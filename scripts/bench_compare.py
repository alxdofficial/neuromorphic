"""Phase 1 + Phase 2 throughput comparison: vanilla Llama vs trajectory-memory.

Runs each path at the production config (medium tier, BS=2, --compile)
and prints a comparison table. All paths use bf16 + Flash Attention via
HF transformers; backward time is included for training paths.

Phase 1 (long-doc TF NTP) — paths:
  V1.A — vanilla Llama forward (no_grad)
  V1.B — vanilla Llama lm_head-only TF training step (frozen backbone)
  T1   — Llama + trajectory-memory full Phase1Trainer step

Phase 2 (GRPO) — paths:
  V2   — vanilla Llama GRPO (J-sample + TF-replay, lm_head-only train)
  T2   — Llama + trajectory-memory two-pass GRPO step (Phase2Trainer.step)

Usage:
    PYTHONPATH=. python scripts/bench_compare.py --bs 2 --compile
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import bench, cleanup_cuda  # noqa: E402

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training import (  # noqa: E402
    Phase1Trainer, Phase2Trainer, build_optimizer,
)


def _make_fake_tokenizer():
    """Minimal tokenizer stub for Phase2Trainer.step (decode + eos_id)."""
    class T:
        eos_token_id = 128001
        pad_token_id = 128001
        unk_token_id = -1

        def decode(self, ids, skip_special_tokens=False):
            return "fake_decoded"

        def convert_tokens_to_ids(self, tok):
            return 128009 if tok == "<|eot_id|>" else self.unk_token_id

    return T()


def bench_phase1_vanilla(args, device, llama, vocab):
    """V1.A: vanilla Llama forward (no_grad). V1.B: vanilla Llama lm_head step."""
    BS = args.bs
    T = args.t_phase1
    input_ids = torch.randint(0, vocab, (BS, T), device=device)

    # V1.A — forward only
    llama.train(False)
    for p in llama.parameters():
        p.requires_grad = False

    def vanilla_fwd():
        with torch.no_grad():
            return llama(input_ids).logits

    print("\n[V1.A] Vanilla Llama forward (no_grad)")
    tps_a, mem_a, _ = bench("vanilla Llama fwd (no_grad)",
                             vanilla_fwd, args.warmup, args.iter, BS, T)

    # V1.B — lm_head-only train step
    for p in llama.lm_head.parameters():
        p.requires_grad = True
    llama.train(True)
    opt = torch.optim.AdamW(
        [p for p in llama.lm_head.parameters() if p.requires_grad],
        lr=1e-4, fused=True,
    )

    def vanilla_step():
        opt.zero_grad(set_to_none=True)
        out = llama(input_ids)
        logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        opt.step()

    print("\n[V1.B] Vanilla Llama lm_head TF step")
    tps_b, mem_b, _ = bench("vanilla Llama step (lm_head)",
                             vanilla_step, args.warmup, args.iter, BS, T)

    # Reset llama to frozen for next caller.
    for p in llama.parameters():
        p.requires_grad = False
    llama.train(False)
    cleanup_cuda()
    return (tps_a, mem_a), (tps_b, mem_b)


def bench_phase1_trajmem(args, device):
    """T1: Llama + trajmem full Phase1Trainer.step_wave1."""
    cfg = getattr(TrajMemConfig, args.config_tier)()
    T = cfg.D * cfg.T_window
    BS = args.bs

    model = IntegratedLM(cfg, model_name=args.model, attach_lm=True).to(device)
    if args.compile:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )
    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(model, optimizer, scheduler=None, grad_clip=1.0)
    vocab = model.llama.config.vocab_size
    chunk = torch.randint(0, vocab, (BS, T), device=device)

    def step():
        trainer.step_wave1(chunk)

    print("\n[T1] Llama + trajmem Phase1Trainer.step_wave1")
    label = f"trajmem step (tier={args.config_tier}" + (
        ", compile" if args.compile else "") + ")"
    tps, mem, _ = bench(label, step, args.warmup, args.iter, BS, T)

    del trainer, optimizer, model
    cleanup_cuda()
    return tps, mem


def bench_phase2_vanilla(args, device, llama, vocab):
    """V2: vanilla Llama GRPO (sample J responses no_grad, TF replay,
    backward through lm_head). Mimics what `Phase2Trainer.step` does
    with vanilla Llama as the policy (no memory module)."""
    BS = 1  # Phase 2 is per-prompt
    T_pre = args.t_prompt
    T_gen = args.t_gen
    K = args.num_samples
    prompt_ids = torch.randint(0, vocab, (BS, T_pre), device=device)

    # Set up: only lm_head trainable.
    for p in llama.parameters():
        p.requires_grad = False
    for p in llama.lm_head.parameters():
        p.requires_grad = True
    llama.train(True)
    opt = torch.optim.AdamW(
        [p for p in llama.lm_head.parameters() if p.requires_grad],
        lr=1e-4, fused=True,
    )

    @torch.no_grad()
    def sample_one():
        # Greedy AR sample for K_gen tokens — speed reference, not real
        # GRPO sampling (which would do multinomial; cost is similar).
        cur = prompt_ids.clone()
        for _ in range(T_gen):
            logits = llama(cur).logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)
        return cur[:, T_pre:]

    def grpo_step():
        opt.zero_grad(set_to_none=True)
        # Pass 1 — K samples
        samples = [sample_one() for _ in range(K)]
        advantages = torch.tensor([1.0, -1.0] * (K // 2), device=device)[:K]
        # Pass 2 — TF replay
        loss = torch.zeros((), device=device)
        for s, adv in zip(samples, advantages):
            full = torch.cat([prompt_ids, s], dim=1)
            out = llama(full)
            shift = out.logits[:, T_pre - 1: -1, :]   # logits predicting sample tokens
            target = s
            logp = F.log_softmax(shift.float(), dim=-1)
            sample_logp = logp.gather(2, target.unsqueeze(-1)).squeeze(-1)
            loss = loss + (-adv * sample_logp.sum())
        loss = loss / K
        loss.backward()
        opt.step()

    print(f"\n[V2] Vanilla Llama GRPO step (K={K} samples, "
          f"T_pre={T_pre}, T_gen={T_gen}, lm_head trainable)")
    # Phase 2 token-count for tok/s: prompt + K × (T_pre + T_gen) for
    # pass 2 forwards. We charge by sampled tokens (K × T_gen) for
    # comparability with phase 1.
    eff_tokens = K * T_gen
    tps, mem, ms = bench("vanilla GRPO step", grpo_step,
                         args.warmup_phase2, args.iter_phase2, BS, eff_tokens)

    for p in llama.parameters():
        p.requires_grad = False
    llama.train(False)
    del opt
    cleanup_cuda()
    return tps, mem, ms


def bench_phase2_trajmem(args, device):
    """T2: Llama + trajmem Phase2Trainer.step (two-pass GRPO)."""
    cfg = getattr(TrajMemConfig, args.config_tier)()
    BS = 1
    T_pre = args.t_prompt
    T_gen = args.t_gen
    K = args.num_samples

    model = IntegratedLM(cfg, model_name=args.model, attach_lm=True).to(device)
    if args.compile:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )
    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase2Trainer(model, optimizer, scheduler=None, grad_clip=1.0)
    vocab = model.llama.config.vocab_size
    prompt_ids = torch.randint(0, vocab, (BS, T_pre), device=device)
    tokenizer = _make_fake_tokenizer()

    def step():
        trainer.step(
            prompt_ids[0], num_samples=K, max_new_tokens=T_gen,
            reward_kind="exact_match", gold="x", meta=None,
            tokenizer=tokenizer, temperature=1.0,
        )

    print(f"\n[T2] Llama + trajmem two-pass GRPO step (K={K}, "
          f"T_pre={T_pre}, T_gen={T_gen})")
    eff_tokens = K * T_gen
    label = f"trajmem GRPO step ({args.config_tier}"
    if args.compile:
        label += ", compile"
    label += ")"
    tps, mem, ms = bench(label, step, args.warmup_phase2, args.iter_phase2,
                         BS, eff_tokens)

    del trainer, optimizer, model
    cleanup_cuda()
    return tps, mem, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--bs", type=int, default=2,
                    help="BS for Phase 1 paths (V1.A, V1.B, T1)")
    ap.add_argument("--t-phase1", type=int, default=1024,
                    help="Sequence length for Phase 1 paths (= D × T_window)")
    ap.add_argument("--t-prompt", type=int, default=1024,
                    help="Phase 2 prompt length")
    ap.add_argument("--t-gen", type=int, default=64,
                    help="Phase 2 max generated tokens per sample")
    ap.add_argument("--num-samples", type=int, default=4,
                    help="Phase 2 GRPO group size K")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile model.forward_window for trajmem paths")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iter", type=int, default=8)
    ap.add_argument("--warmup-phase2", type=int, default=1)
    ap.add_argument("--iter-phase2", type=int, default=3)
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    print("=" * 76)
    print("PHASE 1 + PHASE 2 COMPARISON: vanilla Llama vs trajectory-memory")
    print("=" * 76)
    print(f"  device:      {torch.cuda.get_device_name()} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"  model:       {args.model}")
    print(f"  config tier: {args.config_tier}")
    print(f"  compile:     {'on' if args.compile else 'off'}")
    print(f"  Phase 1:     BS={args.bs}, T={args.t_phase1}")
    print(f"  Phase 2:     T_prompt={args.t_prompt}, T_gen={args.t_gen}, "
          f"K={args.num_samples}")

    # Load Llama once for vanilla paths.
    if not args.skip_phase1 or not args.skip_phase2:
        print("\nLoading vanilla Llama for baseline paths...")
        llama = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16,
        ).to(device)
        vocab = llama.config.vocab_size

    p1_a = p1_b = p1_t = None
    p2_v = p2_t = None

    if not args.skip_phase1:
        p1_a, p1_b = bench_phase1_vanilla(args, device, llama, vocab)
        p1_t = bench_phase1_trajmem(args, device)

    if not args.skip_phase2:
        # Re-load llama since trajmem path freed it (also reset frozen
        # state in case it's needed).
        if "llama" not in dir() or llama is None:
            llama = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16,
            ).to(device)
            vocab = llama.config.vocab_size
        p2_v = bench_phase2_vanilla(args, device, llama, vocab)
        del llama
        cleanup_cuda()
        p2_t = bench_phase2_trajmem(args, device)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)

    def row(label: str, tps, mem):
        if tps is None:
            print(f"  {label:<50}      n/a")
            return
        print(f"  {label:<50} {tps/1000:>8.1f}k tok/s   {mem:>5.2f} GB")

    if not args.skip_phase1:
        print("Phase 1 (long-doc TF NTP):")
        row(f"  V1.A — vanilla Llama fwd (no_grad)", *p1_a)
        row(f"  V1.B — vanilla Llama lm_head TF step", *p1_b)
        row(f"  T1   — Llama + trajmem step", *p1_t)
        if p1_b[0] and p1_t[0]:
            print(f"\n  T1 vs V1.B slowdown: "
                  f"{p1_b[0] / p1_t[0]:.2f}× "
                  f"(adding the memory module costs {p1_b[0] - p1_t[0]:.0f} tok/s)")

    if not args.skip_phase2:
        print("\nPhase 2 (GRPO, K samples × prompt+gen tokens, lm_head trainable):")
        row(f"  V2   — vanilla Llama GRPO step", p2_v[0], p2_v[1])
        row(f"  T2   — Llama + trajmem two-pass GRPO step", p2_t[0], p2_t[1])
        if p2_v[0] and p2_t[0]:
            print(f"\n  T2 vs V2 slowdown: {p2_v[0] / p2_t[0]:.2f}×")


if __name__ == "__main__":
    main()
