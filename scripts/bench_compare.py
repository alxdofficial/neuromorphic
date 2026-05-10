"""Phase 1 + Phase 2 throughput comparison: vanilla Llama vs trajectory-memory.

Each path runs at its own max-fitting BS (project memory: "Bench at each
path's own optimal BS — for cross-path throughput comparisons, each
scenario reported at its own max-fitting BS, not forced to share BS").
Per-path values hard-coded below; adjust if hardware/config changes.

Phase 1 (long-doc TF NTP) — paths:
  V1.A — vanilla Llama forward (no_grad)
  V1.B — vanilla Llama lm_head-only TF training step (frozen backbone)
  T1   — Llama + trajectory-memory full Phase1Trainer step

Phase 2 (GRPO) — paths:
  V2   — vanilla Llama GRPO (J-sample + TF-replay, lm_head-only train)
  T2   — Llama + trajectory-memory two-pass GRPO step (Phase2Trainer.step)

All paths use bf16 + Flash Attention via HF transformers; backward time
included for training paths.

Usage:
    PYTHONPATH=. python scripts/bench_compare.py
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


# ──────────────────────────────────────────────────────────────────────
# Per-path BS / K — each scenario tuned to its own max-fitting size on
# RTX 4090 24 GB, medium config, eager. Aim is each path peaking near
# ~21 GB. Adjust if hardware / config changes.
# ──────────────────────────────────────────────────────────────────────
BS_V1A_FWD = 48     # vanilla Llama fwd no_grad — small footprint, lots of headroom
BS_V1B_STEP = 5     # vanilla Llama lm_head step — backward activations
BS_T1_TRAJMEM = 4   # trajmem step (KV cache) — sweep shows BS=8 OOMs eager
K_V2_VANILLA = 12   # vanilla GRPO with KV cache (K=14 OOMs; pass-2 holds K forwards' activations)
K_T2_TRAJMEM = 6    # trajmem GRPO with KV cache (K=8 OOMs)


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


def bench_phase1_vanilla_fwd(args, device, llama, vocab, BS):
    """V1.A: vanilla Llama forward (no_grad)."""
    T = args.t_phase1
    input_ids = torch.randint(0, vocab, (BS, T), device=device)

    llama.train(False)
    for p in llama.parameters():
        p.requires_grad = False

    def vanilla_fwd():
        with torch.no_grad():
            return llama(input_ids).logits

    print(f"\n[V1.A] Vanilla Llama forward (no_grad) BS={BS}")
    tps, mem, _ = bench("vanilla Llama fwd (no_grad)",
                         vanilla_fwd, args.warmup, args.iter, BS, T)
    cleanup_cuda()
    return tps, mem


def bench_phase1_vanilla_step(args, device, llama, vocab, BS):
    """V1.B: vanilla Llama lm_head-only TF training step (backward included)."""
    T = args.t_phase1
    input_ids = torch.randint(0, vocab, (BS, T), device=device)

    for p in llama.parameters():
        p.requires_grad = False
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

    print(f"\n[V1.B] Vanilla Llama lm_head TF step BS={BS}")
    tps, mem, _ = bench("vanilla Llama step (lm_head)",
                         vanilla_step, args.warmup, args.iter, BS, T)

    # Reset llama to frozen for next caller.
    for p in llama.parameters():
        p.requires_grad = False
    llama.train(False)
    del opt
    cleanup_cuda()
    return tps, mem


def bench_phase1_trajmem(args, device, BS):
    """T1: Llama + trajmem full Phase1Trainer.step_wave1."""
    cfg = getattr(TrajMemConfig, args.config_tier)()
    T = cfg.D * cfg.T_window

    model = IntegratedLM(cfg, model_name=args.model, attach_lm=True).to(device)
    if args.compile:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )
    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(
        model, optimizer, scheduler=None, grad_clip=1.0,
        use_kv_cache=True,  # ~1.79× speedup; matches what trainers use in production
    )
    vocab = model.llama.config.vocab_size
    chunk = torch.randint(0, vocab, (BS, T), device=device)

    def step():
        trainer.step_wave1(chunk)

    print("\n[T1] Llama + trajmem Phase1Trainer.step_wave1 (KV cache enabled)")
    label = f"trajmem step (tier={args.config_tier}" + (
        ", compile" if args.compile else "") + ", kv-cache)"
    tps, mem, _ = bench(label, step, args.warmup, args.iter, BS, T)

    del trainer, optimizer, model
    cleanup_cuda()
    return tps, mem


def bench_phase2_vanilla(args, device, llama, vocab, K):
    """V2: vanilla Llama GRPO (sample J responses no_grad, TF replay,
    backward through lm_head). Mimics what `Phase2Trainer.step` does
    with vanilla Llama as the policy (no memory module)."""
    BS = 1  # Phase 2 is per-prompt
    T_pre = args.t_prompt
    T_gen = args.t_gen
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
        # AR sample for T_gen tokens with HF KV cache (= what production
        # vanilla GRPO does — TRL, OpenRLHF, vLLM all use KV cache for
        # sampling). Without KV cache vanilla pays O(T_pre+T_gen) per token,
        # which is unrealistic and unfairly handicaps the vanilla baseline.
        from transformers import DynamicCache
        cache = DynamicCache()
        # Prefill the prompt.
        out = llama(prompt_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logit = out.logits[:, -1, :]
        next_tok = last_logit.argmax(dim=-1, keepdim=True)
        gen = [next_tok]
        for _ in range(T_gen - 1):
            out = llama(next_tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last_logit = out.logits[:, -1, :]
            next_tok = last_logit.argmax(dim=-1, keepdim=True)
            gen.append(next_tok)
        return torch.cat(gen, dim=1)

    def grpo_step():
        opt.zero_grad(set_to_none=True)
        # Pass 1 — K samples (KV-cached AR).
        samples = [sample_one() for _ in range(K)]
        advantages = torch.tensor([1.0, -1.0] * (K // 2), device=device)[:K]
        # Pass 2 — TF replay over (prompt + sample). One forward per sample
        # at full length, returning logits for shifted positions.
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


def bench_phase2_trajmem(args, device, K):
    """T2: Llama + trajmem Phase2Trainer.step (two-pass GRPO)."""
    cfg = getattr(TrajMemConfig, args.config_tier)()
    BS = 1
    T_pre = args.t_prompt
    T_gen = args.t_gen

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
    ap.add_argument("--t-phase1", type=int, default=1024,
                    help="Sequence length for Phase 1 paths (= D × T_window)")
    ap.add_argument("--t-prompt", type=int, default=1024,
                    help="Phase 2 prompt length")
    ap.add_argument("--t-gen", type=int, default=64,
                    help="Phase 2 max generated tokens per sample")
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
    print(f"  Per-path BS / K (each at its own max-fitting size):")
    print(f"    V1.A vanilla fwd:   BS={BS_V1A_FWD}")
    print(f"    V1.B vanilla step:  BS={BS_V1B_STEP}")
    print(f"    T1   trajmem step:  BS={BS_T1_TRAJMEM}")
    print(f"    V2   vanilla GRPO:  K={K_V2_VANILLA}")
    print(f"    T2   trajmem GRPO:  K={K_T2_TRAJMEM}")
    print(f"  Phase 1: T={args.t_phase1}")
    print(f"  Phase 2: T_prompt={args.t_prompt}, T_gen={args.t_gen}")

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
        p1_a = bench_phase1_vanilla_fwd(args, device, llama, vocab, BS_V1A_FWD)
        p1_b = bench_phase1_vanilla_step(args, device, llama, vocab, BS_V1B_STEP)
        p1_t = bench_phase1_trajmem(args, device, BS_T1_TRAJMEM)

    if not args.skip_phase2:
        # Re-load llama since trajmem path freed it (also reset frozen
        # state in case it's needed).
        if "llama" not in dir() or llama is None:
            llama = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16,
            ).to(device)
            vocab = llama.config.vocab_size
        p2_v = bench_phase2_vanilla(args, device, llama, vocab, K_V2_VANILLA)
        del llama
        cleanup_cuda()
        p2_t = bench_phase2_trajmem(args, device, K_T2_TRAJMEM)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("SUMMARY (each path at its own max-fitting BS / K)")
    print("=" * 76)

    def row(label: str, tps, mem):
        if tps is None:
            print(f"  {label:<55}      n/a")
            return
        # Phase 1 numbers are k tok/s; Phase 2 (GRPO) is tens of tok/s — show
        # both at the same precision regardless of magnitude.
        if tps >= 1000:
            tps_str = f"{tps/1000:>7.1f}k tok/s"
        else:
            tps_str = f"{tps:>7.1f}  tok/s"
        print(f"  {label:<55} {tps_str}   {mem:>5.2f} GB")

    if not args.skip_phase1:
        print("Phase 1 (long-doc TF NTP):")
        row(f"  V1.A — vanilla Llama fwd (no_grad)   BS={BS_V1A_FWD}", *p1_a)
        row(f"  V1.B — vanilla Llama lm_head step    BS={BS_V1B_STEP}", *p1_b)
        row(f"  T1   — Llama + trajmem step          BS={BS_T1_TRAJMEM}", *p1_t)
        if p1_b[0] and p1_t[0]:
            print(f"\n  T1 vs V1.B (per-token tput, each at own max BS): "
                  f"{p1_b[0] / p1_t[0]:.2f}×")

    if not args.skip_phase2:
        print("\nPhase 2 (GRPO, lm_head trainable for vanilla):")
        row(f"  V2   — vanilla Llama GRPO step       K={K_V2_VANILLA}", *p2_v[:2])
        row(f"  T2   — Llama + trajmem two-pass GRPO K={K_T2_TRAJMEM}", *p2_t[:2])
        if p2_v[0] and p2_t[0]:
            print(f"\n  T2 vs V2 (per-token tput, each at own max K): "
                  f"{p2_v[0] / p2_t[0]:.2f}×")


if __name__ == "__main__":
    main()
