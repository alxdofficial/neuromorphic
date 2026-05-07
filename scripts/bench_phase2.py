"""Phase 2 bench — GRPO sample/score/replay throughput.

Runs two paths:
  E) vanilla Llama realistic GRPO  (REINFORCE on lm_head, frozen backbone, BERT reward)
  F) frozen Llama + GraphWalker realistic GRPO  (production grpo_step)

Both paths use BERT-cosine reward on real text (passphrase-chat-grpo data
in production, or random tokens for diagnostic runs).

E is the "what does training Llama via REINFORCE on its own token
sampling cost" baseline. The policy is `softmax(lm_head_logits)`; log-π
sums per-token log-probs of the GENERATED tokens (prefix tokens are
teacher-forced data, not policy decisions, so they are NOT credited).

Examples:
  # Production-shape Phase-2 bench (Wave 3 config: B=8, K=8, T_pre=256)
  PYTHONPATH=. python scripts/bench_phase2.py --B 8 --K 8

  # Target-config Phase-2 bench (eager mode)
  PYTHONPATH=. python scripts/bench_phase2.py --target-config --B 1 --K 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import (  # noqa: E402
    add_walker_config_args, bench, cleanup_cuda,
    print_config_summary, walker_cfg_from_args,
)

from src.graph_walker.pretrained.config import PretrainedGWConfig  # noqa: E402
from src.graph_walker.pretrained.integrated_lm import IntegratedLM  # noqa: E402
from src.graph_walker.pretrained.train_phase1 import (  # noqa: E402
    Phase1Batch, phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import grpo_step  # noqa: E402


# ----------------------------------------------------------------------
# Vanilla Llama realistic GRPO — Path E
# ----------------------------------------------------------------------


def _sample_token(logits: torch.Tensor, temp: float, top_p: float) -> torch.Tensor:
    if temp <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temp
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        keep = cum <= top_p
        keep[:, 0] = True
        sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        unsort = torch.full_like(logits, float("-inf"))
        unsort.scatter_(1, sorted_idx, sorted_logits)
        logits = unsort
    probs = logits.softmax(dim=-1).clamp(min=1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def vanilla_grpo_step(
    llama,
    opt: torch.optim.Optimizer,
    *,
    prefix_ids: torch.Tensor,            # [B, T_pre]
    references: list[torch.Tensor],      # B variable-length tensors
    reward_fn,
    K: int,
    gen_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    grad_clip: float = 1.0,
    adv_std_floor: float = 1e-3,
):
    """REINFORCE on Llama's own token sampling. Backbone frozen, only
    `lm_head` trains. Mirrors the GW grpo_step's sample → score → replay
    structure but the policy is the LM's softmax-over-vocab.

    Per-token log-π is `log P(tok_t | tok_<t)`; only GEN tokens are
    credited (prefix is teacher-forced data, not policy actions). The
    log-π aggregation is mean-over-gen-tokens to keep gradient
    magnitudes bounded across gen lengths.
    """
    device = next(llama.parameters()).device
    B, T_pre = prefix_ids.shape
    BK = B * K
    prefix_rep = prefix_ids.repeat_interleave(K, dim=0).contiguous()
    opt.zero_grad(set_to_none=True)

    # ---- Sample (no_grad, AR) ----
    with torch.no_grad():
        out = llama(prefix_rep, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[:, -1, :]
        new_tokens = []
        for _ in range(gen_length):
            tok = _sample_token(last_logits, temperature, top_p)
            new_tokens.append(tok)
            out = llama(tok.unsqueeze(-1), past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]
        new_tokens_t = torch.stack(new_tokens, dim=1)               # [B*K, gen_length]
        generated = torch.cat([prefix_rep, new_tokens_t], dim=1)    # [B*K, T_pre + gen_length]

    # ---- Score ----
    rewards = reward_fn(new_tokens_t, references).to(device)        # [B*K]
    r_grp = rewards.view(B, K)
    r_mean = r_grp.mean(dim=1, keepdim=True)
    r_std = r_grp.std(dim=1, keepdim=True).clamp(min=adv_std_floor)
    advantages = ((r_grp - r_mean) / r_std).view(BK)

    # ---- Replay (with grad, teacher-forced) ----
    # replay_seq has length T_pre + gen_length - 1 (drop last token —
    # there's no target for it; matches the GW path's contract).
    replay_seq = generated[:, :-1]
    targets = generated[:, 1:]
    out = llama(replay_seq)
    logits = out.logits                                             # [B*K, T_replay, V]
    # Memory-efficient per-token log-π: at production scale, full
    # log_softmax over [B*K, T_replay, V] would materialize a
    # ~12 GB fp32 tensor at V=128256, B*K=64, T_replay=383. Chunk
    # along T to bound peak memory. Per-token CE = -log P(target):
    #   target_logit = logit at target index
    #   logsumexp = log Σ_j exp(logit_j)
    #   log_pi_per_token = target_logit - logsumexp
    BK_, T_replay, V = logits.shape
    chunk = 256
    log_pi_chunks = []
    for s in range(0, T_replay, chunk):
        e = min(s + chunk, T_replay)
        log_chunk = logits[:, s:e, :].float()
        tgt_chunk = targets[:, s:e]
        target_logit = log_chunk.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
        log_pi_chunks.append(target_logit - log_chunk.logsumexp(dim=-1))
    log_pi_per_token = torch.cat(log_pi_chunks, dim=1)              # [B*K, T_replay]
    # Only credit GEN tokens. logits at position i predicts targets[i].
    # Gen tokens occupy targets[T_pre-1 : T_pre + gen_length - 1].
    log_pi_gen = log_pi_per_token[:, T_pre - 1:].mean(dim=1)        # [B*K]
    del out, logits

    loss = -(log_pi_gen * advantages.detach()).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in llama.parameters() if p.requires_grad], grad_clip,
    )
    opt.step()
    return float(loss.detach()), float(grad_norm), float(rewards.mean())


# ----------------------------------------------------------------------
# Reward — BERT-cosine on Llama tokens
# ----------------------------------------------------------------------


def _make_reward(use_bert: bool, tok_obj):
    if not use_bert:
        from src.graph_walker.pretrained.train_phase2 import _default_token_match_reward
        return _default_token_match_reward
    from src.graph_walker.pretrained.rewards import (
        BertCosineReward, load_default_bert,
    )
    bert = load_default_bert(device="cpu")  # CPU is fine; small batches
    return BertCosineReward(bert_model=bert, tokenizer=tok_obj, device="cpu")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--B", type=int, default=8,
                    help="BS_outer — independent prompts per step")
    ap.add_argument("--K", type=int, default=8,
                    help="rollouts per prompt")
    ap.add_argument("--T-pre", type=int, default=256)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--reward", choices=["bert", "placeholder"], default="bert")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iter", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--skip-vanilla", action="store_true",
                    help="Skip path E (vanilla LM GRPO baseline).")
    ap.add_argument("--skip-gw", action="store_true",
                    help="Skip path F (GW GRPO).")
    add_walker_config_args(ap)
    args = ap.parse_args()

    device = torch.device("cuda")
    B, K = args.B, args.K
    T_pre, L_gen = args.T_pre, args.gen_length
    print(f"\n=== Phase 2 bench (sample/score/replay GRPO) ===")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  model:  {args.model}")
    print(f"  B={B}, K={K}, T_pre={T_pre}, gen_length={L_gen}, "
          f"reward={args.reward}, warmup={args.warmup}, iter={args.iter}")
    print()

    # Tokenizer + reward (shared between E and F).
    base_tok = AutoTokenizer.from_pretrained(args.model)
    reward_fn = _make_reward(args.reward == "bert", base_tok)

    # Random prefix + references for the bench. (Production uses real
    # passphrase-chat data; the bench shape is what matters for timing.)
    vocab = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).config.vocab_size
    prefix_ids = torch.randint(0, vocab, (B, T_pre), device=device)
    references = [
        torch.randint(0, vocab, (16,), device=device) for _ in range(B)
    ]

    # ----- Path E: vanilla LM GRPO -----
    if args.skip_vanilla:
        print("[E] skipped (--skip-vanilla)")
        tps_e, mem_e = None, None
    else:
        print("Loading vanilla Llama for path E...")
        llama = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
        ).to(device)
        for p in llama.parameters():
            p.requires_grad = False
        for p in llama.lm_head.parameters():
            p.requires_grad = True
        llama.train(True)
        opt_e = torch.optim.AdamW(
            [p for p in llama.lm_head.parameters() if p.requires_grad],
            lr=1e-5, fused=True,
        )

        def step_e():
            vanilla_grpo_step(
                llama, opt_e,
                prefix_ids=prefix_ids, references=references,
                reward_fn=reward_fn,
                K=K, gen_length=L_gen,
                temperature=args.temperature, top_p=args.top_p,
            )

        print()
        print("[E] Vanilla Llama, realistic GRPO (REINFORCE on lm_head)")
        tps_e, mem_e, ms_e = bench(
            "vanilla LM GRPO", step_e,
            args.warmup, args.iter, B * K, T_pre + L_gen,
        )
        del llama, opt_e
        cleanup_cuda()

    # ----- Path F: frozen Llama + GW GRPO -----
    if args.skip_gw:
        print("\n[F] skipped (--skip-gw)")
        tps_f, mem_f = None, None
    else:
        print()
        print(f"Loading frozen Llama + GraphWalker for path F...")
        # GW path: T == segment_T. Use T_pre as the segment length so
        # walker absorbs all prefix tokens in one segment.
        walker_cfg = walker_cfg_from_args(args, T=T_pre, vocab=vocab)
        d_mem = walker_cfg.D_s
        cfg = PretrainedGWConfig(
            model_name=args.model, inject_layer=args.inject_layer,
            d_mem=d_mem, memory=walker_cfg, T=T_pre, bs=B, llama_dtype="bf16",
        )
        model = IntegratedLM(cfg).to(device)
        model.train(True)

        # Phase-2 trains the same surface as Phase-1 by default.
        opt_f = torch.optim.AdamW(
            [p for _, p in model.trainable_parameters()], lr=1e-5, fused=True,
        )
        # Phase-1 prime so neuromod._neuromod_input_* exists — without
        # it, _begin_plastic_window has nothing to read and routing
        # carries no neuromod gradient.
        prime_in = torch.randint(0, vocab, (1, T_pre), device=device)
        phase1_pretrained_step(
            model, opt_f,
            Phase1Batch(input_ids=prime_in, target_ids=prime_in),
            amp_dtype=torch.bfloat16,
        )

        label = "TARGET ~110M" if args.target_config else "production ~25M"
        print_config_summary(walker_cfg, label)

        # Optional compile-block. Phase 2 mixes prefix forward + AR-gen
        # (single-token forwards through KV cache); historically
        # `--compile-block` hung here because each AR step is a different
        # shape and inductor recompiled per token. With `--dynamic-shapes`
        # (dynamic=None), inductor compiles a single shape-polymorphic
        # artifact that handles all shapes — the experiment is whether
        # that resolves the hang.
        if args.compile_walk_block:
            kind = "regional" if args.regional_compile else "whole-block"
            dyn = None if args.dynamic_shapes else False
            dyn_label = "dynamic=None" if args.dynamic_shapes else "dynamic=False"
            print(f"  Compiling walker {kind} (mode={args.compile_mode}, {dyn_label}) ...")
            model.compile_walker_block(
                mode=args.compile_mode,
                regional=args.regional_compile,
                dynamic=dyn,
            )

        def step_f():
            grpo_step(
                model, opt_f,
                prefix_ids=prefix_ids,
                reference_cont=references,
                reward_fn=reward_fn,
                num_rollouts=K,
                gen_length=L_gen,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_id=base_tok.eos_token_id,
            )

        print()
        print("[F] Frozen Llama + GW, realistic GRPO (grpo_step)")
        tps_f, mem_f, ms_f = bench(
            "Llama + GW grpo_step", step_f,
            args.warmup, args.iter, B * K, T_pre + L_gen,
        )
        del model, opt_f
        cleanup_cuda()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=== Summary ===")
    print(f"  Step throughput is `B * K * (T_pre + gen_length) / time` (gpu-fwd-tok/s).")
    print(f"  Sessions/sec = B / time_per_step.")
    print()
    rows = [
        ("E", "vanilla LM GRPO", tps_e, mem_e),
        ("F", "Llama + GW GRPO", tps_f, mem_f),
    ]
    print(f"  {'Path':<6}{'name':<28}{'gpu-tok/s':>12}{'peak GB':>10}")
    for tag, name, tps, mem in rows:
        if tps is None:
            print(f"  {tag:<6}{name:<28}{'OOM/skip':>12}{'-':>10}")
        else:
            print(f"  {tag:<6}{name:<28}{tps/1000:>11.1f}k{mem:>9.2f}")
    if tps_e is not None and tps_f is not None:
        print()
        print(f"  F vs E slowdown: {tps_e/tps_f:.2f}x  "
              f"(vanilla {tps_e/1000:.1f}k → GW {tps_f/1000:.1f}k tok/s)")


if __name__ == "__main__":
    main()
