"""Bench the Phase-2 GRPO step for graph_walker + frozen Llama.

Measures the cost of one full `grpo_step` (prefix-pass-with-grad +
AR generation no-grad + reward + REINFORCE backward + opt.step) at
varying K (number of rollouts per step). K is the natural scale knob
in GRPO — each rollout costs about as much as one full AR sequence,
so K=8 is roughly 8× a K=1 rollout. We want to confirm the linear-K
scaling holds and find the K beyond which we OOM or fall off a perf
cliff.

Per-K, we additionally sweep BS_outer (the true batch dim before K
replication) to find each K's max-fitting outer batch — the production
default is BS_outer=1, K=8, but if the GPU has headroom at higher
BS_outer×K we want to know.

Tokens-per-step:
    BS_outer × K × (T_pre + gen_length)

So a fair tok/s comparison to the AR baseline (--mode ar in
bench_llama_full_training.py) reads BS_outer*K (effective batch dim)
against BS in the AR run, with the same T_pre + gen_length.

Default trainable surface is `freeze_all_but_E_bias_and_neuromod` —
the production Phase-2 minimal policy surface (only neuromod params
move under REINFORCE). Use `--all-trainable` to bench the larger
surface (everything except frozen Llama).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/bench_grpo.py \\
        --k-list 2 4 8 16 \\
        --bs-outer-list 1 2 4 \\
        --t-pre 256 --gen-length 128 \\
        --compile-block

Outputs:
    A table per (K, BS_outer), peak K's optimum row at the bottom.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.integrated_lm import IntegratedLM
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch, phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import grpo_step


def _walker_cfg_for(d_mem: int, T: int) -> GraphWalkerConfig:
    """PRODUCTION walker config — same defaults as bench_pretrained_gw,
    so GRPO numbers are read against the same walker shape that drives
    the Phase-1 headlines."""
    return GraphWalkerConfig(
        D_s=d_mem,
        D_model=d_mem,
        vocab_size=128_256,         # Llama-3.2-1B
        segment_T=T,
        mod_period=T,
        tbptt_block=T,
        compile_on_train=False,
    )


def _placeholder_reward(
    generated: torch.Tensor,        # [K, T_pre + L]
    reference: torch.Tensor,        # [L]
) -> torch.Tensor:
    """Constant reward — bench is timing the step, not optimizing.

    The grpo_step shape (rollout + reward + REINFORCE backward) is what
    we're measuring, not the reward content. A constant reward gives
    advantages=0 and learning signal=0, but the wall-clock + VRAM
    profile of the step is identical to a real-reward run. (We add a
    tiny per-rollout perturbation so r.std() > adv_std_floor and the
    advantage path doesn't degenerate to NaN.)
    """
    K = generated.shape[0]
    return torch.linspace(0.0, 0.01, K, device=generated.device)


def _bench_one(
    model_name: str, inject_layer: int, d_mem: int,
    T_pre: int, gen_length: int,
    BS_outer: int, K: int,
    warmup: int, n_iter: int,
    compile_walk_block: bool, all_trainable: bool, fused_adam: bool,
) -> tuple[float, float, float] | None:
    """Returns (tok/s, peak_gb, ms_per_iter), or None on OOM.

    The "BS" inside `bench_pretrained_gw` is the parallel-segment
    BS; here the rollout sees an effective batch dim of BS_outer × K.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        walker_cfg = _walker_cfg_for(d_mem=d_mem, T=T_pre)
        cfg = PretrainedGWConfig(
            model_name=model_name,
            inject_layer=inject_layer,
            d_mem=d_mem,
            memory=walker_cfg,
            T=T_pre,
            bs=BS_outer,
            llama_dtype="bf16",
            grpo_K=K,
            grpo_rollout_len=gen_length,
        )
        model = IntegratedLM(cfg).cuda()
        if compile_walk_block:
            model.compile_walker_block()
        model.train(True)

        vocab = model.llama.config.vocab_size

        # Priming pass: a fresh model has `_neuromod_input_*=None` so the
        # first GRPO step's `_active_neuromod_delta` would be None and routing
        # would have no grad. Production training builds these snapshots
        # over Wave 1+2 phase-1 steps. For the bench we run a single
        # phase-1 step with the full trainable surface to seed the
        # snapshots, THEN apply the phase-2 freeze.
        prime_opt = torch.optim.AdamW(
            [p for _, p in model.trainable_parameters()],
            lr=1e-5, fused=fused_adam,
        )
        prime_ids = torch.randint(0, vocab, (BS_outer, T_pre), device="cuda")
        phase1_pretrained_step(
            model, prime_opt,
            Phase1Batch(input_ids=prime_ids, target_ids=prime_ids),
            amp_dtype=torch.bfloat16,
        )
        del prime_opt, prime_ids
        torch.cuda.empty_cache()

        if not all_trainable:
            model.freeze_all_but_E_bias_and_neuromod()

        trainable = sum(
            p.numel() for _, p in model.named_parameters()
            if p.requires_grad
        )
        opt = torch.optim.AdamW(
            [p for _, p in model.trainable_parameters()],
            lr=1e-5, fused=fused_adam,
        )
        # GRPO step takes a single (prefix, reference) pair. We pass
        # BS_outer prefixes by replicating — caller-side batching.
        # Easiest: run BS_outer separate grpo_steps per timed iteration
        # so the harness mirrors realistic per-pair training. For
        # BS_outer=1 (production) this is just one call.
        prefixes = [
            torch.randint(0, vocab, (1, T_pre), device="cuda")
            for _ in range(BS_outer)
        ]
        reference = torch.randint(0, vocab, (gen_length,), device="cuda")

        def step():
            for prefix in prefixes:
                grpo_step(
                    model, opt,
                    prefix_ids=prefix,
                    reference_cont=reference,
                    reward_fn=_placeholder_reward,
                    num_rollouts=K,
                    gen_length=gen_length,
                )

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        ms_per_iter = elapsed / n_iter * 1000
        steps_per_sec = n_iter / elapsed
        # Three throughput metrics, each useful for a different question:
        #
        # GPU-fwd-tok/sec  = BS_outer × K × (T_pre + gen_length) / time
        #   How much raw work the GPU did per second. Scales with K because
        #   K rollouts forward in parallel. NOT a measure of training speed.
        #
        # dataset-tok/sec  = BS_outer × (T_pre + gen_length) / time
        #   Unique training tokens consumed per second. K rollouts share the
        #   same prefix, and generated tokens are model samples (not data),
        #   so K does NOT change this. Useful for "how long to consume
        #   Wave-N's dataset".
        #
        # steps/sec        = 1 / time_per_step
        #   Optimizer-step throughput. The right unit for sample-efficiency-
        #   bound training (Wave 3 with finite fact corpus). K scales this
        #   only weakly — going K=2→16 costs ~7% in steps/sec but gives 8×
        #   variance reduction in the policy gradient.
        gpu_tok_per_sec = BS_outer * K * (T_pre + gen_length) / (
            elapsed / n_iter
        )
        dataset_tok_per_sec = BS_outer * (T_pre + gen_length) / (
            elapsed / n_iter
        )
        print(f"  BS_outer={BS_outer:>2} K={K:>3}  "
              f"trainable={trainable/1e6:>5.1f}M  "
              f"{steps_per_sec:>5.2f} steps/s   "
              f"data {dataset_tok_per_sec/1000:>5.2f}k tok/s   "
              f"gpu {gpu_tok_per_sec/1000:>6.1f}k tok/s   "
              f"peak {peak_gb:>5.2f} GB   "
              f"{ms_per_iter:>7.1f} ms/iter", flush=True)
        return gpu_tok_per_sec, peak_gb, ms_per_iter
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or \
                "out of memory" in str(e).lower():
            print(f"  BS_outer={BS_outer:>2} K={K:>3}    OOM", flush=True)
            return None
        raise
    finally:
        try:
            del model
        except NameError:
            pass
        try:
            del opt
        except NameError:
            pass
        try:
            del prefixes, reference
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--d-mem", type=int, default=256,
                    help="Walker D_s + MemInjectLayer d_mem. Production=256.")
    ap.add_argument("--t-pre", type=int, default=256,
                    help="Prefix length. Default 256 matches PretrainedGWConfig.T.")
    ap.add_argument("--gen-length", type=int, default=128,
                    help="Rollout generation length. Default 128 matches "
                         "PretrainedGWConfig.grpo_rollout_len.")
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[4, 8, 16],
                    help="K values (rollouts per step) to sweep. "
                         "Production GRPO papers (DeepSeek-R1, etc.) cap "
                         "at K=8 — past K=16 the marginal cost of more "
                         "rollouts grows faster than the variance "
                         "reduction is worth. Sweep K=4..64 explicitly "
                         "if you want to see the full diminishing-returns "
                         "curve.")
    ap.add_argument("--bs-outer-list", type=int, nargs="+",
                    default=[1],
                    help="BS_outer values to sweep per K. Default [1] "
                         "matches production GRPO. Pass [1, 2, 4] to find "
                         "headroom at higher outer batch.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iter", type=int, default=3)
    ap.add_argument("--compile-block", action="store_true",
                    help="Run model.compile_walker_block() before bench.")
    ap.add_argument("--all-trainable", action="store_true",
                    help="Don't apply freeze_all_but_E_bias_and_neuromod — "
                         "bench the larger trainable surface (everything "
                         "except frozen Llama). Production Phase-2 uses "
                         "the minimal surface; this is for diagnostic.")
    ap.add_argument("--no-fused-adam", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"=== GRPO step bench ===", flush=True)
    print(f"  device:    {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  model:     {args.model}", flush=True)
    print(f"  T_pre={args.t_pre}, gen_length={args.gen_length}, "
          f"d_mem={args.d_mem}, inject_layer={args.inject_layer}",
          flush=True)
    print(f"  trainable_surface: "
          f"{'all_phase2' if args.all_trainable else 'neuromod_only'}",
          flush=True)
    print(f"  compile_walk_block: {args.compile_walk_block}", flush=True)
    print(f"  warmup={args.warmup}, iter={args.iter}", flush=True)
    print()

    # rows: list of (BS_outer, K, tps, peak_gb, ms_per_iter)
    rows: list[tuple[int, int, float, float, float]] = []
    for K in args.k_list:
        print(f"--- K={K} ---", flush=True)
        for BS_outer in args.bs_outer_list:
            r = _bench_one(
                args.model, args.inject_layer, args.d_mem,
                args.t_pre, args.gen_length,
                BS_outer, K, args.warmup, args.iter,
                compile_walk_block=args.compile_walk_block,
                all_trainable=args.all_trainable,
                fused_adam=not args.no_fused_adam,
            )
            if r is None:
                print(f"  Stopping K={K} BS_outer sweep at BS_outer={BS_outer} "
                      f"(OOM).", flush=True)
                break
            rows.append((BS_outer, K, *r))
        print(flush=True)

    print("=" * 72, flush=True)
    if not rows:
        print("  No (BS_outer, K) fit on this device.", flush=True)
        return
    # Per-K best at max-fitting BS_outer. Steps/sec is derived from
    # ms/iter (= 1000 / ms). Dataset-tok/sec is also derived (BS_outer ×
    # (T_pre + gen_length) / time — does NOT scale with K because rollouts
    # share the prefix).
    print("  Per-K optimum (max BS_outer that fit):", flush=True)
    by_k: dict[int, tuple[int, float, float, float]] = {}
    for bs_outer, K, gpu_tps, gb, ms in rows:
        cur = by_k.get(K)
        if cur is None or gpu_tps > cur[1]:
            by_k[K] = (bs_outer, gpu_tps, gb, ms)
    for K in sorted(by_k):
        bs_outer, gpu_tps, gb, ms = by_k[K]
        steps_per_sec = 1000.0 / ms
        dataset_tok_per_sec = bs_outer * (args.t_pre + args.gen_length) / (
            ms / 1000.0
        )
        print(f"    K={K:>3}  BS_outer={bs_outer:>2}  "
              f"{steps_per_sec:>5.2f} steps/s  "
              f"data {dataset_tok_per_sec/1000:>5.2f}k tok/s  "
              f"gpu {gpu_tps/1000:>6.1f}k tok/s  "
              f"peak {gb:>5.2f} GB  "
              f"{ms:>7.1f} ms/iter", flush=True)
    print(flush=True)
    print("  Note: K is a variance-reduction lever, not a throughput lever. "
          "All K rollouts share the same prefix, so dataset-tok/sec does "
          "not scale with K. Use steps/sec or dataset-tok/sec to estimate "
          "wave wall-clock; gpu-tok/sec is GPU compute throughput only.",
          flush=True)


if __name__ == "__main__":
    main()
