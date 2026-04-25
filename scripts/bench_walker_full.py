"""Comprehensive walker bench: eager vs whole-block-compile vs cudagraph.

Each config is benched in a fresh process to avoid cache contamination.
Run with:

    PYTHONPATH=. python scripts/bench_walker_full.py [B]

Where ``B`` defaults to 4. Outputs the table in docs/triton_rewrite_plan.md.
"""

from __future__ import annotations

import sys
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step, phase1_step_cudagraph


def make_lm(use_neuromod: bool = False) -> StandaloneLM:
    cfg = GraphWalkerConfig(
        plane_rows=8, plane_cols=8, L=4, K=16,
        D_model=512, D_s=256, D_id=32,
        n_heads=4, n_score_heads=4, D_q_per_head=32, D_q_in=32,
        K_horizons=4, K_buf=4,
        mod_period=64, tbptt_block=64, segment_T=128,
        ffn_mult_content=4,
        content_mlp_depth=2,
        post_model_depth=1,
        vocab_size=1024,
        use_neuromod=use_neuromod,
        compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def time_run(step_fn, lm, opt, tokens, n_iters: int) -> tuple[float, float]:
    """Returns (warmup_seconds, throughput_tokens_per_second)."""
    t0 = time.perf_counter()
    for i in range(3):
        step_fn(lm, opt, tokens, training_step=i)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0

    start = time.perf_counter()
    for i in range(n_iters):
        step_fn(lm, opt, tokens, training_step=10 + i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    B, T = tokens.shape
    return warm, n_iters * B * T / elapsed


def bench_one(label: str, build_lm, configure, step_fn, B: int) -> tuple[str, float, float]:
    lm = build_lm()
    cfg = lm.cfg
    if configure is not None:
        configure(lm)
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (B, cfg.segment_T), device="cuda")
    warm, tps = time_run(step_fn, lm, opt, tokens, n_iters=10)
    print(f"  {label:<32} {tps:>10.1f} tok/s   warmup={warm:.1f}s", flush=True)
    return label, warm, tps


def main() -> None:
    if not torch.cuda.is_available():
        print("no cuda")
        return

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"=== walker bench (B={B}, T=128, mod_period=64) ===\n", flush=True)

    results: list[tuple[str, float, float]] = []

    # 1. Eager (no compile, no neuromod)
    print("[eager]", flush=True)
    results.append(bench_one(
        "eager", lambda: make_lm(use_neuromod=False),
        configure=None,
        step_fn=phase1_step, B=B,
    ))

    # 2. Whole-block torch.compile
    print("\n[whole-block compile]", flush=True)
    def configure_block(lm):
        lm.memory.compile_block(mode="default")
    results.append(bench_one(
        "whole-block compile", lambda: make_lm(use_neuromod=False),
        configure=configure_block,
        step_fn=phase1_step, B=B,
    ))

    # 3. Manual cudagraph + inductor-compiled inner block
    print("\n[cudagraph + compile inner]", flush=True)
    results.append(bench_one(
        "cudagraph + compile inner", lambda: make_lm(use_neuromod=False),
        configure=None,
        step_fn=phase1_step_cudagraph, B=B,
    ))

    # Summary table
    print("\n=== summary ===", flush=True)
    print(f"  {'config':<32} {'tok/s':>12} {'rel':>8}", flush=True)
    print("  " + "-" * 53, flush=True)
    base = results[0][2]
    for label, _, tps in results:
        print(f"  {label:<32} {tps:>12.1f} {tps/base:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
