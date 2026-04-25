"""Bench phase1_step end-to-end with block_forward compile.

Compares:
- eager (compile_on_train=False)
- per-step compile (legacy, compile_step)
- whole-block compile (new, compile_block)

End-to-end means: full phase1_step including readout, plasticity, opt.step.
This is what training actually runs.
"""

from __future__ import annotations

import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def make_lm(use_neuromod: bool = True) -> StandaloneLM:
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
        compile_on_train=False,                # we drive compile manually
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def time_phase1(lm: StandaloneLM, B: int, n_iters: int,
                compile_kind: str | None) -> float:
    cfg = lm.cfg
    device = next(lm.parameters()).device
    T = cfg.segment_T

    if compile_kind == "step":
        lm.memory.compile_step()
    elif compile_kind == "block":
        lm.memory.compile_block(mode="default")
    elif compile_kind == "block-reduce-overhead":
        lm.memory.compile_block(mode="reduce-overhead")
    elif compile_kind == "block-max-autotune":
        lm.memory.compile_block(mode="max-autotune-no-cudagraphs")

    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # Warmup
    for i in range(3):
        phase1_step(lm, opt, tokens, training_step=i)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(n_iters):
        phase1_step(lm, opt, tokens, training_step=10 + i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return n_iters * B * T / elapsed


def main() -> None:
    if not torch.cuda.is_available():
        return

    B = 4
    N_ITERS = 5

    configs = [
        ("eager",                              None),
        ("whole-block compile (default)",      "block"),
        ("whole-block (max-autotune)",         "block-max-autotune"),
    ]

    print(f"\n=== phase1_step bench (B={B}, T={128}, mod_period=64) ===\n")
    print(f"{'config':<25} {'tok/s':>10} {'rel':>8}")
    print("-" * 45)
    baseline = None
    for label, kind in configs:
        lm = make_lm()
        try:
            tps = time_phase1(lm, B, N_ITERS, kind)
        except Exception as e:
            print(f"{label:<25} FAILED: {type(e).__name__}: {str(e)[:60]}")
            import traceback
            traceback.print_exc()
            del lm
            torch.cuda.empty_cache()
            continue
        if baseline is None:
            baseline = tps
        print(f"{label:<25} {tps:>10.1f} {tps/baseline:>7.2f}x")
        del lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
