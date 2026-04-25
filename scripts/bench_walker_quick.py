"""Quick bench for one config at a time. No tail buffering; flushes prints.

Usage:
    PYTHONPATH=. python scripts/bench_walker_quick.py [eager|block|cudagraph]
"""

from __future__ import annotations

import sys
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step, phase1_step_cudagraph


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
        compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def main() -> None:
    if not torch.cuda.is_available():
        print("no cuda", flush=True)
        return

    mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
    print(f"=== mode: {mode} ===", flush=True)

    # use_neuromod=False matches what the captured-graph path supports;
    # benchmark all modes that way for an apples-to-apples comparison.
    lm = make_lm(use_neuromod=False)
    cfg = lm.cfg

    if mode == "block":
        lm.memory.compile_block(mode="default")

    B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = cfg.segment_T
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")

    step_fn = phase1_step_cudagraph if mode == "cudagraph" else phase1_step

    print("warmup 3...", flush=True)
    t0 = time.perf_counter()
    for i in range(3):
        step_fn(lm, opt, tokens, training_step=i)
    torch.cuda.synchronize()
    print(f"warmup took {time.perf_counter()-t0:.1f}s", flush=True)

    n_iters = 10
    print(f"timing {n_iters} iters...", flush=True)
    start = time.perf_counter()
    for i in range(n_iters):
        step_fn(lm, opt, tokens, training_step=10 + i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tps = n_iters * B * T / elapsed
    print(f"throughput: {tps:.1f} tok/s   ({elapsed:.2f}s for {n_iters} iters)",
          flush=True)


if __name__ == "__main__":
    main()
