"""Benchmark walker step under {eager, default+fg=False, default+fg=True}.

Each config runs N warmup + N timed iterations of full mod_period segments.
Reports tokens/sec; the spread tells us whether fullgraph=True alone gives
a meaningful win even without reduce-overhead's CUDA graphs.

If fullgraph=True is significantly faster than the current production
(fullgraph=False), we ship that as Phase A.1 immediately and revisit
reduce-overhead only if the gap to Llama-1B is still too wide.
"""

from __future__ import annotations

import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


def make_lm() -> StandaloneLM:
    """Realistic mid-size config matching the production hot path."""
    cfg = GraphWalkerConfig(
        # Match real defaults but a bit smaller to keep bench short.
        plane_rows=8, plane_cols=8, L=4, K=16,
        D_model=512, D_s=256, D_id=32,
        n_heads=4, n_score_heads=4, D_q_per_head=32, D_q_in=32,
        K_horizons=4, K_buf=4,
        mod_period=64, tbptt_block=64, segment_T=64,
        ffn_mult_content=4,
        content_mlp_depth=2,
        post_model_depth=1,
        vocab_size=1024,
        use_neuromod=True,
        compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def time_segment(lm: StandaloneLM, B: int, n_iters: int,
                 amp_dtype: torch.dtype) -> float:
    """Run n_iters segments of segment_T tokens; return tokens/sec."""
    cfg = lm.cfg
    device = next(lm.parameters()).device
    T = cfg.segment_T
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def run_segment() -> None:
        lm.memory.begin_segment(B, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for t in range(T):
                torch.compiler.cudagraph_mark_step_begin()
                r = lm.memory.step_core(tokens[:, t])
                # Clone immediately to take the tensor out of the cudagraph
                # pool — subsequent replays may otherwise overwrite the buffer
                # before we read it during backward.
                ms = r.motor_state.clone()
                loss = loss + ms.float().pow(2).mean()
        loss.backward()

    # Warmup
    for _ in range(3):
        run_segment()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        run_segment()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return n_iters * B * T / elapsed


def setup_compile(lm: StandaloneLM, mode: str | None, fullgraph: bool) -> None:
    lm.memory._compiled_step = None
    if mode is None:
        return
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, 64,
    )
    lm.memory._compiled_step = torch.compile(
        lm.memory._step_core_pure, mode=mode, fullgraph=fullgraph,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    B = 4
    N_ITERS = 5

    configs: list[tuple[str, str | None, bool]] = [
        ("eager",            None,              False),
        ("default+fg=F",     "default",         False),  # current production
        ("default+fg=T",     "default",         True),
        ("reduce-overhead",  "reduce-overhead", False),  # CUDA graphs, fg=F
        ("ro+fg=T",          "reduce-overhead", True),   # CUDA graphs, fg=T
    ]

    print(f"\n=== Walker compile-mode bench (B={B}, T=64, mod_period=64) ===\n")
    print(f"{'config':<20} {'tok/s':>10} {'rel':>8}")
    print("-" * 42)
    baseline = None
    for label, mode, fullgraph in configs:
        lm = make_lm()
        setup_compile(lm, mode, fullgraph)
        try:
            tps = time_segment(lm, B, N_ITERS, torch.bfloat16)
        except Exception as e:
            print(f"{label:<20} FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        if baseline is None:
            baseline = tps
        print(f"{label:<20} {tps:>10.1f} {tps/baseline:>7.2f}x")
        del lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
