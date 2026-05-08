"""Shared helpers for bench_phase1 / bench_phase2.

Defines the walker config CLI surface (production default + target-config
preset + per-knob overrides) and a single timing primitive that handles
warmup, OOM cleanup, and peak VRAM.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from src.graph_walker.config import GraphWalkerConfig


# Production walker config: what's on main today.
# Capacity bump 2026-05-08: N=1024→2048, K=32→64, p_rewire=0.3→0.5,
# radius=3→4. Per-token compute scales linearly in K (slightly slower)
# but N is "free capacity" (walkers visit B·H cols per step regardless
# of N). p_rewire=0.5 keeps the locality prior while shortening graph
# diameter further; radius=4 supports the K=64 sampling.
PRODUCTION_KNOBS = dict(
    grid_rows=64, grid_cols=64, radius=4, p_rewire=0.5,
    K=64, D_s=256, D_id=512, D_model=256,
    content_mlp_depth=4, D_hid_content=1024,
    post_model_depth=2,
    n_heads=4, n_hops=4,
    n_score_heads=4, D_q_per_head=64,
    K_horizons=8, K_buf=8,
    neuromod_D_mod=512, neuromod_n_layers=6, neuromod_n_heads=8,
    neuromod_edge_hidden=384,
)

# Target walker config (~110M trainable). Per memory/walker_target_config.md.
# Currently breaks cudagraph compile; runnable in eager mode.
TARGET_KNOBS = dict(
    grid_rows=64, grid_cols=64, radius=5,         # N = 4096
    K=96, D_s=768, D_id=512, D_model=1024,
    content_mlp_depth=4, D_hid_content=1024,
    post_model_depth=2,
    n_heads=4, n_hops=4,
    n_score_heads=4, D_q_per_head=64,
    K_horizons=8, K_buf=8,
    neuromod_D_mod=512, neuromod_n_layers=6, neuromod_n_heads=8,
    neuromod_edge_hidden=384,
)


def add_walker_config_args(ap: argparse.ArgumentParser) -> None:
    """Walker config CLI surface. Use --target-config to load the ~110M
    preset, OR override individual knobs to sweep."""
    ap.add_argument("--target-config", action="store_true",
                    help="Load the ~110M target walker config preset.")
    # Topology
    ap.add_argument("--grid-rows", type=int, default=None)
    ap.add_argument("--grid-cols", type=int, default=None)
    ap.add_argument("--radius", type=int, default=None,
                    help="Moore-radius for neighbor candidate sampling")
    ap.add_argument("--walker-K", type=int, default=None,
                    help="Out-edges per column (walker topology)")
    # Widths
    ap.add_argument("--D-s", type=int, default=None,
                    help="Walker column-state dim (must equal d_mem)")
    ap.add_argument("--D-id", type=int, default=None)
    ap.add_argument("--D-model", type=int, default=None)
    ap.add_argument("--content-depth", type=int, default=None,
                    help="Number of ResidualFFN blocks in content_mlp")
    ap.add_argument("--D-hid-content", type=int, default=None)
    ap.add_argument("--post-model-depth", type=int, default=None)
    # Heads / scoring
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-hops", type=int, default=None)
    # Neuromod
    ap.add_argument("--neuromod-D-mod", type=int, default=None)
    ap.add_argument("--neuromod-n-layers", type=int, default=None)
    ap.add_argument("--neuromod-n-heads", type=int, default=None)
    ap.add_argument("--neuromod-edge-hidden", type=int, default=None)
    # Compile
    ap.add_argument("--compile-block", action="store_true",
                    help="Compile the walker block via torch.compile.")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode for compile_walk_block.")
    ap.add_argument("--regional-compile", action="store_true",
                    help="Use regional compilation: compile walker_step_from_h "
                         "instead of the whole walk_segment block. ~10x faster "
                         "first compile (1-2 min vs 10-15 min at T=256), at "
                         "~5-15%% lower per-iter throughput. Recommended for "
                         "dev iteration; flip off for final production runs.")
    ap.add_argument("--dynamic-shapes", action="store_true",
                    help="Pass dynamic=None to torch.compile (auto-detect "
                         "shape variation). After the 2nd shape inductor "
                         "compiles a shape-polymorphic kernel — useful for "
                         "BS sweeps where you'd otherwise pay compile cost "
                         "per BS. Cannot be combined with cudagraph "
                         "(--compile-mode=reduce-overhead requires fixed shapes).")


def walker_cfg_from_args(args: argparse.Namespace, T: int, vocab: int) -> GraphWalkerConfig:
    """Build a GraphWalkerConfig from CLI args. Production preset by
    default; --target-config swaps to ~110M target; per-knob CLI args
    override either preset."""
    knobs = dict(TARGET_KNOBS if args.target_config else PRODUCTION_KNOBS)
    # Per-arg overrides (only set if user supplied)
    overrides = {
        "grid_rows": args.grid_rows, "grid_cols": args.grid_cols, "radius": args.radius,
        "K": args.walker_K, "D_s": args.D_s, "D_id": args.D_id, "D_model": args.D_model,
        "content_mlp_depth": args.content_depth,
        "D_hid_content": args.D_hid_content,
        "post_model_depth": args.post_model_depth,
        "n_heads": args.n_heads, "n_hops": args.n_hops,
        "neuromod_D_mod": args.neuromod_D_mod,
        "neuromod_n_layers": args.neuromod_n_layers,
        "neuromod_n_heads": args.neuromod_n_heads,
        "neuromod_edge_hidden": args.neuromod_edge_hidden,
    }
    for k, v in overrides.items():
        if v is not None:
            knobs[k] = v
    # Integration constraint: T == segment_T == mod_period == tbptt_block.
    knobs["segment_T"] = T
    knobs["mod_period"] = T
    knobs["tbptt_block"] = T
    knobs["vocab_size"] = vocab
    knobs["plasticity_mode"] = "neuromod_only"
    knobs["compile_on_train"] = False
    return GraphWalkerConfig(**knobs)


def print_config_summary(walker_cfg: GraphWalkerConfig, label: str) -> None:
    print(f"  walker config ({label}):")
    print(f"    N = {walker_cfg.N} ({walker_cfg.grid_rows}x{walker_cfg.grid_cols}), K = {walker_cfg.K}, radius = {walker_cfg.radius}")
    print(f"    D_s = {walker_cfg.D_s}, D_id = {walker_cfg.D_id}, D_model = {walker_cfg.D_model}")
    print(f"    content depth = {walker_cfg.content_mlp_depth}, D_hid_content = {walker_cfg.D_hid_content}")
    print(f"    n_heads = {walker_cfg.n_heads}, n_hops = {walker_cfg.n_hops}")
    print(f"    neuromod = {walker_cfg.neuromod_n_layers}L/{walker_cfg.neuromod_n_heads}H, "
          f"D_mod = {walker_cfg.neuromod_D_mod}, edge_hidden = {walker_cfg.neuromod_edge_hidden}")


def bench(name: str, fn, n_warmup: int, n_iter: int, BS: int, T: int):
    """Time a callable. Returns (tok_per_sec, peak_gb, ms_per_iter) or
    (None, None, None) on OOM. OOM cleanup is critical so the next bench
    in the same process starts with a clean slate."""
    try:
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"  {name:42s}    OOM        peak n/a       BS={BS} T={T}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return None, None, None
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tps = (BS * T * n_iter) / elapsed
    ms = elapsed / n_iter * 1000
    print(f"  {name:42s} {tps/1000:6.1f}k tok/s   peak {peak_gb:5.2f} GB   "
          f"{ms:6.1f} ms/iter")
    return tps, peak_gb, ms


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
