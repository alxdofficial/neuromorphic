"""Bench the walker STANDALONE at the EXACT same config the Llama
integration uses, so we can isolate "walker is slow" from "Llama+walker
is slow because of Llama".

Mirrors `_walker_cfg_for(d_mem=512, T=256)` from
`scripts/bench_pretrained_gw.py`. Runs `walk_segment` over a synthetic
h_mem (shape [B, T, D_s]) and backwards a scalar loss, with optional
`compile_walk_block_from_h` for inductor whole-block fusion (the same compile
the integration's `--compile-block` uses).

Output: one row per BS in `--bs-list`, throughput in tok/s + peak GB,
plus comparison to the integration bench's published GW step numbers
(2.6k tok/s @ BS=4 with --compile-block).
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.graph_walker import GraphWalkerMemory
import torch.nn as nn


def _integration_cfg(
    T: int = 256, d_mem: int = 256, *, use_neuromod: bool = True,
) -> GraphWalkerConfig:
    """PRODUCTION walker config — mirrors the (now production-aligned)
    `_walker_cfg_for` in `scripts/bench_pretrained_gw.py`. All knobs come
    from `GraphWalkerConfig()` defaults; only the few that must vary for
    the integration (vocab, segment_T, mod_period, D_s=d_mem) are overridden.

    `use_neuromod=False` is the cudagraph-compatible variant — disables
    neuromod's per-window grad-carrying delta so the activation memory
    drops dramatically (no graph-transformer per window). Matches the
    constraint of `CapturedBlockTrainer`.
    """
    mod_period = 128
    while T % mod_period != 0 and mod_period > 1:
        mod_period //= 2
    return GraphWalkerConfig(
        D_s=d_mem,
        D_model=d_mem,
        vocab_size=128_256,
        segment_T=T,
        mod_period=mod_period,
        tbptt_block=mod_period,
        compile_on_train=False,
        use_neuromod=use_neuromod,
    )


def _build(cfg: GraphWalkerConfig) -> GraphWalkerMemory:
    emb = nn.Embedding(cfg.vocab_size, cfg.D_model)
    nn.init.normal_(emb.weight, std=0.02)
    m = GraphWalkerMemory(cfg, tied_token_emb=emb).cuda()
    # Match the integration's frozen-in-the-integration set so trainable
    # surface area (and thus backward cost) is comparable.
    m.token_to_state.weight.requires_grad = False
    return m


def bench_one(B: int, T: int, *, compile_walk_block: bool, use_neuromod: bool,
              n_warmup: int, n_iter: int) -> dict:
    cfg = _integration_cfg(T=T, use_neuromod=use_neuromod)
    m = _build(cfg)
    if compile_walk_block:
        m.compile_walk_block_from_h(mode="default", fullgraph=True)
    opt = torch.optim.AdamW(
        [p for _, p in m.named_parameters() if p.requires_grad], lr=1e-4,
        fused=True,
    )

    device = torch.device("cuda")
    # h_mem comes from W_in(llama_hidden); we synthesise it as a leaf
    # tensor with requires_grad=True so backward through the walker
    # propagates upstream just like in the integration path (where
    # h_mem.grad would flow back into Llama).
    h_mem = torch.randn(B, T, cfg.D_s, device=device, requires_grad=True)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def step():
        m.begin_segment(B, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            readouts = m.walk_segment(h_mem, preserve_graph=False)
            # Standalone bench has no Llama upstream, so we use a
            # sum-of-readouts proxy for the downstream gradient that Llama
            # would otherwise inject through W_out in the real path.
            loss = readouts.float().pow(2).mean()
        loss.backward()
        opt.step()
        # Synthetic surprise: random per-token "CE" so plasticity still
        # runs end-to-end at bench representative shapes.
        with torch.no_grad():
            fake_surprise = torch.randn(B, T, device=device).abs()
        m.update_plasticity(fake_surprise)
        m.detach_state()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tps = B * T * n_iter / elapsed
    return {"tps": tps, "peak_gb": peak_gb, "ms_per_iter": elapsed / n_iter * 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs-list", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--no-compile-block", action="store_true",
                    help="Disable compile_walk_block_from_h (eager comparison).")
    ap.add_argument("--no-neuromod", action="store_true",
                    help="Disable use_neuromod. Matches the cudagraph-compatible "
                         "variant in CapturedBlockTrainer; cuts activation "
                         "memory significantly (no per-window graph-transformer).")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iter", type=int, default=10)
    args = ap.parse_args()

    cfg = _integration_cfg(T=args.T, use_neuromod=not args.no_neuromod)
    n_params = sum(p.numel() for p in _build(cfg).parameters() if p.requires_grad) / 1e6
    print(f"=== Standalone walker bench at INTEGRATION config ===")
    print(f"  N={cfg.N}, K={cfg.K}, D_s={cfg.D_s}, D_id={cfg.D_id}, "
          f"D_hid_content={cfg.D_hid_content}, H={cfg.n_heads}, T={args.T}")
    print(f"  use_neuromod={cfg.use_neuromod}, mod_period={cfg.mod_period}, "
          f"tbptt={cfg.tbptt_block}")
    print(f"  trainable params: {n_params:.1f}M")
    print(f"  compile_walk_block: {not args.no_compile_block}")
    print(f"  Reference (integration bench, --compile-block):")
    print(f"    BS=4 GW step → 2.6k tok/s at 12.22 GB peak")
    print()
    print(f"  {'BS':>4}  {'tok/s':>8}  {'peak GB':>8}  {'ms/iter':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for B in args.bs_list:
        try:
            r = bench_one(
                B, args.T,
                compile_walk_block=not args.no_compile_block,
                use_neuromod=not args.no_neuromod,
                n_warmup=args.warmup, n_iter=args.iter,
            )
            print(f"  {B:>4}  {r['tps']/1000:>7.1f}k  {r['peak_gb']:>8.2f}  "
                  f"{r['ms_per_iter']:>7.1f}", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                print(f"  {B:>4}    OOM", flush=True)
            else:
                raise
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
