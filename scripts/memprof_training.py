"""Memory profile of one TRAINING step (not synthetic forward/backward).

Runs the real training-step entry point — `phase1_pretrained_step` for the
integration target, walker-only `phase1_step` for the standalone target —
under bf16 autocast with the production walker config and AdamW. Records
GPU allocations via `torch.cuda.memory._record_memory_history` and writes
the snapshot to disk for inspection at https://pytorch.org/memory_viz.

Also prints `torch.cuda.memory_stats` highlights (active / reserved / peak)
after the step so you get a quick text summary even without the visualizer.

Pair this with `scripts/memprof_analyze.py` for an in-terminal top-N call
site breakdown derived from the same snapshot.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/memprof_training.py \\
        --target integration --bs 4
    PYTHONPATH=. .venv/bin/python scripts/memprof_training.py \\
        --target standalone --bs 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def _print_stats(label: str) -> None:
    """Quick text summary of allocator state."""
    s = torch.cuda.memory_stats()
    active = s.get("active_bytes.all.current", 0) / 1e9
    peak = s.get("active_bytes.all.peak", 0) / 1e9
    reserved = s.get("reserved_bytes.all.peak", 0) / 1e9
    print(f"  {label:<30s}  active {active:>6.2f} GB  "
          f"peak {peak:>6.2f} GB  reserved {reserved:>6.2f} GB", flush=True)


# ----------------------------------------------------------------
# Target: integration (Llama + walker, phase1_pretrained_step)
# ----------------------------------------------------------------

def run_integration(bs: int, T: int, snapshot_path: Path) -> None:
    from src.graph_walker.pretrained.config import PretrainedGWConfig
    from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
    from src.graph_walker.pretrained.train_phase1 import (
        Phase1Batch, phase1_pretrained_step,
    )
    from scripts.bench_pretrained_gw import _walker_cfg_for

    device = torch.device("cuda")
    print(f"  Loading vanilla Llama-3.2-1B (frozen backbone)...", flush=True)
    walker_cfg = _walker_cfg_for(d_mem=256, T=T)
    cfg = PretrainedGWConfig(
        model_name="meta-llama/Llama-3.2-1B",
        inject_layer=8,
        d_mem=256,
        memory=walker_cfg,
        T=T, bs=bs,
        llama_dtype="bf16",
    )
    wrapper = GraphWalkerPretrainedLM(cfg).to(device)
    wrapper.train(True)
    wrapper.compile_walker_block()

    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-4,
        fused=True,
    )
    input_ids = torch.randint(0, cfg.vocab_size_lm, (bs, T), device=device)
    targets = input_ids.clone()
    batch = Phase1Batch(input_ids=input_ids, target_ids=targets)

    print(f"  Warmup (compile + 2 steps to settle)...", flush=True)
    for _ in range(2):
        phase1_pretrained_step(wrapper, opt, batch, amp_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _print_stats("after warmup")

    torch.cuda.memory._record_memory_history(
        enabled='all', max_entries=200_000, stacks='python', context='all',
    )

    print(f"\n  Profiling one phase1_pretrained_step (BS={bs}, T={T})...",
          flush=True)
    t = time.perf_counter()
    stats = phase1_pretrained_step(wrapper, opt, batch, amp_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t) * 1000.0
    _print_stats(f"after profiled step ({elapsed_ms:.0f}ms)")

    print(f"\n  Loss components: ce={stats.ce_loss:.3f}  "
          f"aux={stats.aux_loss:.3f}  grad_norm={stats.grad_norm:.3f}",
          flush=True)

    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"\n  Snapshot written → {snapshot_path}")
    print(f"  Open in https://pytorch.org/memory_viz for flame-graph view.")
    print(f"  Or run: scripts/memprof_analyze.py {snapshot_path}")


# ----------------------------------------------------------------
# Target: standalone walker (no Llama, walker phase1_step)
# ----------------------------------------------------------------

def run_standalone(bs: int, T: int, snapshot_path: Path) -> None:
    from src.graph_walker.config import GraphWalkerConfig
    from src.graph_walker.standalone import StandaloneLM
    from src.graph_walker.train_phase1 import phase1_step

    device = torch.device("cuda")
    mod_period = 128
    while T % mod_period != 0 and mod_period > 1:
        mod_period //= 2
    cfg = GraphWalkerConfig(
        D_s=256, D_model=256,
        vocab_size=128_256,
        segment_T=T, mod_period=mod_period, tbptt_block=mod_period,
        compile_on_train=False,
    )
    print(f"  Standalone walker  N={cfg.N}  D_s={cfg.D_s}  "
          f"D_id={cfg.D_id}  use_neuromod={cfg.use_neuromod}", flush=True)
    lm = StandaloneLM(cfg).to(device)
    lm.train(True)
    lm.memory.compile_step()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4, fused=True)

    tokens = torch.randint(0, cfg.vocab_size, (bs, T), device=device)

    print(f"  Warmup (compile + 2 steps to settle)...", flush=True)
    for i in range(2):
        phase1_step(lm, opt, tokens, tbptt_block=mod_period,
                    amp_dtype=torch.bfloat16, training_step=i)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _print_stats("after warmup")

    torch.cuda.memory._record_memory_history(
        enabled='all', max_entries=200_000, stacks='python', context='all',
    )

    print(f"\n  Profiling one phase1_step (BS={bs}, T={T})...", flush=True)
    t = time.perf_counter()
    stats = phase1_step(lm, opt, tokens, tbptt_block=mod_period,
                       amp_dtype=torch.bfloat16, training_step=99)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t) * 1000.0
    _print_stats(f"after profiled step ({elapsed_ms:.0f}ms)")
    print(f"  ce_loss={stats.ce_loss:.3f}  grad_norm={stats.grad_norm:.3f}",
          flush=True)

    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"\n  Snapshot written → {snapshot_path}")
    print(f"  Open in https://pytorch.org/memory_viz for flame-graph view.")
    print(f"  Or run: scripts/memprof_analyze.py {snapshot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("integration", "standalone"),
                    required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--snapshot",
                    default="/tmp/walker_memprof.pkl",
                    help="Path to write the memory-history pickle.")
    args = ap.parse_args()

    snapshot = Path(args.snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this profile.")

    print(f"=== Memory profile: target={args.target}  BS={args.bs}  T={args.T} ===",
          flush=True)
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"snapshot: {snapshot}", flush=True)
    print()

    if args.target == "integration":
        run_integration(args.bs, args.T, snapshot)
    else:
        run_standalone(args.bs, args.T, snapshot)


if __name__ == "__main__":
    main()
