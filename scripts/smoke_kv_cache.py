"""Smoke test + bench for the KV cache path.

Verifies:
  - Cache mode runs end-to-end without crashing
  - Loss values are similar to rolling-buffer mode (sanity check —
    they won't be bit-identical because of cache vs re-encode noise,
    but should be in the same ballpark)
  - Speed comparison

Run:
    PYTHONPATH=. python scripts/smoke_kv_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import bench, cleanup_cuda  # noqa: E402

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training import (  # noqa: E402
    Phase1Trainer, build_optimizer,
)


MODEL_NAME = "meta-llama/Llama-3.2-1B"
BS = 4
WARMUP, ITER = 3, 8


def _build(use_kv_cache: bool):
    cfg = TrajMemConfig.medium()
    model = IntegratedLM(cfg, model_name=MODEL_NAME, attach_lm=True).to("cuda")
    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(
        model, optimizer, scheduler=None, grad_clip=1.0,
        use_kv_cache=use_kv_cache,
    )
    return cfg, model, optimizer, trainer


def main():
    torch.set_float32_matmul_precision("high")
    print("=" * 76)
    print("KV CACHE SMOKE + BENCH")
    print("=" * 76)
    print(f"  Hardware:  {torch.cuda.get_device_name()}")
    print(f"  Config:    medium (T_window=256, D=4), BS={BS}")
    print()

    # ── Smoke: rolling buffer + KV cache should produce similar loss ──
    print("[smoke] both paths should run a step without crashing")
    cfg, model_ref, opt_ref, trainer_ref = _build(use_kv_cache=False)
    vocab = model_ref.llama.config.vocab_size
    chunk = torch.randint(0, vocab, (BS, cfg.D * cfg.T_window), device="cuda")

    # Capture initial state for fair comparison.
    state_dict = {k: v.clone() for k, v in model_ref.state_dict().items()
                  if v.is_floating_point()}

    # Run rolling-buffer once.
    metrics_ref = trainer_ref.step_wave1(chunk)
    print(f"  rolling-buffer step: loss={metrics_ref.loss:.4f}, "
          f"grad_norm={metrics_ref.grad_norm:.3f}")
    del trainer_ref, opt_ref, model_ref
    cleanup_cuda()

    # Reset and run KV-cache once.
    cfg, model_kv, opt_kv, trainer_kv = _build(use_kv_cache=True)
    # Restore initial state so comparison is fair.
    with torch.no_grad():
        msd = model_kv.state_dict()
        for k, v in state_dict.items():
            if k in msd:
                msd[k].copy_(v)

    metrics_kv = trainer_kv.step_wave1(chunk)
    print(f"  kv-cache step:       loss={metrics_kv.loss:.4f}, "
          f"grad_norm={metrics_kv.grad_norm:.3f}")
    print(f"  loss delta:          {abs(metrics_kv.loss - metrics_ref.loss):.4f} "
          f"({abs(metrics_kv.loss - metrics_ref.loss) / max(metrics_ref.loss, 1e-9) * 100:.1f}%)")
    del trainer_kv, opt_kv, model_kv
    cleanup_cuda()

    # ── Bench: speed comparison at BS=4 ──
    print("\n[bench] speed comparison at BS=4 medium")
    cfg, model_ref, opt_ref, trainer_ref = _build(use_kv_cache=False)
    chunk = torch.randint(0, vocab, (BS, cfg.D * cfg.T_window), device="cuda")
    def step_ref(): trainer_ref.step_wave1(chunk)
    print("\n[REF] rolling-buffer eager")
    tps_ref, mem_ref, ms_ref = bench("rolling-buffer", step_ref, WARMUP, ITER, BS, cfg.D * cfg.T_window)
    del trainer_ref, opt_ref, model_ref
    cleanup_cuda()

    cfg, model_kv, opt_kv, trainer_kv = _build(use_kv_cache=True)
    def step_kv(): trainer_kv.step_wave1(chunk)
    print("\n[KV] kv-cache eager (single chunk; cache not yet at cap)")
    tps_kv, mem_kv, ms_kv = bench("kv-cache", step_kv, WARMUP, ITER, BS, cfg.D * cfg.T_window)
    del trainer_kv, opt_kv, model_kv
    cleanup_cuda()

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  {'Path':<25} {'tok/s':>10}  {'GB':>6}  {'ms/iter':>9}")
    print(f"  {'rolling-buffer':<25} {tps_ref/1000:>8.2f}k  {mem_ref:>5.2f}  {ms_ref:>8.1f}")
    print(f"  {'kv-cache':<25} {tps_kv/1000:>8.2f}k  {mem_kv:>5.2f}  {ms_kv:>8.1f}")
    if tps_ref and tps_kv:
        print(f"\n  KV-cache speedup: {tps_kv/tps_ref:.2f}× ({ms_ref - ms_kv:+.1f} ms saved)")
        print(f"  Memory delta:     {mem_kv - mem_ref:+.2f} GB")


if __name__ == "__main__":
    main()
