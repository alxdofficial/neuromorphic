"""Test torch.compile mode='default' dynamic=False vs dynamic=True at the
varying-LM-context shapes that real training will see.

The state-threaded W1 trainer feeds Llama lm_input_ids of varying length
within a chunk (256 → 512 → 768 → 1024) and across chunks (rolling buffer
fills toward the cap). dynamic=False forces a recompile per shape and
hits the 8-shape recompile limit; dynamic=True compiles a single graph
that handles varying shapes.

Test paths:
  X1 — eager (no compile)
  X2 — compile dynamic=False (current default in train_wave1.py)
  X3 — compile dynamic=True
  X4 — compile dynamic=True with mode='reduce-overhead'  (cudagraphs if shapes settle)
"""

from __future__ import annotations

import sys
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
WARMUP, ITER = 5, 8


def bench_one(label: str, *, compile_mode: str | None,
              dynamic: bool, ec: int = 2048):
    cfg = TrajMemConfig.medium()
    cfg.effective_lm_context = ec
    cfg.validate()

    model = IntegratedLM(cfg, model_name=MODEL_NAME, attach_lm=True).to("cuda")
    if compile_mode is not None:
        model.forward_window = torch.compile(
            model.forward_window, mode=compile_mode, dynamic=dynamic,
        )

    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(model, optimizer, scheduler=None, grad_clip=1.0)
    vocab = model.llama.config.vocab_size
    chunk = torch.randint(0, vocab, (BS, cfg.D * cfg.T_window), device="cuda")

    def step():
        trainer.step_wave1(chunk)

    print(f"\n[{label}] ec={ec} compile={compile_mode} dynamic={dynamic}")
    w = WARMUP * 4 if compile_mode is not None else WARMUP
    tps, mem, ms = bench(label, step, w, ITER, BS, cfg.D * cfg.T_window)
    del trainer, optimizer, model
    cleanup_cuda()
    return tps, mem, ms


def main():
    torch.set_float32_matmul_precision("high")
    print("=" * 76)
    print("torch.compile DYNAMIC SHAPES SWEEP at ec=2048 (production setting)")
    print("=" * 76)
    print(f"  Hardware: {torch.cuda.get_device_name()}")
    print(f"  Config:   medium (T_window=256, D=4), BS={BS}, ec=2048")
    print()

    results = []
    for label, mode, dyn in [
        ("X1 eager",                    None,      False),
        ("X2 compile default static",   "default", False),
        ("X3 compile default dynamic",  "default", True),
        ("X4 compile reduce-ovhd dyn",  "reduce-overhead", True),
    ]:
        try:
            r = bench_one(label, compile_mode=mode, dynamic=dyn)
            results.append((label, *r))
        except Exception as e:
            print(f"  {label} FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append((label, None, None, None))
            cleanup_cuda()

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  {'Path':<35} {'tok/s':>10}  {'GB':>6}  {'ms/iter':>9}")
    for label, tps, mem, ms in results:
        if tps is None:
            print(f"  {label:<35}        FAILED")
        else:
            print(f"  {label:<35} {tps/1000:>8.2f}k  {mem:>5.2f}  {ms:>8.1f}")


if __name__ == "__main__":
    main()
