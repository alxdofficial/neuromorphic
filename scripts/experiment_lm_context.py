"""Test: how does trajmem step time scale with effective_lm_context?

Currently effective_lm_context=2048 means later windows have growing
LM input (256 → 512 → ... → 2048). Each window pays for its full
context. Hypothesis: most of T1's cost is the rolling buffer, not the
memory module.

Path A: cfg.effective_lm_context = T_window=256 (no rolling buffer)
Path B: cfg.effective_lm_context = 1024 (= 4 windows, same as today's chunk)
Path C: cfg.effective_lm_context = 2048 (= 8 windows, today's default)

Plus --compile vs eager comparison.
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
WARMUP, ITER = 3, 8


def bench_lm_context(eff_context: int, compile_fwd: bool, label: str):
    cfg = TrajMemConfig.medium()
    cfg.effective_lm_context = eff_context
    cfg.validate()

    model = IntegratedLM(cfg, model_name=MODEL_NAME, attach_lm=True).to("cuda")
    if compile_fwd:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )

    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(model, optimizer, scheduler=None, grad_clip=1.0)
    vocab = model.llama.config.vocab_size
    chunk = torch.randint(0, vocab, (BS, cfg.D * cfg.T_window), device="cuda")

    def step():
        trainer.step_wave1(chunk)

    print(f"\n[{label}] eff_context={eff_context}, compile={compile_fwd}")
    # More warmup for compile.
    w = WARMUP * 3 if compile_fwd else WARMUP
    tps, mem, ms = bench(label, step, w, ITER, BS, cfg.D * cfg.T_window)
    del trainer, optimizer, model
    cleanup_cuda()
    return tps, mem, ms


def main():
    torch.set_float32_matmul_precision("high")
    print("=" * 76)
    print("LM CONTEXT WINDOW SWEEP")
    print("=" * 76)
    print(f"  Hardware:  {torch.cuda.get_device_name()}")
    print(f"  Config:    medium (T_window=256, D=4), BS={BS}")
    print()

    results = []
    for ec, compile_fwd, label in [
        (256,  False, "A1 ec=256 eager"),
        (1024, False, "B1 ec=1024 eager"),
        (2048, False, "C1 ec=2048 eager (default)"),
        (256,  True,  "A2 ec=256 compile"),
        (1024, True,  "B2 ec=1024 compile"),
        (2048, True,  "C2 ec=2048 compile"),
    ]:
        try:
            r = bench_lm_context(ec, compile_fwd, label)
            results.append((label, *r))
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:<45} OOM")
            results.append((label, None, None, None))
            cleanup_cuda()

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  {'Path':<35} {'tok/s':>10}  {'GB':>6}  {'ms/iter':>9}")
    for label, tps, mem, ms in results:
        if tps is None:
            print(f"  {label:<35}        OOM")
        else:
            print(f"  {label:<35} {tps/1000:>8.2f}k  {mem:>5.2f}  {ms:>8.1f}")


if __name__ == "__main__":
    main()
