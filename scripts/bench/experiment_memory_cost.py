"""Decompose the T1 vs V1.B slowdown into:
  (A) Memory-module overhead per-se (read + write + bridge)
  (B) "Llama runs 4 times at growing T" overhead (shape cost)
  (C) Backward-only memory overhead

5 paths, each timed at BS=4, T=1024 chunk:

  P0 — vanilla Llama, single forward T=1024, lm_head step (= V1.B)
  P1 — vanilla Llama, 4 forwards at [256,512,768,1024] tokens, lm_head step
       (matches T1's shape WITHOUT memory module)
  P2 — trajmem with scale=0 + read/write skipped (memory module bypassed)
  P3 — trajmem with read/write disabled but bridge active
  P4 — full trajmem (= T1)

Differences:
  P1 - P0 = pure shape cost (multi-forward overhead)
  P4 - P1 = pure memory-module overhead
  P3 - P2 = bridge cross-attn cost
  P4 - P3 = read+write trajectory cost
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import bench, cleanup_cuda  # noqa: E402

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training import (  # noqa: E402
    Phase1Trainer, build_optimizer,
)


MODEL_NAME = "meta-llama/Llama-3.2-1B"
BS = 4
T_FULL = 1024              # full chunk
T_WINDOW = 256             # per-window like medium config
WARMUP, ITER = 3, 8


def path0_vanilla_single_forward(llama, vocab):
    """V1.B equivalent: single forward at T=1024 + lm_head step."""
    input_ids = torch.randint(0, vocab, (BS, T_FULL), device="cuda")
    for p in llama.parameters():
        p.requires_grad = False
    for p in llama.lm_head.parameters():
        p.requires_grad = True
    llama.train(True)
    opt = torch.optim.AdamW([p for p in llama.lm_head.parameters()
                             if p.requires_grad], lr=1e-4, fused=True)

    def step():
        opt.zero_grad(set_to_none=True)
        out = llama(input_ids)
        logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        opt.step()

    print("\n[P0] vanilla Llama single-forward T=1024 + lm_head step (= V1.B)")
    tps, mem, ms = bench("P0 vanilla single-fwd",
                         step, WARMUP, ITER, BS, T_FULL)
    for p in llama.parameters():
        p.requires_grad = False
    llama.train(False)
    del opt
    cleanup_cuda()
    return tps, mem, ms


def path1_vanilla_multi_forward(llama, vocab):
    """Vanilla Llama with 4 forwards at growing context [256, 512, 768, 1024],
    matching T1's per-window shape but no memory module."""
    chunks = [torch.randint(0, vocab, (BS, T_FULL), device="cuda")]
    for p in llama.parameters():
        p.requires_grad = False
    for p in llama.lm_head.parameters():
        p.requires_grad = True
    llama.train(True)
    opt = torch.optim.AdamW([p for p in llama.lm_head.parameters()
                             if p.requires_grad], lr=1e-4, fused=True)

    def step():
        opt.zero_grad(set_to_none=True)
        chunk = chunks[0]
        # 4 forwards at [256, 512, 768, 1024]
        loss = torch.zeros((), device="cuda")
        for d in range(4):
            t_end = (d + 1) * T_WINDOW
            slice_ = chunk[:, :t_end]
            out = llama(slice_)
            # Compute CE only on the new T_WINDOW window's tokens, like T1 does.
            new_logits = out.logits[:, -T_WINDOW:-1, :]
            new_targets = slice_[:, -T_WINDOW + 1:]
            loss = loss + F.cross_entropy(
                new_logits.reshape(-1, new_logits.size(-1)).float(),
                new_targets.reshape(-1),
            )
        loss.backward()
        opt.step()

    print("\n[P1] vanilla Llama 4 forwards at [256,512,768,1024] tokens (no memory)")
    tps, mem, ms = bench("P1 vanilla multi-fwd",
                         step, WARMUP, ITER, BS, T_FULL)
    for p in llama.parameters():
        p.requires_grad = False
    llama.train(False)
    del opt
    cleanup_cuda()
    return tps, mem, ms


def _make_model_with_overrides(scale_zero=False, skip_read_write=False):
    """Build a trajmem model with optional ablations:
      scale_zero: pin MemInjectLayer.scale to zero (memory injection no-op)
      skip_read_write: monkey-patch read/write modules to return zero-tensors
    """
    cfg = TrajMemConfig.medium()
    model = IntegratedLM(cfg, model_name=MODEL_NAME, attach_lm=True).to("cuda")

    if scale_zero:
        # Force scale to all-zero so MemInjectLayer's bypass path triggers.
        # tanh(0)=0, so zeroing scale_raw makes the effective scale zero.
        with torch.no_grad():
            mem_inject = model._mem_inject_layer()
            mem_inject.scale_raw.zero_()

    if skip_read_write:
        # Replace forward methods with zero-returning stubs that match shapes.
        from src.trajectory_memory.read_module import ReadTrajectoryGenerator
        from src.trajectory_memory.write_module import WriteTrajectoryGenerator

        def zero_read(self, prev_window_hiddens, current_states, manifold,
                      *, hard=True):
            BS, T, _ = prev_window_hiddens.shape
            J, K, D = self.cfg.J, self.cfg.K_read, self.cfg.D_concept
            visited = torch.zeros(
                BS, J, K, D,
                dtype=current_states.dtype, device=current_states.device,
            )
            visited_ids = torch.zeros(
                BS, J, K, dtype=torch.int64, device=current_states.device,
            )
            return visited, visited_ids

        def zero_write(self, current_window_hiddens, surprise, prev_states,
                       manifold, *, hard=True):
            BS = prev_states.shape[0]
            J, K = self.cfg.J, self.cfg.K_write
            new_states = prev_states  # pass-through
            visited_ids = torch.zeros(
                BS, J, K, dtype=torch.int64, device=prev_states.device,
            )
            proposed = torch.zeros(
                BS, J, K, self.cfg.D_concept,
                dtype=prev_states.dtype, device=prev_states.device,
            )
            return new_states, visited_ids, proposed

        # Bind to instance.
        import types
        model.read_module.forward = types.MethodType(zero_read, model.read_module)
        model.write_module.forward = types.MethodType(zero_write, model.write_module)

    return model, cfg


def _trajmem_step_fn(model, cfg, vocab):
    """Build a step() closure for a given trajmem model."""
    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(model, optimizer, scheduler=None, grad_clip=1.0)
    chunk = torch.randint(0, vocab, (BS, cfg.D * cfg.T_window), device="cuda")

    def step():
        trainer.step_wave1(chunk)

    return step, trainer, optimizer


def path2_trajmem_no_memory(vocab):
    """Trajmem with scale=0 AND read/write skipped — memory module bypassed."""
    model, cfg = _make_model_with_overrides(scale_zero=True, skip_read_write=True)
    step, trainer, opt = _trajmem_step_fn(model, cfg, vocab)
    print("\n[P2] trajmem with scale=0 + read/write skipped (memory module bypassed)")
    tps, mem, ms = bench("P2 trajmem no-memory",
                         step, WARMUP, ITER, BS, cfg.D * cfg.T_window)
    del trainer, opt, model
    cleanup_cuda()
    return tps, mem, ms


def path3_trajmem_bridge_only(vocab):
    """Trajmem with read/write skipped but bridge cross-attn active.
    Tests cost of MemInjectLayer's cross-attn (with zero KV via skipped read)."""
    model, cfg = _make_model_with_overrides(scale_zero=False, skip_read_write=True)
    step, trainer, opt = _trajmem_step_fn(model, cfg, vocab)
    print("\n[P3] trajmem with read/write skipped, scale=0.1 (bridge active, KV=zeros)")
    tps, mem, ms = bench("P3 trajmem bridge-only",
                         step, WARMUP, ITER, BS, cfg.D * cfg.T_window)
    del trainer, opt, model
    cleanup_cuda()
    return tps, mem, ms


def path4_trajmem_full(vocab):
    """Full trajmem (= T1)."""
    model, cfg = _make_model_with_overrides(scale_zero=False, skip_read_write=False)
    step, trainer, opt = _trajmem_step_fn(model, cfg, vocab)
    print("\n[P4] full trajmem (= T1)")
    tps, mem, ms = bench("P4 trajmem full",
                         step, WARMUP, ITER, BS, cfg.D * cfg.T_window)
    del trainer, opt, model
    cleanup_cuda()
    return tps, mem, ms


def main():
    torch.set_float32_matmul_precision("high")
    print("=" * 76)
    print("DECOMPOSING T1 vs V1.B SLOWDOWN")
    print("=" * 76)
    print(f"  Hardware:    {torch.cuda.get_device_name()}")
    print(f"  BS={BS}, T_full={T_FULL}, T_window={T_WINDOW}, "
          f"warmup={WARMUP}, iter={ITER}")

    # Vanilla paths share llama instance.
    print("\nLoading vanilla Llama once for P0+P1...")
    llama = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to("cuda")
    vocab = llama.config.vocab_size

    p0 = path0_vanilla_single_forward(llama, vocab)
    p1 = path1_vanilla_multi_forward(llama, vocab)

    del llama
    cleanup_cuda()

    p2 = path2_trajmem_no_memory(vocab)
    p3 = path3_trajmem_bridge_only(vocab)
    p4 = path4_trajmem_full(vocab)

    print("\n" + "=" * 76)
    print("DECOMPOSITION SUMMARY")
    print("=" * 76)
    paths = {
        "P0 vanilla single-fwd T=1024 (= V1.B)":    p0,
        "P1 vanilla 4× growing-T forwards (no mem)": p1,
        "P2 trajmem, memory bypassed":               p2,
        "P3 trajmem, read+write skipped (bridge on)":p3,
        "P4 full trajmem (= T1)":                    p4,
    }
    print(f"  {'Path':<48} {'tok/s':>10}  {'GB':>6}  {'ms/iter':>9}")
    for label, (tps, mem, ms) in paths.items():
        print(f"  {label:<48} {tps/1000:>8.2f}k  {mem:>5.2f}  {ms:>8.1f}")

    print("\nDecomposition (ms/iter deltas):")
    print(f"  P0 vanilla baseline:                          {p0[2]:>7.1f} ms")
    print(f"  P1 - P0  (multi-forward shape cost):           +{p1[2] - p0[2]:>6.1f} ms")
    print(f"  P2 - P1  (trajmem trainer overhead):           +{p2[2] - p1[2]:>6.1f} ms")
    print(f"  P3 - P2  (bridge cross-attn cost):             +{p3[2] - p2[2]:>6.1f} ms")
    print(f"  P4 - P3  (read+write trajectory hops):         +{p4[2] - p3[2]:>6.1f} ms")
    print(f"  P4 - P0  (TOTAL T1 vs V1.B gap):               +{p4[2] - p0[2]:>6.1f} ms "
          f"({p4[2]/p0[2]:.2f}× slowdown)")

    # Throughput-form summary
    print("\nMemory-module-only overhead vs P1 (matched-shape vanilla):")
    if p1[0] and p4[0]:
        pure_mem_cost_pct = (p4[2] - p1[2]) / p4[2] * 100
        print(f"  Memory module adds {pure_mem_cost_pct:.1f}% to step time "
              f"(P4 {p4[2]:.0f} ms vs P1 {p1[2]:.0f} ms)")


if __name__ == "__main__":
    main()
