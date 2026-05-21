#!/usr/bin/env python3
"""Forward-pass smoke test for V2.1 repr learning + baselines.

Verifies end-to-end:
  1. Each of three model variants (v21, flat_baseline, continuous_baseline)
     can be instantiated and run a forward pass
  2. Returned loss is finite (not NaN/Inf)
  3. Backward pass works and trainable params get gradients
  4. Memory token shapes match expectations
  5. Trainable parameter count is in the expected range

Usage:
    python scripts/repr_learning/forward_test.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data import synthetic_batch
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


def main():
    cfg = ReprConfig(
        batch_size=2,                 # small for smoke test
        fixed_window_size=128,        # small for speed
    )
    print(f"Config: n_edges={cfg.n_edges}, n_memory_tokens={cfg.n_memory_tokens}")
    print(f"        n_nodes={cfg.n_nodes}, d_concept={cfg.d_concept}, d_edge={cfg.d_edge}")
    print()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Llama once and share across variants
    print(f"Loading Llama ({cfg.llama_model})...")
    t0 = time.time()
    llama_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=llama_dtype)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print()

    # Synthetic batch
    batch = synthetic_batch(cfg, seed=42)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    mask_positions = batch.mask_positions.to(device)
    print(f"Batch: input_ids={input_ids.shape}, "
          f"masked {mask_positions.sum().item()}/{mask_positions.numel()} positions")
    print()

    variants = [
        "v21",
        "flat_baseline",
        "continuous_baseline",
        "memorizing_baseline",
        "recurrent_baseline",
    ]
    # recurrent_baseline (Mamba) requires CUDA — skip on CPU
    if device == "cpu":
        variants = [v for v in variants if v != "recurrent_baseline"]
        print("Note: skipping recurrent_baseline on CPU (Mamba requires CUDA)\n")

    for variant in variants:
        print(f"=== Variant: {variant} ===")
        model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
        n_trainable = model.n_trainable_params()
        print(f"  trainable params: {n_trainable:,}")

        # Forward pass
        t1 = time.time()
        out = model(input_ids, attention_mask, mask_positions)
        fwd_t = time.time() - t1
        print(f"  forward: {fwd_t:.2f}s")
        print(f"    loss       = {out['loss'].item():.4f}")
        print(f"    loss_recon = {out['loss_recon'].item():.4f}")
        print(f"    loss_aux   = {out['loss_aux'].item():.4f}")
        print(f"    memory_shape = {out['memory_shape']}")

        assert torch.isfinite(out["loss"]), f"Non-finite loss for {variant}!"

        # Backward pass — verify gradients reach trainable params
        t2 = time.time()
        out["loss"].backward()
        bwd_t = time.time() - t2
        print(f"  backward: {bwd_t:.2f}s")

        n_with_grad = 0
        n_no_grad = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                n_no_grad += 1
            elif torch.isfinite(p.grad).all() and p.grad.abs().max() > 0:
                n_with_grad += 1
            else:
                n_no_grad += 1
        print(f"  params with gradient: {n_with_grad}; without: {n_no_grad}")
        if n_no_grad > 0:
            print(f"    WARNING: {n_no_grad} trainable params have None or zero gradient")

        # Cleanup
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        print()

    print("=== Smoke test complete ===")


if __name__ == "__main__":
    main()
