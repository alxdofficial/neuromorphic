#!/usr/bin/env python3
"""Detailed correctness sweep for V2.1 + 4 baselines.

Beyond the basic forward_test (does it run without crashing?), this
verifies subtler invariants that a correct implementation must satisfy:

  1. Forward determinism - same input + same seed -> same output
  2. Input-sensitivity - different inputs -> meaningfully different memory
  3. Memory magnitude - memory tokens have reasonable norm
  4. Gradient magnitude - all trainable params get reasonable gradients
  5. Memory dispersion - memory tokens span the manifold (not collapsed)
  6. Mask sensitivity - flipping mask positions changes the loss

Failures here indicate implementation bugs the basic smoke test misses.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data import synthetic_batch
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


COLORS = {"pass": "\033[32m", "fail": "\033[31m", "warn": "\033[33m", "end": "\033[0m"}


def report(check_name: str, status: str, detail: str = ""):
    color = COLORS.get(status, "")
    end = COLORS["end"] if color else ""
    symbol = {"pass": "[OK]", "fail": "[FAIL]", "warn": "[WARN]"}.get(status, "[?]")
    line = f"  {color}{symbol} {check_name:<50}{end}"
    if detail:
        line += f" - {detail}"
    print(line)


def run_variant_checks(variant: str, model, batch, device) -> dict:
    results = {}
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    mask_positions = batch.mask_positions.to(device)

    # Check 1: forward determinism
    model.train(False)
    with torch.no_grad():
        out1 = model(input_ids, attention_mask, mask_positions)
        out2 = model(input_ids, attention_mask, mask_positions)
    loss_diff = (out1["loss"] - out2["loss"]).abs().item()
    if loss_diff < 1e-5:
        report("forward determinism (inference mode)", "pass",
               f"dloss = {loss_diff:.2e}")
        results["determinism"] = "pass"
    else:
        report("forward determinism (inference mode)", "fail",
               f"dloss = {loss_diff:.2e}")
        results["determinism"] = "fail"

    # Check 2: input sensitivity
    perturbed_ids = input_ids.clone()
    perturbed_ids[0] = perturbed_ids[0].flip(0)
    with torch.no_grad():
        embed = model.decoder.llama.get_input_embeddings()
        mem1 = model.encoder(embed(input_ids), attention_mask,
                             mask_positions=mask_positions)[0]
        mem_p = model.encoder(embed(perturbed_ids), attention_mask,
                              mask_positions=mask_positions)[0]
        diff = (mem1[0] - mem_p[0]).abs().mean().item()
    if diff > 1e-4:
        report("input sensitivity", "pass", f"mean |dmem| = {diff:.4f}")
        results["input_sensitivity"] = "pass"
    else:
        report("input sensitivity", "fail", f"mean |dmem| = {diff:.2e}")
        results["input_sensitivity"] = "fail"

    # Check 3: memory magnitude
    model.train()
    out = model(input_ids, attention_mask, mask_positions)
    with torch.no_grad():
        mem = model.encoder(embed(input_ids), attention_mask,
                            mask_positions=mask_positions)[0]
        rms = mem.float().pow(2).mean(dim=-1).sqrt()
        mean_rms = rms.mean().item()
        max_rms = rms.max().item()
    if 0.1 < mean_rms < 10.0:
        report("memory RMS in [0.1, 10]", "pass",
               f"mean={mean_rms:.3f}, max={max_rms:.3f}")
        results["memory_magnitude"] = "pass"
    else:
        report("memory RMS in [0.1, 10]", "warn",
               f"mean={mean_rms:.3f}, max={max_rms:.3f}")
        results["memory_magnitude"] = "warn"

    # Check 4: gradient flow
    model.zero_grad()
    out["loss"].backward()
    n_zero_grad = 0
    grad_norms = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or p.grad.abs().sum() == 0:
            n_zero_grad += 1
        else:
            grad_norms.append(p.grad.norm().item())
    if n_zero_grad == 0 and grad_norms:
        median_grad = sorted(grad_norms)[len(grad_norms) // 2]
        max_grad = max(grad_norms)
        if max_grad > 1e3:
            report("gradient flow", "warn",
                   f"max norm = {max_grad:.2e} (large)")
            results["gradient_flow"] = "warn"
        else:
            report("gradient flow", "pass",
                   f"{len(grad_norms)} params, median={median_grad:.4f}, max={max_grad:.4f}")
            results["gradient_flow"] = "pass"
    else:
        report("gradient flow", "fail", f"{n_zero_grad} zero/None grad")
        results["gradient_flow"] = "fail"

    # Check 5: memory dispersion
    with torch.no_grad():
        mem = model.encoder(embed(input_ids), attention_mask,
                            mask_positions=mask_positions)[0]
        m0 = F.normalize(mem[0].float(), dim=-1)
        cos = m0 @ m0.T
        M = cos.shape[0]
        off_diag = cos[~torch.eye(M, dtype=torch.bool, device=cos.device)]
        mean_cos = off_diag.mean().item()
    if mean_cos < 0.9:
        report("memory token dispersion", "pass",
               f"mean off-diag cos = {mean_cos:.3f}")
        results["memory_dispersion"] = "pass"
    else:
        report("memory token dispersion", "warn",
               f"mean off-diag cos = {mean_cos:.3f} (collapsed?)")
        results["memory_dispersion"] = "warn"

    # Check 6: mask sensitivity
    perturbed_mask = mask_positions.clone()
    perturbed_mask[0] = ~perturbed_mask[0]
    if perturbed_mask[0, 1:].any():
        model.train(False)
        with torch.no_grad():
            out_pm = model(input_ids, attention_mask, perturbed_mask)
        loss_orig = out["loss_recon"].item()
        loss_pm = out_pm["loss_recon"].item()
        if abs(loss_pm - loss_orig) > 1e-3:
            report("mask sensitivity", "pass",
                   f"dloss = {abs(loss_pm - loss_orig):.3f}")
            results["mask_sensitivity"] = "pass"
        else:
            report("mask sensitivity", "fail",
                   f"dloss = {abs(loss_pm - loss_orig):.2e}")
            results["mask_sensitivity"] = "fail"

    return results


def main():
    cfg = ReprConfig(batch_size=2, fixed_window_size=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print(f"Loading Llama once and sharing across variants...")
    llama_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=llama_dtype)

    batch = synthetic_batch(cfg, seed=42)
    variants = ["v21", "flat_baseline", "continuous_baseline",
                "memorizing_baseline"]
    if device == "cuda":
        variants.append("recurrent_baseline")

    all_results = {}
    for variant in variants:
        print(f"\n=== {variant} ===")
        model = ReprLearningModel(cfg, variant=variant,
                                  llama_model=llama).to(device)
        results = run_variant_checks(variant, model, batch, device)
        all_results[variant] = results
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    checks = list(all_results[variants[0]].keys())
    header = f"{'Check':<24}" + "".join(f"{v:<18}" for v in variants)
    print(header)
    print("-" * 100)
    for check in checks:
        row = f"{check:<24}"
        for v in variants:
            row += f"{all_results[v].get(check, '?'):<18}"
        print(row)
    print("=" * 100)

    fails = sum(1 for v in variants for c in checks
                if all_results[v].get(c) == "fail")
    warns = sum(1 for v in variants for c in checks
                if all_results[v].get(c) == "warn")
    print(f"\nTotal: {fails} failures, {warns} warnings across {len(variants)} variants")
    if fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
