#!/usr/bin/env python3
"""Audit script — verify bottleneck width parity across V2.1 + 4 baselines.

Prints for each variant:
  - Pre-projection floats per chunk (the matched bottleneck quantity)
  - Per-chunk independent info (bits + floats that vary per input)
  - Memory token shape (post-projection, what Llama sees)
  - Trainable param count

Does NOT load Llama — uses the config directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig


def fmt(n: int) -> str:
    return f"{n:,}"


def main():
    cfg = ReprConfig()

    print(f"Config: window={cfg.fixed_window_size}, n_memory_tokens={cfg.n_memory_tokens}")
    print(f"        d_llama={cfg.d_llama}, d_concept={cfg.d_concept}")
    print()

    raw_input_floats = cfg.fixed_window_size * cfg.d_llama
    post_proj_floats = cfg.n_memory_tokens * cfg.d_llama

    print("=" * 92)
    print(f"{'Variant':<28} {'Pre-proj floats':>16} {'Independent info':>30} {'Memory shape':>15}")
    print("=" * 92)

    # V2.1: 32 × (1024 + 128 + 1024) = 69,632
    v21_pre = cfg.n_edges * (cfg.d_concept + cfg.d_edge + cfg.d_concept)
    v21_indep = f"{cfg.n_edges * 24}b + {cfg.n_edges * cfg.d_edge}f"
    print(
        f"{'V2.1 (32 edges × triples)':<28} {fmt(v21_pre):>16} {v21_indep:>30} "
        f"({cfg.n_memory_tokens}, {cfg.d_llama})"
    )

    # A: 96 × 725
    a_pre = cfg.n_flat_codes * cfg.d_concept_baseline
    a_indep = f"{cfg.n_flat_codes * 12}b"
    print(
        f"{'A (flat codebook)':<28} {fmt(a_pre):>16} {a_indep:>30} "
        f"({cfg.n_memory_tokens}, {cfg.d_llama})"
    )

    # B: 96 × 725
    b_pre = cfg.n_flat_codes * cfg.d_continuous
    b_indep = f"{cfg.n_flat_codes * cfg.d_continuous}f"
    print(
        f"{'B (continuous)':<28} {fmt(b_pre):>16} {b_indep:>30} "
        f"({cfg.n_memory_tokens}, {cfg.d_llama})"
    )

    # MT: 96 retrieved × 725 (bank itself is 256 × 725 but only 96 reach Llama)
    mt_pre = cfg.n_flat_codes * cfg.d_mt_value
    mt_bits_per_pick = (cfg.fixed_window_size.bit_length() - 1) + 1  # log2(256) ≈ 8
    mt_indep = f"{cfg.n_flat_codes * mt_bits_per_pick}b + {cfg.n_flat_codes * cfg.d_mt_value}f"
    print(
        f"{'MT (retrieval, 256→96)':<28} {fmt(mt_pre):>16} {mt_indep:>30} "
        f"({cfg.n_memory_tokens}, {cfg.d_llama})"
    )

    # Mamba: 96 × 725 after adaptive pool
    mamba_pre = cfg.n_flat_codes * cfg.d_recurrent
    mamba_indep = f"{cfg.n_flat_codes * cfg.d_recurrent}f"
    print(
        f"{'Mamba (256→96 pool)':<28} {fmt(mamba_pre):>16} {mamba_indep:>30} "
        f"({cfg.n_memory_tokens}, {cfg.d_llama})"
    )

    print("=" * 92)
    print()

    target = 69_600
    deviations = {
        "V2.1": v21_pre - target,
        "A": a_pre - target,
        "B": b_pre - target,
        "MT": mt_pre - target,
        "Mamba": mamba_pre - target,
    }
    print(f"Target pre-projection budget: {fmt(target)} floats/chunk")
    print("Deviation from target:")
    for name, dev in deviations.items():
        pct = 100.0 * dev / target
        print(f"  {name:<8} {dev:+5d}  ({pct:+.3f}%)")
    print()

    if max(abs(d) for d in deviations.values()) <= 64:
        print("✓ All five variants match pre-projection budget within ±64 floats (±0.1%)")
    else:
        print("✗ One or more variants deviate >64 floats from target")
        sys.exit(1)

    print()
    print(f"For reference:")
    print(f"  Raw embedded input per chunk: {fmt(raw_input_floats)} floats "
          f"(256 tokens × {cfg.d_llama})")
    print(f"  Post-projection (Llama input):  {fmt(post_proj_floats)} floats "
          f"({cfg.n_memory_tokens} tokens × {cfg.d_llama})")
    print(f"  Compression ratio (raw → pre-proj): {raw_input_floats / target:.2f}×")


if __name__ == "__main__":
    main()
