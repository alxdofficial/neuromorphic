#!/usr/bin/env python3
"""Training-step throughput bench for V2.1 + 4 baselines + vanilla Llama.

Each path benched at its own max-fitting batch size (no shared-BS constraint).
Reports tokens/sec at training step granularity (forward + backward + opt step).

"Vanilla Llama" baseline: frozen Llama on the same span-masked reconstruction
task, but with NO encoder and NO memory tokens. Only mask_embed is trainable.
Backward still flows through frozen Llama to reach mask_embed. This is the
"no-overhead" reference — anything faster is suspect, anything much slower
is overhead we should be aware of.

Usage:
    python scripts/repr_learning/bench_throughput.py
    python scripts/repr_learning/bench_throughput.py --variants v21
    python scripts/repr_learning/bench_throughput.py --warmup 5 --iters 30
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.decoder import FrozenLlamaDecoder, load_frozen_llama
from src.repr_learning.model import ReprLearningModel

REPO = Path(__file__).resolve().parents[2]


class VanillaLlamaBaseline(nn.Module):
    """Frozen Llama + trainable mask_embed only. No encoder, no memory tokens.

    Forward signature matches ReprLearningModel for use with the same bench loop.
    """

    def __init__(self, cfg: ReprConfig, llama_model=None):
        super().__init__()
        self.cfg = cfg
        self.decoder = FrozenLlamaDecoder(cfg, llama_model=llama_model)

    def forward(self, input_ids, attention_mask, mask_positions):
        # No memory tokens — pass an empty memory of shape [B, 0, d_llama].
        B = input_ids.shape[0]
        device = input_ids.device
        empty_mem = torch.zeros(
            B, 0, self.cfg.d_llama,
            device=device,
            dtype=self.decoder.mask_embed.dtype,
        )
        _, loss = self.decoder(input_ids, mask_positions, empty_mem)
        return {"loss": loss, "memory_shape": (B, 0, self.cfg.d_llama)}

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p


def make_synthetic_batch(cfg: ReprConfig, batch_size: int, device: str):
    """Random batch for bench (don't need real data — only timing matters)."""
    T = cfg.fixed_window_size
    input_ids = torch.randint(
        256, cfg.llama_vocab_size, (batch_size, T),
        device=device, dtype=torch.long,
    )
    attention_mask = torch.ones(batch_size, T, device=device, dtype=torch.bool)
    mask_positions = torch.zeros(batch_size, T, device=device, dtype=torch.bool)
    # ~70% mask ratio with simple stride pattern (real masking is similar)
    mask_positions[:, ::3] = True  # every 3rd position masked
    mask_positions[:, 1::3] = True
    return input_ids, attention_mask, mask_positions


def bench_at_bs(
    variant: str,
    cfg: ReprConfig,
    llama,
    batch_size: int,
    warmup: int,
    iters: int,
    device: str,
) -> dict:
    """Build model at given BS, run warmup + timed iters. Returns timing dict."""
    if variant == "vanilla_llama":
        model = VanillaLlamaBaseline(cfg, llama_model=llama).to(device)
    else:
        model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
    )

    input_ids, attention_mask, mask_positions = make_synthetic_batch(
        cfg, batch_size, device,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        out = model(input_ids, attention_mask, mask_positions)
        out["loss"].backward()
        opt.step()
    torch.cuda.synchronize()

    # Timed
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        out = model(input_ids, attention_mask, mask_positions)
        out["loss"].backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    del model, opt, input_ids, attention_mask, mask_positions
    gc.collect()
    torch.cuda.empty_cache()

    text_tok_per_step = batch_size * cfg.fixed_window_size
    if variant == "vanilla_llama":
        seq_len = cfg.fixed_window_size
    else:
        seq_len = cfg.n_memory_tokens + cfg.fixed_window_size
    effective_tok_per_step = batch_size * seq_len

    return {
        "variant": variant,
        "batch_size": batch_size,
        "step_ms": dt * 1000,
        "text_tok_per_sec": text_tok_per_step / dt,
        "effective_tok_per_sec": effective_tok_per_step / dt,
        "seq_len_seen_by_llama": seq_len,
        "peak_gb": peak_gb,
    }


def find_max_bs(
    variant: str, cfg: ReprConfig, llama, candidates: list[int],
    warmup: int, iters: int, device: str,
) -> dict:
    """Try candidates in decreasing order; first one that fits is max BS."""
    best = None
    for bs in candidates:
        print(f"  trying BS={bs}...", flush=True)
        try:
            r = bench_at_bs(
                variant, cfg, llama, bs, warmup, iters, device,
            )
            print(
                f"    OK: {r['step_ms']:.1f}ms/step, "
                f"{r['text_tok_per_sec']:.0f} text-tok/s, "
                f"peak {r['peak_gb']:.2f}GB",
                flush=True,
            )
            best = r
            break
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "vanilla_llama",
        "v21",
        "flat_baseline",
        "continuous_baseline",
        "memorizing_baseline",
        "recurrent_baseline",
    ])
    ap.add_argument("--bs-candidates", type=int, nargs="+",
                    default=[64, 32, 16, 8, 4, 2, 1])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="outputs/repr_learning/bench_throughput.json")
    args = ap.parse_args()

    cfg = ReprConfig(fixed_window_size=256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA required for bench.", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {device}")
    print(f"Window: {cfg.fixed_window_size} text tokens")
    print(f"Memory tokens (variants): {cfg.n_memory_tokens}")
    print(f"BS candidates (high to low): {args.bs_candidates}")
    print(f"Warmup: {args.warmup}, timed iters: {args.iters}\n")

    print("Loading Llama once (bf16) and sharing across variants...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    results = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        r = find_max_bs(
            variant, cfg, llama,
            args.bs_candidates, args.warmup, args.iters, device,
        )
        if r is not None:
            results.append(r)
        else:
            print(f"  no BS fit even at BS={args.bs_candidates[-1]}")

    print("\n" + "=" * 92)
    print(
        f"{'variant':<22}{'BS':>4}{'step ms':>10}"
        f"{'text tok/s':>14}{'eff tok/s':>14}{'seq len':>10}{'peak GB':>10}"
    )
    print("-" * 92)
    if not results:
        print("No successful benches.")
        sys.exit(1)

    vanilla_baseline = next(
        (r for r in results if r["variant"] == "vanilla_llama"), None,
    )

    for r in results:
        print(
            f"{r['variant']:<22}{r['batch_size']:>4}{r['step_ms']:>10.1f}"
            f"{r['text_tok_per_sec']:>14,.0f}{r['effective_tok_per_sec']:>14,.0f}"
            f"{r['seq_len_seen_by_llama']:>10}{r['peak_gb']:>10.2f}"
        )
    print("=" * 92)

    if vanilla_baseline:
        print("\nSlowdown vs vanilla Llama (text-tok/s ratio):")
        for r in results:
            if r["variant"] == "vanilla_llama":
                continue
            ratio = vanilla_baseline["text_tok_per_sec"] / r["text_tok_per_sec"]
            print(f"  {r['variant']:<22} {ratio:.2f}× slower "
                  f"({r['text_tok_per_sec']:.0f} vs {vanilla_baseline['text_tok_per_sec']:.0f} tok/s)")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
