#!/usr/bin/env python3
"""Smoke test the v1e hidden-state-matching objective on all variants.

Per variant:
 - Train ~500 steps on FineWeb-edu chunk pairs (256 + 256 tokens).
 - chunk_2 has 50% mask applied; same mask for teacher + student.
 - Loss = MSE between final-layer Llama hidden states at chunk_2 positions
   under [chunk_1, chunk_2_masked] (teacher) vs [encoder(chunk_1), chunk_2_masked] (student).
 - Frozen Llama (no LoRA in smoke).

Verify:
 1. Loss DESCENDS over 500 steps (not stuck — task is learnable).
 2. Variants DIFFERENTIATE (different final loss — architecture matters).
 3. Mamba doesn't dominate (cross-chunk avoids the positional-alignment bias).

Writes per-variant JSONL to outputs/repr_learning/smoke_hsm/<variant>.jsonl.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_real import make_chunkpair_dataloader
from src.repr_learning.data import sample_span_mask
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_TRAIN = REPO / "data/wave1/fineweb_edu.train.parquet"


def make_chunk_mask(B: int, T: int, cfg: ReprConfig, seed: int = 0) -> torch.Tensor:
    """Independent span-mask per row, target ratio from cfg.mask_ratio_min."""
    out = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        rng = random.Random(seed + b)
        ratio = rng.uniform(cfg.mask_ratio_min, cfg.mask_ratio_max)
        positions = sample_span_mask(
            T, ratio, (cfg.mask_span_min, cfg.mask_span_max), rng,
        )
        for p in positions:
            out[b, p] = True
    return out


def smoke_one_variant(
    variant: str, llama, cfg: ReprConfig, n_steps: int,
    log_every: int, out_dir: Path, fresh_llama_loader,
) -> dict:
    device = "cuda"
    # If LoRA is on we need a fresh Llama per variant; otherwise share.
    llama_to_use = llama if not cfg.use_llama_lora else fresh_llama_loader()
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_to_use).to(device)
    n_train = model.n_trainable_params()
    print(f"\n=== {variant}  ({n_train:,} trainable) ===")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    dl = make_chunkpair_dataloader(
        cfg, FINEWEB_TRAIN, chunk_size=256, num_workers=0, seed=42,
    )

    jsonl_path = out_dir / f"{variant}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    losses = []
    for step, batch in enumerate(dl):
        if step >= n_steps:
            break
        chunk_1 = batch.chunk_1.to(device, non_blocking=True)
        chunk_2 = batch.chunk_2.to(device, non_blocking=True)
        B, T2 = chunk_2.shape
        mask_pos = make_chunk_mask(B, T2, cfg, seed=42 + step).to(device)

        # LR warmup
        if step < cfg.warmup_steps:
            lr = cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
        else:
            lr = cfg.learning_rate
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        out = model.compute_hsm_loss(chunk_1, chunk_2, mask_pos)
        loss = out["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        opt.step()
        losses.append(float(out["loss_hsm"]))

        row = {
            "step": step,
            "variant": variant,
            "loss": float(out["loss"]),
            "loss_hsm": float(out["loss_hsm"]),
            "loss_aux": float(out["loss_aux"]) if torch.is_tensor(out["loss_aux"]) else float(out["loss_aux"]),
            "loss_orth": float(out["loss_orth"]) if torch.is_tensor(out["loss_orth"]) else float(out["loss_orth"]),
            "loss_z": float(out["loss_z"]) if torch.is_tensor(out["loss_z"]) else float(out["loss_z"]),
            "grad_norm": float(gn),
            "lr": lr,
            "memory_M": out["memory"].shape[1],
        }
        jsonl_fp.write(json.dumps(row) + "\n")

        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step {step:4d}/{n_steps}  hsm={float(out['loss_hsm']):.4f}  "
                  f"loss={float(out['loss']):.4f}  gnorm={float(gn):.2f}  lr={lr:.2e}")

    jsonl_fp.close()

    elapsed = time.time() - t_start
    initial = losses[:20]
    final = losses[-50:]
    summary = {
        "variant": variant,
        "n_steps": len(losses),
        "elapsed_s": elapsed,
        "loss_init_mean": sum(initial) / max(len(initial), 1),
        "loss_final_mean": sum(final) / max(len(final), 1),
    }
    summary["loss_descent"] = summary["loss_init_mean"] - summary["loss_final_mean"]
    print(f"  DONE: {len(losses)} steps in {elapsed/60:.1f} min")
    print(f"  loss init→final: {summary['loss_init_mean']:.4f} → {summary['loss_final_mean']:.4f} "
          f"(Δ {summary['loss_descent']:+.4f})")

    del model, opt
    if cfg.use_llama_lora:
        del llama_to_use
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "v21", "flat_baseline", "continuous_baseline",
        "memorizing_baseline", "recurrent_baseline", "vanilla_llama",
    ])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--mask-ratio-min", type=float, default=0.4)
    ap.add_argument("--mask-ratio-max", type=float, default=0.6)
    ap.add_argument("--out-dir", type=str, default="outputs/repr_learning/smoke_hsm")
    args = ap.parse_args()

    # v1e config: fused V2.1, M=16 baselines, target ~11.6k bottleneck floats
    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=256,
        max_steps=args.steps,
        warmup_steps=50,
        # v1e architectural defaults
        d_node_state=128,
        n_edges=30,
        n_flat_codes=16,
        edge_token_packing="fused",
        # mask range
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
    )

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out: {out_dir}")
    print(f"Config: M_baseline={cfg.n_flat_codes}, n_edges={cfg.n_edges}, "
          f"d_node_state={cfg.d_node_state}, edge_packing={cfg.edge_token_packing}")

    print("\nLoading Llama (shared, frozen)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    summaries = []
    for v in args.variants:
        s = smoke_one_variant(
            variant=v, llama=llama, cfg=cfg,
            n_steps=args.steps, log_every=args.log_every,
            out_dir=out_dir,
            fresh_llama_loader=lambda: load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)[0],
        )
        summaries.append(s)

    print("\n" + "=" * 78)
    print("SMOKE SUMMARY")
    print("=" * 78)
    print(f"  {'variant':<25}{'init':>10}{'final':>10}{'Δ':>10}")
    print("  " + "-" * 55)
    for s in summaries:
        print(f"  {s['variant']:<25}{s['loss_init_mean']:>10.4f}"
              f"{s['loss_final_mean']:>10.4f}{s['loss_descent']:>+10.4f}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
