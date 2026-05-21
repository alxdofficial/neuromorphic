#!/usr/bin/env python3
"""Production v1e training: hidden-state-matching on cross-chunk FineWeb pairs.

Per variant:
 - Train N steps (default 30K) on FineWeb-edu chunk pairs (256 + 256 tokens).
 - chunk_2 has span-mask applied with ratio in [mask_ratio_min, mask_ratio_max].
 - Loss = MSE between final-layer Llama hidden states at chunk_2 positions
   under [chunk_1, chunk_2_masked] (teacher, no grad) vs
   [encoder(chunk_1), chunk_2_masked] (student, encoder + mask_embed trainable).
 - Frozen Llama (no LoRA in v1e baseline).

Per-variant outputs in outputs/repr_learning/v1e_<variant>/:
  jsonl/<variant>.jsonl     — per-step training metrics
  ckpts/<variant>.last.pt   — encoder weights checkpoint (Llama base excluded)
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_real import make_chunkpair_dataloader
from src.repr_learning.data import sample_span_mask
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_TRAIN = REPO / "data/wave1/fineweb_edu.train.parquet"
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"


def make_chunk_mask(B: int, T: int, cfg: ReprConfig, step: int = 0, base_seed: int = 42) -> torch.Tensor:
    """Per-row independent span-mask. Per-row seeding is `base_seed + step*B + b`
    so consecutive steps don't alias (previous bug: seed=42+step + row b meant
    step s row 1 == step s+1 row 0)."""
    out = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        rng = random.Random(base_seed + step * B + b)
        ratio = rng.uniform(cfg.mask_ratio_min, cfg.mask_ratio_max)
        positions = sample_span_mask(
            T, ratio, (cfg.mask_span_min, cfg.mask_span_max), rng,
        )
        for p in positions:
            out[b, p] = True
    return out


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup → cosine decay to 10% of base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)


@torch.no_grad()
def run_val(model, val_dl, cfg, device, n_batches: int = 25, seed: int = 7) -> dict:
    model.train(False)
    losses_hsm = []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        chunk_1 = batch.chunk_1.to(device, non_blocking=True)
        chunk_2 = batch.chunk_2.to(device, non_blocking=True)
        # Use distinct step counter for val masks (offset by huge number to
        # avoid colliding with training step seeds).
        mask_pos = make_chunk_mask(chunk_2.shape[0], chunk_2.shape[1], cfg,
                                    step=10_000_000 + seed + i).to(device)
        out = model.compute_hsm_loss(chunk_1, chunk_2, mask_pos)
        losses_hsm.append(float(out["loss_hsm"]))
    model.train(True)
    return {
        "val_loss_hsm": sum(losses_hsm) / max(len(losses_hsm), 1),
        "val_n_batches": len(losses_hsm),
    }


def save_checkpoint(model, opt, step, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop frozen Llama base weights (keep LoRA params if any — "lora_" in name)
    def keep(k: str) -> bool:
        if not k.startswith("decoder.llama."):
            return True
        return "lora_" in k
    torch.save({
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
    }, path)


def train_one_variant(
    variant: str, llama, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path,
) -> dict:
    device = "cuda"
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    n_trainable = model.n_trainable_params()
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps)")
    print(f"{'='*78}")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    train_dl = make_chunkpair_dataloader(
        cfg, FINEWEB_TRAIN, chunk_size=256, num_workers=0, seed=42,
    )
    # Build val dataloader ONCE (the prior version reloaded the parquet at
    # every val pass — 30 val passes × 720-doc parquet reread = wasted I/O).
    val_dl = make_chunkpair_dataloader(
        cfg, FINEWEB_VAL, chunk_size=256, num_workers=0, seed=7,
    )

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    last_print_step, last_print_time = 0, t_start

    for step, batch in enumerate(train_dl):
        if step >= n_steps:
            break

        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        chunk_1 = batch.chunk_1.to(device, non_blocking=True)
        chunk_2 = batch.chunk_2.to(device, non_blocking=True)
        mask_pos = make_chunk_mask(
            chunk_2.shape[0], chunk_2.shape[1], cfg, step=step,
        ).to(device)

        opt.zero_grad()
        out = model.compute_hsm_loss(chunk_1, chunk_2, mask_pos)
        loss = out["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        opt.step()

        # JSONL row
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

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            print(f"  step {step:6d}/{n_steps}  hsm={float(out['loss_hsm']):.4f}  "
                  f"loss={float(out['loss']):.4f}  gnorm={float(gn):6.2f}  "
                  f"lr={lr:.2e}  ({sps:.1f} step/s)", flush=True)

        if step > 0 and step % val_every == 0:
            vm = run_val(model, val_dl, cfg, device, val_batches)
            val_row = {"phase": "val", "step": step, "variant": variant, **vm}
            jsonl_fp.write(json.dumps(val_row) + "\n")
            print(f"    [val @ {step}]  hsm={vm['val_loss_hsm']:.4f}  "
                  f"({vm['val_n_batches']} batches)", flush=True)

        if step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, ckpt_path)

    # Final ckpt + val
    save_checkpoint(model, opt, step, ckpt_path)
    final_val = run_val(model, val_dl, cfg, device, val_batches)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_loss_hsm={final_val['val_loss_hsm']:.4f}", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "n_steps": step,
        "elapsed_s": elapsed,
        "final_val_loss_hsm": final_val["val_loss_hsm"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "v21", "flat_baseline", "continuous_baseline",
        "memorizing_baseline", "recurrent_baseline", "vanilla_llama",
    ])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--val-batches", type=int, default=25)
    ap.add_argument("--mask-ratio-min", type=float, default=0.4)
    ap.add_argument("--mask-ratio-max", type=float, default=0.6)
    args = ap.parse_args()

    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=256,
        max_steps=args.steps,
        warmup_steps=500,
        # v1e architectural defaults
        d_node_state=128,
        n_edges=30,
        n_flat_codes=16,
        edge_token_packing="fused",
        # mask range
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
    )

    print(f"Config: M_baseline={cfg.n_flat_codes}, n_edges={cfg.n_edges}, "
          f"d_node_state={cfg.d_node_state}, edge_packing={cfg.edge_token_packing}")
    print(f"Mask ratio: [{cfg.mask_ratio_min:.2f}, {cfg.mask_ratio_max:.2f}]")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")

    print("\nLoading Llama (shared across variants, frozen)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    summaries = []
    for variant in args.variants:
        out_dir = REPO / f"outputs/repr_learning/v1e_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        s = train_one_variant(
            variant=variant, llama=llama, cfg=cfg,
            n_steps=args.steps, log_every=args.log_every,
            val_every=args.val_every, save_every=args.save_every,
            val_batches=args.val_batches, out_dir=out_dir,
        )
        summaries.append(s)

    print("\n" + "=" * 78)
    print("v1e SUMMARY")
    print("=" * 78)
    print(f"  {'variant':<25}{'params':>12}{'final_val_hsm':>16}{'time(min)':>12}")
    print("  " + "-" * 65)
    for s in summaries:
        print(f"  {s['variant']:<25}{s['trainable_params']:>12,}"
              f"{s['final_val_loss_hsm']:>16.4f}{s['elapsed_s']/60:>12.1f}")

    summary_path = REPO / "outputs/repr_learning/v1e_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
