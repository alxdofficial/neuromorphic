#!/usr/bin/env python3
"""Production v1f training: JEPA cross-chunk encoding prediction.

For each variant:
 - Online encoder: encodes chunk_1 → memory_1
 - Predictor (small transformer): predicts target memory_2 from memory_1
 - Target encoder: EMA copy of online encoder, encodes chunk_2 (no grad)
 - Loss: MSE(predicted, target) + VicReg variance + covariance regularization
 - No Llama in the training loop (only Llama's embedding layer)

Critical anti-collapse: stop-gradient on target side + EMA target weights +
asymmetric predictor (only online side has it). Cross-chunk task forces
real predictive structure to be encoded; collapse → uniform memory_2_target
makes prediction trivial but VicReg variance term keeps memory_1 spread out.

Per-variant outputs in outputs/repr_learning/v1f_<variant>/:
  jsonl/<variant>.jsonl   — per-step training metrics
  ckpts/<variant>.last.pt — encoder + predictor + target weights
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_real import make_chunkpair_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_TRAIN = REPO / "data/wave1/fineweb_edu.train.parquet"
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)


def ema_tau_at_step(step: int, max_steps: int,
                    tau_start: float = 0.996, tau_end: float = 1.0) -> float:
    """EMA momentum schedule: linearly increase τ toward 1.0 over training.
    Standard JEPA practice (I-JEPA / BYOL use this)."""
    progress = step / max(max_steps, 1)
    return tau_start + progress * (tau_end - tau_start)


@torch.no_grad()
def run_val(model, val_dl, device, n_batches: int = 25) -> dict:
    model.train(False)
    losses_jepa, losses_var, losses_cov = [], [], []
    target_vars, target_covs = [], []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        chunk_1 = batch.chunk_1.to(device, non_blocking=True)
        chunk_2 = batch.chunk_2.to(device, non_blocking=True)
        out = model.compute_jepa_loss(chunk_1, chunk_2)
        losses_jepa.append(float(out["loss_jepa"]))
        losses_var.append(float(out["loss_var"]))
        losses_cov.append(float(out["loss_cov"]))
        target_vars.append(float(out["loss_var_target"]))
        target_covs.append(float(out["loss_cov_target"]))
    model.train(True)
    n = max(len(losses_jepa), 1)
    return {
        "val_loss_jepa": sum(losses_jepa) / n,
        "val_loss_var": sum(losses_var) / n,
        "val_loss_cov": sum(losses_cov) / n,
        "val_loss_var_target": sum(target_vars) / n,
        "val_loss_cov_target": sum(target_covs) / n,
        "val_n_batches": len(losses_jepa),
    }


def save_checkpoint(model, opt, step, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    model.init_jepa(predictor_d_hidden=512, predictor_n_layers=2, predictor_n_heads=8)
    # n_trainable_params counts encoder + predictor + mask_embed (target encoder
    # has requires_grad=False so it's excluded). Confirm by listing.
    n_trainable = model.n_trainable_params()
    n_target = sum(p.numel() for p in model.jepa_target_encoder.parameters())
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable + {n_target:,} EMA target, {n_steps} steps)")
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

        opt.zero_grad()
        out = model.compute_jepa_loss(chunk_1, chunk_2)
        loss = out["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        opt.step()

        # EMA update of target encoder (critical for stability)
        tau = ema_tau_at_step(step, n_steps, tau_start=0.996, tau_end=1.0)
        model.update_jepa_target(tau=tau)

        row = {
            "step": step,
            "variant": variant,
            "loss": float(out["loss"]),
            "loss_jepa": float(out["loss_jepa"]),
            "loss_var": float(out["loss_var"]),
            "loss_cov": float(out["loss_cov"]),
            "loss_var_target": float(out["loss_var_target"]),
            "loss_cov_target": float(out["loss_cov_target"]),
            "grad_norm": float(gn),
            "lr": lr,
            "ema_tau": tau,
            "memory_M": out["memory"].shape[1],
        }
        jsonl_fp.write(json.dumps(row) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            print(f"  step {step:6d}/{n_steps}  jepa={float(out['loss_jepa']):.4f}  "
                  f"var={float(out['loss_var']):.3f}  cov={float(out['loss_cov']):.3f}  "
                  f"gnorm={float(gn):6.2f}  lr={lr:.2e}  τ={tau:.4f}  ({sps:.1f} step/s)",
                  flush=True)

        if step > 0 and step % val_every == 0:
            vm = run_val(model, val_dl, device, val_batches)
            val_row = {"phase": "val", "step": step, "variant": variant, **vm}
            jsonl_fp.write(json.dumps(val_row) + "\n")
            print(f"    [val @ {step}]  jepa={vm['val_loss_jepa']:.4f}  "
                  f"var={vm['val_loss_var']:.3f}  cov={vm['val_loss_cov']:.3f}",
                  flush=True)

        if step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, ckpt_path)

    save_checkpoint(model, opt, step, ckpt_path)
    final_val = run_val(model, val_dl, device, val_batches)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_loss_jepa={final_val['val_loss_jepa']:.4f}", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "ema_target_params": n_target,
        "n_steps": step,
        "elapsed_s": elapsed,
        "final_val_loss_jepa": final_val["val_loss_jepa"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "v21", "flat_baseline", "continuous_baseline",
        "recurrent_baseline",
    ])
    # Excluded from JEPA:
    # - vanilla_llama: NullEncoder produces empty memory; predictor has
    #   nothing to operate on. The "no-memory floor" doesn't apply.
    # - memorizing_baseline (MT): retrieval-based memory selects content
    #   FROM the input chunk itself rather than producing an abstract latent
    #   representation. Predicting chunk_2's retrieved tokens from chunk_1's
    #   retrieved tokens is closer to a generative LM task than a
    #   representation-prediction task. Structural mismatch with JEPA.
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--val-batches", type=int, default=25)
    args = ap.parse_args()

    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=256,
        max_steps=args.steps,
        warmup_steps=500,
        d_node_state=128,
        n_edges=30,
        n_flat_codes=16,
        edge_token_packing="fused",
    )

    print(f"Config: M_baseline={cfg.n_flat_codes}, n_edges={cfg.n_edges}, "
          f"d_node_state={cfg.d_node_state}, edge_packing={cfg.edge_token_packing}")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")

    print("\nLoading Llama (shared across variants, frozen — only embed used)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    summaries = []
    for variant in args.variants:
        out_dir = REPO / f"outputs/repr_learning/v1f_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        s = train_one_variant(
            variant=variant, llama=llama, cfg=cfg,
            n_steps=args.steps, log_every=args.log_every,
            val_every=args.val_every, save_every=args.save_every,
            val_batches=args.val_batches, out_dir=out_dir,
        )
        summaries.append(s)

    print("\n" + "=" * 78)
    print("v1f SUMMARY")
    print("=" * 78)
    print(f"  {'variant':<25}{'params':>12}{'final_val_jepa':>16}{'time(min)':>12}")
    print("  " + "-" * 65)
    for s in summaries:
        print(f"  {s['variant']:<25}{s['trainable_params']:>12,}"
              f"{s['final_val_loss_jepa']:>16.4f}{s['elapsed_s']/60:>12.1f}")

    summary_path = REPO / "outputs/repr_learning/v1f_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
