#!/usr/bin/env python3
"""Production training script for repr_learning: 30K-step v0 run.

Trains all 5 variants sequentially. Per-step JSONL metrics, periodic
val pass, checkpointing, LR warmup + cosine decay.

Usage:
    # Full v0 run (30K steps per variant, ~3 hours on RTX 4090):
    python scripts/repr_learning/train_repr.py

    # Single variant:
    python scripts/repr_learning/train_repr.py --variants v21

    # Shorter for testing:
    python scripts/repr_learning/train_repr.py --steps 1000 --val-every 200

    # Resume one variant:
    python scripts/repr_learning/train_repr.py --variants v21 --resume

Monitor live (separate terminal):
    python scripts/diagnostics/plot_repr.py \\
        --jsonl-dir outputs/repr_learning/v0/jsonl \\
        --out outputs/repr_learning/v0/plot.png \\
        --watch 60
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_real import (
    MixedReprDataset,
    collate,
    make_dataloader,
)
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.metrics import compute_metrics
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_TRAIN = REPO / "data/wave1/fineweb_edu.train.parquet"
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"
COMPOSITE_TRAIN = REPO / "data/wave1/composite_v1/train/passages.jsonl"
COMPOSITE_VAL = REPO / "data/wave1/composite_v1/val/passages.jsonl"


def make_val_dataloader(cfg: ReprConfig, seed: int = 7) -> DataLoader:
    """Held-out val loader using *val* parquet/jsonl and a different seed.
    Same masking distribution as train (we want val recon CE comparable).
    """
    return make_dataloader(
        cfg, FINEWEB_VAL, COMPOSITE_VAL,
        fineweb_ratio=0.5, num_workers=0, seed=seed,
    )


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup → cosine decay to 10% of base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)  # decay from 1.0 → 0.1


def set_lr(opt: torch.optim.Optimizer, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr


@torch.no_grad()
def run_val(model: ReprLearningModel, val_dl: DataLoader, device: str,
            n_batches: int = 50) -> dict:
    """Compute mean recon CE on `n_batches` val batches."""
    model.train(False)
    losses = []
    disps = []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        mask_positions = batch.mask_positions.to(device, non_blocking=True)
        out = model(input_ids, attention_mask, mask_positions)
        losses.append(out["loss_recon"].item() if isinstance(out["loss_recon"], torch.Tensor)
                      else float(out["loss_recon"]))
        # Memory dispersion on val (sample 0)
        m = F.normalize(out["memory"][0].float(), dim=-1)
        cos = m @ m.T
        M = cos.shape[0]
        mask = ~torch.eye(M, dtype=torch.bool, device=cos.device)
        disps.append(cos[mask].mean().item())
    model.train(True)
    return {
        "val_loss_recon": sum(losses) / max(len(losses), 1),
        "val_mem_dispersion": sum(disps) / max(len(disps), 1),
        "val_n_batches": len(losses),
    }


class DeadCodeReviver:
    """Periodically re-seed never-picked codebook entries from heavily-used ones.

    Tracks per-code pick counts over a rolling window. Every `interval`
    steps (after `warmup`), revives codes with zero picks by copying from
    the top-K most-picked codes + small noise, then resets the window.
    Logs # revived per pass.

    Only runs if `model.encoder` has a `concept_id` parameter (V2.1, A).
    """
    def __init__(self, model, cfg: ReprConfig):
        self.model = model
        self.cfg = cfg
        self.enabled = (
            hasattr(model.encoder, "concept_id")
            and cfg.dead_code_revival_interval > 0
        )
        if not self.enabled:
            self.recent_picks = None
            return
        n_nodes = model.encoder.concept_id.shape[0]
        device = model.encoder.concept_id.device
        self.recent_picks = torch.zeros(n_nodes, dtype=torch.long, device=device)
        self.steps_since_reset = 0
        self.last_revived_count = 0

    @torch.no_grad()
    def observe(self, aux: dict):
        if not self.enabled:
            return
        picks = aux.get("picked_ids")
        if picks is None:
            return
        flat = picks.flatten().to(self.recent_picks.device)
        self.recent_picks.scatter_add_(
            0, flat, torch.ones_like(flat, dtype=self.recent_picks.dtype),
        )
        self.steps_since_reset += 1

    @torch.no_grad()
    def maybe_revive(self, step: int) -> int:
        """Return number of codes revived this pass (0 if not the moment)."""
        if not self.enabled:
            return 0
        if step < self.cfg.dead_code_revival_warmup:
            return 0
        if step % self.cfg.dead_code_revival_interval != 0:
            return 0
        if self.steps_since_reset < self.cfg.dead_code_revival_window // 2:
            # Not enough observation yet — skip this pass.
            return 0

        codebook = self.model.encoder.concept_id  # nn.Parameter [N, D]
        dead = (self.recent_picks == 0).nonzero(as_tuple=False).flatten()
        if len(dead) == 0:
            self.recent_picks.zero_()
            self.steps_since_reset = 0
            return 0

        # Sample heavy users to copy from
        topk = max(len(dead), 32)
        sorted_idx = self.recent_picks.argsort(descending=True)
        heavy = sorted_idx[:topk]
        src = heavy[torch.randint(0, len(heavy), (len(dead),), device=codebook.device)]
        noise = torch.randn_like(codebook.data[dead]) * self.cfg.dead_code_revival_noise_std
        codebook.data[dead] = codebook.data[src] + noise

        n_revived = len(dead)
        self.last_revived_count = n_revived
        self.recent_picks.zero_()
        self.steps_since_reset = 0
        return n_revived


def save_checkpoint(model, opt, step, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Skip the frozen Llama base weights (1.2 GB) but KEEP LoRA params
    # which live at e.g. `decoder.llama.model.layers.<i>.self_attn.q_proj.lora_A`.
    # Also keep encoder params whose names contain "llama" (Mamba's
    # proj_to_llama). The conservative filter: drop only base Llama
    # weights — anything with `lora_` in the key survives.
    def keep(k: str) -> bool:
        if not k.startswith("decoder.llama."):
            return True
        # Inside decoder.llama, keep only the LoRA adapter parameters.
        return "lora_" in k

    torch.save({
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
    }, path)


def load_checkpoint(model, opt, path: Path) -> int:
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]


def train_variant(
    variant: str,
    llama,
    cfg: ReprConfig,
    n_steps: int,
    log_every: int,
    val_every: int,
    save_every: int,
    val_batches: int,
    out_dir: Path,
    resume: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 78}\nVariant: {variant}  ({n_steps} steps)\n{'=' * 78}")

    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    n_trainable = model.n_trainable_params()
    print(f"  trainable params: {n_trainable:,}")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    start_step = 0
    if resume:
        start_step = load_checkpoint(model, opt, ckpt_path)
        if start_step > 0:
            print(f"  resumed from step {start_step}")

    reviver = DeadCodeReviver(model, cfg)
    if reviver.enabled:
        print(f"  dead-code revival: interval={cfg.dead_code_revival_interval} "
              f"warmup={cfg.dead_code_revival_warmup}")

    train_dl = make_dataloader(
        cfg, FINEWEB_TRAIN, COMPOSITE_TRAIN,
        fineweb_ratio=0.5, num_workers=0, seed=42,
    )
    val_dl_factory = lambda: make_val_dataloader(cfg, seed=7)

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)  # line-buffered

    t_start = time.time()
    step = start_step
    last_print_step = step
    last_print_time = t_start

    for batch in train_dl:
        if step >= n_steps:
            break

        # LR schedule
        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        set_lr(opt, lr)

        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        mask_positions = batch.mask_positions.to(device, non_blocking=True)

        t_step = time.time()
        opt.zero_grad()
        out = model(input_ids, attention_mask, mask_positions)
        loss = out["loss"]

        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {loss.item()}",
                  flush=True)
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(), cfg.grad_clip,
        )
        step_ms = (time.time() - t_step) * 1000.0
        text_tok = input_ids.shape[0] * input_ids.shape[1]
        text_tok_per_sec = text_tok / max(step_ms / 1000.0, 1e-9)
        metrics = compute_metrics(
            model=model,
            out=out,
            memory=out["memory"],
            aux=out["aux"],
            grad_norm=grad_norm.item(),
            step=step,
            variant=variant,
            step_ms=step_ms,
            text_tok_per_sec=text_tok_per_sec,
        )

        opt.step()

        # Dead-code revival (V2.1 + A): track picks, periodically re-seed.
        reviver.observe(out["aux"])
        n_revived = reviver.maybe_revive(step)
        if n_revived > 0:
            print(f"  [step {step}] revived {n_revived} dead codes", flush=True)

        # Per-step JSONL row
        row = asdict(metrics)
        row["lr"] = lr
        if n_revived > 0:
            row["dead_codes_revived"] = n_revived
        jsonl_fp.write(json.dumps(row) + "\n")

        # Console log
        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            print(
                f"  step {step:6d}/{n_steps}  "
                f"recon={metrics.loss_recon:6.3f}  "
                f"aux={metrics.loss_aux:6.2f}  "
                f"gnorm={metrics.grad_norm:6.2f}  "
                f"disp={metrics.mem_dispersion:.3f}  "
                f"H={metrics.routing_entropy:5.2f}  "
                f"uniq={metrics.unique_codes_per_batch:4d}  "
                f"lr={lr:.2e}  ({sps:.1f} step/s)",
                flush=True,
            )

        # Validation
        if step > 0 and step % val_every == 0:
            val_dl = val_dl_factory()
            val_metrics = run_val(model, val_dl, device, n_batches=val_batches)
            val_row = {
                "phase": "val",
                "step": step,
                "variant": variant,
                **val_metrics,
            }
            jsonl_fp.write(json.dumps(val_row) + "\n")
            print(
                f"    [val @ {step}]  recon={val_metrics['val_loss_recon']:.3f}  "
                f"disp={val_metrics['val_mem_dispersion']:.3f}  "
                f"({val_metrics['val_n_batches']} batches)",
                flush=True,
            )

        # Checkpoint
        if step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, ckpt_path)
            print(f"    [ckpt @ {step}]  wrote {ckpt_path}", flush=True)

        step += 1

    # Final checkpoint + val
    save_checkpoint(model, opt, step, ckpt_path)
    val_dl = val_dl_factory()
    final_val = run_val(model, val_dl, device, n_batches=val_batches)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, **final_val,
    }) + "\n")
    jsonl_fp.close()

    total_t = time.time() - t_start
    print(
        f"  DONE: {step} steps in {total_t/60:.1f} min "
        f"({step/total_t:.1f} step/s).  "
        f"final val recon={final_val['val_loss_recon']:.3f}",
        flush=True,
    )

    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "v21", "flat_baseline", "continuous_baseline",
        "memorizing_baseline", "recurrent_baseline", "vanilla_llama",
    ])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2_500)
    ap.add_argument("--save-every", type=int, default=5_000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="outputs/repr_learning/v0")
    ap.add_argument("--resume", action="store_true",
                    help="Load last checkpoint per variant before training")
    # v1b: variable-length sentence-aware packing
    ap.add_argument("--variable-length", action="store_true",
                    help="Use SentencePackedDataset (v1b): variable-length, "
                         "document-aware FineWeb windows + passage-packed composite")
    ap.add_argument("--min-window", type=int, default=128)
    ap.add_argument("--max-window", type=int, default=1024)
    ap.add_argument("--mask-ratio-min", type=float, default=None)
    ap.add_argument("--mask-ratio-max", type=float, default=None)
    # v1c: V2.1 architectural improvements — roles/z-loss/clip are ON by
    # default; Q-Former is opt-in.
    ap.add_argument("--qformer", action="store_true",
                    help="Enable V2.1 Q-Former adapter (BLIP-2 style)")
    ap.add_argument("--no-role-embeddings", action="store_true",
                    help="Disable V2.1 role embeddings (restore v0/v1b behavior)")
    ap.add_argument("--llama-lora", action="store_true",
                    help="Apply LoRA to Llama q_proj+v_proj (unfreezes via low-rank)")
    ap.add_argument("--llama-lora-rank", type=int, default=16)
    args = ap.parse_args()

    cfg_kwargs = dict(
        batch_size=args.batch_size,
        fixed_window_size=args.max_window if args.variable_length else 256,
        max_steps=args.steps,
        use_variable_length=args.variable_length,
        min_window_size=args.min_window,
        max_window_size=args.max_window,
        use_qformer_adapter=args.qformer,
        use_role_embeddings=not args.no_role_embeddings,
        use_llama_lora=args.llama_lora,
        llama_lora_rank=args.llama_lora_rank,
    )
    if args.mask_ratio_min is not None:
        cfg_kwargs["mask_ratio_min"] = args.mask_ratio_min
    if args.mask_ratio_max is not None:
        cfg_kwargs["mask_ratio_max"] = args.mask_ratio_max
    cfg = ReprConfig(**cfg_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.use_variable_length:
        print(f"Device: {device}, batch={cfg.batch_size}, "
              f"window=variable[{cfg.min_window_size},{cfg.max_window_size}], "
              f"mask=[{cfg.mask_ratio_min:.2f},{cfg.mask_ratio_max:.2f}], "
              f"steps={args.steps}")
    else:
        print(f"Device: {device}, batch={cfg.batch_size}, "
              f"window={cfg.fixed_window_size}, "
              f"mask=[{cfg.mask_ratio_min:.2f},{cfg.mask_ratio_max:.2f}], "
              f"steps={args.steps}")

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    if device == "cpu" and "recurrent_baseline" in args.variants:
        print("[warn] skipping recurrent_baseline on CPU (Mamba needs CUDA)")
        args.variants = [v for v in args.variants if v != "recurrent_baseline"]

    llama_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # When LoRA is on, each variant needs its own fresh Llama (since LoRA
    # wraps the modules in-place). Otherwise share one Llama across variants.
    shared_llama = None
    if not cfg.use_llama_lora:
        print(f"\nLoading Llama (shared across variants)...")
        shared_llama, _ = load_frozen_llama(cfg.llama_model, dtype=llama_dtype)
        print(f"  loaded ({llama_dtype})")
    else:
        print(f"\nLoRA enabled — Llama will be loaded fresh per variant.")

    for variant in args.variants:
        if shared_llama is None:
            llama_for_variant, _ = load_frozen_llama(cfg.llama_model, dtype=llama_dtype)
        else:
            llama_for_variant = shared_llama
        train_variant(
            variant=variant,
            llama=llama_for_variant,
            cfg=cfg,
            n_steps=args.steps,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
            val_batches=args.val_batches,
            out_dir=out_dir,
            resume=args.resume,
        )
        if shared_llama is None:
            del llama_for_variant
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("All variants done.")
    print("=" * 78)
    print(f"  Plot:  python scripts/diagnostics/plot_repr.py \\")
    print(f"           --jsonl-dir {out_dir}/jsonl \\")
    print(f"           --out {out_dir}/plot.png")


if __name__ == "__main__":
    main()
