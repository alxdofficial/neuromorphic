"""Wave 2 entry point — long-chat TF NTP (plan §4.5).

Reads TurnPair parquet from preprocess_chat.py, length-buckets, runs
Phase1Trainer.step_wave2.

Usage:
    python scripts/train_wave2.py \\
        --data-paths \\
            data/wave2/ultrachat.parquet \\
            data/wave2/wildchat_long.parquet \\
        --batch-size 2 --num-steps 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.metrics import (
    grad_norms_by_component, surprise_stats, vram_stats,
)
from src.trajectory_memory.training.plotting import (
    save_training_plots, dump_history_json,
)
from src.trajectory_memory.training import (
    Phase1Trainer,
    WarmupCosineScheduler,
    build_optimizer,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from src.trajectory_memory.training.loaders import TurnPairDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--lr-memory", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="checkpoint to resume from OR warm-start from. "
                         "When loading a Wave 1 checkpoint to start Wave 2, "
                         "always pass --warm-start.")
    ap.add_argument("--warm-start", action="store_true",
                    help="Load only model weights from --checkpoint-in (no "
                         "optimizer/scheduler/step). Use this when starting "
                         "Wave 2 from a Wave 1 checkpoint.")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-data-paths", nargs="+", type=Path, default=None,
                    help="held-out TurnPair val parquets. If set, eval at each save.")
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="Disable torch.compile (default ON). ~28% speedup at "
                         "low BS, ~2 min cold-start.")
    ap.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false",
                    help="Disable sliding KV cache (default ON). ~1.79× speedup.")
    ap.add_argument("--prior-loss-weight", type=float, default=0.1,
                    help="Weight on NTP CE for prior tokens (B12 fix; default 0.1). "
                         "Without this (=0), prior memory writes get no gradient "
                         "signal because per-chunk backward + detach cuts grad "
                         "before response loss arrives. Set 0 for legacy §4.5 "
                         "behavior; 0.1 matches §4.8 surprise table intent.")
    ap.set_defaults(compile=True, use_kv_cache=True)
    ap.add_argument("--plot-path", type=Path, default=None,
                    help="If set, save a multi-panel diagnostic plot here every "
                         "--plot-every-seconds. PNG; overwritten in place.")
    ap.add_argument("--plot-every-seconds", type=float, default=180.0)
    args = ap.parse_args()

    # Allow TF32 for fp32 matmul (memory params, bridge, lm_head).
    torch.set_float32_matmul_precision("high")

    cfg = getattr(TrajMemConfig, args.config_tier)()
    tokenizer = get_tokenizer()

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    # NOTE: gradient_checkpointing is INCOMPATIBLE with use_cache=True
    # (HF silently sets use_cache=False, returning None for past_key_values).
    # KV cache (1.79× speedup) wins over checkpointing's activation savings.
    if args.compile:
        # dynamic=True so the rolling LM context's varying length doesn't
        # trigger dynamo recompiles per shape (hits recompile_limit=8 within
        # a few chunks otherwise). See `scripts/experiment_compile_dynamic.py`.
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=True,
        )
        print("Compiled model.forward_window (cold-start on first step ~1-3 min).")
    optimizer = build_optimizer(model, lr_memory=args.lr_memory, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    trainer = Phase1Trainer(
        model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        pad_token_id=tokenizer.pad_token_id,
        use_kv_cache=args.use_kv_cache,
        prior_loss_weight=args.prior_loss_weight,
    )
    if args.use_kv_cache:
        print("KV cache enabled.")

    if args.checkpoint_in:
        if args.warm_start:
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=None, scheduler=None,
                map_location=args.device,
            )
            print(f"Warm-started model from {args.checkpoint_in} "
                  f"(optimizer/scheduler/step reset)")
        else:
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=optimizer, scheduler=scheduler,
                map_location=args.device,
            )
            trainer.load_state_dict({"step_count": ckpt.get("step", 0)})
            from src.trajectory_memory.training.checkpoint import restore_rng_state
            if "rng_state" in ckpt:
                restore_rng_state(ckpt["rng_state"])
            print(f"Resumed from {args.checkpoint_in} step {ckpt.get('step')}")

    dataset = TurnPairDataset(
        args.data_paths, batch_size=args.batch_size, pad_id=tokenizer.pad_token_id,
    )
    # B6 — on resume, advance dataset epoch counter so resumed run gets a
    # fresh shuffle instead of replaying the head of the deterministic
    # shuffle order. Same logic as train_wave1.py.
    if trainer.step_count > 0:
        dataset._epoch = trainer.step_count
        print(f"Dataset epoch advanced to {trainer.step_count} for resume")
    print(f"Wave 2 dataset: {len(dataset._rows)} TurnPairs, {len(dataset)} batches/epoch")

    val_dataset = None
    if args.val_data_paths:
        val_dataset = TurnPairDataset(
            args.val_data_paths, batch_size=args.batch_size, pad_id=tokenizer.pad_token_id,
        )
        print(f"Validation: {len(val_dataset._rows)} TurnPairs, "
              f"{args.val_batches} batches per eval")

    def to_device(b):
        b.prior_ids = b.prior_ids.to(args.device)
        b.response_ids = b.response_ids.to(args.device)
        b.prior_mask = b.prior_mask.to(args.device)
        b.response_mask = b.response_mask.to(args.device)
        return b

    def run_val() -> float:
        if val_dataset is None:
            return float("nan")
        losses_v: list[float] = []
        for i, batch_v in enumerate(val_dataset):
            if i >= args.val_batches:
                break
            losses_v.append(trainer.eval_wave2(to_device(batch_v)))
        return sum(losses_v) / max(len(losses_v), 1)

    losses: list = []
    t_start = time.time()
    last_plot_t = time.time()
    last_step_t = time.time()
    history: dict = {
        "step": [], "loss": [], "grad_norm": [], "lr": [],
        "surprise_mean": [], "surprise_std": [],
        "tok_per_sec": [], "vram_peak_gb": [],
        "val_step": [], "val_loss": {},
    }

    while trainer.step_count < args.num_steps:
        for batch in dataset:
            if trainer.step_count >= args.num_steps:
                break
            to_device(batch)

            metrics = trainer.step_wave2(batch)
            losses.append(metrics.loss)

            step = trainer.step_count
            now = time.time()
            history["step"].append(step)
            history["loss"].append(metrics.loss)
            history["grad_norm"].append(metrics.grad_norm)
            history["lr"].append(list(metrics.lr))
            if metrics.surprise_history is not None:
                ss = surprise_stats(metrics.surprise_history)
                history["surprise_mean"].append(ss["mean"])
                history["surprise_std"].append(ss["std"])
            else:
                history["surprise_mean"].append(0.0)
                history["surprise_std"].append(0.0)
            try:
                for comp, val in grad_norms_by_component(model).items():
                    history.setdefault(f"grad_norm_{comp}", []).append(val)
            except Exception:
                pass
            n_tok = batch.prior_ids.shape[1] + batch.response_ids.shape[1]
            history["tok_per_sec"].append(
                batch.prior_ids.shape[0] * n_tok / max(now - last_step_t, 1e-6),
            )
            last_step_t = now
            history["vram_peak_gb"].append(vram_stats()["peak_gb"])

            if step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1)
                elapsed = time.time() - t_start
                lrs = " ".join(f"{lr:.2e}" for lr in metrics.lr)
                print(f"  step {step:>5}  loss={metrics.loss:.4f}  avg10={avg:.4f}  "
                      f"grad_norm={metrics.grad_norm:.2f}  lr=[{lrs}]  "
                      f"({elapsed/max(step, 1):.2f}s/step)")

            if step > 0 and step % args.save_every == 0:
                if val_dataset is not None:
                    val_loss = run_val()
                    print(f"  step {step:>5}  val_loss={val_loss:.4f}")
                    history["val_step"].append(step)
                    history["val_loss"].setdefault("val", []).append(val_loss)
                if args.checkpoint_out is not None:
                    save_checkpoint(
                        args.checkpoint_out,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        step=step,
                        rng_state=capture_rng_state(),
                        extra={"config": cfg.__dict__, "losses": losses},
                    )
                    print(f"  saved checkpoint to {args.checkpoint_out} at step {step}")

            # Live plot refresh.
            if args.plot_path is not None:
                if now - last_plot_t > args.plot_every_seconds:
                    save_training_plots(history, args.plot_path)
                    dump_history_json(history, args.plot_path.with_suffix(".json"))
                    last_plot_t = now
                    print(f"  step {step:>5}  plot saved to {args.plot_path}")

    if args.checkpoint_out:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "losses": losses},
        )
        print(f"Final checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
