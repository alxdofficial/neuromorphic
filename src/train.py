"""Training entry point.

Usage:
    python -m src.train                         # defaults
    python -m src.train --bs 8 --steps 5000     # override
    python -m src.train --no-memory              # LM-only baseline
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .model.config import Config
from .model.model import Model
from .trainer import Trainer
from .data import create_dataloader, get_tokenizer, get_special_token_ids

BS = 96   # default matches train_loop.py's orchestrated pipeline
LR = 3e-4
LR_MIN = 3e-5
WARMUP_STEPS = 1000
MAX_STEPS = 10000
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42
TOKENIZER = "tinyllama"
LOG_INTERVAL = 50
SAVE_INTERVAL = 5000
EVAL_INTERVAL = 500
EVAL_BATCHES = 8
SAVE_DIR = "outputs/v12"


def parse_args():
    p = argparse.ArgumentParser(description="Neuromorphic LM training")
    p.add_argument("--bs", type=int, default=BS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--save-dir", type=str, default=SAVE_DIR)
    p.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL,
                   help="Run eval every N steps (0 = disable)")
    p.add_argument("--eval-batches", type=int, default=EVAL_BATCHES,
                   help="Number of scored batches in eval (excl. warmup)")
    p.add_argument("--eval-warmup-batches", type=int, default=4,
                   help="Batches to run before scoring (warm memory state)")
    p.add_argument("--eval-bs", type=int, default=None,
                   help="Batch size for eval (defaults to --bs)")
    p.add_argument("--tokenizer", type=str, default=TOKENIZER)
    p.add_argument("--no-memory", action="store_true",
                   help="Disable memory graph (LM-only baseline)")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--d-inner", type=int, default=None)
    p.add_argument("--lr-target-step", type=int, default=None,
                   help="Cosine schedule denominator (LR reaches LR_MIN at "
                        "this optimizer step count). Defaults to --steps. "
                        "Pass a separate value to decouple the LR schedule "
                        "from the loop termination target — e.g. when "
                        "action collection steps should not count toward "
                        "LR decay.")
    p.add_argument("--freeze-modulator", action="store_true",
                   help="Freeze modulator params (cycle 1+ phase 1)")
    p.add_argument("--collect-actions", action="store_true",
                   help="Collect modulator actions to a buffer (for codebook fit)")
    p.add_argument("--action-db-out", type=str, default=None,
                   help="Path to save action database when --collect-actions")
    p.add_argument("--no-train", action="store_true",
                   help="Pure inference: skip backward and optimizer step. "
                        "Used for action collection so the codebook is fit on "
                        "a stationary LM rather than a moving target.")
    p.add_argument("--merge-interval", type=int, default=100,
                   help="Periodically average the memory's W/decay/hebbian "
                        "across the batch dimension every N steps and "
                        "broadcast back. Enforces 'one shared memory graph' "
                        "semantics. 0 disables.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed ALL randomness sources that affect model init / dataloader jitter /
    # numpy.random / Python random. Previously only `args.seed` was passed to
    # the dataloader and `torch.manual_seed` was never called, so two runs
    # with the same --seed produced different model initializations. Verified
    # empirically before this fix.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    tokenizer = get_tokenizer(args.tokenizer)
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
    print(f"Tokenizer: {args.tokenizer} (vocab={vocab_size}, eot={eot_id})")

    config = Config.tier_a()
    config.vocab_size = vocab_size
    config.eot_id = eot_id
    if args.d_inner is not None:
        config.d_inner = args.d_inner
    config.validate()

    bs = args.bs
    T = config.T
    print(f"\nConfig: D={config.D}, D_n={config.D_n}, N_cells={config.N_cells}")
    print(f"  Scan: L_total={config.L_total}, split_at={config.scan_split_at}, "
          f"d_inner={config.d_inner}")
    print(f"  Memory: {config.N} neurons, K={config.K}, D_n={config.D_n}")
    print(f"  Ports: {config.N_port} input + {config.N_port} output, "
          f"alpha={config.alpha}")
    print(f"  Modulator: hidden={config.cell_mod_hidden}")
    print(f"  Training: BS={bs}, T={T}, mem_lr_scale={config.mem_lr_scale}")

    model = Model(config).to(device)

    lm_params = model.lm_param_count()
    mem_params = model.memory_param_count()
    total_params = model.param_count()
    print(f"\nParameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  LM: {lm_params:,} ({lm_params / 1e6:.1f}M)")
    print(f"  Memory: {mem_params:,} ({mem_params / 1e6:.1f}M)")

    if args.no_memory:
        print("\n*** MEMORY DISABLED — LM-only baseline ***")

    # Apply modulator freeze BEFORE building the optimizer. Note that
    # frozen params are STILL included in the optimizer's param groups —
    # we only skip weight-decay gating via `requires_grad`. PyTorch's AdamW
    # skips frozen-param updates (they have no .grad), and keeping them in
    # the optimizer preserves param-group sizes across the bootstrap →
    # cycle-1 transition so optimizer.load_state_dict() doesn't silently
    # fail on a param-count mismatch. See audit #2.
    if args.freeze_modulator:
        for p in (model.memory.mod_w1, model.memory.mod_b1,
                  model.memory.mod_w2, model.memory.mod_b2):
            p.requires_grad = False

    # Optimizer — include ALL params (including frozen ones) so param-group
    # shapes are stable across freeze/unfreeze transitions.
    lm_decay, lm_no_decay = [], []
    for name, param in model.lm.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            lm_no_decay.append(param)
        else:
            lm_decay.append(param)

    mem_lr = args.lr * config.mem_lr_scale
    mem_decay, mem_no_decay = [], []
    for name, param in model.memory.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            mem_no_decay.append(param)
        else:
            mem_decay.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lm_decay, "weight_decay": WEIGHT_DECAY, "lr": args.lr},
        {"params": lm_no_decay, "weight_decay": 0.0, "lr": args.lr},
        {"params": mem_decay, "weight_decay": WEIGHT_DECAY, "lr": mem_lr},
        {"params": mem_no_decay, "weight_decay": 0.0, "lr": mem_lr},
    ], betas=(0.9, 0.95), fused=(device.type == "cuda"))

    lr_target_step = args.lr_target_step if args.lr_target_step is not None else args.steps

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(lr_target_step - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / args.lr + (1.0 - LR_MIN / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_step = 0
    pending_runtime_state = None
    pending_dataloader_state = None
    loaded_phase = "phase1"
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  WARN: {len(missing)} missing keys in checkpoint "
                  f"(will stay at fresh init): {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  WARN: {len(unexpected)} unexpected keys in checkpoint "
                  f"(ignored): {unexpected[:5]}"
                  f"{'...' if len(unexpected) > 5 else ''}")
        optimizer_loaded = False
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                optimizer_loaded = True
            except Exception as e:
                # Loud warning — silent restart-with-fresh-Adam bit us before
                # (cycle-boundary param_group size mismatch from freeze).
                print(f"  !!!! WARN: could not load optimizer state: {e}")
                print(f"  !!!! Proceeding with FRESH Adam momentum. "
                      f"This is almost certainly not what you want — "
                      f"investigate the checkpoint/param-group shape.")
        scheduler_loaded = False
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                scheduler_loaded = True
            except Exception:
                pass
        pending_runtime_state = ckpt.get("runtime_state")
        pending_dataloader_state = ckpt.get("dataloader_state")
        start_step = ckpt.get("step", 0)
        loaded_phase = ckpt.get("phase", "phase1")
        print(f"  Loaded from step {start_step} "
              f"(opt={'Y' if optimizer_loaded else 'N'} "
              f"sched={'Y' if scheduler_loaded else 'N'} "
              f"runtime={'Y' if pending_runtime_state else 'N'})")
        del ckpt

        # If scheduler state is missing (e.g. old checkpoint), reconstruct
        # at the correct position instead of restarting from warmup.
        if not scheduler_loaded and start_step > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda, last_epoch=start_step - 1)
            print(f"  Reconstructed scheduler at step {start_step}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

    remaining_steps = args.steps - start_step
    dataloader = create_dataloader(
        phase="A", tokenizer=tokenizer, batch_size=bs,
        seq_length=T, seed=args.seed, max_steps=remaining_steps)
    if pending_dataloader_state is not None and hasattr(dataloader, "load_state_dict"):
        dataloader.load_state_dict(pending_dataloader_state)

    # Eval uses the held-out validation shard (phase="A-val"), NOT the
    # training shard. Previously both used phase="A" which meant eval
    # numbers were in-sample. Eval BS defaults to training BS but can
    # be overridden for parity across phase 1 and phase 2.
    eval_bs = args.eval_bs if args.eval_bs is not None else bs
    total_eval_batches = args.eval_batches + args.eval_warmup_batches
    eval_loader_factory = None
    if args.eval_interval > 0:
        def _make_eval_loader():
            return create_dataloader(
                phase="A-val", tokenizer=tokenizer, batch_size=eval_bs,
                seq_length=T, seed=args.seed + 1000,
                max_steps=total_eval_batches)
        eval_loader_factory = _make_eval_loader

    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        dataloader=dataloader, config=config, device=device,
        max_grad_norm=MAX_GRAD_NORM, log_interval=args.log_interval,
        use_memory=not args.no_memory,
        freeze_modulator=args.freeze_modulator,
        collect_actions=args.collect_actions,
        metrics_path=metrics_path,
        no_train=args.no_train,
        merge_interval=args.merge_interval,
    )
    if args.freeze_modulator:
        print("*** Modulator FROZEN — phase 1 of iterative cycle ***")
    if args.collect_actions:
        print("*** Action collection ENABLED — will flush to disk ***")
    trainer.global_step = start_step
    if pending_runtime_state is not None:
        model.load_runtime_state(pending_runtime_state)
        # Loaded state may have been saved at a different BS (e.g. phase 2
        # at BS=8 → cycle phase 1 at BS=96). Detect mismatch and reshape via
        # collapse + broadcast so the next phase 1 starts with a consensus
        # state at the correct BS. The phase-2 saver already collapses
        # before saving so this is a no-op divergence-wise, but the
        # broadcast is still needed to land on phase 1's BS.
        bs_mismatch = (model.memory._initialized
                       and model.memory.W.shape[0] != bs)
        if bs_mismatch:
            print(f"  Runtime state BS={model.memory.W.shape[0]} != "
                  f"phase-1 BS={bs}; collapsing + broadcasting.")
            model.memory.collapse_batch_dim()
            model.memory.broadcast_to_bs(bs)
        # Phase 2 ckpts: drop LM carries even if BS happens to match. They
        # were last touched mid-rollout (per-trajectory upper carries reset
        # frequently, lower carries follow rollout sequences) and don't
        # represent a coherent "ready for next batch" state for phase 1.
        # The phase 1 warmup loop repopulates them.
        if bs_mismatch or loaded_phase == "phase2":
            model.lm._carries = [None] * config.L_total
            model._initialized = model.memory._initialized

    os.makedirs(args.save_dir, exist_ok=True)

    def save_checkpoint(step):
        path = os.path.join(args.save_dir, f"ckpt_{step:06d}.pt")
        # Mark the true consumer position + consumer-side prev_tokens before
        # dumping dataloader state. This decouples the checkpoint from the
        # prefetch thread's runahead (both step count and prev_tokens).
        if hasattr(dataloader, "mark_consumed"):
            consumer_prev = getattr(trainer, "last_consumed_prev_tokens", None)
            dataloader.mark_consumed(step - start_step, prev_tokens=consumer_prev)
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "runtime_state": model.runtime_state_dict(),
            "dataloader_state": (
                dataloader.state_dict() if hasattr(dataloader, "state_dict") else {}
            ),
            "config": config,
        }, path)
        print(f"  Saved checkpoint: {path}")
        # Also regenerate plots from the current jsonl so the user has an
        # always-fresh visual after each save. Done in a try/except so a
        # plotting bug never crashes training.
        try:
            from scripts.plot_training import (
                load_metrics, plot_phase1_training, plot_phase1_gradients,
                plot_phase1_memory,
            )
            metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
            if os.path.exists(metrics_path):
                records = load_metrics(metrics_path)
                if records:
                    plots_dir = os.path.join(args.save_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    plot_phase1_training(records,
                        os.path.join(plots_dir, "phase1_training.png"))
                    plot_phase1_gradients(records,
                        os.path.join(plots_dir, "phase1_gradients.png"))
                    plot_phase1_memory(records,
                        os.path.join(plots_dir, "phase1_memory.png"))
        except Exception as e:
            print(f"  (plot regen failed: {e})")

    def step_callback(metrics):
        step = trainer.global_step
        if step % args.log_interval == 0:
            print(f"[step {step}] loss={metrics['loss']:.3f} "
                  f"ppl={metrics['ppl']:.1f} "
                  f"tok/s={metrics['tok_s']/1e3:.1f}K "
                  f"lm_gn={metrics['lm_grad_norm']:.2f} "
                  f"dyn_gn={metrics['dyn_grad_norm']:.2f} "
                  f"mod_gn={metrics['mod_clip_norm']:.3f} "
                  f"a_norm={metrics['mod_action_norm']:.4f} "
                  f"W_off={metrics.get('W_offdiag_norm', 0):.3f} "
                  f"h={metrics.get('h_norm', 0):.3f}")
        if (
            eval_loader_factory is not None
            and args.eval_interval > 0
            and step > 0
            and step % args.eval_interval == 0
        ):
            eval_metrics = trainer.evaluate(
                eval_loader_factory(), n_batches=args.eval_batches,
                eval_loader_off=eval_loader_factory(),
                warmup_batches=args.eval_warmup_batches)
            mem_gap_str = ""
            if "mem_leverage_ce" in eval_metrics:
                mem_gap_str = (
                    f" mem_off={eval_metrics['eval_ce_loss_no_mem']:.3f} "
                    f"leverage={eval_metrics['mem_leverage_ce']:+.3f}")
            print(f"[eval {step}] "
                  f"ce={eval_metrics['eval_ce_loss']:.3f} "
                  f"ppl={eval_metrics['eval_ppl']:.1f} "
                  f"mem_pred={eval_metrics['eval_aux_loss']:.3f}"
                  f"{mem_gap_str} "
                  f"({eval_metrics['eval_batches']} batches)")
            # Append to metrics jsonl as a dedicated eval row so plotting can
            # distinguish train vs eval curves.
            trainer._append_metrics({
                "step": step,
                "event": "eval",
                **eval_metrics,
            })
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(step)

    print(f"\nTraining for {remaining_steps} steps...")
    trainer.train_epoch(remaining_steps, step_callback=step_callback)
    save_checkpoint(trainer.global_step)

    if args.collect_actions and args.action_db_out:
        os.makedirs(os.path.dirname(args.action_db_out) or ".", exist_ok=True)
        n = trainer.flush_action_database(args.action_db_out)
        print(f"Flushed {n:,} action samples to {args.action_db_out}")

    print("Done.")


if __name__ == "__main__":
    main()
