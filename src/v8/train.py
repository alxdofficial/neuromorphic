"""Entry point for v8/v9 training.

Usage:
    python -m src.v8.train                        # defaults
    python -m src.v8.train --bs 8 --steps 5000    # override
    python -m src.v8.train --no-memory             # LM-only baseline (no memory graph)
    python -m src.v8.train --no-compile            # disable torch.compile

End-to-end backprop: LM + differentiable memory graph trained jointly.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from .config import V8Config
from .model import V8Model
from .trainer import V8Trainer
from .diagnostics import V8Diagnostics
from ..data import create_dataloader, get_tokenizer, get_special_token_ids

# ============================================================================
# Defaults
# ============================================================================

TIER = "a"
BS = 8
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
SAVE_DIR = "outputs/v8"


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="v8/v9 training")
    p.add_argument("--bs", type=int, default=BS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--save-dir", type=str, default=SAVE_DIR)
    p.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    p.add_argument("--tokenizer", type=str, default=TOKENIZER)
    p.add_argument("--no-memory", action="store_true",
                   help="Disable memory graph (LM-only baseline)")
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--grad-ckpt", action="store_true", default=False,
                   help="Enable gradient checkpointing")
    p.add_argument("--keep-checkpoints", type=int, default=3,
                   help="Keep only the last N checkpoints (0=keep all)")
    p.add_argument("--snapshot-interval", type=int, default=1000,
                   help="Memory graph snapshot interval (0=disabled)")
    p.add_argument("--plot-interval", type=int, default=500,
                   help="Plot generation interval")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--mem-lr-scale", type=float, default=0.3,
                   help="Memory graph LR = LM_LR * this scale (default 0.3)")
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # Tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
    print(f"Tokenizer: {args.tokenizer} (vocab={vocab_size}, eot={eot_id})")

    # Config
    config = V8Config.tier_a()
    config.vocab_size = vocab_size
    config.eot_id = eot_id
    if args.compile is not None:
        config.use_compile = args.compile
    config.gradient_checkpointing = args.grad_ckpt
    config.validate()

    bs = args.bs
    T = config.T
    tokens_per_step = bs * T
    print(f"\nConfig: D={config.D}, C={config.C}, D_cc={config.D_cc}")
    print(f"  Scan: L_total={config.L_total}, split_at={config.scan_split_at}, "
          f"d_inner={config.d_inner}")
    print(f"  Memory: {config.N_neurons} neurons, {config.K_connections} connections, "
          f"D_mem={config.D_mem}")
    print(f"  Modulator: hidden={config.modulator_hidden}")
    print(f"  Training: BS={bs}, T={T}, tokens/step={tokens_per_step:,}")
    print(f"  Segments: {config.segments_per_chunk} per chunk, "
          f"{config.action_every} tokens each")

    # Model
    model = V8Model(config)
    if device.type == "cuda":
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device)

    # Resume from checkpoint
    start_step = 0
    _resume_ckpt = None
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        _resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(_resume_ckpt["model_state_dict"], strict=False)
        start_step = _resume_ckpt.get("step", 0)
        print(f"  Loaded model from step {start_step}")
        if "memory_runtime_state" in _resume_ckpt:
            model.memory.load_runtime_state(_resume_ckpt["memory_runtime_state"])
            model._states_initialized = True
            print(f"  Restored memory graph runtime state")

    # Promote modulator params to float32 AFTER load_state_dict
    # (bf16 precision rounds away small RL-like gradients)
    if device.type == "cuda":
        for name, param in model.memory.named_parameters():
            if 'fc1' in name or 'fc2' in name or 'mod_lr' in name:
                param.data = param.data.float()

    total_params = model.param_count()
    lm_params = model.lm_param_count()
    mem_params = model.memory_param_count()
    print(f"\nParameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  LM: {lm_params:,} ({lm_params/1e6:.1f}M)")
    print(f"  Memory graph: {mem_params:,} ({mem_params/1e6:.1f}M)")

    # Compile
    if config.use_compile and device.type == "cuda":
        print("Compiling model methods...")
        model.lm.forward_scan_lower = torch.compile(model.lm.forward_scan_lower)
        model.lm.forward_scan_upper = torch.compile(model.lm.forward_scan_upper)
        model.lm.forward_output = torch.compile(model.lm.forward_output)

    # Single optimizer with param groups
    lm_decay_params = []
    lm_no_decay_params = []
    for name, param in model.lm.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            lm_no_decay_params.append(param)
        else:
            lm_decay_params.append(param)

    mem_params_list = [p for p in model.memory.parameters() if p.requires_grad]
    mem_lr = args.lr * args.mem_lr_scale

    optimizer = torch.optim.AdamW(
        [
            {"params": lm_decay_params, "weight_decay": WEIGHT_DECAY, "lr": args.lr},
            {"params": lm_no_decay_params, "weight_decay": 0.0, "lr": args.lr},
            {"params": mem_params_list, "weight_decay": 0.0, "lr": mem_lr},
        ],
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # LR scheduler: warmup + cosine decay
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / args.lr + (1.0 - LR_MIN / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume optimizer/scheduler state (reuse checkpoint loaded earlier)
    if _resume_ckpt is not None:
        if "optimizer_state_dict" in _resume_ckpt:
            try:
                optimizer.load_state_dict(_resume_ckpt["optimizer_state_dict"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")
        if "scheduler_state_dict" in _resume_ckpt:
            try:
                scheduler.load_state_dict(_resume_ckpt["scheduler_state_dict"])
                print(f"  Loaded scheduler state")
            except Exception:
                pass
        del _resume_ckpt

    # Data
    dataloader = create_dataloader(
        phase="A",
        tokenizer=tokenizer,
        batch_size=bs,
        seq_length=T,
        seed=args.seed,
        max_steps=args.steps,
    )

    # Trainer
    trainer = V8Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        config=config,
        device=device,
        max_grad_norm=MAX_GRAD_NORM,
        log_interval=args.log_interval,
        use_memory=not args.no_memory,
    )
    trainer.global_step = start_step

    # Output dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({k: v for k, v in config.__dict__.items()
                   if not k.startswith("_")}, f, indent=2, default=str)

    # Diagnostics
    diag = V8Diagnostics(model, save_dir, snapshot_every=args.snapshot_interval)
    saved_checkpoints = []

    print(f"\nOutputs: {save_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Snapshots: every {args.snapshot_interval} steps")
    print(f"Checkpoints: every {args.save_interval} steps (keep last {args.keep_checkpoints})")
    print(f"Memory graph LR: {mem_lr:.2e} ({args.mem_lr_scale}x LM LR)")
    print(f"\nStarting training ({args.steps} steps)...")
    print("=" * 60)

    # Training loop
    def on_step(metrics):
        step = trainer.global_step

        metrics = diag.extend_metrics(metrics, step)

        record = {"step": step, **metrics}
        metrics_file.write(json.dumps(
            {k: round(v, 6) if isinstance(v, float) else v
             for k, v in record.items()}
        ) + "\n")
        if step % 10 == 0:
            metrics_file.flush()

        if step % args.log_interval == 0:
            mem_str = ""
            if "mem_h_norm" in metrics:
                mem_str = (f" | h={metrics['mem_h_norm']:.1f}"
                           f" prim_drift={metrics.get('mem_prim_drift', 0):.4f}"
                           f" gate={metrics.get('mem_mod_gate_prim_mean', 0):.4f}")
            print(f"  step {step:5d} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"ppl={metrics['ppl']:.1f} | "
                  f"tok/s={metrics['tok_s']/1e3:.1f}K | "
                  f"lr={metrics['lr']:.2e}"
                  f"{mem_str}")

        diag.maybe_snapshot(step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"v8_step{step}.pt")
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "config": config,
            }
            if model.memory is not None and model.memory.is_initialized():
                ckpt["memory_runtime_state"] = model.memory.runtime_state_dict()
            torch.save(ckpt, ckpt_path)
            saved_checkpoints.append(ckpt_path)
            print(f"  Saved: {ckpt_path}")

            if args.keep_checkpoints > 0:
                while len(saved_checkpoints) > args.keep_checkpoints:
                    old = saved_checkpoints.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                        print(f"  Removed old: {old}")

        should_plot = (
            (args.plot_interval > 0 and step % args.plot_interval == 0) or
            (args.snapshot_interval > 0 and step % args.snapshot_interval == 0) or
            (args.save_interval > 0 and step % args.save_interval == 0) or
            step == 50
        )
        if should_plot:
            try:
                from scripts.plot_training import (
                    load_metrics as _load_m, plot_training_curves,
                    plot_modulator_health, plot_memory_health, plot_pcm_health,
                    plot_connectivity_snapshot, plot_neuron_graph,
                )
                plot_dir = os.path.join(save_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                _records = _load_m(metrics_path)
                plot_training_curves(_records, os.path.join(plot_dir, "training_curves.png"))
                plot_modulator_health(_records, os.path.join(plot_dir, "modulator_health.png"))
                plot_memory_health(_records, os.path.join(plot_dir, "memory_health.png"))
                plot_pcm_health(_records, os.path.join(plot_dir, "pcm_health.png"))
                snap_dir = os.path.join(save_dir, "snapshots")
                if os.path.exists(snap_dir):
                    snaps = sorted(os.listdir(snap_dir))
                    if snaps:
                        latest_snap = os.path.join(snap_dir, snaps[-1])
                        plot_connectivity_snapshot(
                            latest_snap, os.path.join(plot_dir, "connectivity_latest.png"))
                        plot_neuron_graph(
                            latest_snap, os.path.join(plot_dir, "neuron_graph_latest.png"))
            except Exception as e:
                print(f"  Plot generation failed: {e}")

    all_metrics = trainer.train_epoch(args.steps, step_callback=on_step)

    # Final checkpoint
    final_path = os.path.join(save_dir, f"v8_step{trainer.global_step}.pt")
    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": trainer.global_step,
        "config": config,
    }
    if model.memory is not None and model.memory.is_initialized():
        final_ckpt["memory_runtime_state"] = model.memory.runtime_state_dict()
    torch.save(final_ckpt, final_path)

    metrics_file.close()

    if all_metrics:
        final = all_metrics[-1]
        print(f"\nDone. Final loss: {final['loss']:.4f}, ppl: {final['ppl']:.1f}")
    print(f"Checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
