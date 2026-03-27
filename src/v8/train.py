"""Entry point for v8/v9 training.

Usage:
    python -m src.v8.train                        # defaults (LM backprop + ES memory)
    python -m src.v8.train --no-memory             # LM-only baseline
    python -m src.v8.train --no-compile            # disable torch.compile

LM trained by backprop. Memory graph trained by Evolution Strategies.
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


def parse_args():
    p = argparse.ArgumentParser(description="v8/v9 training (LM backprop + ES memory)")
    p.add_argument("--bs", type=int, default=BS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--save-dir", type=str, default=SAVE_DIR)
    p.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    p.add_argument("--tokenizer", type=str, default=TOKENIZER)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--snapshot-interval", type=int, default=1000)
    p.add_argument("--plot-interval", type=int, default=500)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--es-warmup", type=int, default=None,
                   help="Steps before ES starts (default: from config)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    tokenizer = get_tokenizer(args.tokenizer)
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
    print(f"Tokenizer: {args.tokenizer} (vocab={vocab_size}, eot={eot_id})")

    config = V8Config.tier_a()
    config.vocab_size = vocab_size
    config.eot_id = eot_id
    if args.compile is not None:
        config.use_compile = args.compile
    if args.es_warmup is not None:
        config.es_warmup = args.es_warmup
    config.validate()

    bs = args.bs
    T = config.T
    print(f"\nConfig: D={config.D}, C={config.C}, D_cc={config.D_cc}")
    print(f"  Scan: L_total={config.L_total}, split_at={config.scan_split_at}, d_inner={config.d_inner}")
    print(f"  Memory: {config.N_neurons} neurons, {config.K_connections} connections")
    print(f"  ES: K={config.es_k_neurons}, N_traj={config.es_n_trajectories}, "
          f"σ={config.es_sigma}, lr={config.es_lr}, collect={config.es_collect_chunks}")
    print(f"  Training: BS={bs}, T={T}")

    # Model
    model = V8Model(config)
    if device.type == "cuda":
        model = model.to(device).to(torch.bfloat16)

    # Resume
    start_step = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_step = ckpt.get("step", 0)
        print(f"  Loaded model from step {start_step}")
        if "memory_runtime_state" in ckpt:
            model.memory.load_runtime_state(ckpt["memory_runtime_state"])
            model._states_initialized = True
        del ckpt

    lm_params = model.lm_param_count()
    mem_params = model.memory_param_count()
    print(f"\nParameters: {model.param_count():,} total")
    print(f"  LM (backprop): {lm_params:,} ({lm_params/1e6:.1f}M)")
    print(f"  Memory (ES): {mem_params:,} ({mem_params/1e6:.1f}M)")

    # Compile
    if config.use_compile and device.type == "cuda":
        print("Compiling model methods...")
        model.lm.forward_scan_lower = torch.compile(model.lm.forward_scan_lower)
        model.lm.forward_scan_upper = torch.compile(model.lm.forward_scan_upper)
        model.lm.forward_output = torch.compile(model.lm.forward_output)

    # LM optimizer (memory graph params have requires_grad=False, excluded automatically)
    decay_params = []
    no_decay_params = []
    for name, param in model.lm.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    lm_optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
        fused=(device.type == "cuda"))

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / args.lr + (1.0 - LR_MIN / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(lm_optimizer, lr_lambda)

    # Resume optimizer
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ckpt:
            try:
                lm_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  Could not load optimizer: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        del ckpt

    # Data
    dataloader = create_dataloader(
        phase="A", tokenizer=tokenizer, batch_size=bs,
        seq_length=T, seed=args.seed, max_steps=args.steps)

    # Trainer
    trainer = V8Trainer(
        model=model, lm_optimizer=lm_optimizer, scheduler=scheduler,
        dataloader=dataloader, config=config, device=device,
        max_grad_norm=MAX_GRAD_NORM, log_interval=args.log_interval,
        use_memory=not args.no_memory)
    trainer.global_step = start_step

    # Output dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({k: v for k, v in config.__dict__.items()
                   if not k.startswith("_")}, f, indent=2, default=str)

    diag = V8Diagnostics(model, save_dir, snapshot_every=args.snapshot_interval)
    saved_checkpoints = []

    print(f"\nOutputs: {save_dir}")
    print(f"\nStarting training ({args.steps} steps)...")
    print("=" * 60)

    def on_step(metrics):
        step = trainer.global_step
        metrics = diag.extend_metrics(metrics, step)

        record = {"step": step, **metrics}
        metrics_file.write(json.dumps(
            {k: round(v, 6) if isinstance(v, float) else v
             for k, v in record.items()}) + "\n")
        if step % 10 == 0:
            metrics_file.flush()

        if step % args.log_interval == 0:
            es_str = ""
            if "es_loss_mean" in metrics:
                es_str = (f" | es_mean={metrics['es_loss_mean']:.4f}"
                          f" best={metrics['es_loss_best']:.4f}")
            mem_str = ""
            if "mem_h_norm" in metrics:
                mem_str = (f" | h={metrics['mem_h_norm']:.1f}"
                           f" key_drift={metrics.get('mem_key_drift', 0):.4f}"
                           f" gate={metrics.get('mem_mod_gate_prim_mean', 0):.4f}")
            print(f"  step {step:5d} | loss={metrics['loss']:.4f} | "
                  f"ppl={metrics['ppl']:.1f} | tok/s={metrics['tok_s']/1e3:.1f}K | "
                  f"lr={metrics['lr']:.2e}{es_str}{mem_str}")

        diag.maybe_snapshot(step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"v8_step{step}.pt")
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": lm_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step, "config": config,
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

        should_plot = (
            (args.plot_interval > 0 and step % args.plot_interval == 0) or
            (args.save_interval > 0 and step % args.save_interval == 0) or
            step == 50)
        if should_plot:
            try:
                from scripts.plot_training import (
                    load_metrics as _load_m, plot_training_curves,
                    plot_es_health, plot_modulator_health,
                    plot_memory_health, plot_pcm_health,
                    plot_connectivity_snapshot, plot_neuron_graph)
                plot_dir = os.path.join(save_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                _records = _load_m(metrics_path)
                plot_training_curves(_records, os.path.join(plot_dir, "training_curves.png"))
                plot_es_health(_records, os.path.join(plot_dir, "es_health.png"))
                plot_modulator_health(_records, os.path.join(plot_dir, "modulator_health.png"))
                plot_memory_health(_records, os.path.join(plot_dir, "memory_health.png"))
                plot_pcm_health(_records, os.path.join(plot_dir, "pcm_health.png"))
            except Exception as e:
                print(f"  Plot failed: {e}")

    all_metrics = trainer.train_epoch(args.steps, step_callback=on_step)

    final_path = os.path.join(save_dir, f"v8_step{trainer.global_step}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": lm_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": trainer.global_step, "config": config,
        "memory_runtime_state": model.memory.runtime_state_dict()
            if model.memory.is_initialized() else None,
    }, final_path)

    metrics_file.close()
    if all_metrics:
        print(f"\nDone. Final loss: {all_metrics[-1]['loss']:.4f}")
    print(f"Checkpoint: {final_path}")


if __name__ == "__main__":
    main()
