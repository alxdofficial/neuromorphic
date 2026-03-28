"""
Entry point for v9-backprop training.

Usage:
    python -m src.v8.train                        # defaults
    python -m src.v8.train --bs 8 --steps 5000    # override
    python -m src.v8.train --no-memory             # LM-only baseline
    python -m src.v8.train --no-compile            # disable torch.compile
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

# Defaults
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
    p = argparse.ArgumentParser(description="v9-backprop training")
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
    p.add_argument("--grad-ckpt", action="store_true", default=False)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--snapshot-interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--action-every", type=int, default=None)
    return p.parse_args()


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
    if args.action_every is not None:
        config.action_every = args.action_every
    config.validate()

    bs = args.bs
    T = config.T
    print(f"\nConfig: D={config.D}, C={config.C}")
    print(f"  Scan: L_total={config.L_total}, split_at={config.scan_split_at}")
    print(f"  Memory: {config.N_neurons} scalar neurons, K={config.K_connections}, "
          f"{config.n_groups} groups × {config.group_size}, "
          f"{config.replicas_per_dim} replicas/dim")
    print(f"  Modulator: hidden={config.neuromod_hidden}, "
          f"plasticity={'on' if config.structural_plasticity else 'off'}")
    print(f"  Training: BS={bs}, T={T}, mem_lr_scale={config.mem_lr_scale}")

    # Model — everything in bf16, optimizer keeps f32 master weights
    model = V8Model(config)
    if device.type == "cuda":
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device)

    # Resume
    start_step = 0
    _pending_mg_state = None
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.lm.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "memory_params" in ckpt:
            model.memory.load_state_dict(ckpt["memory_params"], strict=False)
        if "memory_state" in ckpt:
            _pending_mg_state = ckpt["memory_state"]
            print(f"  Memory state found (will restore on first batch)")
        start_step = ckpt.get("step", 0)
        print(f"  Loaded from step {start_step}")

    lm_params = model.lm_param_count()
    mem_params = model.memory_param_count()
    total_params = model.param_count()
    print(f"\nParameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  LM: {lm_params:,} ({lm_params/1e6:.1f}M)")
    print(f"  Memory: {mem_params:,} ({mem_params/1e6:.1f}M)")

    if args.no_memory:
        print("\n*** MEMORY DISABLED — LM-only baseline ***")

    # Compile
    if config.use_compile and device.type == "cuda":
        print("Compiling model methods...")
        model.lm.forward_scan_lower = torch.compile(model.lm.forward_scan_lower)
        model.lm.forward_scan_upper = torch.compile(model.lm.forward_scan_upper)
        model.lm.forward_output = torch.compile(model.lm.forward_output)

    # Optimizer — LM + memory param groups
    lm_decay, lm_no_decay = [], []
    for name, param in model.lm.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            lm_no_decay.append(param)
        else:
            lm_decay.append(param)

    mem_lr = args.lr * config.mem_lr_scale
    mem_decay, mem_no_decay = [], []
    for name, param in model.memory.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            mem_no_decay.append(param)
        else:
            mem_decay.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lm_decay, "weight_decay": WEIGHT_DECAY, "lr": args.lr},
        {"params": lm_no_decay, "weight_decay": 0.0, "lr": args.lr},
        {"params": mem_decay, "weight_decay": WEIGHT_DECAY * 0.1, "lr": mem_lr},
        {"params": mem_no_decay, "weight_decay": 0.0, "lr": mem_lr},
    ], betas=(0.9, 0.95), fused=(device.type == "cuda"))

    # LR scheduler
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / args.lr + (1.0 - LR_MIN / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume optimizer
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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
        model=model, optimizer=optimizer, scheduler=scheduler,
        dataloader=dataloader, config=config, device=device,
        max_grad_norm=MAX_GRAD_NORM, log_interval=args.log_interval,
        use_memory=not args.no_memory)

    trainer.global_step = start_step

    # Restore memory state
    if _pending_mg_state is not None:
        model.initialize_states(bs, device)
        model.memory.load_runtime_state(_pending_mg_state)
        trainer._states_initialized = True
        print(f"  Restored memory state")

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
    print(f"Starting training ({args.steps} steps)...")
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
            mem_str = ""
            if "mem_h_norm" in metrics:
                mem_str = (f" | h={metrics['mem_h_norm']:.1f}"
                           f" mg={metrics.get('mem_grad_norm', 0):.3f}")
            print(f"  step {step:5d} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"ppl={metrics['ppl']:.1f} | "
                  f"tok/s={metrics['tok_s']/1e3:.1f}K | "
                  f"lr={metrics['lr']:.2e}"
                  f"{mem_str}")

        diag.maybe_snapshot(step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"v9_step{step}.pt")
            ckpt = {
                "model_state_dict": model.lm.state_dict(),
                "memory_params": model.memory.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step, "config": config,
            }
            if model.memory.is_initialized():
                ckpt["memory_state"] = {
                    k: v.clone() for k, v in model.memory.runtime_state_dict().items()}
            torch.save(ckpt, ckpt_path)
            saved_checkpoints.append(ckpt_path)
            print(f"  Saved: {ckpt_path}")

            if args.keep_checkpoints > 0:
                while len(saved_checkpoints) > args.keep_checkpoints:
                    old = saved_checkpoints.pop(0)
                    if os.path.exists(old):
                        os.remove(old)

    all_metrics = trainer.train_epoch(args.steps, step_callback=on_step)

    # Final checkpoint
    final_path = os.path.join(save_dir, f"v9_step{trainer.global_step}.pt")
    final_ckpt = {
        "model_state_dict": model.lm.state_dict(),
        "memory_params": model.memory.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": trainer.global_step, "config": config,
    }
    if model.memory.is_initialized():
        final_ckpt["memory_state"] = {
            k: v.clone() for k, v in model.memory.runtime_state_dict().items()}
    torch.save(final_ckpt, final_path)
    metrics_file.close()

    if all_metrics:
        final = all_metrics[-1]
        print(f"\nDone. Final loss: {final['loss']:.4f}, ppl: {final['ppl']:.1f}")
    print(f"Checkpoint: {final_path}")


if __name__ == "__main__":
    main()
