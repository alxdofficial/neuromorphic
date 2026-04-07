"""Training entry point.

Usage:
    python -m src.train                         # defaults
    python -m src.train --bs 8 --steps 5000     # override
    python -m src.train --no-memory              # LM-only baseline
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

from .model.config import Config
from .model.model import Model
from .trainer import Trainer
from .data import create_dataloader, get_tokenizer, get_special_token_ids

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
    p.add_argument("--tokenizer", type=str, default=TOKENIZER)
    p.add_argument("--no-memory", action="store_true",
                   help="Disable memory graph (LM-only baseline)")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--d-inner", type=int, default=None)
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

    config = Config.tier_a()
    config.vocab_size = vocab_size
    config.eot_id = eot_id
    if args.d_inner is not None:
        config.d_inner = args.d_inner
    config.validate()

    bs = args.bs
    T = config.T
    print(f"\nConfig: D={config.D}, D_n={config.D_n}, C={config.C}")
    print(f"  Scan: L_total={config.L_total}, split_at={config.scan_split_at}, "
          f"d_inner={config.d_inner}")
    print(f"  Memory: {config.N} neurons, K={config.K}, D_n={config.D_n}")
    print(f"  Ports: {config.N_port} input + {config.N_port} output, "
          f"alpha={config.alpha}")
    print(f"  Modulator: hidden={config.neuromod_hidden}")
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

    # Optimizer
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

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / args.lr + (1.0 - LR_MIN / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_step = 0
    pending_runtime_state = None
    pending_dataloader_state = None
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  Could not load optimizer: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        pending_runtime_state = ckpt.get("runtime_state")
        pending_dataloader_state = ckpt.get("dataloader_state")
        start_step = ckpt.get("step", 0)
        print(f"  Loaded from step {start_step}")
        del ckpt

    remaining_steps = args.steps - start_step
    dataloader = create_dataloader(
        phase="A", tokenizer=tokenizer, batch_size=bs,
        seq_length=T, seed=args.seed, max_steps=remaining_steps)
    if pending_dataloader_state is not None and hasattr(dataloader, "load_state_dict"):
        dataloader.load_state_dict(pending_dataloader_state)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        dataloader=dataloader, config=config, device=device,
        max_grad_norm=MAX_GRAD_NORM, log_interval=args.log_interval,
        use_memory=not args.no_memory,
    )
    trainer.global_step = start_step
    if pending_runtime_state is not None:
        model.load_runtime_state(pending_runtime_state)

    os.makedirs(args.save_dir, exist_ok=True)

    def save_checkpoint(step):
        path = os.path.join(args.save_dir, f"ckpt_{step:06d}.pt")
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

    def step_callback(metrics):
        step = trainer.global_step
        if step % args.log_interval == 0:
            print(f"[step {step}] loss={metrics['loss']:.3f} "
                  f"ppl={metrics['ppl']:.1f} "
                  f"tok/s={metrics['tok_s']/1e3:.1f}K "
                  f"lm_gn={metrics['lm_grad_norm']:.2f} "
                  f"mem_gn={metrics['mem_grad_norm']:.2f}")
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(step)

    print(f"\nTraining for {remaining_steps} steps...")
    trainer.train_epoch(remaining_steps, step_callback=step_callback)
    save_checkpoint(trainer.global_step)
    print("Done.")


if __name__ == "__main__":
    main()
