"""
Entry point for v8 Neural Memory Graph training.

Usage:
    python -m src.v8.train                        # defaults
    python -m src.v8.train --bs 8 --steps 5000    # override
    python -m src.v8.train --no-memory             # LM-only baseline (no memory graph)
    python -m src.v8.train --no-compile            # disable torch.compile

Trains the v8 model with joint LM backprop + sampling-based RL for neuromodulator.
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
    p = argparse.ArgumentParser(description="v8 Neural Memory Graph training")
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
    p.add_argument("--n-samples", type=int, default=4,
                   help="Number of neuromod action trajectories to sample (default 4)")
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
    print(f"  Scan: L_total={config.L_total}, d_inner={config.d_inner} (single pass, end-injection)")
    print(f"  Memory: {config.N_neurons} neurons, {config.C} blocks × {config.M_per_block}, "
          f"D_mem={config.D_mem}")
    print(f"  Neuromod: hidden={config.neuromod_hidden}, layers={config.neuromod_layers}, "
          f"action_every={config.action_every}")
    print(f"  Training: BS={bs}, T={T}, tokens/step={tokens_per_step:,}")
    print(f"  RL: sampling-based, n_samples={args.n_samples}, "
          f"action_every={config.action_every}")

    # Model
    model = V8Model(config)
    if device.type == "cuda":
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device)

    lm_params = model.lm.param_count()
    total_params = model.param_count()
    print(f"\nParameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  LM: {lm_params:,} ({lm_params/1e6:.1f}M)")
    print(f"  Neuromod: {total_params - lm_params:,} ({(total_params-lm_params)/1e6:.1f}M)")

    # Disable memory if requested (LM-only baseline)
    if args.no_memory:
        print("\n*** MEMORY DISABLED — LM-only baseline ***")

    # Compile
    if config.use_compile and device.type == "cuda":
        print("Compiling model...")
        model.lm = torch.compile(model.lm)

    # LM Optimizer — exclude biases and norms from weight decay
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
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # Neuromodulator optimizer (separate, no weight decay)
    neuromod_optimizer = torch.optim.Adam(
        model.neuromod.parameters(), lr=config.neuromod_lr, eps=1e-5,
    )

    # LR scheduler: warmup + cosine decay
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / args.lr + (1.0 - LR_MIN / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(lm_optimizer, lr_lambda)

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
    trainer_use_memory = not args.no_memory
    n_samples = args.n_samples if trainer_use_memory else 1
    trainer = V8Trainer(
        model=model,
        lm_optimizer=lm_optimizer,
        neuromod_optimizer=neuromod_optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        config=config,
        device=device,
        max_grad_norm=MAX_GRAD_NORM,
        log_interval=args.log_interval,
        use_memory=trainer_use_memory,
        n_samples=n_samples,
    )

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

    print(f"\nOutputs: {save_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"\nStarting training ({args.steps} steps)...")
    print("=" * 60)

    # Training loop
    def on_step(metrics):
        # Log to JSONL
        record = {"step": trainer.global_step, **metrics}
        metrics_file.write(json.dumps(
            {k: round(v, 6) if isinstance(v, float) else v
             for k, v in record.items()}
        ) + "\n")
        if trainer.global_step % 10 == 0:
            metrics_file.flush()

        # Print periodic summary
        if trainer.global_step % args.log_interval == 0:
            rl_str = ""
            if "rl_policy_loss" in metrics:
                rl_str = (f" | rl_loss={metrics['rl_policy_loss']:.4f}"
                          f" adv_std={metrics.get('rl_advantage_std', 0):.4f}")
            print(f"  step {trainer.global_step:5d} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"ppl={metrics['ppl']:.1f} | "
                  f"tok/s={metrics['tok_s']/1e3:.1f}K | "
                  f"lr={metrics['lr']:.2e}"
                  f"{rl_str}")

        # Save checkpoint
        if args.save_interval > 0 and trainer.global_step % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"v8_step{trainer.global_step}.pt")
            torch.save({
                "model_state_dict": model.lm.state_dict(),
                "neuromod_state_dict": model.neuromod.state_dict(),
                "optimizer_state_dict": lm_optimizer.state_dict(),
                "neuromod_optimizer_state_dict": neuromod_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": trainer.global_step,
                "config": config,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    all_metrics = trainer.train_epoch(args.steps, step_callback=on_step)

    # Final checkpoint
    final_path = os.path.join(save_dir, f"v8_step{trainer.global_step}.pt")
    torch.save({
        "model_state_dict": model.lm.state_dict(),
        "neuromod_state_dict": model.neuromod.state_dict(),
        "optimizer_state_dict": lm_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": trainer.global_step,
        "config": config,
    }, final_path)

    metrics_file.close()

    if all_metrics:
        final = all_metrics[-1]
        print(f"\nDone. Final loss: {final['loss']:.4f}, ppl: {final['ppl']:.1f}")
    print(f"Checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
