"""
Entry point for neuromorphic LM training.

Usage:
    python -m src.train

All configuration is hardcoded below. Edit the values directly,
comment/uncomment alternatives as needed.
"""

import math
import os
import torch

from .model import ModelConfig, NeuromorphicLM
from .model.state import save_runtime_state, load_runtime_state
from .data import get_tokenizer, get_special_token_ids, create_dataloader
from .training import TBPTTTrainer
from .debug import MetricsCollector


# ============================================================================
# Configuration — edit these directly
# ============================================================================

# -- Model tier --
TIER = "a"          # ~41M params, fast iteration
# TIER = "b"        # ~67M params, competitive
# TIER = "c"        # ~107M params, strong

# -- Training phase --
PHASE = "A"         # WM only (sanity check on TinyStories)
# PHASE = "B"       # WM + PM
# PHASE = "C"       # WM + PM + EM
# PHASE = "D"       # WM + PM + EM + RL controllers

# -- Data phase (which datasets to use) --
DATA_PHASE = "A"    # TinyStories
# DATA_PHASE = "B"  # FineWeb-Edu + DCLM

# -- Tokenizer --
TOKENIZER = "tinyllama"
# TOKENIZER = "gpt2"
# TOKENIZER = "smollm"

# -- Batch size (persistent streams) --
BS = 32             # Tier A default
# BS = 16           # Tier B default
# BS = 8            # Tier C default

# -- Learning rate --
LR = 3e-4
LR_MIN = 1e-5
WARMUP_STEPS = 1000

# -- Training length --
MAX_STEPS = 5000

# -- Regularization --
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# -- Logging --
LOG_INTERVAL = 50

# -- Checkpointing --
SAVE_DIR = "checkpoints"
SAVE_INTERVAL = 1000
RESUME = None       # set to checkpoint path to resume, e.g. "checkpoints/neuromorphic_a_A_step5000.pt"

# -- Metrics collection --
METRICS_FILE = "checkpoints/metrics.jsonl"
COLLECT_EVERY = 50  # full collection every N steps (0 = disabled)

# -- Device --
DEVICE = None       # auto-detect (cuda if available, else cpu)
# DEVICE = "cuda"
# DEVICE = "cpu"


# ============================================================================
# Training script — no need to edit below
# ============================================================================

def get_device() -> torch.device:
    if DEVICE:
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_config() -> ModelConfig:
    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b, "c": ModelConfig.tier_c}[TIER]
    config = tier_fn()
    config.set_phase(PHASE)
    return config


def main():
    device = get_device()
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer(TOKENIZER)
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = tokenizer.vocab_size
    eot_id = special_ids.get("eos", tokenizer.eos_token_id)
    print(f"Tokenizer: {TOKENIZER} (vocab={vocab_size}, eot={eot_id})")

    # Config
    config = build_config()
    config.vocab_size = vocab_size
    config.eot_id = eot_id

    print(f"Tier: {TIER.upper()} | Phase: {PHASE} | "
          f"D={config.D}, L={config.L}, B={config.B}, D_h={config.D_h}")
    print(f"BS={BS}, T={config.T}, P={config.P}")
    print(f"WM={config.wm_enabled}, PM={config.pm_enabled}, EM={config.em_enabled}")

    # Model
    model = NeuromorphicLM(config).to(device)
    param_count = model.param_count()
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # LR scheduler: linear warmup + cosine decay
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(MAX_STEPS - WARMUP_STEPS, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return LR_MIN / LR + (1.0 - LR_MIN / LR) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    if RESUME:
        print(f"Resuming from {RESUME}")
        ckpt = torch.load(RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "runtime_state" in ckpt:
            load_runtime_state(model, ckpt["runtime_state"])
        start_step = ckpt.get("step", 0)
    else:
        start_step = 0

    # Dataloader
    dataloader = create_dataloader(
        phase=DATA_PHASE,
        tokenizer=tokenizer,
        batch_size=BS,
        seq_length=config.T,
        max_steps=MAX_STEPS,
    )
    print(f"Data phase: {DATA_PHASE}")

    # Metrics collector
    collector = MetricsCollector(
        model=model,
        config=config,
        output_path=METRICS_FILE,
        collect_every=COLLECT_EVERY,
    )
    print(f"Metrics: {METRICS_FILE} (full every {COLLECT_EVERY} steps)")

    # Trainer
    trainer = TBPTTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        config=config,
        device=device,
        max_grad_norm=MAX_GRAD_NORM,
        log_interval=LOG_INTERVAL,
        collector=collector,
    )
    trainer.global_step = start_step

    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    remaining = MAX_STEPS - start_step
    metrics = trainer.train_epoch(remaining)

    # Final save
    if metrics:
        save_path = os.path.join(
            SAVE_DIR,
            f"neuromorphic_{TIER}_{PHASE}_step{trainer.global_step}.pt"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": trainer.global_step,
            "config": config,
            "runtime_state": save_runtime_state(model),
        }, save_path)
        print(f"\nSaved checkpoint: {save_path}")

    collector.close()

    final = metrics[-1] if metrics else {}
    print(f"\nTraining complete. Final loss: {final.get('loss', 'N/A')}")


if __name__ == "__main__":
    main()
