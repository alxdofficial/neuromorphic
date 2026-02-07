"""
Entry point for neuromorphic LM training.

Usage:
    python -m src.train

All configuration is hardcoded below. Edit the values directly,
comment/uncomment alternatives as needed.
"""

import hashlib
import json
import math
import os
import subprocess
import time
import torch

from .model import ModelConfig, NeuromorphicLM
from .model.state import save_runtime_state, load_runtime_state
from .data import get_tokenizer, get_special_token_ids, create_dataloader
from .training import TBPTTTrainer, evaluate_validation
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

# -- Seeds --
TRAIN_SEED = 42

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

# -- Validation / monitoring --
VAL_INTERVAL = 200      # run held-out eval every N train steps (0 = disabled)
VAL_STEPS = 20          # validation chunks per eval pass
VAL_DATA_PHASE = DATA_PHASE
VAL_SEED = 4242
ABLATE_INTERVAL = 1000  # run PM/EM ablation eval every N steps (0 = disabled)

# -- Safety checks --
SAFETY_FAIL_FAST = True
MAX_CONSEC_ZERO_VALID = 3

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


def get_git_commit() -> str:
    """Best-effort current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def config_digest(config: ModelConfig) -> str:
    """Stable short hash for the active config."""
    payload = json.dumps(config.__dict__, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def main():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    device = get_device()
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer(TOKENIZER)
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = tokenizer.vocab_size
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
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

    # Optimizer — exclude biases and LayerNorm from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
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
        seed=TRAIN_SEED,
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
    collector.log_record({
        "mode": "run_meta",
        "run_id": run_id,
        "start_unix": time.time(),
        "git_commit": get_git_commit(),
        "tier": TIER.upper(),
        "phase": PHASE,
        "data_phase": DATA_PHASE,
        "val_data_phase": VAL_DATA_PHASE,
        "tokenizer": TOKENIZER,
        "seed_train": TRAIN_SEED,
        "seed_val": VAL_SEED,
        "config_digest": config_digest(config),
        "config": dict(config.__dict__),
    })

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
        fail_fast=SAFETY_FAIL_FAST,
        max_consecutive_zero_valid=MAX_CONSEC_ZERO_VALID,
    )
    trainer.global_step = start_step

    # Restore last prev_token to prevent false doc-boundary reset on resume
    if RESUME and ckpt.get("last_prev_token") is not None:
        trainer.override_prev_token = ckpt["last_prev_token"]

    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    def save_checkpoint(step: int):
        save_path = os.path.join(
            SAVE_DIR,
            f"neuromorphic_{TIER}_{PHASE}_{run_id}_step{step}.pt"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": config,
            "runtime_state": save_runtime_state(model),
            "last_prev_token": getattr(trainer, "_last_prev_token", None),
        }, save_path)
        print(f"\nSaved checkpoint: {save_path}")

    def run_validation(step: int, pm_enabled: bool, em_enabled: bool,
                       mode: str) -> dict:
        val_loader = create_dataloader(
            phase=VAL_DATA_PHASE,
            tokenizer=tokenizer,
            batch_size=BS,
            seq_length=config.T,
            seed=VAL_SEED,
            max_steps=VAL_STEPS,
        )
        t0 = time.time()
        metrics = evaluate_validation(
            model=model,
            dataloader=val_loader,
            config=config,
            device=device,
            num_steps=VAL_STEPS,
            pm_enabled=pm_enabled,
            em_enabled=em_enabled,
        )
        elapsed = time.time() - t0
        record = {
            "mode": mode,
            "step": step,
            "val_loss": metrics["loss"],
            "val_ppl": metrics["ppl"],
            "val_valid_tokens": metrics["valid_tokens"],
            "val_steps_done": metrics["steps_done"],
            "val_valid_fraction": metrics["valid_fraction"],
            "val_eot_input_fraction": metrics["eot_input_fraction"],
            "val_reset_fraction": metrics["reset_fraction"],
            "val_elapsed": elapsed,
            "val_pm_enabled": pm_enabled,
            "val_em_enabled": em_enabled,
        }
        collector.log_record(record)
        return record

    def on_step(_step_metrics: dict):
        if SAVE_INTERVAL > 0 and trainer.global_step % SAVE_INTERVAL == 0:
            save_checkpoint(trainer.global_step)

        val_record = None
        if VAL_INTERVAL > 0 and trainer.global_step % VAL_INTERVAL == 0:
            val_record = run_validation(
                trainer.global_step,
                pm_enabled=config.pm_enabled,
                em_enabled=config.em_enabled,
                mode="val",
            )
            print(
                f"[val] step {trainer.global_step:5d} | "
                f"loss {val_record['val_loss']:.4f} | "
                f"ppl {val_record['val_ppl']:.1f} | "
                f"valid {val_record['val_valid_fraction']:.3f}"
            )

        if ABLATE_INTERVAL > 0 and trainer.global_step % ABLATE_INTERVAL == 0:
            if val_record is None:
                val_record = run_validation(
                    trainer.global_step,
                    pm_enabled=config.pm_enabled,
                    em_enabled=config.em_enabled,
                    mode="val",
                )
            off_record = run_validation(
                trainer.global_step,
                pm_enabled=False,
                em_enabled=False,
                mode="ablate",
            )
            ablate = {
                "mode": "ablate_summary",
                "step": trainer.global_step,
                "ablate_ppl_normal": val_record["val_ppl"],
                "ablate_ppl_plasticity_off": off_record["val_ppl"],
                "ablate_gap": off_record["val_ppl"] - val_record["val_ppl"],
            }
            collector.log_record(ablate)
            print(
                f"[ablate] step {trainer.global_step:5d} | "
                f"normal {ablate['ablate_ppl_normal']:.1f} | "
                f"off {ablate['ablate_ppl_plasticity_off']:.1f} | "
                f"gap {ablate['ablate_gap']:+.2f}"
            )

    remaining = MAX_STEPS - start_step
    metrics = trainer.train_epoch(remaining, step_callback=on_step)

    # Final save
    if metrics:
        save_checkpoint(trainer.global_step)

    collector.close()

    final = metrics[-1] if metrics else {}
    print(f"\nTraining complete. Final loss: {final.get('loss', 'N/A')}")


if __name__ == "__main__":
    main()
