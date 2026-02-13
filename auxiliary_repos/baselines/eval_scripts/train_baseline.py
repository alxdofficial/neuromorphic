"""
Train baseline models from scratch on the same data as the neuromorphic LM.

This trains Pythia-160M and Mamba-130M architectures from random initialization
on FineWeb-Edu (60%) + DCLM (40%), using the same TinyLlama tokenizer and
token budget as our model for an apples-to-apples comparison.

Usage:
    python train_baseline.py --model pythia-160m --steps 300000 --bs 48
    python train_baseline.py --model mamba-130m --steps 300000 --bs 48

    # Quick test run
    python train_baseline.py --model pythia-160m --steps 100 --bs 8

Notes:
    - Models are initialized from scratch (random weights), NOT from pretrained
    - Uses the same tokenizer as our neuromorphic LM (TinyLlama, 32K vocab)
    - Uses the same data mix (FineWeb-Edu 60% + DCLM 40%)
    - Logs metrics to JSONL for comparison plotting
    - Saves checkpoints periodically
"""

import argparse
import json
import math
import os
import sys
import time
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ============================================================================
# Constants (match neuromorphic LM training)
# ============================================================================

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEQ_LENGTH = 256  # Must match our T=256
GRAD_ACCUM = 1

# Datasets (same as our Phase B)
FINEWEB_EDU_PATH = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_NAME = "sample-10BT"
DCLM_PATH = "mlfoundations/dclm-baseline-1.0"
MIX_WEIGHTS = [0.6, 0.4]  # FineWeb-Edu 60%, DCLM 40%

# Optimizer (standard transformer training)
LR = 6e-4
MIN_LR = 6e-5  # 10% of max LR
WARMUP_FRACTION = 0.01  # 1% of total steps
WEIGHT_DECAY = 0.01
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# Logging
LOG_INTERVAL = 50
VAL_INTERVAL = 500
SAVE_INTERVAL = 5000

# ============================================================================
# Model configs (from-scratch architecture definitions)
# ============================================================================

MODEL_CONFIGS = {
    "pythia-160m": {
        "model_type": "gpt_neox",
        "config_class": "GPTNeoXConfig",
        "config_kwargs": {
            "vocab_size": 32000,  # TinyLlama tokenizer
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,  # 4x hidden
            "max_position_embeddings": 2048,
            "rotary_pct": 0.25,
            "use_parallel_residual": True,
            "layer_norm_eps": 1e-5,
        },
    },
    "mamba-130m": {
        "model_type": "mamba",
        "config_class": "MambaConfig",
        "config_kwargs": {
            "vocab_size": 32000,  # TinyLlama tokenizer
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "state_size": 16,
            "expand": 2,
            "conv_kernel": 4,
            "use_bias": False,
            "use_conv_bias": True,
        },
    },
}


# ============================================================================
# Data pipeline
# ============================================================================

class StreamingTokenDataset:
    """Stream tokens from HuggingFace datasets, matching our training pipeline."""

    def __init__(self, tokenizer, seq_length: int, batch_size: int, seed: int = 42):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.eos_token_id = tokenizer.eos_token_id

        # Load datasets in streaming mode
        print("Loading FineWeb-Edu (streaming)...")
        ds_fineweb = load_dataset(
            FINEWEB_EDU_PATH, FINEWEB_EDU_NAME,
            split="train", streaming=True,
        )
        print("Loading DCLM (streaming)...")
        ds_dclm = load_dataset(
            DCLM_PATH, split="train", streaming=True,
        )

        # Interleave with mix weights
        self.dataset = interleave_datasets(
            [ds_fineweb, ds_dclm],
            probabilities=MIX_WEIGHTS,
            seed=seed,
            stopping_strategy="all_exhausted",
        )
        self._iter = iter(self.dataset)
        self._token_buffer = []

    def _fill_buffer(self, min_tokens: int):
        """Fill token buffer from streaming dataset."""
        while len(self._token_buffer) < min_tokens:
            try:
                example = next(self._iter)
            except StopIteration:
                # Restart stream
                self._iter = iter(self.dataset)
                example = next(self._iter)

            text = example.get("text", "")
            if not text.strip():
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.eos_token_id)
            self._token_buffer.extend(tokens)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of (input_ids, labels) tensors."""
        needed = self.batch_size * (self.seq_length + 1)
        self._fill_buffer(needed)

        batch_tokens = []
        for _ in range(self.batch_size):
            chunk = self._token_buffer[:self.seq_length + 1]
            self._token_buffer = self._token_buffer[self.seq_length + 1:]
            batch_tokens.append(chunk)

        tokens = torch.tensor(batch_tokens, dtype=torch.long)
        input_ids = tokens[:, :-1]   # [BS, T]
        labels = tokens[:, 1:]       # [BS, T]
        return input_ids, labels


# ============================================================================
# Metrics logger
# ============================================================================

class MetricsLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a")

    def log(self, record: dict):
        clean = {}
        for k, v in record.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    clean[k] = None
                else:
                    clean[k] = round(v, 6)
            else:
                clean[k] = v
        self._f.write(json.dumps(clean) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ============================================================================
# Training loop
# ============================================================================

def create_model(model_name: str, device: str) -> AutoModelForCausalLM:
    """Create a model from scratch (random init)."""
    cfg_spec = MODEL_CONFIGS[model_name]

    if cfg_spec["model_type"] == "gpt_neox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        config = GPTNeoXConfig(**cfg_spec["config_kwargs"])
        model = GPTNeoXForCausalLM(config)
    elif cfg_spec["model_type"] == "mamba":
        from transformers import MambaConfig, MambaForCausalLM
        config = MambaConfig(**cfg_spec["config_kwargs"])
        model = MambaForCausalLM(config)
    else:
        raise ValueError(f"Unknown model type: {cfg_spec['model_type']}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Config: {config}")

    return model.to(device)


def train(
    model_name: str,
    max_steps: int,
    batch_size: int,
    output_dir: str,
    resume_from: str = None,
    seed: int = 42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # Output paths
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    logger = MetricsLogger(metrics_path)

    # Tokenizer (same as neuromorphic LM)
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer)}")

    # Model
    model = create_model(model_name, device)
    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    # LR scheduler (cosine with warmup, same shape as our training)
    warmup_steps = max(int(max_steps * WARMUP_FRACTION), 100)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # Mixed precision — Mamba needs bf16 (fp16 causes NaN with selective scan)
    is_mamba = "mamba" in model_name
    if device == "cuda" and is_mamba:
        amp_dtype = torch.bfloat16
        scaler = None  # bf16 doesn't need GradScaler
        use_amp = True
    elif device == "cuda":
        amp_dtype = torch.float16
        scaler = GradScaler("cuda")
        use_amp = True
    else:
        amp_dtype = torch.float32
        scaler = None
        use_amp = False

    # Data
    print("Setting up data pipeline...")
    data = StreamingTokenDataset(tokenizer, SEQ_LENGTH, batch_size, seed=seed)

    # Resume
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed at step {start_step}")

    # Save config
    config_record = {
        "model_name": model_name,
        "n_params": n_params,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "seq_length": SEQ_LENGTH,
        "lr": LR,
        "min_lr": MIN_LR,
        "warmup_steps": warmup_steps,
        "tokenizer": TOKENIZER_NAME,
        "data_mix": "fineweb-edu(0.6)+dclm(0.4)",
        "tokens_per_step": batch_size * SEQ_LENGTH,
        "total_tokens": max_steps * batch_size * SEQ_LENGTH,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    # Training
    tokens_per_step = batch_size * SEQ_LENGTH
    total_tokens = max_steps * tokens_per_step

    print(f"\n{'='*60}")
    print(f"Training {model_name} from scratch")
    print(f"  Steps: {max_steps:,}")
    print(f"  Tokens/step: {tokens_per_step:,} (BS={batch_size} x T={SEQ_LENGTH})")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  LR: {LR} -> {MIN_LR} (warmup {warmup_steps} steps)")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    model.train()
    t_start = time.time()
    running_loss = 0.0
    running_tokens = 0

    for step in range(start_step, max_steps):
        t_step = time.time()

        # Get batch
        input_ids, labels = data.get_batch()
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward + backward
        optimizer.zero_grad()
        if use_amp:
            with autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        # Metrics
        loss_val = loss.item()
        ppl = min(math.exp(loss_val), 1e6) if not math.isnan(loss_val) else 1e6
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_step
        tok_s = tokens_per_step / elapsed if elapsed > 0 else 0

        running_loss += loss_val
        running_tokens += tokens_per_step

        # Log
        if (step + 1) % LOG_INTERVAL == 0 or step == start_step:
            avg_loss = running_loss / LOG_INTERVAL if step > start_step else loss_val
            total_elapsed = time.time() - t_start
            tokens_done = (step + 1 - start_step) * tokens_per_step
            avg_tok_s = tokens_done / total_elapsed if total_elapsed > 0 else 0

            record = {
                "step": step + 1,
                "mode": "train",
                "model": model_name,
                "loss": loss_val,
                "avg_loss": avg_loss,
                "ppl": ppl,
                "lr": lr,
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "tok_s": tok_s,
                "avg_tok_s": avg_tok_s,
                "elapsed": total_elapsed,
                "tokens_seen": tokens_done,
            }
            logger.log(record)

            print(
                f"[{step+1:>7d}/{max_steps}] "
                f"loss={loss_val:.4f} ppl={ppl:.1f} "
                f"lr={lr:.2e} gnorm={record['grad_norm']:.2f} "
                f"tok/s={tok_s:.0f} (avg={avg_tok_s:.0f}) "
                f"elapsed={total_elapsed:.0f}s"
            )
            running_loss = 0.0

        # Checkpoint
        if (step + 1) % SAVE_INTERVAL == 0 or step + 1 == max_steps:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step+1}.pt")
            ckpt = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss_val,
                "config": config_record,
            }
            if scaler:
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            # Also save as "latest"
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            torch.save(ckpt, latest_path)

    total_time = time.time() - t_start
    total_tokens_actual = (max_steps - start_step) * tokens_per_step
    print(f"\n{'='*60}")
    print(f"Training complete: {model_name}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Total tokens: {total_tokens_actual:,} ({total_tokens_actual/1e9:.2f}B)")
    print(f"  Avg throughput: {total_tokens_actual/total_time:.0f} tok/s")
    print(f"  Final loss: {loss_val:.4f}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    logger.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models from scratch on neuromorphic LM data"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS.keys()), required=True,
        help="Which baseline architecture to train",
    )
    parser.add_argument(
        "--steps", type=int, default=300_000,
        help="Total training steps (default: 300K to match Phase B)",
    )
    parser.add_argument(
        "--bs", type=int, default=48,
        help="Batch size (default: 48 to match our training)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/baseline_<model>/)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        "outputs", f"baseline_{args.model.replace('-', '_')}"
    )

    train(
        model_name=args.model,
        max_steps=args.steps,
        batch_size=args.bs,
        output_dir=output_dir,
        resume_from=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
