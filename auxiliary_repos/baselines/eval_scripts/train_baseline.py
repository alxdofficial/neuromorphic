"""
Train baseline models from scratch on the same data as the neuromorphic LM.

Trains baseline architectures from random initialization on FineWeb-Edu (60%)
+ DCLM (40%), using the same TinyLlama tokenizer and token budget as our
model for an apples-to-apples comparison.

Tier A (~100M):
    python train_baseline.py --model gpt2-small    # Transformer baseline (GPT-2)
    python train_baseline.py --model pythia-160m   # Transformer baseline (GPT-NeoX)
    python train_baseline.py --model mamba-130m    # SSM baseline
    python train_baseline.py --model rwkv7-168m    # Recurrent baseline

Tier B (~400M):
    python train_baseline.py --model gpt2-medium   # Transformer baseline (GPT-2)
    python train_baseline.py --model pythia-410m   # Transformer baseline (GPT-NeoX)
    python train_baseline.py --model mamba-370m    # SSM baseline
    python train_baseline.py --model rwkv7-421m    # Recurrent baseline

Tier C (~1B):
    python train_baseline.py --model pythia-1b     # Transformer baseline
    python train_baseline.py --model tinyllama-1b  # Transformer baseline (LLaMA arch)
    python train_baseline.py --model mamba-1.4b    # SSM baseline
    python train_baseline.py --model rwkv7-1.5b    # Recurrent baseline

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

import torch
import torch.nn.functional as F
from torch.amp import autocast
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
TOKEN_BUDGET = 1_500_000_000  # 1.5B tokens (matches neuromorphic training)
GRAD_ACCUM = 1

# Per-model optimal batch sizes (RTX 4090 24GB, bf16)
# Tier B/C models need smaller BS due to VRAM; adjust if using A100.
MODEL_OPTIMAL_BS = {
    # Tier A (~100M)
    "gpt2-small": 96,
    "pythia-160m": 96,
    "mamba-130m": 64,
    "rwkv7-169m": 64,
    # Tier B (~400M)
    "gpt2-medium": 32,
    "pythia-410m": 32,
    "mamba-370m": 32,
    "rwkv7-421m": 32,
    # Tier C (~1B)
    "pythia-1b": 8,
    "tinyllama-1b": 8,
    "mamba-1.4b": 8,
    "rwkv7-1.5b": 8,
}

# Datasets (same as our Phase B) — local pre-downloaded parquet files
# Run `python scripts/prepare_data.py` first to create these.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_ROOT, "..", "..", "..", "data", "phase_B"))
FINEWEB_EDU_LOCAL = os.path.join(_DATA_DIR, "fineweb_edu.parquet")
DCLM_LOCAL = os.path.join(_DATA_DIR, "dclm.parquet")
VAL_LOCAL = os.path.join(_DATA_DIR, "val_fineweb_edu.parquet")
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
SAVE_INTERVAL = 2000

# ============================================================================
# Model configs (from-scratch architecture definitions)
# ============================================================================

MODEL_CONFIGS = {
    # =================================================================
    # Tier A (~100M params)
    # =================================================================
    "gpt2-small": {
        "model_type": "gpt2",
        "config_kwargs": {
            "vocab_size": 32000,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_positions": 2048,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
        },
    },
    "pythia-160m": {
        "model_type": "gpt_neox",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 2048,
            "rotary_pct": 0.25,
            "use_parallel_residual": True,
            "layer_norm_eps": 1e-5,
        },
    },
    "mamba-130m": {
        "model_type": "mamba",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "state_size": 16,
            "expand": 2,
            "conv_kernel": 4,
            "use_bias": False,
            "use_conv_bias": True,
        },
    },
    "rwkv7-168m": {
        "model_type": "rwkv7",
        "hf_repo": "RWKV/RWKV7-Goose-Pile-168M-HF",  # for auto_map
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "head_dim": 64,
            "hidden_act": "sqrelu",
            "hidden_ratio": 4.0,
            "a_low_rank_dim": 64,
            "decay_low_rank_dim": 64,
            "gate_low_rank_dim": 128,
            "v_low_rank_dim": 32,
            "norm_first": True,
            "norm_bias": True,
            "norm_eps": 1e-5,
            "tie_word_embeddings": False,
        },
    },
    # =================================================================
    # Tier B (~400M params)
    # =================================================================
    "gpt2-medium": {
        "model_type": "gpt2",
        "config_kwargs": {
            "vocab_size": 32000,
            "n_embd": 1024,
            "n_layer": 24,
            "n_head": 16,
            "n_positions": 2048,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
        },
    },
    "mamba-370m": {
        "model_type": "mamba",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 48,
            "state_size": 16,
            "expand": 2,
            "conv_kernel": 4,
            "use_bias": False,
            "use_conv_bias": True,
        },
    },
    "pythia-410m": {
        "model_type": "gpt_neox",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "max_position_embeddings": 2048,
            "rotary_pct": 0.25,
            "use_parallel_residual": True,
            "layer_norm_eps": 1e-5,
        },
    },
    "rwkv7-421m": {
        "model_type": "rwkv7",
        "hf_repo": "RWKV/RWKV7-Goose-Pile-421M-HF",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "head_dim": 64,
            "hidden_act": "sqrelu",
            "hidden_ratio": 4.0,
            "a_low_rank_dim": 64,
            "decay_low_rank_dim": 64,
            "gate_low_rank_dim": 128,
            "v_low_rank_dim": 64,
            "norm_first": True,
            "norm_bias": True,
            "norm_eps": 1e-5,
            "tie_word_embeddings": False,
        },
    },
    # =================================================================
    # Tier C (~1B params)
    # =================================================================
    "pythia-1b": {
        "model_type": "gpt_neox",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 8,
            "intermediate_size": 8192,
            "max_position_embeddings": 2048,
            "rotary_pct": 0.25,
            "use_parallel_residual": True,
            "layer_norm_eps": 1e-5,
        },
    },
    "tinyllama-1b": {
        "model_type": "llama",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "max_position_embeddings": 2048,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": False,
        },
    },
    "mamba-1.4b": {
        "model_type": "mamba",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "state_size": 16,
            "expand": 2,
            "conv_kernel": 4,
            "use_bias": False,
            "use_conv_bias": True,
        },
    },
    "rwkv7-1.5b": {
        "model_type": "rwkv7",
        "hf_repo": "RWKV/RWKV7-Goose-Pile-1.47B-HF",
        "config_kwargs": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "intermediate_size": 8192,
            "head_dim": 64,
            "hidden_act": "sqrelu",
            "hidden_ratio": 4.0,
            "a_low_rank_dim": 64,
            "decay_low_rank_dim": 64,
            "gate_low_rank_dim": 128,
            "v_low_rank_dim": 64,
            "norm_first": True,
            "norm_bias": True,
            "norm_eps": 1e-5,
            "tie_word_embeddings": False,
        },
    },
}


# ============================================================================
# Data pipeline
# ============================================================================

class LocalTokenDataset:
    """Load tokens from local parquet files (no network calls during training).

    Run ``python scripts/prepare_data.py`` first to create the local files.
    """

    def __init__(self, tokenizer, seq_length: int, batch_size: int, seed: int = 42):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.eos_token_id = tokenizer.eos_token_id
        self._seed = seed

        self._build_dataset()
        self._token_buffer = []

    def _build_dataset(self):
        """Load and interleave local parquet datasets."""
        for path in (FINEWEB_EDU_LOCAL, DCLM_LOCAL):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Local data not found: {path}\n"
                    f"Run: python scripts/prepare_data.py"
                )

        print(f"Loading FineWeb-Edu (local): {FINEWEB_EDU_LOCAL}")
        ds_fineweb = load_dataset(
            "parquet", data_files=FINEWEB_EDU_LOCAL, split="train",
        )
        print(f"Loading DCLM (local): {DCLM_LOCAL}")
        ds_dclm = load_dataset(
            "parquet", data_files=DCLM_LOCAL, split="train",
        )

        self.dataset = interleave_datasets(
            [ds_fineweb, ds_dclm],
            probabilities=MIX_WEIGHTS,
            seed=self._seed,
            stopping_strategy="all_exhausted",
        )
        self._iter = iter(self.dataset)

    def _fill_buffer(self, min_tokens: int):
        """Fill token buffer from dataset."""
        while len(self._token_buffer) < min_tokens:
            try:
                example = next(self._iter)
            except StopIteration:
                # Recycle dataset
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
# Validation
# ============================================================================

def build_val_dataset(tokenizer, batch_size: int, n_batches: int = 100,
                      seed: int = 1337) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-fetch a fixed val set from local parquet (held-out from training).

    Materialised once at startup so val evaluation is fast and deterministic.
    """
    if not os.path.exists(VAL_LOCAL):
        raise FileNotFoundError(
            f"Local validation data not found: {VAL_LOCAL}\n"
            f"Run: python scripts/prepare_data.py"
        )
    print(f"Building validation set from {VAL_LOCAL} "
          f"({n_batches} batches, seed={seed})...", flush=True)
    ds = load_dataset("parquet", data_files=VAL_LOCAL,
                      split="train").shuffle(seed=seed)
    eos = tokenizer.eos_token_id
    buf = []
    batches = []
    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue
        toks = tokenizer.encode(text, add_special_tokens=False)
        toks.append(eos)
        buf.extend(toks)
        while len(buf) >= batch_size * (SEQ_LENGTH + 1):
            batch_tokens = []
            for _ in range(batch_size):
                chunk = buf[:SEQ_LENGTH + 1]
                buf = buf[SEQ_LENGTH + 1:]
                batch_tokens.append(chunk)
            t = torch.tensor(batch_tokens, dtype=torch.long)
            batches.append((t[:, :-1], t[:, 1:]))
            if len(batches) >= n_batches:
                break
        if len(batches) >= n_batches:
            break
    print(f"  Val set ready: {len(batches)} batches x {batch_size} x {SEQ_LENGTH}")
    return batches


@torch.no_grad()
def run_validation(model, val_batches: list, device: str,
                   amp_dtype: torch.dtype) -> dict:
    """Compute val loss/ppl over the pre-fetched val set."""
    model.train(False)
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in val_batches:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with autocast("cuda", dtype=amp_dtype):
            out = model(input_ids=input_ids, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
    model.train(True)
    avg_loss = total_loss / max(total_tokens, 1)
    return {"val_loss": avg_loss, "val_ppl": min(math.exp(avg_loss), 1e6)}


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
    model_type = cfg_spec["model_type"]

    if model_type == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(**cfg_spec["config_kwargs"])
        model = GPT2LMHeadModel(config)
    elif model_type == "gpt_neox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        config = GPTNeoXConfig(**cfg_spec["config_kwargs"])
        model = GPTNeoXForCausalLM(config)
    elif model_type == "mamba":
        from transformers import MambaConfig, MambaForCausalLM
        config = MambaConfig(**cfg_spec["config_kwargs"])
        model = MambaForCausalLM(config)
    elif model_type == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(**cfg_spec["config_kwargs"])
        model = LlamaForCausalLM(config)
    elif model_type == "rwkv7":
        # RWKV-7 uses custom code from the HF repo (flash-linear-attention)
        hf_repo = cfg_spec["hf_repo"]
        config = AutoConfig.from_pretrained(
            hf_repo, trust_remote_code=True,
        )
        # Override vocab_size for our tokenizer
        config.vocab_size = cfg_spec["config_kwargs"]["vocab_size"]
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Config: {config}")

    model = model.to(device)

    # torch.compile for transformer architectures
    # (Mamba's selective scan and RWKV's custom kernels don't compile cleanly)
    if model_type in ("gpt2", "gpt_neox", "llama"):
        print("  Compiling with torch.compile...")
        model = torch.compile(model)

    return model


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
        fused=(device == "cuda"),
    )

    # LR scheduler (cosine with warmup, same shape as our training)
    warmup_steps = max(int(max_steps * WARMUP_FRACTION), 100)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # Mixed precision — bf16 for all models (matches neuromorphic training)
    if device == "cuda":
        amp_dtype = torch.bfloat16
        scaler = None  # bf16 doesn't need GradScaler
        use_amp = True
    else:
        amp_dtype = torch.float32
        scaler = None
        use_amp = False

    # Data
    print("Setting up data pipeline...")
    data = LocalTokenDataset(tokenizer, SEQ_LENGTH, batch_size, seed=seed)
    val_batches = build_val_dataset(tokenizer, batch_size=min(batch_size, 32))

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
        optimizer.zero_grad(set_to_none=True)
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
            session_tokens = (step + 1 - start_step) * tokens_per_step
            avg_tok_s = session_tokens / total_elapsed if total_elapsed > 0 else 0

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
                "tokens_seen": (step + 1) * tokens_per_step,
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

        # Validation
        if (step + 1) % VAL_INTERVAL == 0:
            val = run_validation(model, val_batches, device, amp_dtype)
            val_record = {
                "step": step + 1,
                "mode": "val",
                "model": model_name,
                **val,
                "tokens_seen": (step + 1) * tokens_per_step,
            }
            logger.log(val_record)
            print(
                f"  [val] step={step+1} "
                f"val_loss={val['val_loss']:.4f} val_ppl={val['val_ppl']:.2f}"
            )

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
        "--steps", type=int, default=None,
        help="Total training steps (default: derived from 2B token budget and BS)",
    )
    parser.add_argument(
        "--bs", type=int, default=None,
        help="Batch size (default: per-model optimal for RTX 4090)",
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

    batch_size = args.bs or MODEL_OPTIMAL_BS.get(args.model, 32)
    tokens_per_step = batch_size * SEQ_LENGTH
    max_steps = args.steps or (TOKEN_BUDGET // tokens_per_step)

    output_dir = args.output_dir or os.path.join(
        "outputs", f"baseline_{args.model.replace('-', '_')}"
    )

    train(
        model_name=args.model,
        max_steps=max_steps,
        batch_size=batch_size,
        output_dir=output_dir,
        resume_from=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
