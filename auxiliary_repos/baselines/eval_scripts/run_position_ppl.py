"""
Long-document perplexity binned by position.

Tests whether memory systems help with later parts of long documents.
Uses Project Gutenberg books as a PG19 replacement.

Usage:
    python auxiliary_repos/baselines/eval_scripts/run_position_ppl.py
"""

import json
import math
import os
import sys
import time
from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda"
BOOKS_DIR = ROOT / "outputs/comparison/pg_books"
OUTPUT_DIR = ROOT / "outputs/comparison"

# Position bins
BINS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
BIN_LABELS = ["0-256", "256-512", "512-1K", "1K-2K", "2K-4K"]

OUR_CKPT = ROOT / "outputs/phase_d/20260212_214745/checkpoints/neuromorphic_a_D_20260212_214745_step5000.pt"
PYTHIA_CKPT_DIR = ROOT / "outputs/baseline_pythia_160m/checkpoints"
MAMBA_CKPT_DIR = ROOT / "outputs/baseline_mamba_130m/checkpoints"


def load_books(tokenizer, max_tokens=4096, max_books=5):
    """Load and tokenize Gutenberg books, keeping first max_tokens tokens."""
    docs = []
    for path in sorted(glob(str(BOOKS_DIR / "*.txt"))):
        with open(path) as f:
            text = f.read()
        # Strip Gutenberg header/footer (rough heuristic)
        start = text.find("*** START OF")
        if start > 0:
            text = text[text.find("\n", start) + 1:]
        end = text.find("*** END OF")
        if end > 0:
            text = text[:end]

        tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
        if len(tokens) >= max_tokens:
            docs.append((Path(path).stem, tokens[:max_tokens]))
            if len(docs) >= max_books:
                break
    return docs


def measure_our_model(docs):
    """Per-position perplexity for neuromorphic model."""
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    config = ModelConfig.tier_a()
    config.set_phase("C")
    model = NeuromorphicLM(config).to(DEVICE)
    ckpt = torch.load(str(OUR_CKPT), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.requires_grad_(False)

    bin_losses = {label: [] for label in BIN_LABELS}
    no_reset = torch.tensor([False], dtype=torch.bool, device=DEVICE)
    reset = torch.tensor([True], dtype=torch.bool, device=DEVICE)

    for name, tokens in docs:
        token_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            model.forward_one_token(token_t[0:1], reset)

            for t in range(1, len(tokens)):
                logits, _, _ = model.forward_one_token(token_t[t:t+1], no_reset)
                loss = F.cross_entropy(logits, token_t[t:t+1]).item()
                if not math.isnan(loss):
                    for (lo, hi), label in zip(BINS, BIN_LABELS):
                        if lo <= t < hi:
                            bin_losses[label].append(loss)
                            break

        print(f"    {name} done ({len(tokens)} tokens)")

    del model
    torch.cuda.empty_cache()
    return bin_losses


def measure_baseline(docs, model_name, loader_fn):
    """Per-position perplexity for a HuggingFace baseline (full-context forward)."""
    model = loader_fn()

    # Mamba needs bf16 (fp16 causes NaN with selective scan)
    amp_dtype = torch.bfloat16 if "mamba" in model_name else torch.float16

    bin_losses = {label: [] for label in BIN_LABELS}

    for name, tokens in docs:
        token_t = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(token_t)
                logits = outputs.logits[0].float()

                for t in range(1, len(tokens)):
                    loss = F.cross_entropy(logits[t-1:t], token_t[0, t:t+1]).item()
                    if not math.isnan(loss):
                        for (lo, hi), label in zip(BINS, BIN_LABELS):
                            if lo <= t < hi:
                                bin_losses[label].append(loss)
                                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    OOM on {name}, skipping")
            continue

        print(f"    {name} done ({len(tokens)} tokens)")

    del model
    torch.cuda.empty_cache()
    return bin_losses


def load_pythia():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    config = GPTNeoXConfig(
        vocab_size=32000, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=2048, rotary_pct=0.25,
        use_parallel_residual=True, layer_norm_eps=1e-5,
    )
    model = GPTNeoXForCausalLM(config).to(DEVICE)
    ckpt_path = PYTHIA_CKPT_DIR / "step_15000.pt"
    if not ckpt_path.exists():
        ckpt_path = PYTHIA_CKPT_DIR / "latest.pt"
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.requires_grad_(False)
    return model


def load_mamba():
    from transformers import MambaConfig, MambaForCausalLM
    config = MambaConfig(
        vocab_size=32000, hidden_size=768, num_hidden_layers=24,
        state_size=16, expand=2, conv_kernel=4,
        use_bias=False, use_conv_bias=True,
    )
    model = MambaForCausalLM(config).to(DEVICE)
    ckpt_path = MAMBA_CKPT_DIR / "step_15000.pt"
    if not ckpt_path.exists():
        ckpt_path = MAMBA_CKPT_DIR / "latest.pt"
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.requires_grad_(False)
    return model


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("=" * 60)
    print("LONG-DOCUMENT PERPLEXITY VS POSITION")
    print("=" * 60)

    docs = load_books(tokenizer, max_tokens=4096, max_books=5)
    print(f"Loaded {len(docs)} books: {[n for n, _ in docs]}")

    results = {}

    # Our model
    print("\nNeuromorphic model:")
    our_losses = measure_our_model(docs)
    results["neuromorphic"] = {}
    for label in BIN_LABELS:
        losses = our_losses[label]
        if losses:
            avg = sum(losses) / len(losses)
            results["neuromorphic"][label] = {
                "avg_loss": round(avg, 4),
                "ppl": round(math.exp(min(avg, 20)), 1),
                "n_tokens": len(losses),
            }

    # Pythia — note: max_position_embeddings=2048, so 2K-4K bin may extrapolate
    print("\nPythia baseline:")
    pythia_losses = measure_baseline(docs, "pythia", load_pythia)
    results["pythia"] = {}
    for label in BIN_LABELS:
        losses = pythia_losses[label]
        if losses:
            avg = sum(losses) / len(losses)
            results["pythia"][label] = {
                "avg_loss": round(avg, 4),
                "ppl": round(math.exp(min(avg, 20)), 1),
                "n_tokens": len(losses),
            }

    # Mamba
    print("\nMamba baseline:")
    mamba_losses = measure_baseline(docs, "mamba", load_mamba)
    results["mamba"] = {}
    for label in BIN_LABELS:
        losses = mamba_losses[label]
        if losses:
            avg = sum(losses) / len(losses)
            results["mamba"][label] = {
                "avg_loss": round(avg, 4),
                "ppl": round(math.exp(min(avg, 20)), 1),
                "n_tokens": len(losses),
            }

    # Print table
    print(f"\n{'Position':<12}", end="")
    for name in results:
        print(f" {name + ' (PPL)':>20}", end="")
    print()
    print("-" * (12 + 20 * len(results)))
    for label in BIN_LABELS:
        print(f"{label:<12}", end="")
        for name in results:
            if label in results[name]:
                ppl = results[name][label]["ppl"]
                print(f" {ppl:>20.1f}", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()

    # Also show delta from first bin (does PPL improve or degrade with position?)
    print(f"\n{'Position':<12}", end="")
    for name in results:
        print(f" {name + ' (delta)':>20}", end="")
    print()
    print("-" * (12 + 20 * len(results)))
    for label in BIN_LABELS:
        print(f"{label:<12}", end="")
        for name in results:
            first_label = BIN_LABELS[0]
            if label in results[name] and first_label in results[name]:
                delta = results[name][label]["avg_loss"] - results[name][first_label]["avg_loss"]
                direction = "better" if delta < -0.01 else ("worse" if delta > 0.01 else "same")
                print(f" {delta:>+12.3f} ({direction})", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR / "position_ppl.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'position_ppl.json'}")


if __name__ == "__main__":
    main()
