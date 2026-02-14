"""
Comprehensive comparison: Neuromorphic LM vs Pythia vs Mamba baselines.

Metrics:
  1. Loss vs FLOPs / Loss vs Tokens (from training logs)
  2. Inference throughput & memory scaling
  3. PG19 perplexity vs document position
  4. WikiText-2 perplexity and bits-per-byte

Usage:
    python auxiliary_repos/baselines/eval_scripts/run_comparison.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]  # neuromorphic/
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoints
OUR_CKPT = ROOT / "outputs/phase_d/20260212_214745/checkpoints/neuromorphic_a_D_20260212_214745_step5000.pt"
PYTHIA_CKPT_DIR = ROOT / "outputs/baseline_pythia_160m/checkpoints"
MAMBA_CKPT_DIR = ROOT / "outputs/baseline_mamba_130m/checkpoints"

# Training logs
OUR_METRICS = [
    ROOT / "outputs/phase_a_to_d/20260212_114315/metrics.jsonl",
    ROOT / "outputs/phase_b/20260212_160136/metrics.jsonl",
    ROOT / "outputs/phase_d/20260212_214745/metrics.jsonl",
]
PYTHIA_METRICS = ROOT / "outputs/baseline_pythia_160m/metrics.jsonl"
MAMBA_METRICS = ROOT / "outputs/baseline_mamba_130m/metrics.jsonl"

OUTPUT_DIR = ROOT / "outputs/comparison"

OUR_PARAMS = 56.1e6
PYTHIA_PARAMS = 134.2e6
MAMBA_PARAMS = 115.1e6


# ===========================================================================
# Model loaders
# ===========================================================================

def load_our_model():
    """Load neuromorphic model from checkpoint."""
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    config = ModelConfig.tier_a()
    config.set_phase("C")
    model = NeuromorphicLM(config).to(DEVICE)
    ckpt = torch.load(str(OUR_CKPT), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.requires_grad_(False)
    return model


def load_pythia(ckpt_dir=None):
    """Load from-scratch Pythia from training checkpoint."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    ckpt_dir = ckpt_dir or PYTHIA_CKPT_DIR
    config = GPTNeoXConfig(
        vocab_size=32000, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=2048, rotary_pct=0.25,
        use_parallel_residual=True, layer_norm_eps=1e-5,
    )
    model = GPTNeoXForCausalLM(config).to(DEVICE)

    ckpt_path = Path(ckpt_dir) / "step_15000.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(ckpt_dir) / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"    Loaded weights from {ckpt_path.name}")
    model.requires_grad_(False)
    return model


def load_mamba(ckpt_dir=None):
    """Load from-scratch Mamba from training checkpoint."""
    from transformers import MambaConfig, MambaForCausalLM

    ckpt_dir = ckpt_dir or MAMBA_CKPT_DIR
    config = MambaConfig(
        vocab_size=32000, hidden_size=768, num_hidden_layers=24,
        state_size=16, expand=2, conv_kernel=4,
        use_bias=False, use_conv_bias=True,
    )
    model = MambaForCausalLM(config).to(DEVICE)

    ckpt_path = Path(ckpt_dir) / "step_15000.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(ckpt_dir) / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"    Loaded weights from {ckpt_path.name}")
    model.requires_grad_(False)
    return model


# ===========================================================================
# 1. Loss vs FLOPs / Loss vs Tokens (from training logs)
# ===========================================================================

def compute_training_curves():
    """Extract loss vs tokens and loss vs FLOPs from training logs."""
    print("\n" + "=" * 60)
    print("1. TRAINING EFFICIENCY (Loss vs Tokens / FLOPs)")
    print("=" * 60)

    results = {}

    # --- Our model (multiple phases, concatenated) ---
    our_records = []
    cumulative_tokens = 0
    for log_path in OUR_METRICS:
        if not log_path.exists():
            print(f"  SKIP: {log_path} not found")
            continue
        with open(log_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("mode") != "train":
                    continue
                loss = d.get("loss")
                if loss is None or math.isnan(loss):
                    continue
                vt = d.get("valid_tokens", 8192)
                cumulative_tokens += vt
                our_records.append({
                    "tokens": cumulative_tokens,
                    "loss": loss,
                    "step": d.get("step", 0),
                    "phase": d.get("phase", "?"),
                })

    results["neuromorphic"] = {
        "params": OUR_PARAMS,
        "records": our_records,
        "final_loss": our_records[-1]["loss"] if our_records else None,
        "total_tokens": our_records[-1]["tokens"] if our_records else 0,
    }

    # --- Baselines ---
    for name, path, params in [
        ("pythia", PYTHIA_METRICS, PYTHIA_PARAMS),
        ("mamba", MAMBA_METRICS, MAMBA_PARAMS),
    ]:
        records = []
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                loss = d.get("loss")
                if loss is None:
                    continue
                tokens = d.get("tokens_seen", 0)
                records.append({"tokens": tokens, "loss": loss, "step": d["step"]})
        results[name] = {
            "params": params,
            "records": records,
            "final_loss": records[-1]["loss"] if records else None,
            "total_tokens": records[-1]["tokens"] if records else 0,
        }

    # Print summary
    print(f"\n{'Model':<25} {'Params':>10} {'Tokens':>12} {'FLOPs':>12} {'Final Loss':>12} {'PPL':>8}")
    print("-" * 85)
    for name, data in results.items():
        tokens = data["total_tokens"]
        params = data["params"]
        flops = 6 * params * tokens
        loss = data["final_loss"]
        ppl = math.exp(loss) if loss else float("nan")
        print(
            f"{name:<25} {params/1e6:>8.1f}M {tokens/1e6:>10.1f}M "
            f"{flops/1e15:>10.2f}PF {loss:>12.3f} {ppl:>8.0f}"
        )

    # Save curves as JSON
    curves_path = OUTPUT_DIR / "training_curves.json"
    serializable = {}
    for name, data in results.items():
        recs = data["records"]
        sampled = [r for i, r in enumerate(recs) if i % 10 == 0 or i == len(recs) - 1]
        serializable[name] = {
            "params": data["params"],
            "tokens": [r["tokens"] for r in sampled],
            "loss": [r["loss"] for r in sampled],
            "flops": [6 * data["params"] * r["tokens"] for r in sampled],
        }
    with open(curves_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Curves saved to {curves_path}")

    return results


# ===========================================================================
# 2. Inference throughput & memory scaling
# ===========================================================================

def benchmark_inference():
    """Measure tokens/s and GPU memory at various context lengths."""
    print("\n" + "=" * 60)
    print("2. INFERENCE BENCHMARK (Throughput & Memory Scaling)")
    print("=" * 60)

    context_lengths = [256, 512, 1024, 2048, 4096]
    results = {}

    # --- Our model ---
    print("\n  Loading neuromorphic model...")
    our_model = load_our_model()

    our_results = []
    for ctx_len in context_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tokens = torch.randint(0, 32000, (ctx_len,), device=DEVICE)
        reset = torch.tensor([True], dtype=torch.bool, device=DEVICE)
        no_reset = torch.tensor([False], dtype=torch.bool, device=DEVICE)

        # Warmup
        with torch.no_grad():
            for t in range(min(32, ctx_len)):
                our_model.forward_one_token(tokens[t:t+1], reset if t == 0 else no_reset)

        # Reset state and measure
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            our_model.forward_one_token(tokens[0:1], reset)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for t in range(1, ctx_len):
                our_model.forward_one_token(tokens[t:t+1], no_reset)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        tok_s = (ctx_len - 1) / elapsed if elapsed > 0 else 0

        our_results.append({
            "ctx_len": ctx_len, "tok_s": round(tok_s),
            "peak_mem_gb": round(peak_mem, 3), "elapsed": round(elapsed, 4),
        })
        print(f"    Neuromorphic ctx={ctx_len:>5}: {tok_s:>8.0f} tok/s, {peak_mem:.3f} GB")

    results["neuromorphic"] = our_results
    del our_model
    torch.cuda.empty_cache()

    # --- Baselines ---
    # Mamba needs bf16 (fp16 causes NaN with selective scan)
    for model_name, loader_fn in [("pythia", load_pythia), ("mamba", load_mamba)]:
        amp_dtype = torch.bfloat16 if "mamba" in model_name else torch.float16
        print(f"\n  Loading {model_name} baseline...")
        try:
            model = loader_fn()
        except Exception as e:
            print(f"    SKIP {model_name}: {e}")
            continue

        model_results = []
        for ctx_len in context_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            tokens = torch.randint(0, 32000, (1, ctx_len), device=DEVICE)

            # Warmup
            try:
                with torch.no_grad():
                    _ = model(tokens[:, :min(256, ctx_len)])
            except Exception:
                pass

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(tokens)
                torch.cuda.synchronize()
                elapsed = time.time() - t0

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                tok_s = ctx_len / elapsed if elapsed > 0 else 0

                model_results.append({
                    "ctx_len": ctx_len, "tok_s": round(tok_s),
                    "peak_mem_gb": round(peak_mem, 3), "elapsed": round(elapsed, 4),
                })
                print(f"    {model_name} ctx={ctx_len:>5}: {tok_s:>8.0f} tok/s, {peak_mem:.3f} GB")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                model_results.append({
                    "ctx_len": ctx_len, "tok_s": 0,
                    "peak_mem_gb": -1, "elapsed": -1,
                })
                print(f"    {model_name} ctx={ctx_len:>5}: OOM")

        results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    with open(OUTPUT_DIR / "inference_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'inference_benchmark.json'}")

    return results


# ===========================================================================
# 3. PG19 perplexity vs document position
# ===========================================================================

def measure_pg19_position_ppl():
    """Compute perplexity binned by position within long documents."""
    print("\n" + "=" * 60)
    print("3. PG19 PERPLEXITY VS DOCUMENT POSITION")
    print("=" * 60)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load PG19 test set
    print("  Loading PG19 test set (first 10 books)...")
    try:
        ds = load_dataset("deepmind/pg19", split="test", streaming=True)
    except Exception as e:
        print(f"  ERROR loading PG19: {e}")
        return {}

    # Position bins (in tokens)
    bins = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    bin_labels = ["0-256", "256-512", "512-1K", "1K-2K", "2K-4K"]

    results = {}

    # Tokenize documents
    docs = []
    for i, example in enumerate(ds):
        if i >= 10:
            break
        text = example.get("text", "")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= 4096:
            docs.append(tokens[:4096])
    print(f"  {len(docs)} documents with >= 4096 tokens")

    if not docs:
        print("  No documents long enough, skipping")
        return {}

    # --- Our model ---
    print("\n  Measuring neuromorphic model...")
    our_model = load_our_model()

    our_bin_losses = {label: [] for label in bin_labels}
    no_reset = torch.tensor([False], dtype=torch.bool, device=DEVICE)
    reset = torch.tensor([True], dtype=torch.bool, device=DEVICE)

    for doc_idx, tokens in enumerate(docs):
        token_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            our_model.forward_one_token(token_t[0:1], reset)

            for t in range(1, len(tokens)):
                logits, _, _ = our_model.forward_one_token(token_t[t:t+1], no_reset)
                loss = F.cross_entropy(logits, token_t[t:t+1]).item()

                for (lo, hi), label in zip(bins, bin_labels):
                    if lo <= t < hi:
                        our_bin_losses[label].append(loss)
                        break

        if (doc_idx + 1) % 5 == 0:
            print(f"    Doc {doc_idx + 1}/{len(docs)} done")

    results["neuromorphic"] = {}
    for label in bin_labels:
        losses = our_bin_losses[label]
        if losses:
            avg = sum(losses) / len(losses)
            results["neuromorphic"][label] = {
                "avg_loss": round(avg, 4),
                "ppl": round(math.exp(min(avg, 20)), 1),
                "n_tokens": len(losses),
            }

    del our_model
    torch.cuda.empty_cache()

    # --- Baselines ---
    for model_name, loader_fn in [("pythia", load_pythia), ("mamba", load_mamba)]:
        amp_dtype = torch.bfloat16 if "mamba" in model_name else torch.float16
        print(f"\n  Measuring {model_name} baseline...")
        try:
            model = loader_fn()
        except Exception as e:
            print(f"    SKIP {model_name}: {e}")
            continue

        bin_losses = {label: [] for label in bin_labels}

        for doc_idx, tokens in enumerate(docs):
            token_t = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

            try:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(token_t)
                    logits = outputs.logits[0].float()

                    for t in range(1, len(tokens)):
                        loss = F.cross_entropy(
                            logits[t - 1:t], token_t[0, t:t+1]
                        ).item()

                        for (lo, hi), label in zip(bins, bin_labels):
                            if lo <= t < hi:
                                bin_losses[label].append(loss)
                                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    OOM on doc {doc_idx} (len={len(tokens)}), skipping")
                continue

            if (doc_idx + 1) % 5 == 0:
                print(f"    Doc {doc_idx + 1}/{len(docs)} done")

        results[model_name] = {}
        for label in bin_labels:
            losses = bin_losses[label]
            if losses:
                avg = sum(losses) / len(losses)
                results[model_name][label] = {
                    "avg_loss": round(avg, 4),
                    "ppl": round(math.exp(min(avg, 20)), 1),
                    "n_tokens": len(losses),
                }

        del model
        torch.cuda.empty_cache()

    # Print table
    print(f"\n  {'Position':<12}", end="")
    for name in results:
        print(f" {name:>20}", end="")
    print()
    print("  " + "-" * (12 + 20 * len(results)))
    for label in bin_labels:
        print(f"  {label:<12}", end="")
        for name in results:
            if label in results[name]:
                ppl = results[name][label]["ppl"]
                print(f" {ppl:>20.1f}", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()

    with open(OUTPUT_DIR / "pg19_position_ppl.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'pg19_position_ppl.json'}")

    return results


# ===========================================================================
# 4. WikiText-2 perplexity (standard LM benchmark)
# ===========================================================================

def measure_wikitext2_ppl():
    """Standard WikiText-2 perplexity for all models."""
    print("\n" + "=" * 60)
    print("4. WIKITEXT-2 PERPLEXITY & BITS-PER-BYTE")
    print("=" * 60)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("  Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    max_tokens = min(len(all_tokens), 50000)
    tokens_list = all_tokens[:max_tokens]
    n_bytes = len(full_text[:max_tokens * 5].encode("utf-8"))  # approximate
    print(f"  Tokens: {len(tokens_list):,}, approx bytes: {n_bytes:,}")

    results = {}

    # --- Our model (sequential, no sliding window needed) ---
    print("\n  Measuring neuromorphic model...")
    our_model = load_our_model()

    total_loss = 0.0
    n_scored = 0
    token_t = torch.tensor(tokens_list, dtype=torch.long, device=DEVICE)
    no_reset = torch.tensor([False], dtype=torch.bool, device=DEVICE)
    reset_mask = torch.tensor([True], dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        our_model.forward_one_token(token_t[0:1], reset_mask)
        for t in range(1, len(tokens_list)):
            logits, _, _ = our_model.forward_one_token(token_t[t:t+1], no_reset)
            loss = F.cross_entropy(logits, token_t[t:t+1]).item()
            if not math.isnan(loss):
                total_loss += loss
                n_scored += 1

            if (t + 1) % 10000 == 0:
                avg = total_loss / n_scored
                print(f"    {t+1}/{len(tokens_list)} tokens, running ppl={math.exp(avg):.1f}")

    avg_loss = total_loss / n_scored
    ppl = math.exp(avg_loss)
    bpb = avg_loss / math.log(2)
    results["neuromorphic"] = {
        "ppl": round(ppl, 2), "loss": round(avg_loss, 4),
        "bpb": round(bpb, 4), "n_tokens": n_scored,
    }
    print(f"    Neuromorphic: PPL={ppl:.2f}, loss={avg_loss:.4f}, BPB={bpb:.4f}")

    del our_model
    torch.cuda.empty_cache()

    # --- Baselines (sliding window) ---
    stride = 256
    max_length = 1024

    for model_name, loader_fn in [("pythia", load_pythia), ("mamba", load_mamba)]:
        amp_dtype = torch.bfloat16 if "mamba" in model_name else torch.float16
        print(f"\n  Measuring {model_name} baseline...")
        try:
            model = loader_fn()
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        total_loss = 0.0
        n_scored = 0
        token_t = torch.tensor([tokens_list], dtype=torch.long, device=DEVICE)

        for start in range(0, len(tokens_list) - 1, stride):
            end = min(start + max_length, len(tokens_list))
            chunk = token_t[:, start:end]
            target_start = max(start, stride) - start if start > 0 else 1

            try:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(chunk)
                    logits = outputs.logits[0].float()

                    for t in range(max(1, target_start), chunk.shape[1]):
                        loss = F.cross_entropy(
                            logits[t-1:t], chunk[0, t:t+1]
                        ).item()
                        if not math.isnan(loss):
                            total_loss += loss
                            n_scored += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        if n_scored > 0:
            avg_loss = total_loss / n_scored
            ppl = math.exp(avg_loss)
            bpb = avg_loss / math.log(2)
            results[model_name] = {
                "ppl": round(ppl, 2), "loss": round(avg_loss, 4),
                "bpb": round(bpb, 4), "n_tokens": n_scored,
            }
            print(f"    {model_name}: PPL={ppl:.2f}, loss={avg_loss:.4f}, BPB={bpb:.4f}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n  {'Model':<25} {'PPL':>10} {'Loss':>10} {'BPB':>10}")
    print("  " + "-" * 55)
    for name, data in results.items():
        print(f"  {name:<25} {data['ppl']:>10.2f} {data['loss']:>10.4f} {data['bpb']:>10.4f}")

    with open(OUTPUT_DIR / "wikitext2_ppl.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'wikitext2_ppl.json'}")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    compute_training_curves()
    benchmark_inference()
    measure_pg19_position_ppl()
    measure_wikitext2_ppl()

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
