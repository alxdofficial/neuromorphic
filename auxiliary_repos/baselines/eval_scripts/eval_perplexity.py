"""
Baseline perplexity benchmarks.

Computes token-level perplexity on:
  - WikiText-2 (validation)
  - WikiText-103 (validation)
  - PG19 (first N books from validation, perplexity vs position)

Usage:
    python eval_perplexity.py --model pythia-160m
    python eval_perplexity.py --model mamba-130m
    python eval_perplexity.py --model all
"""

import argparse
import json
import math
import os
import time

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "mamba-130m": "state-spaces/mamba-130m-hf",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ============================================================================
# Perplexity computation
# ============================================================================

def compute_perplexity(
    model,
    tokenizer,
    text: str,
    stride: int = 512,
    max_length: int = 1024,
    device: str = "cuda",
) -> dict:
    """Compute perplexity using sliding window with stride.

    Returns dict with overall ppl and token count.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, prev_end)  # only score new tokens

        input_chunk = input_ids[:, begin:end]
        target_chunk = input_chunk.clone()
        # Mask tokens before target_begin (already scored)
        target_chunk[:, :target_begin - begin] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss

        # Number of tokens actually scored
        n_scored = (target_chunk != -100).sum().item()
        if n_scored > 0:
            nlls.append(neg_log_likelihood.item() * n_scored)
            n_tokens += n_scored

        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens) if n_tokens > 0 else float("inf")
    return {"ppl": ppl, "n_tokens": n_tokens, "total_nll": sum(nlls)}


def compute_perplexity_by_position(
    model,
    tokenizer,
    text: str,
    chunk_size: int = 256,
    max_length: int = 1024,
    device: str = "cuda",
) -> list[dict]:
    """Compute per-chunk perplexity to measure perplexity vs position.

    Returns list of {position, ppl, n_tokens} dicts.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    results = []
    for start in range(0, seq_len - 1, chunk_size):
        end = min(start + chunk_size, seq_len)
        # Use context from max(0, start - max_length + chunk_size) to end
        ctx_start = max(0, end - max_length)
        input_chunk = input_ids[:, ctx_start:end]
        target_chunk = input_chunk.clone()
        # Only score tokens in [start, end)
        target_chunk[:, :start - ctx_start] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)

        n_scored = (target_chunk != -100).sum().item()
        if n_scored > 0:
            ppl = math.exp(outputs.loss.item())
            results.append({
                "position": start,
                "ppl": min(ppl, 1e6),
                "n_tokens": n_scored,
            })

    return results


# ============================================================================
# Dataset loading
# ============================================================================

def load_wikitext2_val() -> str:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    return "\n\n".join(ds["text"])


def load_wikitext103_val() -> str:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    return "\n\n".join(ds["text"])


def load_pg19_val(n_books: int = 10) -> list[str]:
    """Load first N books from PG19 validation split."""
    ds = load_dataset("pg19", split="validation", streaming=True)
    books = []
    for i, example in enumerate(ds):
        if i >= n_books:
            break
        text = example["text"]
        if len(text) > 1000:  # skip very short entries
            books.append(text)
    return books


# ============================================================================
# Main
# ============================================================================

def run_model(model_name: str, repo_id: str):
    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({repo_id})")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    results = {
        "model": model_name,
        "repo_id": repo_id,
        "n_params": n_params,
        "device": str(device),
    }

    # --- WikiText-2 ---
    print("\n--- WikiText-2 (validation) ---")
    t0 = time.time()
    wt2_text = load_wikitext2_val()
    wt2_result = compute_perplexity(model, tokenizer, wt2_text, stride=512, max_length=1024, device=device)
    elapsed = time.time() - t0
    print(f"  PPL: {wt2_result['ppl']:.2f}  ({wt2_result['n_tokens']:,} tokens, {elapsed:.1f}s)")
    results["wikitext2"] = {**wt2_result, "elapsed": elapsed}

    # --- WikiText-103 ---
    print("\n--- WikiText-103 (validation) ---")
    t0 = time.time()
    wt103_text = load_wikitext103_val()
    wt103_result = compute_perplexity(model, tokenizer, wt103_text, stride=512, max_length=1024, device=device)
    elapsed = time.time() - t0
    print(f"  PPL: {wt103_result['ppl']:.2f}  ({wt103_result['n_tokens']:,} tokens, {elapsed:.1f}s)")
    results["wikitext103"] = {**wt103_result, "elapsed": elapsed}

    # --- PG19 (perplexity vs position) ---
    print("\n--- PG19 (validation, 10 books, PPL vs position) ---")
    t0 = time.time()
    pg19_books = load_pg19_val(n_books=10)
    print(f"  Loaded {len(pg19_books)} books")

    all_position_results = []
    book_ppls = []
    for i, book_text in enumerate(pg19_books):
        book_tokens = len(tokenizer.encode(book_text))
        book_result = compute_perplexity(model, tokenizer, book_text, stride=512, max_length=1024, device=device)
        book_ppls.append(book_result["ppl"])
        print(f"  Book {i}: {book_tokens:,} tokens, PPL={book_result['ppl']:.2f}")

        # Position-binned perplexity for this book
        pos_results = compute_perplexity_by_position(
            model, tokenizer, book_text,
            chunk_size=512, max_length=1024, device=device,
        )
        all_position_results.extend(pos_results)

    # Aggregate position results into bins
    position_bins = {}
    for pr in all_position_results:
        pos = pr["position"]
        if pos < 1000:
            bin_name = "0-1K"
        elif pos < 5000:
            bin_name = "1K-5K"
        elif pos < 10000:
            bin_name = "5K-10K"
        elif pos < 50000:
            bin_name = "10K-50K"
        else:
            bin_name = "50K+"

        if bin_name not in position_bins:
            position_bins[bin_name] = {"total_nll": 0.0, "n_tokens": 0}
        position_bins[bin_name]["total_nll"] += math.log(pr["ppl"]) * pr["n_tokens"]
        position_bins[bin_name]["n_tokens"] += pr["n_tokens"]

    pg19_position_ppls = {}
    for bin_name, data in sorted(position_bins.items()):
        if data["n_tokens"] > 0:
            bin_ppl = math.exp(data["total_nll"] / data["n_tokens"])
            pg19_position_ppls[bin_name] = {"ppl": bin_ppl, "n_tokens": data["n_tokens"]}
            print(f"  Position {bin_name}: PPL={bin_ppl:.2f} ({data['n_tokens']:,} tokens)")

    elapsed = time.time() - t0
    results["pg19"] = {
        "mean_ppl": sum(book_ppls) / len(book_ppls) if book_ppls else None,
        "book_ppls": book_ppls,
        "position_ppls": pg19_position_ppls,
        "n_books": len(pg19_books),
        "elapsed": elapsed,
    }
    if results["pg19"]["mean_ppl"]:
        print(f"  Mean PPL across books: {results['pg19']['mean_ppl']:.2f}  ({elapsed:.1f}s)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"perplexity_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline perplexity benchmarks")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        default="all", help="Which model to benchmark")
    args = parser.parse_args()

    if args.model == "all":
        targets = MODELS.items()
    else:
        targets = [(args.model, MODELS[args.model])]

    all_results = {}
    for name, repo_id in targets:
        result = run_model(name, repo_id)
        all_results[name] = result

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<15} {'Params':>10} {'WT-2 PPL':>10} {'WT-103 PPL':>12} {'PG19 PPL':>10}")
        print("-" * 60)
        for name, r in all_results.items():
            pg19_ppl = r['pg19']['mean_ppl'] or 0
            print(f"{name:<15} {r['n_params']/1e6:>8.1f}M "
                  f"{r['wikitext2']['ppl']:>10.2f} "
                  f"{r['wikitext103']['ppl']:>12.2f} "
                  f"{pg19_ppl:>10.2f}")

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "perplexity_comparison.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
