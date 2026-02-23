#!/usr/bin/env python3
"""
Download and save training data locally as parquet files.

Streams from HuggingFace, counts tokens with TinyLlama tokenizer, and stops
once the target token count is reached.  All models (neuromorphic + baselines)
then train from these identical local files — no more HTTP streaming crashes
and guaranteed data parity.

Usage:
    # Download 2B tokens for Phase B (default)
    python scripts/prepare_data.py

    # Custom token budget
    python scripts/prepare_data.py --tokens 3B

    # Validation only (fast)
    python scripts/prepare_data.py --val-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Phase B datasets
FINEWEB_EDU_PATH = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_NAME = "sample-10BT"
DCLM_PATH = "mlfoundations/dclm-baseline-1.0"

# Mix weights (same as Phase B)
FINEWEB_WEIGHT = 0.6
DCLM_WEIGHT = 0.4

# Output directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "phase_B"

# Validation set
VAL_TOKEN_TARGET = 5_000_000  # 5M tokens for validation
VAL_SEED = 1337

# How often to print progress
PROGRESS_INTERVAL = 10_000  # every N examples


def parse_token_count(s: str) -> int:
    """Parse human-friendly token counts like '2B', '500M', '1.5B'."""
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def stream_and_save(
    hf_path: str,
    hf_name: str | None,
    target_tokens: int,
    out_path: Path,
    tokenizer,
    seed: int = 42,
    text_column: str = "text",
    label: str = "",
) -> dict:
    """Stream a HF dataset, count tokens, save as parquet when target reached.

    Returns metadata dict with token/example counts.
    """
    print(f"\n{'='*60}")
    print(f"Downloading {label or hf_path}")
    print(f"  Target: {target_tokens:,} tokens")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    ds = load_dataset(hf_path, hf_name, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    texts = []
    total_tokens = 0
    skipped = 0
    t0 = time.time()

    for i, example in enumerate(ds):
        text = example.get(text_column, "")
        if not text or not text.strip():
            skipped += 1
            continue

        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        texts.append(text)
        total_tokens += n_tok

        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1:>9,} examples] "
                f"{total_tokens:>13,} tokens  "
                f"({total_tokens/1e9:.2f}B)  "
                f"{rate/1e6:.1f}M tok/s  "
                f"skipped={skipped}"
            )

        if total_tokens >= target_tokens:
            break

    elapsed = time.time() - t0
    print(f"\n  Done: {len(texts):,} examples, {total_tokens:,} tokens in {elapsed:.0f}s")

    # Save as parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, out_path, compression="zstd")

    file_size = out_path.stat().st_size
    print(f"  Saved: {out_path} ({file_size / 1e9:.2f} GB)")

    return {
        "examples": len(texts),
        "tokens": total_tokens,
        "skipped": skipped,
        "file_size_bytes": file_size,
        "elapsed_s": round(elapsed, 1),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download training data locally as parquet files"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="2B",
        help="Total token budget (e.g. '2B', '500M'). Split 60/40 between "
             "FineWeb-Edu and DCLM. Default: 2B",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)"
    )
    parser.add_argument(
        "--val-tokens",
        type=str,
        default="5M",
        help="Validation token budget (default: 5M)",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only download validation data (fast)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    total_tokens = parse_token_count(args.tokens)
    val_tokens = parse_token_count(args.val_tokens)
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR

    fineweb_tokens = int(total_tokens * FINEWEB_WEIGHT)
    dclm_tokens = int(total_tokens * DCLM_WEIGHT)

    print(f"Data preparation for Phase B training")
    print(f"  Total budget: {total_tokens:,} tokens ({total_tokens/1e9:.1f}B)")
    print(f"  FineWeb-Edu:  {fineweb_tokens:,} tokens (60%)")
    print(f"  DCLM:         {dclm_tokens:,} tokens (40%)")
    print(f"  Validation:   {val_tokens:,} tokens")
    print(f"  Output dir:   {out_dir}")
    print(f"  Train seed:   {args.seed}")
    print(f"  Val seed:     {VAL_SEED}")

    # Load tokenizer (for counting tokens)
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  Vocab size: {len(tokenizer)}")

    manifest = {
        "created": datetime.now().isoformat(),
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": len(tokenizer),
        "total_token_budget": total_tokens,
        "mix_weights": {"fineweb_edu": FINEWEB_WEIGHT, "dclm": DCLM_WEIGHT},
        "datasets": {},
    }

    t_start = time.time()

    # --- Validation set (from FineWeb-Edu, different seed) ---
    val_meta = stream_and_save(
        hf_path=FINEWEB_EDU_PATH,
        hf_name=FINEWEB_EDU_NAME,
        target_tokens=val_tokens,
        out_path=out_dir / "val_fineweb_edu.parquet",
        tokenizer=tokenizer,
        seed=VAL_SEED,
        label="FineWeb-Edu (validation, seed=1337)",
    )
    manifest["datasets"]["val_fineweb_edu"] = val_meta

    if args.val_only:
        manifest["total_elapsed_s"] = round(time.time() - t_start, 1)
        _save_manifest(out_dir, manifest)
        print("\nValidation-only mode — done.")
        return

    # --- FineWeb-Edu (train) ---
    fineweb_meta = stream_and_save(
        hf_path=FINEWEB_EDU_PATH,
        hf_name=FINEWEB_EDU_NAME,
        target_tokens=fineweb_tokens,
        out_path=out_dir / "fineweb_edu.parquet",
        tokenizer=tokenizer,
        seed=args.seed,
        label="FineWeb-Edu (train)",
    )
    manifest["datasets"]["fineweb_edu"] = fineweb_meta

    # --- DCLM (train) ---
    dclm_meta = stream_and_save(
        hf_path=DCLM_PATH,
        hf_name=None,
        target_tokens=dclm_tokens,
        out_path=out_dir / "dclm.parquet",
        tokenizer=tokenizer,
        seed=args.seed,
        label="DCLM (train)",
    )
    manifest["datasets"]["dclm"] = dclm_meta

    manifest["total_elapsed_s"] = round(time.time() - t_start, 1)
    _save_manifest(out_dir, manifest)

    # --- Summary ---
    total_saved = sum(
        d["tokens"]
        for k, d in manifest["datasets"].items()
        if not k.startswith("val_")
    )
    total_size = sum(d["file_size_bytes"] for d in manifest["datasets"].values())
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Train tokens: {total_saved:,} ({total_saved/1e9:.2f}B)")
    print(f"  Val tokens:   {val_meta['tokens']:,}")
    print(f"  Total size:   {total_size/1e9:.2f} GB")
    print(f"  Time:         {manifest['total_elapsed_s']:.0f}s")
    print(f"  Output:       {out_dir}")
    print(f"{'='*60}")


def _save_manifest(out_dir: Path, manifest: dict):
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
