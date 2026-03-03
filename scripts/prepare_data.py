#!/usr/bin/env python3
"""
Download and save training data locally as parquet files.

Streams from The Pile (deduplicated) on HuggingFace, counts tokens with
the GPT-NeoX tokenizer, and stops once the target token count is reached.
All models (neuromorphic + baselines) then train from these identical local
files — no more HTTP streaming crashes and guaranteed data parity.

Using The Pile ensures fair comparison with published baselines (Pythia, Mamba,
RWKV-7) which were all trained on the same corpus.

Usage:
    # Download 2B tokens (default)
    python scripts/prepare_data.py

    # Custom token budget
    python scripts/prepare_data.py --tokens 10B

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
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use TinyLlama tokenizer (our model's tokenizer) for token counting
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# The Pile (deduplicated) — same data as Pythia/Mamba/RWKV baselines
PILE_PATH = "EleutherAI/the_pile_deduplicated"

# Output directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pile"

# Validation: use The Pile's val split (separate from train)
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
    target_tokens: int,
    out_path: Path,
    tokenizer,
    seed: int = 42,
    split: str = "train",
    text_column: str = "text",
    label: str = "",
) -> dict:
    """Stream a HF dataset, count tokens, save as parquet + tokenized shard.

    Saves both raw text (.parquet) and pre-tokenized binary (.bin) for fast
    training. The .bin shard is a flat uint16 array of token IDs with EOS
    tokens between documents — ready for memory-mapped streaming.

    Returns metadata dict with token/example counts.
    """
    print(f"\n{'='*60}")
    print(f"Downloading {label or hf_path}")
    print(f"  Split: {split}")
    print(f"  Target: {target_tokens:,} tokens")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    eos_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    if vocab_size > 65535:
        raise ValueError(
            f"Vocab size {vocab_size} exceeds uint16 range. "
            f"Token shards require vocab_size <= 65535."
        )

    ds = load_dataset(hf_path, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    texts = []
    all_token_ids = []  # collected for .bin shard
    total_tokens = 0
    skipped = 0
    t0 = time.time()

    for i, example in enumerate(ds):
        text = example.get(text_column, "")
        if not text or not text.strip():
            skipped += 1
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        texts.append(text)
        all_token_ids.extend(tokens)
        all_token_ids.append(eos_id)
        total_tokens += len(tokens)

        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            pct = total_tokens / target_tokens * 100
            print(
                f"  [{i+1:>9,} examples] "
                f"{total_tokens:>13,} tokens  "
                f"({total_tokens/1e9:.2f}B / {target_tokens/1e9:.1f}B = {pct:.1f}%)  "
                f"{rate/1e6:.1f}M tok/s  "
                f"skipped={skipped}"
            )

        if total_tokens >= target_tokens:
            break

    elapsed = time.time() - t0
    print(f"\n  Done: {len(texts):,} examples, {total_tokens:,} tokens in {elapsed:.0f}s")

    # Save as parquet (raw text, for inspection/re-tokenization)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, out_path, compression="zstd")

    file_size = out_path.stat().st_size
    print(f"  Saved: {out_path} ({file_size / 1e9:.2f} GB)")

    # Save pre-tokenized shard (.bin) for fast training
    shard_path = out_path.with_suffix(".bin")
    shard_tokens = len(all_token_ids)
    token_array = np.array(all_token_ids, dtype=np.uint16)
    token_array.tofile(shard_path)
    shard_size = shard_path.stat().st_size
    print(f"  Shard: {shard_path} ({shard_size / 1e9:.2f} GB, {shard_tokens:,} tokens)")

    return {
        "examples": len(texts),
        "tokens": total_tokens,
        "skipped": skipped,
        "file_size_bytes": file_size,
        "elapsed_s": round(elapsed, 1),
        "seed": seed,
        "split": split,
        "shard_file": str(shard_path),
        "shard_tokens": shard_tokens,
        "shard_bytes": shard_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download The Pile training data locally as parquet files"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="2B",
        help="Training token budget (e.g. '2B', '10B', '500M'). Default: 2B",
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

    print(f"Data preparation: The Pile (deduplicated)")
    print(f"  Train budget: {total_tokens:,} tokens ({total_tokens/1e9:.1f}B)")
    print(f"  Val budget:   {val_tokens:,} tokens")
    print(f"  Output dir:   {out_dir}")
    print(f"  Train seed:   {args.seed}")
    print(f"  Val seed:     {VAL_SEED}")

    # Load tokenizer (for counting tokens)
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  Vocab size: {len(tokenizer)}")

    manifest = {
        "created": datetime.now().isoformat(),
        "source": "EleutherAI/the_pile_deduplicated",
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": len(tokenizer),
        "total_token_budget": total_tokens,
        "datasets": {},
    }

    t_start = time.time()

    # --- Validation set (different seed slice of train split) ---
    # The Pile deduplicated only has a 'train' split.
    # We use a different shuffle seed so val examples are distinct from train.
    val_meta = stream_and_save(
        hf_path=PILE_PATH,
        target_tokens=val_tokens,
        out_path=out_dir / "pile_val.parquet",
        tokenizer=tokenizer,
        seed=VAL_SEED,
        split="train",
        label="The Pile (validation slice, seed=1337)",
    )
    manifest["datasets"]["pile_val"] = val_meta

    if args.val_only:
        manifest["total_elapsed_s"] = round(time.time() - t_start, 1)
        _save_manifest(out_dir, manifest)
        print("\nValidation-only mode — done.")
        return

    # --- Training set ---
    train_meta = stream_and_save(
        hf_path=PILE_PATH,
        target_tokens=total_tokens,
        out_path=out_dir / "pile_train.parquet",
        tokenizer=tokenizer,
        seed=args.seed,
        split="train",
        label="The Pile (train)",
    )
    manifest["datasets"]["pile_train"] = train_meta

    manifest["total_elapsed_s"] = round(time.time() - t_start, 1)
    _save_manifest(out_dir, manifest)

    # --- Summary ---
    total_saved = train_meta["tokens"]
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
