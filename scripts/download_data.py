"""Single entry point: download all datasets needed for trajectory-memory training.

Per plan §4.5, the four waves need:

  Wave 1 (long-doc TF):
    - Books / long stories: deepmind/pg19 (Project Gutenberg books)
    - Code: bigcode/the-stack-dedup (subset)
    - ArXiv: EleutherAI/proof-pile-2 (arXiv subset) — optional
    - Web: HuggingFaceFW/fineweb-edu (filtered for length)
    - Synthetic needle-in-haystack: generated from above bodies (no download)

  Wave 2 (long-chat TF):
    - HuggingFaceH4/ultrachat_200k
    - allenai/WildChat-1M
    - (LongAlpaca / LongInstruct / AgentInstruct: optional, may need separate auth)

  Wave 3 (verifiable-reward GRPO):
    - openai/gsm8k
    - AI-MO/NuminaMath-TIR (math)
    - openai_humaneval (code, via `openai_humaneval` config)
    - deepmind/narrativeqa (long-context QA)

  Wave 4 (long-session GRPO):
    - allenai/WildChat-1M (already downloaded for W2; re-used)
    - microsoft/orca-agentinstruct-1M-v1 (optional)

This script uses HuggingFace `datasets` to ensure the data is cached in
`~/.cache/huggingface`. For very large datasets (FineWeb at 27GB, TheStack),
it streams a small validation slice by default. Use `--full` to force the
full download.

Usage:
    python scripts/download_data.py                       # check + minimal slices
    python scripts/download_data.py --full --datasets w1  # full Wave 1 download
    python scripts/download_data.py --datasets gsm8k narrativeqa  # specific only
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from datasets import load_dataset


# Each entry: dataset_id, optional config name, default split, comment.
_DATASETS = {
    # ── Wave 1 (long doc) ─────────────────────────────────────────────
    "fineweb-edu": {
        "id": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",      # 10B-token sample of fineweb-edu
        "split": "train",
        "wave": "w1",
        "size": "27GB cached fully; streamable",
    },
    "pg19": {
        "id": "deepmind/pg19",
        "config": None,
        "split": "train",
        "wave": "w1",
        "size": "~12GB; long Project Gutenberg books",
    },
    "the-stack-dedup": {
        "id": "bigcode/the-stack-dedup",
        "config": "data/python",       # python-only is small enough
        "split": "train",
        "wave": "w1",
        "size": "~12GB python-only slice; streamable",
    },
    # ── Wave 2 (long chat) ───────────────────────────────────────────
    "ultrachat-200k": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "config": None,
        "split": "train_sft",
        "wave": "w2",
        "size": "~3GB",
    },
    "wildchat-1m": {
        "id": "allenai/WildChat-1M",
        "config": None,
        "split": "train",
        "wave": "w2,w4",
        "size": "~5GB; multi-turn assistant logs",
    },
    # ── Wave 3 (verifiable reward) ───────────────────────────────────
    "gsm8k": {
        "id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "wave": "w3",
        "size": "~10MB; tiny",
    },
    "narrativeqa": {
        "id": "deepmind/narrativeqa",
        "config": None,
        "split": "train",
        "wave": "w3",
        "size": "~200MB",
    },
    "humaneval": {
        "id": "openai_humaneval",
        "config": None,
        "split": "test",                # HumanEval has no train split
        "wave": "w3",
        "size": "~40KB; evaluation only",
    },
    "numinamath": {
        "id": "AI-MO/NuminaMath-TIR",
        "config": None,
        "split": "train",
        "wave": "w3",
        "size": "~100MB",
    },
    # Wave 4 also uses wildchat-1m (above)
}


def list_datasets():
    print("Datasets registered for trajectory-memory training:\n")
    for name, info in _DATASETS.items():
        print(f"  {name:24s}  {info['wave']:8s}  {info['id']}")
        print(f"  {'':24s}  {'':8s}  size: {info['size']}")
    print()


def download_one(name: str, *, slice_n: int | None = None, full: bool = False):
    """Download (or stream-cache) one dataset. With `slice_n`, only fetch
    that many examples — useful for verification."""
    if name not in _DATASETS:
        print(f"  [skip] unknown dataset: {name}")
        return None
    info = _DATASETS[name]
    print(f"  [download] {name} ({info['id']}) — {info['size']}")
    kwargs = {"path": info["id"], "split": info["split"]}
    if info["config"]:
        kwargs["name"] = info["config"]

    if not full and slice_n:
        # Stream-load N examples to populate cache without full download.
        kwargs["streaming"] = True
        ds = load_dataset(**kwargs)
        examples = []
        for i, ex in enumerate(ds):
            if i >= slice_n:
                break
            examples.append(ex)
        print(f"  [ok]       streamed {len(examples)} examples for verification")
        return examples
    else:
        ds = load_dataset(**kwargs)
        n_examples = len(ds) if hasattr(ds, "__len__") else "streaming"
        print(f"  [ok]       fully loaded ({n_examples} examples)")
        return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(_DATASETS.keys()),
                    help="dataset names or wave tags (w1/w2/w3/w4)")
    ap.add_argument("--slice", type=int, default=200,
                    help="examples per dataset for verification (default 200)")
    ap.add_argument("--full", action="store_true",
                    help="full download (huge for fineweb / wildchat)")
    ap.add_argument("--list", action="store_true",
                    help="list datasets and exit")
    args = ap.parse_args()

    if args.list:
        list_datasets()
        return

    # Expand wave tags to dataset names.
    targets = []
    for d in args.datasets:
        if d.lower() in ("w1", "w2", "w3", "w4"):
            targets.extend(
                name for name, info in _DATASETS.items()
                if d.lower() in info["wave"]
            )
        elif d in _DATASETS:
            targets.append(d)
        else:
            print(f"  [warn] unknown dataset/wave: {d}")
    targets = list(dict.fromkeys(targets))   # dedup, preserve order

    print(f"Downloading {len(targets)} dataset(s):  {', '.join(targets)}")
    print(f"Mode: {'full' if args.full else f'verification (slice_n={args.slice})'}")
    print()

    failed = []
    for name in targets:
        try:
            download_one(name, slice_n=args.slice, full=args.full)
        except Exception as e:
            print(f"  [fail]     {name}: {e}")
            failed.append(name)
        print()

    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
