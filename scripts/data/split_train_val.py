"""Split a preprocessed parquet into train + val parquets.

Used to carve out held-out validation slices for trainers' periodic
val-loss computation.

By default does row-wise split (fine for W1 / single-row-per-example
data). For W2 / W4 chat (TurnPair extraction creates N rows per
session) and W3 NarrativeQA (5 questions per document), use
`--group-by` to ensure all rows of a single session/document land in
the SAME split — otherwise val-loss measures memorization of training
rows, not generalization to held-out documents.

Usage:
    # row-wise (default)
    python scripts/split_train_val.py data/wave1/needle.parquet --val-frac 0.05

    # group-aware (W2/W4 chat by session_id)
    python scripts/split_train_val.py data/wave2/wildchat_long.parquet \\
        --val-frac 0.05 --group-by session_id

    # NarrativeQA: derive group from prompt prefix (no document_id column)
    python scripts/split_train_val.py data/wave3/narrativeqa.parquet \\
        --val-frac 0.05 --group-by-prompt-prefix 256
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _row_split(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    n_val = max(1, int(round(n * val_frac)))
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return sorted(indices[n_val:]), sorted(indices[:n_val])  # train, val


def _group_split(
    group_keys: list,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Group-aware split. All rows sharing a group_key land in the same
    split. Returns (train_idx, val_idx) row lists.
    """
    # Bucket rows by group key.
    buckets: dict = {}
    for row_idx, key in enumerate(group_keys):
        buckets.setdefault(key, []).append(row_idx)
    unique_keys = list(buckets.keys())
    n_groups = len(unique_keys)
    n_val_groups = max(1, int(round(n_groups * val_frac)))
    rng = random.Random(seed)
    rng.shuffle(unique_keys)
    val_keys = set(unique_keys[:n_val_groups])
    train_idx: list[int] = []
    val_idx: list[int] = []
    for key, rows in buckets.items():
        if key in val_keys:
            val_idx.extend(rows)
        else:
            train_idx.extend(rows)
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def _prompt_prefix_keys(table, n_prefix_tokens: int) -> list[str]:
    """Build group keys from the first n_prefix_tokens of `prompt_ids`.
    Used for NarrativeQA where there's no document_id column but rows
    sharing a passage have identical prompt prefixes."""
    prompts = table.column("prompt_ids").to_pylist()
    keys = []
    for p in prompts:
        prefix = tuple(p[:n_prefix_tokens])
        # hash to keep key small in memory
        keys.append(hashlib.md5(repr(prefix).encode()).hexdigest())
    return keys


def split_parquet(
    input_path: Path,
    val_frac: float,
    seed: int,
    group_by: str | None = None,
    group_by_prompt_prefix: int | None = None,
) -> tuple[Path, Path]:
    table = pq.read_table(input_path)
    n = len(table)

    if group_by is not None:
        if group_by not in table.column_names:
            raise SystemExit(
                f"--group-by {group_by!r} not found in {input_path.name}; "
                f"available columns: {table.column_names}"
            )
        group_keys = table.column(group_by).to_pylist()
        train_idx, val_idx = _group_split(group_keys, val_frac, seed)
        n_groups = len(set(group_keys))
        print(f"  group-aware split by `{group_by}`: {n_groups} groups → "
              f"{n - len(val_idx)} train / {len(val_idx)} val rows")
    elif group_by_prompt_prefix is not None:
        group_keys = _prompt_prefix_keys(table, group_by_prompt_prefix)
        train_idx, val_idx = _group_split(group_keys, val_frac, seed)
        n_groups = len(set(group_keys))
        print(f"  group-aware split by prompt-prefix-{group_by_prompt_prefix}-tokens "
              f"hash: {n_groups} groups → {n - len(val_idx)} train / {len(val_idx)} val rows")
    else:
        train_idx, val_idx = _row_split(n, val_frac, seed)

    train_table = table.take(pa.array(train_idx))
    val_table = table.take(pa.array(val_idx))

    train_path = input_path.with_suffix(".train.parquet")
    val_path = input_path.with_suffix(".val.parquet")
    pq.write_table(train_table, train_path)
    pq.write_table(val_table, val_path)
    print(f"  {input_path.name}: {n} rows -> {len(train_table)} train + {len(val_table)} val")
    print(f"    {train_path}  ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"    {val_path}  ({val_path.stat().st_size / 1e6:.1f} MB)")
    return train_path, val_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path, help="parquet file(s) to split")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction held out for validation (default 0.05)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-by", type=str, default=None,
                    help="Column to group by (all rows sharing the value land "
                         "in the same split). Use for chat (session_id) etc.")
    ap.add_argument("--group-by-prompt-prefix", type=int, default=None,
                    help="Group by hash of first N tokens of prompt_ids "
                         "(for NarrativeQA which lacks document_id).")
    args = ap.parse_args()

    if args.group_by and args.group_by_prompt_prefix:
        raise SystemExit("--group-by and --group-by-prompt-prefix are mutually exclusive")

    for path in args.inputs:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        # Skip if already a .train. or .val. file.
        if ".train." in path.name or ".val." in path.name:
            print(f"  [skip] {path} is already a split file")
            continue
        split_parquet(
            path, val_frac=args.val_frac, seed=args.seed,
            group_by=args.group_by,
            group_by_prompt_prefix=args.group_by_prompt_prefix,
        )


if __name__ == "__main__":
    main()
