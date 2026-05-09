"""Split a preprocessed parquet into train + val parquets.

Used to carve out held-out validation slices for trainers' periodic
val-loss computation. Splits row-wise with a fixed seed.

Usage:
    python scripts/split_train_val.py data/wave1/needle.parquet --val-frac 0.05
    # writes data/wave1/needle.train.parquet (95%) + data/wave1/needle.val.parquet (5%)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def split_parquet(input_path: Path, val_frac: float, seed: int) -> tuple[Path, Path]:
    table = pq.read_table(input_path)
    n = len(table)
    n_val = max(1, int(round(n * val_frac)))

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])

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
    args = ap.parse_args()

    for path in args.inputs:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        # Skip if already a .train. or .val. file.
        if ".train." in path.name or ".val." in path.name:
            print(f"  [skip] {path} is already a split file")
            continue
        split_parquet(path, val_frac=args.val_frac, seed=args.seed)


if __name__ == "__main__":
    main()
