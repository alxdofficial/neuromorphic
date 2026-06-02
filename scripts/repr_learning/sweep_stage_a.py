"""Stage-A capacity sweep — recall vs #KV-pairs elbow, per arm.

Capacity ∝ params (Nichani 2024), so with the v2.1 budget the storage ceiling is
large. An EARLY elbow (recall falls off at single-digit pairs) indicts the
write/read MODULES, not the compression bottleneck — that is the clean
"modules vs compression" separation this sweep is built to surface.

Usage: python scripts/repr_learning/sweep_stage_a.py --arms graph_v6_baseline continuous_baseline \
           --n-pairs 1 2 4 8 16 --steps 600
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from scripts.repr_learning.train_stage_a import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["graph_v6_baseline", "continuous_baseline"])
    ap.add_argument("--n-pairs", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    rows = []
    for arm in args.arms:
        for np_ in args.n_pairs:
            print(f"\n########## {arm}  n_pairs={np_} ##########")
            real, off, shuf = train_one(arm, np_, steps=args.steps, batch_size=args.batch_size,
                                        lr=args.lr, tok=tok, verbose=True)
            rows.append((arm, np_, real, off, shuf))

    # elbow table: REAL recall (and the controls) per arm × #pairs
    print("\n\n================ RECALL-vs-#PAIRS ELBOW ================")
    print(f"{'arm':22s} {'n_pairs':>7s} {'REAL':>6s} {'OFF':>6s} {'SHUF':>6s}  {'load-bearing':>12s}")
    for arm, np_, real, off, shuf in rows:
        print(f"{arm:22s} {np_:7d} {real:6.3f} {off:6.3f} {shuf:6.3f}  {real - max(off, shuf):+12.3f}")
    print("\nElbow = #pairs where REAL recall collapses. Early elbow => modules, not bottleneck.")


if __name__ == "__main__":
    main()
