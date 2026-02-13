"""
Run all baseline evaluations: smoke test, perplexity, and zero-shot benchmarks.

Usage:
    python run_all.py
    python run_all.py --skip-benchmarks   # skip lm-eval (slow)
    python run_all.py --model pythia-160m  # single model only
"""

import argparse
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(name: str, args: list[str] = None):
    script = os.path.join(SCRIPT_DIR, name)
    cmd = [sys.executable, script] + (args or [])
    print(f"\n{'#'*70}")
    print(f"# Running: {' '.join(cmd)}")
    print(f"{'#'*70}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all baseline evaluations")
    parser.add_argument("--model", default="all", help="Model to test (default: all)")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Skip lm-eval zero-shot benchmarks (slow)")
    args = parser.parse_args()

    model_args = ["--model", args.model] if args.model != "all" else []

    # 1. Smoke test
    run_script("smoke_test.py")

    # 2. Perplexity
    run_script("eval_perplexity.py", model_args)

    # 3. Zero-shot benchmarks
    if not args.skip_benchmarks:
        run_script("run_benchmarks.py", model_args)
    else:
        print("\nSkipping zero-shot benchmarks (--skip-benchmarks)")

    print(f"\n{'='*70}")
    print("All evaluations complete. Results in: auxiliary_repos/baselines/results/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
