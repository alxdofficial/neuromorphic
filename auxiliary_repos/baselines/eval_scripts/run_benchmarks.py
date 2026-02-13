"""
Run zero-shot benchmarks on baseline models via lm-evaluation-harness.

Benchmarks: PIQA, ARC-Easy, ARC-Challenge, LAMBADA, HellaSwag

Usage:
    python run_benchmarks.py --model pythia-160m
    python run_benchmarks.py --model mamba-130m
    python run_benchmarks.py --model all
    python run_benchmarks.py --model pythia-160m --tasks piqa,lambada_openai

Prerequisites:
    pip install lm-eval>=0.4.0
"""

import argparse
import json
import os
import subprocess
import sys

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "mamba-130m": "state-spaces/mamba-130m-hf",
}

# Tasks that differentiate at small scale (from baseline_comparison_plan.md)
DEFAULT_TASKS = [
    "piqa",
    "arc_easy",
    "arc_challenge",
    "lambada_openai",
    "hellaswag",
    "winogrande",
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_lm_harness(model_name: str, repo_id: str, tasks: list[str]):
    """Run lm-evaluation-harness for a model."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"benchmarks_{model_name}.json")

    task_str = ",".join(tasks)
    print(f"\n{'='*70}")
    print(f"Running lm-eval: {model_name} ({repo_id})")
    print(f"Tasks: {task_str}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={repo_id},dtype=float16",
        "--tasks", task_str,
        "--batch_size", "auto",
        "--output_path", out_path,
        "--log_samples",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"WARNING: lm-eval exited with code {result.returncode}")
    else:
        print(f"\nResults saved to: {out_path}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run zero-shot benchmarks")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        default="all", help="Which model to benchmark")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task list (default: piqa,arc_easy,arc_challenge,lambada_openai,hellaswag,winogrande)")
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASKS

    if args.model == "all":
        targets = MODELS.items()
    else:
        targets = [(args.model, MODELS[args.model])]

    for name, repo_id in targets:
        run_lm_harness(name, repo_id, tasks)


if __name__ == "__main__":
    main()
