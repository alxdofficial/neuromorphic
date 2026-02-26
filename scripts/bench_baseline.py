"""Quick throughput benchmark for baseline models at various batch sizes.

Usage:
    python scripts/bench_baseline.py --model gpt2-small
    python scripts/bench_baseline.py --model rwkv7-168m
    python scripts/bench_baseline.py --model gpt2-medium   # etc.
"""

import argparse
import sys
import time
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast

# Add baseline script to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "auxiliary_repos", "baselines", "eval_scripts"))
from train_baseline import MODEL_CONFIGS, create_model, SEQ_LENGTH

WARMUP_STEPS = 5
MEASURE_STEPS = 20


def bench(model_name: str, batch_sizes: list[int], device: str = "cuda"):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"SEQ_LENGTH={SEQ_LENGTH}, warmup={WARMUP_STEPS}, measure={MEASURE_STEPS}")
    print(f"{'='*60}\n")

    model = create_model(model_name, device)
    amp_dtype = torch.bfloat16

    results = []
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Random input
        input_ids = torch.randint(0, 32000, (bs, SEQ_LENGTH), device=device)
        labels = torch.randint(0, 32000, (bs, SEQ_LENGTH), device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        try:
            # Warmup
            for _ in range(WARMUP_STEPS):
                optimizer.zero_grad()
                with autocast("cuda", dtype=amp_dtype):
                    out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            for _ in range(MEASURE_STEPS):
                optimizer.zero_grad()
                with autocast("cuda", dtype=amp_dtype):
                    out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            tokens_per_step = bs * SEQ_LENGTH
            total_tokens = tokens_per_step * MEASURE_STEPS
            tok_s = total_tokens / elapsed
            ms_per_step = (elapsed / MEASURE_STEPS) * 1000
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

            print(f"  BS={bs:>3d}  |  {tok_s:>8,.0f} tok/s  |  {ms_per_step:>7.1f} ms/step  |  {peak_vram:.2f} GB VRAM")
            results.append((bs, tok_s, ms_per_step, peak_vram))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  BS={bs:>3d}  |  OOM")
            break

        del optimizer

    if results:
        best = max(results, key=lambda r: r[1])
        print(f"\n  ** Best: BS={best[0]}, {best[1]:,.0f} tok/s, {best[2]:.1f} ms/step, {best[3]:.2f} GB **")
        hours_1_5b = 1_500_000_000 / best[1] / 3600
        print(f"  ** Estimated 1.5B tokens: {hours_1_5b:.1f}h **")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: auto-sweep)")
    args = parser.parse_args()

    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    else:
        # Default sweep
        batch_sizes = [8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256]

    bench(args.model, batch_sizes)


if __name__ == "__main__":
    main()
