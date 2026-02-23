"""Sweep batch sizes to find optimal throughput on current GPU.

Each batch size runs in a separate subprocess to avoid torch.compile
recompilation limits (compiled graphs are specialized per tensor shape).

Usage: python -m scripts.sweep_batch_size [--sizes 32,48,64,80,96]
"""
import argparse
import json
import subprocess
import sys

import torch


_WORKER_SCRIPT = '''
import json, sys, time, torch
from src.data.streaming import StreamBatch
from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.training.trainer import TBPTTTrainer

bs, warmup, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

def _rand_batch(bs, t, vocab, device):
    x = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
    y = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
    prev = torch.zeros(bs, dtype=torch.long, device=device)
    return StreamBatch(input_ids=x, target_ids=y, prev_token=prev)

cfg = ModelConfig.tier_a_wide(T=256, use_compile=True)
cfg.set_phase("B")
cfg.vocab_size = 32000
cfg.eot_id = 2

model = NeuromorphicLM(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
trainer = TBPTTTrainer(
    model=model, optimizer=optimizer, scheduler=scheduler,
    dataloader=iter(()), config=cfg, device=device,
    collector=None, log_interval=10_000,
)

for i in range(warmup):
    trainer.train_chunk(_rand_batch(bs, cfg.T, cfg.vocab_size, device))
    trainer.global_step += 1

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

times = []
for _ in range(steps):
    batch = _rand_batch(bs, cfg.T, cfg.vocab_size, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.train_chunk(batch)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    trainer.global_step += 1
    times.append(t1 - t0)

avg = sum(times) / len(times)
tok_per_step = bs * cfg.T
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(json.dumps({
    "bs": bs, "avg_step_s": round(avg, 4),
    "tok_per_s": round(tok_per_step / avg),
    "tok_per_step": tok_per_step,
    "peak_vram_gb": round(peak_vram, 2),
}))
'''


def run_one(bs: int, warmup: int, steps: int) -> dict | None:
    result = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, str(bs), str(warmup), str(steps)],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
            return None  # OOM
        raise RuntimeError(f"BS={bs} failed:\n{stderr[-500:]}")
    # Last line of stdout is the JSON
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"BS={bs}: no JSON in output:\n{result.stdout[-500:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="32,48,64,80,96,112,128")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Config: tier_a_wide (D=768, L=8, B=2), Phase B, T=256, compiled")
    print(f"Warmup: {args.warmup} steps, Timed: {args.steps} steps")
    print(f"(Each BS runs in a separate process — ~6 min compile warmup each)")
    print()
    print(f"{'BS':>4} | {'tok/step':>8} | {'step (s)':>8} | {'tok/s':>8} | {'VRAM (GB)':>9} | {'util%':>5}")
    print("-" * 60)

    results = []
    for bs in sizes:
        print(f"{bs:>4} | running...", end="", flush=True)
        try:
            r = run_one(bs, args.warmup, args.steps)
        except Exception as e:
            print(f"\r{bs:>4} | ERROR: {e}")
            continue
        if r is None:
            print(f"\r{bs:>4} | {'OOM':>8} | {'---':>8} | {'---':>8} | {'>24':>9} | {'---':>5}")
            break
        util = r["peak_vram_gb"] / gpu_mem * 100
        print(f"\r{r['bs']:>4} | {r['tok_per_step']:>8} | {r['avg_step_s']:>8.4f} | {r['tok_per_s']:>8} | {r['peak_vram_gb']:>9.2f} | {util:>5.1f}")
        results.append(r)

    if results:
        best = max(results, key=lambda r: r["tok_per_s"])
        print(f"\nBest throughput: BS={best['bs']} at {best['tok_per_s']} tok/s "
              f"({best['peak_vram_gb']} GB VRAM)")


if __name__ == "__main__":
    main()
