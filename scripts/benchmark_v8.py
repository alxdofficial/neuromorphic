#!/usr/bin/env python3
"""Benchmark v9-backprop training throughput and memory usage.

Automatically finds max batch size, then measures throughput.

Usage:
    python scripts/benchmark_v8.py                     # full model with memory
    python scripts/benchmark_v8.py --no-memory         # LM-only baseline
    python scripts/benchmark_v8.py --no-compile        # disable torch.compile
    python scripts/benchmark_v8.py --bs 4              # force batch size (skip auto)
"""

import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, ".")
from src.v8.config import V8Config
from src.v8.model import V8Model
from src.v8.trainer import V8Trainer
from src.data.streaming import StreamBatch

BS_CANDIDATES = [32, 24, 16, 12, 8, 6, 4, 2, 1]
WARMUP_STEPS = 3
BENCH_STEPS = 5


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def make_batch(bs, T, device):
    return StreamBatch(
        input_ids=torch.randint(0, 32000, (bs, T), device=device),
        target_ids=torch.randint(0, 32000, (bs, T), device=device),
        prev_token=torch.zeros(bs, dtype=torch.long, device=device),
    )


def try_bs(model, cfg, device, bs, use_memory):
    """Try a batch size. Returns True if it fits, False if OOM."""
    cleanup()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        trainer = V8Trainer(
            model=model, optimizer=optimizer,
            scheduler=scheduler, dataloader=iter([]), config=cfg, device=device,
            use_memory=use_memory,
        )
        for _ in range(2):
            trainer.train_chunk(make_batch(bs, cfg.T, device))
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  BS={bs:>2}: OK, {peak:.1f} GB")
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  BS={bs:>2}: OOM")
        cleanup()
        return False


def benchmark(model, cfg, device, bs, use_memory):
    """Measure throughput at given batch size."""
    cleanup()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = V8Trainer(
        model=model, optimizer=optimizer,
        scheduler=scheduler, dataloader=iter([]), config=cfg, device=device,
        use_memory=use_memory,
    )

    # Warmup
    for _ in range(WARMUP_STEPS):
        trainer.train_chunk(make_batch(bs, cfg.T, device))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(BENCH_STEPS):
        torch.cuda.synchronize()
        t0 = time.time()
        trainer.train_chunk(make_batch(bs, cfg.T, device))
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    peak = torch.cuda.max_memory_allocated() / 1e9
    avg = sum(times) / len(times)
    tok_s = bs * cfg.T / avg

    print(f"\n=== Results ===")
    print(f"BS={bs}, T={cfg.T}")
    print(f"Throughput: {tok_s/1e3:.1f}K tok/s ({avg:.3f}s/step)")
    print(f"Peak VRAM: {peak:.2f} GB")
    print(f"Params: {model.param_count()/1e6:.1f}M")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = V8Config.tier_a(vocab_size=32000)
    cfg.validate()

    model = V8Model(cfg).to(device).to(torch.bfloat16)
    for p_param in model.memory.parameters():
        p_param.data = p_param.data.float()
    model.lm.mem_gate.data = model.lm.mem_gate.data.float()
    model.train()

    use_memory = not args.no_memory
    print(f"v9-backprop benchmark | memory={'ON' if use_memory else 'OFF'}")
    print(f"  {cfg.L_total} layers, d_inner={cfg.d_inner}, D={cfg.D}")
    print(f"  N={cfg.N_neurons} neurons, K={cfg.K_connections}, D_neuron={cfg.D_neuron}")
    print(f"  Params: LM={model.lm_param_count()/1e6:.1f}M, Mem={model.memory_param_count()/1e6:.1f}M")

    if args.bs is None:
        print("\nFinding max batch size...")
        best_bs = 1
        for bs in BS_CANDIDATES:
            if try_bs(model, cfg, device, bs, use_memory):
                best_bs = bs
                break
        print(f"Using BS={best_bs}")
    else:
        best_bs = args.bs

    model.initialize_states(best_bs, device)
    benchmark(model, cfg, device, best_bs, use_memory)


if __name__ == "__main__":
    main()
