#!/usr/bin/env python3
"""Benchmark v8 training throughput and memory usage.

Automatically finds max batch size, then measures throughput.

Usage:
    python scripts/benchmark_v8.py                     # full v8 with memory
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


def compile_model(model):
    """Compile individual methods (not module-level — we call named methods, not forward())."""
    model.lm.forward_scan_lower = torch.compile(model.lm.forward_scan_lower)
    model.lm.forward_scan_upper = torch.compile(model.lm.forward_scan_upper)
    model.lm.forward_output = torch.compile(model.lm.forward_output)
    model.neuromod.get_action_and_value = torch.compile(
        model.neuromod.get_action_and_value)


def make_batch(bs, T, device):
    return StreamBatch(
        input_ids=torch.randint(0, 32000, (bs, T), device=device),
        target_ids=torch.randint(0, 32000, (bs, T), device=device),
        prev_token=torch.zeros(bs, dtype=torch.long, device=device),
    )


def try_bs(model, cfg, device, bs, use_memory, use_compile):
    """Try a batch size. Returns True if it fits, False if OOM."""
    cleanup()
    try:
        lm_opt = torch.optim.AdamW(model.lm.parameters(), lr=3e-4)
        nm_opt = torch.optim.Adam(model.neuromod.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(lm_opt, lambda _: 1.0)
        trainer = V8Trainer(
            model=model, lm_optimizer=lm_opt, neuromod_optimizer=nm_opt,
            scheduler=sched, dataloader=iter([]), config=cfg, device=device,
            use_memory=use_memory,
        )
        # Run 2 steps to catch fragmentation
        for _ in range(2):
            trainer.train_chunk(make_batch(bs, cfg.T, device))
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  BS={bs}: OK ({peak:.1f}GB)", flush=True)
        del trainer, lm_opt, nm_opt, sched
        cleanup()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"  BS={bs}: OOM", flush=True)
            del trainer, lm_opt, nm_opt, sched
            cleanup()
            return False
        raise


def bench(model, cfg, device, bs, use_memory):
    """Benchmark at a specific batch size. Returns (tok_s, vram_gb, metrics)."""
    cleanup()
    lm_opt = torch.optim.AdamW(model.lm.parameters(), lr=3e-4, fused=True)
    nm_opt = torch.optim.Adam(model.neuromod.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(lm_opt, lambda _: 1.0)
    trainer = V8Trainer(
        model=model, lm_optimizer=lm_opt, neuromod_optimizer=nm_opt,
        scheduler=sched, dataloader=iter([]), config=cfg, device=device,
        use_memory=use_memory,
    )

    # Warmup
    for i in range(WARMUP_STEPS):
        trainer.train_chunk(make_batch(bs, cfg.T, device))
        print(f"  warmup {i+1}/{WARMUP_STEPS}", flush=True)

    # Timed
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    last_metrics = {}
    for i in range(BENCH_STEPS):
        batch = make_batch(bs, cfg.T, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = trainer.train_chunk(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_metrics = m
        tok_s = bs * cfg.T / (t1 - t0)
        print(f"  step {i+1}/{BENCH_STEPS}: {tok_s/1e3:.1f}K tok/s", flush=True)

    avg = sum(times) / len(times)
    tok_s = bs * cfg.T / avg
    vram = torch.cuda.max_memory_allocated() / 1e9
    del trainer, lm_opt, nm_opt, sched
    return tok_s, vram, last_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=None, help="Force batch size (skip auto)")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Enable gradient checkpointing (off by default, matching training)")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    use_memory = not args.no_memory
    use_compile = not args.no_compile

    cfg = V8Config.tier_a(
        vocab_size=32000,
        gradient_checkpointing=args.grad_ckpt,
    )
    cfg.validate()

    print(f"GPU: {gpu} ({vram_total:.1f}GB)")
    print(f"Config: D={cfg.D}, L={cfg.L_total}, neurons={cfg.N_neurons}")
    print(f"T={cfg.T}, memory={'ON' if use_memory else 'OFF'}, "
          f"compile={use_compile}, grad_ckpt={args.grad_ckpt}")

    model = V8Model(cfg).to(device).to(torch.bfloat16)
    params = model.param_count()
    print(f"Params: {params/1e6:.1f}M")

    if use_compile:
        print("Compiling...", flush=True)
        compile_model(model)

    # Find max BS
    if args.bs is not None:
        bs = args.bs
        print(f"Forced BS={bs}", flush=True)
    else:
        print("Finding max batch size...", flush=True)
        bs = 1
        for candidate in BS_CANDIDATES:
            del model
            cleanup()
            model = V8Model(cfg).to(device).to(torch.bfloat16)
            if use_compile:
                compile_model(model)
            if try_bs(model, cfg, device, candidate, use_memory, use_compile):
                bs = candidate
                break
        print(f"Max BS={bs}", flush=True)

    # Fresh model for benchmark
    del model
    cleanup()
    model = V8Model(cfg).to(device).to(torch.bfloat16)
    if use_compile:
        print("Compiling...", flush=True)
        compile_model(model)

    # Benchmark
    print(f"\nBenchmarking BS={bs}...", flush=True)
    tok_s, vram, metrics = bench(model, cfg, device, bs, use_memory)

    print(f"\n{'='*50}")
    print(f"Result: {tok_s/1e3:.1f}K tok/s | {bs*cfg.T*1000/tok_s:.0f}ms/step | "
          f"{vram:.1f}GB VRAM | BS={bs}")
    print(f"  loss={metrics.get('loss', 0):.3f}")
    if use_memory:
        print(f"  rl_policy_loss={metrics.get('rl_policy_loss', 0):.4f}, "
              f"rl_loss={metrics.get('rl_loss', 0):.3f}, "
              f"adv={metrics.get('rl_adv_mean', 0):.4f}±{metrics.get('rl_adv_std', 0):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
