"""Profile v8 training step — breakdown by component.

Usage:
    python -m scripts.profile_v8 [--bs 4] [--no-memory]
"""

import argparse
import time
import torch
import torch.nn.functional as F

from src.v8.config import V8Config
from src.v8.model import V8Model
from src.v8.trainer import V8Trainer


class FakeBatch:
    def __init__(self, bs, T, device, eot_id=2):
        self.input_ids = torch.randint(0, 32000, (bs, T), device=device)
        self.target_ids = torch.randint(0, 32000, (bs, T), device=device)
        self.prev_token = torch.randint(0, 32000, (bs,), device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = V8Config.tier_a(vocab_size=32000)
    cfg.validate()

    model = V8Model(cfg).to(device).to(torch.bfloat16)
    model.train()

    use_memory = not args.no_memory

    lm_opt = torch.optim.AdamW(model.lm.parameters(), lr=3e-4, fused=True)
    nm_opt = torch.optim.Adam(model.neuromod.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(lm_opt, lambda _: 1.0)

    trainer = V8Trainer(
        model=model, lm_optimizer=lm_opt, neuromod_optimizer=nm_opt,
        scheduler=sched, dataloader=iter([]), config=cfg, device=device,
        use_memory=use_memory,
    )

    print(f"Profiling v8 | BS={args.bs} | T={cfg.T} | memory={'ON' if use_memory else 'OFF'}")
    print(f"  N={cfg.N_neurons} neurons, K={cfg.K_connections} connections, "
          f"action_every={cfg.action_every}")
    print()

    # Warmup
    for i in range(args.warmup):
        batch = FakeBatch(args.bs, cfg.T, device)
        trainer.train_chunk(batch)
        print(f"  warmup {i+1}/{args.warmup}")

    # Now profile with torch profiler
    print(f"\n  Running {args.steps} profiled steps...")
    step_times = []
    for i in range(args.steps):
        batch = FakeBatch(args.bs, cfg.T, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        metrics = trainer.train_chunk(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)
        print(f"  step {i+1}/{args.steps}: {elapsed*1000:.0f}ms, "
              f"{args.bs*cfg.T/elapsed/1e3:.1f}K tok/s")

    mean_ms = sum(step_times) / len(step_times) * 1000
    tok_s = args.bs * cfg.T / (mean_ms / 1000)
    print(f"\n  Mean: {mean_ms:.0f}ms/step, {tok_s/1e3:.1f}K tok/s")

    # Now do a detailed torch.profiler trace
    print(f"\n  Running torch.profiler trace...")
    batch = FakeBatch(args.bs, cfg.T, device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        trainer.train_chunk(batch)

    # Print kernel summary
    print("\n" + "=" * 80)
    print("Top CUDA kernels by total time:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20,
        top_level_events_only=False,
    ))

    # Print summary stats
    print("\n" + "=" * 80)
    print("Top CPU operations by total time:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=15,
        top_level_events_only=True,
    ))

    # Memory summary
    print("\n" + "=" * 80)
    print("Memory summary:")
    print("=" * 80)
    peak = torch.cuda.max_memory_allocated() / 1e9
    current = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  Peak allocated:    {peak:.2f} GB")
    print(f"  Current allocated: {current:.2f} GB")
    print(f"  Reserved:          {reserved:.2f} GB")

    # Count kernel launches (key_averages uses self_cuda_time_total)
    events = prof.key_averages()

    def cuda_time(e):
        for attr in ('self_cuda_time_total', 'cuda_time_total', 'device_time_total'):
            v = getattr(e, attr, None)
            if v is not None:
                return v
        return 0

    total_cuda_time = sum(cuda_time(e) for e in events)
    n_kernels = sum(e.count for e in events if cuda_time(e) > 0)
    print(f"\n  Total CUDA kernel launches: {n_kernels}")
    print(f"  Total CUDA time: {total_cuda_time/1e3:.1f}ms")

    # BMM specifically
    bmm_events = [e for e in events
                  if 'bmm' in e.key.lower() or 'batch_matmul' in e.key.lower()
                  or 'gemm' in e.key.lower()]
    if bmm_events and total_cuda_time > 0:
        bmm_time = sum(cuda_time(e) for e in bmm_events)
        bmm_calls = sum(e.count for e in bmm_events)
        print(f"\n  BMM/GEMM kernels: {bmm_calls} calls, {bmm_time/1e3:.1f}ms total "
              f"({bmm_time/total_cuda_time*100:.0f}% of CUDA time)")


if __name__ == "__main__":
    main()
