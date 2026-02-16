"""Profile one training chunk to see where time is spent.

Usage: python -m scripts.profile_training [--steps N]
"""
import argparse
import time
import torch
from torch.profiler import profile, ProfilerActivity

from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.training.trainer import TBPTTTrainer
from src.data import create_dataloader, get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda")
    BS = 32
    config = ModelConfig.tier_a(use_compile=True)
    config.set_phase("B")

    model = NeuromorphicLM(config).to(device)
    model.initialize_states(BS, device)
    model.compile_for_training()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    tokenizer = get_tokenizer("tinyllama")
    dl = create_dataloader(
        phase="B", tokenizer=tokenizer, batch_size=BS,
        seq_length=config.T,
    )
    trainer = TBPTTTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        dataloader=dl, config=config, device=device,
        collector=None, log_interval=9999,
    )

    # Warmup
    print(f"Warming up ({args.warmup} steps)...")
    data_iter = iter(dl)
    for i in range(args.warmup):
        batch = next(data_iter)
        trainer.train_chunk(batch)
        trainer.global_step += 1
        print(f"  Step {i+1} done")

    torch.cuda.synchronize()

    # Wall-clock timing
    print(f"\nTiming {args.steps} steps...")
    times = []
    for i in range(args.steps):
        batch = next(data_iter)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer.train_chunk(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        trainer.global_step += 1
        tok = BS * config.T
        print(f"  Step {i+1}: {t1-t0:.4f}s  ({tok/(t1-t0):.0f} tok/s)")

    avg = sum(times) / len(times)
    tok_per_step = BS * config.T
    print(f"\nAverage: {avg:.4f}s/step = {tok_per_step/avg:.0f} tok/s")

    # PyTorch profiler
    print(f"\nRunning PyTorch profiler ({args.steps} steps)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
    ) as prof:
        for i in range(args.steps):
            batch = next(data_iter)
            trainer.train_chunk(batch)
            trainer.global_step += 1

    torch.cuda.synchronize()

    print("\n=== Top 40 CUDA kernels by total GPU time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=40,
        max_name_column_width=90,
    ))

    print("\n=== Top 25 CPU ops by self CPU time ===")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=25,
        max_name_column_width=90,
    ))

    # Export
    trace_path = "/tmp/neuromorphic_profile.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace: {trace_path}")


if __name__ == "__main__":
    main()
