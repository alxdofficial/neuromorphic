"""Profile v9-backprop training step — breakdown by component.

Usage:
    python scripts/profile_v8.py [--bs 4] [--no-memory]
"""

import argparse
import time
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

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
    for p_param in model.memory.parameters():
        p_param.data = p_param.data.float()
    model.lm.mem_gate.data = model.lm.mem_gate.data.float()
    model.train()

    use_memory = not args.no_memory

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    trainer = V8Trainer(
        model=model, optimizer=optimizer,
        scheduler=scheduler, dataloader=iter([]), config=cfg, device=device,
        use_memory=use_memory,
    )

    print(f"Profiling v9-backprop | BS={args.bs} | T={cfg.T} | memory={'ON' if use_memory else 'OFF'}")
    print(f"  N={cfg.N_neurons} neurons, K={cfg.K_connections}, D_neuron={cfg.D_neuron}")
    print(f"  Params: LM={model.lm_param_count()/1e6:.1f}M, Mem={model.memory_param_count()/1e6:.1f}M, "
          f"Total={model.param_count()/1e6:.1f}M")
    print()

    # Warmup
    for i in range(args.warmup):
        batch = FakeBatch(args.bs, cfg.T, device)
        trainer.train_chunk(batch)

    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for i in range(args.steps):
        batch = FakeBatch(args.bs, cfg.T, device)
        torch.cuda.synchronize()
        t0 = time.time()
        metrics = trainer.train_chunk(batch)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        tok_s = args.bs * cfg.T / elapsed
        print(f"  Step {i}: {elapsed:.3f}s, {tok_s/1e3:.1f}K tok/s, "
              f"loss={metrics['loss']:.3f}, ppl={metrics['ppl']:.1f}")

    avg = sum(times) / len(times)
    avg_tok = args.bs * cfg.T / avg
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nAverage: {avg:.3f}s/step, {avg_tok/1e3:.1f}K tok/s")
    print(f"Peak VRAM: {peak:.2f} GB")


if __name__ == "__main__":
    main()
