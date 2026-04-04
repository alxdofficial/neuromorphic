"""Quick throughput + VRAM benchmark."""

import time
import torch
import torch.nn.functional as F

from src.model.config import Config
from src.model.model import Model


def benchmark(bs=8, steps=20, warmup=5):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    print(f"Config: N={config.N}, D_n={config.D_n}, K={config.K}, "
          f"D={config.D}, T={config.T}, BS={bs}")
    print(f"  Ports: {config.N_port} in + {config.N_port} out, alpha={config.alpha}")

    model = Model(config).to(device)
    total = model.param_count()
    lm_p = model.lm_param_count()
    mem_p = model.memory_param_count()
    print(f"  Params: {total/1e6:.1f}M (LM={lm_p/1e6:.1f}M, Mem={mem_p/1e6:.1f}M)")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    input_ids = torch.randint(1, config.vocab_size, (bs, config.T), device=device)
    target_ids = torch.randint(0, config.vocab_size, (bs, config.T), device=device)

    # Warmup
    print(f"\nWarmup ({warmup} steps)...")
    for i in range(warmup):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = model.forward_chunk(input_ids, target_ids=target_ids)
        result["loss"].backward()
        model.detach_states()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"Benchmarking ({steps} steps)...")
    times = []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = model.forward_chunk(input_ids, target_ids=target_ids)
        result["loss"].backward()
        model.detach_states()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    peak_gb = peak_mb / 1000

    avg_ms = sum(times) * 1000 / len(times)
    min_ms = min(times) * 1000
    tokens_per_step = bs * config.T
    tok_per_s = tokens_per_step / (avg_ms / 1000)

    print(f"\n{'='*50}")
    print(f"BS={bs}, T={config.T}, tokens/step={tokens_per_step}")
    print(f"Avg step: {avg_ms:.1f} ms  (min: {min_ms:.1f} ms)")
    print(f"Throughput: {tok_per_s:.0f} tok/s ({tok_per_s/1e3:.1f}K)")
    print(f"Peak VRAM: {peak_gb:.2f} GB ({peak_mb:.0f} MB)")
    print(f"{'='*50}")

    # Also measure forward-only for memory graph
    print(f"\nMemory graph forward-only timing...")
    mg = model.memory
    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    mg_times = []
    for i in range(steps):
        mg.detach_states()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            mem_out = mg.forward_segment(H_aug)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mg_times.append(t1 - t0)

    avg_mg = sum(mg_times) * 1000 / len(mg_times)
    per_token_ms = avg_mg / config.T
    mg_tok_s = tokens_per_step / (avg_mg / 1000)
    print(f"  Avg segment: {avg_mg:.1f} ms ({per_token_ms:.2f} ms/token)")
    print(f"  Memory graph throughput: {mg_tok_s:.0f} tok/s ({mg_tok_s/1e3:.1f}K)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    benchmark(bs=args.bs, steps=args.steps)
