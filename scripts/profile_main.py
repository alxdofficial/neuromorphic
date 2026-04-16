"""Profile main-branch throughput to find optimization opportunities."""

import sys
import time
import torch

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def run():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = Config.tier_a()
    cfg.vocab_size = 32000
    cfg.validate()

    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                             fused=True)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    lm = sum(p.numel() for p in model.lm.parameters()) / 1e6
    mem = sum(p.numel() for p in model.memory.parameters()) / 1e6
    dp = sum(p.numel() for p in model.memory.discrete_policy.parameters()) / 1e6
    print(f"Params: total={total:.1f}M  LM={lm:.1f}M  Mem={mem:.1f}M "
          f"(discrete_policy={dp:.1f}M)")
    print(f"Config: N_cells={cfg.N_cells}, neurons_per_cell={cfg.neurons_per_cell}, "
          f"Hmod={cfg.cell_mod_hidden}, K={cfg.num_codes}")

    BS = 72
    def step():
        input_ids = torch.randint(1, cfg.vocab_size, (BS, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (BS, cfg.T), device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        opt.step()
        model.detach_states()

    # Warmup
    print("\nWarmup...")
    for i in range(3):
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        print(f"  step {i}: {time.time()-t0:.2f}s")

    # Profiler trace
    print("\nProfiling 3 steps...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

    tokens = BS * cfg.T * 3
    print(f"\n=== Overall ===")
    print(f"Step time: {elapsed/3*1000:.1f} ms")
    print(f"Throughput: {tokens/elapsed:,.0f} tok/s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print(f"\n=== Top 20 ops by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    run()
