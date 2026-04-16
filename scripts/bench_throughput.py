"""Measure steady-state training throughput."""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def bench(bs=32, warmup=3, measure=10):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = Config.tier_a()
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                             fused=True)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    mem_params = sum(p.numel() for p in model.memory.parameters()) / 1e6
    lm_params = sum(p.numel() for p in model.lm.parameters()) / 1e6
    print(f"Params: total={total_params:.1f}M  LM={lm_params:.1f}M  Mem={mem_params:.1f}M")
    print(f"Config: BS={bs}, T={cfg.T}, N={cfg.N_total}, C_h={cfg.conv_channels}, "
          f"K={cfg.num_codes}, D_code={cfg.code_dim}")
    print(f"  ckpt_memory={cfg.checkpoint_memory}, ckpt_decoder={cfg.checkpoint_decoder}")
    print()

    def step():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.detach_states()

    # Warmup — torch.compile pays its first-call cost here.
    print(f"Warmup ({warmup} steps)...")
    for i in range(warmup):
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        print(f"  warmup step {i}: {time.time() - t0:.2f}s")

    print(f"\nMeasuring ({measure} steps)...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(measure):
        step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens = bs * cfg.T * measure
    tok_s = tokens / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print()
    print(f"=== Results ===")
    print(f"Elapsed: {elapsed:.2f}s for {measure} steps ({tokens:,} tokens)")
    print(f"Throughput: {tok_s:,.0f} tok/s")
    print(f"Step time:  {elapsed / measure * 1000:.1f} ms/step")
    print(f"Peak VRAM:  {peak_gb:.2f} GB")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=10)
    args = p.parse_args()
    bench(bs=args.bs, warmup=args.warmup, measure=args.measure)
