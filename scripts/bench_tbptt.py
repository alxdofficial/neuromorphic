"""Brief sweep: does larger tbptt_block help at modest BS?"""

import sys
import time

import torch

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def bench_one(bs: int, tbptt: int, warmup=3, measure=8):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = Config.tier_a(tbptt_block=tbptt)
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)

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

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(measure):
            step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
    except torch.cuda.OutOfMemoryError:
        return None, None, None

    tok_s = bs * cfg.T * measure / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    step_ms = elapsed / measure * 1000
    del model, opt
    return tok_s, peak_gb, step_ms


def main():
    configs = [
        (16, 8), (16, 16), (16, 32), (16, 64),
        (32, 8), (32, 16), (32, 32), (32, 64),
    ]
    print(f"{'BS':>4} {'tbptt':>6} {'tok/s':>9} {'ms/step':>9} {'VRAM GB':>8}")
    print("-" * 45)
    for bs, tbptt in configs:
        tok_s, peak_gb, step_ms = bench_one(bs, tbptt)
        if tok_s is None:
            print(f"{bs:>4} {tbptt:>6}   OOM")
        else:
            print(f"{bs:>4} {tbptt:>6} {tok_s:>9,.0f} {step_ms:>8.1f}  {peak_gb:>7.2f}")


if __name__ == "__main__":
    main()
