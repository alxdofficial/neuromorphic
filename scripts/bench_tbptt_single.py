"""Single-config benchmark. Run once per process to avoid compile cache pollution."""

import sys
import time

import torch

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--tbptt", type=int, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = Config.tier_a(tbptt_block=args.tbptt)
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)

    def step():
        input_ids = torch.randint(1, cfg.vocab_size, (args.bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (args.bs, cfg.T), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.detach_states()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.measure):
        step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_s = args.bs * cfg.T * args.measure / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    step_ms = elapsed / args.measure * 1000
    print(f"BS={args.bs} tbptt={args.tbptt}: {tok_s:,.0f} tok/s  {step_ms:.1f} ms/step  {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
