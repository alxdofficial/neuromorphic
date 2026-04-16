"""Speed sweep across config variations.

Each variation is constrained to ≥256 total neurons. We measure steady-state
throughput after torch.compile warmup. Reports tok/s, step time, peak VRAM,
and total params.
"""

import sys
import time
import torch

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def bench(BS, overrides, warmup=2, measure=5, label=""):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg = Config.tier_a(**overrides)
        cfg.vocab_size = 32000
        cfg.validate()

        device = torch.device("cuda")
        model = Model(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        def step():
            input_ids = torch.randint(1, cfg.vocab_size, (BS, cfg.T),
                                        device=device)
            target_ids = torch.randint(0, cfg.vocab_size, (BS, cfg.T),
                                         device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = model.forward_chunk(input_ids, target_ids=target_ids)
            opt.zero_grad(set_to_none=True)
            r["loss"].backward()
            opt.step()
            model.detach_states()

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(measure):
            step()
        torch.cuda.synchronize()
        el = time.time() - t0

        tok_s = BS * cfg.T * measure / el
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = sum(p.numel() for p in model.parameters()) / 1e6
        mem = sum(p.numel() for p in model.memory.parameters()) / 1e6

        del model, opt
        return {
            "label": label,
            "BS": BS,
            "tok_s": tok_s,
            "step_ms": el / measure * 1000,
            "peak_gb": peak,
            "total_mb": total,
            "mem_mb": mem,
        }
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"label": label, "BS": BS, "oom": True}


def main():
    print(f"{'Label':<32s}  {'BS':>3s}  {'tok/s':>8s}  {'step':>6s}  "
          f"{'VRAM':>5s}  {'params':>6s}")
    print("-" * 72)

    # Baseline: current NC=1 N=256 D_n=256
    configs = [
        ("NC=1 N=256 D_n=256 (current)", dict()),
        ("NC=1 N=256 D_n=512 (wider)", dict(D_n=512)),
        ("NC=1 N=384 D_n=256 (bigger pool)", dict(N_total=384)),
        ("NC=1 N=512 D_n=256 (bigger pool)", dict(N_total=512)),
        # Shrink D_n: keeps N_total and number of LM interface pools
        ("NC=1 N=256 D_n=128 (narrower)", dict(D_n=128)),
        # decoder hidden variations (affects decoder MLP)
        ("NC=1 N=256 dec_hid=256", dict(decoder_hidden=256)),
        ("NC=1 N=256 dec_hid=1024", dict(decoder_hidden=1024)),
        # Attention width
        ("NC=1 N=256 F=32", dict(attn_token_dim=32, attn_n_heads=2)),
        ("NC=1 N=256 F=128", dict(attn_token_dim=128)),
        # Codebook / code_dim
        ("NC=1 N=256 K=512 D_code=64", dict(num_codes=512, code_dim=64)),
        ("NC=1 N=256 K=4096 D_code=256", dict(num_codes=4096, code_dim=256)),
    ]

    # Each config at a couple of batch sizes
    for label, overrides in configs:
        for BS in [48, 64, 72]:
            r = bench(BS, overrides, warmup=2, measure=5, label=label)
            if r.get("oom"):
                print(f"{label:<32s}  {BS:>3d}  {'OOM':>8s}")
            else:
                print(f"{label:<32s}  {BS:>3d}  {r['tok_s']:>7,.0f}  "
                      f"{r['step_ms']:>5.0f}ms  {r['peak_gb']:>4.1f}G  "
                      f"{r['total_mb']:>5.1f}M")


if __name__ == "__main__":
    main()
