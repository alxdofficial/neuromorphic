"""Find max safe batch size and check for memory leaks."""

import torch
import gc
import time

from src.v8.config import V8Config
from src.v8.model import V8Model
from src.v8.trainer import V8Trainer


class FakeBatch:
    def __init__(self, bs, T, device):
        self.input_ids = torch.randint(0, 32000, (bs, T), device=device)
        self.target_ids = torch.randint(0, 32000, (bs, T), device=device)
        self.prev_token = torch.randint(0, 32000, (bs,), device=device)


def try_bs(bs, device, steps=3):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg = V8Config.tier_a(vocab_size=32000)
        cfg.validate()
        model = V8Model(cfg).to(device).to(torch.bfloat16)
        lm_opt = torch.optim.AdamW(model.lm.parameters(), lr=3e-4, fused=True)
        nm_opt = torch.optim.Adam(model.neuromod.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(lm_opt, lambda _: 1.0)
        trainer = V8Trainer(
            model=model, lm_optimizer=lm_opt, neuromod_optimizer=nm_opt,
            scheduler=sched, dataloader=iter([]), config=cfg, device=device,
            use_memory=True,
        )
        for i in range(steps):
            batch = FakeBatch(bs, cfg.T, device)
            trainer.train_chunk(batch)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  BS={bs}: OK, peak={peak:.2f}GB")
        del model, lm_opt, nm_opt, trainer
        return True, peak
    except torch.cuda.OutOfMemoryError:
        print(f"  BS={bs}: OOM")
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0


def check_memory_leaks(bs, device, steps=50):
    """Run many steps and check if memory grows over time."""
    torch.cuda.empty_cache()
    gc.collect()

    cfg = V8Config.tier_a(vocab_size=32000)
    cfg.validate()
    model = V8Model(cfg).to(device).to(torch.bfloat16)
    lm_opt = torch.optim.AdamW(model.lm.parameters(), lr=3e-4, fused=True)
    nm_opt = torch.optim.Adam(model.neuromod.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(lm_opt, lambda _: 1.0)
    trainer = V8Trainer(
        model=model, lm_optimizer=lm_opt, neuromod_optimizer=nm_opt,
        scheduler=sched, dataloader=iter([]), config=cfg, device=device,
        use_memory=True,
    )

    print(f"\nMemory leak check: BS={bs}, {steps} steps")
    mem_samples = []

    for i in range(steps):
        batch = FakeBatch(bs, cfg.T, device)
        trainer.train_chunk(batch)

        if i % 5 == 0:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            mem_samples.append(allocated)
            if i % 10 == 0:
                print(f"  step {i:3d}: allocated={allocated:.3f}GB, reserved={reserved:.3f}GB")

    # Check for growth
    first_5 = sum(mem_samples[:5]) / 5
    last_5 = sum(mem_samples[-5:]) / 5
    growth = last_5 - first_5
    print(f"\n  First 5 avg: {first_5:.3f}GB")
    print(f"  Last 5 avg:  {last_5:.3f}GB")
    print(f"  Growth:      {growth*1000:.1f}MB")
    if abs(growth) < 0.01:
        print("  PASS: No memory leak detected")
    else:
        print(f"  WARNING: Memory grew by {growth*1000:.1f}MB over {steps} steps")

    del model, lm_opt, nm_opt, trainer


def main():
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({vram:.1f}GB)")

    # Find max BS
    print("\nFinding max batch size...")
    max_bs = 4
    max_peak = 0
    for bs in [16, 14, 12, 10, 8, 6, 4]:
        ok, peak = try_bs(bs, device)
        if ok:
            max_bs = bs
            max_peak = peak
            break

    print(f"\nMax safe BS={max_bs}, peak VRAM={max_peak:.2f}GB "
          f"({max_peak/vram*100:.0f}% of {vram:.1f}GB)")

    # Memory leak check at a safe BS
    safe_bs = min(max_bs, 8)
    check_memory_leaks(safe_bs, device, steps=50)


if __name__ == "__main__":
    main()
