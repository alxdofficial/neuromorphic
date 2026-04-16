"""Detailed breakdown of where forward+backward time goes.

Uses torch.profiler for per-op timings, plus manual section timers for
architecturally-meaningful buckets (encoder, decoder, memory step, LM).
"""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_profiler(bs, warmup=2, measure=5, trace_out=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    cfg = Config.tier_a()
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                             fused=(device.type == "cuda"))

    total = sum(p.numel() for p in model.parameters()) / 1e6
    lm_p = sum(p.numel() for p in model.lm.parameters()) / 1e6
    mem_p = sum(p.numel() for p in model.memory.parameters()) / 1e6
    mod_p = sum(p.numel() for p in model.memory.modulator.parameters()) / 1e6
    dec_p = sum(p.numel() for p in model.memory.decoder.parameters()) / 1e6
    print(f"Params: total={total:.1f}M  LM={lm_p:.1f}M  Mem={mem_p:.1f}M  "
          f"(Mod={mod_p:.1f}M  Dec={dec_p:.1f}M)")
    print(f"Config: BS={bs}, T={cfg.T}, N={cfg.N_total}, C_h={cfg.conv_channels}, "
          f"L={cfg.conv_layers}, k={cfg.conv_kernel}, K={cfg.num_codes}, "
          f"D_code={cfg.code_dim}")
    print(f"  ckpt_memory={cfg.checkpoint_memory}  ckpt_decoder={cfg.checkpoint_decoder}")

    def step():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=device.type == "cuda"):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.detach_states()
        return r["loss"].item()

    # Warmup — compile, allocator warmup.
    print("\nWarming up...")
    for _ in range(warmup):
        step()
    _sync()

    # --- torch.profiler per-op breakdown ---
    print("\nProfiling...")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        _sync()
        t0 = time.time()
        for _ in range(measure):
            step()
        _sync()
        elapsed = time.time() - t0

    tokens = bs * cfg.T * measure
    tok_s = tokens / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

    print(f"\n=== Overall ===")
    print(f"Step time:  {elapsed / measure * 1000:.1f} ms/step")
    print(f"Throughput: {tok_s:,.0f} tok/s")
    print(f"Peak VRAM:  {peak_gb:.2f} GB")

    print(f"\n=== Top 25 ops by CUDA time ===")
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))

    if trace_out:
        prof.export_chrome_trace(trace_out)
        print(f"\nChrome trace written to {trace_out}")


def run_section_timing(bs, iterations=5):
    """Explicit per-section timing with CUDA events. Complements torch.profiler
    by giving us the big-bucket wall time (encoder vs decoder vs LM etc.)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.tier_a()
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device.type == "cuda"))

    # Warmup
    for _ in range(2):
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=device.type == "cuda"):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        opt.step()
        model.detach_states()
    _sync()

    # Isolated section timings using CUDA events.
    def event_time(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)  # ms

    # A. Isolated encoder call
    def isolated_encoder():
        mem = model.memory
        if not mem._initialized:
            mem.initialize_states(bs, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            received = torch.matmul(mem.W, mem.msg)
            role_id = mem.role_id.to(device)
            logits = mem.modulator(
                mem.h, mem.msg, received, mem.W, mem.hebbian, mem.decay,
                mem.s_mem_live, mem.s_mem_ema_fast, role_id)
        return logits

    # B. Isolated decoder call
    def isolated_decoder():
        emb = torch.randn(bs, cfg.code_dim, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            return model.memory.decoder(emb)

    # C. Isolated W @ msg
    def isolated_wmsg():
        mem = model.memory
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            return torch.matmul(mem.W, mem.msg)

    # D. Isolated hebbian msg @ msg.T
    def isolated_hebbian():
        mem = model.memory
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            return torch.matmul(mem.msg, mem.msg.transpose(-1, -2))

    # E. Full model forward_chunk
    def full_forward():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            return model.forward_chunk(input_ids, target_ids=target_ids)

    # F. Full step (forward + backward + step)
    def full_step():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        opt.step()
        model.detach_states()

    print(f"\n=== Section timing (BS={bs}, averaged over {iterations} runs) ===")
    print(f"(Isolated single calls; not per-segment amortized.)")
    sections = [
        ("1× encoder call (single modulation fire)", isolated_encoder),
        ("1× decoder call (single modulation fire)", isolated_decoder),
        ("1× W @ msg (per-token matmul)", isolated_wmsg),
        ("1× msg @ msgᵀ (hebbian update)", isolated_hebbian),
        ("Full forward_chunk (T=128)", full_forward),
        ("Full step (forward + backward + opt)", full_step),
    ]
    for name, fn in sections:
        times = [event_time(fn) for _ in range(iterations)]
        mean_ms = sum(times) / len(times)
        print(f"  {name:50s}  {mean_ms:8.2f} ms")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=5)
    p.add_argument("--trace", type=str, default=None)
    p.add_argument("--sections", action="store_true",
                   help="Also run explicit per-section timing")
    args = p.parse_args()
    run_profiler(args.bs, args.warmup, args.measure, trace_out=args.trace)
    if args.sections:
        run_section_timing(args.bs, iterations=5)
