"""Focused profile of the current attention-neuromod architecture.

Reports:
  1. CUDA-event timing of each hot primitive (at BS=64, T=128).
  2. torch.profiler top ops by CUDA time (forward only and full step).
  3. FLOP accounting and efficiency vs RTX 4090 bf16 peak.
"""

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def event_ms(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return min(times)


def main(bs: int = 64):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = Config.tier_a()
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    lm_p = sum(p.numel() for p in model.lm.parameters()) / 1e6
    mem_p = sum(p.numel() for p in model.memory.parameters()) / 1e6

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Params: total={total:.1f}M  LM={lm_p:.1f}M  Mem={mem_p:.1f}M")
    print(f"Config: BS={bs}, T={cfg.T}, NC={cfg.N_cells}, Nc={cfg.neurons_per_cell}, "
          f"D_n={cfg.D_n}, D={cfg.D}, d_inner={cfg.d_inner}")
    print(f"        mod_interval={cfg.modulation_interval}, tbptt={cfg.tbptt_block}")
    print()

    # Warmup full step (torch.compile)
    def full_step():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        opt.step()
        model.detach_states()

    print("Warming up (compile)...")
    for i in range(3):
        t = time.time()
        full_step()
        torch.cuda.synchronize()
        print(f"  warmup {i}: {time.time()-t:.2f}s")

    # Measure full step
    print()
    print("Measuring full step...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        full_step()
    torch.cuda.synchronize()
    step_ms = (time.time() - t0) * 100
    tok_s = bs * cfg.T * 10 / (time.time() - t0)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Step time: {step_ms:.1f} ms   Throughput: {tok_s:,.0f} tok/s   VRAM: {peak_gb:.1f} GB")
    print()

    # Ensure memory initialized
    model.memory.initialize_states(bs, device)

    # Component isolation — wrap each in autocast and measure with CUDA events.
    mem = model.memory

    def iso_wmsg():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = torch.matmul(mem.W, mem.msg)

    def iso_hebbian():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = torch.matmul(mem.msg, mem.msg.transpose(-1, -2))

    def iso_state_mlp():
        # LIF update: h = tanh(decay*h + (1-decay)*received)
        received = torch.matmul(mem.W, mem.msg)
        decay_gate = mem.decay.unsqueeze(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = torch.tanh(decay_gate * mem.h + (1.0 - decay_gate) * received)

    def iso_msg_mlp():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flat = mem.h.reshape(-1, cfg.D_n)
            h1 = torch.tanh(F.linear(flat, mem.msg_w1, mem.msg_b1))
            _ = torch.tanh(F.linear(h1, mem.msg_w2, mem.msg_b2))

    def iso_modulator():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            received = torch.matmul(mem.W, mem.msg)
            _ = mem.modulator(mem.h, mem.msg, received, mem.W, mem.hebbian, mem.decay,
                              mem.s_mem_live, mem.s_mem_ema_fast, mem.role_id)

    def iso_decoder():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Decoder API: [BS, NC, D_code] + cell_emb [NC, d_cell]
            emb = torch.randn(bs, cfg.N_cells, cfg.code_dim, device=device)
            _ = mem.decoder(emb, mem.modulator.cell_emb)

    def iso_lm_lower():
        ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.lm.forward_scan_lower(ids)

    def iso_lm_upper():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.randn(bs, cfg.T, cfg.D, device=device, dtype=torch.bfloat16)
            _ = model.lm.forward_scan_upper(x)

    def iso_lm_head():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.randn(bs, cfg.T, cfg.D, device=device, dtype=torch.bfloat16)
            _ = model.lm.forward_output(x)

    print("=" * 70)
    print(f"ISOLATED SECTION TIMING (BS={bs}, 1 fire)")
    print("=" * 70)
    n_tok = cfg.T
    n_mod = cfg.T // cfg.modulation_interval

    sections = [
        ("W @ msg (bmm)",           iso_wmsg,      n_tok,  "per-token"),
        ("msg @ msgᵀ (hebbian)",    iso_hebbian,   n_tok,  "per-token"),
        ("state MLP (shared)",      iso_state_mlp, n_tok,  "per-token"),
        ("msg MLP (shared)",        iso_msg_mlp,   n_tok,  "per-token"),
        ("modulator (full fwd)",    iso_modulator, n_mod,  "per-mod-event"),
        ("decoder (full fwd)",      iso_decoder,   n_mod,  "per-mod-event"),
        ("LM scan lower (T=128)",   iso_lm_lower,  1,      "per-segment"),
        ("LM scan upper (T=128)",   iso_lm_upper,  1,      "per-segment"),
        ("LM head (T=128)",         iso_lm_head,   1,      "per-segment"),
    ]

    print(f"{'name':<28s} {'one-fire':>10s} {'calls':>6s} {'total':>10s}  {'note'}")
    print("-" * 75)
    for name, fn, calls, note in sections:
        try:
            t = event_ms(fn, reps=5)
            total_ms = t * calls
            print(f"{name:<28s} {t:>8.3f}ms {calls:>6d} {total_ms:>8.2f}ms  ({note})")
        except Exception as e:
            print(f"{name:<28s} FAILED: {type(e).__name__}: {str(e)[:40]}")
    print()

    # Now torch.profiler for full step with shapes
    print("=" * 70)
    print("TORCH.PROFILER TOP OPS (1 full step)")
    print("=" * 70)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        full_step()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=64)
    args = p.parse_args()
    main(bs=args.bs)
