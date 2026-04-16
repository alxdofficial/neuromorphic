"""Deep profile of the conv-grid modulator to identify exact bottlenecks.

Four angles:
1. torch.profiler trace: per-op CUDA/CPU time, kernel names, tensor-core vs
   fallback use.
2. Isolated component timing: encoder, decoder, memory-step primitives,
   observation tensor build, LM. CUDA events for microsecond resolution.
3. Per-layer breakdown of the conv stack: which conv stages dominate.
4. Theoretical vs measured FLOPS: are we compute-bound, memory-bandwidth-
   bound, or dispatch-bound?
"""

import sys
import time
import torch

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def event_ms(fn, reps=3):
    """Time a callable over reps runs using CUDA events."""
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    return min(times)  # best-of, skips variance


def run_full_profile(bs=8, warmup=2, measure=3):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = Config.tier_a()
    cfg.vocab_size = 32000
    cfg.validate()

    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    lm_p = sum(p.numel() for p in model.lm.parameters()) / 1e6
    mem_p = sum(p.numel() for p in model.memory.parameters()) / 1e6
    mod_p = sum(p.numel() for p in model.memory.modulator.parameters()) / 1e6
    dec_p = sum(p.numel() for p in model.memory.decoder.parameters()) / 1e6

    print("=" * 70)
    print("CONV-GRID MODULATOR — DEEP PROFILE")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"bf16 peak: ~165 TFLOPS (theoretical 4090 tensor-core)")
    print()
    print(f"Params: total={total:.1f}M  LM={lm_p:.1f}M  Mem={mem_p:.1f}M "
          f"(modulator={mod_p:.1f}M, decoder={dec_p:.1f}M)")
    print(f"Config: BS={bs}, T={cfg.T}, N={cfg.N_total}, NC_pools={cfg.NC_pools}")
    print(f"  Encoder: C_h={cfg.conv_channels}, L={cfg.conv_layers}, "
          f"k={cfg.conv_kernel}")
    print(f"  Decoder: seed=[{cfg.decoder_seed_spatial}, {cfg.decoder_seed_spatial}, "
          f"{cfg.decoder_seed_channels}]")
    print(f"  Codebook: K={cfg.num_codes}, D_code={cfg.code_dim}")
    print(f"  mod_interval={cfg.modulation_interval}, tbptt={cfg.tbptt_block}")
    print(f"  checkpoint_memory={cfg.checkpoint_memory}, "
          f"checkpoint_decoder={cfg.checkpoint_decoder}")
    print()

    def step():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model.forward_chunk(input_ids, target_ids=target_ids)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        opt.step()
        model.detach_states()
        return r

    # ---- Warmup (especially important for torch.compile) ----
    print("Warmup...")
    for i in range(warmup):
        t0 = time.time()
        step()
        _sync()
        print(f"  step {i}: {time.time() - t0:.2f}s")
    print()

    # ---- Overall: step time & throughput ----
    _sync()
    t0 = time.time()
    for _ in range(measure):
        step()
    _sync()
    step_ms = (time.time() - t0) / measure * 1000
    tok_s = bs * cfg.T * measure / (time.time() - t0 + 1e-9)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"Step time: {step_ms:.1f} ms")
    print(f"Throughput: {tok_s:,.0f} tok/s")
    print(f"Peak VRAM: {peak_gb:.2f} GB")
    print()

    # ---- Isolated section timing via CUDA events ----
    print("=" * 70)
    print("ISOLATED SECTION TIMING (CUDA events, best-of-3)")
    print("=" * 70)

    # Ensure memory state is initialized
    model.memory.initialize_states(bs, device)

    def iso_build_input():
        mem = model.memory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            received = torch.matmul(mem.W, mem.msg)
            role_id = mem.role_id.to(device)
            _ = mem.modulator.build_input(
                mem.h, mem.msg, received, mem.W, mem.hebbian, mem.decay,
                mem.s_mem_live, mem.s_mem_ema_fast, role_id)

    def iso_encoder_fwd():
        mem = model.memory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            received = torch.matmul(mem.W, mem.msg)
            role_id = mem.role_id.to(device)
            _ = mem.modulator(
                mem.h, mem.msg, received, mem.W, mem.hebbian, mem.decay,
                mem.s_mem_live, mem.s_mem_ema_fast, role_id)

    def iso_decoder_fwd():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = torch.randn(bs, cfg.code_dim, device=device)
            _ = model.memory.decoder(emb)

    def iso_wmsg():
        mem = model.memory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = torch.matmul(mem.W, mem.msg)

    def iso_hebbian():
        mem = model.memory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = torch.matmul(mem.msg, mem.msg.transpose(-1, -2))

    def iso_state_mlp():
        mem = model.memory
        received = torch.matmul(mem.W, mem.msg)
        inp = torch.cat([received, mem.h], dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h1 = torch.tanh(torch.nn.functional.linear(
                inp.reshape(-1, 2*cfg.D_n), mem.state_w1, mem.state_b1))
            _ = torch.tanh(torch.nn.functional.linear(
                h1, mem.state_w2, mem.state_b2))

    def iso_msg_mlp():
        mem = model.memory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h1 = torch.tanh(torch.nn.functional.linear(
                mem.h.reshape(-1, cfg.D_n), mem.msg_w1, mem.msg_b1))
            _ = torch.tanh(torch.nn.functional.linear(
                h1, mem.msg_w2, mem.msg_b2))

    def iso_full_fwd():
        input_ids = torch.randint(1, cfg.vocab_size, (bs, cfg.T), device=device)
        target_ids = torch.randint(0, cfg.vocab_size, (bs, cfg.T), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.forward_chunk(input_ids, target_ids=target_ids)

    def iso_full_step():
        step()

    sections = [
        ("build observation tensor", iso_build_input, "once/fire"),
        ("encoder forward", iso_encoder_fwd, "once/fire"),
        ("decoder forward", iso_decoder_fwd, "once/fire"),
        ("W @ msg", iso_wmsg, "once/token"),
        ("msg @ msgᵀ (hebbian)", iso_hebbian, "once/token"),
        ("state MLP", iso_state_mlp, "once/token"),
        ("msg MLP", iso_msg_mlp, "once/token"),
        ("full forward_chunk", iso_full_fwd, "per segment"),
        ("full step (fwd+bwd+opt)", iso_full_step, "per segment"),
    ]

    for name, fn, freq in sections:
        try:
            t = event_ms(fn, reps=3)
            print(f"  {name:40s} {t:8.3f} ms   ({freq})")
        except Exception as e:
            print(f"  {name:40s} FAILED: {e}")
    print()

    # ---- Per-stage encoder timing ----
    print("=" * 70)
    print("PER-ENCODER-STAGE TIMING (single fire)")
    print("=" * 70)
    mod = model.memory.modulator
    mem = model.memory

    # Build the actual conv input to hand to each stage
    with torch.autocast("cuda", dtype=torch.bfloat16):
        received = torch.matmul(mem.W, mem.msg)
        role_id = mem.role_id.to(device)
        grid_input = mod.build_input(
            mem.h, mem.msg, received, mem.W, mem.hebbian, mem.decay,
            mem.s_mem_live, mem.s_mem_ema_fast, role_id)

    # Run stem + each stage separately
    with torch.autocast("cuda", dtype=torch.bfloat16):
        w_dt = mod.stem.weight.dtype
        x = grid_input.to(w_dt)

        def stem():
            import torch.nn.functional as F_
            return F_.gelu(mod.stem_norm(mod.stem(x)))

        t = event_ms(stem, reps=3)
        print(f"  stem (Conv2d k={cfg.conv_kernel}, "
              f"{cfg.mod_in_channels}→{max(cfg.conv_channels//4, 16)}): "
              f"{t:.3f} ms   "
              f"(input [{bs}, {cfg.mod_in_channels}, {cfg.N_total}, {cfg.N_total}])")

        h = stem()
        for i, stage in enumerate(mod.stages):
            def fn(s=stage, h=h):
                return s(h)
            t = event_ms(fn, reps=3)
            out = stage(h)
            print(f"  stage {i} ({stage.dw.in_channels}→{stage.pw.out_channels}, "
                  f"spatial {h.shape[-1]}→{out.shape[-1]}): "
                  f"{t:.3f} ms")
            h = out
    print()

    # ---- Per-decoder-stage timing ----
    print("=" * 70)
    print("PER-DECODER-STAGE TIMING (single fire)")
    print("=" * 70)
    dec = model.memory.decoder
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = torch.randn(bs, cfg.code_dim, device=device)
        S = dec.seed_spatial
        x = dec.init_proj(emb).reshape(bs, dec.seed_channels, S, S)

        def init_proj():
            e = torch.randn(bs, cfg.code_dim, device=device)
            return dec.init_proj(e).reshape(bs, dec.seed_channels, S, S)

        t = event_ms(init_proj, reps=3)
        print(f"  init_proj (Linear {cfg.code_dim}→{dec.seed_channels * S * S}): "
              f"{t:.3f} ms")

        for i, stage in enumerate(dec.stages):
            def fn(s=stage, x=x):
                return s(x)
            t = event_ms(fn, reps=3)
            out = stage(x)
            print(f"  stage {i} (spatial {x.shape[-1]}→{out.shape[-1]}, "
                  f"C {x.shape[1]}→{out.shape[1]}): {t:.3f} ms")
            x = out
    print()

    # ---- torch.profiler per-op ----
    print("=" * 70)
    print("TORCH.PROFILER: TOP OPS BY CUDA TIME (3 steps)")
    print("=" * 70)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(3):
            step()
        _sync()

    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25))

    # ---- Compute-bound or dispatch-bound ----
    print("=" * 70)
    print("FLOP ACCOUNTING")
    print("=" * 70)
    # Approximate theoretical FLOPs for one step
    N, C_h, k, L = cfg.N_total, cfg.conv_channels, cfg.conv_kernel, cfg.conv_layers
    stem_ch_in = cfg.mod_in_channels
    stem_ch_out = max(C_h // 4, 16)

    # Pyramid encoder: stem at full N×N, each stage halves spatial dim.
    # Channels: [stem_out, C_h/2, C_h, C_h, C_h]
    def dwsep_flops(N_s, c_in, c_out, k_):
        # depthwise: N_s² * c_in * k²
        # pointwise: N_s² * c_in * c_out
        return N_s * N_s * (c_in * k_ * k_ + c_in * c_out)

    # Stem: dense conv
    stem_flops = N * N * stem_ch_in * stem_ch_out * cfg.conv_kernel ** 2
    stage_flops = []
    spatial = N
    ladder = [stem_ch_out, max(C_h // 2, 32), C_h, C_h, C_h]
    for i in range(1, len(ladder)):
        spatial = spatial // 2
        stage_flops.append(dwsep_flops(spatial, ladder[i-1], ladder[i], k))
    encoder_flops_per_fire = stem_flops + sum(stage_flops)

    n_fires = cfg.T // cfg.modulation_interval
    encoder_flops = encoder_flops_per_fire * bs * n_fires
    encoder_flops_tf = encoder_flops / 1e12

    # Memory step compute (per token): W@msg, hebbian, state MLP, msg MLP
    wmsg_flops = bs * N * N * cfg.D_n  # per token
    heb_flops = bs * N * N * cfg.D_n  # per token
    state_flops = bs * N * (2 * cfg.D_n * cfg.state_mlp_hidden +
                             cfg.state_mlp_hidden * cfg.D_n)
    msg_flops = bs * N * (cfg.D_n * cfg.msg_mlp_hidden +
                           cfg.msg_mlp_hidden * cfg.D_n)
    memstep_flops = (wmsg_flops + heb_flops + state_flops + msg_flops) * cfg.T
    memstep_flops_tf = memstep_flops / 1e12

    # LM: approximate scan + lm_head
    lm_scan_flops = bs * cfg.T * cfg.L_total * 2 * cfg.D * cfg.d_inner * 2  # rough
    lm_head_flops = bs * cfg.T * cfg.D_embed * cfg.vocab_size
    lm_flops_tf = (lm_scan_flops + lm_head_flops) / 1e12

    total_fwd_tf = encoder_flops_tf + memstep_flops_tf + lm_flops_tf
    total_fwd_bwd_tf = total_fwd_tf * 3  # rough 2x backward + checkpoint replay

    step_time_s = step_ms / 1000
    achieved_tf_per_s = total_fwd_bwd_tf / step_time_s

    print(f"Theoretical FLOPs per step (forward):")
    print(f"  Encoder (pyramid DW-sep):  {encoder_flops_tf:>8.2f} TFlops")
    print(f"  Memory step (128 tokens):  {memstep_flops_tf:>8.2f} TFlops")
    print(f"  LM (scan + head):          {lm_flops_tf:>8.2f} TFlops")
    print(f"  TOTAL forward:             {total_fwd_tf:>8.2f} TFlops")
    print(f"  TOTAL fwd+bwd (est):       {total_fwd_bwd_tf:>8.2f} TFlops")
    print()
    print(f"Step time:                {step_time_s*1000:>8.1f} ms")
    print(f"Achieved throughput:      {achieved_tf_per_s:>8.1f} TFlops/sec")
    print(f"4090 peak (bf16):         {165:>8.1f} TFlops/sec")
    print(f"Efficiency:               {achieved_tf_per_s/165*100:>8.1f}%")
    print()
    print("If efficiency << 30%: dispatch-bound or memory-bandwidth-bound.")
    print("If efficiency ~ 30%+: compute-bound.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=3)
    args = p.parse_args()
    run_full_profile(args.bs, args.warmup, args.measure)
