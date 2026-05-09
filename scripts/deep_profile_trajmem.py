"""Deep profile of trajectory-memory Phase 1 step at the production config.

Four lenses, each answering a different question:

  1. Whole-step timing — overall tok/s, peak VRAM, ms/iter at the chosen BS.
  2. Per-component CUDA-event timing — how the step time splits between
     Llama forward, read_module, write_module, MemInjectLayer cross-attn,
     manifold writes, backward, and optimizer step.
  3. torch.profiler top ops — which CUDA kernels dominate; exposes
     dispatch overhead, fallback (non-tensorcore) ops, and matmul shape
     hot spots.
  4. FLOP accounting — theoretical compute for memory ops vs Llama;
     diagnoses whether we're compute-bound, bandwidth-bound, or
     dispatch-bound (which indicates fixable inefficiency).

Usage:
    PYTHONPATH=. python scripts/deep_profile_trajmem.py --bs 2 --compile
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training import Phase1Trainer, build_optimizer


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def event_ms(fn, reps: int = 5) -> float:
    """Time a callable via CUDA events. Returns best-of-reps in ms."""
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    return min(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile model.forward_window")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=10)
    ap.add_argument("--profile-steps", type=int, default=3,
                    help="Steps to capture in torch.profiler")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = getattr(TrajMemConfig, args.config_tier)()
    T_chunk = cfg.D * cfg.T_window

    model = IntegratedLM(cfg, model_name=args.model, attach_lm=True).to(device)
    if args.compile:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )
    optimizer = build_optimizer(model, lr_memory=3e-4, lr_adapter=1e-4)
    trainer = Phase1Trainer(model, optimizer, scheduler=None, grad_clip=1.0)

    vocab = model.llama.config.vocab_size

    # Param breakdown.
    groups: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        sub = name.split(".")[1] if "." in name else "<root>"
        key = f"{top}.{sub}" if top in ("host", "llama") else top
        groups.setdefault(key, [0, 0])
        groups[key][0] += p.numel()
        groups[key][1] += p.numel() if p.requires_grad else 0
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 76)
    print("TRAJECTORY-MEMORY DEEP PROFILE")
    print("=" * 76)
    print(f"Device:  {torch.cuda.get_device_name()} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"Tier:    {args.config_tier}")
    print(f"Config:  N={cfg.N}, K_max={cfg.K_max_neighbors}, "
          f"J={cfg.J}, K_read={cfg.K_read}, K_write={cfg.K_write}")
    print(f"         D={cfg.D}, T_window={cfg.T_window}, chunk={T_chunk}")
    print(f"         d_lm={cfg.d_lm}, D_concept={cfg.D_concept}, "
          f"bridge_hidden={cfg.bridge_hidden}, inject_layer={cfg.inject_layer}")
    print(f"Compile: {'on' if args.compile else 'off'}")
    print(f"BS={args.bs}, T_chunk={T_chunk}")
    print()
    print(f"Params: total={total/1e9:.3f}B  trainable={train/1e6:.2f}M")
    for k in sorted(groups):
        tot, tr = groups[k]
        print(f"  {k:<35} {tot/1e6:>9.2f}M total  {tr/1e6:>8.2f}M trainable")
    print()

    chunk = torch.randint(0, vocab, (args.bs, T_chunk), device=device)

    # ── Warmup ────────────────────────────────────────────────────────
    print("Warmup...")
    for i in range(args.warmup):
        t0 = time.time()
        trainer.step_wave1(chunk)
        _sync()
        print(f"  step {i}: {(time.time() - t0)*1000:.1f}ms")
    print()

    # ── 1. Whole-step timing ──────────────────────────────────────────
    print("=" * 76)
    print("1. WHOLE-STEP TIMING")
    print("=" * 76)
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.time()
    for _ in range(args.measure):
        trainer.step_wave1(chunk)
    _sync()
    elapsed = time.time() - t0
    step_ms = elapsed / args.measure * 1000
    tps = args.bs * T_chunk * args.measure / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Step time:  {step_ms:.1f} ms/iter")
    print(f"  Throughput: {tps/1000:.2f}k tok/s")
    print(f"  Peak VRAM:  {peak_gb:.2f} GB")
    print()

    # ── 2. Component isolation via CUDA events ───────────────────────
    print("=" * 76)
    print("2. PER-COMPONENT TIMING (CUDA events, best-of-5)")
    print("=" * 76)

    # Build representative inputs for each stage.
    BS = args.bs
    T = cfg.T_window
    D = cfg.D
    prev_states = model.manifold.reset_states(batch_size=BS)
    prev_window_hiddens = torch.zeros(
        BS, T, cfg.d_lm, dtype=prev_states.dtype, device=device,
    )
    lm_input_ids = torch.randint(0, vocab, (BS, T), device=device)

    # 2a. Whole forward_window (one window).
    def fw_one_window():
        with torch.no_grad():
            return model.forward_window(
                lm_input_ids, prev_window_hiddens, prev_states,
                hard_routing=True,
            )

    # 2b. Llama forward only — keep MemInjectLayer in the path but feed it
    # a zero memory readout so the bridge projection runs but adds nothing.
    # `memory_fn=None` would trigger MemInjectLayer's safety assertion
    # ("called without memory_fn but scale is not all-zero").
    inject_layer = model._mem_inject_layer()
    saved_memory_fn = inject_layer.memory_fn

    def _zero_memory_fn(h_mem):
        # h_mem: [BS, T, D_concept] — return zeros of same shape so the
        # bridge's W_out + scaled-add contributes zero residual delta.
        return torch.zeros_like(h_mem)

    inject_layer.memory_fn = _zero_memory_fn

    def llama_only_fwd():
        with torch.no_grad():
            return model.llama(
                input_ids=lm_input_ids,
                output_hidden_states=False,
                use_cache=False,
            )

    # 2c. read_module only.
    def read_only():
        with torch.no_grad():
            return model.read_module(
                prev_window_hiddens.to(prev_states.dtype),
                prev_states, model.manifold, hard=True,
            )

    # Need to compute current_hiddens + surprise to feed write_module.
    surprise_dummy = torch.zeros(BS, dtype=prev_states.dtype, device=device)
    current_hiddens_dummy = torch.zeros(
        BS, T, cfg.d_lm, dtype=prev_states.dtype, device=device,
    )

    # 2d. write_module only.
    def write_only():
        with torch.no_grad():
            return model.write_module(
                current_hiddens_dummy, surprise_dummy, prev_states,
                model.manifold, hard=True,
            )

    # Restore memory_fn
    def with_memory_fn(fn):
        # Need a non-None memory_fn for the actual injection path.
        # Construct a fresh read_visited and wire it up like forward_window does.
        with torch.no_grad():
            read_visited, _ = model.read_module(
                prev_window_hiddens.to(prev_states.dtype),
                prev_states, model.manifold, hard=True,
            )
        model._mem_inject_layer().memory_fn = model._build_memory_fn(read_visited)
        try:
            return fn()
        finally:
            model._mem_inject_layer().memory_fn = None

    def llama_with_memory_fwd():
        return with_memory_fn(lambda: model.llama(
            input_ids=lm_input_ids,
            output_hidden_states=False,
            use_cache=False,
        ))

    sections = [
        ("Llama forward (no memory_fn, identity adapter)", llama_only_fwd, "1× per window"),
        ("Llama forward + MemInjectLayer cross-attn", llama_with_memory_fwd, "1× per window"),
        ("read_module only", read_only, "1× per window"),
        ("write_module only", write_only, "1× per window"),
        ("forward_window (full, no_grad)", fw_one_window, "1× per window"),
    ]

    print(f"  Per-window costs (chunk has D={D} windows + 1 backward):")
    for name, fn, freq in sections:
        try:
            t = event_ms(fn, reps=5)
            print(f"  {name:<55} {t:7.2f} ms   ({freq})")
        except Exception as e:
            print(f"  {name:<55} FAILED: {e}")
    print()

    # 2e. Backward + optimizer step (estimated as full step minus 4× forward_window).
    fw_one = event_ms(fw_one_window, reps=5)
    full_step = event_ms(lambda: trainer.step_wave1(chunk), reps=5)
    fw_total = fw_one * D  # D forward windows in a chunk
    bw_opt = full_step - fw_total
    print(f"  Estimated backward + opt step (full - {D}×fw_one):")
    print(f"    full step:        {full_step:7.2f} ms")
    print(f"    {D}× forward_window: {fw_total:7.2f} ms")
    print(f"    bw + opt:         {bw_opt:7.2f} ms ({bw_opt/full_step*100:.1f}% of step)")
    print()
    inject_layer.memory_fn = saved_memory_fn

    # ── 3. torch.profiler top ops ────────────────────────────────────
    print("=" * 76)
    print(f"3. TORCH.PROFILER: TOP OPS ({args.profile_steps} steps, sorted by CUDA time)")
    print("=" * 76)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(args.profile_steps):
            trainer.step_wave1(chunk)
        _sync()
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=60,
    ))
    print()

    # ── 4. FLOP accounting (rough) ───────────────────────────────────
    print("=" * 76)
    print("4. FLOP ACCOUNTING (rough; identifies compute-vs-bandwidth-bound)")
    print("=" * 76)

    n_layers = model.llama.config.num_hidden_layers
    d_lm = cfg.d_lm
    # Llama: per chunk, T_chunk tokens * n_layers * (4·d² QKV + 2·d² attn + 8·d·d_ffn FFN)
    # Llama-3.2-1B: d_ffn=8192, d=2048, so per-token-per-layer ≈ 4*d² + 2*d²·(T/2) + 8*d*d_ffn
    # We use the FFN dominant term: 8 * d_lm * d_ffn = 8 * 2048 * 8192 = 134M flops/token/layer
    d_ffn = model.llama.config.intermediate_size
    llama_flops_per_tok_per_layer = 8 * d_lm * d_ffn + 4 * d_lm * d_lm
    llama_flops = (
        BS * T_chunk * n_layers * llama_flops_per_tok_per_layer
    )

    # Memory ops, per chunk:
    # - read_module: D windows × (J × K_read × D² × 4) MLP + (J × K_read × N × D) attn-over-manifold
    # - write_module: D windows × similar
    # - scatter_mean state update: D windows × (BS × J × K_write × D)  — bandwidth-bound
    D_c = cfg.D_concept
    read_flops = D * BS * cfg.J * cfg.K_read * (4 * D_c * D_c + 2 * cfg.N * D_c)
    write_flops = D * BS * cfg.J * cfg.K_write * (5 * D_c * D_c + 2 * cfg.N * D_c)
    bridge_flops = D * BS * T * (2 * d_lm * cfg.bridge_hidden + 2 * cfg.bridge_hidden * d_lm)

    print(f"  Per-chunk theoretical FLOPs (forward only):")
    print(f"    Llama:        {llama_flops/1e9:>10.1f} GFLOPS  "
          f"({100*llama_flops/(llama_flops+read_flops+write_flops+bridge_flops):.1f}% of total)")
    print(f"    Bridge MLP:   {bridge_flops/1e9:>10.1f} GFLOPS  "
          f"({100*bridge_flops/(llama_flops+read_flops+write_flops+bridge_flops):.1f}% of total)")
    print(f"    read_module:  {read_flops/1e9:>10.1f} GFLOPS  "
          f"({100*read_flops/(llama_flops+read_flops+write_flops+bridge_flops):.1f}% of total)")
    print(f"    write_module: {write_flops/1e9:>10.1f} GFLOPS  "
          f"({100*write_flops/(llama_flops+read_flops+write_flops+bridge_flops):.1f}% of total)")
    total_flops_fwd = llama_flops + read_flops + write_flops + bridge_flops
    print(f"    TOTAL fwd:    {total_flops_fwd/1e9:>10.1f} GFLOPS")

    # Triple for fwd+bwd (rough rule of thumb: bwd is ~2× fwd).
    total_flops_step = total_flops_fwd * 3
    achieved_tflops = total_flops_step / (step_ms / 1000) / 1e12
    print(f"    TOTAL step (×3 for bwd):  {total_flops_step/1e12:.2f} TFLOPS")
    print(f"  Achieved: {achieved_tflops:.1f} TFLOPS at {step_ms:.1f} ms/iter")
    print(f"  4090 bf16 tensor-core peak: ~165 TFLOPS")
    print(f"  Utilization: {100*achieved_tflops/165:.1f}% of peak")
    print()
    print("  (Low utilization → dispatch-bound or bandwidth-bound, not compute-bound.")
    print("   High utilization → kernel-bound, optimization needs algorithmic changes.)")


if __name__ == "__main__":
    main()
