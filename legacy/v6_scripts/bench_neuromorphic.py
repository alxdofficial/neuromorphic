"""Benchmark neuromorphic model v5 training/inference throughput.

Measures:
  1. Max training batch size (binary search, no compile — faster)
  2. Training throughput at max BS (with torch.compile)
  3. Inference throughput at max BS (with torch.compile)
  4. VRAM breakdown

With --k-segments K, the benchmark simulates the real TBPTT training loop:
K forward passes accumulated into one backward, matching actual trainer memory
and compute cost.

Usage:
  python scripts/bench_neuromorphic.py --tier a
  python scripts/bench_neuromorphic.py --tier a --k-segments 8 --grad-checkpointing
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM

DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision("high")


def make_model(bs, seq_len, vocab, tier="a", compile_model=False,
               grad_checkpointing=False, k_segments=1):
    """Create neuromorphic model, move to GPU, initialize states."""
    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b, "c": ModelConfig.tier_c}
    config = tier_fn[tier](vocab_size=vocab, N=seq_len, use_compile=False,
                           gradient_checkpointing=grad_checkpointing,
                           K_segments=k_segments)
    config.validate()
    model = NeuromorphicLM(config).to(DEVICE).to(torch.bfloat16)
    model.initialize_states(bs, DEVICE)

    if compile_model:
        model.forward_segment = torch.compile(
            model.forward_segment, mode="default"
        )

    return model, config


def train_step(model, optimizer, input_ids, vocab, k_segments=1):
    """Full TBPTT step: K forward passes accumulated into one backward.

    input_ids: [BS, T] where T = k_segments * N
    Matches real trainer: accumulate K segment losses, one backward, detach.
    """
    BS, T = input_ids.shape
    N = T // k_segments

    optimizer.zero_grad()
    chunk_loss = None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for seg in range(k_segments):
            seg_ids = input_ids[:, seg * N:(seg + 1) * N]
            logits, aux_loss = model.forward_segment(seg_ids)
            seg_loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab).float(),
                seg_ids[:, 1:].reshape(-1),
            ) + aux_loss
            chunk_loss = seg_loss if chunk_loss is None else chunk_loss + seg_loss

    chunk_loss.backward()
    optimizer.step()
    model.detach_states()
    return chunk_loss.item()


def try_train_step(bs, seq_len, vocab, tier, grad_checkpointing=False, k_segments=1):
    """Try a full TBPTT step at given batch size. Returns True if it fits."""
    gc.collect()
    torch.cuda.empty_cache()
    T = seq_len * k_segments
    try:
        model, config = make_model(bs, seq_len, vocab, tier=tier, compile_model=False,
                                   grad_checkpointing=grad_checkpointing,
                                   k_segments=k_segments)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab, (bs, T), device=DEVICE)

        train_step(model, optimizer, input_ids, vocab, k_segments=k_segments)
        train_step(model, optimizer, input_ids, vocab, k_segments=k_segments)
        torch.cuda.synchronize()

        del model, optimizer, input_ids
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return False
        raise


def find_max_bs(seq_len, vocab, tier, grad_checkpointing=False, k_segments=1):
    """Binary search for max training batch size."""
    lo, hi = 4, 4
    while hi <= 512:
        print(f"  Trying BS={hi}...", end=" ", flush=True)
        if try_train_step(hi, seq_len, vocab, tier,
                          grad_checkpointing=grad_checkpointing,
                          k_segments=k_segments):
            print("OK", flush=True)
            lo = hi
            hi *= 2
        else:
            print("OOM", flush=True)
            break
    hi = min(hi, 512)

    # Refine
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"  Refine BS={mid}...", end=" ", flush=True)
        if try_train_step(mid, seq_len, vocab, tier,
                          grad_checkpointing=grad_checkpointing,
                          k_segments=k_segments):
            print("OK", flush=True)
            best = mid
            lo = mid + 1
        else:
            print("OOM", flush=True)
            hi = mid - 1

    best = (best // 4) * 4  # align to 4
    if best == 0:
        best = 4
    return best


def measure_throughput(model, input_ids, vocab, k_segments=1, warmup=5, measure=30):
    """Measure training throughput (tok/s) with proper CUDA timing."""
    bs, T = input_ids.shape
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Warmup (step 1 triggers torch.compile tracing — can take 5-10 min)
    for i in range(warmup):
        t0 = time.perf_counter()
        train_step(model, optimizer, input_ids, vocab, k_segments=k_segments)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"    Warmup step {i+1}/{warmup}: {dt:.1f}s", flush=True)

    # Measure
    print(f"    Measuring ({measure} steps)...", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    for i in range(measure):
        train_step(model, optimizer, input_ids, vocab, k_segments=k_segments)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = bs * T * measure
    tok_s = tokens / elapsed
    ms_per_step = elapsed / measure * 1000
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / 1e9

    return tok_s, ms_per_step, peak_vram


def measure_inference(model, input_ids, vocab, warmup=5, measure=50):
    """Measure inference throughput (single segment, forward only, no grad)."""
    bs, N = input_ids.shape
    model.set_eval_mode()

    # Warmup
    for i in range(warmup):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.forward_segment(input_ids)
        model.detach_states()

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(measure):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.forward_segment(input_ids)
        model.detach_states()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = bs * N * measure
    return tokens / elapsed, elapsed / measure * 1000


def main():
    parser = argparse.ArgumentParser(description="Benchmark neuromorphic v5 model")
    parser.add_argument("--tier", choices=["a", "b", "c"], default="a")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=30)
    parser.add_argument("--bs", type=int, default=0,
                        help="Override batch size (0=auto find max)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (halves scan activation memory)")
    parser.add_argument("--k-segments", type=int, default=1,
                        help="TBPTT segments per step — simulates real training loop memory/compute")
    args = parser.parse_args()

    seq_len = args.seq_len
    vocab = args.vocab
    use_compile = not args.no_compile
    grad_checkpointing = args.grad_checkpointing
    k_segments = args.k_segments
    T = seq_len * k_segments

    print(f"=" * 70, flush=True)
    print(f"Neuromorphic v5 Benchmark — Tier {args.tier.upper()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    print(f"N={seq_len}, T={T}, vocab={vocab}, compile={use_compile}, "
          f"grad_ckpt={grad_checkpointing}, k_segments={k_segments}", flush=True)

    # Param count
    model_tmp, config = make_model(1, seq_len, vocab, tier=args.tier, compile_model=False,
                                   grad_checkpointing=grad_checkpointing,
                                   k_segments=k_segments)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} ({param_count/1e6:.1f}M)", flush=True)
    print(f"Config: D={config.D}, D_embed={config.D_embed}, B={config.B}, "
          f"C={config.C}, L_total={config.L_total}, L_mem={config.L_mem}, M={config.M}", flush=True)
    del model_tmp
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 1: Find max BS (no compile — faster)
    if args.bs > 0:
        max_bs = args.bs
        print(f"\nUsing specified BS={max_bs}", flush=True)
    else:
        print(f"\n--- Phase 1: Finding max training batch size (no compile) ---",
              flush=True)
        max_bs = find_max_bs(seq_len, vocab, args.tier,
                             grad_checkpointing=grad_checkpointing,
                             k_segments=k_segments)
    print(f"Max training BS: {max_bs}  (tok/step = {max_bs * T:,})", flush=True)
    compile_bs = max_bs

    # Phase 2: Training throughput (with compile)
    print(f"\n--- Phase 2: Training throughput (BS={compile_bs}, compile={use_compile}) ---",
          flush=True)
    gc.collect()
    torch.cuda.empty_cache()

    model, config = make_model(
        compile_bs, seq_len, vocab, tier=args.tier, compile_model=use_compile,
        grad_checkpointing=grad_checkpointing, k_segments=k_segments
    )
    input_ids = torch.randint(0, vocab, (compile_bs, T), device=DEVICE)

    tok_s, ms_step, peak_vram = measure_throughput(
        model, input_ids, vocab, k_segments=k_segments,
        warmup=args.warmup, measure=args.measure
    )
    print(f"  Training: {tok_s:,.0f} tok/s", flush=True)
    print(f"  ms/step:  {ms_step:.1f}  ({compile_bs * T:,} tok/step)", flush=True)
    print(f"  Peak VRAM: {peak_vram:.2f} GB", flush=True)

    # Phase 3: Inference throughput (single segment)
    infer_ids = torch.randint(0, vocab, (compile_bs, seq_len), device=DEVICE)
    print(f"\n--- Phase 3: Inference throughput (BS={compile_bs}, single segment) ---",
          flush=True)
    infer_tok_s, infer_ms = measure_inference(
        model, infer_ids, vocab, warmup=args.warmup, measure=args.measure
    )
    print(f"  Inference: {infer_tok_s:,.0f} tok/s", flush=True)
    print(f"  ms/step:   {infer_ms:.1f}", flush=True)

    # Phase 4: VRAM breakdown
    print(f"\n--- Phase 4: VRAM breakdown ---", flush=True)
    del model, input_ids, infer_ids
    gc.collect()
    torch.cuda.empty_cache()

    # Weights only
    torch.cuda.reset_peak_memory_stats(DEVICE)
    model, _ = make_model(compile_bs, seq_len, vocab, tier=args.tier, compile_model=False,
                          grad_checkpointing=grad_checkpointing, k_segments=k_segments)
    weights_gb = torch.cuda.memory_allocated(DEVICE) / 1e9

    # After optimizer step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, vocab, (max_bs, T), device=DEVICE)
    train_step(model, optimizer, input_ids, vocab, k_segments=k_segments)
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    after_opt_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
    optimizer_gb = after_opt_gb - weights_gb

    # Peak during training
    torch.cuda.reset_peak_memory_stats(DEVICE)
    train_step(model, optimizer, input_ids, vocab, k_segments=k_segments)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    activations_gb = max(peak_gb - after_opt_gb, 0)

    print(f"  Weights:     {weights_gb:.2f} GB", flush=True)
    print(f"  Optimizer:   {optimizer_gb:.2f} GB", flush=True)
    print(f"  Activations: {activations_gb:.2f} GB", flush=True)
    print(f"  Peak:        {peak_gb:.2f} GB", flush=True)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"SUMMARY — Neuromorphic v5 Tier {args.tier.upper()}", flush=True)
    print(f"  Params:        {param_count/1e6:.1f}M", flush=True)
    print(f"  Max train BS:  {max_bs}  (tok/step = {max_bs * T:,})", flush=True)
    print(f"  Train tok/s:   {tok_s:,.0f} (compile={use_compile}, K={k_segments})", flush=True)
    print(f"  Infer tok/s:   {infer_tok_s:,.0f} (single segment)", flush=True)
    print(f"  ms/step:       {ms_step:.1f} (train), {infer_ms:.1f} (infer)", flush=True)
    print(f"  Peak VRAM:     {peak_vram:.2f} GB", flush=True)
    print(f"  VRAM split:    {weights_gb:.2f} wt / {optimizer_gb:.2f} opt / "
          f"{activations_gb:.2f} act", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
