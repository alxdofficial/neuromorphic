"""Benchmark training/inference throughput and efficiency metrics for all models.

Measures: FLOPs/tok, training tok/s, inference tok/s, peak VRAM, VRAM breakdown.
Uses conservative binary search to find max training batch size, then benchmarks.

v5: NTP only (no FITB).
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

from src.metrics.efficiency import (
    EfficiencyReport,
    format_comparison_table,
    measure_flops_per_token,
    measure_inference_throughput,
    measure_training_throughput,
    measure_vram_breakdown,
    save_reports_json,
)


DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+


# ---------------------------------------------------------------------------
# step_fn builder — NTP training step
# ---------------------------------------------------------------------------

def make_ntp_step_fn(model, forward_fn, optimizer, input_ids, vocab):
    """Build a step_fn closure for the NTP training path."""
    def step_fn():
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if hasattr(model, 'detach_states'):
            model.detach_states()
    return step_fn


# ---------------------------------------------------------------------------
# Binary search for max batch size
# ---------------------------------------------------------------------------

def try_train_step(create_model_fn, bs, seq_len, vocab):
    """Try a FULL training step (fwd + bwd + optimizer) at given batch size."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model, forward_fn = create_model_fn(bs)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)

        step_fn = make_ntp_step_fn(model, forward_fn, optimizer, input_ids, vocab)

        # Full training step + second step to check stability
        step_fn()
        step_fn()

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


def find_max_bs(name, create_model_fn, seq_len, vocab):
    """Binary search for max training batch size (must survive full train step)."""
    lo, hi = 8, 8
    while hi <= 1024:
        if try_train_step(create_model_fn, hi, seq_len, vocab):
            lo = hi
            hi *= 2
        else:
            break
    hi = min(hi, 1024)

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_train_step(create_model_fn, mid, seq_len, vocab):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    best = (best // 8) * 8
    if best == 0:
        best = 8
    print(f"  {name}: max training BS = {best}")
    return best


# ---------------------------------------------------------------------------
# Model factories (keep in script — depend on optional packages)
# ---------------------------------------------------------------------------

def make_pythia_160m(bs, seq_len, vocab):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, vocab_size=vocab,
        max_position_embeddings=2048, use_cache=False,
    )
    model = GPTNeoXForCausalLM(cfg).to(DEVICE).to(torch.bfloat16)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids).logits
    return model, fwd


def make_mamba(bs, seq_len, vocab):
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    cfg = MambaConfig(d_model=768, n_layer=24, vocab_size=vocab)
    model = MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.bfloat16)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids).logits
    return model, fwd


def make_gpt2(bs, seq_len, vocab):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        n_embd=768, n_layer=12, n_head=12, vocab_size=vocab,
        n_positions=max(seq_len, 1024), use_cache=False,
    )
    model = GPT2LMHeadModel(cfg).to(DEVICE).to(torch.bfloat16)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids).logits
    return model, fwd


def make_neuromorphic(bs, seq_len, vocab, tier="a", compile_model=False):
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b}
    config = tier_fn[tier](vocab_size=vocab, N=seq_len, use_compile=False)
    model = NeuromorphicLM(config).to(DEVICE).to(torch.bfloat16)
    model.initialize_states(bs, DEVICE)

    if compile_model:
        model.forward_segment = torch.compile(model.forward_segment, mode="default")

    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _aux = m.forward_segment(ids)
            return logits
    return model, fwd


# ---------------------------------------------------------------------------
# Full benchmark for one model
# ---------------------------------------------------------------------------

def benchmark_model(
    name, create_model_fn, bs, seq_len, vocab, warmup, measure_iters,
    skip_flops, skip_inference,
):
    """Run all measurements for a single model. Returns EfficiencyReport."""
    print(f"\n  Benchmarking {name} (BS={bs}) ...")
    gc.collect()
    torch.cuda.empty_cache()

    # --- FLOPs ---
    flops_result = {"total_flops": 0, "flops_per_token": 0}
    if not skip_flops:
        print(f"    Counting FLOPs ...")
        model, fwd = create_model_fn(bs)
        input_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)
        flops_result = measure_flops_per_token(model, input_ids, fwd)
        print(f"    FLOPs/tok: {flops_result['flops_per_token']:,}")
        del model, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    # --- Training throughput ---
    print(f"    Measuring training throughput ...")
    model, fwd = create_model_fn(bs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    detach = (lambda m: m.detach_states()) if hasattr(model, 'detach_states') else None
    param_count = sum(p.numel() for p in model.parameters())

    input_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)
    step_fn = make_ntp_step_fn(model, fwd, optimizer, input_ids, vocab)

    train_result = measure_training_throughput(
        model, fwd, optimizer,
        bs=bs, seq_len=seq_len, vocab=vocab, device=DEVICE,
        warmup=warmup, measure=measure_iters, detach_fn=detach,
        step_fn=step_fn,
    )
    print(f"    Train: {train_result['tok_per_sec']:,.0f} tok/s, "
          f"{train_result['ms_per_step']:.1f} ms/step")

    del model, optimizer, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    # --- Inference throughput ---
    infer_result = {"tok_per_sec": 0.0, "ms_per_step": 0.0}
    if not skip_inference:
        print(f"    Measuring inference throughput ...")
        model, fwd = create_model_fn(bs)
        detach = (lambda m: m.detach_states()) if hasattr(model, 'detach_states') else None

        infer_result = measure_inference_throughput(
            model, fwd,
            bs=bs, seq_len=seq_len, vocab=vocab, device=DEVICE,
            warmup=warmup, measure=measure_iters, detach_fn=detach,
        )
        print(f"    Infer: {infer_result['tok_per_sec']:,.0f} tok/s, "
              f"{infer_result['ms_per_step']:.1f} ms/step")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- VRAM breakdown ---
    vram_result = {"weights_gb": 0.0, "optimizer_gb": 0.0,
                   "activations_gb": 0.0, "peak_gb": 0.0}
    try:
        print(f"    Measuring VRAM breakdown ...")
        model, fwd = create_model_fn(bs)
        detach = (lambda m: m.detach_states()) if hasattr(model, 'detach_states') else None

        vram_result = measure_vram_breakdown(
            model, fwd, torch.optim.AdamW,
            bs=bs, seq_len=seq_len, vocab=vocab, device=DEVICE,
            detach_fn=detach,
        )
        print(f"    VRAM: wt={vram_result['weights_gb']:.2f} "
              f"opt={vram_result['optimizer_gb']:.2f} "
              f"act={vram_result['activations_gb']:.2f} "
              f"peak={vram_result['peak_gb']:.2f} GB")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"    VRAM breakdown: OOM at BS={bs} (use lower BS for breakdown)")
        else:
            raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return EfficiencyReport(
        model_name=name,
        param_count=param_count,
        train_tok_per_sec=train_result["tok_per_sec"],
        infer_tok_per_sec=infer_result["tok_per_sec"],
        flops_per_token_fwd=flops_result["flops_per_token"],
        peak_vram_train_gb=train_result["peak_vram_gb"],
        vram_weights_gb=vram_result["weights_gb"],
        vram_optimizer_gb=vram_result["optimizer_gb"],
        vram_activations_gb=vram_result["activations_gb"],
        batch_size=bs,
        seq_len=seq_len,
        device_name=torch.cuda.get_device_name(),
        ms_per_step_train=train_result["ms_per_step"],
        ms_per_step_infer=infer_result["ms_per_step"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark training/inference throughput and efficiency metrics."
    )
    parser.add_argument("--tier", choices=["a", "b"], default="a",
                        help="Neuromorphic tier (default: a)")
    parser.add_argument("--json", type=str, default=None, metavar="FILE",
                        help="Write JSON results to FILE")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default: 256)")
    parser.add_argument("--vocab", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--measure", type=int, default=20,
                        help="Measurement iterations (default: 20)")
    parser.add_argument("--skip-flops", action="store_true",
                        help="Skip FLOP counting (slow on first run)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference throughput measurement")
    args = parser.parse_args()

    seq_len = args.seq_len
    vocab = args.vocab

    # Model factories — closure over seq_len/vocab
    # Two versions of neuromorphic factory:
    #   - no compile for BS search (avoids compile memory overhead skewing max BS)
    #   - with compile for actual benchmark measurement
    def _neuro_nocompile(bs):
        return make_neuromorphic(bs, seq_len, vocab, tier=args.tier,
                                 compile_model=False)
    def _neuro_compiled(bs):
        return make_neuromorphic(bs, seq_len, vocab, tier=args.tier,
                                 compile_model=True)

    def _pythia(bs):
        return make_pythia_160m(bs, seq_len, vocab)

    def _mamba(bs):
        return make_mamba(bs, seq_len, vocab)

    def _gpt2(bs):
        return make_gpt2(bs, seq_len, vocab)

    neuro_name = f"Neuromorphic-Tier{args.tier.upper()}"

    # (name, bs_search_factory, benchmark_factory)
    models = [
        ("Pythia-160M", _pythia, _pythia),
        ("Mamba-130M", _mamba, _mamba),
        ("GPT2-124M", _gpt2, _gpt2),
        (neuro_name, _neuro_nocompile, _neuro_compiled),
    ]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"seq_len={seq_len}, vocab={vocab}, warmup={args.warmup}, "
          f"measure={args.measure}")
    print(f"{'=' * 95}")

    # Phase 1: find max batch sizes (no compile — avoids overhead skewing max BS)
    print("\n--- Finding max training batch sizes ---")
    max_bs = {}
    for name, bs_factory, bench_factory in models:
        try:
            bs = find_max_bs(name, bs_factory, seq_len, vocab)
            max_bs[name] = (bench_factory, bs)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {name}: FAILED - {e}")

    # Phase 2: full benchmarks (with compile for neuromorphic)
    print(f"\n{'=' * 95}")
    print("--- Benchmarking at max batch size ---")

    reports = []
    for name, (factory, bs) in max_bs.items():
        try:
            report = benchmark_model(
                name, factory, bs, seq_len, vocab,
                args.warmup, args.measure,
                args.skip_flops, args.skip_inference,
            )
            reports.append(report)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {name}: FAILED - {e}")

    # Phase 3: print comparison table
    print(f"\n{'=' * 95}")
    print("--- Comparison Table ---\n")
    print(format_comparison_table(reports))

    # Phase 4: save JSON if requested
    if args.json and reports:
        save_reports_json(reports, args.json)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
