"""Benchmark training/inference throughput and efficiency metrics for all models.

Measures: FLOPs/tok, training tok/s, inference tok/s, peak VRAM, VRAM breakdown.
Uses conservative binary search to find max training batch size, then benchmarks.

Use --fitb to benchmark the neuromorphic model with its real FITB training path
(masked token prediction, R passes per step) instead of NTP.
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


# ---------------------------------------------------------------------------
# step_fn builders — encapsulate full train step for NTP / FITB
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


def make_fitb_step_fn(model, optimizer, input_ids, config):
    """Build a step_fn closure for the FITB training path.

    Each call generates a fresh mask, replaces masked tokens with <FITB>,
    runs forward_segment with fitb_mask, computes fitb_cross_entropy,
    backward, optimizer step, and detach.
    """
    from src.training.masking import generate_fitb_mask
    from src.training.loss import fitb_cross_entropy

    bs, seq_len = input_ids.shape
    device = input_ids.device

    def step_fn():
        optimizer.zero_grad()
        fitb_mask = generate_fitb_mask(
            bs, seq_len, config.mask_rate, config.span_mask_prob,
            config.span_mask_mean_len, device,
        )
        ids_masked = input_ids.clone()
        ids_masked[fitb_mask] = config.fitb_id

        with torch.autocast("cuda", dtype=torch.bfloat16):
            per_pass_logits, aux = model.forward_segment(
                ids_masked, fitb_mask=fitb_mask,
            )
            loss, _ = fitb_cross_entropy(per_pass_logits, input_ids, fitb_mask)
            loss = loss + aux

        loss.backward()
        optimizer.step()
        model.detach_states()

    return step_fn


# ---------------------------------------------------------------------------
# Binary search for max batch size
# ---------------------------------------------------------------------------

def try_train_step(create_model_fn, bs, seq_len, vocab, fitb=False):
    """Try a FULL training step (fwd + bwd + optimizer) at given batch size."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model, forward_fn = create_model_fn(bs)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)

        if fitb and hasattr(model, 'config') and model.config.fitb_id >= 0:
            step_fn = make_fitb_step_fn(model, optimizer, input_ids, model.config)
        else:
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


def find_max_bs(name, create_model_fn, seq_len, vocab, fitb=False):
    """Binary search for max training batch size (must survive full train step)."""
    lo, hi = 8, 8
    while hi <= 1024:
        if try_train_step(create_model_fn, hi, seq_len, vocab, fitb=fitb):
            lo = hi
            hi *= 2
        else:
            break
    hi = min(hi, 1024)

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_train_step(create_model_fn, mid, seq_len, vocab, fitb=fitb):
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


def make_neuromorphic(bs, seq_len, vocab, tier="a"):
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b, "c": ModelConfig.tier_c}
    config = tier_fn[tier](vocab_size=vocab, N=seq_len, use_compile=False)
    model = NeuromorphicLM(config).to(DEVICE).to(torch.bfloat16)
    model.initialize_states(bs, DEVICE)

    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _aux = m.forward_segment(ids)
            return logits
    return model, fwd


def make_neuromorphic_fitb(bs, seq_len, vocab, tier="a"):
    """Create neuromorphic model configured for FITB benchmarking.

    Sets fitb_id / null_id to the last two vocab entries and mask_rate=0.3.
    Returns (model, fwd_fn) where fwd_fn wraps the FITB forward path.
    """
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b, "c": ModelConfig.tier_c}
    config = tier_fn[tier](
        vocab_size=vocab, N=seq_len, use_compile=False,
        fitb_id=vocab - 2, null_id=vocab - 1, mask_rate=0.3,
    )
    model = NeuromorphicLM(config).to(DEVICE).to(torch.bfloat16)
    model.initialize_states(bs, DEVICE)

    def fwd(m, ids):
        """FITB forward for FLOPs counting — generates mask and runs FITB path."""
        from src.training.masking import generate_fitb_mask
        fitb_mask = generate_fitb_mask(
            ids.shape[0], ids.shape[1], config.mask_rate,
            config.span_mask_prob, config.span_mask_mean_len, ids.device,
        )
        ids_masked = ids.clone()
        ids_masked[fitb_mask] = config.fitb_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            per_pass_logits, _aux = m.forward_segment(
                ids_masked, fitb_mask=fitb_mask,
            )
            # Return last-pass logits for FLOPs compatibility (shape [BS,N,V])
            return per_pass_logits[-1]

    return model, fwd


# ---------------------------------------------------------------------------
# Full benchmark for one model
# ---------------------------------------------------------------------------

def benchmark_model(
    name, create_model_fn, bs, seq_len, vocab, warmup, measure_iters,
    skip_flops, skip_inference, fitb=False,
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

    if fitb and hasattr(model, 'config') and model.config.fitb_id >= 0:
        step_fn = make_fitb_step_fn(model, optimizer, input_ids, model.config)
    else:
        step_fn = make_ntp_step_fn(model, fwd, optimizer, input_ids, vocab)

    train_result = measure_training_throughput(
        model, fwd, optimizer,
        bs=bs, seq_len=seq_len, vocab=vocab, device=DEVICE,
        warmup=warmup, measure=measure_iters, detach_fn=detach,
        step_fn=step_fn,
    )
    print(f"    Train: {train_result['tok_per_sec']:,.0f} tok/s, "
          f"{train_result['ms_per_step']:.1f} ms/step")

    # Compute predicted_tok_per_sec for FITB
    predicted_tok_per_sec = 0.0
    if fitb and hasattr(model, 'config') and model.config.fitb_id >= 0:
        cfg = model.config
        ms_per_step = train_result["ms_per_step"]
        if ms_per_step > 0:
            predicted_tok_per_sec = (
                bs * seq_len * cfg.mask_rate * cfg.R / (ms_per_step / 1000.0)
            )
        print(f"    Pred tok/s: {predicted_tok_per_sec:,.0f} "
              f"(mask_rate={cfg.mask_rate}, R={cfg.R})")

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
    # NTP: let measure_vram_breakdown manage its own optimizer lifecycle
    #       (weights-only baseline must be measured before optimizer exists).
    # FITB: must pass step_fn because internal NTP loss path won't work,
    #        but we defer optimizer creation to keep weights-only phase clean.
    # Note: VRAM breakdown can OOM at max BS (it needs two full train steps
    # plus measurement overhead). Catch and report zeros rather than losing
    # the throughput numbers.
    vram_result = {"weights_gb": 0.0, "optimizer_gb": 0.0,
                   "activations_gb": 0.0, "peak_gb": 0.0}
    try:
        print(f"    Measuring VRAM breakdown ...")
        model, fwd = create_model_fn(bs)
        detach = (lambda m: m.detach_states()) if hasattr(model, 'detach_states') else None
        is_fitb_model = fitb and hasattr(model, 'config') and model.config.fitb_id >= 0

        if is_fitb_model:
            # For FITB VRAM: create optimizer + step_fn lazily via wrapper
            # so measure_vram_breakdown sees weights-only first
            _vram_opt = [None]
            _vram_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)
            _vram_cfg = model.config

            def _vram_fitb_step():
                if _vram_opt[0] is None:
                    _vram_opt[0] = torch.optim.AdamW(model.parameters(), lr=1e-4)
                make_fitb_step_fn(model, _vram_opt[0], _vram_ids, _vram_cfg)()

            vram_result = measure_vram_breakdown(
                model, fwd, torch.optim.AdamW,
                bs=bs, seq_len=seq_len, vocab=vocab, device=DEVICE,
                detach_fn=detach, step_fn=_vram_fitb_step,
            )
        else:
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
        predicted_tok_per_sec=predicted_tok_per_sec,
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
    parser.add_argument("--tier", choices=["a", "b", "c"], default="a",
                        help="Neuromorphic tier (default: a)")
    parser.add_argument("--fitb", action="store_true",
                        help="Benchmark neuromorphic model with FITB training path "
                             "(masked token prediction, R passes). Baselines stay NTP.")
    parser.add_argument("--json", type=str, default=None, metavar="FILE",
                        help="Write JSON results to FILE")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length (default: 128)")
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
    use_fitb = args.fitb

    # Model factories — closure over seq_len/vocab
    if use_fitb:
        def _neuromorphic(bs):
            return make_neuromorphic_fitb(bs, seq_len, vocab, tier=args.tier)
    else:
        def _neuromorphic(bs):
            return make_neuromorphic(bs, seq_len, vocab, tier=args.tier)

    def _pythia(bs):
        return make_pythia_160m(bs, seq_len, vocab)

    def _mamba(bs):
        return make_mamba(bs, seq_len, vocab)

    def _gpt2(bs):
        return make_gpt2(bs, seq_len, vocab)

    neuro_name = f"Neuromorphic-Tier{args.tier.upper()}"
    if use_fitb:
        neuro_name += "-FITB"

    models = [
        ("Pythia-160M", _pythia, False),
        ("Mamba-130M", _mamba, False),
        ("GPT2-124M", _gpt2, False),
        (neuro_name, _neuromorphic, use_fitb),
    ]

    print(f"GPU: {torch.cuda.get_device_name()}")
    mode_str = "FITB" if use_fitb else "NTP"
    print(f"seq_len={seq_len}, vocab={vocab}, warmup={args.warmup}, "
          f"measure={args.measure}, neuro_mode={mode_str}")
    print(f"{'=' * 95}")

    # Phase 1: find max batch sizes
    print("\n--- Finding max training batch sizes ---")
    max_bs = {}
    for name, factory, is_fitb in models:
        try:
            bs = find_max_bs(name, factory, seq_len, vocab, fitb=is_fitb)
            max_bs[name] = (factory, bs, is_fitb)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {name}: FAILED - {e}")

    # Phase 2: full benchmarks
    print(f"\n{'=' * 95}")
    print("--- Benchmarking at max batch size ---")

    reports = []
    for name, (factory, bs, is_fitb) in max_bs.items():
        try:
            report = benchmark_model(
                name, factory, bs, seq_len, vocab,
                args.warmup, args.measure,
                args.skip_flops, args.skip_inference,
                fitb=is_fitb,
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
