"""Benchmark neuromorphic FITB training + generation throughput.

Sweeps batch sizes to find throughput-maximizing BS (not just max-fitting),
benchmarks FITB training, FITB generation (generate_segments), and
compares against Pythia/GPT2 baselines.

Usage:
    python scripts/bench_fitb_speed.py
    python scripts/bench_fitb_speed.py --tier b --no-baselines
    python scripts/bench_fitb_speed.py --json results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def _timed(fn, warmup=5, measure=20):
    """Run fn with warmup, then measure. Returns avg ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measure):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / measure * 1000  # ms per step


# ---------------------------------------------------------------------------
# Neuromorphic FITB training step
# ---------------------------------------------------------------------------

def make_fitb_step(model, optimizer, input_ids, config):
    """Closure: full FITB train step (mask, forward, loss, backward, step)."""
    from src.training.masking import generate_fitb_mask

    bs, seq_len = input_ids.shape
    device = input_ids.device

    def step():
        optimizer.zero_grad()
        fitb_mask = generate_fitb_mask(
            bs, seq_len, config.mask_rate, config.span_mask_prob,
            config.span_mask_mean_len, device,
        )
        ids_masked = input_ids.clone()
        ids_masked[fitb_mask] = config.fitb_id

        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce_loss, aux, valid = model.forward_segment(
                ids_masked, fitb_mask=fitb_mask, target_ids=input_ids,
            )
            loss = ce_loss / valid.float().clamp(min=1) + aux

        loss.backward()
        optimizer.step()
        model.detach_states()

    return step


# ---------------------------------------------------------------------------
# Baseline NTP training step
# ---------------------------------------------------------------------------

def make_ntp_step(model, forward_fn, optimizer, input_ids, vocab):
    """Closure: standard NTP train step."""
    def step():
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if hasattr(model, 'detach_states'):
            model.detach_states()
    return step


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def create_neuromorphic(bs, seq_len, vocab, tier="a", compile_model=False):
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b, "c": ModelConfig.tier_c}
    config = tier_fn[tier](
        vocab_size=vocab, N=seq_len, use_compile=False,
        fitb_id=vocab - 2, null_id=vocab - 1, mask_rate=0.3,
    )
    model = NeuromorphicLM(config).to(DEVICE).to(torch.bfloat16)
    model.initialize_states(bs, DEVICE)

    if compile_model:
        model.forward_segment = torch.compile(model.forward_segment, mode="default")

    return model


def create_pythia_160m(bs, seq_len, vocab):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, vocab_size=vocab,
        max_position_embeddings=max(seq_len, 2048), use_cache=False,
    )
    model = GPTNeoXForCausalLM(cfg).to(DEVICE).to(torch.bfloat16)

    def fwd(m, ids):
        return m(ids).logits
    return model, fwd


def create_gpt2(bs, seq_len, vocab):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        n_embd=768, n_layer=12, n_head=12, vocab_size=vocab,
        n_positions=max(seq_len, 1024), use_cache=False,
    )
    model = GPT2LMHeadModel(cfg).to(DEVICE).to(torch.bfloat16)

    def fwd(m, ids):
        return m(ids).logits
    return model, fwd


def create_mamba_130m(bs, seq_len, vocab):
    from mamba_ssm import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    cfg = MambaConfig(
        d_model=768, n_layer=24, vocab_size=vocab,
    )
    model = MambaLMHeadModel(cfg).to(DEVICE).to(torch.bfloat16)

    def fwd(m, ids):
        return m(ids).logits
    return model, fwd


# ---------------------------------------------------------------------------
# Batch size sweep for neuromorphic FITB
# ---------------------------------------------------------------------------

def sweep_neuro_fitb(tier, seq_len, vocab, bs_candidates, compile_model, measure_iters):
    """Sweep batch sizes for neuromorphic FITB training. Returns results list."""
    results = []
    best_bs, best_tps = 0, 0.0

    for bs in bs_candidates:
        _cleanup()
        try:
            model = create_neuromorphic(bs, seq_len, vocab, tier=tier,
                                         compile_model=compile_model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            input_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)
            step = make_fitb_step(model, optimizer, input_ids, model.config)

            # Warmup
            for _ in range(3):
                step()
            torch.cuda.synchronize()

            # Measure
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(measure_iters):
                step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            tok_per_sec = bs * seq_len * measure_iters / elapsed
            peak_vram = torch.cuda.max_memory_allocated() / 1e9

            results.append((bs, tok_per_sec, peak_vram))
            marker = ""
            if tok_per_sec > best_tps:
                best_tps = tok_per_sec
                best_bs = bs
                marker = " <-- best"
            print(f"    BS={bs:>4d}: {tok_per_sec:>9,.0f} tok/s  "
                  f"({peak_vram:.1f} GB VRAM){marker}")

            del model, optimizer, input_ids
            _cleanup()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"    BS={bs:>4d}: OOM")
                _cleanup()
                break
            raise

    return best_bs, best_tps, results


# ---------------------------------------------------------------------------
# Batch size sweep for baselines
# ---------------------------------------------------------------------------

def sweep_baseline(name, factory_fn, seq_len, vocab, bs_candidates, measure_iters):
    """Sweep batch sizes for a baseline model. Returns results."""
    results = []
    best_bs, best_tps = 0, 0.0
    params = 0

    for bs in bs_candidates:
        _cleanup()
        try:
            model, fwd = factory_fn(bs, seq_len, vocab)
            params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            input_ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)
            step = make_ntp_step(model, fwd, optimizer, input_ids, vocab)

            for _ in range(3):
                step()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(measure_iters):
                step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            tps = bs * seq_len * measure_iters / elapsed
            peak = torch.cuda.max_memory_allocated() / 1e9

            results.append((bs, tps, peak))
            marker = ""
            if tps > best_tps:
                best_tps = tps
                best_bs = bs
                marker = " <-- best"
            print(f"    BS={bs:>4d}: {tps:>9,.0f} tok/s  ({peak:.1f} GB){marker}")

            del model, optimizer, input_ids
            _cleanup()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"    BS={bs:>4d}: OOM")
                _cleanup()
                break
            raise

    return best_bs, best_tps, params, results


# ---------------------------------------------------------------------------
# Generate segments benchmark
# ---------------------------------------------------------------------------

def sweep_generate(tier, seq_len, vocab, gen_tokens, bs_candidates, measure_iters):
    """Sweep batch sizes for generate_segments."""
    results = []
    best_bs, best_tps = 0, 0.0

    for bs in bs_candidates:
        _cleanup()
        try:
            model = create_neuromorphic(bs, seq_len, vocab, tier=tier,
                                         compile_model=False)
            model.eval()
            prompt = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)

            def gen_step():
                model.initialize_states(bs, DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.generate_segments(prompt, max_new_tokens=gen_tokens)

            ms = _timed(gen_step, warmup=3, measure=measure_iters)
            tok_per_sec = bs * gen_tokens / (ms / 1000)

            results.append((bs, tok_per_sec, ms))
            marker = ""
            if tok_per_sec > best_tps:
                best_tps = tok_per_sec
                best_bs = bs
                marker = " <-- best"
            print(f"    BS={bs:>4d}: {tok_per_sec:>9,.0f} tok/s  "
                  f"({ms:.0f} ms for {gen_tokens} tokens){marker}")

            del model; _cleanup()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"    BS={bs:>4d}: OOM")
                _cleanup()
                break
            raise

    return best_bs, best_tps, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark FITB training + generation speed")
    parser.add_argument("--tier", choices=["a", "b", "c"], default="a")
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline model benchmarks")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (faster startup, slower throughput)")
    parser.add_argument("--json", type=str, default=None, metavar="FILE")
    parser.add_argument("--measure", type=int, default=20,
                        help="Measurement iterations per batch size (default: 20)")
    args = parser.parse_args()

    from src.model.config import ModelConfig
    tier_fn = {"a": ModelConfig.tier_a, "b": ModelConfig.tier_b, "c": ModelConfig.tier_c}
    ref_config = tier_fn[args.tier](vocab_size=args.vocab)
    seq_len = ref_config.N  # Use model's native N (512)

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Count actual params
    _tmp = create_neuromorphic(1, seq_len, args.vocab, tier=args.tier)
    param_count = sum(p.numel() for p in _tmp.parameters())
    del _tmp; _cleanup()

    print(f"Tier {args.tier.upper()}: D={ref_config.D}, D_embed={ref_config.D_embed}, "
          f"B={ref_config.B_blocks}, C={ref_config.C}, R={ref_config.R}, N={seq_len}")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"Training: FITB (mask_rate=0.3, R={ref_config.R})")
    print(f"Compile: {'yes' if not args.no_compile else 'no'}")
    print(f"{'='*80}")

    use_compile = not args.no_compile
    bs_candidates = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]

    # ------------------------------------------------------------------
    # 1. FITB Training sweep
    # ------------------------------------------------------------------
    print(f"\n--- FITB Training: Batch Size Sweep ---")
    best_train_bs, best_train_tps, train_results = sweep_neuro_fitb(
        args.tier, seq_len, args.vocab, bs_candidates, use_compile, args.measure,
    )

    pred_tps = best_train_tps * ref_config.mask_rate * ref_config.R
    ms_train = best_train_bs * seq_len / best_train_tps * 1000 if best_train_tps > 0 else 0

    print(f"\n  Best training: BS={best_train_bs}, {best_train_tps:,.0f} tok/s "
          f"({ms_train:.1f} ms/step)")
    print(f"  Predicted tok/s (mask=0.3 x R={ref_config.R}): {pred_tps:,.0f}")

    # ------------------------------------------------------------------
    # 2. FITB Generation sweep
    # ------------------------------------------------------------------
    print(f"\n--- FITB Generation: generate_segments ---")
    gen_bs_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
    gen_tokens = seq_len * 2  # 1024 tokens

    best_gen_bs, best_gen_tps, gen_results = sweep_generate(
        args.tier, seq_len, args.vocab, gen_tokens, gen_bs_candidates,
        max(args.measure // 2, 5),
    )

    if best_gen_tps > 0:
        print(f"\n  Best generation: BS={best_gen_bs}, {best_gen_tps:,.0f} tok/s")

    # ------------------------------------------------------------------
    # 3. Baselines
    # ------------------------------------------------------------------
    baseline_data = {}
    if not args.no_baselines:
        print(f"\n{'='*80}")
        print(f"--- Baselines (NTP, seq_len={seq_len}) ---")

        baselines = []
        try:
            from transformers import GPTNeoXConfig
            baselines.append(("Pythia-160M", create_pythia_160m))
        except ImportError:
            print("  Pythia: skipped (no transformers)")
        try:
            from transformers import GPT2Config
            baselines.append(("GPT2-124M", create_gpt2))
        except ImportError:
            print("  GPT2: skipped (no transformers)")
        try:
            from mamba_ssm import MambaLMHeadModel
            baselines.append(("Mamba-130M", create_mamba_130m))
        except ImportError:
            print("  Mamba: skipped (no mamba_ssm)")

        for name, factory in baselines:
            print(f"\n  {name}:")
            b_bs, b_tps, b_params, b_results = sweep_baseline(
                name, factory, seq_len, args.vocab, bs_candidates, args.measure,
            )
            baseline_data[name] = {
                "best_bs": b_bs, "best_tps": b_tps, "params": b_params,
                "sweep": b_results,
            }

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  SUMMARY  —  Tier {args.tier.upper()} on {torch.cuda.get_device_name()}")
    print(f"{'='*80}\n")

    header = f"{'Model':<30s} {'Params':>10s} {'Best BS':>8s} {'tok/s':>12s} {'ms/step':>10s}"
    print(header)
    print("-" * len(header))

    # Neuromorphic training
    print(f"{'Neuro-FITB (train)':<30s} {param_count/1e6:>9.1f}M {best_train_bs:>8d} "
          f"{best_train_tps:>12,.0f} {ms_train:>9.1f}")
    print(f"{'  pred (mask*.3 x R)':<30s} {'':>10s} {'':>8s} {pred_tps:>12,.0f}")

    # Neuromorphic generation
    if best_gen_tps > 0:
        ms_gen = best_gen_bs * gen_tokens / best_gen_tps * 1000
        print(f"{'Neuro-FITB (generate)':<30s} {'':>10s} {best_gen_bs:>8d} "
              f"{best_gen_tps:>12,.0f} {ms_gen:>9.1f}")

    # Baselines
    for name, info in baseline_data.items():
        ms_b = info["best_bs"] * seq_len / info["best_tps"] * 1000 if info["best_tps"] > 0 else 0
        print(f"{name:<30s} {info['params']/1e6:>9.1f}M {info['best_bs']:>8d} "
              f"{info['best_tps']:>12,.0f} {ms_b:>9.1f}")

    # Ratios
    print()
    for name, info in baseline_data.items():
        if info["best_tps"] > 0 and best_train_tps > 0:
            ratio = best_train_tps / info["best_tps"]
            print(f"  Neuro train / {name}: {ratio:.2f}x")
            pred_ratio = pred_tps / info["best_tps"]
            print(f"  Neuro pred  / {name}: {pred_ratio:.2f}x")

    # Save JSON
    if args.json:
        import json
        from pathlib import Path
        data = {
            "gpu": torch.cuda.get_device_name(),
            "tier": args.tier,
            "seq_len": seq_len,
            "vocab": args.vocab,
            "compiled": use_compile,
            "neuromorphic": {
                "params": param_count,
                "R": ref_config.R,
                "mask_rate": 0.3,
                "train": {
                    "best_bs": best_train_bs,
                    "best_tok_per_sec": best_train_tps,
                    "predicted_tok_per_sec": pred_tps,
                    "ms_per_step": ms_train,
                    "sweep": [(bs, tps, vram) for bs, tps, vram in train_results],
                },
                "generate": {
                    "best_bs": best_gen_bs,
                    "best_tok_per_sec": best_gen_tps,
                    "gen_tokens": gen_tokens,
                    "sweep": [(bs, tps, ms) for bs, tps, ms in gen_results],
                },
            },
            "baselines": {
                name: {
                    "params": info["params"],
                    "best_bs": info["best_bs"],
                    "best_tok_per_sec": info["best_tps"],
                    "sweep": info["sweep"],
                }
                for name, info in baseline_data.items()
            },
        }
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
