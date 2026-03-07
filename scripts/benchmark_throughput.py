#!/usr/bin/env python3
"""
Throughput benchmark: neuromorphic LM vs all baselines.

Measures training throughput (tok/s) and peak VRAM for each model at
T=1024 and T=2048. Uses random data to isolate model speed from I/O.
Outputs a clean comparison table + optional JSON for cost extrapolation.

Usage:
    python scripts/benchmark_throughput.py
    python scripts/benchmark_throughput.py --models neuromorphic-a pythia-160m
    python scripts/benchmark_throughput.py --tiers A B
    python scripts/benchmark_throughput.py --tiers A B C --json results/throughput.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

# Ensure repo root is on sys.path so `from src.xxx` works
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOCAB = 32000
WARMUP_STEPS = 5
BENCH_STEPS = 20
SEQ_LENGTHS = [1024, 2048]

# Batch sizes to try (descending). We pick the largest that doesn't OOM.
BS_CANDIDATES = [64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]

# Models that support gradient checkpointing
GRAD_CKPT_TYPES = {"gpt2", "gpt_neox", "llama"}

# Models that support torch.compile
COMPILE_TYPES = {"gpt2", "gpt_neox", "llama"}

TIER_MODELS = {
    "A": ["neuromorphic-a", "gpt2-small", "pythia-160m"],
    "B": ["neuromorphic-b", "gpt2-medium", "pythia-410m"],
    "C": ["neuromorphic-c", "pythia-1b", "tinyllama-1b"],
}


# Known GPU peak BF16 TFLOPS (fma, tensor cores) for MFU calculation.
# Source: manufacturer specs. Add entries as needed.
GPU_PEAK_TFLOPS = {
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA RTX 4090": 165.2,
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA A100-PCIE-40GB": 312.0,
    "NVIDIA A100 80GB PCIe": 312.0,
    "NVIDIA H100 SXM5": 989.0,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA L40S": 362.0,
    "NVIDIA A6000": 155.0,
}


@dataclass
class BenchResult:
    model: str
    tier: str
    seq_len: int
    batch_size: int
    n_params: int
    tok_per_s: float
    avg_step_ms: float
    peak_vram_gb: float
    steps: int
    compile: bool
    grad_ckpt: bool
    flops_per_tok: float = 0.0   # estimated FLOPs per token (6*N_params)
    tflops_achieved: float = 0.0 # measured TFLOPS
    mfu: float = 0.0             # model FLOPs utilization (%)
    error: str = ""


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "   ", "STEP": " > ", "OK": "[+]", "ERR": "[!]", "===": "==="}
    print(f"{ts} {prefix.get(level, '   ')} {msg}", flush=True)


def log_header(msg: str):
    print(f"\n{'='*70}", flush=True)
    log(msg, "===")
    print(f"{'='*70}", flush=True)


def log_progress(step: int, total: int, tok_per_s: float, label: str = ""):
    bar_len = 20
    filled = int(bar_len * step / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = step / total * 100
    extra = f"  {tok_per_s/1e3:.1f}K tok/s" if tok_per_s > 0 else ""
    label_str = f" {label}" if label else ""
    print(f"\r  [{bar}] {step}/{total} ({pct:.0f}%){extra}{label_str}    ",
          end="", flush=True)


def _cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_neuromorphic(tier: str, seq_len: int, device: torch.device):
    """Create neuromorphic model for given tier."""
    from src.model.config import ModelConfig
    from src.model.model import NeuromorphicLM

    tier_fn = {"A": ModelConfig.tier_a, "B": ModelConfig.tier_b,
               "C": ModelConfig.tier_c}
    if tier not in tier_fn:
        raise ValueError(f"Unknown tier: {tier}")
    cfg = tier_fn[tier](N=seq_len)

    cfg.set_phase("A")
    cfg.vocab_size = VOCAB
    cfg.eot_id = 2
    cfg.use_compile = False  # we handle compile separately

    model = NeuromorphicLM(cfg).to(device)
    return model, cfg


def create_baseline(model_name: str, device: torch.device):
    """Create a baseline model from train_baseline.py configs."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                    "..", "auxiliary_repos", "baselines", "eval_scripts"))
    from train_baseline import MODEL_CONFIGS

    cfg_spec = MODEL_CONFIGS[model_name]
    model_type = cfg_spec["model_type"]

    if model_type == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(**cfg_spec["config_kwargs"])
        model = GPT2LMHeadModel(config)
    elif model_type == "gpt_neox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        config = GPTNeoXConfig(**cfg_spec["config_kwargs"])
        model = GPTNeoXForCausalLM(config)
    elif model_type == "mamba":
        from transformers import MambaConfig, MambaForCausalLM
        config = MambaConfig(**cfg_spec["config_kwargs"])
        model = MambaForCausalLM(config)
    elif model_type == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(**cfg_spec["config_kwargs"])
        model = LlamaForCausalLM(config)
    elif model_type == "rwkv7":
        from transformers import AutoConfig, AutoModelForCausalLM
        hf_repo = cfg_spec["hf_repo"]
        config = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
        config.vocab_size = cfg_spec["config_kwargs"]["vocab_size"]
        config.dtype = "bfloat16"
        config.fuse_cross_entropy = False
        config.fuse_linear_cross_entropy = False
        config.use_l2warp = False
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    use_grad_ckpt = model_type in GRAD_CKPT_TYPES
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()

    return model, model_type, use_grad_ckpt


# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------

def _run_neuromorphic_step(model, cfg, x, y, optimizer, bs, device):
    """Run one training step for neuromorphic model."""
    from src.data.streaming import StreamBatch
    from src.training.trainer import TBPTTTrainer

    if not hasattr(model, '_bench_trainer'):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        model._bench_trainer = TBPTTTrainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            dataloader=iter(()), config=cfg, device=device,
            collector=None, log_interval=10_000,
        )

    batch = StreamBatch(
        input_ids=x, target_ids=y,
        prev_token=torch.zeros(bs, dtype=torch.long, device=device),
    )
    model._bench_trainer.train_chunk(batch)
    model._bench_trainer.global_step += 1


def _run_baseline_step(model, x, y, optimizer):
    """Run one training step for a baseline HF model."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids=x, labels=y).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


# ---------------------------------------------------------------------------
# Batch size finder
# ---------------------------------------------------------------------------

def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _cleanup_model(model):
    if model is not None:
        if hasattr(model, '_bench_trainer'):
            del model._bench_trainer
        del model


def find_max_bs(
    model_name: str, tier: str, seq_len: int, device: torch.device,
    is_neuro: bool, model_type: str | None = None,
    max_bs: int | None = None,
) -> int:
    """Try descending batch sizes until one fits.

    Probes WITHOUT torch.compile for speed — eager mode uses slightly less
    memory than compiled, so we subtract one BS step as safety margin.
    Runs 2 steps per candidate to catch fragmentation.

    If max_bs is given, skips candidates above it (e.g. reuse the BS found
    at a shorter seq_len as the ceiling for a longer one).
    """
    PROBE_STEPS = 2
    found_bs = None

    candidates = [bs for bs in BS_CANDIDATES if max_bs is None or bs <= max_bs]
    for bs in candidates:
        _cleanup()
        model = None
        optimizer = None
        try:
            if is_neuro:
                model, cfg = create_neuromorphic(tier, seq_len, device)
                # No compile — eager mode for fast probing
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            else:
                model, _, _ = create_baseline(model_name, device)
                # No compile — eager mode for fast probing
                optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

            for _ in range(PROBE_STEPS):
                x = torch.randint(0, VOCAB, (bs, seq_len), device=device)
                y = torch.randint(0, VOCAB, (bs, seq_len), device=device)
                if is_neuro:
                    _run_neuromorphic_step(model, cfg, x, y, optimizer, bs, device)
                else:
                    _run_baseline_step(model, x, y, optimizer)
                del x, y

            peak = torch.cuda.max_memory_allocated() / 1e9
            log(f"  BS={bs} OK (eager, peak {peak:.1f}GB)")
            found_bs = bs

            _cleanup_model(model)
            del optimizer
            _cleanup()
            break

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                log(f"  BS={bs} OOM")
                _cleanup_model(model)
                if optimizer is not None:
                    del optimizer
                _cleanup()
                continue
            raise

    if found_bs is None:
        log("  All batch sizes OOM! Returning BS=1 as fallback.", "ERR")
        return 1

    # torch.compile uses extra memory for kernel caches and intermediates.
    # Step down one level from the eager-mode max as safety margin.
    use_compile = is_neuro or (model_type in COMPILE_TYPES)
    if use_compile and found_bs > 1:
        idx = BS_CANDIDATES.index(found_bs)
        if idx + 1 < len(BS_CANDIDATES):
            safe_bs = BS_CANDIDATES[idx + 1]
            log(f"  Compile safety margin: {found_bs} -> {safe_bs}")
            return safe_bs

    return found_bs


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def bench_model(
    model_name: str,
    tier: str,
    seq_len: int,
    device: torch.device,
    fixed_bs: int | None = None,
    max_bs: int | None = None,
) -> BenchResult:
    """Benchmark a single model at a given seq_len."""
    is_neuro = model_name.startswith("neuromorphic")
    use_compile = False
    use_grad_ckpt = False
    model_type = None

    log(f"Model: {model_name} | T={seq_len} | Tier {tier}")
    _cleanup()

    try:
        # --- Determine model properties ---
        if is_neuro:
            use_compile = True
            model_type = "neuromorphic"
        else:
            cfg_spec_path = os.path.join(os.path.dirname(__file__),
                            "..", "auxiliary_repos", "baselines", "eval_scripts")
            sys.path.insert(0, cfg_spec_path)
            from train_baseline import MODEL_CONFIGS
            model_type = MODEL_CONFIGS[model_name]["model_type"]
            use_compile = model_type in COMPILE_TYPES
            use_grad_ckpt = model_type in GRAD_CKPT_TYPES

        # --- Quick param count ---
        if is_neuro:
            tmp_model, _ = create_neuromorphic(tier, seq_len, torch.device("cpu"))
        else:
            tmp_model, _, _ = create_baseline(model_name, torch.device("cpu"))
        n_params = sum(p.numel() for p in tmp_model.parameters())
        del tmp_model
        gc.collect()

        log(f"  Params: {n_params:,} ({n_params/1e6:.1f}M) | "
            f"compile={'yes' if use_compile else 'no'} | "
            f"grad_ckpt={'yes' if use_grad_ckpt else 'no'}")

        # --- Find batch size ---
        if fixed_bs:
            bs = fixed_bs
            log(f"  Batch size: {bs} (fixed)")
        else:
            ceiling = f" (ceiling={max_bs})" if max_bs else ""
            log(f"  Finding max batch size...{ceiling}")
            bs = find_max_bs(model_name, tier, seq_len, device,
                             is_neuro, model_type, max_bs)
            log(f"  Batch size: {bs} (auto)")

        # --- Create fresh model for benchmark ---
        _cleanup()
        log(f"  Creating fresh model for benchmark...")

        if is_neuro:
            model, cfg = create_neuromorphic(tier, seq_len, device)
            if use_compile:
                log(f"  Compiling (first step will be slow)...")
                model = torch.compile(model)
                cfg.use_compile = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        else:
            model, _, use_grad_ckpt = create_baseline(model_name, device)
            if use_compile:
                log(f"  Compiling (first step will be slow)...")
                model = torch.compile(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)

        # --- Warmup (includes compile on first step) ---
        log(f"  Warmup: {WARMUP_STEPS} steps @ BS={bs}...")
        for i in range(WARMUP_STEPS):
            x = torch.randint(0, VOCAB, (bs, seq_len), device=device)
            y = torch.randint(0, VOCAB, (bs, seq_len), device=device)
            if is_neuro:
                _run_neuromorphic_step(model, cfg, x, y, optimizer, bs, device)
            else:
                _run_baseline_step(model, x, y, optimizer)
            log_progress(i + 1, WARMUP_STEPS, 0, "warmup")
        print(flush=True)
        log(f"  Warmup done.")

        # --- Timed runs ---
        log(f"  Benchmarking: {BENCH_STEPS} steps...")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []

        for i in range(BENCH_STEPS):
            x = torch.randint(0, VOCAB, (bs, seq_len), device=device)
            y = torch.randint(0, VOCAB, (bs, seq_len), device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if is_neuro:
                _run_neuromorphic_step(model, cfg, x, y, optimizer, bs, device)
            else:
                _run_baseline_step(model, x, y, optimizer)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            tok_s = bs * seq_len / (t1 - t0)
            log_progress(i + 1, BENCH_STEPS, tok_s, "bench")

        print(flush=True)

        avg_step = sum(times) / len(times)
        std_step = (sum((t - avg_step) ** 2 for t in times) / len(times)) ** 0.5
        tok_per_s = bs * seq_len / avg_step
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        # FLOPs estimation: standard 6*N approximation (fwd + bwd + optimizer)
        flops_per_tok = 6 * n_params
        tflops_achieved = (flops_per_tok * tok_per_s) / 1e12
        gpu_name_local = torch.cuda.get_device_name(0)
        gpu_peak = GPU_PEAK_TFLOPS.get(gpu_name_local, 0)
        mfu = (tflops_achieved / gpu_peak * 100) if gpu_peak > 0 else 0

        result = BenchResult(
            model=model_name, tier=tier, seq_len=seq_len,
            batch_size=bs, n_params=n_params,
            tok_per_s=tok_per_s,
            avg_step_ms=round(avg_step * 1000, 1),
            peak_vram_gb=round(peak_vram, 2),
            steps=BENCH_STEPS,
            compile=use_compile,
            grad_ckpt=use_grad_ckpt,
            flops_per_tok=flops_per_tok,
            tflops_achieved=round(tflops_achieved, 1),
            mfu=round(mfu, 1),
        )

        log(f"  DONE: {tok_per_s/1e3:.1f}K tok/s | "
            f"BS={bs} | {peak_vram:.1f}GB VRAM | "
            f"{avg_step*1000:.1f} +/- {std_step*1000:.1f} ms/step | "
            f"{tflops_achieved:.1f} TFLOPS | MFU={mfu:.1f}%", "OK")

        # Cleanup
        if hasattr(model, '_bench_trainer'):
            del model._bench_trainer
        del model, optimizer
        _cleanup()
        return result

    except Exception as e:
        log(f"  FAILED: {e}", "ERR")
        traceback.print_exc()
        _cleanup()
        return BenchResult(
            model=model_name, tier=tier, seq_len=seq_len,
            batch_size=0, n_params=0, tok_per_s=0,
            avg_step_ms=0, peak_vram_gb=0, steps=0,
            compile=False, grad_ckpt=False, error=str(e),
        )


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results_table(results: list[BenchResult]):
    """Print a clean comparison table."""
    log_header("THROUGHPUT COMPARISON")

    tiers = sorted(set(r.tier for r in results))
    for tier in tiers:
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue

        print(f"\n  Tier {tier}:", flush=True)
        print(f"  {'Model':<20} {'Params':>8} {'T=1024':>12} {'T=2048':>12} "
              f"{'BS@2K':>6} {'VRAM@2K':>8} {'TFLOPS':>7} {'MFU':>6}",
              flush=True)
        print(f"  {'-'*85}", flush=True)

        # Our model first, then baselines sorted alphabetically
        model_names = sorted(set(r.model for r in tier_results),
                             key=lambda m: (not m.startswith("neuro"), m))
        for name in model_names:
            model_results = {r.seq_len: r for r in tier_results if r.model == name}
            r1k = model_results.get(1024)
            r2k = model_results.get(2048)
            ref = r2k or r1k
            if not ref or ref.error:
                err = (ref.error[:40] if ref else "skipped")
                print(f"  {name:<20} {'ERR':>8} {'':>12} {'':>12} "
                      f"{'':>6} {'':>8} {'':>7} {'':>6}  {err}",
                      flush=True)
                continue

            params_str = f"{ref.n_params/1e6:.0f}M"
            t1k_str = f"{r1k.tok_per_s/1e3:.1f}K" if r1k and not r1k.error else "---"
            t2k_str = f"{r2k.tok_per_s/1e3:.1f}K" if r2k and not r2k.error else "---"
            bs2k = str(r2k.batch_size) if r2k and not r2k.error else "---"
            vram = f"{r2k.peak_vram_gb:.1f}GB" if r2k and not r2k.error else "---"
            tflops = f"{ref.tflops_achieved:.1f}" if ref.tflops_achieved > 0 else "---"
            mfu = f"{ref.mfu:.1f}%" if ref.mfu > 0 else "---"

            print(f"  {name:<20} {params_str:>8} {t1k_str:>12} {t2k_str:>12} "
                  f"{bs2k:>6} {vram:>8} {tflops:>7} {mfu:>6}",
                  flush=True)

    # Cost extrapolation table
    print(f"\n  --- Cost Extrapolation (1.5B tokens) ---", flush=True)
    print(f"  {'Model':<20} {'Tier':>4} {'T=1024':>12} {'T=2048':>12}", flush=True)
    print(f"  {'-'*52}", flush=True)

    # Deduplicate: one row per model, both seq lens
    seen = set()
    for r in results:
        if r.model in seen:
            continue
        seen.add(r.model)
        model_results = {o.seq_len: o for o in results
                         if o.model == r.model and not o.error and o.tok_per_s > 0}
        if not model_results:
            continue

        def _fmt_hours(tok_s):
            h = 1_500_000_000 / tok_s / 3600
            return f"{h:.1f}h" if h < 100 else f"{h:.0f}h"

        t1k = _fmt_hours(model_results[1024].tok_per_s) if 1024 in model_results else "---"
        t2k = _fmt_hours(model_results[2048].tok_per_s) if 2048 in model_results else "---"
        print(f"  {r.model:<20} {r.tier:>4} {t1k:>12} {t2k:>12}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global WARMUP_STEPS, BENCH_STEPS  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="Benchmark training throughput across all models"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to benchmark (e.g., neuromorphic-a pythia-160m)"
    )
    parser.add_argument(
        "--tiers", nargs="+", default=["A"],
        choices=["A", "B", "C"],
        help="Model tiers to benchmark (default: A)"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=SEQ_LENGTHS,
        help="Sequence lengths to test (default: 1024 2048)"
    )
    parser.add_argument(
        "--bs", type=int, default=None,
        help="Force a specific batch size (default: auto-detect max)"
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_STEPS,
        help=f"Warmup steps (default: {WARMUP_STEPS})"
    )
    parser.add_argument(
        "--steps", type=int, default=BENCH_STEPS,
        help=f"Benchmark steps (default: {BENCH_STEPS})"
    )
    parser.add_argument(
        "--json", type=str, default="",
        help="Save results to JSON file"
    )
    args = parser.parse_args()

    WARMUP_STEPS = args.warmup
    BENCH_STEPS = args.steps

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for benchmarking.", flush=True)
        sys.exit(1)

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_free = (torch.cuda.mem_get_info(0)[0]) / 1e9

    log_header("THROUGHPUT BENCHMARK")
    log(f"GPU: {gpu_name} ({gpu_vram:.1f}GB total, {gpu_free:.1f}GB free)")
    if gpu_free < gpu_vram * 0.8:
        log(f"WARNING: Only {gpu_free:.1f}/{gpu_vram:.1f}GB free! "
            f"Other processes may cause OOM. Kill them for accurate results.", "ERR")
    log(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    log(f"Seq lengths: {args.seq_lens}")
    log(f"Tiers: {args.tiers}")
    log(f"Steps: {WARMUP_STEPS} warmup + {BENCH_STEPS} timed")

    # Build model list
    if args.models:
        models_to_run = []
        for m in args.models:
            for t, tier_models in TIER_MODELS.items():
                if m in tier_models:
                    models_to_run.append((m, t))
                    break
            else:
                models_to_run.append((m, "A"))
    else:
        models_to_run = []
        for tier in args.tiers:
            for model_name in TIER_MODELS.get(tier, []):
                models_to_run.append((model_name, tier))

    total_runs = len(models_to_run) * len(args.seq_lens)
    log(f"Models: {[m for m, _ in models_to_run]}")
    log(f"Total runs: {total_runs}")
    print(flush=True)

    # Run benchmarks
    results = []
    done = 0
    # Cache: reuse BS from shorter seq_len as ceiling for longer ones
    bs_cache = {}  # model_name -> best BS found so far

    seq_lens_sorted = sorted(args.seq_lens)  # shortest first
    for model_name, tier in models_to_run:
        for seq_len in seq_lens_sorted:
            done += 1
            log_header(f"[{done}/{total_runs}] {model_name} @ T={seq_len}")

            # Use previous BS as ceiling (longer seq can only fit <= shorter seq BS)
            ceiling = bs_cache.get(model_name)

            result = bench_model(
                model_name=model_name,
                tier=tier,
                seq_len=seq_len,
                device=device,
                fixed_bs=args.bs,
                max_bs=ceiling,
            )
            results.append(result)

            # Cache the found BS for this model
            if result.batch_size > 0 and not result.error:
                bs_cache[model_name] = result.batch_size

            time.sleep(2)  # GPU cooldown

    # Print summary
    print_results_table(results)

    # Save JSON
    if args.json:
        out_dir = os.path.dirname(args.json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        output = {
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_name,
            "gpu_vram_gb": round(gpu_vram, 1),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        log(f"Results saved to {args.json}", "OK")

    log_header("DONE")


if __name__ == "__main__":
    main()
