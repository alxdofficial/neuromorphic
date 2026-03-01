"""Efficiency measurement utilities for model benchmarking.

Pure PyTorch — no external model dependencies (transformers, mamba_ssm, etc.).
Model-specific factories live in the bench script.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Default English sample for avg_bytes_per_token (no data deps)
# ---------------------------------------------------------------------------
_DEFAULT_SAMPLE = (
    "The quick brown fox jumps over the lazy dog. In the beginning was the "
    "Word, and the Word was with God, and the Word was God. It is a truth "
    "universally acknowledged, that a single man in possession of a good "
    "fortune, must be in want of a wife. Call me Ishmael. Some years ago, "
    "never mind how long precisely, having little or no money in my purse, "
    "and nothing particular to interest me on shore, I thought I would sail "
    "about a little and see the watery part of the world. It was the best "
    "of times, it was the worst of times, it was the age of wisdom, it was "
    "the age of foolishness. All happy families are alike; each unhappy "
    "family is unhappy in its own way. In a hole in the ground there lived "
    "a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms "
    "and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it "
    "to sit down on or to eat: it was a hobbit-hole, and that means comfort."
)


# ---------------------------------------------------------------------------
# EfficiencyReport
# ---------------------------------------------------------------------------
@dataclass
class EfficiencyReport:
    """All efficiency metrics for a single model configuration."""

    model_name: str
    param_count: int
    train_tok_per_sec: float
    infer_tok_per_sec: float
    flops_per_token_fwd: int
    peak_vram_train_gb: float
    vram_weights_gb: float
    vram_optimizer_gb: float
    vram_activations_gb: float
    batch_size: int
    seq_len: int
    device_name: str
    predicted_tok_per_sec: float = 0.0  # predicted effective tok/s for generation
    bpb: float | None = None
    ms_per_step_train: float = 0.0
    ms_per_step_infer: float = 0.0


# ---------------------------------------------------------------------------
# FLOPs measurement
# ---------------------------------------------------------------------------
def measure_flops_per_token(
    model: nn.Module,
    input_ids: Tensor,
    forward_fn: Callable[[nn.Module, Tensor], Tensor],
) -> dict[str, int]:
    """Count forward-pass FLOPs using torch.utils.flop_counter.FlopCounterMode.

    For the neuromorphic model, forward_fn should call forward_segment which
    includes all R passes — the counter sees them all.

    Must run with use_compile=False (compiled ops fuse ATen ops that the
    counter hooks into).

    Returns:
        {"total_flops": int, "flops_per_token": int}
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    # Support both [BS, N] and [BS, N, D] inputs
    n_tokens = input_ids.shape[0] * input_ids.shape[1]

    flop_counter = FlopCounterMode(display=False)
    with flop_counter, torch.no_grad():
        forward_fn(model, input_ids)

    total = flop_counter.get_total_flops()
    return {
        "total_flops": total,
        "flops_per_token": total // max(n_tokens, 1),
    }


# ---------------------------------------------------------------------------
# Training throughput
# ---------------------------------------------------------------------------
def measure_training_throughput(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    bs: int,
    seq_len: int,
    vocab: int,
    device: torch.device,
    warmup: int = 5,
    measure: int = 20,
    detach_fn: Callable[[nn.Module], None] | None = None,
    step_fn: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Full fwd+bwd+step timing.

    When ``step_fn`` is provided it replaces the internal NTP step entirely —
    the caller is responsible for forward, loss, backward, optimizer step, and
    detach.  ``forward_fn`` / ``optimizer`` / ``detach_fn`` are unused in that
    case.

    Returns:
        {"tok_per_sec": float, "ms_per_step": float, "peak_vram_gb": float}
    """
    model.train()

    if step_fn is None:
        input_ids = torch.randint(0, vocab, (bs, seq_len), device=device)

        def _step():
            optimizer.zero_grad()
            logits = forward_fn(model, input_ids)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab).float(),
                input_ids[:, 1:].reshape(-1),
            )
            loss.backward()
            optimizer.step()
            if detach_fn is not None:
                detach_fn(model)
    else:
        _step = step_fn

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        _step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(measure):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens = bs * seq_len * measure
    tok_s = tokens / elapsed
    ms = elapsed / measure * 1000

    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9

    return {"tok_per_sec": tok_s, "ms_per_step": ms, "peak_vram_gb": peak_vram}


# ---------------------------------------------------------------------------
# Inference throughput
# ---------------------------------------------------------------------------
def measure_inference_throughput(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tensor], Tensor],
    bs: int,
    seq_len: int,
    vocab: int,
    device: torch.device,
    warmup: int = 5,
    measure: int = 20,
    detach_fn: Callable[[nn.Module], None] | None = None,
) -> dict[str, float]:
    """Forward-only (mode, no_grad) prefill throughput.

    Returns:
        {"tok_per_sec": float, "ms_per_step": float}
    """
    model.eval()
    input_ids = torch.randint(0, vocab, (bs, seq_len), device=device)

    def _step():
        with torch.no_grad():
            forward_fn(model, input_ids)
        if detach_fn is not None:
            detach_fn(model)

    for _ in range(warmup):
        _step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(measure):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens = bs * seq_len * measure
    return {
        "tok_per_sec": tokens / elapsed,
        "ms_per_step": elapsed / measure * 1000,
    }


# ---------------------------------------------------------------------------
# VRAM breakdown
# ---------------------------------------------------------------------------
def measure_vram_breakdown(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tensor], Tensor],
    optimizer_cls: type,
    bs: int,
    seq_len: int,
    vocab: int,
    device: torch.device,
    detach_fn: Callable[[nn.Module], None] | None = None,
    step_fn: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Three-phase VRAM measurement.

    1. After model.to(device) -> weights
    2. After first optimizer step -> + optimizer state
    3. Peak during training -> + activations/grads

    When ``step_fn`` is provided it replaces the internal NTP forward+loss+
    backward+step for phases 2 and 3.  The caller must ensure the step_fn
    uses the same optimizer (created externally) so optimizer state is
    captured correctly.

    Returns:
        {"weights_gb": float, "optimizer_gb": float,
         "activations_gb": float, "peak_gb": float}
    """
    if device.type != "cuda":
        return {
            "weights_gb": 0.0,
            "optimizer_gb": 0.0,
            "activations_gb": 0.0,
            "peak_gb": 0.0,
        }

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Phase 1: weights only
    torch.cuda.reset_peak_memory_stats(device)
    weights_bytes = torch.cuda.memory_allocated(device)

    model.train()

    if step_fn is not None:
        # step_fn encapsulates fwd+loss+bwd+step+detach
        # Phase 2: materialize optimizer state
        step_fn()

        gc.collect()
        torch.cuda.empty_cache()
        after_opt_bytes = torch.cuda.memory_allocated(device)

        # Phase 3: peak during second step
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:
        # Phase 2: add optimizer
        optimizer = optimizer_cls(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab, (bs, seq_len), device=device)

        # Run one step to materialize optimizer state
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if detach_fn is not None:
            detach_fn(model)

        # After step: optimizer state is materialized, grads cleared next zero_grad
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        after_opt_bytes = torch.cuda.memory_allocated(device)

        # Phase 3: peak during training (second step)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if detach_fn is not None:
            detach_fn(model)
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)

        del optimizer, input_ids, logits, loss

    gc.collect()
    torch.cuda.empty_cache()

    weights_gb = weights_bytes / 1e9
    optimizer_gb = (after_opt_bytes - weights_bytes) / 1e9
    peak_gb = peak_bytes / 1e9
    activations_gb = peak_gb - (after_opt_bytes / 1e9)

    return {
        "weights_gb": weights_gb,
        "optimizer_gb": optimizer_gb,
        "activations_gb": max(activations_gb, 0.0),
        "peak_gb": peak_gb,
    }


# ---------------------------------------------------------------------------
# BPB (Bits per byte)
# ---------------------------------------------------------------------------
def compute_bpb(ce_loss_nats: float, avg_bytes_per_token: float) -> float:
    """Convert CE loss (nats) to bits-per-byte.

    BPB = CE / ln(2) / avg_bytes_per_token
    """
    if ce_loss_nats == 0.0:
        return 0.0
    return ce_loss_nats / math.log(2) / avg_bytes_per_token


def compute_avg_bytes_per_token(
    tokenizer: Any,
    sample_texts: list[str] | None = None,
) -> float:
    """Compute average UTF-8 bytes per token for a tokenizer.

    Args:
        tokenizer: anything with an ``encode(text) -> list[int]`` method.
        sample_texts: texts to sample from. Uses a default English passage if None.

    Returns:
        Average bytes per token (float).
    """
    if sample_texts is None:
        sample_texts = [_DEFAULT_SAMPLE]

    total_bytes = 0
    total_tokens = 0
    for text in sample_texts:
        encoded = tokenizer.encode(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(encoded)

    if total_tokens == 0:
        return 1.0
    return total_bytes / total_tokens


# ---------------------------------------------------------------------------
# Formatting / output
# ---------------------------------------------------------------------------
def format_comparison_table(reports: list[EfficiencyReport]) -> str:
    """Format reports as an ASCII table suitable for terminal or paper."""
    if not reports:
        return "(no reports)"

    has_pred = any(r.predicted_tok_per_sec > 0 for r in reports)

    headers = [
        "Model",
        "Params",
        "Train tok/s",
    ]
    if has_pred:
        headers.append("Pred tok/s")
    headers += [
        "Infer tok/s",
        "FLOPs/tok",
        "Peak VRAM",
        "Wt/Opt/Act GB",
        "BPB",
    ]

    rows: list[list[str]] = []
    for r in reports:
        row = [
            r.model_name,
            f"{r.param_count / 1e6:.1f}M",
            f"{r.train_tok_per_sec:,.0f}",
        ]
        if has_pred:
            row.append(
                f"{r.predicted_tok_per_sec:,.0f}" if r.predicted_tok_per_sec > 0 else "-"
            )
        row += [
            f"{r.infer_tok_per_sec:,.0f}",
            _fmt_flops(r.flops_per_token_fwd),
            f"{r.peak_vram_train_gb:.1f} GB",
            f"{r.vram_weights_gb:.1f}/{r.vram_optimizer_gb:.1f}/{r.vram_activations_gb:.1f}",
            f"{r.bpb:.3f}" if r.bpb is not None else "-",
        ]
        rows.append(row)

    # Column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 0:
                parts.append(cell.ljust(widths[i]))
            else:
                parts.append(cell.rjust(widths[i]))
        return " | ".join(parts)

    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt_row(headers), sep]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def _fmt_flops(flops: int) -> str:
    """Human-readable FLOPs (e.g. 1.2G, 450M)."""
    if flops >= 1e12:
        return f"{flops / 1e12:.1f}T"
    if flops >= 1e9:
        return f"{flops / 1e9:.1f}G"
    if flops >= 1e6:
        return f"{flops / 1e6:.1f}M"
    if flops >= 1e3:
        return f"{flops / 1e3:.1f}K"
    return str(flops)


def save_reports_json(reports: list[EfficiencyReport], path: str | Path) -> None:
    """Write reports to a JSON file."""
    data = [asdict(r) for r in reports]
    Path(path).write_text(json.dumps(data, indent=2))
