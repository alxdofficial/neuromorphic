#!/usr/bin/env python3
"""End-to-end smoke test: 100-200 steps with real data on each variant.

Trains a ReprLearningModel (encoder + frozen Llama decoder) for a short
number of optimizer steps on a 50/50 mix of FineWeb-edu + composite_v1.

Reports loss curve, memory dispersion progression, gradient stability.
Used to verify the full pipeline (data -> encoder -> Llama -> loss ->
backward -> optimizer step) works end-to-end before launching long runs.

Per-step health metrics are logged as JSONL via ReprMetrics. The companion
script scripts/diagnostics/plot_repr.py renders a multi-panel diagnostic
figure from those logs.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_real import make_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.metrics import compute_metrics
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB = REPO / "data/wave1/fineweb_edu.train.parquet"
COMPOSITE = REPO / "data/wave1/composite_v1/train/passages.jsonl"


def memory_dispersion(memory: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity within each sample's 96 tokens.
    Lower = more dispersed (good); higher = collapsed (bad)."""
    m = F.normalize(memory.float(), dim=-1)  # [B, M, d]
    B, M, _ = m.shape
    cos = m @ m.transpose(1, 2)              # [B, M, M]
    mask = ~torch.eye(M, dtype=torch.bool, device=cos.device)
    off_diag = cos[:, mask].reshape(B, -1)
    return off_diag.mean().item()


def train_variant(
    variant: str,
    llama,
    cfg: ReprConfig,
    n_steps: int,
    log_every: int = 10,
    jsonl_path: Path | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 70}\nVariant: {variant}\n{'=' * 70}")

    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    n_trainable = model.n_trainable_params()
    print(f"  trainable params: {n_trainable:,}")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    dl = make_dataloader(
        cfg, FINEWEB, COMPOSITE,
        fineweb_ratio=0.5, num_workers=0, seed=42,
    )

    history: list[dict] = []
    jsonl_fp = open(jsonl_path, "a") if jsonl_path else None
    t_start = time.time()
    step = 0

    for batch in dl:
        if step >= n_steps:
            break

        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        mask_positions = batch.mask_positions.to(device, non_blocking=True)

        t_step = time.time()
        opt.zero_grad()
        out = model(input_ids, attention_mask, mask_positions)
        loss = out["loss"]

        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {loss.item()}")
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(), cfg.grad_clip,
        )

        # Compute metrics AFTER backward + grad-clip, BEFORE opt.step()
        # so grads are still attached to parameters.
        step_ms = (time.time() - t_step) * 1000.0
        text_tok = input_ids.shape[0] * input_ids.shape[1]
        text_tok_per_sec = text_tok / max(step_ms / 1000.0, 1e-9)
        metrics = compute_metrics(
            model=model,
            out=out,
            memory=out["memory"],
            aux=out["aux"],
            grad_norm=grad_norm.item(),
            step=step,
            variant=variant,
            step_ms=step_ms,
            text_tok_per_sec=text_tok_per_sec,
        )

        opt.step()

        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps(asdict(metrics)) + "\n")
            jsonl_fp.flush()

        if step % log_every == 0 or step == n_steps - 1:
            print(
                f"  step {step:4d}  recon={metrics.loss_recon:6.3f}  "
                f"aux={metrics.loss_aux:5.2f}  gnorm={metrics.grad_norm:5.2f}  "
                f"disp={metrics.mem_dispersion:.3f}  "
                f"H={metrics.routing_entropy:5.2f}  "
                f"uniq={metrics.unique_codes_per_batch:4d}  "
                f"({metrics.step_ms:.0f}ms/step)"
            )
            history.append(asdict(metrics))

        step += 1

    if jsonl_fp is not None:
        jsonl_fp.close()
    total_t = time.time() - t_start
    print(f"  Done in {total_t:.1f}s ({step / total_t:.1f} steps/s)")

    summary = {
        "variant": variant,
        "n_trainable": n_trainable,
        "n_steps": step,
        "loss_recon_first": history[0]["loss_recon"] if history else None,
        "loss_recon_last": history[-1]["loss_recon"] if history else None,
        "loss_recon_delta": (
            history[-1]["loss_recon"] - history[0]["loss_recon"]
            if len(history) >= 2 else None
        ),
        "dispersion_first": history[0]["mem_dispersion"] if history else None,
        "dispersion_last": history[-1]["mem_dispersion"] if history else None,
        "total_time_s": total_t,
        "history": history,
    }

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "v21", "flat_baseline", "continuous_baseline",
        "memorizing_baseline", "recurrent_baseline",
    ])
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="outputs/repr_learning/smoke.json")
    ap.add_argument("--jsonl-dir", type=str,
                    default="outputs/repr_learning/jsonl",
                    help="Directory for per-variant JSONL training logs")
    args = ap.parse_args()

    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=256,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nConfig: batch_size={cfg.batch_size}, "
          f"window={cfg.fixed_window_size}, n_steps={args.n_steps}")

    if device == "cpu" and "recurrent_baseline" in args.variants:
        print("[warn] dropping recurrent_baseline on CPU (Mamba needs CUDA)")
        args.variants = [v for v in args.variants if v != "recurrent_baseline"]

    print(f"\nLoading Llama (shared across variants)...")
    llama_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=llama_dtype)

    jsonl_dir = REPO / args.jsonl_dir
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for variant in args.variants:
        jsonl_path = jsonl_dir / f"{variant}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()  # fresh log per run
        summary = train_variant(
            variant, llama, cfg, args.n_steps, args.log_every,
            jsonl_path=jsonl_path,
        )
        all_summaries[variant] = summary

    print(f"\n{'=' * 70}\nFINAL SUMMARY\n{'=' * 70}")
    print(f"{'variant':<25}{'n_train':>10}{'recon_0':>10}{'recon_T':>10}"
          f"{'Δrecon':>10}{'disp_T':>10}")
    print("-" * 75)
    for v, s in all_summaries.items():
        d = s.get("loss_recon_delta")
        d_str = f"{d:+.3f}" if d is not None else "N/A"
        print(
            f"{v:<25}{s['n_trainable']:>10,}"
            f"{s['loss_recon_first']:>10.3f}{s['loss_recon_last']:>10.3f}"
            f"{d_str:>10}{s['dispersion_last']:>10.3f}"
        )
    print("=" * 75)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
