#!/usr/bin/env python3
"""v1g training: sentence-level shuffled-retrieval reconstruction.

For each variant:
 - Encoder ingests 4096-token chunk via 4 × 1024 streaming writes
   (V2.1 deferred: uses single-pass for now until per-sentence design lands)
 - Per queried sentence (K=3 per chunk), decoder reconstructs the
   still-masked positions from (memory + visible + revealed)
 - Restricted attention: still-masked sees only visible + self
 - Random-reveal curriculum (MaskGIT-style, r ∈ [0, 0.9])

Bottleneck scaled to ~26k floats at d_node_state (20× compression of
4096-token input). MT and V2.1 excluded — they need separate handling.

Per-variant outputs in outputs/repr_learning/v1g_<variant>/:
  jsonl/<variant>.jsonl   — per-step training metrics
  ckpts/<variant>.last.pt — encoder + decoder.mask_embed weights
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_sentence import make_sentence_chunk_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_TRAIN = REPO / "data/wave1/fineweb_edu.train.parquet"
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)


@torch.no_grad()
def run_val(model, val_dl, device, n_batches: int = 10, window_size: int = 1024) -> dict:
    model.train(False)
    losses, masked_counts = [], []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        batch.input_ids = batch.input_ids.to(device, non_blocking=True)
        batch.attention_mask = batch.attention_mask.to(device, non_blocking=True)
        batch.query_input_ids = batch.query_input_ids.to(device, non_blocking=True)
        batch.mask_positions = batch.mask_positions.to(device, non_blocking=True)
        batch.reveal_positions = batch.reveal_positions.to(device, non_blocking=True)
        batch.query_lengths = batch.query_lengths.to(device, non_blocking=True)
        batch.query_starts = batch.query_starts.to(device, non_blocking=True)
        out = model.compute_sentence_recon_loss(batch, window_size=window_size)
        losses.append(float(out["loss_recon"]))
        masked_counts.append(int(out["n_still_masked"]))
    model.train(True)
    n = max(len(losses), 1)
    return {
        "val_loss_recon": sum(losses) / n,
        "val_mean_still_masked": sum(masked_counts) / n,
        "val_n_batches": len(losses),
    }


def save_checkpoint(model, opt, step, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def keep(k: str) -> bool:
        # Drop frozen Llama base weights to keep ckpt small. Keep LoRA params.
        if not k.startswith("decoder.llama."):
            return True
        return "lora_" in k

    torch.save({
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
    }, path)


def train_one_variant(
    variant: str, llama, tokenizer, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path, chunk_size: int, window_size: int,
    n_queries: int, mask_ratio: float, reveal_lo: float, reveal_hi: float,
    sentence_min_len: int, sentence_max_len: int, resume: bool = False,
) -> dict:
    device = "cuda"
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    n_trainable = model.n_trainable_params()
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps)")
    print(f"{'='*78}")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    train_dl = make_sentence_chunk_dataloader(
        cfg, fineweb_path=FINEWEB_TRAIN, tokenizer=tokenizer,
        chunk_size=chunk_size, n_queries=n_queries,
        mask_ratio=mask_ratio, reveal_lo=reveal_lo, reveal_hi=reveal_hi,
        sentence_min_len=sentence_min_len, sentence_max_len=sentence_max_len,
        num_workers=0, seed=42,
    )
    val_dl = make_sentence_chunk_dataloader(
        cfg, fineweb_path=FINEWEB_VAL, tokenizer=tokenizer,
        chunk_size=chunk_size, n_queries=n_queries,
        mask_ratio=mask_ratio, reveal_lo=reveal_lo, reveal_hi=reveal_hi,
        sentence_min_len=sentence_min_len, sentence_max_len=sentence_max_len,
        num_workers=0, seed=7,
    )

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load ckpt if requested and it exists; otherwise start fresh.
    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_result = model.load_state_dict(sd["model_state_dict"], strict=False)
        opt.load_state_dict(sd["optimizer_state_dict"])
        start_step = int(sd.get("step", 0)) + 1
        n_missing_non_llama = sum(
            1 for k in load_result.missing_keys if not k.startswith("decoder.llama.")
        )
        print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1}; "
              f"missing(non-llama)={n_missing_non_llama}, "
              f"unexpected={len(load_result.unexpected_keys)}")
    else:
        if jsonl_path.exists():
            jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    last_print_step, last_print_time = start_step, t_start

    for step, batch in enumerate(train_dl):
        if step < start_step:
            continue
        if step >= n_steps:
            break

        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        # Move batch tensors to device
        batch.input_ids = batch.input_ids.to(device, non_blocking=True)
        batch.attention_mask = batch.attention_mask.to(device, non_blocking=True)
        batch.query_input_ids = batch.query_input_ids.to(device, non_blocking=True)
        batch.mask_positions = batch.mask_positions.to(device, non_blocking=True)
        batch.reveal_positions = batch.reveal_positions.to(device, non_blocking=True)
        batch.query_lengths = batch.query_lengths.to(device, non_blocking=True)
        batch.query_starts = batch.query_starts.to(device, non_blocking=True)

        opt.zero_grad()
        out = model.compute_sentence_recon_loss(batch, window_size=window_size)
        loss = out["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        opt.step()

        row = {
            "step": step,
            "variant": variant,
            "loss": float(out["loss"]),
            "loss_recon": float(out["loss_recon"]),
            "loss_aux": float(out["loss_aux"]),
            "n_still_masked": int(out["n_still_masked"]),
            "n_revealed": int(out["n_revealed"]),
            "n_visible_in_query": int(out["n_visible_in_query"]),
            "grad_norm": float(gn),
            "lr": lr,
            "memory_M": out["memory"].shape[1],
        }
        # B-specific diagnostics: track diversity to see if it actually drops
        aux = out.get("aux", {})
        if "diversity_slots_raw" in aux:
            row["diversity_slots"] = float(aux["diversity_slots_raw"])
            row["diversity_mem"] = float(aux["diversity_mem_raw"])
        jsonl_fp.write(json.dumps(row) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            print(f"  step {step:6d}/{n_steps}  recon={float(out['loss_recon']):.4f}  "
                  f"aux={float(out['loss_aux']):.3f}  "
                  f"masked={int(out['n_still_masked']):4d}  "
                  f"gnorm={float(gn):6.2f}  lr={lr:.2e}  ({sps:.1f} step/s)",
                  flush=True)

        if step > 0 and step % val_every == 0:
            vm = run_val(model, val_dl, device, val_batches, window_size=window_size)
            val_row = {"phase": "val", "step": step, "variant": variant, **vm}
            jsonl_fp.write(json.dumps(val_row) + "\n")
            print(f"    [val @ {step}]  recon={vm['val_loss_recon']:.4f}",
                  flush=True)

        if step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, ckpt_path)

    save_checkpoint(model, opt, step, ckpt_path)
    final_val = run_val(model, val_dl, device, val_batches, window_size=window_size)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_loss_recon={final_val['val_loss_recon']:.4f}", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "n_steps": step,
        "elapsed_s": elapsed,
        "final_val_loss_recon": final_val["val_loss_recon"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    # Variants — V2.1 still excluded (needs per-sentence streaming design).
    # MT now uses per-sentence retrieval cap K = n_flat_codes (= 36).
    ap.add_argument("--variants", nargs="+", default=[
        "flat_baseline", "continuous_baseline", "memorizing_baseline",
        "recurrent_baseline", "graph_baseline", "vanilla_llama",
    ])
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--mask-ratio", type=float, default=0.8)
    ap.add_argument("--reveal-lo", type=float, default=0.0)
    ap.add_argument("--reveal-hi", type=float, default=0.9)
    ap.add_argument("--sentence-min-len", type=int, default=8)
    ap.add_argument("--sentence-max-len", type=int, default=80)
    ap.add_argument("--b-diversity-scale", type=float, default=50.0,
                    help="B's diversity-loss multiplier (was 1000 in v1e/v1f)")
    ap.add_argument("--mt-diversity-scale", type=float, default=50.0,
                    help="MT's diversity-loss multiplier (was 1000 in v1e)")
    ap.add_argument("--out-tag", type=str, default="v1g",
                    help="Prefix for outputs/repr_learning/<tag>_<variant>/")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from each variant's last.pt if present (append to JSONL)")
    args = ap.parse_args()

    # v1g doesn't support v21 (per-sentence streaming not built)
    if "v21" in args.variants:
        raise SystemExit("v21 is not supported in v1g — per-sentence streaming "
                         "design pending. Pick from flat_baseline, "
                         "continuous_baseline, memorizing_baseline, "
                         "recurrent_baseline, vanilla_llama.")

    # v1g bottleneck scale-up:
    #   target ~26k floats at d_node_state level (4096 × 128 / 20)
    #   - Baselines: 36 slots × 725 = 26,100 floats  ✓
    #   - V2.1 (when added later): 68 edges × 384 fused = 26,112 floats
    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,      # encoder pos_embed needs chunk_size capacity
        max_steps=args.steps,
        warmup_steps=500,
        d_node_state=128,
        n_edges=68,                            # for V2.1 when wired in
        n_flat_codes=36,                       # baselines: 36 × 725 ≈ 26k floats
        edge_token_packing="fused",
        b_diversity_scale=args.b_diversity_scale,
        mt_diversity_scale=args.mt_diversity_scale,
        # v1g fix: align Mamba param count with the other variants
        # (default d_mamba=1024 made Mamba ~25% heavier than A/B/MT;
        #  d_mamba=512 overshot the other way to ~50% lighter; 768 lands
        #  closer to A/B/MT's ~14M trainable.)
        d_mamba=768,
    )

    print(f"v1g config: chunk={args.chunk_size}, window={args.window_size}, "
          f"n_queries={args.n_queries}, mask_ratio={args.mask_ratio}, "
          f"reveal=[{args.reveal_lo}, {args.reveal_hi}]")
    print(f"Bottleneck (baselines): {cfg.n_flat_codes} × {cfg.d_continuous} "
          f"= {cfg.n_flat_codes * cfg.d_continuous} floats")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")

    print(f"\nLoading tokenizer {cfg.llama_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    print("Loading Llama (shared across variants, frozen)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    summaries = []
    for variant in args.variants:
        out_dir = REPO / f"outputs/repr_learning/{args.out_tag}_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        s = train_one_variant(
            variant=variant, llama=llama, tokenizer=tokenizer, cfg=cfg,
            n_steps=args.steps, log_every=args.log_every,
            val_every=args.val_every, save_every=args.save_every,
            val_batches=args.val_batches, out_dir=out_dir,
            chunk_size=args.chunk_size, window_size=args.window_size,
            n_queries=args.n_queries, mask_ratio=args.mask_ratio,
            reveal_lo=args.reveal_lo, reveal_hi=args.reveal_hi,
            sentence_min_len=args.sentence_min_len,
            sentence_max_len=args.sentence_max_len,
            resume=args.resume,
        )
        summaries.append(s)

    print("\n" + "=" * 78)
    print("v1g SUMMARY")
    print("=" * 78)
    print(f"  {'variant':<25}{'params':>12}{'final_val_recon':>17}{'time(min)':>12}")
    print("  " + "-" * 66)
    for s in summaries:
        print(f"  {s['variant']:<25}{s['trainable_params']:>12,}"
              f"{s['final_val_loss_recon']:>17.4f}{s['elapsed_s']/60:>12.1f}")

    summary_path = REPO / "outputs/repr_learning/v1g_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
