#!/usr/bin/env python3
"""bench_vanilla_llama.py — measure vanilla Llama-3.2-1B NTP CE on the
same val parquets the trajectory-memory model is trained against.

Gives us the Llama-only baseline the memory adapter has to beat.
See `docs/research_backlog.md` item #1c.

Usage:
    python scripts/diagnostics/bench_vanilla_llama.py \\
        --val-data-paths data/wave1/needle.val.parquet \\
                         data/wave1/fineweb_edu.val.parquet \\
        --val-batches 100

Outputs:
- Per-source NTP CE (avg over all valid tokens in val)
- Per-source ANSWER-SPAN-only CE (needle only)
- JSON dump for downstream comparison
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.training.loaders import LongDocDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data-paths", nargs="+", required=True)
    ap.add_argument("--val-batches", type=int, default=100,
                    help="Max chunks per source")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/vanilla_llama_baseline.json"))
    ap.add_argument("--context-cap", type=int, default=0,
                    help="If >0, only Llama's last N tokens have attention "
                         "(simulates effective_lm_context cap). Each 1024-token "
                         "chunk independently — we don't thread state, so this "
                         "is per-chunk cap.")
    args = ap.parse_args()

    cfg = TrajMemConfig.medium()  # for T_window=256, D=4 framing

    print(f"Loading vanilla Llama-3.2-1B (no memory module)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.train(False)   # inference mode (equivalent to .eval())
    print(f"  device: {next(model.parameters()).device}, "
          f"dtype: {next(model.parameters()).dtype}")

    results: dict[str, dict] = {}

    for val_path in args.val_data_paths:
        path = Path(val_path)
        source = path.stem.split(".")[0]
        print(f"\n=== Source: {source} ({path}) ===")

        ds = LongDocDataset(
            [path],
            chunk_tokens=cfg.D * cfg.T_window,   # 4 × 256 = 1024 tokens
            pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            drop_short=False,
        )

        total_ce_sum = torch.zeros((), device=args.device, dtype=torch.float32)
        total_token_count = 0.0
        total_answer_ce_sum = torch.zeros((), device=args.device, dtype=torch.float32)
        total_answer_count = 0.0
        n_chunks = 0
        t0 = time.time()

        with torch.no_grad():
            for i, item in enumerate(ds):
                if i >= args.val_batches:
                    break
                if item is None:
                    continue

                chunk = item.input_ids.unsqueeze(0).to(args.device)
                valid_mask = item.valid_mask.to(args.device)

                out = model(input_ids=chunk, use_cache=False)
                logits = out.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk[:, 1:].contiguous()
                shift_mask = valid_mask[1:]

                ce_per_token = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view_as(shift_labels).squeeze(0)

                # Full-doc CE (all valid, weighted)
                token_mask = (shift_mask > 0).float()
                token_weight = shift_mask.float() * token_mask
                ce_sum = (ce_per_token * token_weight).sum()
                ce_count = token_weight.sum()
                total_ce_sum = total_ce_sum + ce_sum.to(total_ce_sum.dtype)
                total_token_count += float(ce_count.item())

                # Answer-span only (synthesize_needle marks answer tokens with weight 100)
                answer_mask = (shift_mask >= 50.0).float()
                if answer_mask.sum() > 0:
                    ans_ce_sum = (ce_per_token * answer_mask).sum()
                    ans_ce_count = answer_mask.sum()
                    total_answer_ce_sum = (
                        total_answer_ce_sum + ans_ce_sum.to(total_answer_ce_sum.dtype)
                    )
                    total_answer_count += float(ans_ce_count.item())
                n_chunks += 1
                if (i + 1) % 20 == 0:
                    cur_ce = (total_ce_sum / max(total_token_count, 1)).item()
                    print(f"  chunk {i+1}: cumulative CE={cur_ce:.4f} "
                          f"({total_token_count:.0f} tokens)")

        dt = time.time() - t0
        full_ce = (total_ce_sum / max(total_token_count, 1)).item()
        ans_ce = (
            (total_answer_ce_sum / max(total_answer_count, 1)).item()
            if total_answer_count > 0 else None
        )
        results[source] = {
            "n_chunks": n_chunks,
            "n_tokens_weighted": total_token_count,
            "ntp_ce_full": full_ce,
            "ntp_ce_answer_only": ans_ce,
            "n_answer_tokens_weighted": total_answer_count,
            "wall_time_s": dt,
        }
        print(f"\n  {source}: full CE = {full_ce:.4f}  "
              f"({total_token_count:.0f} weighted tokens, {n_chunks} chunks, {dt:.1f}s)")
        if ans_ce is not None:
            print(f"  {source}: ANSWER-ONLY CE = {ans_ce:.4f}  "
                  f"({total_answer_count:.0f} answer tokens)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model_name,
            "results": results,
            "config": {
                "val_batches": args.val_batches,
                "T_window": cfg.T_window,
                "D": cfg.D,
            },
        }, f, indent=2)
    print(f"\nSaved: {args.output}")

    print("\n--- Summary ---")
    print(f"{'source':<20} {'full CE':>10} {'answer CE':>12}")
    for src, r in results.items():
        ans = f"{r['ntp_ce_answer_only']:.4f}" if r['ntp_ce_answer_only'] else "—"
        print(f"{src:<20} {r['ntp_ce_full']:>10.4f} {ans:>12}")


if __name__ == "__main__":
    main()
