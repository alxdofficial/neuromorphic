#!/usr/bin/env python3
"""Memory training harness: every memory variant × objective, with teacher-forced CE.

Default objective = masked_reconstruction (sentence-pair MAE); other objectives
(qa, conditioned_reconstruction[_bio], continuation) select via --task.

For each variant:
 - Encoder ingests a context via streaming writes → memory tokens.
 - Decoder forward on [memory, question, answer]. The original context
   tokens are NOT visible to the decoder — only memory carries that info.
 - TF-CE on the answer's content-mask positions (load-bearing tokens; the
   rest of the answer span is filler).

Per-variant outputs in outputs/memory/<out_tag>_<variant>/:
  jsonl/<variant>.jsonl   — per-step training metrics
  ckpts/<variant>.last.pt — encoder + decoder.mask_embed weights

Thin entrypoint (harness reorg): the reusable harness now lives in
``src.memory.training`` and the CLI in ``scripts.train.cli``; ``main`` only
orchestrates parse → build model/tokenizer → trainer loop → save.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.data.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS
from src.memory.data.mixes import DEFAULT_TRAIN_MIX, TASK_SPEC
from src.memory.decoder import load_frozen_llama
from src.memory.training import (
    train_one_variant, train_mixed_variant, probe_bs,
    make_mixed_val_sets, run_val, _continuation_early_loss, to_device,
    _infonce_logits_weights, _same_answer_valid_mask, _grad_cached_objective_step,
)
from scripts.train.cli import build_parser, args_to_config

# back-compat re-exports for diagnostics (removed in Phase 4). Diagnostics still do
# `from scripts.train.train import <name>` (and a few `from train import <name>` under sys.path
# hacks); keep every such name resolvable here until those imports are repointed.
MIXED_TASK_MODE = {n: s.task_mode for n, s in TASK_SPEC.items()}
MIXED_TASKS_DEFAULT = DEFAULT_TRAIN_MIX

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = build_parser()
    args = ap.parse_args()
    cfg, composite_task_weights = args_to_config(args, ap)
    M = args.mem_tokens

    print(f"\nLoading tokenizer {cfg.llama_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading Llama (shared across variants, frozen)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    if args.probe_bs:
        print(f"\n[probe-bs] per-arm max batch size, conditioned-reconstruction N={args.cond_recon_n_pairs}, "
              f"chunk={args.chunk_size}, mem_tokens={M}")
        probe_bs(args.variants, llama, tokenizer, cfg, args)
        return

    summaries = []
    # MIXED multi-task: vanilla floor/ceiling have no trainable params in the mixed
    # round-robin (they are single-task eval-only references) — skip them.
    if args.task == "mixed":
        mixed_variants = [v for v in args.variants if not v.startswith("vanilla_")]
        skipped = [v for v in args.variants if v.startswith("vanilla_")]
        if skipped:
            print(f"[mixed] skipping eval-only vanilla references: {skipped}")
        for variant in mixed_variants:
            out_dir = REPO / f"outputs/memory/{args.out_tag}_{variant}"
            out_dir.mkdir(parents=True, exist_ok=True)
            s = train_mixed_variant(
                variant=variant, llama=llama, tokenizer=tokenizer, cfg=cfg,
                n_steps=args.steps, log_every=args.log_every,
                val_every=args.val_every, save_every=args.save_every,
                val_batches=args.val_batches, out_dir=out_dir,
                window_size=args.window_size,
                mixed_tasks=tuple(args.mixed_tasks), mixed_ctx=args.mixed_ctx,
                mixed_M=args.mixed_M, babi_tasks=tuple(args.babi_tasks),
                predict_len=args.predict_len, mae_src_tok=args.src_tokenizer,
                resume=args.resume, train_seed=args.seed,
            )
            summaries.append(s)
    else:
        for variant in args.variants:
            out_dir = REPO / f"outputs/memory/{args.out_tag}_{variant}"
            out_dir.mkdir(parents=True, exist_ok=True)
            s = train_one_variant(
                variant=variant, llama=llama, tokenizer=tokenizer, cfg=cfg,
                n_steps=args.steps, log_every=args.log_every,
                val_every=args.val_every, save_every=args.save_every,
                val_batches=args.val_batches, out_dir=out_dir,
                chunk_size=args.chunk_size, window_size=args.window_size,
                passages_per_chunk=args.passages_per_chunk,
                resume=args.resume,
                use_hotpot=not args.no_hotpot,
                use_narrative=args.narrative,
                use_musique=args.musique,
                use_babilong=args.babilong,
                babilong_config=args.babilong_config,
                mix_weights=tuple(args.mix_weights),
                composite_task_weights=composite_task_weights,
                patience=args.patience,
                min_step_for_stop=args.min_step_for_stop,
                min_delta=args.early_stop_min_delta,
                task=args.task,
                cond_recon_n_pairs=args.cond_recon_n_pairs, cond_recon_n_query=args.cond_recon_n_query,
                cond_recon_value_len=args.cond_recon_value_len,
                cond_recon_bio_n_facts=args.cond_recon_bio_n_facts, cond_recon_bio_world_seed=args.cond_recon_bio_world_seed,
                babi_tasks=tuple(args.babi_tasks),
                compress_len=args.compress_len, predict_len=args.predict_len,
                mae_src_tok=args.src_tokenizer,
            )
            summaries.append(s)

    print("\n" + "=" * 78)
    print("v1h SUMMARY")
    print("=" * 78)
    if args.task == "mixed":
        print(f"  {'variant':<25}{'params':>12}   per-task final val_loss")
        print("  " + "-" * 70)
        for s in summaries:
            pt = "  ".join(f"{t}={s['per_task_final'][t]['val_loss']:.3f}"
                           for t in s["mixed_tasks"])
            print(f"  {s['variant']:<25}{s['trainable_params']:>12,}   {pt}")
    else:
        print(f"  {'variant':<25}{'params':>12}{'final_recon':>13}{'top1':>8}{'time(min)':>12}")
        print("  " + "-" * 70)
        for s in summaries:
            print(f"  {s['variant']:<25}{s['trainable_params']:>12,}"
                  f"{s['final_val_loss_recon']:>13.4f}"
                  f"{s['final_val_top1_acc']*100:>7.1f}%"
                  f"{s['elapsed_s']/60:>12.1f}")

    summary_path = REPO / f"outputs/memory/{args.out_tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
