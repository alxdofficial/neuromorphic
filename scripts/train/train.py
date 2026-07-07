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

from src.memory.decoder import load_frozen_llama
from src.memory.training import train_mixed_variant, probe_bs
from scripts.train.cli import build_parser, args_to_config

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = build_parser()
    args = ap.parse_args()
    cfg = args_to_config(args, ap)
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
    # MIXED multi-task (the only training path): vanilla floor/ceiling have no
    # trainable params in the mixed round-robin (they are eval-only references) — skip them.
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

    print("\n" + "=" * 78)
    print("v1h SUMMARY")
    print("=" * 78)
    print(f"  {'variant':<25}{'params':>12}   per-task final val_loss")
    print("  " + "-" * 70)
    for s in summaries:
        pt = "  ".join(f"{t}={s['per_task_final'][t]['val_loss']:.3f}"
                       for t in s["mixed_tasks"])
        print(f"  {s['variant']:<25}{s['trainable_params']:>12,}   {pt}")

    summary_path = REPO / f"outputs/memory/{args.out_tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
