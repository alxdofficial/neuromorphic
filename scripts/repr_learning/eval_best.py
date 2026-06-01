#!/usr/bin/env python3
"""Re-evaluate a saved best.pt on a freshly-materialized fixed val set.

This is the "trustworthy number" script — fixes the two eval pathologies in
the trainer's in-line eval:
  (a) val sampling variance (#614) — uses the same fixed-val protocol
  (b) wrong-checkpoint final-eval — explicitly loads best.pt

Usage:
  python -m scripts.repr_learning.eval_best \\
      --variant graph_baseline \\
      --ckpt outputs/repr_learning/v1h_t4k_v2_graph_baseline/ckpts/graph_baseline.best.pt \\
      --chunk-size 4096 \\
      --val-batches 32

If --ckpt is omitted for vanilla_llama or vanilla_full_context, runs eval-only
on a fresh model (those have no trainable encoder).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_qa import make_mixed_qa_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel

from scripts.repr_learning.train_repr_qa import (
    materialize_val_set, run_val, to_device,
    COMPOSITE_VAL_P, COMPOSITE_VAL_Q,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="path to best.pt. REQUIRED for trainable variants; "
                         "optional for vanilla_llama and vanilla_full_context "
                         "(which have no trainable encoder).")
    # Defaults MATCH train_repr_qa.py CLI defaults exactly. If the trained
    # ckpt was made with different flags (e.g. tranche-1-v2 explicitly passed
    # --narrative --mix-weights 0.5 0.25 0.25), pass the SAME flags to eval.
    # See audit issue: eval_best.py defaults previously mismatched trainer →
    # eval was on a different protocol than training → numbers were not
    # directly comparable.
    # Defaults updated post-tranche-3 to match the hard-only protocol:
    # chunk_size=8192, Narrative+MuSiQue on, mix 0.30/0.25/0.25/0.20/0.
    # For pre-tranche-3 ckpts (4096, no Narrative), pass legacy flags.
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--window-size", type=int, default=1024,
                    help="MATCHES trainer default. 16 windows/chunk at 256 vs "
                         "4 windows at 1024 — significantly different streaming "
                         "dynamics for variants with per-window state (graph u-EMA, "
                         "plastic fast weights, splat blob updates).")
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override cfg.batch_size. Useful for memory-heavy variants "
                         "(plastic with chunk_size=4096 OOMs at BS=2; needs BS=1).")
    ap.add_argument("--narrative", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable NarrativeQA source. Default ON (matches "
                         "tranche-3+ trainer default). Pass --no-narrative for "
                         "ckpts trained without narrative_qa in the mix.")
    ap.add_argument("--musique", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable MuSiQue source. Default ON (matches "
                         "tranche-3+ trainer default).")
    ap.add_argument("--mix-weights", type=float, nargs="+",
                    default=[0.30, 0.25, 0.25, 0.20, 0.0],
                    metavar="W",
                    help="Sampling weights: composite, hotpot, narrative, musique, babilong. "
                         "Default MATCHES tranche-3+ trainer (hard-only). "
                         "Pre-tranche-3 ckpts: pass --mix-weights 0.7 0.3 0 0 0.")
    ap.add_argument("--passages-per-chunk", type=int, default=0,
                    help="0 = auto-scale by trainer's formula max(75, (chunk_size//1024)*75) "
                         "(= 300 at 4096, 600 at 8192, 1200 at 16384). MATCHES trainer.")
    ap.add_argument("--seed", type=int, default=7,
                    help="Same seed as trainer val_dl → identical batches → identical numbers")
    ap.add_argument("--json-out", default=None, help="optional: dump result as JSON to this path")
    args = ap.parse_args()

    # Validation: trainable variants MUST have a ckpt. Without one we'd eval
    # random-init weights and silently produce garbage (audit issue #5).
    TRAINABLE = {"flat_baseline", "continuous_baseline", "memorizing_baseline",
                 "recurrent_baseline", "plastic_baseline", "splat_baseline",
                 "graph_baseline", "graph_v5_baseline", "v21"}
    if args.variant in TRAINABLE and args.ckpt is None:
        raise SystemExit(
            f"ERROR: --ckpt is required for trainable variant {args.variant!r}. "
            f"Evaluating random init weights would silently produce garbage. "
            f"Pass --ckpt path/to/best.pt or use vanilla_llama/vanilla_full_context."
        )

    cfg = ReprConfig()
    # Match the v1h training-time config overrides (see scripts/repr_learning/verify_v1h.py).
    # Without these the saved ckpts won't load (size mismatches on n_flat_codes,
    # d_mamba, pos_embed). These were the values used to train the tranche 1 v2
    # baselines; mismatching them silently loads default-init tensors for the
    # missed weights → garbage output.
    cfg.n_flat_codes = 36
    cfg.d_mamba = 768
    cfg.max_window_size = max(cfg.max_window_size, args.chunk_size)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.passages_per_chunk == 0:
        # MATCH trainer's formula in train_repr_qa.py: max(75, (chunk_size//1024)*75)
        # gives 300 @ chunk=4096, 600 @ 8192, 1200 @ 16384.
        # The old formula max(80, chunk_size//50) gave 81 @ 4096 — way too sparse
        # vs trainer, so the val passage-density was off and per-passage QA difficulty
        # didn't match training conditions.
        args.passages_per_chunk = max(75, (args.chunk_size // 1024) * 75)

    print(f"=== {args.variant} ===")
    if args.ckpt:
        print(f"  ckpt: {args.ckpt}")
    print(f"  chunk_size={args.chunk_size}  window={args.window_size}  val_batches={args.val_batches}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading Llama-3.2-1B...")
    llama, _ = load_frozen_llama(cfg.llama_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)

    model = ReprLearningModel(cfg, variant=args.variant, llama_model=llama).to(device)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(sd["model_state_dict"], strict=False)
        n_loaded = len(sd["model_state_dict"]) - len(unexpected)
        print(f"  loaded {n_loaded} tensors from ckpt (step={sd.get('step','?')})")

        # Surface mismatches (audit issue #6). Previously strict=False silently
        # swallowed both kinds; missed missing encoder weights = eval'ing random
        # init for some layers, missed unexpected = ckpt has stale params under
        # an old architecture name.
        # Allowed-missing: Llama (frozen, not in ckpt), buffers that aren't
        # persisted. Anything ELSE missing is a real problem.
        suspicious_missing = [
            k for k in missing
            if not k.startswith("llama.") and not k.startswith("decoder.")
        ]
        if suspicious_missing:
            print(f"\n  WARN: {len(suspicious_missing)} encoder/head tensors are "
                  f"MISSING from ckpt — these are random-init in eval:")
            for k in suspicious_missing[:10]:
                print(f"    - {k}")
            if len(suspicious_missing) > 10:
                print(f"    ... +{len(suspicious_missing)-10} more")
            raise SystemExit(
                "Refusing to eval with random-init encoder tensors. "
                "Either the ckpt is from a different architecture or the "
                "current model has new parameters that weren't trained."
            )
        if unexpected:
            print(f"\n  WARN: {len(unexpected)} tensors in ckpt have NO HOME in "
                  f"current model — probably a renamed/removed module:")
            for k in unexpected[:10]:
                print(f"    - {k}")
            if len(unexpected) > 10:
                print(f"    ... +{len(unexpected)-10} more")
            raise SystemExit(
                "Refusing to eval with mismatched architecture. The ckpt was "
                "trained against a different code revision; either revert the "
                "model code or retrain on the current revision."
            )

    print(f"  building val set...")
    val_dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=COMPOSITE_VAL_P,
        composite_questions_path=COMPOSITE_VAL_Q,
        use_hotpot=True, use_narrative=args.narrative,
        use_musique=False, use_babilong=False,
        babilong_config="4k",
        split="validation",
        chunk_size=args.chunk_size, passages_per_chunk=args.passages_per_chunk,
        weights=args.mix_weights, num_workers=0, seed=args.seed,
    )
    val_set = materialize_val_set(val_dl, args.val_batches)
    print(f"  fixed val set: {len(val_set)} batches")

    t0 = time.time()
    result = run_val(model, val_set, device, args.val_batches, args.window_size)
    elapsed = time.time() - t0

    print(f"\n  val_loss_recon: {result['val_loss_recon']:.4f}")
    print(f"  val_top1_acc:   {result['val_top1_acc']*100:.1f}%")
    print(f"  val_n_batches:  {result['val_n_batches']}")
    print(f"  elapsed:        {elapsed:.1f}s")
    print(f"\n  Per-family loss (smallest n=1, treat sparse families with caution):")
    fam_sorted = sorted(result["val_per_family"].items(), key=lambda x: x[1]["mean_loss"])
    for fam, d in fam_sorted:
        print(f"    {fam:18}  n={d['n']:3d}  loss={d['mean_loss']:.3f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fp:
            json.dump({
                "variant": args.variant,
                "ckpt": args.ckpt,
                "step": sd.get("step") if args.ckpt else None,
                "chunk_size": args.chunk_size,
                "val_batches": args.val_batches,
                "seed": args.seed,
                **result,
            }, fp, indent=2)
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
