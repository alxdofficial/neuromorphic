#!/usr/bin/env python3
"""Zero-memory ablation: for each trained checkpoint, run val twice (with
real memory, with memory zeroed) and report the gap.

A large gap (real ≪ zeroed) means memory is doing real work. A small gap
means the architecture is a soft-prompt / parametric overfit and the
encoder is largely irrelevant. The vanilla_llama floor is the absolute
parametric Llama score.

Usage:
  python scripts/repr_learning/eval_zero_mem.py \
    --val-batches 20 --batch-size 2 \
    --pairs flat_baseline:outputs/repr_learning/v1h_X/ckpts/flat_baseline.last.pt \
            graph_baseline:outputs/repr_learning/v1h_Y/ckpts/graph_baseline.last.pt \
    --output outputs/repr_qa/zero_mem.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_qa import make_mixed_qa_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel
from scripts.repr_learning.train_repr_qa import materialize_val_set, to_device


@torch.no_grad()
def run_val_pass(model, val_set, device, n_batches: int, window_size: int,
                 zero_memory: bool) -> Dict:
    """Run val over a fixed materialized list of batches. Critical: pass the
    SAME list to both `normal` and `zero` calls so the two scores compare
    apples-to-apples. Previously this took a streaming val_dl and the second
    call drained a different chunk of the stream — gaps were noisy and biased.
    """
    model.train(False)
    losses, accs, per_fam = [], [], {}
    for i, batch in enumerate(val_set):
        if i >= n_batches:
            break
        batch = to_device(batch, device)
        out = model.compute_qa_loss(batch, window_size=window_size,
                                    zero_memory=zero_memory)
        losses.append(float(out["loss_recon"]))
        accs.append(float(out["top1_acc"]))
        if "per_example_loss" in out:
            per_ex = out["per_example_loss"].detach().cpu().tolist()
        else:
            per_ex = [float(out["loss_recon"])] * len(batch.task_family)
        for fam, l in zip(batch.task_family, per_ex):
            d = per_fam.setdefault(fam, {"n": 0, "loss": 0.0})
            d["n"] += 1
            d["loss"] += float(l)
    n = max(len(losses), 1)
    fam_summary = {f: {"n": v["n"], "mean_loss": v["loss"] / max(v["n"], 1)}
                   for f, v in per_fam.items()}
    return {
        "val_loss_recon": sum(losses) / n,
        "val_top1_acc": sum(accs) / n,
        "val_n_batches": len(losses),
        "val_per_family": fam_summary,
    }


def evaluate_variant(variant: str, ckpt_path: Path, llama, tokenizer,
                     cfg: ReprConfig, val_set, device, n_batches: int,
                     window_size: int) -> Dict:
    print(f"\n=== {variant}  ({ckpt_path}) ===", flush=True)
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = sd.get("model_state_dict", sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # missing keys are expected (we don't save Llama params) — only flag if
    # encoder-specific keys are missing
    enc_missing = [k for k in missing if k.startswith("encoder.")
                                       or k.startswith("decoder.mask_embed")]
    if enc_missing:
        print(f"  WARN: missing encoder/mask_embed keys: {enc_missing[:5]}"
              f"{'...' if len(enc_missing) > 5 else ''}", flush=True)
    if unexpected:
        print(f"  WARN: unexpected keys: {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}", flush=True)

    # Both passes over the SAME materialized list — gap is now a pure
    # within-batch comparison, not contaminated by streaming sampling noise.
    normal = run_val_pass(model, val_set, device, n_batches, window_size,
                          zero_memory=False)
    zero = run_val_pass(model, val_set, device, n_batches, window_size,
                        zero_memory=True)
    del model
    torch.cuda.empty_cache()

    # Per-family gap
    fam_gap = {}
    for fam in normal["val_per_family"]:
        n_loss = normal["val_per_family"][fam]["mean_loss"]
        z_loss = zero["val_per_family"].get(fam, {}).get("mean_loss")
        if z_loss is not None:
            fam_gap[fam] = {
                "normal": n_loss,
                "zero":   z_loss,
                "gap":    z_loss - n_loss,
                "n":      normal["val_per_family"][fam]["n"],
            }

    result = {
        "variant": variant,
        "ckpt": str(ckpt_path),
        "normal_recon": normal["val_loss_recon"],
        "normal_top1":  normal["val_top1_acc"],
        "zero_recon":   zero["val_loss_recon"],
        "zero_top1":    zero["val_top1_acc"],
        "gap_recon":    zero["val_loss_recon"] - normal["val_loss_recon"],
        "n_batches":    normal["val_n_batches"],
        "per_family":   fam_gap,
    }
    print(f"  normal:  recon={result['normal_recon']:.4f}  top1={result['normal_top1']:.1%}", flush=True)
    print(f"  zero:    recon={result['zero_recon']:.4f}  top1={result['zero_top1']:.1%}", flush=True)
    print(f"  gap (zero-normal): {result['gap_recon']:+.4f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="variant:ckpt_path tuples, e.g. "
                         "flat_baseline:outputs/.../flat_baseline.last.pt")
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--passages-per-chunk", type=int, default=300)
    ap.add_argument("--output", type=str, default="outputs/repr_qa/zero_mem.json")
    ap.add_argument("--no-hotpot", action="store_true")
    ap.add_argument("--narrative", action="store_true")
    ap.add_argument("--mix-weights", nargs=3, type=float, default=[0.7, 0.3, 0.0],
                    help="composite, hotpot, narrative weights")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    pairs: List[Tuple[str, Path]] = []
    for p in args.pairs:
        if ":" not in p:
            raise ValueError(f"--pairs entry must be variant:path, got {p!r}")
        var, path = p.split(":", 1)
        pairs.append((var.strip(), Path(path.strip())))

    # Sanity: ckpts exist
    for var, path in pairs:
        if not path.exists():
            raise FileNotFoundError(f"{var}: ckpt {path} not found")

    # IMPORTANT: ReprConfig's default llama_model is now Llama-3.2-1B-Instruct
    # (post-tranche-4). Old v1h checkpoints were trained against base Llama
    # with raw-concat prompts; loading them under Instruct's chat-template
    # scaffold silently invalidates results. Pin the backbone explicitly to
    # the legacy base model when evaluating pre-tranche-4 ckpts.
    BACKBONE_FOR_LEGACY_CKPTS = "meta-llama/Llama-3.2-1B"
    print(f"Loading tokenizer + Llama (frozen, shared) — pinned backbone: "
          f"{BACKBONE_FOR_LEGACY_CKPTS}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_FOR_LEGACY_CKPTS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llama, _ = load_frozen_llama(BACKBONE_FOR_LEGACY_CKPTS, dtype=torch.bfloat16)
    llama = llama.to(device)

    # Build cfg with the SAME overrides train_repr_qa.py main() uses for v1h.
    # Default ReprConfig has older shapes (n_flat_codes=96, max_window=1024)
    # which cause load_state_dict size-mismatch failures on v1h checkpoints.
    cfg = ReprConfig(
        llama_model=BACKBONE_FOR_LEGACY_CKPTS,   # also pin in cfg so chat
                                                  # template build sees no
                                                  # chat_template and skips
                                                  # scaffold (matches training).
        batch_size=args.batch_size,
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,
        max_steps=10_000,
        warmup_steps=500,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=36,
        edge_token_packing="fused",
        d_mamba=768,
    )
    print(f"Building val dataloader: chunk={args.chunk_size} window={args.window_size}", flush=True)
    val_dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=REPO / "data/wave1/composite_v1/val/passages.jsonl",
        composite_questions_path=REPO / "data/wave1/composite_v1/val/questions.jsonl",
        use_hotpot=not args.no_hotpot,
        use_narrative=args.narrative,
        split="validation",
        chunk_size=args.chunk_size,
        passages_per_chunk=args.passages_per_chunk,
        weights=tuple(args.mix_weights),
        batch_size=args.batch_size,
    )
    # Materialize once — all variants and both (normal, zero) passes see the
    # IDENTICAL list of batches. Without this, the streaming val_dl gives a
    # different sample to each pass and gaps get contaminated by sampling
    # variance (same root cause as the trainer-side val noise fix).
    val_set = materialize_val_set(val_dl, args.val_batches)
    print(f"Materialized val_set: {len(val_set)} batches (shared across variants and passes)",
          flush=True)

    results = []
    for var, path in pairs:
        try:
            r = evaluate_variant(var, path, llama, tokenizer, cfg, val_set,
                                 device, args.val_batches, args.window_size)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {var}: {type(e).__name__}: {e}", flush=True)
            results.append({"variant": var, "ckpt": str(path), "error": str(e)})

    out_path = REPO / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "mix_weights": args.mix_weights,
            "use_hotpot": not args.no_hotpot,
            "use_narrative": args.narrative,
            "val_batches": args.val_batches,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'variant':<22} {'normal':>9} {'zero':>9} {'gap':>9} {'n_top1':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<22}  ERROR: {r['error'][:50]}")
            continue
        print(f"{r['variant']:<22} {r['normal_recon']:>9.4f} {r['zero_recon']:>9.4f} "
              f"{r['gap_recon']:>+9.4f} {r['normal_top1']:>8.1%}")
    print("=" * 80)
    print("gap = zero - normal. Large gap → memory carries info. Small gap → memory irrelevant.")


if __name__ == "__main__":
    main()
