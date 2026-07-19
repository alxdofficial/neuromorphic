#!/usr/bin/env python3
"""Phase-2 Tier-2 baseline: LCLM — Latent Context Language Models over LongMemEval (POD-ONLY).

LCLM (Li, McLeish et al., "End-to-End Context Compression at Scale", arXiv:2606.09659) is the closest
concurrent competitor to our architecture family: an encoder (Qwen3-Embedding-0.6B) pools token chunks into
SOFT TOKENS, an adapter projects them, and a 4B decoder (Qwen3-4B-Instruct-2507) consumes them as latent
context. Released weights: HF `latent-context/0.6b-4b-LCLM-{4x,8x,16x}` (compression ratio). Unlike our
FROZEN-decoder design, LCLM trains the decoder end-to-end (frozen-init → continual pretrain → SFT) — the
axis to engage in related work (see docs/baselines/PHASE2_BASELINES.md §2.5).

⚠ LICENSE: none stated on the repo/HF org as of 2026-07-18 — clearance is the user's call; this file is the
technical wiring only. ⚠ The published checkpoints are NOT loadable via vanilla transformers/vllm — the LCLM
repo (github.com/LeonLixyz/LCLM) must be cloned and on PYTHONPATH (`--repo-dir`), and the long context MUST be
wrapped in `<|memory_start|> ... <|memory_end|>` (the encoder boundary markers).

Per-item INFERENCE compression (no per-corpus training) → fits LongMemEval's private-haystack shape directly:
encode each item's ~115k history to soft tokens, decode the answer. Scores with the SAME deterministic scorer
as Tier-1. RESUMABLE + crash-safe (per-item ResultStore). `--help` works without torch/the LCLM repo (lazy).

VRAM: ~4.6B params ≈ 9–10GB bf16 → fits one 24GB GPU, inference-only. Example (on the pod):
  python scripts/baselines/tier2/run_lclm.py --checkpoint latent-context/0.6b-4b-LCLM-16x --max-examples 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_DEFAULT_CHECKPOINT = "latent-context/0.6b-4b-LCLM-16x"   # highest compression; swap 16x→8x/4x to trade fidelity
_DEFAULT_REPO_DIR = "~/tier2_repos/LCLM"                  # git clone https://github.com/LeonLixyz/LCLM


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _record(it, hyp="", error=None):
    return {"question_id": it["question_id"], "question": it["question"], "answer": it["answer"],
            "hypothesis": hyp, "question_type": it["question_type"],
            "finish_reason": "error" if error else "stop", "error": error}


def run_lclm(args, items, checkpoint: str, repo_dir: str, store) -> None:
    """Load LCLM once, then per item: wrap the dated history in memory markers, encode→decode the answer.
    Exact entry points from the LCLM repo's inference/hf.py (load_model / generate_text)."""
    sys.path.insert(0, repo_dir)
    # POD-ONLY: needs the LCLM repo on PYTHONPATH (checkpoints aren't vanilla-transformers loadable).
    from inference.hf import load_model, generate_text

    model, dec_tok, processor = load_model(checkpoint, device="cuda", dtype="bf16")

    done = store.done_ids()
    for it in items:
        if str(it["question_id"]) in done:
            continue
        try:
            # context MUST be wrapped in the encoder boundary markers; the question follows outside them.
            # anchor temporal questions to their date, like the Tier-1 panel.
            q = it["question"]
            if it.get("question_date"):
                q = f"Current Date: {it['question_date']}\n{q}"
            prompt = f"<|memory_start|>{it['full_history']}<|memory_end|> {q}"
            hyp = generate_text(model, dec_tok, processor, prompt,
                                max_tokens=args.max_new_tokens, temperature=0.0)
            store.append(_record(it, hyp=hyp))
        except Exception as e:  # noqa: BLE001 — crash-safe: record, continue, resume later
            print(f"[run_lclm] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(_record(it, error=f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT,
                    help=f"HF checkpoint (default {_DEFAULT_CHECKPOINT}); 4x/8x/16x = compression ratio")
    ap.add_argument("--repo-dir", default=_DEFAULT_REPO_DIR, help="path to the cloned LCLM repo (PYTHONPATH)")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (EM+containment only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    repo_dir = str(Path(args.repo_dir).expanduser())

    # --- everything below needs torch/the LCLM repo — lazy on purpose (see docstring: --help works without) ---
    _seed_everything(args.seed)
    from src.memory.data.longmemeval import load_longmemeval_text
    from src.memory.eval import score_longmemeval
    from src.memory.eval.results import ResultStore

    print(f"[run_lclm] checkpoint={args.checkpoint} repo_dir={repo_dir} variant={args.variant} "
          f"max_examples={args.max_examples} seed={args.seed}")
    items = load_longmemeval_text(variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_lclm] {len(items)} items; types={types}")

    commit = _git_commit()
    ckpt_slug = args.checkpoint.split("/")[-1]
    tag = (f"longmemeval__lclm__{ckpt_slug}__{args.variant}"
           f"__n{len(items)}__g{args.max_new_tokens}__seed{args.seed}__{commit}")
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_lclm] resume: {n_done}/{len(items)} already done — generating the rest")

    run_lclm(args, items, args.checkpoint, repo_dir, store)

    records = [r for r in store.all_records() if not r.get("error")]
    agg = score_longmemeval(records, use_bem=not args.no_bem)
    store.merge_verdicts(agg.get("details", [])); store.compact()
    n_err = sum(1 for r in store.all_records() if r.get("error"))
    print(f"\n[run_lclm] overall_acc={agg.get('overall_accuracy', float('nan')):.3f}  "
          f"task_avg={agg.get('task_averaged_accuracy', float('nan')):.3f}  "
          f"abstention={agg.get('abstention_accuracy')}  n={agg.get('n_nonabstention')}  errors={n_err}")

    payload = {
        "dataset": "longmemeval", "method": "lclm", "model": args.checkpoint,
        "meta": {"n": len(records), "n_errors": n_err, "variant": args.variant, "seed": args.seed,
                 "max_new_tokens": args.max_new_tokens, "commit": commit,
                 "coverage": round(len(records) / len(items), 4) if items else None},
        "aggregate": {k: v for k, v in agg.items() if k != "details"},
        "store": str(store.path),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"[run_lclm] wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
