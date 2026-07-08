#!/usr/bin/env python
"""Peek at episodes through the EXACT training harness data path — decode a sample for eyeballing
and report per-(task × datasource) fill/padding/causality/frequency stats.

Pulls pre-collate samples straight from ``make_mixed_train_dataloaders`` round-robin, exactly as
``loops.py`` trains, so what you see is what the model sees. Use it to sanity-check goldens after any
data change, or to catch fill/padding/frequency regressions.

Usage:
  python scripts/diagnostics/mixed/episode_peek.py [--n-detail 40] [--n-stats 1000]
                                                   [--ctx 2048] [--window 256] [--m 96]
"""
from __future__ import annotations

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from src.memory.config import ReprConfig
from src.memory.data.mixes import DEFAULT_TRAIN_MIX, DEFAULT_MIXED_M
from src.memory.training.data_mix import make_mixed_train_dataloaders

MODEL = "HuggingFaceTB/SmolLM2-135M"
BABI_TASKS = [1, 2, 3, 7, 8, 11, 12, 13, 14]
MAE_SRC_TOK = "meta-llama/Llama-3.2-1B"


def _causal_frac(tok, sample) -> float:
    """Fraction of SCORED answer tokens that also appear in the context token-set — a multi-query-
    robust causality proxy (≈1.0 for qa/recon where the answer is reconstructible from context;
    low for continuation, whose target is the unseen future). Returns -1 for the trivial mae case."""
    if sample["task_family"] == "masked_reconstruction":
        return -1.0
    ctx = set(sample["context_ids"][sample["context_mask"].bool()].tolist())
    scored = [t for t, m in zip(sample["answer_ids"].tolist(), sample["answer_content_mask_list"]) if m]
    if not scored:
        return 0.0
    return sum(1 for t in scored if t in ctx) / len(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-detail", type=int, default=40, help="episodes to decode in full for eyeballing")
    ap.add_argument("--n-stats", type=int, default=1000, help="episodes to aggregate for the stats table")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--m", type=int, default=DEFAULT_MIXED_M)
    ap.add_argument("--predict-len", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    pad = tok.eos_token_id or 0
    cfg = ReprConfig(llama_model=MODEL, pad_token_id=pad)
    cfg.batch_size = 1

    dls = make_mixed_train_dataloaders(DEFAULT_TRAIN_MIX, tok, cfg, ctx_len=args.ctx, m_slots=args.m,
                                       mae_src_tok=MAE_SRC_TOK, babi_tasks=BABI_TASKS,
                                       predict_len=args.predict_len, train_seed=42,
                                       window_size=args.window, bio_query_window=None)
    datasets = {t: dl.dataset for t, dl in dls.items()}
    iters = {t: iter(ds) for t, ds in datasets.items()}
    tasks = list(dls)
    print(f"[loaders] tasks={tasks} ctx={args.ctx} M={args.m} window={args.window}\n")

    def dec(ids):
        return tok.decode(ids)

    def scored(s):
        return tok.decode([t for t, m in zip(s["answer_ids"].tolist(), s["answer_content_mask_list"]) if m])

    stats = collections.defaultdict(lambda: {"n": 0, "in": [], "pad": [], "scored": [], "causal": []})
    for i in range(args.n_stats):
        t = tasks[i % len(tasks)]
        s = next(iters[t])
        ds = s["task_family"]
        valid = int(s["context_mask"].sum())
        r = stats[(t, ds)]
        r["n"] += 1
        r["in"].append(valid)
        r["pad"].append(args.ctx - valid)
        r["scored"].append(sum(s["answer_content_mask_list"]))
        r["causal"].append(_causal_frac(tok, s))

        if i < args.n_detail:
            ctxf = dec(s["context_ids"][s["context_mask"].bool()])
            print(f"{'='*100}\n#{i:03d} task={t} datasource={ds} qtype={s['question_type']} "
                  f"input={valid} pad={args.ctx-valid} scored={sum(s['answer_content_mask_list'])} "
                  f"causal={_causal_frac(tok, s):.2f}")
            print(f"  CTX head: {ctxf[:300].replace(chr(10),'⏎')!r}")
            print(f"  CTX tail: …{ctxf[-200:].replace(chr(10),'⏎')!r}")
            print(f"  Q: {dec(s['question_ids'])!r}")
            print(f"  GOLDEN(scored): {scored(s)[:300]!r}")

    # ---- per (task × datasource) stats ----
    print(f"\n{'task':15} {'datasource':24} {'n':>5} {'freq%':>6} {'in_avg':>7} {'pad%':>6} "
          f"{'scored':>7} {'causal':>7} {'builds/ep':>9}")
    print("-" * 100)
    total = sum(r["n"] for r in stats.values())
    for (t, ds) in sorted(stats):
        r = stats[(t, ds)]
        causal = [c for c in r["causal"] if c >= 0]
        cstr = f"{st.mean(causal):.2f}" if causal else "n/a"
        bpe = datasets[t].n_builds / max(1, datasets[t].n_episodes)   # resample telemetry
        print(f"{t:15} {ds:24} {r['n']:>5} {100*r['n']/total:>5.1f}% {st.mean(r['in']):>7.0f} "
              f"{100*st.mean(r['pad'])/args.ctx:>5.0f}% {st.mean(r['scored']):>7.1f} {cstr:>7} {bpe:>9.2f}")
    print("-" * 100)
    print("causal = frac of scored answer-tokens present in context (≈1 = reconstructible; low continuation = future; n/a mae)")
    print("builds/ep = task's cumulative build calls ÷ episodes (>1 ⇒ resampling; a too-tight spec / heavy padding)")


if __name__ == "__main__":
    main()
