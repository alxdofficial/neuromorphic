#!/usr/bin/env python
"""Per-source data statistics — quantity, input/answer token lengths, padding, answer-in-context.

The "how much padding / how long is the input / how long is the answer" tool. For each
(source × task × EpisodeSpec) it draws N episodes and reports the distributions, so we can see —
per source — how much of the budget is real signal vs padding, and whether sources have drastically
different input/answer sizes (wasted compute). `episode_stats()` is importable for ad-hoc use.

Usage:
  python scripts/diagnostics/mixed/data_stats.py [--group qa|continuation|mae|recon|all]
                                                 [--total-len 1024] [--n 200]
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from src.memory.data.sources import SOURCE_REGISTRY
from src.memory.data.tasks import get_task
from src.memory.data.tasks.base import TaskDataset
from src.memory.data.schedule import EpisodeSpec

# (source, task_style, per-source construction kwargs) — the training sources, individually.
GROUPS = {
    "qa": [
        ("babi", "qa", {}),
        ("squad", "qa", {}),
        ("triviaqa", "qa", {}),
        ("hotpot_train", "qa", {}),
        ("musique_train", "qa", {}),
        ("multiwoz", "qa", {}),
    ],
    "continuation": [
        ("fineweb", "continuation", {"src_tokenizer_name": "meta-llama/Llama-3.2-1B"}),
        ("pile", "continuation", {}),
        ("redpajama", "continuation", {}),
        ("code", "continuation", {}),
    ],
    "mae": [("fineweb", "mae", {"src_tokenizer_name": "meta-llama/Llama-3.2-1B"})],
    "recon": [("bio", "reconstruction", {"world_seed": 0, "n_facts": 3})],
}


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def episode_stats(source, task, spec, tokenizer, pad, n=200):
    """Draw n episodes; return a stats dict (input/answer token lengths, padding, answer-in-context)."""
    ds = TaskDataset(source, task, spec, tokenizer, pad_token_id=pad, seed=1)
    it = iter(ds)
    ctx_valid, ctx_pad_frac, ans_len, scored_len, in_ctx = [], [], [], [], 0
    T = spec.total_len
    for _ in range(n):
        b = next(it)                                         # pre-collate SAMPLE dict
        v = int(b["context_mask"].sum())
        ctx_valid.append(v)
        ctx_pad_frac.append(1.0 - v / T)
        ans_len.append(int(b["answer_ids"].shape[0]))        # all answer_ids valid pre-collate
        scored_len.append(int(sum(b["answer_content_mask_list"])))
        # answer-in-context (the causality/binding invariant for qa/recon)
        ctx = tokenizer.decode(b["context_ids"][b["context_mask"].bool()])
        ans = tokenizer.decode(b["answer_ids"]).strip()
        if ans and ans.lower() in ctx.lower():
            in_ctx += 1
    return dict(
        n=n, total_len=T,
        input_p10=_q(ctx_valid, 10), input_p50=_q(ctx_valid, 50), input_p90=_q(ctx_valid, 90),
        input_max=max(ctx_valid), input_mean=st.mean(ctx_valid),
        pad_frac_mean=st.mean(ctx_pad_frac),
        ans_p50=_q(ans_len, 50), ans_p90=_q(ans_len, 90), ans_max=max(ans_len),
        scored_p50=_q(scored_len, 50),
        in_ctx=in_ctx,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="qa", choices=[*GROUPS, "all"])
    ap.add_argument("--total-len", type=int, default=1024)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-docs", type=int, default=1500, help="bounded per-source load size")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    pad = tok.eos_token_id or 0
    groups = list(GROUPS) if args.group == "all" else [args.group]

    print(f"\n{'source':16} {'qty':>7} {'in_p50':>7} {'in_p90':>7} {'in_max':>7} "
          f"{'pad%':>6} {'ans_p50':>8} {'ans_p90':>8} {'scored':>7} {'ans∈ctx':>8}")
    print("─" * 90)
    for g in groups:
        for sname, tstyle, skw in GROUPS[g]:
            try:
                src = SOURCE_REGISTRY[sname](tok, split="train", seed=1,
                                             **({"n_docs": args.n_docs} if "n_docs" not in skw and
                                                sname not in ("babi", "bio") else {}), **skw)
                qty = len(getattr(src, "docs", getattr(src, "rows",
                          getattr(src, "subs", []) or [])) or [])
                spec = EpisodeSpec(source=sname, task=tstyle, total_len=args.total_len,
                                   predict_len=64, n_inputs=24)
                s = episode_stats(src, get_task(tstyle), spec, tok, pad, n=args.n)
                print(f"{sname:16} {qty:>7} {s['input_p50']:>7} {s['input_p90']:>7} {s['input_max']:>7} "
                      f"{100*s['pad_frac_mean']:>5.0f}% {s['ans_p50']:>8} {s['ans_p90']:>8} "
                      f"{s['scored_p50']:>7} {s['in_ctx']:>6}/{s['n']}")
            except Exception as e:
                print(f"{sname:16} ERROR {type(e).__name__}: {str(e)[:60]}")
    print()


if __name__ == "__main__":
    main()
