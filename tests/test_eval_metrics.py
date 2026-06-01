#!/usr/bin/env python3
"""Layer-1 regression test: the eval METRICS (em / containment / recall / f1).

Pins down correct behavior so the scoring plumbing is locked down and bug-free.
Run:  python tests/test_eval_metrics.py   (exits non-zero on any failure)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.repr_learning.eval_per_family import (
    em_score, f1_score, recall_score, containment_score, max_over_refs, normalize_answer,
)

A = 0.01  # float tolerance

# (name, fn, pred, gold, expected)
CASES = [
    # ── CONTAINMENT (headline) ────────────────────────────────────────────
    ("con_exact",          containment_score, "amateur lithography", "amateur lithography", 1.0),
    ("con_verbose_ok",     containment_score, "Kirsten Oseth's hobby is amateur lithography.", "amateur lithography", 1.0),
    ("con_wrong_number",   containment_score, "1996", "1994", 0.0),
    ("con_legit_number",   containment_score, "established in 1994", "1994", 1.0),
    ("con_substr_num_BUG", containment_score, "34 years", "4", 0.0),     # '4' must NOT match inside '34'
    ("con_substr_num_BUG2",containment_score, "40 years", "4", 0.0),
    ("con_substr_word_BUG",containment_score, "cabinet", "cab", 0.0),    # 'cab' must NOT match inside 'cabinet'
    ("con_reordered",      containment_score, "lithography amateur", "amateur lithography", 0.0),  # order matters
    ("con_partial",        containment_score, "amateur", "amateur lithography", 0.0),
    ("con_articles",       containment_score, "the amateur lithography", "amateur lithography", 1.0),
    ("con_case",           containment_score, "AMATEUR LITHOGRAPHY", "amateur lithography", 1.0),
    ("con_wrong_entity",   containment_score, "John Cabot", "Sebastian Cabot", 0.0),
    ("con_empty_pred",     containment_score, "", "1994", 0.0),
    # ── EM ────────────────────────────────────────────────────────────────
    ("em_exact",           em_score, "amateur lithography", "amateur lithography", 1.0),
    ("em_verbose",         em_score, "his hobby is amateur lithography", "amateur lithography", 0.0),
    ("em_articles",        em_score, "the amateur lithography", "amateur lithography", 1.0),
    ("em_case",            em_score, "1994", "1994", 1.0),
    # ── RECALL (verbosity-robust partial credit) ──────────────────────────
    ("rec_verbose_ok",     recall_score, "Kirsten Oseth's hobby is amateur lithography.", "amateur lithography", 1.0),
    ("rec_half",           recall_score, "John Cabot", "Sebastian Cabot", 0.5),
    ("rec_wrong_number",   recall_score, "34 years", "4", 0.0),
    ("rec_none",           recall_score, "maritime history", "amateur lithography", 0.0),
    # ── F1 ────────────────────────────────────────────────────────────────
    ("f1_exact",           f1_score, "amateur lithography", "amateur lithography", 1.0),
    ("f1_verbose",         f1_score, "Kirsten Oseth's hobby is amateur lithography", "amateur lithography", 0.5),
    ("f1_none",            f1_score, "1996", "1994", 0.0),
]


def main():
    fails = []
    print(f"{'case':24s} {'exp':>5} {'got':>6}  result")
    print("-" * 48)
    for name, fn, pred, gold, exp in CASES:
        got = fn(pred, gold)
        ok = abs(got - exp) < A
        print(f"{name:24s} {exp:>5.2f} {got:>6.2f}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    # multi-ref behavior
    mref = max_over_refs("the famous Ponte di Rialto", ["Rialto Bridge", "Ponte di Rialto"], containment_score)
    ok = abs(mref - 1.0) < A
    print(f"{'maxref_alias':24s} {1.0:>5.2f} {mref:>6.2f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("maxref_alias")

    print("-" * 48)
    if fails:
        print(f"FAILED {len(fails)}/{len(CASES)+1}: {fails}")
        sys.exit(1)
    print(f"ALL {len(CASES)+1} PASS")


if __name__ == "__main__":
    main()
