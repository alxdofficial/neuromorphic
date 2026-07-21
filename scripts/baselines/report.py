#!/usr/bin/env python3
"""Aggregate Phase-2 baseline run JSONs into a comparison table keyed by `dataset-subtask`.

Reads every `outputs/baselines/<dataset>__<model>__<mode>.json` (written by run_api_eval.py / the Tier-2
runners), and emits a table whose ROWS are `<dataset>-<subtask>` (e.g. `longmemeval-temporal`,
`memoryagentbench-factconsolidation_mh_6k`) and COLUMNS are `<model> · <mode>`, cells = accuracy (n).
Writes Markdown + CSV to outputs/baselines/report.{md,csv} and prints the Markdown.

Usage:
  python scripts/baselines/report.py                     # all runs under outputs/baselines
  python scripts/baselines/report.py --glob 'longmemeval__*'
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "outputs" / "baselines"

# Below this many scored questions an artifact is a smoke test, not a result, and must never reach the
# published table. LongMemEval-S is 500 Q and MAB 3071, so any real run clears this by two orders of
# magnitude; the smokes we actually produce are n=1/n=2. Deliberately low: a genuinely partial run should
# still show up (with its n visible) rather than vanish silently.
_MIN_REPORTABLE_N = 10


def collect(pattern: str):
    """-> (rows_sorted, cols_sorted, cell{(row,col): (acc,n)}, overall{col: (acc,n)})."""
    cells, cols = {}, set()
    # A (dataset, model, mode) can have SEVERAL artifacts on disk — an n=1/n=2 smoke next to the real run.
    # Without this, the smoke's cells overwrite the real ones per-row (a 1.000 (1) OVERALL sitting above a
    # 0.364 (121) subtask row, from two different files). Keep only the highest-n artifact per column key,
    # and drop smokes outright, so a stray smoke JSON can never be published as a result.
    best: dict[tuple, tuple] = {}
    for fp in sorted(glob.glob(str(OUT / pattern))):
        if fp.endswith(("_api_summary.json", "report.json")):
            continue
        try:
            d0 = json.loads(Path(fp).read_text())
        except Exception:  # noqa: BLE001
            continue
        m0 = d0.get("meta") or {}
        n0 = m0.get("n_scored") if m0.get("n_scored") is not None else m0.get("n") or 0
        key = (d0.get("dataset"), d0.get("model"), d0.get("mode") or d0.get("method"))
        if n0 < _MIN_REPORTABLE_N:
            print(f"[report] skip {Path(fp).name}: n={n0} < {_MIN_REPORTABLE_N} (smoke, not a result)")
            continue
        if key not in best or n0 > best[key][0]:
            best[key] = (n0, fp)
    for _n, fp in sorted(best.values(), key=lambda t: t[1]):
        try:
            d = json.loads(Path(fp).read_text())
        except Exception:  # noqa: BLE001
            continue
        agg = d.get("aggregate") or {}
        meta = d.get("meta") or {}
        dataset = d.get("dataset", "?")
        # Tier-1 API runs carry `mode` (floor/full_context/rag_*); Tier-2 + agent-memory runs carry `method`
        # (lclm/kvzip/memoryllm/a-mem/…). Fall back so BOTH show up as columns in the comparison table.
        mode_or_method = d.get("mode") or d.get("method") or "?"
        col = f"{d.get('model', '?').split('/')[-1]} · {mode_or_method}"
        if "per_subtask" not in agg:
            print(f"[report] WARN: {Path(fp).name} has no per_subtask — skipped (rerun to regenerate)")
            continue
        cols.add(col)
        # OVERALL folded in as a per-DATASET row so runs of the same model+mode on different datasets
        # don't collide on a single shared OVERALL row. n uses an explicit-None check (0 is a valid count).
        n_over = agg.get("n_scored")
        if n_over is None:
            n_over = agg.get("n_nonabstention")
        cells[(f"{dataset}-OVERALL", col)] = (agg.get("overall_accuracy"), n_over)
        # COVERAGE QC row: fraction that produced a scoreable model output (provider/execution failures are
        # excluded). Token-cap outputs are scored; EOS_COMPLETION reports their separate completion signal.
        cov = meta.get("coverage")
        if cov is not None:
            cells[(f"{dataset}-COVERAGE", col)] = (cov, meta.get("n"))
        eos_rate = meta.get("eos_completion_rate")
        if eos_rate is not None:
            cells[(f"{dataset}-EOS_COMPLETION", col)] = (eos_rate, meta.get("n"))
        # audit #15: overall_accuracy is a per-source MICRO-average (MAB is dominated by Accurate_Retrieval's
        # 1,700 Q). Surface the competency MACRO-average + per-competency + the intent-parsed lenient overall
        # so architecture comparisons aren't read off the AR-skewed micro number.
        cma = agg.get("competency_averaged_accuracy")
        if cma is not None:
            cells[(f"{dataset}-COMPETENCY_MACRO", col)] = (cma, n_over)
        lenient = agg.get("overall_accuracy_lenient")
        if lenient is not None:
            cells[(f"{dataset}-OVERALL_lenient", col)] = (lenient, n_over)
        for comp, v in (agg.get("per_competency") or {}).items():
            cells[(f"{dataset}-comp:{comp}", col)] = (v.get("accuracy"), v.get("n"))
        for sub, v in agg["per_subtask"].items():
            cells[(f"{dataset}-{sub}", col)] = (v.get("accuracy"), v.get("n"))
    return sorted({r for r, _ in cells}, key=_row_sort_key), sorted(cols), cells


def _row_sort_key(r: str):
    # group by dataset, OVERALL first within each dataset, then subtasks alphabetically
    ds, _, sub = r.partition("-")
    return (ds, sub != "OVERALL", sub)


def _esc(s) -> str:
    return str(s).replace("|", "\\|")   # keep a literal pipe from breaking the markdown table


def _fmt(cell):
    if cell is None:
        return ""
    acc, n = cell
    return f"{acc:.3f} ({n})" if acc is not None else ""


def to_markdown(rows, cols, cells) -> str:
    if not cols:
        return "_(no runs found — run run_api_eval.py first)_"
    lines = ["| dataset-subtask | " + " | ".join(_esc(c) for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        bold = r.endswith("-OVERALL")
        cellstrs = [_fmt(cells.get((r, c))) for c in cols]
        if bold:
            cellstrs = [f"**{s}**" if s else s for s in cellstrs]
        label = f"**{_esc(r)}**" if bold else _esc(r)
        lines.append(f"| {label} | " + " | ".join(cellstrs) + " |")
    return "\n".join(lines)


def to_csv(rows, cols, cells, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset-subtask", *cols])
        for r in rows:
            w.writerow([r, *[(_fmt(cells.get((r, c))) or "") for c in cols]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.json", help="filename glob under outputs/baselines/")
    ap.add_argument("--no-published", action="store_true",
                    help="omit the CITED published-reference block (judge-scored, not comparable)")
    args = ap.parse_args()
    rows, cols, cells = collect(args.glob)
    md = to_markdown(rows, cols, cells)
    if not args.no_published:
        # append the CITED published numbers as a SEPARATE labeled block (LLM-judge, NOT our deterministic
        # scale) — never merged into the table above.
        from src.memory.eval.published_baselines import render_published_markdown
        md = md + "\n" + render_published_markdown()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(md + "\n")
    to_csv(rows, cols, cells, OUT / "report.csv")   # CSV stays deterministic-only (our runs)
    print(md)
    print(f"\n[report] wrote {OUT}/report.md and report.csv")


if __name__ == "__main__":
    main()
