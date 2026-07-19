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
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "baselines"


def collect(pattern: str):
    """-> (rows_sorted, cols_sorted, cell{(row,col): (acc,n)}, overall{col: (acc,n)})."""
    cells, cols = {}, set()
    for fp in sorted(glob.glob(str(OUT / pattern))):
        if fp.endswith(("_api_summary.json", "report.json")):
            continue
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
        # COVERAGE QC row: fraction of selected items that produced a scorable answer (<1.0 ⇒ errors/cutoffs
        # were EXCLUDED from accuracy, not counted wrong — the accuracy above is over fewer items).
        cov = meta.get("coverage")
        if cov is not None:
            cells[(f"{dataset}-COVERAGE", col)] = (cov, meta.get("n"))
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
        w = csv.writer(f)
        w.writerow(["dataset-subtask", *cols])
        for r in rows:
            w.writerow([r, *[(_fmt(cells.get((r, c))) or "") for c in cols]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.json", help="filename glob under outputs/baselines/")
    args = ap.parse_args()
    rows, cols, cells = collect(args.glob)
    md = to_markdown(rows, cols, cells)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(md + "\n")
    to_csv(rows, cols, cells, OUT / "report.csv")
    print(md)
    print(f"\n[report] wrote {OUT}/report.md and report.csv")


if __name__ == "__main__":
    main()
