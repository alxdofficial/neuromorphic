#!/usr/bin/env python3
"""Live per-task dashboard for a MIXED multi-task run.

Tails a run's jsonl and prints a per-task table line each time a NEW eval step
lands (i.e. once all per-task val rows for a step are in). Designed for the mixed
harness (scripts/train/train.py --task mixed), which writes one PER-TASK val row
per eval, tagged with `task` + `step`:

  step  1500 | MAE  loss=4.80 top1=.28 | bAbI EM=41% loss=2.10 | cont early=0.62

Line-buffered stdout so it works under the Monitor tool / `tail -f`. Pass the
jsonl path directly, or a run directory (we resolve <dir>/jsonl/*.jsonl, or a
bare *.jsonl inside it).

  python scripts/diagnostics/mixed_dashboard.py outputs/memory/<tag>_<variant>
  python scripts/diagnostics/mixed_dashboard.py path/to/<variant>.jsonl
  python scripts/diagnostics/mixed_dashboard.py <jsonl> --once   # render existing rows then exit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _resolve_jsonl(arg: str) -> Path:
    """Accept a jsonl file, or a run dir containing jsonl/<variant>.jsonl."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("jsonl/*.jsonl")) or sorted(p.glob("*.jsonl"))
        if not cands:
            raise SystemExit(f"no *.jsonl under {p} (looked in {p}/jsonl/ and {p}/)")
        if len(cands) > 1:
            print(f"[dashboard] multiple jsonl found, using {cands[0].name}: "
                  f"{[c.name for c in cands]}", file=sys.stderr)
        return cands[0]
    raise SystemExit(f"not a file or directory: {arg}")


def _fmt_task(task: str, row: dict) -> str:
    """One task's cell. val_loss is the primary metric; bAbI adds EM, continuation
    adds the early-token loss."""
    loss = row.get("val_loss", row.get("val_loss_recon"))
    top1 = row.get("top1", row.get("val_top1_acc"))
    if task in ("mae", "masked_reconstruction"):
        return f"MAE  loss={loss:.3f} top1={top1:.2f}"
    if task == "babi":
        em = row.get("val_babi_em")
        em_s = f"EM={em*100:.0f}% " if em is not None else ""
        return f"bAbI {em_s}loss={loss:.3f}"
    if task == "continuation":
        early = row.get("val_cont_early_loss")
        early_s = f" early={early:.3f}" if early is not None else ""
        return f"cont loss={loss:.3f}{early_s}"
    return f"{task} loss={loss:.3f}"


# canonical column order; any other task is appended in first-seen order.
_ORDER = ["mae", "masked_reconstruction", "babi", "continuation"]


def _render(step: int, rows_by_task: dict, final: bool) -> str:
    tasks = sorted(rows_by_task, key=lambda t: (_ORDER.index(t) if t in _ORDER else 99, t))
    cells = " | ".join(_fmt_task(t, rows_by_task[t]) for t in tasks)
    tag = " [final]" if final else ""
    return f"step {step:6d}{tag} | {cells}"


def _emit_complete(pending: dict, emitted: set, final_steps: set):
    """Print each eval step's line ONCE its per-task rows are all in, never a
    partial line. The trainer writes one row per task per eval step; we consider a
    step complete when (a) it has as many task rows as the widest step seen so far,
    (b) it is the final eval, or (c) a strictly-later step has appeared (which means
    the earlier step's rows are fully flushed). `final` rows are re-emitted as an
    upgrade (keyed on (step, is_final))."""
    if not pending:
        return
    n_expected = max(len(v) for v in pending.values())   # widest step = full task set
    max_step = max(pending)
    for step in sorted(pending):
        is_final = step in final_steps
        # `is_final` is a cosmetic TAG only — it must NOT bypass the all-tasks
        # completeness gate, or the first-arriving task's final row prints a partial line.
        complete = (len(pending[step]) >= n_expected) or (step < max_step)
        if not complete:
            continue
        key = (step, is_final)
        if key in emitted:
            continue
        if not is_final and (step, False) in emitted:   # non-final already printed
            continue
        print(_render(step, pending[step], is_final), flush=True)
        emitted.add(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="jsonl file OR run dir (resolves jsonl/<variant>.jsonl)")
    ap.add_argument("--once", action="store_true",
                    help="render all eval rows currently in the file, then exit "
                         "(no tailing). Default: tail and follow.")
    ap.add_argument("--poll", type=float, default=1.0, help="tail poll interval (s)")
    args = ap.parse_args()

    # The run may not have created its jsonl yet (dashboard started first). For the
    # tailing path, wait for it to appear; --once renders what exists and exits.
    if args.once:
        jsonl = _resolve_jsonl(args.path)
    else:
        jsonl = None
        while jsonl is None:
            try:
                jsonl = _resolve_jsonl(args.path)
            except SystemExit:                      # not created yet — keep waiting
                time.sleep(args.poll)
    print(f"[dashboard] tailing {jsonl}", file=sys.stderr, flush=True)

    pending: dict[int, dict] = {}   # step -> {task: val_row}
    final_steps: set[int] = set()
    emitted: set = set()

    def ingest(line: str):
        line = line.strip()
        if not line:
            return
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            return
        if r.get("phase") != "val" or "task" not in r:
            return
        step = int(r.get("step", -1))
        pending.setdefault(step, {})[r["task"]] = r
        if r.get("final"):
            final_steps.add(step)

    if args.once:
        with open(jsonl) as fp:
            for line in fp:
                ingest(line)
        _emit_complete(pending, emitted, final_steps)
        return

    # tail -f loop: re-open-safe, handles truncation (run restart).
    with open(jsonl) as fp:
        # drain existing content first
        for line in fp:
            ingest(line)
        _emit_complete(pending, emitted, final_steps)
        while True:
            where = fp.tell()
            line = fp.readline()
            if not line:
                time.sleep(args.poll)
                fp.seek(where)
                continue
            ingest(line)
            _emit_complete(pending, emitted, final_steps)


if __name__ == "__main__":
    main()
