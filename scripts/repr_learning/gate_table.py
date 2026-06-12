#!/usr/bin/env python3
"""Cross-model EMAT-bio binding-gate table from the trainer's per-variant JSONL.

The gate is computed in-training at every validation (REAL/SHUF/OFF over the
held-out val set), so this just reads the LAST val row per variant — no re-run.

Columns:
  REAL       = val_loss_recon                         (lower better; the answer CE)
  top1       = val_top1_acc                            (answer-token accuracy)
  SHUF-REAL  = val_loss_recon_shuf - val_loss_recon    (>0 ⇒ ADDRESSABLE binding:
                                                         shuffling the key→value map
                                                         hurts, so reads are key-specific)
  OFF-REAL   = val_loss_recon_off  - val_loss_recon    (>0 ⇒ memory is USED at all,
                                                         key-specific or not)

For graph_v8 only, also surfaces the recently-added health telemetry from the
final val row (key-collapse cosine, state-effect, reader-gate) so a flat gate can
be read against the machinery: collapse-cos↑ = write-router hub collapse;
state-effect≈0 = nothing written; reader-gate≈0 = read inert.

Usage:
  .venv/bin/python scripts/repr_learning/gate_table.py \
      --jsonl-dir outputs/repr_learning/emat_bio_v8_gate/jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFERRED_ORDER = [
    "graph_v8_baseline", "beacon_baseline", "icae_baseline",
    "ccm_baseline", "autocompressor_baseline",
]


def last_val_row(path: Path) -> dict | None:
    last = None
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("phase") == "val" and r.get("val_loss_recon") is not None:
                last = r
    return last


def fmt(x, nd=4, w=9):
    return (f"{x:+.{nd}f}" if x is not None else "  n/a  ").rjust(w)


def resolve_paths(out_tag: str, jsonl_dir: Path | None) -> dict[str, Path]:
    """Per-variant outputs live at outputs/repr_learning/<out_tag>_<variant>/jsonl/
    <variant>.jsonl (one dir per arm). A flat --jsonl-dir overrides that."""
    if jsonl_dir is not None:
        return {p.stem: p for p in sorted(jsonl_dir.glob("*.jsonl"))}
    base = Path("outputs/repr_learning")
    found = {}
    for d in sorted(base.glob(f"{out_tag}_*")):
        for p in d.glob("jsonl/*.jsonl"):
            found[p.stem] = p
    return found


def collect(found: dict[str, Path], variants: list[str] | None):
    names = variants or [v for v in PREFERRED_ORDER if v in found] + \
            [v for v in found if v not in PREFERRED_ORDER]
    rows = []
    for v in names:
        p = found.get(v)
        row = last_val_row(p) if p else None
        if row is None:
            rows.append({"variant": v, "status": "no val yet"})
            continue
        real = row.get("val_loss_recon")
        shuf = row.get("val_loss_recon_shuf")
        off = row.get("val_loss_recon_off")
        rows.append({
            "variant": v,
            "step": row.get("step"),
            "REAL": real,
            "top1": row.get("val_top1_acc"),
            "SHUF_REAL": (shuf - real) if (shuf is not None and real is not None) else None,
            "OFF_REAL": (off - real) if (off is not None and real is not None) else None,
            # graph_v8-only health telemetry (deepest written layer L3)
            "key_collapse_L3": row.get("val_graph_v8_key_collapse_cos_L3"),
            "state_effect_L3": row.get("val_graph_v8_state_effect_L3"),
            "reader_gate": row.get("val_graph_v8_reader_gate_mean"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="emat_bio_v8_gate",
                    help="Finds outputs/repr_learning/<out_tag>_<variant>/jsonl/*.jsonl")
    ap.add_argument("--jsonl-dir", type=Path, default=None,
                    help="Override: a single flat dir of <variant>.jsonl files.")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--bind-thresh", type=float, default=0.05,
                    help="SHUF-REAL above this is called BINDS.")
    args = ap.parse_args()

    found = resolve_paths(args.out_tag, args.jsonl_dir)
    rows = collect(found, args.variants)
    done = [r for r in rows if "REAL" in r]
    done.sort(key=lambda r: (r["SHUF_REAL"] is not None, r.get("SHUF_REAL") or -1e9),
              reverse=True)

    print(f"\nEMAT-bio binding gate  —  {args.jsonl_dir or args.out_tag}")
    print("=" * 92)
    print(f"{'variant':<26}{'step':>5}{'REAL':>9}{'top1':>8}"
          f"{'SHUF-REAL':>11}{'OFF-REAL':>10}   verdict")
    print("-" * 92)
    for r in rows:
        if "REAL" not in r:
            print(f"{r['variant']:<26}{'':>5}{'  (' + r['status'] + ')':>9}")
            continue
        sr = r["SHUF_REAL"]
        verdict = "BINDS ✓" if (sr is not None and sr > args.bind_thresh) else "flat"
        uses = "" if (r["OFF_REAL"] is None or r["OFF_REAL"] <= 0) else " · uses-mem"
        t1 = f"{r['top1']*100:5.1f}%" if r["top1"] is not None else "  n/a"
        print(f"{r['variant']:<26}{r['step']:>5}{r['REAL']:>9.4f}{t1:>8}"
              f"{fmt(r['SHUF_REAL']):>11}{fmt(r['OFF_REAL']):>10}   {verdict}{uses}")
    print("=" * 92)
    print("SHUF-REAL>0 ⇒ addressable key→value binding (the headline).  "
          "OFF-REAL>0 ⇒ memory used at all.")

    g = next((r for r in done if r["variant"] == "graph_v8_baseline"), None)
    if g and any(g.get(k) is not None for k in
                 ("key_collapse_L3", "state_effect_L3", "reader_gate")):
        print("\ngraph_v8 health telemetry (final val, L3):")
        print(f"  key_collapse_cos = {g['key_collapse_L3']!s:<8} "
              "(↑→ write-router hub collapse; nodes share one direction)")
        print(f"  state_effect     = {g['state_effect_L3']!s:<8} "
              "(≈0 → nothing written into the bank)")
        print(f"  reader_gate_mean = {g['reader_gate']!s:<8} "
              "(≈0.1 init; growth → read injection strengthening)")


if __name__ == "__main__":
    main()
