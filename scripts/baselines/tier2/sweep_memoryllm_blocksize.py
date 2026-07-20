#!/usr/bin/env python3
"""M+ block-size sweep — pick the injection block size (the cost lever) BEFORE the full M+ pod run.

WHY (memory `project_mplus_batching_verdict`): M+ can't be batched within one model (single shared
`[32, 10240, 4096]` pool, batch=1 baked in) and is **overhead-bound, not compute-bound** — each
`inject_memory()` call costs ~0.4 s wall for ~15-25 ms of actual compute, GPU idle ~20×. So the ONLY cheap
speed lever is the injection block size: bigger `--inject-block-tokens` → FEWER sequential inject calls over
the same ~115k history → proportionally less overhead. Each block still compresses to a fixed 256 mem-tokens
regardless of size, so a bigger block is coarser compression — this sweep measures whether accuracy holds
while wall time drops, so we buy the ~2× for free only if it does.

WHAT: runs `run_memoryllm.py` on a FIXED LongMemEval subsample at each `--inject-block-tokens`, then tabulates
overall accuracy vs. the GPU-synced inject time recorded in each artifact's `meta.timing`. The subsample is
deterministic (tier2_common.load_items stratified `--max-examples`) so every block sees the SAME items.

COST NOTE: this drives one subprocess per block → the 8B model reloads each time (~1-2 min/load). That's a
one-time calibration tax (3 loads ≈ 3-6 min); the reported inject/answer seconds are measured INTERNALLY and
exclude load, so the comparison is clean. Run it on a small subsample (default 50) on the M+ pod, read the
table, set `--inject-block-tokens` for the full run, terminate.

Example (on the M+ pod):
  python scripts/baselines/tier2/sweep_memoryllm_blocksize.py --max-examples 50 --blocks 512,1024,2048
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts" / "baselines" / "tier2" / "run_memoryllm.py"


def _run_one(block: int, args) -> Path | None:
    """Invoke run_memoryllm.py for one block size, streaming its output; return the artifact path it wrote."""
    cmd = [sys.executable, str(RUNNER),
           "--dataset", args.dataset, "--variant", args.variant,
           "--inject-block-tokens", str(block), "--seed", str(args.seed),
           "--attn-impl", args.attn_impl, "--repo-dir", args.repo_dir, "--out-dir", args.out_dir]
    if args.max_examples is not None:
        cmd += ["--max-examples", str(args.max_examples)]
    if args.model:
        cmd += ["--model", args.model]
    if args.no_bem:
        cmd += ["--no-bem"]
    print(f"\n{'=' * 90}\n[sweep] block={block} → {' '.join(cmd)}\n{'=' * 90}", flush=True)
    # stream live (inject/answer progress + the runner's own timing line) while capturing the artifact path.
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wrote = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if "] wrote " in line:
            wrote = line.split("] wrote ", 1)[1].strip()
    proc.wait()
    if proc.returncode != 0:
        print(f"[sweep] ✗ block={block} exited {proc.returncode}")
        return None
    return Path(wrote) if wrote else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", default="512,1024,2048", help="comma-separated inject-block-tokens to sweep")
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--variant", default="s")
    ap.add_argument("--max-examples", type=int, default=50, help="fixed subsample (deterministic/stratified)")
    ap.add_argument("--model", default=None, help="override HF repo id (default = runner's mplus-8b)")
    ap.add_argument("--repo-dir", default=str(REPO.parent / "baselines" / "MemoryLLM"))
    ap.add_argument("--attn-impl", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bem", action="store_true", help="skip BEM (faster scoring; accuracy read is coarser)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max accuracy drop vs the smallest block still counted as 'quality holds' (default 2pt)")
    args = ap.parse_args()

    blocks = [int(b) for b in args.blocks.split(",") if b.strip()]
    rows = []
    for b in blocks:
        art = _run_one(b, args)
        if art is None or not art.exists():
            print(f"[sweep] block={b}: no artifact — skipping")
            continue
        d = json.loads(art.read_text())
        t = d.get("meta", {}).get("timing", {})
        rows.append({
            "block": b,
            "acc": d.get("aggregate", {}).get("overall_accuracy"),
            "cov": d.get("meta", {}).get("coverage"),
            "n_injects": t.get("n_injects"),
            "encode_s": t.get("encode_s"),
            "s_per_inject": t.get("s_per_inject"),
            "answer_s": t.get("answer_s"),
            "art": art.name,
        })

    if not rows:
        print("[sweep] no results — nothing to report")
        return

    def _f(x, p="6.3f"):
        return format(x, p) if isinstance(x, (int, float)) else str(x)

    print(f"\n{'=' * 90}\n[sweep] M+ block-size sweep — {args.dataset} n≈{args.max_examples}\n{'=' * 90}")
    print(f"{'block':>6} {'acc':>7} {'cov':>6} {'injects':>8} {'encode_s':>9} {'s/inject':>9} {'answer_s':>9}")
    for r in rows:
        print(f"{r['block']:>6} {_f(r['acc'],'7.3f')} {_f(r['cov'],'6.3f')} {_f(r['n_injects'],'8d')} "
              f"{_f(r['encode_s'],'9.1f')} {_f(r['s_per_inject'],'9.3f')} {_f(r['answer_s'],'9.1f')}")

    # recommendation: the LARGEST block whose accuracy is within --tol of the smallest block's (the finest,
    # highest-fidelity compression) — i.e. the cheapest injection cost that doesn't cost us quality.
    base = rows[0]
    baseline_acc = base["acc"] if isinstance(base["acc"], (int, float)) else None
    pick = base
    for r in rows[1:]:
        if isinstance(r["acc"], (int, float)) and baseline_acc is not None and r["acc"] >= baseline_acc - args.tol:
            pick = r
    speedup = (base["encode_s"] / pick["encode_s"]) if (base.get("encode_s") and pick.get("encode_s")) else None
    print(f"\n[sweep] baseline block={base['block']} acc={_f(base['acc'],'.3f')} encode={_f(base['encode_s'],'.1f')}s")
    msg = f"[sweep] → RECOMMEND --inject-block-tokens {pick['block']} (acc {_f(pick['acc'],'.3f')}"
    if speedup:
        msg += f", {speedup:.2f}× faster injection"
    print(msg + f", within {args.tol:.02f} of baseline)")


if __name__ == "__main__":
    main()
