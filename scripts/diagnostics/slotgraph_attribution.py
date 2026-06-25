"""slotgraph clean 2x2 attribution → docs/slotgraph_attribution.md.

The eval-time ablation was confounded (weights trained WITH struct+id). This
reads the FROM-SCRATCH retrained controls (all at slotgraph LoRA rank 82, same
config, 3 seeds) so each innovation's contribution is unconfounded:

  full (struct+id)  vs  struct-only (id OFF)  vs  id-only (struct OFF)  vs
  neither (same-code icae)  vs  icae (native).

Reuses the cohort generator's extractors. Run after run_slotgraph_attribution.sh.
"""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
from scripts.diagnostics.cohort_results import seed_dir, extract_jsonl, eval_babi_em, ms

TASKS = ["mae", "babi", "continuation", "condrecon_bio"]
SEEDS = [42, 1, 2]
# name -> (prefix, variant)
MODELS = [
    ("full (struct+id)",        "valrun_slotgraph",     "slotgraph_baseline"),
    ("struct-only (id OFF)",    "valrun_sg_structonly", "slotgraph_baseline"),
    ("id-only (struct OFF)",    "valrun_sg_idonly",     "slotgraph_baseline"),
    ("neither (same-code icae)","valrun_sg_neither",    "slotgraph_baseline"),
    ("icae (native)",           "valrun_icae",          "icae_baseline"),
]


def main():
    data, evals, nseed = {}, {}, {}
    for name, prefix, variant in MODELS:
        data[name], evals[name] = {}, {}
        for s in SEEDS:
            d = seed_dir(prefix, variant, s)
            j = extract_jsonl(d)
            if j is None:
                continue
            data[name][s] = j
            try:
                evals[name][s] = eval_babi_em(d, variant)
                print(f"[eval] {name} seed {s}: {evals[name][s]}", file=sys.stderr)
            except Exception as e:
                print(f"[eval] {name} seed {s} FAILED: {e}", file=sys.stderr)
        nseed[name] = len(data[name])

    present = [(n, p, v) for n, p, v in MODELS if nseed[n] > 0]

    def col(name, task, key):
        return [data[name][s].get(task, {}).get(key) for s in data[name]]

    L = ["# slotgraph attribution — clean 2×2 (from-scratch controls)\n",
         "Each cell trained from scratch, same config, LoRA rank 82, 3 seeds {42,1,2}. "
         "Isolates the contribution of the **id-tags** and the **graph structure (MP read)** without "
         "the eval-time distribution-shift confound. `neither` = slotgraph code with both off "
         "(same-code icae anchor at rank 82); `icae (native)` = the real icae baseline.\n",
         "Seeds present: " + ", ".join(f"{n} (n={nseed[n]})" for n, *_ in present) + "\n"]

    L.append("## Table 1 — REAL loss (↓) + babi EM (↑)\n")
    L.append("| model | mae | babi | continuation | condrecon_bio | babi EM |")
    L.append("|" + "---|" * 6)
    for name, *_ in present:
        cells = [ms(col(name, t, "real_loss"), 3) for t in TASKS]
        em = ms(col(name, "babi", "babi_em"), 0, 100, True)
        L.append(f"| {name} | " + " | ".join(cells) + f" | {em} |")
    L.append("")

    L.append("## Table 2 — example-specificity (mae / continuation SHUF−REAL, ↑=binds)\n")
    L.append("Reliable on mae/continuation (no babi correlated-batch dilution).\n")
    L.append("| model | mae SHUF−REAL | continuation SHUF−REAL |")
    L.append("|---|---|---|")
    for name, *_ in present:
        L.append(f"| {name} | {ms(col(name,'mae','shuf_real'),3)} | {ms(col(name,'continuation','shuf_real'),3)} |")
    L.append("")

    if any(evals[n] for n, *_ in present):
        L.append("## Table 3 — babi_em binding (exact-match REAL / SHUF / OFF)\n")
        L.append("| model | REAL | SHUF | OFF |")
        L.append("|---|---|---|---|")
        for name, *_ in present:
            if not evals[name]:
                continue
            ss = list(evals[name])
            r = ms([evals[name][s]["REAL"] for s in ss], 1, 100, True)
            sh = ms([evals[name][s]["SHUF"] for s in ss], 1, 100, True)
            of = ms([evals[name][s]["OFF"] for s in ss], 1, 100, True)
            L.append(f"| {name} | {r} | {sh} | {of} |")
        L.append("")

    out = Path("docs/slotgraph_attribution.md")
    out.write_text("\n".join(L))
    print(f"wrote {out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
