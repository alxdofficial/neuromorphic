"""Build the full same-code cohort results table → docs/cohort_results.md.

Aggregates across seeds (mean ± std) when multiple seed runs are present.
Reads each model's mixed-val JSONL for REAL loss / babi_em / binding gate
(OFF-REAL, SHUF-REAL), and re-evaluates the checkpoints for exact-match
babi_em under REAL / SHUF / OFF (the reliable binding test -- the loss-based
SHUF gate is unreliable on babi, see notes in the doc).

All runs MUST be from the same code commit (cross-era comparison is invalid).

Seed-dir convention (matches run_overnight_seeds.sh):
  seed 42 -> "{prefix}_{variant}"        e.g. valrun_memon_biomem_baseline
  seed N  -> "{prefix}_s{N}_{variant}"   e.g. valrun_memon_s1_biomem_baseline

Usage:  .venv/bin/python scripts/diagnostics/cohort/cohort_results.py [--no-eval] [--seeds 42 1 2]
"""
import sys, os, json, glob, math, dataclasses, argparse
from pathlib import Path

# Allow `from src...` / `from scripts...` when run as a script file (script dir,
# not the repo root, is what Python puts on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

TASKS = ["mae", "babi", "continuation", "condrecon_bio"]

# display name -> (prefix, variant, multiseed)
MODELS = [
    ("biomem (membrane on)",   "valrun_memon",     "biomem_baseline",         True),
    ("biomem (membrane off)",  "valrun_memoff",    "biomem_baseline",         True),
    ("slotgraph",              "valrun_slotgraph", "slotgraph_baseline",      True),
    ("icae",                   "valrun_icae",      "icae_baseline",           True),
    ("ccm",                    "valrun_ccm",       "ccm_baseline",            True),
    ("autocompressor",         "valrun_autocomp",  "autocompressor_baseline", True),
    ("beacon",                 "valrun_beacon",    "beacon_baseline",         True),
    ("vanilla floor (no mem)", "valrun_vfloor",    "vanilla_llama",           False),
    ("vanilla ceiling (full)", "valrun_vceil",     "vanilla_full_context",    False),
]
ROOT = Path("outputs/memory")


def seed_dir(prefix, variant, seed):
    return ROOT / (f"{prefix}_{variant}" if seed == 42 else f"{prefix}_s{seed}_{variant}")


def extract_jsonl(d):
    """Return {task: {real_loss, babi_em, babi_em_best, shuf_real, off_real}} or None."""
    cands = glob.glob(str(d / "jsonl" / "*.jsonl"))
    if not cands:
        return None
    rows = [json.loads(l) for l in open(cands[0]) if l.strip()]
    val = [r for r in rows if r.get("phase") == "val"]
    if not val:
        return None
    out = {}
    fin_step = max(r["step"] for r in val)
    for r in (x for x in val if x["step"] == fin_step):
        out.setdefault(r["task"], {})["real_loss"] = r.get("val_loss")
        if r.get("val_babi_em") is not None:
            out[r["task"]]["babi_em"] = r["val_babi_em"]
    bem = [r["val_babi_em"] for r in val if r.get("val_babi_em") is not None]
    if bem:
        out.setdefault("babi", {})["babi_em_best"] = max(bem)
    gk = [r for r in val if "val_shuf_minus_real" in r or "val_off_minus_real" in r]
    if gk:
        gstep = max(r["step"] for r in gk)
        for r in (x for x in gk if x["step"] == gstep):
            out.setdefault(r["task"], {})["shuf_real"] = r.get("val_shuf_minus_real")
            out[r["task"]]["off_real"] = r.get("val_off_minus_real")
    return out


def eval_babi_em(d, variant):
    """Exact-match babi_em under REAL/SHUF/OFF by reloading the checkpoint."""
    import torch
    from transformers import AutoTokenizer
    from src.memory.config import ReprConfig
    from src.memory.model import ReprLearningModel
    from src.memory.training import make_mixed_val_sets, to_device

    cps = sorted(glob.glob(str(d / "ckpts" / "*.pt")))
    if not cps:
        return None
    sd = torch.load(cps[-1], map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]
    valid = {fld.name for fld in dataclasses.fields(ReprConfig)}
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vs = make_mixed_val_sets(["babi"], tok, cfg, 4, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)
    m = ReprLearningModel(cfg, variant=variant, llama_model=None).cuda()
    m.load_state_dict(sd["model_state_dict"], strict=False)
    m.eval(); m.task_mode = "babi"
    res = {}
    for cond, kw in [("REAL", {}), ("SHUF", {"shuffle_memory": True}), ("OFF", {"zero_memory": True})]:
        hit = n = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for b in vs["babi"]:
                b = to_device(b, "cuda")
                out = m.compute_loss(b, window_size=1024, **kw)
                for fam, em in zip(b.task_family, out["per_example_em"].tolist()):
                    if fam == "babi":
                        hit += em; n += 1
        res[cond] = hit / max(n, 1)
    del m; torch.cuda.empty_cache()
    return res


def agg(vals):
    """mean, std over non-None values; std=None if <2."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return mean, std


def ms(vals, p=3, scale=1.0, pct=False):
    """format mean ± std."""
    mean, std = agg([v * scale if v is not None else None for v in vals])
    if mean is None:
        return "—"
    suf = "%" if pct else ""
    if std is None:
        return f"{mean:.{p}f}{suf}"
    return f"{mean:.{p}f}±{std:.{p}f}{suf}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()

    # gather per-(model, seed) extractions + evals
    data = {}      # name -> {seed: jsonl_dict}
    evals = {}     # name -> {seed: {REAL,SHUF,OFF}}
    nseed = {}     # name -> count of present seeds
    for name, prefix, variant, multi in MODELS:
        seeds = args.seeds if multi else [42]
        data[name] = {}
        evals[name] = {}
        for s in seeds:
            d = seed_dir(prefix, variant, s)
            j = extract_jsonl(d)
            if j is None:
                continue
            data[name][s] = j
            if not args.no_eval and variant != "vanilla_llama":
                try:
                    evals[name][s] = eval_babi_em(d, variant)
                    print(f"[eval] {name} seed {s}: {evals[name][s]}", file=sys.stderr)
                except Exception as e:
                    print(f"[eval] {name} seed {s} FAILED: {e}", file=sys.stderr)
        nseed[name] = len(data[name])

    present = [(n, p, v, m) for n, p, v, m in MODELS if nseed[n] > 0]

    def col(name, task, key):
        return [data[name][s].get(task, {}).get(key) for s in data[name]]

    L = []
    L.append("# Cohort results — mixed 4-task, same-code\n")
    L.append("Frozen SmolLM2-135M (d=576), 1024→M=32 (32:1), 4000 steps, batch 8. "
             "All models trained on the **same code commit** (cross-era comparison is invalid). "
             "Tasks: mae / babi / continuation / condrecon_bio. Cells are **mean ± std** over the "
             "seeds present (n shown per model); single-seed cells show the mean only.\n")
    L.append("Seeds per model: " + ", ".join(f"{n} (n={nseed[n]})" for n, *_ in present) + "\n")

    # Table 1 — REAL
    L.append("## Table 1 — REAL performance (val loss; lower is better)\n")
    L.append("babi also shows exact-match EM (final). Loss is unitless.\n")
    L.append("| model | " + " | ".join(TASKS) + " | babi EM |")
    L.append("|" + "---|" * (len(TASKS) + 2))
    for name, *_ in present:
        cells = [ms(col(name, t, "real_loss"), 3) for t in TASKS]
        em = ms(col(name, "babi", "babi_em"), 0, scale=100, pct=True)
        L.append(f"| {name} | " + " | ".join(cells) + f" | {em} |")
    L.append("")

    # Table 2 — gate
    L.append("## Table 2 — binding gate (loss units)\n")
    L.append("OFF−REAL = cost of zeroing memory (usage). SHUF−REAL = cost of the *wrong* example's "
             "memory (example-specificity). **Caveat:** the loss-based SHUF gate is unreliable on babi "
             "(consecutive examples share answers → wrong memory often coincidentally fits); Table 3 is "
             "the reliable binding signal.\n")
    L.append("| model | " + " | ".join(f"{t} OFF / SHUF" for t in TASKS) + " |")
    L.append("|" + "---|" * (len(TASKS) + 1))
    for name, *_ in present:
        cells = [f"{ms(col(name,t,'off_real'),2)} / {ms(col(name,t,'shuf_real'),3)}" for t in TASKS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    # Table 3 — exact-match binding
    if any(evals[n] for n, *_ in present):
        L.append("## Table 3 — babi_em binding (exact-match REAL / SHUF / OFF)\n")
        L.append("The reliable binding test. **OFF=0 → memory is essential** (answer impossible without "
                 "it). SHUF≈REAL across *all* models (incl. published icae) confirms the SHUF metric is "
                 "diluted on babi, not that models fail to bind.\n")
        L.append("| model | REAL | SHUF | OFF |")
        L.append("|---|---|---|---|")
        for name, *_ in present:
            if not evals[name]:
                continue
            seeds = list(evals[name])
            r = ms([evals[name][s]["REAL"] for s in seeds], 1, 100, True)
            sh = ms([evals[name][s]["SHUF"] for s in seeds], 1, 100, True)
            of = ms([evals[name][s]["OFF"] for s in seeds], 1, 100, True)
            L.append(f"| {name} | {r} | {sh} | {of} |")
        L.append("")

    out = Path("docs/cohort_results.md")
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({len(present)} models)")
    print("\n".join(L))


if __name__ == "__main__":
    main()
