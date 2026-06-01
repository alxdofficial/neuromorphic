#!/usr/bin/env python3
"""Re-score ALL trained variants under the new headline metric (containment) +
EM + recall + F1. Thin driver over eval_per_family's building blocks, with an
explicit per-variant checkpoint map (the ckpts live under 3 different prefixes,
so a single --ckpt-pattern can't reach them all).

  CORR = containment (headline)   EM = exact match   REC = token recall   F1 = legacy
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

ROOT = Path(__file__).resolve().parents[2]
CKPTS = {
    "graph_v5_baseline":   ROOT / "outputs/repr_learning/v56_graph_v5_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt",
    "recurrent_baseline":  ROOT / "outputs/repr_learning/tranche6_mamba_1280_recurrent_baseline/ckpts/recurrent_baseline.best.pt",
    "flat_baseline":       ROOT / "outputs/repr_learning/tranche6_baselines_flat_baseline/ckpts/flat_baseline.best.pt",
    "continuous_baseline": ROOT / "outputs/repr_learning/tranche6_baselines_continuous_baseline/ckpts/continuous_baseline.best.pt",
    "memorizing_baseline": ROOT / "outputs/repr_learning/tranche6_baselines_memorizing_baseline/ckpts/memorizing_baseline.best.pt",
}

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--families", nargs="+",
                default=["biographical", "hotpot_qa", "musique", "narrative_qa", "babilong"])
ap.add_argument("--variants", nargs="+", default=list(CKPTS.keys()))
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--chunk-size", type=int, default=8192)
ap.add_argument("--window-size", type=int, default=1024)
ap.add_argument("--max-new-tokens", type=int, default=40)
ap.add_argument("--tag", default="rescore_newmetric")
ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/repr_learning/eval_per_family")
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
device = "cuda"

# Config identical to eval_per_family.main() (tranche-3 / v5.5 matched).
cfg = ReprConfig(
    fixed_window_size=args.window_size, max_window_size=args.chunk_size,
    d_node_state=128, n_edges=68, n_flat_codes=128,
    d_continuous=1398, d_concept_baseline=1398, d_mt_value=1398, d_recurrent=1398,
    graph_v5_K_node=128, graph_v5_K_edge=196, graph_v5_K_proposal=196,
    graph_v5_d_node=384, graph_v5_d_state=384, graph_v5_d_updater=640,
    graph_v5_updater_layers=5, graph_v5_n_message_rounds=6, graph_v5_mp_d_hidden=1024,
    d_enc=768, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3072,
    d_mamba=1280, edge_token_packing="fused",
)
passages = max(75, (args.chunk_size // 1024) * 75)

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.bfloat16).to(device)
llama.train(False)

# Shared, fixed sample set across all variants.
samples = EPF.collect_samples(args.families, args.n, tokenizer=tok, cfg=cfg,
                              chunk_size=args.chunk_size, passages_per_chunk=passages)
by_fam_all = defaultdict(list)
for s in samples:
    by_fam_all[s.family].append(s)

results = {}
steps = {}
for variant in args.variants:
    ckpt = CKPTS[variant]
    print(f"\n══ {variant}\n   {ckpt}", flush=True)
    try:
        model, step = EPF.load_variant(variant, ckpt, cfg, llama)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"   SKIP: {e}")
        continue
    model = model.to(device)
    model.train(False)
    steps[variant] = step
    results[variant] = {fam: [] for fam in args.families}
    ct = getattr(model, "chat_template", None)
    for fam in args.families:
        fs = by_fam_all[fam]
        with torch.no_grad():
            for i in range(0, len(fs), args.batch_size):
                batch = fs[i:i + args.batch_size]
                mem, aux = EPF._stream_encode_batch(model, batch, device, args.window_size)
                _, clean = EPF.generate_answers(
                    llama, tok, mem.detach(), batch, args.max_new_tokens, device,
                    memory_mask=aux.get("memory_mask"), chat_template=ct)
                for j, s in enumerate(batch):
                    results[variant][fam].append({
                        "pred": clean[j], "refs": s.answer_refs,
                        "em": EPF.max_over_refs(clean[j], s.answer_refs, EPF.em_score),
                        "contain": EPF.max_over_refs(clean[j], s.answer_refs, EPF.containment_score),
                        "recall": EPF.max_over_refs(clean[j], s.answer_refs, EPF.recall_score),
                        "f1": EPF.max_over_refs(clean[j], s.answer_refs, EPF.f1_score),
                    })
        rs = results[variant][fam]

        def _a(k, rs=rs):
            return 100 * sum(r[k] for r in rs) / max(1, len(rs))
        print(f"   {fam:14s} n={len(rs):3d}  CORR={_a('contain'):5.1f}  "
              f"EM={_a('em'):5.1f}  REC={_a('recall'):5.1f}  F1={_a('f1'):5.1f}", flush=True)
    del model
    torch.cuda.empty_cache()

# ── Table ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 92)
print(f"RE-SCORE — headline=CORRECTNESS (containment) / EM   (n={args.n}/family)")
print("=" * 92)
hdr = "  ".join(f"{f[:12]:>13s}" for f in args.families)
print(f"\n  {'variant':22s} {'macroCORR':>9s}  {hdr}")
print(f"  {'-'*22} {'-'*9}  " + "  ".join("-" * 13 for _ in args.families))
ranking = []
for v in args.variants:
    if v not in results:
        continue
    cells, cons = [], []
    for fam in args.families:
        rs = results[v][fam]
        if not rs:
            cells.append(f"{'—':>13s}"); continue
        c = 100 * sum(r["contain"] for r in rs) / len(rs)
        e = 100 * sum(r["em"] for r in rs) / len(rs)
        cons.append(c)
        cells.append(f"C{c:4.1f}/EM{e:4.1f}")
    macro = sum(cons) / len(cons) if cons else 0.0
    ranking.append((macro, v))
    print(f"  {v:22s} {macro:9.1f}  " + "  ".join(cells))
print("\n  ranking by macro containment:")
for m, v in sorted(ranking, reverse=True):
    print(f"    {m:5.1f}  {v}")

# ── Persist ────────────────────────────────────────────────────────────────
summary = {
    "tag": args.tag, "n_per_family": args.n, "families": args.families,
    "headline_metric": "containment", "variant_steps": steps,
    "by_variant": {
        v: {fam: {
            "n": len(rs),
            "containment": sum(r["contain"] for r in rs) / max(1, len(rs)),
            "em": sum(r["em"] for r in rs) / max(1, len(rs)),
            "recall": sum(r["recall"] for r in rs) / max(1, len(rs)),
            "f1": sum(r["f1"] for r in rs) / max(1, len(rs)),
        } for fam, rs in fr.items()}
        for v, fr in results.items()
    },
}
(args.out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
with open(args.out_dir / f"{args.tag}_per_sample.jsonl", "w") as f:
    for v, fr in results.items():
        for fam, rs in fr.items():
            for r in rs:
                f.write(json.dumps({"variant": v, "family": fam, **r}) + "\n")
print(f"\n[output] {args.out_dir}/{args.tag}_summary.json")
