"""slotgraph instrument panel + reporting standard.

Run after every fix/training run to get a comparable table. Every metric is
defined ONCE in METRICS (label / meaning / how-measured / good-direction); that
registry drives both the table and the glossary, so a report is always
self-documenting. Compare two designs with --vs.

Single run:   .venv/bin/python scripts/diagnostics/slotgraph_metrics.py --prefix valrun_slotgraph --label baseline
Compare two:  .venv/bin/python scripts/diagnostics/slotgraph_metrics.py --prefix valrun_slotgraph --vs valrun_myfix \
                  --label baseline --vs-label myfix

Lenses (the recurring question for every selection):
  • decisiveness — sharp given ONE input?           (↑ good)
  • cross-input diversity — responds to the input?  (↑ good = the topology signal)

Writes docs/slotgraph_metrics.md (single) or docs/slotgraph_compare.md (--vs).
"""
import sys, os, glob, math, statistics, dataclasses, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

DEV = "cuda"
ROOT = "outputs/memory"
TASKS = ("mae", "babi")

# ── metric registry: the single source of truth for table + glossary ──────────
# key, stage, scope ('task'|'perf'), label, meaning, how, dir ('up'|'down'|'zero'|'ctx')
METRICS = [
    ("w_slot_rank",  "WRITE",  "task", "slot rank",
     "distinct directions among the 32 slots within one input",
     "participation ratio of slot_final (final-layer slot hiddens) over the M slots, per example, mean", "up"),
    ("w_input_dep",  "WRITE",  "task", "per-slot rank across inputs",
     "how much each slot position varies with the input (write input-dependence)",
     "mean over slot positions of the participation ratio of slot_final[:,pos] across samples", "up"),
    ("w_redund",     "WRITE",  "task", "inter-slot cosine",
     "redundancy among the slots (1 = identical)",
     "mean pairwise cosine of the M slot vectors, per example, mean", "down"),
    ("sel_conf",     "SELECT", "task", "endpoint decisiveness",
     "how peaked each edge's endpoint choice is given one input (1 = one-hot)",
     "1 − normalized entropy of the soft endpoint distribution, mean over edges", "up"),
    ("sel_div",      "SELECT", "task", "endpoint cross-input diversity",
     "does an edge pick different nodes for different inputs — THE topology signal",
     "mean over edges of the normalized entropy of its argmax node pick across samples", "up"),
    ("coverage",     "SELECT", "task", "node coverage",
     "fraction of nodes used as an endpoint somewhere in the batch",
     "fraction of K nodes selected ≥1 time (pooled over samples)", "up"),
    ("usage_ent",    "SELECT", "task", "node-usage entropy",
     "balance of pooled node usage (1 = uniform, 0 = one hub)",
     "normalized entropy of pooled endpoint counts over the K nodes", "up"),
    ("node_select",  "SELECT", "task", "per-node cross-sample variance",
     "do individual nodes swing in usage with the input (input-selective) vs fixed roles",
     "per node, std across samples of its per-sample usage fraction; mean over nodes", "up"),
    ("edge_distinct","SELECT", "task", "edge distinctness",
     "fraction of edges with a distinct (src,dst) pair within an input",
     "mean over samples of #unique (src,dst) / E", "up"),
    ("selfloop",     "SELECT", "task", "self-loop frac",
     "edges pointing a node to itself (should be 0 by construction)",
     "fraction of edges with src == dst", "zero"),
    ("router_id_frac","SELECT","task", "router id-vs-content",
     "fraction of routing key/query magnitude from the FIXED id-stream vs input-dependent content — high = id drowns content (a cause of input-blindness)",
     "‖id-projection‖/(‖content‖+‖id‖) through the node-key + edge-query heads, mean", "down"),
    ("key_inputdep", "SELECT", "task", "routing-key input-dependence",
     "do the per-node routing keys vary across inputs, BEFORE the argmax (separates 'keys are fixed' from 'argmax kills variation')",
     "mean over nodes of the participation ratio of the node key across samples", "up"),
    ("sel_margin",   "SELECT", "task", "selection margin",
     "gap between the top-1 and top-2 endpoint probability (≈0 = fragile near-tie)",
     "mean over edges of (p_top1 − p_top2) of the soft endpoint distribution", "up"),
    ("temp",         "SELECT", "task", "routing temperature",
     "softmax temperature; tiny → argmax saturates and input-gradient dies",
     "exp(log_temp)", "ctx"),
    ("mp_delta",     "MP-READ","task", "mp_delta",
     "how much the message-passing read changes the output vs a plain prepend",
     "1 − cos(slot_final, memory), mean over slots", "ctx"),
    ("mem_rank",     "OUTPUT", "task", "memory rank",
     "distinct directions in the prepended memory (compare to WRITE slot rank)",
     "participation ratio of the final memory over the M slots, per example, mean", "up"),
    ("babi_em",          "PERF", "perf", "babi exact-match",
     "babi answer accuracy", "exact-match over babi val", "up"),
    ("mae_loss",         "PERF", "perf", "mae loss",
     "masked-reconstruction loss", "recon CE on mae val", "down"),
    ("mae_shuf_real",    "PERF", "perf", "mae SHUF−REAL",
     "example-specificity on mae (reliable)", "loss(shuffled memory) − loss(real)", "up"),
    ("continuation_loss","PERF", "perf", "continuation loss",
     "next-token loss", "recon CE on continuation val", "down"),
    ("continuation_shuf_real","PERF","perf","continuation SHUF−REAL",
     "example-specificity on continuation (reliable)", "loss(shuffled) − loss(real)", "up"),
]
DIR_SYM = {"up": "↑", "down": "↓", "zero": "→0", "ctx": "·"}


def seed_dir(prefix, variant, s):
    return f"{prefix}_{variant}" if s == 42 else f"{prefix}_s{s}_{variant}"


def load(d):
    sd = torch.load(sorted(glob.glob(f"{ROOT}/{d}/ckpts/*.pt"))[-1], map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]; valid = {f.name for f in dataclasses.fields(ReprConfig)}
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    m.load_state_dict(sd["model_state_dict"], strict=False); m.eval()
    return m, cfg


def effrank(X):
    X = X.float()
    if X.shape[0] < 2:
        return 0.0
    Xc = X - X.mean(0, keepdim=True); C = Xc.t() @ Xc
    return float(C.trace() ** 2 / (C * C).sum().clamp_min(1e-12))


def ent_norm(counts, K):
    p = counts.float() / counts.sum().clamp_min(1e-9); p = p[p > 0]
    return float(-(p * p.log()).sum() / math.log(K)) if K > 1 else 0.0


def trace_pass(m, batches):
    """Return per-task scalar metrics + trajectories (layer/hop/node) for one task's batches."""
    m.encoder.use_structure = True; m.encoder.trace = True
    embed = m.decoder.llama.get_input_embeddings(); K = m.encoder.K
    SF, MEM, SRC, DST, SOFT = [], [], [], [], []
    hop, LAYER, u_mag, u_sat = {}, {}, {}, {}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b in batches:
            b = to_device(b, DEV)
            m.encoder(m._encode_for_memory(embed(b.context_ids), b.context_mask), b.context_mask)
            t = m.encoder._trace
            SF.append(t["slot_final"]); MEM.append(t["memory"]); SRC.append(t["src_pick"])
            DST.append(t["dst_pick"]); SOFT.append(t["soft_src"])
            for hi, h in enumerate(t["hop_h"]):
                hop.setdefault(hi, []).append(statistics.mean(effrank(h[bi]) for bi in range(h.shape[0])))
            for li, ls in enumerate(t.get("layer_slots", [])):
                LAYER.setdefault(li, []).append(ls)
    m.encoder.trace = False
    sf = torch.cat(SF); mem = torch.cat(MEM); src = torch.cat(SRC); dst = torch.cat(DST); soft = torch.cat(SOFT)
    N, M, d = sf.shape; E = src.shape[1]; eye = torch.eye(M).bool()
    cos = F.cosine_similarity(sf.unsqueeze(2), sf.unsqueeze(1), dim=-1)
    # router input decomposition: input-dependent content vs FIXED-id contribution to keys/queries
    enc = m.encoder; sf_g = sf.to(DEV)
    content = enc.struct_norm(sf_g).float(); id_stream = (enc.id_head_scale * enc.id_embed).float()
    def _decomp(head, sl):
        W = head.weight.float(); c = content[:, sl] @ W[:, :d].t(); i = id_stream[sl] @ W[:, d:].t()
        frac = float(i.norm(dim=-1).mean() / (c.norm(dim=-1).mean() + i.norm(dim=-1).mean() + 1e-9))
        return frac, (c + i.unsqueeze(0))
    k_frac, key_full = _decomp(enc.k_head, slice(0, K))
    q_frac, _ = _decomp(enc.q_src_head, slice(K, M))
    top2 = soft.topk(2, dim=-1).values
    # per-node usage matrix [N,K] → histogram + cross-sample variance
    usage = torch.stack([torch.bincount(src[i], minlength=K).float() + torch.bincount(dst[i], minlength=K).float()
                         for i in range(N)])
    usage = usage / usage.sum(1, keepdim=True).clamp_min(1)
    allpick = torch.cat([src.reshape(-1), dst.reshape(-1)])
    sc = {
        "w_slot_rank": statistics.mean(effrank(sf[i]) for i in range(N)),
        "w_input_dep": statistics.mean(effrank(sf[:, p, :]) for p in range(M)),
        "w_redund": float(cos[:, ~eye].mean()),
        "sel_conf": 1.0 - float((-(soft.clamp_min(1e-9).log() * soft).sum(-1) / math.log(K)).mean()),
        "sel_div": statistics.mean(ent_norm(torch.bincount(src[:, e], minlength=K), K) for e in range(E)),
        "coverage": float((torch.bincount(allpick, minlength=K) > 0).float().mean()),
        "usage_ent": ent_norm(torch.bincount(allpick, minlength=K), K),
        "node_select": float(usage.std(0).mean()),
        "edge_distinct": statistics.mean(len({(int(src[i, e]), int(dst[i, e])) for e in range(E)}) / E for i in range(N)),
        "selfloop": float((src == dst).float().mean()),
        "mp_delta": float((1 - F.cosine_similarity(sf.reshape(-1, d), mem.reshape(-1, d), dim=-1)).mean()),
        "mem_rank": statistics.mean(effrank(mem[i]) for i in range(N)),
        "router_id_frac": (k_frac + q_frac) / 2,
        "key_inputdep": statistics.mean(effrank(key_full[:, j, :].cpu()) for j in range(K)),
        "sel_margin": float((top2[..., 0] - top2[..., 1]).mean()),
        "temp": float(enc.log_temp.exp()),
    }
    layer_rank, layer_cos = {}, {}
    for li, chunks in LAYER.items():
        Lt = torch.cat(chunks)
        layer_rank[li] = statistics.mean(effrank(Lt[i]) for i in range(Lt.shape[0]))
        lc = F.cosine_similarity(Lt.unsqueeze(2), Lt.unsqueeze(1), dim=-1)
        layer_cos[li] = float(lc[:, ~eye].mean())
    traj = {"hop_rank": {hi: statistics.mean(v) for hi, v in hop.items()},
            "layer_rank": layer_rank, "layer_cos": layer_cos,
            "node_hist": sorted(usage.mean(0).tolist(), reverse=True)}
    return sc, traj


def perf_pass(m, vs):
    out = {}
    def run(fam, shuf):
        m.task_mode = MIXED_TASK_MODE[fam]; m.encoder.use_structure = True
        ls, eh, en = [], 0, 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for b in vs[fam]:
                b = to_device(b, DEV); o = m.compute_loss(b, window_size=1024, shuffle_memory=shuf)
                ls.append(o["loss_recon"].item())
                if fam == "babi":
                    for f, e in zip(b.task_family, o["per_example_em"].tolist()):
                        if f == "babi":
                            eh += e; en += 1
        return statistics.mean(ls), (eh / en if en else None)
    out["babi_em"] = run("babi", False)[1]
    for fam in ("mae", "continuation"):
        r, _ = run(fam, False); s, _ = run(fam, True)
        out[f"{fam}_loss"] = r; out[f"{fam}_shuf_real"] = s - r
    return out


def compute(prefix, seeds, variant="slotgraph_baseline"):
    seeds = [s for s in seeds if glob.glob(f"{ROOT}/{seed_dir(prefix, variant, s)}/ckpts/*.pt")]
    sc = {t: {} for t in TASKS}; trj = {t: {"hop_rank": {}, "layer_rank": {}, "layer_cos": {}, "node_hist": []} for t in TASKS}
    perf = {}
    tok = vs = None
    for s in seeds:
        m, cfg = load(seed_dir(prefix, variant, s))
        if tok is None:
            tok = AutoTokenizer.from_pretrained(cfg.llama_model); tok.pad_token = tok.pad_token or tok.eos_token
            vs = make_mixed_val_sets(["babi", "mae", "continuation"], tok, cfg, 8, ctx_len=1024, m_slots=32,
                                     mae_src_tok="meta-llama/Llama-3.2-1B",
                                     babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)
        for t in TASKS:
            s_sc, s_trj = trace_pass(m, vs[t])
            for k, v in s_sc.items():
                sc[t].setdefault(k, []).append(v)
            for grp in ("hop_rank", "layer_rank", "layer_cos"):
                for idx, v in s_trj[grp].items():
                    trj[t][grp].setdefault(idx, []).append(v)
            trj[t]["node_hist"].append(s_trj["node_hist"])
        for k, v in perf_pass(m, vs).items():
            perf.setdefault(k, []).append(v)
        del m; torch.cuda.empty_cache()
    # aggregate
    agg_sc = {t: {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in sc[t].items()} for t in TASKS}
    agg_perf = {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in perf.items() if v and v[0] is not None}
    agg_trj = {t: {grp: {idx: statistics.mean(v) for idx, v in trj[t][grp].items()} for grp in ("hop_rank", "layer_rank", "layer_cos")} for t in TASKS}
    for t in TASKS:
        nh = trj[t]["node_hist"]
        agg_trj[t]["node_hist"] = [statistics.mean(x) for x in zip(*nh)] if nh else []
    return {"scalars": agg_sc, "perf": agg_perf, "traj": agg_trj, "n": len(seeds)}


def val(R, key, scope, task=None):
    d = R["perf"] if scope == "perf" else R["scalars"][task]
    return d.get(key)


def spark(d):  # trajectory dict {idx:val} → compact, downsampled to ≤12 points
    if not d:
        return "—"
    xs = [d[i] for i in sorted(d)]
    if len(xs) > 12:
        step = math.ceil(len(xs) / 12); xs = xs[::step] + [d[max(d)]]
    return " ".join(f"{x:.1f}" for x in xs)


def fmt(v):
    return "—" if v is None else (f"{v[0]:.3f}±{v[1]:.3f}" if v[1] else f"{v[0]:.3f}")


def verdict(a, b, d):
    if a is None or b is None:
        return "—"
    delta = b[0] - a[0]
    if d in ("up", "down", "zero"):
        good = delta > 0 if d == "up" else delta < 0 if d == "down" else abs(b[0]) <= abs(a[0])
        return f"{delta:+.3f} {'✓' if good else '✗'}"
    return f"{delta:+.3f}"


def render(RA, RB, labA, labB):
    cmp = RB is not None
    L = [f"# slotgraph metrics — {'`'+labA+'` vs `'+labB+'`' if cmp else '`'+labA+'`'}"]
    L.append(f"\n{labA}: n={RA['n']} seeds" + (f" · {labB}: n={RB['n']} seeds" if cmp else "") +
             ". Every selection under two lenses — **decisiveness** (sharp given one input) and "
             "**cross-input diversity** (responds to the input = the topology signal). ✓/✗ = moved in the good direction.\n")
    stages = ["WRITE", "SELECT", "MP-READ", "OUTPUT"]
    L.append("## Structural metrics (per task: mae / babi)")
    head = "| metric | dir | "
    head += ("mae A→B | babi A→B |" if cmp else "mae | babi |")
    L.append(head); L.append("|" + "---|" * (4 if cmp else 4))
    for key, stage, scope, label, *_rest, d in [(m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for m in METRICS]:
        if scope != "task" or stage not in stages:
            continue
        a_m, a_b = val(RA, key, scope, "mae"), val(RA, key, scope, "babi")
        if cmp:
            b_m, b_b = val(RB, key, scope, "mae"), val(RB, key, scope, "babi")
            L.append(f"| {stage} {label} | {DIR_SYM[d]} | {fmt(a_m)}→{fmt(b_m)} {verdict(a_m,b_m,d)} "
                     f"| {fmt(a_b)}→{fmt(b_b)} {verdict(a_b,b_b,d)} |")
        else:
            L.append(f"| {stage} {label} | {DIR_SYM[d]} | {fmt(a_m)} | {fmt(a_b)} |")
    L.append("\n## Performance")
    L.append("| metric | dir | " + ("A→B |" if cmp else "value |")); L.append("|---|---|---|")
    for key, stage, scope, label, *_rest, d in [(m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for m in METRICS]:
        if scope != "perf":
            continue
        a = val(RA, key, scope)
        if cmp:
            b = val(RB, key, scope); L.append(f"| {label} | {DIR_SYM[d]} | {fmt(a)}→{fmt(b)} {verdict(a,b,d)} |")
        else:
            L.append(f"| {label} | {DIR_SYM[d]} | {fmt(a)} |")
    # trajectories
    L.append("\n## Depth profile (effective rank: write per LM-layer, then read per MP-hop)")
    for t in TASKS:
        L.append(f"- **{t}** write-layer rank: {spark(RA['traj'][t]['layer_rank'])}"
                 + (f"   ‖ {labB}: {spark(RB['traj'][t]['layer_rank'])}" if cmp else ""))
        L.append(f"  - read-hop rank:   {spark(RA['traj'][t]['hop_rank'])}"
                 + (f"   ‖ {labB}: {spark(RB['traj'][t]['hop_rank'])}" if cmp else ""))
        L.append(f"  - write-layer inter-slot cosine: {spark(RA['traj'][t]['layer_cos'])}"
                 + (f"   ‖ {labB}: {spark(RB['traj'][t]['layer_cos'])}" if cmp else ""))
    L.append("\n## Node usage histogram over samples (sorted desc; flat = fixed roles, peaked = hub)")
    for t in TASKS:
        nh = RA["traj"][t]["node_hist"]
        L.append(f"- **{t}** {labA}: " + " ".join(f"{x:.2f}" for x in nh))
        if cmp:
            L.append(f"  {labB}: " + " ".join(f"{x:.2f}" for x in RB['traj'][t]['node_hist']))
    # glossary
    L.append("\n## Metric glossary (what / how / good direction)")
    L.append("| metric | means | how measured | good |"); L.append("|---|---|---|---|")
    for key, stage, scope, label, meaning, how, d in METRICS:
        L.append(f"| {stage} {label} | {meaning} | {how} | {DIR_SYM[d]} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="valrun_slotgraph")
    ap.add_argument("--vs", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--vs-label", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()
    labA = args.label or args.prefix
    RA = compute(args.prefix, args.seeds)
    RB = compute(args.vs, args.seeds) if args.vs else None
    labB = args.vs_label or args.vs
    md = render(RA, RB, labA, labB)
    out = Path("docs/slotgraph_compare.md" if args.vs else "docs/slotgraph_metrics.md")
    out.write_text(md); print(f"wrote {out}\n"); print(md)


if __name__ == "__main__":
    main()
