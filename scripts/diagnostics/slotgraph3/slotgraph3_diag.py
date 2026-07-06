"""slotgraph3 diagnosis — why is SHUF−REAL ≈ 0 on babi (vs healthy on mae)?

Ports the prior slotgraph instrumentation (slotgraph_diag structure probe, slotgraph_gradflow
component-gradient groups, the depth×time per-window trace) onto the slotgraph3 checkpoints.
Per task (mae = healthy control, babi = the failure, condrecon_bio = known-gameable):

  A. MEMORY SPECIFICITY   cross-example cosine + eff-rank of the emitted memory.
                          SHUF−REAL≈0 ⇔ memory ≈ identical across examples — measure it directly.
  B. WRITE over TIME      per-window across-example divergence of node/edge latents + per-window
                          write magnitude ‖Δlat‖/‖lat‖. Does babi's write ignore the input?
  C. TOPOLOGY formation   per-window A from the traced latents → routing_diversity, edges/node,
                          nodes_used + cross-example JACCARD of the top-k edge SET (is the wiring
                          input-dependent?) + within-sample node distinctness (collapse).
  D. GRADIENT probe       real-path compute_loss fwd+bwd (train mode, autocast bf16, same as the
                          trainer) → per-component grad norms (router vs phi vs heads vs mixer)
                          + per-window latent grads. Is babi's router gradient-starved vs mae?

Usage:
  python scripts/diagnostics/slotgraph3/slotgraph3_diag.py --ckpt outputs/memory/sg3_customx_slotgraph3_baseline/ckpts/slotgraph3_baseline.best.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.training import make_mixed_val_sets, to_device
from src.memory.data.mixes import TASK_MODE
from src.memory.data.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS

DEV = "cuda"
TASKS = ["mae", "babi", "condrecon_bio"]
N_BATCHES = 2                       # 2 × BS8 = 16 examples per task


def load_model(ckpt: Path):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]
    valid = {f.name for f in dataclasses.fields(ReprConfig)}
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    m = ReprLearningModel(cfg, variant="slotgraph3_baseline", llama_model=None).to(DEV)
    res = m.load_state_dict(sd["model_state_dict"], strict=False)
    bad = [k for k in res.missing_keys
           if "llama" not in k.lower() and not k.startswith(("encoder.base.", "decoder.llama."))]
    if bad:
        print(f"  [warn] missing non-base keys: {bad[:8]}")
    m.eval()
    step = sd["metadata"].get("step", "?")
    print(f"loaded {ckpt.name} (step {step}, write={getattr(cfg, 'slotgraph3_write', 'lm')})")
    return m, cfg, AutoTokenizer.from_pretrained(cfg.llama_model)


def xexample_cos(x):
    """[B,S,d] → mean over example pairs i<j of mean_s cos(x[i,s], x[j,s]). 1.0 = identical across examples."""
    xn = F.normalize(x.float(), dim=-1)
    sim = torch.einsum("isd,jsd->ijs", xn, xn).mean(-1)
    B = sim.shape[0]
    off = sim.sum() - sim.diagonal().sum()
    return float(off / (B * (B - 1)))


def within_cos(x):
    """[B,S,d] → within-example mean pairwise cos of the S rows (collapse metric)."""
    xn = F.normalize(x.float(), dim=-1)
    c = xn @ xn.transpose(-1, -2)
    S = x.shape[1]
    off = c.sum((-1, -2)) - c.diagonal(dim1=-2, dim2=-1).sum(-1)
    return float((off / (S * (S - 1))).mean())


def effrank(x):
    x = x.detach().float().reshape(-1, x.shape[-1])
    xc = x - x.mean(0, keepdim=True)
    C = xc.t() @ xc
    return float(torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-12))


def topo_stats(enc, edge_lat, node_lat=None):
    """Recompute A from the traced latents → routing diversity / support / usage / cross-example Jaccard."""
    with torch.no_grad():
        key = node_lat if (getattr(enc, "route_key", "edge") == "node" and node_lat is not None) else edge_lat
        A = enc._route(key.detach())
        K = enc.K
        dp = A.argmax(-1)                                    # [B,K]
        oh = F.one_hot(dp, K).float().mean(0)                # [K,K] across-batch argmax distribution
        rdiv = float((-(oh.clamp_min(1e-9).log() * oh).sum(-1)).mean() / math.log(K))
        epn = float((A > 0).float().sum(-1).mean())
        used = float(dp.reshape(-1).unique().numel())
        topi = A.topk(enc.read_topk, dim=-1).indices         # [B,K,k]
        B = A.shape[0]
        sets = [set((i, int(j)) for i in range(K) for j in topi[b, i]) for b in range(B)]
        jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
               for a in range(B) for b in range(a + 1, B)]
        return rdiv, epn, used, sum(jac) / len(jac)


GROUPS = {
    "ROUTER q_route":  ("encoder.q_route",),
    "ROUTER k_proj":   ("encoder.k_proj",),
    "EDGE   phi":      ("encoder.phi",),
    "STATE  lat_init": ("encoder.node_lat_init", "encoder.edge_lat_init"),
    "STATE  id/role":  ("encoder.node_id", "encoder.role", "encoder.id_scale"),
    "WRITE  heads":    ("encoder.n_head", "encoder.head_node", "encoder.head_edge"),
    "WRITE  betas":    ("encoder.beta_node", "encoder.beta_edge"),
    "READ   norm":     ("encoder.norm",),
    "MIXER  blocks":   ("encoder.blocks",),
    "MIXER  enc-LoRA": ("encoder.base",),
    "decoder LoRA":    ("decoder.llama",),
}


def grad_norms(model):
    out = {}
    for g, prefixes in GROUPS.items():
        tot, n = 0.0, 0
        for name, p in model.named_parameters():
            if p.grad is not None and p.requires_grad and any(name.startswith(pre) for pre in prefixes):
                tot += float(p.grad.norm() ** 2); n += 1
        out[g] = (math.sqrt(tot) if n else None, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    model, cfg, tok = load_model(Path(args.ckpt))
    enc = model.encoder
    vs = make_mixed_val_sets(TASKS, tok, cfg, N_BATCHES, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=BABI_DEFAULT_TASKS, predict_len=64)

    # ── probes A/B/C: eval mode, grad enabled only so the _trace retain_grad hook is legal ──
    print(f"\n{'='*100}\nA/B/C — memory specificity, write-over-time, topology  ({N_BATCHES}×{cfg.batch_size} examples/task)\n{'='*100}")
    for task in TASKS:
        model.task_mode = TASK_MODE[task]
        mem_x, mem_er, nlat_x, elat_x = [], [], [], []
        win_rows = {}                                             # w → aggregated stats
        for batch in vs[task]:
            batch = to_device(batch, DEV)
            emb = model.decoder.llama.get_input_embeddings()(batch.context_ids)
            enc._trace = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                st = enc.init_streaming_state(emb.shape[0], emb.device, emb.dtype)
                st, _ = enc.streaming_write(st, emb, batch.context_mask)
                memory, aux = enc.finalize_memory(st)
            mem_x.append(xexample_cos(memory)); mem_er.append(effrank(memory))
            for w, nl, el in enc._trace:
                nl, el = nl.detach(), el.detach()
                rdiv, epn, used, jac = topo_stats(enc, el, nl)
                r = win_rows.setdefault(w, dict(nx=[], ex=[], wc=[], rd=[], ep=[], us=[], ja=[], dn=[], de=[]))
                r["nx"].append(xexample_cos(nl)); r["ex"].append(xexample_cos(el))
                r["wc"].append(within_cos(nl)); r["rd"].append(rdiv); r["ep"].append(epn)
                r["us"].append(used); r["ja"].append(jac)
            # per-window write magnitude ‖Δ‖/‖lat‖ (trace holds post-update latents)
            tr = enc._trace
            for w in range(len(tr)):
                prev_n = enc.node_lat_init.detach()[None] if w == 0 else tr[w - 1][1].detach()
                prev_e = enc.edge_lat_init.detach()[None] if w == 0 else tr[w - 1][2].detach()
                dn = float(((tr[w][1].detach() - prev_n).norm() / prev_n.norm().clamp_min(1e-9)))
                de = float(((tr[w][2].detach() - prev_e).norm() / prev_e.norm().clamp_min(1e-9)))
                win_rows[w]["dn"].append(dn); win_rows[w]["de"].append(de)
            nlat_x.append(xexample_cos(tr[-1][1].detach())); elat_x.append(xexample_cos(tr[-1][2].detach()))
            enc._trace = None
        avg = lambda l: sum(l) / len(l)
        print(f"\n[{task}]  MEMORY cross-example cos = {avg(mem_x):.4f}  (1.0 = generic ⇒ SHUF−REAL≈0)   "
              f"mem effrank = {avg(mem_er):.1f}")
        print(f"        final-latent cross-example cos: node={avg(nlat_x):.4f}  edge={avg(elat_x):.4f}")
        print(f"        {'win':>4} {'node_xcos':>10} {'edge_xcos':>10} {'node_wcos':>10} {'rdiv':>7} "
              f"{'edges/n':>8} {'used':>5} {'jaccard':>8} {'|Δn|/|n|':>9} {'|Δe|/|e|':>9}")
        for w in sorted(win_rows):
            r = win_rows[w]
            print(f"        {w:>4} {avg(r['nx']):>10.4f} {avg(r['ex']):>10.4f} {avg(r['wc']):>10.4f} "
                  f"{avg(r['rd']):>7.3f} {avg(r['ep']):>8.2f} {avg(r['us']):>5.1f} {avg(r['ja']):>8.3f} "
                  f"{avg(r['dn']):>9.4f} {avg(r['de']):>9.4f}")

    # ── probe D: real-path gradient probe (train mode, like the trainer) ──
    print(f"\n{'='*100}\nD — real-path gradient probe (compute_loss fwd+bwd, train mode)\n{'='*100}")
    hdr = f"{'group':<18}" + "".join(f"{t:>16}" for t in TASKS)
    per_task = {}
    for task in TASKS:
        model.task_mode = TASK_MODE[task]
        model.train()
        model.zero_grad(set_to_none=True)
        enc._trace = []
        batch = to_device(vs[task][0], DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(batch, window_size=1024)
        out["loss"].backward()
        per_task[task] = dict(gn=grad_norms(model), loss=float(out["loss"]),
                              wg=[(w, float(nl.grad.norm()) if nl.grad is not None else None,
                                   float(el.grad.norm()) if el.grad is not None else None)
                                  for w, nl, el in enc._trace])
        enc._trace = None
        model.zero_grad(set_to_none=True)
        model.eval()
    print(hdr)
    for g in GROUPS:
        row = f"{g:<18}"
        for t in TASKS:
            v, n = per_task[t]["gn"][g]
            row += f"{('—' if v is None or n == 0 else f'{v:.3f}'):>16}"
        print(row)
    print("\nper-window latent grads (node/edge):")
    for t in TASKS:
        wg = per_task[t]["wg"]
        s = "  ".join(f"w{w}:{n:.2f}/{e:.2f}" for w, n, e in wg if n is not None)
        print(f"  {t:<14} loss={per_task[t]['loss']:.3f}   {s}")


if __name__ == "__main__":
    main()
