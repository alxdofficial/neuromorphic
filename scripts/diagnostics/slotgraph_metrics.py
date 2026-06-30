"""slotgraph v3.1 instrument panel — measure the model on ALL metrics in one run.

Builds the model (optionally from --ckpt), runs mae+babi forwards in the REAL bf16 path, and prints
organized, self-documenting panels. Run pre-train (untrained sanity) and post-train (--ckpt) to compare.

  SUBSTRATE — vocab/edge/key/read-token eff-rank + norms: is the memory SPREAD, not collapsed? (v2 → ~2)
  ROUTING   — selection decisiveness, cross-input diversity (THE topology signal), node usage, self-loops
  CONNECTEDNESS — avg node degree / n_components / nodes_used: do clauses link into ONE graph, or stay a
              bag of separate dyads? (the "clauses connect, not separate" acceptance criterion)
  WRITE     — learned bounded delta rates (β), read gate
  GRADIENT  — LAST-layer selection grad/param vs content head (the v2 dead-zone canary; v2 was ~30,000×)
  READ      — read-xattn attention sharpness over the 144 graph tokens (binding=sharp vs pooling=diffuse)

  (binding — SHUF−REAL / OFF−REAL — is read from the TRAINING jsonl, not here: this panel's compute_loss
   doesn't replicate the training eval's QA loss, so its loss-diffs are unreliable. This panel is structure-only.)

Usage:
  .venv/bin/python scripts/diagnostics/slotgraph_metrics.py
  .venv/bin/python scripts/diagnostics/slotgraph_metrics.py --ckpt outputs/memory/slotgraph_baseline.best.pt
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer, AutoConfig
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.common import resolve_special_ids
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
DEV = "cuda"
TASKS = ("mae", "babi")

# glossary: key → (panel, label, one-line meaning, good-direction)
GLOSS = {
    "node_effrank":  ("SUBSTRATE", "vocab eff-rank /N", "distinct node meanings (v2 collapsed to ~2)", "up"),
    "edge_effrank":  ("SUBSTRATE", "edge eff-rank /E", "distinct edge states", "up"),
    "key_effrank":   ("SUBSTRATE", "key eff-rank /N", "spread of the fixed selection addresses", "up"),
    "mem_effrank":   ("SUBSTRATE", "read-token eff-rank /E", "distinct READ tokens the LM sees", "up"),
    "node_norm":     ("SUBSTRATE", "node norm", "post-delta-norm magnitude (~sqrt(dn)=8, stable)", "ctx"),
    "edge_norm":     ("SUBSTRATE", "edge norm", "post-delta-norm magnitude (~sqrt(dn)=8, stable)", "ctx"),
    "src_entropy":   ("ROUTING",   "src entropy", "decisiveness of src pick (down=sharper; max ln144=4.97)", "down"),
    "dst_entropy":   ("ROUTING",   "dst entropy", "decisiveness of dst pick (down=sharper)", "down"),
    "routing_diversity": ("ROUTING", "cross-input diversity", "edges pick diff nodes per input — topology signal", "up"),
    "node_entropy":  ("ROUTING",   "node-usage entropy", "balance of node usage (up=spread, down=hub)", "up"),
    "selfloop_frac": ("ROUTING",   "self-loop frac", "edges with src==dst (should be ~0; masked)", "zero"),
    "sel_scale":     ("ROUTING",   "sel temperature", "learned cosine->logit scale (init sqrt(dk)=8)", "ctx"),
    "beta_node":     ("WRITE",     "beta node", "learned bounded node-meaning delta rate (continuity)", "ctx"),
    "beta_edge":     ("WRITE",     "beta edge", "learned bounded edge-state delta rate", "ctx"),
    "read_gate":     ("WRITE",     "read gate", "mean tanh gate of the per-layer read (bootstraps from 0.1)", "ctx"),
    "avg_degree":    ("CONNECTEDNESS", "avg node degree", "edges per used node (>1 = reuse = clauses connect)", "up"),
    "n_components":  ("CONNECTEDNESS", "connected components", "1-few = ONE graph; ~E = a bag of separate clauses", "down"),
    "nodes_used":    ("CONNECTEDNESS", "nodes used", "distinct endpoint nodes (lower vs 2E=288 ⇒ more reuse)", "ctx"),
    "sel_gradparam": ("GRADIENT",  "last-layer sel grad/param", "selection-head grad mag (v2 starved to ~1e-5)", "up"),
    "sel_gap":       ("GRADIENT",  "sel-gap (content/sel)", "content-head / selection-head grad (v2 ~30,000x)", "down"),
    "read_entropy":  ("READ",      "read attn entropy", "read-xattn spread over 144 tokens (down=sharper=binding)", "down"),
    "read_maxw":     ("READ",      "read attn max-weight", "peak read attention weight (up=sharper)", "up"),
}


def build(ckpt=None):
    cfg = ReprConfig(use_llama_lora=True, batch_size=8)
    bc = AutoConfig.from_pretrained(BACKBONE)
    cfg.llama_model = BACKBONE; cfg.d_llama = bc.hidden_size; cfg.llama_vocab_size = bc.vocab_size
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.pad_token_id, cfg.sep_token_id = resolve_special_ids(tok)
    cfg.task_mode = "mixed"
    cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.slotgraph_d_node = 64; cfg.slotgraph_n_nodes = 144; cfg.slotgraph_n_edges = 144
    cfg.slotgraph_enc_layers = 4; cfg.slotgraph_xattn_heads = 4
    cfg.slotgraph_lora_rank = 56; cfg.slotgraph_lora_alpha = 112
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    if ckpt:
        sd = torch.load(ckpt, map_location=DEV)
        state = sd
        if isinstance(sd, dict):
            for k in ("model_state_dict", "model", "state_dict"):
                if k in sd:
                    state = sd[k]; break
        miss, unexp = m.load_state_dict(state, strict=False)
        print(f"loaded {ckpt}: {len(miss)} missing, {len(unexp)} unexpected keys")
    return m, tok, cfg


def read_sharpness(m, enc, b):
    cap = {}
    mid = len(enc.read_xattn) // 2
    def hook(mod, args):
        cap["h"] = args[0].detach(); cap["G"] = args[1].detach()
    hd = enc.read_xattn[mid].register_forward_pre_hook(hook)
    m.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        m.compute_loss(b, window_size=1024)
    hd.remove()
    if "h" not in cap:
        return None, None
    xa = enc.read_xattn[mid]
    with torch.no_grad():
        h = cap["h"].float(); G = cap["G"].float()
        B, S, _ = h.shape; U = G.shape[1]
        q = xa.q(h).view(B, S, xa.h, xa.hd).transpose(1, 2)
        k = xa.k(G).view(B, U, xa.h, xa.hd).transpose(1, 2)
        p = ((q @ k.transpose(-1, -2)) / math.sqrt(xa.hd)).softmax(-1)    # [B,h,S,U]
        ent = float((-(p.clamp_min(1e-9).log() * p).sum(-1)).mean() / math.log(U))
        return ent, float(p.max(-1).values.mean())


def measure_task(m, enc, b):
    # NOTE: binding (SHUF-REAL / OFF-REAL) is read from the TRAINING jsonl, not here — this panel's
    # compute_loss doesn't replicate the training eval's QA loss exactly, so its loss-diffs are unreliable.
    # Everything below is model-INTERNAL (structure/connectedness/read) and is reliable.
    r = {}
    m.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        real = m.compute_loss(b, window_size=1024)
    for k in ("node_effrank", "edge_effrank", "key_effrank", "mem_effrank", "node_norm", "edge_norm",
              "src_entropy", "dst_entropy", "routing_diversity", "node_entropy", "selfloop_frac",
              "sel_scale", "beta_node", "beta_edge", "read_gate",
              "avg_degree", "n_components", "nodes_used"):
        v = real.get(f"slotgraph_{k}")
        if v is not None:
            r[k] = float(v)
    ent, mw = read_sharpness(m, enc, b)
    if ent is not None:
        r["read_entropy"] = ent; r["read_maxw"] = mw
    return r


def grad_canary(m, enc, b, task):
    m.train(True); m.task_mode = MIXED_TASK_MODE[task]
    m.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = m.compute_loss(b, window_size=1024)
    out["loss"].backward()
    def gp(ps):
        ps = [p for p in ps if p.grad is not None]
        nd = sum(p.numel() for p in ps) or 1
        gd = sum(p.grad.abs().sum().item() for p in ps) / nd
        pm = sum(p.detach().abs().sum().item() for p in ps) / nd
        return gd / max(1e-12, pm)
    last = enc.gt_layers[-1]
    sel = min(gp(list(last.q_src.parameters())), gp(list(last.q_dst.parameters())))
    ec = gp([p for L in enc.gt_layers for p in L.headB.parameters()])
    m.zero_grad(set_to_none=True)
    return {"sel_gradparam": sel, "sel_gap": ec / max(1e-12, sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    m, tok, cfg = build(args.ckpt); enc = m.encoder
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    state = "trained ckpt" if args.ckpt else "UNTRAINED (init sanity)"
    print(f"\n{'='*86}\nslotgraph v3.1 instrument panel — {state}  ({tot/1e6:.2f}M trainable)\n{'='*86}")
    vs = make_mixed_val_sets(list(TASKS), tok, cfg, 8, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)
    data = {}
    for task in TASKS:
        b = to_device(vs[task][0], DEV)
        r = measure_task(m, enc, b)
        r.update(grad_canary(m, enc, b, task))
        data[task] = r

    for panel in ("SUBSTRATE", "ROUTING", "CONNECTEDNESS", "WRITE", "GRADIENT", "READ"):
        keys = [k for k, g in GLOSS.items() if g[0] == panel]
        print(f"\n-- {panel} " + "-" * (80 - len(panel)))
        print(f"  {'metric':26}{'meaning':44}{'mae':>8}{'babi':>8}")
        for k in keys:
            _, label, meaning, _ = GLOSS[k]
            def fmt(x):
                return "   --  " if x is None else f"{x:8.3f}"
            print(f"  {label:26}{meaning[:42]:44}{fmt(data['mae'].get(k))}{fmt(data['babi'].get(k))}")

    print("\n-- HOW TO READ " + "-" * 71)
    print("  no collapse       -> eff-ranks stay high (v2 collapsed to ~2)")
    print("  clauses connect   -> avg degree > 1 AND n_components << E (else a bag of separate clauses)")
    print("  selection sharp   -> src/dst entropy FALLING toward 0; sel-gap << 100x (v2 ~30,000x)")
    print("  binding (SHUF-REAL) is in the TRAINING jsonl, not here — this panel is structure-only.")
    print(f"{'='*86}\n")


if __name__ == "__main__":
    main()
