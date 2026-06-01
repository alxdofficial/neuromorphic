#!/usr/bin/env python3
"""VERIFY the edge-ablation finding two independent ways (beyond the cos probe).

A. MECHANISM: record edge_state / routing stats at the readout boundary.
   - edge_state norm vs node norm vs round-0 src_ctx norm
       -> small edge_state ⇒ write-side dead (edges never populated)
       -> large edge_state but tiny answer effect ⇒ read-side dead (ignored)
   - alpha_src/alpha_dst entropy + max over K_node (routing peakedness)
       -> entropy≈log(K_node)=4.85 ⇒ diffuse routing (no real topology)
   - per-round agg-norm / buf-norm (how much each MP round moves the buffer)

B. ANSWER-LEVEL (decisive): EM/F1 with edges ON (full) vs OFF (nodes-only),
   on the multi-hop families where edges *should* matter (hotpot, musique),
   plus biographical as a control (graph's winning family — is the win node-only?).
   If F1 is unchanged when edges are off, edges are causally dead for the task.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

cfg = ReprConfig(
    n_flat_codes=128, d_continuous=1398, d_concept_baseline=1398,
    d_mt_value=1398, d_recurrent=1398,
    d_enc=768, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3072,
    d_node_state=128, n_edges=68,
    graph_v5_K_node=128, graph_v5_K_edge=196, graph_v5_K_proposal=196,
    graph_v5_d_node=384, graph_v5_d_state=384, graph_v5_d_updater=640,
    graph_v5_updater_layers=5, graph_v5_n_message_rounds=6, graph_v5_mp_d_hidden=1024,
    d_mamba=1280, edge_token_packing="fused",
    max_window_size=8192, fixed_window_size=1024,
)
device = "cuda"
FAMS = ["hotpot_qa", "musique", "biographical"]
NPF = 32

print("loading llama + tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.float32).to(device)
llama.train(False)
ckpt = Path("outputs/repr_learning/v56_graph_v5_graph_v5_baseline/"
            "ckpts/graph_v5_baseline.best.pt")
model, step = EPF.load_variant("graph_v5_baseline", ckpt, cfg, llama)
model.to(device)
model.train(False)
ro = model.encoder.readout
chat_template = getattr(model, "chat_template", None)
print(f"loaded graph_v5 @ step {step}; readout.T={ro.T}; chat_template={chat_template is not None}", flush=True)

samples = EPF.collect_samples(FAMS, NPF, tokenizer=tok, cfg=cfg,
                              chunk_size=8192, passages_per_chunk=600)
by_fam = defaultdict(list)
for s in samples:
    by_fam[s.family].append(s)
print(f"collected {len(samples)} samples: " +
      ", ".join(f"{f}={len(v)}" for f, v in by_fam.items()), flush=True)

# ── monkeypatch readout: record stats (full mode) + apply ablation ──────────
_orig = ro.forward
_mode = {"m": "full", "rec": False}
stats = defaultdict(list)
captured = {}


def _entropy_over_nodes(al):                       # al: [B, K_edge, K_node], rows≈softmax
    p = al.float().clamp_min(1e-12)
    p = p / p.sum(-1, keepdim=True)
    return (-(p * p.log()).sum(-1)).flatten().cpu()  # [B*K_edge]


def patched(N, alpha_src, alpha_dst, edge_state, *a, **k):
    if _mode["rec"]:
        with torch.no_grad():
            stats["edge_state_norm"].append(edge_state.float().norm(dim=-1).flatten().cpu())
            stats["node_norm"].append(N.float().norm(dim=-1).flatten().cpu())
            # round-0 src context the message MLP actually sees (cat with edge_state)
            src_ctx0 = torch.matmul(alpha_src, ro.pre_norm(ro.W_init(N)))
            stats["src_ctx0_norm"].append(src_ctx0.float().norm(dim=-1).flatten().cpu())
            stats["alpha_src_ent"].append(_entropy_over_nodes(alpha_src))
            stats["alpha_dst_ent"].append(_entropy_over_nodes(alpha_dst))
            stats["alpha_src_max"].append(alpha_src.float().max(-1).values.flatten().cpu())
            stats["alpha_dst_max"].append(alpha_dst.float().max(-1).values.flatten().cpu())
            if "args" not in captured:               # stash 1 batch for telemetry pass
                captured["args"] = (N.detach().clone(), alpha_src.detach().clone(),
                                    alpha_dst.detach().clone(), edge_state.detach().clone())
    if _mode["m"] == "noedge":
        alpha_src = torch.zeros_like(alpha_src)
        alpha_dst = torch.zeros_like(alpha_dst)
    elif _mode["m"] == "nostate":
        edge_state = torch.zeros_like(edge_state)
    return _orig(N, alpha_src, alpha_dst, edge_state, *a, **k)


ro.forward = patched


def encode(batch, mode, rec=False):
    _mode["m"], _mode["rec"] = mode, rec
    mem, aux = EPF._stream_encode_batch(model, batch, device, 1024)
    _mode["rec"] = False
    return mem, aux


# ════════════════ PART A: mechanism statistics ════════════════
print("\n[A] recording edge_state / routing stats (full mode)...", flush=True)
with torch.no_grad():
    for f, v in by_fam.items():
        for i in range(0, len(v), 4):
            encode(v[i:i + 4], "full", rec=True)


def summ(key):
    x = torch.cat(stats[key])
    return f"mean={x.mean():.3f} med={x.median():.3f} p10={x.quantile(0.1):.3f} p90={x.quantile(0.9):.3f}"


K_node = cfg.graph_v5_K_node
print(f"\n=== [A] MECHANISM (graph_v5 @ step {step}) ===")
print(f"  node ‖N‖              : {summ('node_norm')}")
print(f"  edge_state ‖e‖        : {summ('edge_state_norm')}")
print(f"  round-0 src_ctx ‖     : {summ('src_ctx0_norm')}   <- the OTHER half of the message MLP input")
es = torch.cat(stats['edge_state_norm']); sc = torch.cat(stats['src_ctx0_norm'])
print(f"  edge_state / src_ctx ratio (mean) : {(es.mean()/sc.mean()):.3f}  "
      f"(<<1 ⇒ MLP input dominated by node ctx, edge payload negligible)")
print(f"  routing entropy  max={math.log(K_node):.3f} (uniform/diffuse), 0 (one-hot/sharp):")
print(f"    alpha_src H : {summ('alpha_src_ent')}")
print(f"    alpha_dst H : {summ('alpha_dst_ent')}")
print(f"    alpha_src max-weight : {summ('alpha_src_max')}")
print(f"    alpha_dst max-weight : {summ('alpha_dst_max')}")

# telemetry pass: per-round agg vs buf norm on one captured batch
with torch.no_grad():
    _, telem = _orig(*captured["args"], compute_telemetry=True)
buf = telem["mp_buf_norm_per_round"].tolist()
agg = telem["mp_agg_norm_per_round"].tolist()
print(f"  per-round  agg‖/buf‖ (how much each MP round moves the buffer):")
for t, (b, g) in enumerate(zip(buf, agg)):
    print(f"    round {t}: buf={b:.3f}  agg={g:.3f}  agg/buf={g/max(b,1e-6):.4f}")

# ════════════════ PART B: answer-level EM/F1 (full vs noedge) ════════════════
print("\n[B] answer-level EM/F1: full vs noedge (routing off) vs nostate (edge_state off)...", flush=True)
MODES = ("full", "noedge", "nostate")
res = {m: defaultdict(list) for m in MODES}
with torch.no_grad():
    for f, v in by_fam.items():
        for mode in MODES:
            for i in range(0, len(v), 4):
                batch = v[i:i + 4]
                mem, aux = encode(batch, mode)
                _, clean = EPF.generate_answers(
                    llama, tok, mem.detach(), batch, 40, device,
                    memory_mask=aux.get("memory_mask"), chat_template=chat_template)
                for j, s in enumerate(batch):
                    res[mode][f].append((
                        EPF.max_over_refs(clean[j], s.answer_refs, EPF.em_score),
                        EPF.max_over_refs(clean[j], s.answer_refs, EPF.f1_score),
                    ))


def f1_of(mode, f):
    return 100 * sum(x[1] for x in res[mode][f]) / len(res[mode][f])


print(f"\n=== [B] ANSWER-LEVEL F1 (graph_v5 @ step {step}, n={NPF}/family) ===")
print(f"  {'family':<14} {'full':>7} {'noedge':>7} {'nostate':>8}   {'ΔMP':>6} {'Δstate':>7}")
for f in by_fam:
    ff, fn, fs = f1_of('full', f), f1_of('noedge', f), f1_of('nostate', f)
    print(f"  {f:<14} {ff:>7.1f} {fn:>7.1f} {fs:>8.1f}   {ff-fn:>+6.1f} {ff-fs:>+7.1f}")
print("\n  ΔMP   = full − noedge  (routing/message-passing contribution)")
print("  Δstate= full − nostate (relational edge_STATE payload contribution)")
print("  Δstate≈0 everywhere ⇒ relational edge_state is read-side DEAD; any MP value is node-pooling.")
