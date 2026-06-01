#!/usr/bin/env python3
"""Edge-ablation diagnostic: is graph_v5's message-passing (edges) load-bearing?

Encodes the SAME multi-hop (hotpot+musique) batches at full message-passing
(T=6 rounds) vs nodes-only (T=0, message-passing ablated) and measures how much
the 128 memory tokens handed to Llama actually change.

cos(T6, T0) ~ 1  =>  6 rounds of edge message-passing barely change the memory
                     => edges are DECORATIVE (fix = relational supervision).
cos(T6, T0) low  =>  message-passing substantially reshapes the memory
                     => edges are load-bearing (escalate to answer-level test).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
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
print("loading llama + tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
llama = AutoModelForCausalLM.from_pretrained(
    cfg.llama_model, torch_dtype=torch.bfloat16).to(device)
llama.train(False)
ckpt = Path("outputs/repr_learning/outputs/repr_learning/"
            "tranche4_graph_v5_baseline_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt")
model, step = EPF.load_variant("graph_v5_baseline", ckpt, cfg, llama)
model.to(device)
model.train(False)
ro = model.encoder.readout
print(f"loaded graph_v5 @ step {step}; readout.T={ro.T}", flush=True)

samples = EPF.collect_samples(["hotpot_qa", "musique"], 24, tokenizer=tok, cfg=cfg,
                              chunk_size=8192, passages_per_chunk=600)
print(f"collected {len(samples)} multi-hop samples", flush=True)


_orig = ro.forward
_mode = {"m": "full"}


def patched(N, alpha_src, alpha_dst, edge_state, *args, **kwargs):
    if _mode["m"] == "noedge":          # zero routing -> agg=0 -> nodes-only
        alpha_src = torch.zeros_like(alpha_src)
        alpha_dst = torch.zeros_like(alpha_dst)
    elif _mode["m"] == "nostate":       # keep routing, drop relational payload
        edge_state = torch.zeros_like(edge_state)
    return _orig(N, alpha_src, alpha_dst, edge_state, *args, **kwargs)


ro.forward = patched


def encode_all(mode):
    _mode["m"] = mode
    mems = []
    for i in range(0, len(samples), 8):
        mem, _ = EPF._stream_encode_batch(model, samples[i:i + 8], device, 1024)
        mems.append(mem.float().cpu())
    return torch.cat(mems, 0)   # [N, K_node, d_llama]


with torch.no_grad():
    m_full = encode_all("full")
    m_noedge = encode_all("noedge")    # message-passing entirely neutralized
    m_nostate = encode_all("nostate")  # relational edge state neutralized
_mode["m"] = "full"


def tokcos(a, b):
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)


c_ne, c_ns = tokcos(m_full, m_noedge), tokcos(m_full, m_nostate)
rel_ne = ((m_full - m_noedge).norm(dim=-1) / m_full.norm(dim=-1).clamp_min(1e-6)).mean()
rel_ns = ((m_full - m_nostate).norm(dim=-1) / m_full.norm(dim=-1).clamp_min(1e-6)).mean()
N, K = m_full.shape[0], m_full.shape[1]
print(f"\n=== EDGE-ABLATION on graph_v5 @ step {step} -- {N} multi-hop samples x {K} memory tokens ===")
print(f"  FULL vs NO-MESSAGE-PASSING (nodes only):")
print(f"    cos mean={c_ne.mean():.4f}  median={c_ne.median():.4f}  "
      f"frac>0.99={(c_ne > 0.99).float().mean():.3f}  frac>0.95={(c_ne > 0.95).float().mean():.3f}")
print(f"    rel ||Δ||/||full|| = {rel_ne:.4f}")
print(f"  FULL vs NO-EDGE-STATE (routing on, relational payload off):")
print(f"    cos mean={c_ns.mean():.4f}  frac>0.99={(c_ns > 0.99).float().mean():.3f}")
print(f"    rel ||Δ||/||full|| = {rel_ns:.4f}")
print("\n  cos~1 / rel~0  => that component is DECORATIVE (memory barely changes).")
print("  If NO-MESSAGE-PASSING is ~identical => edges/MP not load-bearing => relational supervision.")
