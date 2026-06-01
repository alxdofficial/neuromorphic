#!/usr/bin/env python3
"""Smoke test for graph_v6 (encoder in isolation — no Llama).

Checks: streaming write, finalize (fact-tokens), per-token inject (Stage B),
gradient flow through every load-bearing module (no dead paths), the FiLM no-op
mechanism (state changes facts once gamma/beta != 0), node anti-collapse, and the
bottleneck-width measurement. Run: python scripts/repr_learning/smoke_graph_v6.py
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphV6BaselineEncoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
results = []
def check(name, cond, extra=""):
    ok = bool(cond)
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    return ok

cfg = ReprConfig(
    d_llama=64,
    graph_v6_K_node=8, graph_v6_K_edge=12, graph_v6_d_node=16, graph_v6_d_state=16,
    graph_v6_d_updater=32, graph_v6_updater_layers=2, graph_v6_updater_heads=4,
    graph_v6_d_read=16, graph_v6_read_heads=2, graph_v6_read_ffn_mult=2,
    graph_v6_film_hidden=16, graph_v6_builder_mlp_hidden=16,
)
enc = GraphV6BaselineEncoder(cfg).to(dev)
B, T_w = 2, 10

print("== write + finalize ==")
state = enc.init_streaming_state(B, dev, torch.float32)
check("init returns N/q_src/q_dst/state", all(k in state for k in ["N", "q_src", "q_dst", "state"]))
check("N shape [B,K_node,d_node]", tuple(state["N"].shape) == (B, 8, 16))
for w in range(2):
    emb = torch.randn(B, T_w, cfg.d_llama, device=dev)
    mask = torch.ones(B, T_w, dtype=torch.bool, device=dev)
    state, _ = enc.streaming_write(state, emb, mask, chunk_offset=w * T_w)
check("n_windows == 2", state["n_windows"] == 2)
check("state finite after writes", torch.isfinite(state["N"]).all() and torch.isfinite(state["state"]).all())
mem, faux = enc.finalize_memory(state)
check("finalize memory M==0 (prepend skipped)", mem.shape[1] == 0)
facts = faux["graph_v6_facts"]
check("facts dict has 'value' only", set(facts.keys()) == {"value"})
check("fact value [B,K_edge,d_read]", tuple(facts["value"].shape) == (B, 12, 16))

print("== per-token inject (READ Stage B) ==")
hidden = torch.randn(B, 7, cfg.d_llama, device=dev, requires_grad=True)
out = enc.inject(hidden, facts)
check("inject preserves [B,T,d_llama]", tuple(out.shape) == (B, 7, cfg.d_llama))
check("inject is a non-trivial residual", not torch.allclose(out, hidden))

print("== backward / gradient flow (no dead paths) ==")
loss = out.float().pow(2).mean()
loss.backward()
def gn(p):
    return None if p.grad is None else p.grad.float().norm().item()
grad_targets = {
    "updater.node_head": enc.updater.node_head[-1].weight,
    "updater.src_head": enc.updater.src_head[-1].weight,
    "updater.state_head": enc.updater.state_head[-1].weight,
    "updater.edge_id (instance tag)": enc.updater.edge_id,
    "fact_builder.film (STATE path, last layer)": enc.fact_builder.film[-1].weight,
    "fact_builder.W_src": enc.fact_builder.W_src.weight,
    "fact_builder.W_dst": enc.fact_builder.W_dst.weight,
    "fact_builder.mlp (post-FiLM)": enc.fact_builder.mlp[-1].weight,
    "fact_reader.q_proj": enc.fact_reader.q_proj.weight,
    "fact_reader.attn": next(enc.fact_reader.attn.parameters()),
    "fact_reader.W_out": enc.fact_reader.W_out.weight,
    "fact_reader.scale_raw": enc.fact_reader.scale_raw,
    "read_pointer (W_k/tau)": next(enc.read_pointer.parameters()),
    "mu_node": enc.mu_node, "mu_state": enc.mu_state, "mu_q": enc.mu_q,
}
for name, p in grad_targets.items():
    g = gn(p)
    check(f"grad flows: {name}", g is not None and g > 0, f"|g|={g:.2e}" if g else "None")
check("hidden grad (read differentiable)", hidden.grad is not None and hidden.grad.norm() > 0)

print("== no-op-free: FiLM makes `state` load-bearing (when gamma/beta != 0) ==")
# At init FiLM output is zero-init (transparent: fact==h); set it nonzero to verify
# the mechanism wires state into the fact. (The real probe runs on the TRAINED model.)
with torch.no_grad():
    enc.fact_builder.film[-1].weight.normal_(0, 0.5)
    enc.fact_builder.film[-1].bias.normal_(0, 0.5)
fv = enc._build_facts(state, zero_state=False)
fz = enc._build_facts(state, zero_state=True)
check("zeroing state changes facts", not torch.allclose(fv, fz, atol=1e-4),
      f"||Δ||={ (fv - fz).norm().item():.3e}")

print("== anti-collapse probe (node cross-slot cosine) ==")
Nf = F.normalize(state["N"][0].float(), dim=-1)
cos = Nf @ Nf.t()
K = cos.shape[0]
off = cos[~torch.eye(K, dtype=torch.bool, device=cos.device)]
check("nodes not collapsed (cos mean < 0.9)", off.mean().item() < 0.9,
      f"mean={off.mean():.3f} max={off.max():.3f}")

print("== bottleneck width (current convention) ==")
prod = ReprConfig()
gv6 = prod.bottleneck_floats_graph_v6
base_op = 128 * 1398  # operative baseline (trainer override n_flat_codes=128, d_inner=1398)
print(f"  graph_v6 substrate (production dims K_node={prod.graph_v6_K_node}, K_edge={prod.graph_v6_K_edge}, "
      f"d_node={prod.graph_v6_d_node}, d_state={prod.graph_v6_d_state}):")
print(f"    = {prod.graph_v6_K_node}·{prod.graph_v6_d_node} + {prod.graph_v6_K_edge}·"
      f"(2·{prod.graph_v6_d_node}+{prod.graph_v6_d_state}) = {gv6} floats")
print(f"  baseline operative budget = 128·1398 = {base_op} floats  "
      f"(graph_v6 = {100*gv6/base_op:.1f}% of baseline)")

nfail = sum(1 for _, ok in results if not ok)
print(f"\n{'ALL PASS' if nfail == 0 else f'{nfail}/{len(results)} FAILED'}")
sys.exit(1 if nfail else 0)
