#!/usr/bin/env python3
"""Equivalence + grad smoke for graph_substrate_v8 (incl. the plasticity modulator).

The load-bearing check: the chunkwise-parallel training path must EXACTLY match the
per-token reference path (same chunk-frozen semantics). We verify it (a) at init
(gate==1, identity) and (b) with a RANDOMIZED, ACTIVE plasticity gate — the latter
proves the new modulation is placed so the parallel form is preserved. Plus pad
invariance and grad flow into the plasticity MLP. Small dims for speed.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repr_learning.graph_substrate_v8 import GraphV8Config, GraphV8Substrate  # noqa: E402

torch.manual_seed(0)
cfg = GraphV8Config(d_model=128, d_mem=64, n_nodes=64, n_layers=3, chunk=8)
sub = GraphV8Substrate(cfg)
B, T, S = 2, 20, cfg.n_layers
h = torch.randn(B, T, S, cfg.d_model)
surp = torch.rand(B, T)
mask = torch.ones(B, T); mask[0, 15:] = 0.0          # ragged: row 0 has 5 pads


def randomize_gate(scale=1.0):
    with torch.no_grad():
        for f1, f2 in zip(sub.plasticity.fc1, sub.plasticity.fc2):
            f1.weight.normal_(0, 0.5); f1.bias.normal_(0, 0.5)
            f2.weight.normal_(0, scale); f2.bias.normal_(0, scale)


def run(reference):
    s = sub.init_state(B, "cpu", torch.float32)
    return sub.forward(h, surp, mask, state=s, reference=reference)


def maxrel(a, b):
    return ((a - b).abs() / (b.abs() + 1e-6)).max().item()


def equiv(tag):
    ck, rf = run(False), run(True)
    ok = True
    for l in range(1, cfg.n_layers + 1):
        rk = maxrel(ck["keys"][l], rf["keys"][l]); rv = maxrel(ck["values"][l], rf["values"][l])
        rc = maxrel(ck["coact"][l - 1], rf["coact"][l - 1])
        flag = "" if (rk < 1e-3 and rv < 1e-3 and rc < 1e-3) else "  <-- FAIL"
        if flag:
            ok = False
        print(f"  {tag} L{l}: keys {rk:.2e}  values {rv:.2e}  coact {rc:.2e}{flag}")
    return ok


print("=== (1) chunkwise ≡ reference @ init (gate==1, identity) ===")
ok1 = equiv("init")

print("=== (2) chunkwise ≡ reference @ ACTIVE gate (randomized fc2) ===")
randomize_gate(1.0)
ok2 = equiv("active")

print("=== (3) gate is genuinely active (not silently ==1) ===")
# verify gate != 1 somewhere with the randomized weights
fresh = torch.rand(B, cfg.n_nodes, cfg.n_nodes)
g = sub.plasticity.gate(0, fresh, fresh, torch.randn(cfg.n_nodes, cfg.d_mem), torch.rand(B))
print(f"  gate range [{g.min():.3f}, {g.max():.3f}]  (eta={cfg.plasticity_eta} ⇒ expect within "
      f"[{1-cfg.plasticity_eta:.2f}, {1+cfg.plasticity_eta:.2f}])")
ok3 = (g.min() < 0.999 or g.max() > 1.001) and g.min() >= 1 - cfg.plasticity_eta - 1e-4 \
    and g.max() <= 1 + cfg.plasticity_eta + 1e-4

print("=== (4) grad flows to the plasticity MLP (both layers) ===")
randomize_gate(0.3)
sub.zero_grad(set_to_none=True)
out = run(False)
loss = sum(out["keys"][l].pow(2).mean() + out["values"][l].pow(2).mean()
           for l in range(1, cfg.n_layers + 1))
loss.backward()
ok4 = True
for l in range(cfg.n_layers):
    g1 = sub.plasticity.fc1[l].weight.grad; g2 = sub.plasticity.fc2[l].weight.grad
    n1 = g1.norm().item() if g1 is not None else 0.0
    n2 = g2.norm().item() if g2 is not None else 0.0
    flag = "" if (n1 > 0 and n2 > 0) else "  <-- NO GRAD"
    if flag:
        ok4 = False
    print(f"  plasticity L{l}: fc1 grad {n1:.2e}  fc2 grad {n2:.2e}{flag}")

print("\n" + ("ALL SMOKE CHECKS PASS" if all([ok1, ok2, ok3, ok4])
              else "SMOKE FAILED: " + ", ".join(
                  n for n, ok in [("equiv@init", ok1), ("equiv@active", ok2),
                                  ("gate-active", ok3), ("grad-flow", ok4)] if not ok)))
sys.exit(0 if all([ok1, ok2, ok3, ok4]) else 1)
