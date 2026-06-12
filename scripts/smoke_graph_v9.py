#!/usr/bin/env python3
"""Correctness smoke for graph_substrate_v9 + GraphV9FlowReader (no training).

Checks, in order of load-bearing-ness:
  (1) ALGEBRA AT INIT (cups-and-ball): a chain of exact transposition factors
      (strength=2, dir=(e_i-e_j)/sqrt(2)) tracks a streamed permutation EXACTLY —
      the falsifiable "operator chain works from step 0" claim (S5 word problem).
  (2) WY fast apply == sequential reference apply (incl. the adversarial
      strength=2 case).
  (3) chunkwise forward == per-token reference forward (same chunk-frozen
      semantics), arms A/B/C, absorb on and off.
  (4) ABSORPTION CONSERVES strength exactly (debit == landed, per batch row).
  (5) Identity at init: zero-strength state => apply is exactly the input;
      reader injection is exactly zero (o_proj zero-init), both arms.
  (6) Pad invariance: padded rows are exact no-ops (state matches trimmed run).
  (7) Grad flow into every write-path param group, arm C (zero-init final MLP
      layers wake first — the §7c starvation telemetry contract).
  (8) pack_state/unpack_layer roundtrip.
Small dims for speed; CPU.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repr_learning.graph_substrate_v9 import (  # noqa: E402
    GraphV9Config, GraphV9Substrate)

torch.manual_seed(0)
results = {}


def check(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def maxrel(a, b):
    return ((a - b).abs() / (b.abs() + 1e-6)).max().item()


# ── (1) algebra at init: streamed transpositions tracked exactly ───────────────
print("=== (1) cups-and-ball: permutation tracking at init ===")
d_code = 32
cfg1 = GraphV9Config(d_model=64, d_code=d_code, d_key=32, nodes=(64,), slots=(1,),
                     chunk=8, arm="A", absorb_enabled=False)
sub1 = GraphV9Substrate(cfg1)
n_swaps = 24                              # 24 streamed swaps over 5 "cups" (S5 words)
perm = list(range(5))
factor_dirs = torch.zeros(1, n_swaps, 1, d_code)
factor_strengths = torch.full((1, n_swaps, 1), 2.0)   # strength 2: exact reflections
swaps = []
gen = torch.Generator().manual_seed(1)
for k in range(n_swaps):
    i, j = torch.randperm(5, generator=gen)[:2].tolist()
    direction = torch.zeros(d_code)
    direction[i], direction[j] = 1.0, -1.0
    factor_dirs[0, k, 0] = direction / direction.norm()   # H = I - 2vv^T == swap i,j
    swaps.append((i, j))
for (i, j) in swaps:                      # ground-truth composed permutation
    perm[i], perm[j] = perm[j], perm[i]
ball = 3
codes = torch.zeros(1, 1, d_code)
codes[0, 0, ball] = 1.0
routing_scores = torch.ones(1, 1, n_swaps)
out_fast = sub1.apply_chain(codes, factor_dirs, factor_strengths, routing_scores)
out_ref = sub1.apply_chain_reference(codes, factor_dirs, factor_strengths, routing_scores)
expect = perm.index(ball)                 # where the ball ended up
ok = out_fast[0, 0, expect].item() > 0.999 and out_fast[0, 0].abs().sum().item() < 1.001
check("permutation tracked exactly at init", ok,
      f"ball {ball} -> coord {expect}, mass {out_fast[0,0,expect]:.6f}")

# ── (2) WY fast apply == sequential reference ──────────────────────────────────
print("=== (2) WY blocked apply == sequential reference ===")
swap_abs_err = (out_fast - out_ref).abs().max().item()
check("swap-case (strength=2) fast==ref", swap_abs_err < 1e-5,
      f"max abs {swap_abs_err:.2e}")    # abs, not rel: off-coords are ~1e-7 on both sides
batch_size, chunk_len = 2, 12
cfg2 = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(48, 16), slots=(2, 4),
                     chunk=8, arm="A", wy_block=16)
sub2 = GraphV9Substrate(cfg2)
codes2 = torch.randn(batch_size, chunk_len, 32)
dirs2 = torch.randn(batch_size, 48, 2, 32)
strengths2 = 2.0 * torch.rand(batch_size, 48, 2)
scores2 = torch.softmax(torch.randn(batch_size, chunk_len, 48), -1)
check("random-case fast==ref",
      maxrel(sub2.apply_chain(codes2, dirs2, strengths2, scores2),
             sub2.apply_chain_reference(codes2, dirs2, strengths2, scores2)) < 1e-4)

# ── (3) chunkwise == reference forward ─────────────────────────────────────────
print("=== (3) chunkwise forward == per-token reference ===")
for arm in ("A", "B", "C"):
    for absorb in (False, True):
        if arm == "C" and not absorb:
            continue                       # arm C requires absorb (it IS the write)
        cfg = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(48, 16),
                            slots=(1, 4), chunk=8, arm=arm, absorb_enabled=absorb)
        sub = GraphV9Substrate(cfg).eval()
        seq_len = 20
        hiddens = torch.randn(batch_size, seq_len, 64)
        surprise = torch.rand(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len); mask[0, 15:] = 0.0
        with torch.no_grad():
            state_a = sub.forward(hiddens, surprise, mask, reference=False)
            state_b = sub.forward(hiddens, surprise, mask, reference=True)
        worst = 0.0
        for l in range(sub.depth):
            worst = max(worst,
                        maxrel(state_a["factor_dirs"][l], state_b["factor_dirs"][l]),
                        maxrel(state_a["factor_strengths"][l], state_b["factor_strengths"][l]),
                        maxrel(state_a["coact"][l], state_b["coact"][l]),
                        maxrel(state_a["trace"][l], state_b["trace"][l]))
        check(f"arm {arm}, absorb={'on' if absorb else 'off'}", worst < 1e-3,
              f"max rel {worst:.2e}")

# ── (4) absorption conserves strength exactly ──────────────────────────────────
print("=== (4) absorption conservation (debit == landed) ===")
cfg4 = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(32,), slots=(4,),
                     chunk=8, arm="A", absorb_enabled=True)
sub4 = GraphV9Substrate(cfg4)
with torch.no_grad():
    sub4.absorb_strength_logit.fill_(2.0)   # open the gate hard: non-trivial test
    dirs4 = torch.randn(batch_size, 32, 4, 32)
    strengths4 = 2.0 * torch.rand(batch_size, 32, 4)
    coact4 = torch.rand(batch_size, 32, 32)
    surprise4 = torch.rand(batch_size)
    dirs_out, strengths_out, flux = sub4._absorb(0, dirs4, strengths4, coact4, surprise4)
budget_err = (strengths_out.sum(dim=(1, 2)) - strengths4.sum(dim=(1, 2))).abs().max().item()
check("total strength conserved per row", budget_err < 1e-4,
      f"|delta| {budget_err:.2e}, transfer flux {flux.mean():.4f}")
check("strengths stay in [0,2]",
      (strengths_out.min().item() >= -1e-6) and (strengths_out.max().item() <= 2.0 + 1e-5),
      f"range [{strengths_out.min():.4f}, {strengths_out.max():.4f}]")

# arm C strict invariant: a FULL forward must leave each writable layer's total
# strength exactly at its base total (no deposits anywhere; absorption conserves)
cfg4c = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(32, 16), slots=(1, 4),
                      chunk=8, arm="C", absorb_enabled=True)
sub4c = GraphV9Substrate(cfg4c).eval()
with torch.no_grad():
    sub4c.absorb_strength_logit.fill_(2.0)
    state4c = sub4c.forward(torch.randn(batch_size, 20, 64), torch.rand(batch_size, 20))
    base_total = (2.0 * torch.sigmoid(sub4c.base_strength_logit[0])).sum()
    end_total = state4c["factor_strengths"][1].sum(dim=(1, 2))
invariant_err = (end_total - base_total).abs().max().item()
check("arm C full-forward strength invariant (no injection anywhere)",
      invariant_err < 1e-3, f"|delta| {invariant_err:.2e} on base total {base_total:.2f}")

# ── (5) identity at init ───────────────────────────────────────────────────────
print("=== (5) identity at init ===")
cfg5 = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(48,), slots=(2,),
                     chunk=8, arm="A")
sub5 = GraphV9Substrate(cfg5)
state5 = sub5.init_state(batch_size, "cpu")
codes5 = torch.randn(batch_size, chunk_len, 32)
scores5 = torch.softmax(torch.randn(batch_size, chunk_len, 48), -1)
out5 = sub5.apply_chain(codes5, state5["factor_dirs"][0],
                        state5["factor_strengths"][0], scores5)
check("empty state apply == input exactly", (out5 - codes5).abs().max().item() == 0.0)

from src.repr_learning.graph_read import GraphV9FlowReader  # noqa: E402
for arm in ("A", "B", "C"):
    cfg_r = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(48, 16),
                          slots=(1, 4), chunk=8, arm=arm)
    sub_r = GraphV9Substrate(cfg_r)
    reader = GraphV9FlowReader(sub_r, (3, 7), inner_dim=32)
    with torch.no_grad():
        state_r = sub_r.forward(torch.randn(batch_size, 20, 64), torch.rand(batch_size, 20))
        reader.set_memory(sub_r.pack_state(state_r))
        injection = reader.read(1, torch.randn(batch_size, 7, 64))
    check(f"reader injection exactly zero at init (arm {arm}, o_proj zero)",
          injection.abs().max().item() == 0.0)
    reader.clear_memory()
    check(f"reader OFF is a true no-op (arm {arm})",
          reader.read(1, torch.randn(batch_size, 7, 64)).abs().max().item() == 0.0)

# ── (6) pad invariance ─────────────────────────────────────────────────────────
print("=== (6) pad invariance (pads neither write, route, nor decay) ===")
cfg6 = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(48, 16), slots=(1, 4),
                     chunk=8, arm="B", absorb_enabled=True)
sub6 = GraphV9Substrate(cfg6).eval()
seq_len = 20
hiddens6 = torch.randn(2, seq_len, 64); surprise6 = torch.rand(2, seq_len)
mask6 = torch.ones(2, seq_len); mask6[0, 13:] = 0.0
with torch.no_grad():
    state_pad = sub6.forward(hiddens6, surprise6, mask6)
    state_trim = sub6.forward(hiddens6[0:1, :13], surprise6[0:1, :13])
worst = 0.0
for l in range(sub6.depth):
    worst = max(worst,
                maxrel(state_pad["factor_strengths"][l][0:1], state_trim["factor_strengths"][l]),
                maxrel(state_pad["factor_dirs"][l][0:1], state_trim["factor_dirs"][l]),
                maxrel(state_pad["coact"][l][0:1], state_trim["coact"][l]))
check("padded row state == trimmed run", worst < 1e-3, f"max rel {worst:.2e}")

# ── (7) grad flow into the write path (arm B — the primary design) ─────────────
print("=== (7) grad flow, arm C (write-path starvation telemetry contract) ===")
cfg7 = GraphV9Config(d_model=64, d_code=32, d_key=32, nodes=(48, 16), slots=(1, 4),
                     chunk=8, arm="C", absorb_enabled=True)
sub7 = GraphV9Substrate(cfg7).train()
hiddens7 = torch.randn(2, 20, 64); surprise7 = torch.rand(2, 20)
state7 = sub7.forward(hiddens7, surprise7)
loss = sum(state7["factor_strengths"][l].pow(2).mean()
           + state7["factor_dirs"][l].pow(2).mean() for l in range(sub7.depth))
loss.backward()
groups = {
    "seed_proj": sub7.seed_proj.weight,
    "atom_dirs": sub7.atom_dirs,
    "atom_strength": sub7.atom_strength_logit,
    "base_dirs_L1": sub7.base_dirs[0],
    "base_strength_L1": sub7.base_strength_logit[0],
    "node_keys_L0": sub7.node_keys[0], "node_keys_L1": sub7.node_keys[1],
    "route_proj_L0": sub7.route_projs[0].weight,
    "route_proj_L1": sub7.route_projs[1].weight,
    "plasticity_out_L1": sub7.plasticity_mlps["1"].fc_out.weight,
    "absorb_strength": sub7.absorb_strength_logit,
    "route_temp": sub7.log_route_temp,
    "match_weight": sub7.match_weight,
}
all_ok = True
for name, param in groups.items():
    norm = param.grad.norm().item() if param.grad is not None else 0.0
    ok = norm > 0
    all_ok &= ok
    print(f"    {name:<18} grad {norm:.3e}" + ("" if ok else "  <-- NO GRAD"))
check("all write-path groups receive gradient", all_ok)

# ── (8) pack/unpack roundtrip ──────────────────────────────────────────────────
print("=== (8) pack_state / unpack_layer roundtrip ===")
packed = sub7.pack_state(state7)
ok = True
for l in range(sub7.depth):
    dirs_rt, strengths_rt = sub7.unpack_layer(packed, l)
    ok &= (dirs_rt - state7["factor_dirs"][l]).abs().max().item() == 0.0
    ok &= (strengths_rt - state7["factor_strengths"][l]).abs().max().item() == 0.0
check("roundtrip exact", ok)

failed = [k for k, v in results.items() if not v]
print("\n" + ("ALL SMOKE CHECKS PASS" if not failed else "SMOKE FAILED: " + ", ".join(failed)))
sys.exit(0 if not failed else 1)
