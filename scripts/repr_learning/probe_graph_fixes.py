#!/usr/bin/env python3
"""Audit-probe suite for graph_baseline fixes.

Runs four deterministic probes the external audit recommended:
  P1: same input twice -> memory must match (was |dmem| ~ 0.57 before C1 fix)
  P2: snap_gate=1 + close-to-existing proposal must produce that endpoint
      with cosine ~ 1.0 (was ~1.1% attention max before C2 fix)
  P3: L_connectivity must not reward self-snap (was 99.3% self-argmax for
      src, 97.4% for dst before C3 fix)
  P4 (keep=1 saliency): forcing keep_gate=1 must produce zero saliency
      drift (was |dsaliency|=3.39 before M2 fix)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.repr_learning.config import ReprConfig
from src.repr_learning.graph_substrate import init_graph_state, soft_snap


def p1_determinism(device):
    """Same input twice -> identical memory output."""
    print("\n[P1] determinism probe", flush=True)
    from src.repr_learning.encoder import GraphBaselineEncoder
    cfg = ReprConfig()
    enc = GraphBaselineEncoder(cfg).to(device)
    enc.train(False)
    B, T = 2, 256
    token_embeds = torch.randn(B, T, cfg.d_llama, device=device, dtype=torch.float32)
    attn = torch.ones(B, T, dtype=torch.bool, device=device)

    with torch.no_grad():
        s1 = enc.init_streaming_state(B, device, torch.float32)
        s1, _ = enc.streaming_write(s1, token_embeds, attn)
        mem1, _ = enc.finalize_memory(s1)

        s2 = enc.init_streaming_state(B, device, torch.float32)
        s2, _ = enc.streaming_write(s2, token_embeds, attn)
        mem2, _ = enc.finalize_memory(s2)

    delta = (mem1 - mem2).abs()
    l2 = delta.pow(2).sum().sqrt().item()
    mad = delta.mean().item()
    max_diff = delta.max().item()
    print(f"  L2 |dmem| = {l2:.6f}", flush=True)
    print(f"  mean abs dmem = {mad:.6e}", flush=True)
    print(f"  max abs dmem  = {max_diff:.6e}", flush=True)
    ok = l2 < 1e-3
    print(f"  -> {'PASS' if ok else 'FAIL'} (expect L2 < 1e-3 in inference mode)", flush=True)
    return ok


def p2_snap_actually_snaps(device):
    """snap_gate=1 with a proposal very close to an existing endpoint must
    output that endpoint (cosine ~ 1)."""
    print("\n[P2] soft_snap actually snaps", flush=True)
    B, K, d = 1, 16, 64
    torch.manual_seed(0)
    bank = torch.randn(B, 2 * K, d, device=device)
    target = bank[:, 7:8, :]                                     # [B, 1, d]
    noise = 0.01 * torch.randn(B, 1, d, device=device)
    proposal_one = (target + noise)                              # close to bank[7]
    proposal = proposal_one.expand(B, K, d).contiguous()

    snap_gate = torch.ones(B, K, device=device)
    out, max_sim = soft_snap(
        proposal, bank, snap_gate, temperature=0.1, top_k=4, exclude_self=False,
    )
    out0 = out[:, 0, :]                                          # [B, d]
    target0 = target[:, 0, :]                                    # [B, d]
    cos_to_target = F.cosine_similarity(out0, target0, dim=-1).item()
    cos_to_proposal = F.cosine_similarity(out0, proposal[:, 0], dim=-1).item()
    print(f"  cos(snap_out, target_endpoint) = {cos_to_target:.4f}", flush=True)
    print(f"  cos(snap_out, raw_proposal)    = {cos_to_proposal:.4f}", flush=True)
    print(f"  max_sim (slot 0) = {max_sim[0, 0].item():.4f}", flush=True)
    ok = cos_to_target > 0.99
    print(f"  -> {'PASS' if ok else 'FAIL'} (expect cos>0.99 with snap_gate=1)", flush=True)
    return ok


def p3_no_self_snap(device):
    """L_connectivity's max_sim must reflect nearest *other* endpoint, not self."""
    print("\n[P3] L_connect excludes self", flush=True)
    B, K, d = 1, 32, 64
    torch.manual_seed(0)
    # Build a bank arranged as [src(0..K-1), dst(0..K-1)]; slot i's own
    # src is bank[i], own dst is bank[K+i]. Set the proposal for slot i
    # to exactly = bank[i] (self-src). The OLD soft_snap would report
    # max_sim = 1.0 (matching self). The NEW one masks self and should
    # report a much lower max_sim.
    bank = torch.randn(B, 2 * K, d, device=device)
    proposal = bank[:, :K, :].clone()                            # slot i = self-src

    snap_gate = torch.zeros(B, K, device=device)                 # no actual snap
    _, max_sim = soft_snap(
        proposal, bank, snap_gate, temperature=0.1, top_k=4, exclude_self=True,
    )
    self_match_fraction = (max_sim >= 0.999).float().mean().item()
    mean_max = max_sim.mean().item()
    print(f"  mean(max_sim) = {mean_max:.4f}  (should be << 1.0)", flush=True)
    print(f"  fraction at >= 0.999 = {self_match_fraction:.2%}  (should be 0% - self excluded)", flush=True)
    ok = self_match_fraction < 0.05
    print(f"  -> {'PASS' if ok else 'FAIL'} (self-match should be near 0)", flush=True)
    return ok


def p4_keep_gate_keeps_saliency(device):
    """With keep_gate forced to 1, saliency_logit must not drift."""
    print("\n[P4] keep_gate=1 freezes saliency", flush=True)
    from src.repr_learning.encoder import GraphBaselineEncoder
    cfg = ReprConfig()
    enc = GraphBaselineEncoder(cfg).to(device)
    enc.train(False)
    B, T = 2, 256
    token_embeds = torch.randn(B, T, cfg.d_llama, device=device, dtype=torch.float32)
    attn = torch.ones(B, T, dtype=torch.bool, device=device)

    # Patch the updater to force keep_gate=1 (override the sigmoid output)
    original_forward = enc.updater.forward

    def patched_forward(pins, edges_old, pins_pad_mask=None):
        proposed, gates = original_forward(pins, edges_old, pins_pad_mask=pins_pad_mask)
        gates["keep_gate"] = torch.ones_like(gates["keep_gate"])
        return proposed, gates

    enc.updater.forward = patched_forward
    try:
        with torch.no_grad():
            state = enc.init_streaming_state(B, device, torch.float32)
            sal_before = state["edges"]["saliency_logit"].clone()
            state, _ = enc.streaming_write(state, token_embeds, attn)
            sal_after = state["edges"]["saliency_logit"]

        sal_delta = (sal_after - sal_before).abs()
        max_d = sal_delta.max().item()
        mean_d = sal_delta.mean().item()
        print(f"  max |dsaliency| = {max_d:.6f}", flush=True)
        print(f"  mean |dsaliency| = {mean_d:.6f}", flush=True)
        ok = max_d < 1e-3
        print(f"  -> {'PASS' if ok else 'FAIL'} (expect ~ 0 with keep=1)", flush=True)
        return ok
    finally:
        enc.updater.forward = original_forward


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    results = {
        "P1_determinism":         p1_determinism(device),
        "P2_snap_actually_snaps": p2_snap_actually_snaps(device),
        "P3_no_self_snap":        p3_no_self_snap(device),
        "P4_keep_keeps_saliency": p4_keep_gate_keeps_saliency(device),
    }
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("-" * 50)
    for k, v in results.items():
        print(f"  {k:<28}  {'PASS' if v else 'FAIL'}")
    print("=" * 50)
    if all(results.values()):
        print("OVERALL: PASS - all 4 audit probes green.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL - at least one probe regressed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
