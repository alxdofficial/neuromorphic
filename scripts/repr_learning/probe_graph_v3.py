#!/usr/bin/env python3
"""[v3-ONLY — does not run on v4 architecture]
Graph baseline v3 structure probe.

Replaces the v1/v2 probes that tested snap_gate / saliency_logit / soft_snap
(all removed). The v3 probes target the new mechanics:

  P1 — Self-pick rate. Routing must pick NON-self proposals every window.
       At init or trained, self_pick_fraction should be 0.0.

  P2 — Update-strength margin. update_alpha should be small (~0) when
       pick_strength is below the random-nearest cosine margin (≈0.28 for
       K=68, d=128). High alpha on noise = broken margin.

  P3 — Saliency gate response. Memory token norms must differ by ≥10× between
       u=0 slots and u=1 slots. If close to equal, the gate is ineffective.

  P4 — Saliency differentiation over windows. With random data, u should
       still drift apart across slots after many windows (different slots
       see different competition strengths). max(u) - min(u) > 0.01 after
       50 windows of random data is a healthy floor.

  P5 — Overwrite activation. On REAL text (composite_v1 streamed), the
       overwrite_count should be > 0 at least once across 50 windows. Zero
       overwrites over many windows ⇒ admission gate too strict OR u
       never decays enough.

  P6 — Endpoint clustering. Mean off-diagonal cosine across all
       (src, dst) endpoints. v1/v2 was stuck at ≈0.0 (no reuse); v3 should
       climb above 0.1 over training as endpoints cluster around frequent
       entities. (Probe-only; needs trained checkpoint for the steady-state
       value.)

Usage:
    python -m scripts.repr_learning.probe_graph_v3 [--ckpt path/to/best.pt]

Without --ckpt: runs probes on a fresh-init model (sanity checks P1, P2,
P3; P4 and P6 on random data).

With --ckpt: loads weights and runs all probes.
"""
from __future__ import annotations

import argparse
import torch

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphBaselineEncoder, GraphReadout
# v3-only — `expert_choice_routing` was removed in v4. Lazy import inside
# probe_self_pick so this module is at least importable under v4.
# from src.repr_learning.graph_substrate import expert_choice_routing


def _streaming_step(enc, state, te, am, w, T):
    return enc.streaming_write(state, te, am, chunk_offset=w * T)


def probe_self_pick(enc, n_windows=5, seed=0):
    """P1: routing must NOT pick self proposals."""
    torch.manual_seed(seed)
    B, T = 2, 256
    state = enc.init_streaming_state(B, torch.device("cpu"), torch.float32)
    # v3-only API — lazy import; raises a clear error against v4.
    try:
        from src.repr_learning.graph_substrate import expert_choice_routing
    except ImportError:
        raise RuntimeError(
            "This v3-only probe uses `expert_choice_routing`, which was "
            "removed in v4. Checkout tag `graph_baseline_v3_lb_locked` to run."
        ) from None
    te = torch.randn(B, T, enc.cfg.d_llama)
    K = enc.K_max
    self_pick_fractions, update_alphas = [], []
    for w in range(n_windows):
        edges_old = state["edges"]
        pins = enc.pin_encoder(te.to(torch.float32))
        proposals = enc.updater(pins, edges_old, pins_pad_mask=None)
        endpoints = torch.cat([edges_old["src"], edges_old["dst"]], dim=1)
        proposed_endpoints = torch.cat([proposals["src"], proposals["dst"]], dim=1)
        picked_idx, alpha, _novelty, _pc, _lb = expert_choice_routing(
            endpoints, proposed_endpoints, strength_scale=enc.update_strength_scale,
        )
        self_pick = (picked_idx == torch.arange(2 * K).unsqueeze(0)).float().mean()
        self_pick_fractions.append(self_pick.item())
        update_alphas.append(alpha.mean().item())
        state, _ = _streaming_step(enc, state, te, torch.ones(B, T, dtype=torch.bool), w, T)
    return {
        "self_pick_mean": sum(self_pick_fractions) / len(self_pick_fractions),
        "self_pick_max": max(self_pick_fractions),
        "update_alpha_mean": sum(update_alphas) / len(update_alphas),
    }


def probe_gate_response():
    """P3: readout gate must differ ≥10× between u=0 and u=1."""
    readout = GraphReadout(d_node=128, d_state=128, d_llama=2048,
                            d_hidden=512, n_heads=4)
    src = torch.randn(2, 68, 128)
    dst = torch.randn(2, 68, 128)
    state = torch.randn(2, 68, 128)
    with torch.no_grad():
        norm_zero = readout(src, dst, state, torch.zeros(2, 68)).norm(dim=-1).mean().item()
        norm_one = readout(src, dst, state, torch.ones(2, 68)).norm(dim=-1).mean().item()
    return {
        "memory_norm_u0": norm_zero,
        "memory_norm_u1": norm_one,
        "ratio_u1_to_u0": norm_one / max(norm_zero, 1e-6),
    }


def probe_saliency_drift(enc, n_windows=50, seed=0):
    """P4: u should differentiate across slots over many windows."""
    torch.manual_seed(seed)
    B, T = 2, 256
    state = enc.init_streaming_state(B, torch.device("cpu"), torch.float32)
    te = torch.randn(B, T, enc.cfg.d_llama)
    am = torch.ones(B, T, dtype=torch.bool)
    for w in range(n_windows):
        state, _ = _streaming_step(enc, state, te, am, w, T)
    u = state["edges"]["u"]
    return {
        "u_min": u.min().item(),
        "u_max": u.max().item(),
        "u_mean": u.mean().item(),
        "u_spread": (u.max() - u.min()).item(),
        "u_std": u.std().item(),
    }


def probe_endpoint_clustering(enc, n_windows=50, seed=0):
    """P6: mean off-diagonal endpoint cosine — clustering signature."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    B, T = 2, 256
    state = enc.init_streaming_state(B, torch.device("cpu"), torch.float32)
    te = torch.randn(B, T, enc.cfg.d_llama)
    am = torch.ones(B, T, dtype=torch.bool)
    for w in range(n_windows):
        state, _ = _streaming_step(enc, state, te, am, w, T)
    bank = torch.cat([state["edges"]["src"], state["edges"]["dst"]], dim=1)
    bn = F.normalize(bank, dim=-1, eps=1e-6)
    cos = bn @ bn.transpose(-1, -2)
    K2 = bank.shape[1]
    off_diag = ~torch.eye(K2, dtype=torch.bool)
    return {"endpoint_reuse_mean_cos": cos[:, off_diag].mean().item()}


def probe_overwrite_activation(enc, n_windows=50, seed=0):
    """P5: at least some overwrites should happen over many windows."""
    torch.manual_seed(seed)
    B, T = 2, 256
    state = enc.init_streaming_state(B, torch.device("cpu"), torch.float32)
    te = torch.randn(B, T, enc.cfg.d_llama)
    am = torch.ones(B, T, dtype=torch.bool)
    total_overwrites = 0.0
    for w in range(n_windows):
        state, diag = _streaming_step(enc, state, te, am, w, T)
        total_overwrites += float(diag.get("graph_overwrite_count", 0.0))
    return {
        "overwrite_count_total": total_overwrites,
        "overwrite_rate_per_slot_per_window": total_overwrites / (enc.K_max * n_windows * B),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="optional ckpt path; if absent, fresh-init")
    args = ap.parse_args()

    cfg = ReprConfig()
    enc = GraphBaselineEncoder(cfg)

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        enc_sd = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                  if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        print(f"[ckpt loaded] missing={len(missing)} unexpected={len(unexpected)}")

    enc.training = False

    print("\n=== P1 — Self-pick rate (must be 0.0) ===")
    r1 = probe_self_pick(enc)
    print(f"  self_pick_mean: {r1['self_pick_mean']:.4f}  (PASS if 0.0)")
    print(f"  self_pick_max:  {r1['self_pick_max']:.4f}")
    print(f"  update_alpha_mean: {r1['update_alpha_mean']:.4f}  (sane: 0.1-0.8 on random data)")

    print("\n=== P3 — Gate response (ratio u=1/u=0 must be ≥ 10) ===")
    r3 = probe_gate_response()
    print(f"  memory_norm at u=0: {r3['memory_norm_u0']:.4f}")
    print(f"  memory_norm at u=1: {r3['memory_norm_u1']:.4f}")
    print(f"  ratio: {r3['ratio_u1_to_u0']:.2f}x (PASS if >= 10x)")

    print("\n=== P4 — Saliency drift over 50 windows ===")
    r4 = probe_saliency_drift(enc, n_windows=50)
    print(f"  u_min={r4['u_min']:.4f}  u_max={r4['u_max']:.4f}  u_mean={r4['u_mean']:.4f}")
    print(f"  u_spread (max-min): {r4['u_spread']:.4f}  (PASS if > 0.01)")

    print("\n=== P5 — Overwrite activation over 50 windows ===")
    r5 = probe_overwrite_activation(enc, n_windows=50)
    print(f"  total overwrites: {r5['overwrite_count_total']:.0f}")
    print(f"  rate per (slot x window): {r5['overwrite_rate_per_slot_per_window']:.5f}")
    print(f"  NOTE: random data may give 0. Real text + trained ckpt should fire.")

    print("\n=== P6 — Endpoint clustering ===")
    r6 = probe_endpoint_clustering(enc, n_windows=50)
    print(f"  mean off-diagonal cosine: {r6['endpoint_reuse_mean_cos']:.4f}")
    print(f"  NOTE: v1/v2 was ~0.0; v3 trained should climb > 0.1.")


if __name__ == "__main__":
    main()
