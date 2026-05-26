#!/usr/bin/env python3
"""[v3-ONLY — does not run on v4 architecture]

Diagnostic plots for v3 graph_baseline (residual proposals + expert-choice
routing + recycle). Uses APIs that no longer exist in v4
(`expert_choice_routing`, `recycle_dead_slots`). To re-run on a v3 ckpt:
  git checkout graph_baseline_v3_lb_locked
v4 needs its own diagnostic suite (gate distribution over training, endpoint
reuse curve under free targets, etc).

Produces 6 figures into docs/plots/ for the meeting:
  1. Per-family val_recon bar chart
  2. State diversity across windows
  3. Eigenvalue spectrum of state matrix
  5. Endpoint similarity heatmap
  7. Training-time telemetry curves
  8. Routing-decision matrix
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphBaselineEncoder
# v3-only — `expert_choice_routing` was removed in v4. Import lazily in main()
# so this module can still be imported by code-scanners / collectors.
# from src.repr_learning.graph_substrate import expert_choice_routing

ROOT = Path("/home/alex/code/neuromorphic")
OUT = ROOT / "docs/plots"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_LB = ROOT / "outputs/repr_learning/v1h_t4k_v3_lb_graph_baseline/ckpts/graph_baseline.best.pt"
CKPT_NO = ROOT / "outputs/repr_learning/v1h_t4k_v3_graph_baseline/ckpts/graph_baseline.best.pt"
JSONL_LB = ROOT / "outputs/repr_learning/v1h_t4k_v3_lb_graph_baseline/jsonl/graph_baseline.jsonl"
JSONL_NO = ROOT / "outputs/repr_learning/v1h_t4k_v3_graph_baseline/jsonl/graph_baseline.jsonl"
JSONL_DIR = ROOT / "outputs/repr_learning"


def load_encoder(ckpt_path):
    cfg = ReprConfig()
    cfg.n_flat_codes = 36
    cfg.d_mamba = 768
    enc = GraphBaselineEncoder(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                 if k.startswith("encoder.")}
    enc.load_state_dict(enc_state, strict=False)
    enc.train(False)
    return enc, sd.get("step", "?")


def diversity(x):
    if x.dim() == 1:
        return 0.0
    x_n = F.normalize(x, dim=-1, eps=1e-8)
    sim = x_n @ x_n.T
    eye = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return float(1.0 - sim[~eye].abs().mean())


def run_chunk(enc, B=2, T_w=1024, n_windows=4, seed=7):
    torch.manual_seed(seed)
    cfg = enc.cfg
    te = torch.randn(B, T_w * n_windows, cfg.d_llama) * 0.5
    state = enc.init_streaming_state(B, te.device, te.dtype)
    per_window = [{"src": state["edges"]["src"].clone(),
                   "dst": state["edges"]["dst"].clone(),
                   "state": state["edges"]["state"].clone(),
                   "u": state["edges"]["u"].clone()}]
    for w in range(n_windows):
        te_w = te[:, w * T_w:(w + 1) * T_w, :]
        state, _ = enc.streaming_write(state, te_w, torch.ones(B, T_w, dtype=torch.bool),
                                       chunk_offset=w * T_w)
        per_window.append({"src": state["edges"]["src"].clone(),
                           "dst": state["edges"]["dst"].clone(),
                           "state": state["edges"]["state"].clone(),
                           "u": state["edges"]["u"].clone()})
    return per_window


def fig_per_family():
    print("=== Fig 1: per-family val_recon ===")
    sources = [
        ("graph + LB", "v1h_t4k_v3_lb_graph_baseline", "graph_baseline", "tab:red"),
        ("mamba", "v1h_t4k_v3_recurrent_baseline", "recurrent_baseline", "tab:green"),
        ("continuous", "v1h_t4k_v3_continuous_baseline", "continuous_baseline", "tab:blue"),
        ("memorizing", "v1h_t4k_v3_memorizing_baseline", "memorizing_baseline", "tab:purple"),
        ("graph (no LB)", "v1h_t4k_v3_graph_baseline", "graph_baseline", "tab:orange"),
    ]
    data = {}
    families = set()
    for label, dirname, vid, color in sources:
        jsonl = JSONL_DIR / dirname / "jsonl" / f"{vid}.jsonl"
        if not jsonl.exists():
            continue
        per_fam = None
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("phase") == "val" and r.get("final"):
                    per_fam = r.get("val_per_family")
                    break
        if per_fam is None:
            with jsonl.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("phase") == "val" and "val_per_family" in r:
                        per_fam = r["val_per_family"]
        if per_fam is None:
            continue
        data[label] = (color, {fam: v["mean_loss"] for fam, v in per_fam.items()})
        families.update(data[label][1].keys())

    families = sorted(families)
    n_var = len(data); n_fam = len(families)
    bar_w = 0.8 / n_var
    x = np.arange(n_fam)
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (label, (color, fam_dict)) in enumerate(data.items()):
        vals = [fam_dict.get(f, np.nan) for f in families]
        offsets = x + (i - (n_var - 1) / 2) * bar_w
        ax.bar(offsets, vals, bar_w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("val_loss_recon  (lower = better)")
    ax.set_title("Per-family val_recon  -  where graph wins/loses vs baselines")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = OUT / "v1h_t4k_v3_per_family.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def fig_state_diversity():
    print("=== Fig 2: state diversity across windows ===")
    enc_lb, step_lb = load_encoder(CKPT_LB)
    enc_no, step_no = load_encoder(CKPT_NO)
    wins_lb = run_chunk(enc_lb)
    wins_no = run_chunk(enc_no)
    div = lambda wins, key: [diversity(w[key][0]) for w in wins]
    labels = ["init"] + [f"win {i}" for i in range(len(wins_lb) - 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, name in zip(axes, ["src", "dst", "state"],
                              ["src endpoint", "dst endpoint", "edge state"]):
        ax.plot(labels, div(wins_lb, key), marker="o", linewidth=2.2,
                color="tab:red", label=f"graph + LB (step {step_lb})")
        ax.plot(labels, div(wins_no, key), marker="s", linewidth=1.6,
                color="tab:orange", linestyle="--", alpha=0.85,
                label=f"graph no-LB (step {step_no})")
        ax.axhline(0.1, color="black", linestyle=":", alpha=0.4,
                   label="success threshold (0.1)")
        ax.set_title(f"{name} diversity per window")
        ax.set_ylabel("diversity (1 - avg|cos| across slots)")
        ax.set_ylim(-0.02, 1.0)
        ax.grid(True, alpha=0.3)
        if key == "src":
            ax.legend(loc="upper right", fontsize=8)
    plt.suptitle("Slot diversity across windows  -  LB loss fixes within-chunk collapse", fontsize=13)
    plt.tight_layout()
    p = OUT / "v1h_t4k_v3_state_diversity.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def fig_eigenspectrum():
    print("=== Fig 3: eigenvalue spectrum of state ===")
    enc_lb, _ = load_encoder(CKPT_LB)
    enc_no, _ = load_encoder(CKPT_NO)
    wins_lb = run_chunk(enc_lb)
    wins_no = run_chunk(enc_no)
    final_lb = wins_lb[-1]["state"][0]
    final_no = wins_no[-1]["state"][0]
    init_state = wins_lb[0]["state"][0]
    K = final_lb.shape[0]

    def spectrum(x):
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, s, _ = torch.svd(x_centered)
        sq = (s ** 2).numpy()
        cum = np.cumsum(sq) / sq.sum()
        return sq, cum

    sq_lb, cum_lb = spectrum(final_lb)
    sq_no, cum_no = spectrum(final_no)
    sq_init, cum_init = spectrum(init_state)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.arange(1, len(sq_lb) + 1)
    ax1.semilogy(idx, sq_lb, marker="o", color="tab:red", label="graph + LB", linewidth=2)
    ax1.semilogy(idx, sq_no, marker="s", color="tab:orange", label="graph no-LB", linewidth=1.5, linestyle="--")
    ax1.semilogy(idx, sq_init, marker="^", color="tab:gray", label="init (untrained)", linewidth=1, alpha=0.7)
    ax1.set_xlabel("singular value index (rank)")
    ax1.set_ylabel("sigma^2 (variance captured)")
    ax1.set_title("Singular value spectrum of state matrix\n(at end of 4-window chunk)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(K, len(sq_lb)))

    ax2.plot(idx, cum_lb, marker="o", color="tab:red", label="graph + LB", linewidth=2)
    ax2.plot(idx, cum_no, marker="s", color="tab:orange", label="graph no-LB", linewidth=1.5, linestyle="--")
    ax2.plot(idx, cum_init, marker="^", color="tab:gray", label="init", linewidth=1, alpha=0.7)
    ax2.axhline(0.95, color="black", linestyle=":", alpha=0.5, label="95% variance")
    ax2.set_xlabel("rank used")
    ax2.set_ylabel("cumulative variance fraction")
    ax2.set_title("Effective dimensionality of state field")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(K, len(sq_lb))); ax2.set_ylim(0, 1.05)

    def n95(cum):
        return int((cum < 0.95).sum() + 1)
    n_lb = n95(cum_lb); n_no = n95(cum_no); n_init = n95(cum_init)
    ax2.text(0.55, 0.05,
             f"n@95%:  LB={n_lb}   no-LB={n_no}   init={n_init}",
             transform=ax2.transAxes, fontsize=11,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.suptitle("Effective rank of substrate state  -  LB restores high-dim representation", fontsize=13)
    plt.tight_layout()
    p = OUT / "v1h_t4k_v3_eigenspectrum.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}  (n_eff@95%: LB={n_lb}, no-LB={n_no}, init={n_init})")


def fig_similarity_heatmap():
    print("=== Fig 5: endpoint similarity heatmap ===")
    enc_lb, _ = load_encoder(CKPT_LB)
    wins_lb = run_chunk(enc_lb)
    end = wins_lb[-1]
    src = F.normalize(end["src"][0], dim=-1)
    dst = F.normalize(end["dst"][0], dim=-1)
    state = F.normalize(end["state"][0], dim=-1)
    sim_src = (src @ src.T).numpy()
    sim_dst = (dst @ dst.T).numpy()
    sim_state = (state @ state.T).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sim, name in zip(axes, [sim_src, sim_dst, sim_state],
                              ["src", "dst", "state"]):
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{name} pairwise cosine\nslot x slot (K=68)")
        ax.set_xlabel("slot j"); ax.set_ylabel("slot i")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Slot specialization  -  block/cluster structure = slots took on distinct roles",
                 fontsize=13)
    plt.tight_layout()
    p = OUT / "v1h_t4k_v3_similarity_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def fig_telemetry_curves():
    print("=== Fig 7: training telemetry over time ===")
    def parse(jsonl_path):
        rows = []
        with jsonl_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "step" in r and r.get("phase") != "val":
                    rows.append(r)
        return rows
    rows_lb = parse(JSONL_LB); rows_no = parse(JSONL_NO)

    def extract(rows, key, bucket_size=500):
        buckets = {}
        for r in rows:
            if key not in r: continue
            b = r["step"] // bucket_size
            buckets.setdefault(b, []).append(r[key])
        xs = sorted(buckets.keys())
        ys = [float(np.mean(buckets[b])) for b in xs]
        return [b * bucket_size for b in xs], ys

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    pairs = [
        ("loss_recon", "Train loss_recon (rolling mean per 500 steps)", axes[0, 0]),
        ("graph_u_mean", "Saliency u (mean across slots)", axes[0, 1]),
        ("graph_pick_strength_avg", "Pick strength alpha (avg endpoint update strength)", axes[1, 0]),
        ("graph_overwrites_per_row_per_window", "Slot overwrites per row per window", axes[1, 1]),
    ]
    for key, title, ax in pairs:
        for label, rows, color, ls in [
            ("graph + LB", rows_lb, "tab:red", "-"),
            ("graph no-LB", rows_no, "tab:orange", "--"),
        ]:
            xs, ys = extract(rows, key)
            if xs:
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, label=label)
        ax.set_title(title); ax.set_xlabel("training step"); ax.grid(True, alpha=0.3); ax.legend()
    plt.suptitle("Training-time telemetry  -  LB vs no-LB", fontsize=13)
    plt.tight_layout()
    p = OUT / "v1h_t4k_v3_telemetry_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def fig_routing_matrix():
    print("=== Fig 8: routing decision matrix ===")
    enc_lb, _ = load_encoder(CKPT_LB)
    enc_no, _ = load_encoder(CKPT_NO)

    def collect(enc):
        torch.manual_seed(7)
        cfg = enc.cfg
        B, T_w = 1, 1024
        te = torch.randn(B, T_w * 4, cfg.d_llama) * 0.5
        state = enc.init_streaming_state(B, te.device, te.dtype)
        for w in range(3):
            state, _ = enc.streaming_write(state, te[:, w*T_w:(w+1)*T_w, :],
                                           torch.ones(B, T_w, dtype=torch.bool),
                                           chunk_offset=w*T_w)
        pins = enc.pin_encoder(te[:, 3*T_w:4*T_w, :].to(torch.float32))
        proposals = enc.updater(pins, state["edges"])
        endpoints = torch.cat([state["edges"]["src"], state["edges"]["dst"]], dim=1)
        prop_ep = torch.cat([proposals["src"], proposals["dst"]], dim=1)
        # v3-only API — lazy import; raises a clear error against v4.
        try:
            from src.repr_learning.graph_substrate import expert_choice_routing
        except ImportError:
            raise RuntimeError(
                "This v3-only diagnostic uses `expert_choice_routing`, which "
                "was removed in v4. Checkout tag `graph_baseline_v3_lb_locked` "
                "to run this script."
            ) from None
        picked_idx, _, _, pick_count, _ = expert_choice_routing(
            endpoints, prop_ep, strength_scale=enc.update_strength_scale,
        )
        K = enc.K_max
        twoK = 2 * K
        matrix = torch.zeros(twoK, twoK)
        for i in range(twoK):
            matrix[i, picked_idx[0, i]] = 1.0
        return matrix.numpy(), pick_count[0].numpy()

    mat_lb, pc_lb = collect(enc_lb)
    mat_no, pc_no = collect(enc_no)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for col, (mat, pc, label) in enumerate([(mat_lb, pc_lb, "graph + LB"),
                                              (mat_no, pc_no, "graph no-LB")]):
        ax = axes[0, col]
        ax.imshow(mat, cmap="binary", aspect="auto")
        ax.set_title(f"{label}  -  routing matrix\n(rows=endpoints, cols=proposals; black=pick)")
        ax.set_xlabel("proposal idx (0..K-1 = src side, K..2K = dst side)")
        ax.set_ylabel("endpoint idx")
        ax = axes[1, col]
        ax.bar(np.arange(len(pc)), pc, color="tab:red" if "LB" in label else "tab:orange")
        ax.set_title(f"{label}  -  pick_count per proposal\n(max={int(pc.max())}, mean={pc.mean():.1f}, "
                     f"dead={int((pc==0).sum())}/{len(pc)})")
        ax.set_xlabel("proposal idx"); ax.set_ylabel("# endpoints that picked it")
        ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="uniform = 1")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Routing concentration  -  LB loss spreads picks more evenly", fontsize=13)
    plt.tight_layout()
    p = OUT / "v1h_t4k_v3_routing_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


if __name__ == "__main__":
    import torch as _t_for_nograd
    _ng = _t_for_nograd.no_grad()
    _ng.__enter__()
    fig_per_family()
    fig_state_diversity()
    fig_eigenspectrum()
    fig_similarity_heatmap()
    fig_telemetry_curves()
    fig_routing_matrix()
    print()
    print("All figures saved to docs/plots/")
