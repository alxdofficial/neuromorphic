"""Telemetry — collect per-step training stats and stream them to disk.

Coverage (~40 metrics across 8 categories):
- training        loss, ce, aux_ce, load_balance, grad_norm, lr, tok/sec
- column_state    ||s|| percentiles, touched_frac, visit_entropy
- walker_heads    ||walker_state|| per head, walker_state_alpha per head,
                  head co-location rate, head-plane distribution
- routing         tau, epsilon, routing_entropy, edge_diversity,
                  exploration_rate
- neuromod        ||delta_nm||, gamma, ||E_bias|| / max / >thresh frac,
                  per-layer grad norm
- surprise        per-horizon surprise EMA, plast_eta, hebb-vs-nm magnitude
- gradients       per-component grad norms (token_emb, content_mlp,
                  q_proj, k_all, nbr_id_to_s, mem_input_v_proj,
                  state_to_model, walker_state_alpha, neuromod.*)
- llama           W_in/W_out grad+value norms, scale norm,
                  inject_residual_norm, ce_minus_vanilla (when available)

Usage:
    collector = StatsCollector(work_dir="outputs/run1")
    for step in range(N):
        stats = phase1_pretrained_step(...)
        collector.snapshot(wrapper, step=step, phase="phase1", stats=stats)
    collector.close()

    # Plotting (offline):
    from src.graph_walker.telemetry import plot_dashboard
    plot_dashboard("outputs/run1/stats.jsonl", out_dir="outputs/run1/plots")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _norm(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    return float(t.detach().float().norm().item())


def _grad_norm(p: torch.nn.Parameter) -> float:
    if p.grad is None:
        return 0.0
    return float(p.grad.detach().float().norm().item())


def _percentiles(t: torch.Tensor, qs: tuple[float, ...]) -> dict[str, float]:
    if t.numel() == 0:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    flat = t.detach().float().flatten()
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(torch.quantile(flat, q).item())
    return out


def _entropy_normalized(counts: torch.Tensor) -> float:
    """H(counts) / log(N), in [0, 1]. 1.0 = uniform, 0.0 = single bucket."""
    total = counts.sum()
    if total <= 0 or counts.numel() == 0:
        return 1.0
    p = (counts.float() / total).clamp(min=1e-12)
    ent = -(p * p.log()).sum()
    ent_uniform = math.log(counts.numel())
    if ent_uniform <= 0:
        return 1.0
    return float((ent / ent_uniform).item())


@dataclass
class _StepRow:
    step: int
    phase: str
    payload: dict[str, Any]


class StatsCollector:
    """Per-step stats collector. Writes one JSONL row per step.

    All metrics are scalars (or short lists per horizon / per head). Keep
    the row size small enough that millions of steps fit on disk without
    careful pruning."""

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / "stats.jsonl"
        self._fh = open(self.path, "a")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Core: collect everything from the wrapper after a training step.
    # ------------------------------------------------------------------

    def snapshot(
        self,
        wrapper,
        *,
        step: int,
        phase: str,
        stats: Any,           # Phase1Stats | Phase1ARStats | GRPOStats
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture all metrics and append a JSONL row.

        Returns the row dict so callers can also log it to console / wandb.
        """
        m = wrapper.memory
        cfg = wrapper.config
        memcfg = cfg.memory

        row: dict[str, Any] = {"step": step, "phase": phase}

        # ---- training ----
        for fld in ("loss", "ce_loss", "aux_loss", "load_balance_loss",
                    "grad_norm", "tok_per_sec", "reward_mean", "reward_std",
                    "log_pi_mean", "advantage_max", "inject_residual_norm"):
            if hasattr(stats, fld):
                row[f"train.{fld}"] = float(getattr(stats, fld))

        # ---- column state ----
        if m is not None and m.s is not None:
            s_norm = m.s.detach().float().norm(dim=-1)            # [B, N]
            row.update({
                f"col.s_norm.{k}": v
                for k, v in _percentiles(
                    s_norm, (0.5, 0.9, 0.99),
                ).items()
            })
            if m.visit_count is not None:
                vc = m.visit_count.detach().float()
                row["col.touched_frac"] = float((vc > 0).float().mean().item())
                row["col.visit_entropy_norm"] = _entropy_normalized(vc)

        # ---- walker heads ----
        if m is not None and m.walker_state is not None:
            ws = m.walker_state.detach().float()                  # [B, H, D_s]
            ws_norm_per_head = ws.norm(dim=-1).mean(dim=0)        # [H]
            row["walker.state_norm_per_head"] = ws_norm_per_head.tolist()
            alpha_w = torch.sigmoid(m.walker_state_alpha.detach()).tolist()
            row["walker.alpha_per_head"] = alpha_w
            # Head co-location rate: fraction of (b, h1, h2 != h1) head pairs
            # at the same column. High = walkers collapsed; low = differentiated.
            wp = m.walker_pos.detach()                            # [B, H]
            B, H = wp.shape
            if H > 1:
                eq = (wp.unsqueeze(2) == wp.unsqueeze(1)).float()  # [B, H, H]
                # Subtract diagonal (always equal), divide by H*(H-1) per b.
                eq = eq - torch.eye(H, device=eq.device).unsqueeze(0)
                row["walker.colocation_rate"] = float(
                    (eq.sum() / (B * H * (H - 1))).item()
                )

        # ---- routing schedule ----
        if m is not None:
            tau, eps = m._schedule_tensors(torch.zeros(1, dtype=torch.long))
            row["routing.tau"] = float(tau)
            row["routing.epsilon"] = float(eps)

        # ---- neuromod ----
        if m is not None and m.neuromod is not None:
            if m._active_delta_nm is not None:
                row["neuromod.delta_nm_norm"] = _norm(m._active_delta_nm)
            row["neuromod.gamma"] = float(
                torch.sigmoid(m.neuromod.blend_logit.detach()).item()
            )
            row["neuromod.E_bias_norm"] = _norm(m.E_bias_flat)
            E_abs = m.E_bias_flat.detach().abs()
            row["neuromod.E_bias_max"] = float(E_abs.max().item())
            row["neuromod.E_bias_active_frac"] = float(
                (E_abs > 0.1).float().mean().item()
            )

        # ---- surprise ----
        if m is not None and m.surprise_ema is not None:
            sema = m.surprise_ema.detach().float().mean(dim=0)    # [K_h]
            row["surprise.per_horizon"] = sema.tolist()
            row["surprise.mean"] = float(sema.mean().item())
            # Plast eta is computed on the fly inside _plasticity_step;
            # mirror the formula here as a diagnostic:
            surprise_scalar = sema.mean()
            row["surprise.plast_eta"] = float(
                memcfg.plast_eta * torch.sigmoid(
                    surprise_scalar - memcfg.plast_surprise_bias,
                ).item()
            )

        # ---- per-component gradient norms ----
        if m is not None:
            for prefix in (
                "tied_token_emb", "cols.content_mlp", "cols.q_proj",
                "cols.k_proj", "nbr_id_to_s", "mem_input_v_proj",
                "state_to_model", "walker_state_alpha",
                "neuromod.edge_mlp", "neuromod.feature_proj",
                "neuromod.layers", "neuromod.blend_logit",
                "input_q_proj", "out_k_proj", "out_v_proj",
            ):
                gn = 0.0
                for name, p in m.named_parameters():
                    if name.startswith(prefix):
                        gn += _grad_norm(p) ** 2
                gn = math.sqrt(gn)
                row[f"grad.{prefix}"] = gn

        # ---- Llama-side ----
        try:
            inj = wrapper.mem_inject
            row["llama.W_in_norm"] = _norm(inj.W_in.weight)
            row["llama.W_out_norm"] = _norm(inj.W_out.weight)
            row["llama.scale_norm"] = _norm(inj.scale)
            row["llama.W_in_grad_norm"] = _grad_norm(inj.W_in.weight)
            row["llama.W_out_grad_norm"] = _grad_norm(inj.W_out.weight)
            row["llama.scale_grad_norm"] = _grad_norm(inj.scale)
        except Exception:
            pass

        if extra:
            row.update(extra)

        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        return row


# ---------------------------------------------------------------------
# Plotting — offline helper. Reads a stats.jsonl and produces 8 PNGs.
# ---------------------------------------------------------------------

def plot_dashboard(stats_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Read a stats.jsonl produced by `StatsCollector.snapshot` and write
    8 PNGs to `out_dir`. Returns the list of paths written.

    Plot panels:
        1. training_curves.png          loss / ce / aux_ce / grad_norm
        2. column_state.png             s_norm percentiles, touched_frac, visit_entropy
        3. walker_heads.png             per-head walker_state norm, alpha, colocation
        4. routing.png                  tau, epsilon, plast_eta
        5. neuromod.png                 delta_nm norm, gamma, E_bias norm/max/active_frac
        6. surprise.png                 per-horizon surprise EMA over time
        7. gradient_flow.png            grad norm per component
        8. llama_side.png               W_in/W_out/scale grad+value norms, inject_residual

    matplotlib is imported lazily so the production import path stays light.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats_path = Path(stats_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return []

    steps = [r["step"] for r in rows]

    def _series(key: str) -> list[float]:
        return [r.get(key, math.nan) for r in rows]

    def _save(fig, name: str) -> Path:
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        return p

    written: list[Path] = []

    # 1. training curves
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key in zip(
        axes.flat,
        ["train.loss", "train.ce_loss", "train.aux_loss", "train.grad_norm"],
    ):
        ax.plot(steps, _series(key), lw=1)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.suptitle("Training curves")
    written.append(_save(fig, "01_training_curves.png"))

    # 2. column state
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, title in [
        (axes[0], "col.s_norm.p90", "||s|| P90"),
        (axes[1], "col.touched_frac", "Touched fraction"),
        (axes[2], "col.visit_entropy_norm", "Visit entropy (normalized)"),
    ]:
        ax.plot(steps, _series(key), lw=1)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Column state utilization")
    written.append(_save(fig, "02_column_state.png"))

    # 3. walker heads
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # Per-head series — extract from walker.state_norm_per_head (list)
    norms_by_head: list[list[float]] = []
    alphas_by_head: list[list[float]] = []
    for r in rows:
        norms_by_head.append(r.get("walker.state_norm_per_head", []))
        alphas_by_head.append(r.get("walker.alpha_per_head", []))
    if norms_by_head and norms_by_head[0]:
        H = len(norms_by_head[0])
        for h in range(H):
            axes[0].plot(steps, [r[h] if h < len(r) else math.nan for r in norms_by_head],
                         lw=1, label=f"head {h}")
            axes[1].plot(steps, [r[h] if h < len(r) else math.nan for r in alphas_by_head],
                         lw=1, label=f"head {h}")
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    axes[0].set_title("||walker_state|| per head")
    axes[1].set_title("σ(walker_state_alpha) per head")
    axes[2].plot(steps, _series("walker.colocation_rate"), lw=1)
    axes[2].set_title("Head co-location rate")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Walker heads")
    written.append(_save(fig, "03_walker_heads.png"))

    # 4. routing schedule
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes,
        ["routing.tau", "routing.epsilon", "surprise.plast_eta"]):
        ax.plot(steps, _series(key), lw=1)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.suptitle("Routing & plasticity schedules")
    written.append(_save(fig, "04_routing.png"))

    # 5. neuromod
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key in zip(axes.flat, [
        "neuromod.delta_nm_norm", "neuromod.gamma",
        "neuromod.E_bias_norm", "neuromod.E_bias_active_frac",
    ]):
        ax.plot(steps, _series(key), lw=1)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.suptitle("Neuromod")
    written.append(_save(fig, "05_neuromod.png"))

    # 6. surprise per horizon
    fig, ax = plt.subplots(figsize=(8, 4))
    per_h: list[list[float]] = []
    for r in rows:
        per_h.append(r.get("surprise.per_horizon", []))
    if per_h and per_h[0]:
        K_h = len(per_h[0])
        for k in range(K_h):
            ax.plot(steps, [r[k] if k < len(r) else math.nan for r in per_h],
                    lw=1, label=f"h{k+1}")
        ax.legend(fontsize=8, ncol=K_h)
    ax.set_title("Surprise EMA per horizon")
    ax.grid(alpha=0.3)
    written.append(_save(fig, "06_surprise.png"))

    # 7. gradient flow per component
    grad_keys = sorted({k for r in rows for k in r if k.startswith("grad.")})
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in grad_keys:
        ax.plot(steps, _series(key), lw=1, label=key.replace("grad.", ""))
    ax.set_yscale("log")
    ax.set_title("Gradient norm per component")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    written.append(_save(fig, "07_gradient_flow.png"))

    # 8. llama-side
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key in zip(axes.flat, [
        "llama.scale_norm", "llama.W_out_norm",
        "llama.W_out_grad_norm", "train.inject_residual_norm",
    ]):
        ax.plot(steps, _series(key), lw=1)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.suptitle("Llama integration")
    written.append(_save(fig, "08_llama_side.png"))

    return written
