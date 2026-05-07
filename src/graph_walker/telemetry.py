"""Telemetry — collect per-step training stats and stream them to disk.

Coverage (~40 metrics across 8 categories):
- training        loss, ce, aux_ce, load_balance, grad_norm, lr, tok/sec
- column_state    ||s|| percentiles, touched_frac, visit_entropy
- walker_heads    ||walker_state|| per head, walker_state_alpha per head,
                  head co-location rate
- routing         tau, epsilon, routing_entropy, edge_diversity,
                  exploration_rate
- neuromod        ||delta_nm||, gamma, ||E_bias|| / max / >thresh frac,
                  per-layer grad norm
- surprise        per-horizon surprise EMA, plast_eta, hebb-vs-nm magnitude
- gradients       per-component grad norms (token_emb, content_mlp,
                  q_proj, k_all, nbr_id_to_s,
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
        # State for delta-style metrics (across snapshot calls). We keep
        # the previous E_bias_flat tensor (clone) so we can compute the
        # TRUE delta-norm `||E_t - E_{t-1}||`, not just `abs(||E_t|| -
        # ||E_{t-1}||)` which is blind to rotations / sign flips.
        # Memory cost: one [N*K] fp32 buffer ≈ 64KB at default config.
        self._prev_E_bias: torch.Tensor | None = None
        # Step counter for throttling expensive checks (per-param NaN scan
        # is the biggest culprit). Currently disabled — we run the scan
        # every step but only on params with non-None grad to keep cost
        # bounded.
        self._snap_count: int = 0

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
                    "ce_p50", "ce_p90", "ce_p99", "ce_max",
                    "grad_norm", "tok_per_sec",
                    "reward_mean", "reward_std", "reward_min", "reward_max",
                    "log_pi_mean", "log_pi_max_abs",
                    "advantage_max", "advantage_std",
                    "inject_residual_norm", "inject_residual_ratio"):
            if hasattr(stats, fld):
                row[f"train.{fld}"] = float(getattr(stats, fld))
        # Integer fields (don't float-cast).
        for fld in ("gen_unique_count",):
            if hasattr(stats, fld):
                row[f"train.{fld}"] = int(getattr(stats, fld))

        # ---- AR phase-1 specifics: per-position CE diagnostics ----
        # Phase1ARStats has `ce_per_step: list[float]`. The MEMORY-effect
        # signature is: late-position CE drops below early-position CE
        # (walker has had time to write the prefix's relevant info; LM
        # uses it to predict the late tokens better). If ce_first ≈
        # ce_last, walker isn't being used. If ce_first < ce_last, the
        # walker is actively HURTING (or just absent).
        ce_per_step = getattr(stats, "ce_per_step", None)
        if isinstance(ce_per_step, (list, tuple)) and len(ce_per_step) > 0:
            ce_arr = [float(c) for c in ce_per_step]
            row["train.ar.ce_first"] = ce_arr[0]
            row["train.ar.ce_last"] = ce_arr[-1]
            row["train.ar.ce_mean"] = sum(ce_arr) / len(ce_arr)
            # Drop = first - last; positive if memory helps.
            row["train.ar.ce_drop"] = ce_arr[0] - ce_arr[-1]
            row["train.ar.ce_per_step_len"] = len(ce_arr)

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
            if m._active_neuromod_delta is not None:
                row["neuromod.delta_nm_norm"] = _norm(m._active_neuromod_delta)
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
                "cols.k_proj", "nbr_id_to_s",
                "state_to_model", "walker_state_alpha",
                "neuromod.edge_mlp", "neuromod.feature_proj",
                "neuromod.layers", "neuromod.blend_logit",
                "neuromod.edge_bias_proj",
                "out_k_proj", "out_v_proj",
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

        # ---- walker plasticity health (cheap sanity flags) ----
        # These catch silent dead-walker / dead-neuromod situations:
        # - has_active_neuromod_delta=False means routing this step had no
        #   neuromod gradient signal (only plain E_bias_flat used).
        # - window_len reaching 0 between steps = plasticity fired
        #   correctly. Stuck nonzero across calls = plasticity didn't
        #   fire (update_plasticity wasn't called or returned early).
        # - snapshot_size == 0 = no touched cols recorded; neuromod has
        #   no input to condition on next step.
        if m is not None:
            row["walker.window_len"] = int(getattr(m, "window_len", 0))
            row["walker.has_active_neuromod_delta"] = bool(
                getattr(m, "_active_neuromod_delta", None) is not None
            )
            snap_ids = getattr(m, "_neuromod_input_ids", None)
            row["walker.snapshot_size"] = int(snap_ids.numel()) if snap_ids is not None else 0
            row["walker.phase"] = str(getattr(m, "phase", "phase1"))

        # ---- E_bias delta from previous snapshot ----
        # TRUE delta-norm `||E_t - E_{t-1}||` (not abs of norm-difference,
        # which would miss rotations / sign flips). If E_bias_delta_norm
        # stays near 0 across many steps, neuromod is not learning OR
        # plasticity-window is misfiring.
        if m is not None and m.E_bias_flat is not None:
            cur = m.E_bias_flat.detach().float()
            if self._prev_E_bias is not None and self._prev_E_bias.shape == cur.shape:
                row["mem.E_bias_delta_norm"] = float((cur - self._prev_E_bias).norm().item())
            self._prev_E_bias = cur.clone()

        # ---- VRAM peak (this step's high-water mark) ----
        if torch.cuda.is_available():
            row["vram.peak_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
            row["vram.allocated_mb"] = float(torch.cuda.memory_allocated() / 1e6)
            torch.cuda.reset_peak_memory_stats()

        # ---- NaN / Inf detection across ALL trainable params ----
        # Cheap aggregate (one bool per step). Scope: ALL wrapper params,
        # not just walker — earlier version missed MemInjectLayer's
        # W_in/W_out/scale, which are some of the most numerically active
        # params (gradient flowing from Llama CE through frozen layers).
        # Cost: O(n_trainable_tensors) calls to isfinite().all() ≈ 5-10ms
        # at our scale. Skip non-trainable to keep cost bounded.
        any_nan_grad = False
        any_nan_param = False
        for name, p in wrapper.named_parameters():
            if not p.requires_grad:
                continue
            if not torch.isfinite(p.data).all():
                any_nan_param = True
            if p.grad is not None and not torch.isfinite(p.grad).all():
                any_nan_grad = True
            if any_nan_grad and any_nan_param:
                break
        row["nan.any_nan_grad"] = any_nan_grad
        row["nan.any_nan_param"] = any_nan_param

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

    # 9. VRAM + throughput
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(steps, _series("vram.peak_mb"), lw=1, label="peak")
    axes[0].plot(steps, _series("vram.allocated_mb"), lw=1, label="allocated")
    axes[0].set_title("VRAM (MB)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, _series("train.tok_per_sec"), lw=1)
    axes[1].set_title("Throughput (tok/sec)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Resource utilization")
    written.append(_save(fig, "09_vram_throughput.png"))

    # 10. Walker plasticity health (sanity flags)
    # If `has_active_neuromod_delta` drops to False (and stays there), neuromod
    # has no gradient signal → stuck. If `snapshot_size` is 0, no touched
    # cols recorded → neuromod has no input to condition on.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(steps, _series("walker.window_len"), lw=1)
    axes[0, 0].set_title("walker.window_len  (resets on plasticity fire)")
    axes[0, 0].grid(alpha=0.3)
    has_delta = [1.0 if r.get("walker.has_active_neuromod_delta", False) else 0.0
                  for r in rows]
    axes[0, 1].plot(steps, has_delta, lw=1, drawstyle="steps-post")
    axes[0, 1].set_title("has_active_neuromod_delta  (1=neuromod live, 0=stuck)")
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(steps, _series("walker.snapshot_size"), lw=1)
    axes[1, 0].set_title("snapshot_size  (touched cols feeding neuromod)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(steps, _series("mem.E_bias_delta_norm"), lw=1)
    axes[1, 1].set_title("E_bias_delta_norm  (plasticity moving E_bias?)")
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle("Walker plasticity health")
    written.append(_save(fig, "10_walker_health.png"))

    # 11. Phase-2 GRPO diagnostics (only meaningful when phase=phase2 stats present)
    # Reward distribution + variance + log_pi sanity + generation diversity.
    # Skip silently if no GRPO rows.
    has_grpo = any("train.reward_mean" in r for r in rows)
    if has_grpo:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        # Reward band: min/max/mean across the K rollouts at each step.
        r_mean = _series("train.reward_mean")
        r_min = _series("train.reward_min")
        r_max = _series("train.reward_max")
        axes[0, 0].plot(steps, r_mean, lw=1.5, label="mean", color="C0")
        axes[0, 0].plot(steps, r_min, lw=0.7, label="min", color="C1", alpha=0.7)
        axes[0, 0].plot(steps, r_max, lw=0.7, label="max", color="C2", alpha=0.7)
        axes[0, 0].set_title("Reward (mean / min / max across K rollouts)")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)
        # Reward + advantage std: if std → 0, no learning signal.
        axes[0, 1].plot(steps, _series("train.reward_std"), lw=1, label="reward.std")
        axes[0, 1].plot(steps, _series("train.advantage_std"), lw=1, label="adv.std")
        axes[0, 1].set_title("Reward / advantage std (→0 = no signal)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)
        # log_pi magnitude: should stay ~|log p| (a few nats); regression detector.
        axes[1, 0].plot(steps, _series("train.log_pi_mean"), lw=1, label="mean")
        axes[1, 0].plot(steps, _series("train.log_pi_max_abs"), lw=1, label="max |·|")
        axes[1, 0].set_title("log_pi (mean per step / max |·|)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        # Generation diversity: # unique rollouts. Floor of 1 = collapse.
        axes[1, 1].plot(steps, _series("train.gen_unique_count"), lw=1)
        axes[1, 1].set_title("gen_unique_count  (1 = all rollouts identical)")
        axes[1, 1].grid(alpha=0.3)
        fig.suptitle("GRPO diagnostics")
        written.append(_save(fig, "11_grpo_diagnostics.png"))

    # 12. AR-specific: per-position CE diagnostic (memory-effect signature)
    has_ar = any("train.ar.ce_first" in r for r in rows)
    if has_ar:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(steps, _series("train.ar.ce_first"), lw=1, label="first")
        axes[0].plot(steps, _series("train.ar.ce_last"), lw=1, label="last")
        axes[0].plot(steps, _series("train.ar.ce_mean"), lw=1, label="mean", linestyle="--")
        axes[0].set_title("AR CE: first / last / mean position")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(steps, _series("train.ar.ce_drop"), lw=1, color="C3")
        axes[1].axhline(0.0, color="k", lw=0.5, alpha=0.5)
        axes[1].set_title("CE drop = first - last  (>0 = memory helps)")
        axes[1].grid(alpha=0.3)
        fig.suptitle("AR memory-effect signature")
        written.append(_save(fig, "12_ar_memory_effect.png"))

    # 13. NaN / Inf flags
    has_nan = any(r.get("nan.any_nan_grad") or r.get("nan.any_nan_param")
                   for r in rows)
    if has_nan:
        fig, ax = plt.subplots(figsize=(10, 3))
        nan_grad = [1.0 if r.get("nan.any_nan_grad") else 0.0 for r in rows]
        nan_param = [1.0 if r.get("nan.any_nan_param") else 0.0 for r in rows]
        ax.plot(steps, nan_grad, lw=1, drawstyle="steps-post", label="any_nan_grad")
        ax.plot(steps, nan_param, lw=1, drawstyle="steps-post", label="any_nan_param")
        ax.set_title("⚠️  NaN / Inf detected at these steps")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        written.append(_save(fig, "13_nan_flags.png"))

    return written
