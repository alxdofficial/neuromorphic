"""V8 training diagnostics — lightweight metrics + periodic snapshots.

Two tiers:
  1. Extended metrics (every step): memory graph summary stats added to
     the existing JSONL. Cheap — just a few reductions on existing tensors.
  2. Snapshots (every N steps): memory graph connectivity, neuron
     specialization, weight distributions. Written to snapshot dir.

Usage:
    diagnostics = V8Diagnostics(model, save_dir, snapshot_every=1000)
    # In training loop:
    metrics = trainer.train_chunk(batch)
    metrics = diagnostics.extend_metrics(metrics, step)
    diagnostics.maybe_snapshot(step)
"""

import json
import os
import torch
import numpy as np
from torch import Tensor


class V8Diagnostics:
    """Lightweight diagnostics for v8 training."""

    def __init__(self, model, save_dir: str, snapshot_every: int = 1000):
        self.model = model
        self.save_dir = save_dir
        self.snapshot_dir = os.path.join(save_dir, "snapshots")
        self.snapshot_every = snapshot_every
        os.makedirs(self.snapshot_dir, exist_ok=True)
        # Snapshot of initial primitives/keys for tracking drift
        self._init_primitives = None
        self._init_keys = None

    def extend_metrics(self, metrics: dict, step: int) -> dict:
        """Add memory graph + LM coupling stats to the per-step metrics dict.

        Cheap operations only — reductions on existing tensors.
        Does NOT create new tensors or run forward passes.
        """
        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return metrics

        with torch.no_grad():
            # === Memory graph state ===
            metrics["mem_h_norm"] = round(mg.h.norm().item(), 4)
            metrics["mem_h_mean_abs"] = round(mg.h.abs().mean().item(), 6)
            metrics["mem_msg_norm"] = round(mg.prev_messages.norm().item(), 4)

            # Primitive diversity (std across neurons — higher = more specialized)
            prim_std = mg.primitives.std(dim=1).mean().item()
            metrics["mem_prim_std"] = round(prim_std, 6)

            # Key stats (RMS-normalized, unit direction vectors)
            key_div = mg.key.std(dim=1).mean().item()  # diversity across neurons
            metrics["mem_key_diversity"] = round(key_div, 6)

            # Drift from init — tracks whether neuromod is actually changing primitives/keys
            if self._init_primitives is None:
                self._init_primitives = mg.primitives.clone()
                self._init_keys = mg.key.clone()
            prim_drift = (mg.primitives - self._init_primitives).norm(dim=-1).mean().item()
            key_drift = (mg.key - self._init_keys).norm(dim=-1).mean().item()
            metrics["mem_prim_drift"] = round(prim_drift, 6)
            metrics["mem_key_drift"] = round(key_drift, 6)

            # Decay distribution
            decay = torch.sigmoid(mg.decay_logit)
            metrics["mem_decay_mean"] = round(decay.mean().item(), 4)
            metrics["mem_decay_std"] = round(decay.std().item(), 4)

            # Message magnitude (all neurons + port vs non-port split)
            C = self.model.config.C
            mm = mg.msg_magnitude  # [BS, N]
            metrics["mem_msg_mag_mean"] = round(mm.mean().item(), 4)
            metrics["mem_msg_mag_port"] = round(mm[:, :C].mean().item(), 4)
            metrics["mem_msg_mag_nonport"] = round(mm[:, C:].mean().item(), 4)

            # Active neurons: fraction with msg_magnitude > 0.01
            metrics["mem_usage_frac"] = round(
                (mm.mean(dim=0) > 0.01).float().mean().item(), 4)

            # Plasticity counter (cumulative rewires)
            if hasattr(mg, '_plasticity_rewires'):
                metrics["mem_plasticity_rewires"] = mg._plasticity_rewires

            # tanh saturation: fraction of |msg| > 0.95
            msg_abs = mg.prev_messages.abs()
            metrics["mem_tanh_saturated"] = round(
                (msg_abs > 0.95).float().mean().item(), 4)
            # Message magnitude stats
            msg_rms = (mg.prev_messages ** 2).mean(dim=-1).sqrt()
            metrics["mem_msg_rms_mean"] = round(msg_rms.mean().item(), 4)
            metrics["mem_msg_rms_std"] = round(msg_rms.std().item(), 4)

            # Co-activation signal quality
            phi = mg.co_activation_ema
            metrics["mem_phi_mean"] = round(phi.mean().item(), 6)
            metrics["mem_phi_std"] = round(phi.std().item(), 6)
            metrics["mem_phi_pos_frac"] = round((phi > 0).float().mean().item(), 4)
            metrics["mem_phi_neg_frac"] = round((phi < 0).float().mean().item(), 4)
            metrics["mem_phi_abs_max"] = round(phi.abs().max().item(), 4)

            # === LM coupling ===
            lm = self.model.lm

            # Memory gate: sigmoid(mem_gate) per CC — critical coupling signal
            gate = torch.sigmoid(lm.mem_gate)  # [C]
            metrics["mem_gate_mean"] = round(gate.mean().item(), 4)
            metrics["mem_gate_min"] = round(gate.min().item(), 4)
            metrics["mem_gate_max"] = round(gate.max().item(), 4)

            # === PCM diagnostics (from cached forward stats) ===
            pcm_stats = getattr(lm, '_pcm_stats', None)
            if pcm_stats is not None:
                metrics["pcm_surprise_mean"] = round(pcm_stats["surprise_mean"], 4)
                metrics["pcm_surprise_std"] = round(pcm_stats["surprise_std"], 4)
                metrics["pcm_surprise_max"] = round(pcm_stats["surprise_max"], 4)
                # Per-CC breakdown (min/max/spread across columns)
                per_cc_surp = pcm_stats["surprise_per_cc"]
                metrics["pcm_surprise_cc_min"] = round(min(per_cc_surp), 4)
                metrics["pcm_surprise_cc_max"] = round(max(per_cc_surp), 4)
                per_cc_loss = pcm_stats["pred_loss_per_cc"]
                metrics["pcm_pred_loss_mean"] = round(
                    sum(per_cc_loss) / len(per_cc_loss), 6)
                metrics["pcm_pred_loss_min"] = round(min(per_cc_loss), 6)
                metrics["pcm_pred_loss_max"] = round(max(per_cc_loss), 6)

            # === Neuromod policy stats ===
            nm = self.model.neuromod
            metrics["nm_logstd_gate"] = round(nm.gate_logstd.mean().item(), 8)
            metrics["nm_logstd_decay"] = round(nm.decay_logstd.mean().item(), 8)

            # Eligibility trace stats
            metrics["nm_trace_prim_norm"] = round(
                mg.trace_prim.norm(dim=-1).mean().item(), 4)
            metrics["nm_trace_key_norm"] = round(
                mg.trace_key.norm(dim=-1).mean().item(), 4)

            # Action stats (gate + decay_target from a sample forward)
            if mg.h is not None and mg.h.shape[0] > 0:
                obs = mg.get_neuron_obs()[:1]
                obs_flat = obs.reshape(-1, obs.shape[-1])
                nm_dtype = next(nm.parameters()).dtype
                with torch.no_grad():
                    action, _, _, _ = nm.get_action_and_value(obs_flat.to(nm_dtype))
                gate = action[:, 0]
                metrics["nm_gate_mean"] = round(gate.mean().item(), 4)
                metrics["nm_gate_std"] = round(gate.std().item(), 4)

        return metrics

    def maybe_snapshot(self, step: int):
        """Save a detailed snapshot of memory graph state if due."""
        if self.snapshot_every <= 0 or step % self.snapshot_every != 0:
            return
        if step == 0:
            return

        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return

        snap_path = os.path.join(self.snapshot_dir, f"step_{step:06d}.pt")

        with torch.no_grad():
            snapshot = {}

            # Per-neuron summary stats (averaged over batch)
            # [N] vectors — small, safe to save
            snapshot["prim_norm_per_neuron"] = mg.primitives.norm(dim=-1).mean(dim=0).cpu()
            snapshot["h_norm_per_neuron"] = mg.h.norm(dim=-1).mean(dim=0).cpu()
            snapshot["msg_norm_per_neuron"] = mg.prev_messages.norm(dim=-1).mean(dim=0).cpu()
            snapshot["decay_per_neuron"] = torch.sigmoid(mg.decay_logit).mean(dim=0).cpu()
            snapshot["msg_magnitude_per_neuron"] = mg.msg_magnitude.mean(dim=0).cpu()

            # Connection weight distribution per neuron [N, K]
            snapshot["key_per_neuron"] = mg.key.mean(dim=0).cpu()
            # Connectivity structure [N, K] — topology doesn't change often
            snapshot["conn_indices"] = mg.conn_indices.cpu()

            # Co-activation matrix [N, N] and primitives [N, D]
            snapshot["co_activation"] = mg.co_activation_ema.cpu().float()
            snapshot["primitives_mean"] = mg.primitives.mean(dim=0).cpu()  # [N, D]
            snapshot["key_mean"] = mg.key.mean(dim=0).cpu()  # [N, D]

            # Routing weights (if computed) [N, K] — batch-averaged
            if hasattr(mg, '_routing_weights') and mg._routing_weights is not None:
                snapshot["routing_weights"] = mg._routing_weights.mean(dim=0).cpu()

            snapshot["step"] = step
            snapshot["config"] = {
                "N": mg.config.N_neurons,
                "K": mg.config.K_connections,
                "D": mg.config.D_mem,
                "C": mg.config.C,
            }

        torch.save(snapshot, snap_path)
