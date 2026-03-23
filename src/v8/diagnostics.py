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

    def extend_metrics(self, metrics: dict, step: int) -> dict:
        """Add memory graph stats to the per-step metrics dict.

        Cheap operations only — reductions on existing tensors.
        Does NOT create new tensors or run forward passes.
        """
        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return metrics

        with torch.no_grad():
            # Neuron state magnitudes
            metrics["mem_h_norm"] = round(mg.h.norm().item(), 4)
            metrics["mem_h_mean_abs"] = round(mg.h.abs().mean().item(), 6)
            metrics["mem_msg_norm"] = round(mg.prev_messages.norm().item(), 4)

            # Primitive diversity (std across neurons — higher = more specialized)
            prim_std = mg.primitives.std(dim=1).mean().item()
            metrics["mem_prim_std"] = round(prim_std, 6)

            # Connection weight stats (L1-normalized, so sum |w| = 1 per neuron)
            cw = mg.conn_weights
            metrics["mem_cw_mean"] = round(cw.mean().item(), 6)
            metrics["mem_cw_std"] = round(cw.std().item(), 6)
            metrics["mem_cw_max"] = round(cw.abs().max().item(), 6)

            # Decay distribution
            decay = torch.sigmoid(mg.decay_logit)
            metrics["mem_decay_mean"] = round(decay.mean().item(), 4)
            metrics["mem_decay_std"] = round(decay.std().item(), 4)

            # Firing rate and plasticity
            metrics["mem_firing_rate"] = round(mg.firing_rate.mean().item(), 4)
            metrics["mem_flow_mean"] = round(mg.flow_ema.mean().item(), 6)
            metrics["mem_corr_mean"] = round(mg.corr_ema.mean().item(), 6)

            # Co-activation stats (phi coefficient matrix)
            phi = mg.co_activation_ema
            metrics["mem_phi_mean"] = round(phi.mean().item(), 6)
            metrics["mem_phi_pos_frac"] = round((phi > 0).float().mean().item(), 4)
            metrics["mem_phi_neg_frac"] = round((phi < 0).float().mean().item(), 4)

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
            snapshot["firing_rate_per_neuron"] = mg.firing_rate.mean(dim=0).cpu()

            # Connection weight distribution per neuron [N, K]
            snapshot["cw_per_neuron"] = mg.conn_weights.mean(dim=0).cpu()
            snapshot["flow_per_neuron"] = mg.flow_ema.mean(dim=0).cpu()
            snapshot["corr_per_neuron"] = mg.corr_ema.mean(dim=0).cpu()

            # Connectivity structure [N, K] — topology doesn't change often
            snapshot["conn_indices"] = mg.conn_indices.cpu()

            # Adjacency matrix summary (not the full N×N, just stats)
            A = mg._build_adjacency()
            snapshot["adj_row_norms"] = A.norm(dim=-1).mean(dim=0).cpu()  # [N]
            snapshot["adj_sparsity"] = (A.abs() < 1e-6).float().mean().item()

            snapshot["step"] = step
            snapshot["config"] = {
                "N": mg.config.N_neurons,
                "K": mg.config.K_connections,
                "D": mg.config.D_mem,
                "C": mg.config.C,
            }

        torch.save(snapshot, snap_path)
