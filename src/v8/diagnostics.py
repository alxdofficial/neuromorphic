"""V8/v9.1 training diagnostics — lightweight metrics + periodic snapshots.

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
    """Lightweight diagnostics for v9.1 training."""

    def __init__(self, model, save_dir: str, snapshot_every: int = 1000):
        self.model = model
        self.save_dir = save_dir
        self.snapshot_dir = os.path.join(save_dir, "snapshots")
        self.snapshot_every = snapshot_every
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._init_w_conn = None

    def extend_metrics(self, metrics: dict, step: int) -> dict:
        """Add memory graph stats to the per-step metrics dict."""
        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return metrics

        with torch.no_grad():
            # === Memory graph state ===
            metrics["mem_h_norm"] = round(mg.h.norm().item(), 4)
            metrics["mem_h_mean_abs"] = round(mg.h.abs().mean().item(), 6)
            metrics["mem_msg_norm"] = round(mg.prev_messages.norm().item(), 4)

            # === Connection weights ===
            w = torch.sigmoid(mg.w_conn.data)  # [N, K]
            metrics["mem_w_conn_mean"] = round(w.mean().item(), 4)
            metrics["mem_w_conn_std"] = round(w.std().item(), 4)
            metrics["mem_w_conn_min"] = round(w.min().item(), 4)
            metrics["mem_w_conn_max"] = round(w.max().item(), 4)

            # w_conn drift from init
            if self._init_w_conn is None:
                self._init_w_conn = mg.w_conn.data.clone()
            w_drift = (mg.w_conn.data - self._init_w_conn).abs().mean().item()
            metrics["mem_w_conn_drift"] = round(w_drift, 6)

            # === Decay distribution ===
            decay = torch.sigmoid(mg.decay_logit.data)  # [N]
            metrics["mem_decay_mean"] = round(decay.mean().item(), 4)
            metrics["mem_decay_std"] = round(decay.std().item(), 4)

            # === Message magnitude ===
            mm = mg.msg_magnitude  # [BS, N]
            metrics["mem_msg_mag_mean"] = round(mm.mean().item(), 4)
            metrics["mem_msg_mag_std"] = round(mm.std().item(), 4)
            metrics["mem_msg_mag_max"] = round(mm.max().item(), 4)

            # Inject/readout weight stats
            inject_w = torch.sigmoid(mg.inject_w.data)
            readout_w = torch.sigmoid(mg.readout_w.data)
            metrics["mem_inject_mean"] = round(inject_w.mean().item(), 4)
            metrics["mem_readout_mean"] = round(readout_w.mean().item(), 4)

            # Active neurons
            metrics["mem_usage_frac"] = round(
                (mm.mean(dim=0) > 0.01).float().mean().item(), 4)

            # === Hebbian learning rate ===
            metrics["mem_hebbian_lr"] = round(
                torch.sigmoid(mg.hebbian_lr_logit).item(), 6)

            # === LM coupling ===
            lm = self.model.lm
            gate = torch.sigmoid(lm.mem_gate)  # [C]
            metrics["mem_gate_mean"] = round(gate.mean().item(), 4)
            metrics["mem_gate_min"] = round(gate.min().item(), 4)
            metrics["mem_gate_max"] = round(gate.max().item(), 4)

            # === PCM diagnostics ===
            pcm_stats = getattr(lm, '_pcm_stats', None)
            if pcm_stats is not None:
                metrics["pcm_surprise_mean"] = round(pcm_stats["surprise_mean"], 4)
                metrics["pcm_surprise_std"] = round(pcm_stats["surprise_std"], 4)
                metrics["pcm_surprise_max"] = round(pcm_stats["surprise_max"], 4)
                per_cc_surp = pcm_stats["surprise_per_cc"]
                metrics["pcm_surprise_cc_min"] = round(min(per_cc_surp), 4)
                metrics["pcm_surprise_cc_max"] = round(max(per_cc_surp), 4)
                per_cc_loss = pcm_stats["pred_loss_per_cc"]
                metrics["pcm_pred_loss_mean"] = round(
                    sum(per_cc_loss) / len(per_cc_loss), 6)

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

        with torch.no_grad():
            N = self.model.config.N_neurons
            C = self.model.config.C
            K = self.model.config.K_connections
            D = self.model.config.D_mem

            snapshot = {
                "step": step,
                "config": {"N": N, "C": C, "K": K, "D": D},
            }

            # Per-neuron norms (batch-averaged)
            snapshot["h_norm_per_neuron"] = mg.h.norm(dim=-1).mean(dim=0).cpu()
            snapshot["msg_norm_per_neuron"] = mg.prev_messages.norm(dim=-1).mean(dim=0).cpu()
            snapshot["msg_magnitude_per_neuron"] = mg.msg_magnitude.mean(dim=0).cpu()

            # Decay per neuron
            snapshot["decay_per_neuron"] = torch.sigmoid(mg.decay_logit.data).cpu()

            # Connection weights per neuron
            snapshot["w_conn"] = torch.sigmoid(mg.w_conn.data).cpu()

            # Connectivity
            snapshot["conn_indices"] = mg.conn_indices.cpu()

            # W1 weight norms per neuron (proxy for neuron "complexity")
            snapshot["W1_norm_per_neuron"] = mg.W1.data.norm(dim=(1, 2)).cpu()

            snap_path = os.path.join(
                self.snapshot_dir, f"step_{step:06d}.pt")
            torch.save(snapshot, snap_path)
