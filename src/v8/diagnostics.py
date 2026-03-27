"""V8/v9 training diagnostics — lightweight metrics + periodic snapshots.

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
    """Lightweight diagnostics for v8/v9 training."""

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
        """Add memory graph stats to the per-step metrics dict."""
        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return metrics

        with torch.no_grad():
            # === Memory graph state ===
            metrics["mem_h_norm"] = round(mg.h.norm().item(), 4)
            metrics["mem_h_mean_abs"] = round(mg.h.abs().mean().item(), 6)
            metrics["mem_msg_norm"] = round(mg.prev_messages.norm().item(), 4)

            # Primitive diversity (std across neurons — higher = more specialized)
            prim = mg.primitives.data  # [N, D] nn.Parameter
            prim_std = prim.std(dim=0).mean().item()  # std across neurons per dim
            metrics["mem_prim_std"] = round(prim_std, 6)

            # Key diversity
            key = mg.key.data  # [N, D]
            key_div = key.std(dim=0).mean().item()
            metrics["mem_key_diversity"] = round(key_div, 6)

            # Drift from init
            if self._init_primitives is None:
                self._init_primitives = prim.clone()
                self._init_keys = key.clone()
            prim_drift = (prim - self._init_primitives).norm(dim=-1).mean().item()
            key_drift = (key - self._init_keys).norm(dim=-1).mean().item()
            metrics["mem_prim_drift"] = round(prim_drift, 6)
            metrics["mem_key_drift"] = round(key_drift, 6)

            # Decay distribution
            decay = torch.sigmoid(mg.decay_logit.data)  # [N]
            metrics["mem_decay_mean"] = round(decay.mean().item(), 4)
            metrics["mem_decay_std"] = round(decay.std().item(), 4)

            # Message magnitude
            mm = mg.msg_magnitude  # [BS, N]
            metrics["mem_msg_mag_mean"] = round(mm.mean().item(), 4)
            metrics["mem_msg_mag_std"] = round(mm.std().item(), 4)
            metrics["mem_msg_mag_max"] = round(mm.max().item(), 4)

            # Active neurons
            metrics["mem_usage_frac"] = round(
                (mm.mean(dim=0) > 0.01).float().mean().item(), 4)

            # Inject/readout weight stats
            inject_w = torch.sigmoid(mg.inject_w.data)  # [N, C_mem]
            readout_w = torch.sigmoid(mg.readout_w.data)  # [C_mem, N]
            metrics["mem_inject_mean"] = round(inject_w.mean().item(), 4)
            metrics["mem_inject_std"] = round(inject_w.std().item(), 4)
            metrics["mem_readout_mean"] = round(readout_w.mean().item(), 4)
            metrics["mem_readout_std"] = round(readout_w.std().item(), 4)

            # tanh saturation
            msg_abs = mg.prev_messages.abs()
            metrics["mem_tanh_saturated"] = round(
                (msg_abs > 0.95).float().mean().item(), 4)

            # Message RMS
            msg_rms = (mg.prev_messages ** 2).mean(dim=-1).sqrt()
            metrics["mem_msg_rms_mean"] = round(msg_rms.mean().item(), 4)

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
                metrics["pcm_pred_loss_min"] = round(min(per_cc_loss), 6)
                metrics["pcm_pred_loss_max"] = round(max(per_cc_loss), 6)

            # === Per-neuron modulator stats ===
            # Run modulator on current h to get gate/decay_mod (first batch element)
            if mg.h is not None and mg.h.shape[0] > 0:
                gate_p, gate_k, decay_mod = mg._modulator_forward(mg.h[:1, :],
                    _trace_prim=mg.trace_prim[:1], _trace_key=mg.trace_key[:1])
                metrics["mem_mod_gate_prim_mean"] = round(gate_p.mean().item(), 4)
                metrics["mem_mod_gate_prim_std"] = round(gate_p.std().item(), 4)
                metrics["mem_mod_gate_key_mean"] = round(gate_k.mean().item(), 4)
                metrics["mem_mod_decay_mod_mean"] = round(decay_mod.mean().item(), 4)
                metrics["mem_mod_decay_mod_std"] = round(decay_mod.std().item(), 4)
                metrics["mem_mod_lr"] = round(
                    torch.sigmoid(mg.mod_lr_logit).item(), 6)

            # Trace norms
            metrics["mem_trace_prim_norm"] = round(
                mg.trace_prim.norm(dim=-1).mean().item(), 4)
            metrics["mem_trace_key_norm"] = round(
                mg.trace_key.norm(dim=-1).mean().item(), 4)

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

            snapshot = {
                "step": step,
                "config": {
                    "N": N, "C": C, "K": K,
                    "D": self.model.config.D_mem,
                },
            }

            # Per-neuron norms (batch-averaged)
            snapshot["h_norm_per_neuron"] = mg.h.norm(dim=-1).mean(dim=0).cpu()
            snapshot["msg_norm_per_neuron"] = mg.prev_messages.norm(dim=-1).mean(dim=0).cpu()
            snapshot["msg_magnitude_per_neuron"] = mg.msg_magnitude.mean(dim=0).cpu()

            # Decay per neuron
            snapshot["decay_per_neuron"] = torch.sigmoid(mg.decay_logit.data).cpu()

            # Learned parameter snapshots
            snapshot["primitives_mean"] = mg.primitives.data.cpu()  # [N, D]
            snapshot["key_per_neuron"] = mg.key.data.cpu()  # [N, D]

            # Connectivity
            snapshot["conn_indices"] = mg.conn_indices.cpu()

            # Modulator gate distribution
            if mg.h is not None and mg.h.shape[0] > 0:
                gate_p, gate_k, decay_mod = mg._modulator_forward(
                    mg.h[:1], _trace_prim=mg.trace_prim[:1], _trace_key=mg.trace_key[:1])
                snapshot["mod_gate_prim"] = gate_p[0].cpu()  # [N, 1]
                snapshot["mod_gate_key"] = gate_k[0].cpu()
                snapshot["mod_decay_mod"] = decay_mod[0].cpu()

            snap_path = os.path.join(
                self.snapshot_dir, f"step_{step:06d}.pt")
            torch.save(snapshot, snap_path)
