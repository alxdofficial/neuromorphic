"""v10 training diagnostics — scalar neuron memory graph."""

import os
import torch
from torch import Tensor


class V8Diagnostics:
    """Lightweight diagnostics for v10 training."""

    def __init__(self, model, save_dir: str, snapshot_every: int = 1000):
        self.model = model
        self.save_dir = save_dir
        self.snapshot_dir = os.path.join(save_dir, "snapshots")
        self.snapshot_every = snapshot_every
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def extend_metrics(self, metrics: dict, step: int) -> dict:
        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return metrics

        with torch.no_grad():
            # Neuron state
            metrics["mem_V_mean"] = round(mg.V.mean().item(), 4)
            metrics["mem_V_std"] = round(mg.V.std().item(), 4)
            metrics["mem_act_mean"] = round(mg.activation.mean().item(), 4)

            # Neuron properties (set by modulator)
            metrics["mem_w_conn_mean"] = round(mg.w_conn.mean().item(), 4)
            metrics["mem_w_conn_std"] = round(mg.w_conn.std().item(), 4)
            metrics["mem_decay_mean"] = round(mg.decay.mean().item(), 4)
            metrics["mem_decay_std"] = round(mg.decay.std().item(), 4)
            metrics["mem_threshold_mean"] = round(mg.threshold.mean().item(), 4)

            # Activation magnitude (neuron usage)
            am = mg.activation_magnitude
            metrics["mem_act_mag_mean"] = round(am.mean().item(), 4)
            metrics["mem_usage_frac"] = round(
                (am.mean(dim=0) > 0.01).float().mean().item(), 4)

            # Hebbian traces
            metrics["mem_hebbian_mean"] = round(mg.hebbian.mean().item(), 4)
            metrics["mem_hebbian_std"] = round(mg.hebbian.std().item(), 4)

            # LM coupling
            lm = self.model.lm
            gate = torch.sigmoid(lm.mem_gate)
            metrics["mem_gate_mean"] = round(gate.mean().item(), 4)
            metrics["mem_gate_min"] = round(gate.min().item(), 4)
            metrics["mem_gate_max"] = round(gate.max().item(), 4)

            # Modulator param norms
            metrics["mod_w1_norm"] = round(mg.mod_w1.norm().item(), 4)
            metrics["mod_w2_norm"] = round(mg.mod_w2.norm().item(), 4)

            # Gradient norms
            if mg.mod_w1.grad is not None:
                metrics["mem_grad_norm"] = round(
                    sum(p.grad.norm().item() ** 2
                        for p in mg.parameters() if p.grad is not None) ** 0.5, 6)

            # Structural plasticity
            if hasattr(mg, '_last_rewire_swaps'):
                metrics["plasticity_swaps"] = mg._last_rewire_swaps

            # PCM
            pcm_stats = getattr(lm, '_pcm_stats', None)
            if pcm_stats is not None:
                metrics["pcm_surprise_mean"] = round(pcm_stats["surprise_mean"], 4)
                metrics["pcm_surprise_std"] = round(pcm_stats["surprise_std"], 4)

        return metrics

    def maybe_snapshot(self, step: int):
        if self.snapshot_every <= 0 or step % self.snapshot_every != 0:
            return
        if step == 0:
            return

        mg = self.model.memory
        if mg is None or not mg.is_initialized():
            return

        snap_path = os.path.join(self.snapshot_dir, f"step_{step:06d}.pt")

        with torch.no_grad():
            snapshot = {
                "step": step,
                "V_mean_per_neuron": mg.V.mean(dim=0).cpu(),
                "act_mean_per_neuron": mg.activation.mean(dim=0).cpu(),
                "decay_per_neuron": mg.decay.mean(dim=0).cpu(),
                "threshold_per_neuron": mg.threshold.mean(dim=0).cpu(),
                "activation_magnitude": mg.activation_magnitude.mean(dim=0).cpu(),
                "conn_indices": mg.conn_indices.cpu(),
                "config": {
                    "N": mg.config.N_neurons,
                    "K": mg.config.K_connections,
                    "n_groups": mg.config.n_groups,
                    "group_size": mg.config.group_size,
                },
            }

        torch.save(snapshot, snap_path)
