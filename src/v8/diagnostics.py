"""v9-backprop training diagnostics."""

import os
import torch
from torch import Tensor


class V8Diagnostics:
    """Lightweight diagnostics for v9-backprop training."""

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
            # Memory graph state
            metrics["mem_h_norm"] = round(mg.h.norm().item(), 4)
            metrics["mem_msg_norm"] = round(mg.prev_messages.norm().item(), 4)

            # Neuron properties (set by modulator)
            w_sig = torch.sigmoid(mg.w_conn)
            metrics["mem_w_conn_mean"] = round(w_sig.mean().item(), 4)
            metrics["mem_w_conn_std"] = round(w_sig.std().item(), 4)

            decay = torch.sigmoid(mg.decay_logit)
            metrics["mem_decay_mean"] = round(decay.mean().item(), 4)
            metrics["mem_decay_std"] = round(decay.std().item(), 4)

            metrics["mem_prim_norm"] = round(mg.primitives_state.norm().item(), 4)

            # Message magnitude
            mm = mg.msg_magnitude
            metrics["mem_msg_mag_mean"] = round(mm.mean().item(), 4)
            metrics["mem_msg_mag_std"] = round(mm.std().item(), 4)

            # Active neurons
            metrics["mem_usage_frac"] = round(
                (mm.mean(dim=0) > 0.01).float().mean().item(), 4)

            # Hebbian traces
            metrics["mem_hebbian_mean"] = round(mg.hebbian_traces.mean().item(), 4)
            metrics["mem_hebbian_std"] = round(mg.hebbian_traces.std().item(), 4)
            metrics["mem_hebbian_max"] = round(mg.hebbian_traces.max().item(), 4)

            # LM coupling
            lm = self.model.lm
            gate = torch.sigmoid(lm.mem_gate)
            metrics["mem_gate_mean"] = round(gate.mean().item(), 4)
            metrics["mem_gate_min"] = round(gate.min().item(), 4)
            metrics["mem_gate_max"] = round(gate.max().item(), 4)

            # Modulator param stats
            metrics["mod_w1_norm"] = round(mg.mod_w1.norm().item(), 4)
            metrics["mod_w2_norm"] = round(mg.mod_w2.norm().item(), 4)

            # Per-step MLP param stats
            metrics["state_mlp_w1_norm"] = round(mg.state_w1.norm().item(), 4)
            metrics["state_mlp_w2_norm"] = round(mg.state_w2.norm().item(), 4)
            metrics["msg_mlp_w1_norm"] = round(mg.msg_w1.norm().item(), 4)
            metrics["msg_mlp_w2_norm"] = round(mg.msg_w2.norm().item(), 4)

            # Neuron ID
            metrics["neuron_id_norm"] = round(mg.neuron_id.norm().item(), 4)
            metrics["neuron_id_std"] = round(mg.neuron_id.std().item(), 4)

            # Gradient norms (from last backward, if available)
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
                "h_norm_per_neuron": mg.h.norm(dim=-1).mean(dim=0).cpu(),
                "msg_norm_per_neuron": mg.prev_messages.norm(dim=-1).mean(dim=0).cpu(),
                "decay_per_neuron": torch.sigmoid(mg.decay_logit).mean(dim=0).cpu(),
                "msg_magnitude_per_neuron": mg.msg_magnitude.mean(dim=0).cpu(),
                "w_conn_mean_per_neuron": torch.sigmoid(mg.w_conn).mean(dim=(0, 2)).cpu(),
                "prim_norm_per_neuron": mg.primitives_state.norm(dim=-1).mean(dim=0).cpu(),
                "hebbian_mean_per_neuron": mg.hebbian_traces.mean(dim=0).cpu(),
                "neuron_id_norms": mg.neuron_id.norm(dim=-1).cpu(),
                "conn_indices": mg.conn_indices.cpu(),
                "config": {
                    "N": mg.config.N_neurons,
                    "K": mg.config.K_connections,
                    "D": mg.config.D_neuron,
                    "C_mem": mg.config.C_mem,
                },
            }

        torch.save(snapshot, snap_path)
