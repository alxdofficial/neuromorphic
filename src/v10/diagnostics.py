"""v10-gnn training diagnostics."""

import os
import torch
from torch import Tensor


class V10Diagnostics:
    """Lightweight diagnostics for v10-gnn training."""

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
            # Memory graph state norms
            metrics["mem_h_norm"] = round(mg.h.norm().item(), 4)
            metrics["mem_msg_norm"] = round(mg.messages.norm().item(), 4)

            # Connection weights (sigmoid of logits)
            w_sig = torch.sigmoid(mg.w_conn)
            metrics["mem_w_conn_mean"] = round(w_sig.mean().item(), 4)
            metrics["mem_w_conn_std"] = round(w_sig.std().item(), 4)

            # Neuron diversity: std of per-neuron h norms
            # (high = neurons differentiating, low = homogeneous)
            h_per_neuron = mg.h.norm(dim=-1).mean(dim=0)  # [N]
            metrics["mem_h_diversity"] = round(h_per_neuron.std().item(), 4)

            # Message diversity
            msg_per_neuron = mg.messages.norm(dim=-1).mean(dim=0)  # [N]
            metrics["mem_msg_diversity"] = round(msg_per_neuron.std().item(), 4)

            # Hebbian traces
            metrics["mem_hebbian_mean"] = round(
                mg.hebbian_traces.mean().item(), 4)
            metrics["mem_hebbian_std"] = round(
                mg.hebbian_traces.std().item(), 4)

            # Neuron identity embeddings
            metrics["neuron_id_norm"] = round(mg.identity.norm().item(), 4)
            metrics["neuron_id_std"] = round(mg.identity.std().item(), 4)

            # Per-component gradient norms (from last backward)
            ns = mg.neuron_step
            _has_grads = (hasattr(ns, 'state_mlp')
                          and ns.state_mlp[0].weight.grad is not None)
            if _has_grads:
                # State MLP grads
                state_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in ns.state_mlp.parameters()
                    if p.grad is not None) ** 0.5
                metrics["grad_state_mlp"] = round(state_grad, 6)

                # Message MLP grads
                msg_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in ns.msg_mlp.parameters()
                    if p.grad is not None) ** 0.5
                metrics["grad_msg_mlp"] = round(msg_grad, 6)

                # Modulator MLP grads
                mod_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in ns.mod_mlp.parameters()
                    if p.grad is not None) ** 0.5
                metrics["grad_mod_mlp"] = round(mod_grad, 6)

                # Identity embedding grads
                if mg.identity.grad is not None:
                    metrics["grad_identity"] = round(
                        mg.identity.grad.norm().item(), 6)

                # Decoder grads
                dec_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in self.model.decoder.parameters()
                    if p.grad is not None) ** 0.5
                metrics["grad_decoder"] = round(dec_grad, 6)

            # Structural plasticity swaps
            if hasattr(mg, '_last_rewire_swaps'):
                metrics["plasticity_swaps"] = mg._last_rewire_swaps

            # PCM surprise stats
            lm = self.model.lm
            pcm_stats = getattr(lm, '_pcm_stats', None)
            if pcm_stats is not None:
                metrics["pcm_surprise_mean"] = round(
                    pcm_stats["surprise_mean"], 4)
                metrics["pcm_surprise_std"] = round(
                    pcm_stats["surprise_std"], 4)

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
                "msg_norm_per_neuron":
                    mg.messages.norm(dim=-1).mean(dim=0).cpu(),
                "w_conn_mean_per_neuron":
                    torch.sigmoid(mg.w_conn).mean(dim=(0, 2)).cpu(),
                "hebbian_mean_per_neuron":
                    mg.hebbian_traces.mean(dim=0).cpu(),
                "identity_norms": mg.identity.norm(dim=-1).cpu(),
                "conn_indices": mg.conn_indices.cpu(),
                "config": {
                    "N": mg.config.N_neurons,
                    "K": mg.config.K_connections,
                    "D": mg.config.D_neuron,
                    "D_id": mg.config.D_id,
                    "num_words": mg.config.num_words,
                },
            }

        torch.save(snapshot, snap_path)
