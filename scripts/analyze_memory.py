"""Deep analysis of memory graph behavior on a single chunk.

Loads a checkpoint, runs one chunk, captures per-token neuron dynamics,
and generates detailed visualizations.

Usage:
    python -m scripts.analyze_memory outputs/v8/<run_id>/v8_step5000.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from src.v8.config import V8Config
from src.v8.model import V8Model


def load_model(ckpt_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    config.validate()

    # Detect obs_dim from checkpoint to handle version differences
    neuromod_sd = ckpt["neuromod_state_dict"]
    ckpt_obs_dim = neuromod_sd["backbone.0.weight"].shape[1]

    # Override obs_dim if it differs from current code
    class PatchedConfig(type(config)):
        pass
    patched = config
    original_obs_dim = config.D_mem * 3 + 3
    if ckpt_obs_dim != original_obs_dim:
        # Monkey-patch the model to use checkpoint's obs_dim
        pass  # handled by strict=False below

    model = V8Model(config)
    if device.type == "cuda":
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device)

    # Load with strict=False to handle architecture changes between versions
    model.lm.load_state_dict(ckpt["model_state_dict"], strict=False)
    # For neuromod, only load if dimensions match
    try:
        model.neuromod.load_state_dict(neuromod_sd, strict=False)
    except RuntimeError:
        print(f"  Warning: neuromod weights incompatible (obs_dim {ckpt_obs_dim} vs {original_obs_dim}), using random init")
    if "memory_graph_state" in ckpt:
        model.initialize_states(1, device)
        model.memory.load_state_dict(ckpt["memory_graph_state"])
    return model, config


def capture_chunk(model, input_ids, device):
    """Run one chunk and capture per-token neuron dynamics."""
    model.eval()
    config = model.config
    BS, T = input_ids.shape
    C = config.C
    D_mem = config.D_mem
    N = config.N_neurons
    action_every = config.action_every

    model.initialize_states(BS, device)

    with torch.no_grad():
        H_mid, x, surprise, aux_loss = model.lm.forward_scan_lower(input_ids)

        model._ensure_memory(BS, device, next(model.lm.parameters()).dtype)
        cc_signals_all = H_mid.detach().view(BS, T, C, D_mem)
        n_segments = T // action_every
        cc_segments = cc_signals_all.view(BS, n_segments, action_every, C, D_mem)

        all_h = torch.zeros(BS, T, N, D_mem)
        all_msg = torch.zeros(BS, T, N, D_mem)
        all_received = torch.zeros(BS, T, N, D_mem)
        all_activation = torch.zeros(BS, T, N)
        all_fired = torch.zeros(BS, T, N, dtype=torch.bool)
        neuromod_actions = []

        mg = model._mem_graph
        A = mg._build_adjacency()

        for seg in range(n_segments):
            obs = mg.get_neuron_obs()
            obs_flat = obs.reshape(BS * N, -1)
            action, _, _, _ = model.neuromod.get_action_and_value(obs_flat)
            neuromod_actions.append(action.detach().cpu())
            model._apply_neuromod_action(action, BS)
            A = mg._build_adjacency()

            decay = torch.sigmoid(mg.decay_logit).unsqueeze(-1)
            one_minus_decay = 1.0 - decay

            for t_local in range(action_every):
                t_global = seg * action_every + t_local

                received = torch.bmm(A, mg.prev_messages)
                received[:, :C] = received[:, :C] + cc_segments[:, seg, t_local]
                mg.h = decay * mg.h + one_minus_decay * received
                mg.prev_messages = torch.tanh(mg.h * mg.primitives)

                all_h[:, t_global] = mg.h.cpu().float()
                all_msg[:, t_global] = mg.prev_messages.cpu().float()
                all_received[:, t_global] = received.cpu().float()
                act_mag = mg.prev_messages.float().norm(dim=-1)
                all_activation[:, t_global] = act_mag.cpu()

                threshold = mg.activation_ema + mg.activation_std_ema
                all_fired[:, t_global] = (act_mag > threshold).cpu()

        mem_signals = all_msg[:, :, :C].to(device=device, dtype=H_mid.dtype)
        H_enriched = model.lm.inject_memory(H_mid, mem_signals)
        H_final = model.lm.forward_scan_upper(H_enriched)
        logits = model.lm.forward_output(H_final)

    return {
        "h": all_h[0],
        "messages": all_msg[0],
        "received": all_received[0],
        "activation": all_activation[0],
        "fired": all_fired[0],
        "logits": logits[0].cpu().float(),
        "cc_signals": cc_signals_all[0].cpu().float(),
        "surprise": surprise[0].cpu().float(),
        "primitives": mg.primitives[0].cpu().float(),
        "conn_weights": mg.conn_weights[0].cpu().float(),
        "conn_indices": mg.conn_indices.cpu(),
        "decay": torch.sigmoid(mg.decay_logit[0]).cpu().float(),
        "co_activation": mg.co_activation_ema.cpu().float(),
        "neuromod_actions": neuromod_actions,
        "config": config,
    }


def plot_neuron_activity_heatmap(data, output_path):
    """[N x T] heatmap showing activation magnitude per token."""
    act = data["activation"].numpy()
    T, N = act.shape
    C = data["config"].C

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(act.T, aspect='auto', cmap='hot', interpolation='nearest',
                   vmin=0, vmax=np.percentile(act, 95))
    ax.axhline(y=C - 0.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7,
               label='Port neurons boundary')
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Neuron Activation Magnitude (per-token)")
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label="Activation norm")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_firing_raster(data, output_path):
    """Raster plot of binary firing events."""
    fired = data["fired"].numpy().T
    C = data["config"].C

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.spy(fired, aspect='auto', markersize=0.3, color='black')
    ax.axhline(y=C - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Binary Firing Raster (dot = neuron fired)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_co_activation_matrix(data, output_path):
    """Phi coefficient matrix as heatmap with spectral reordering."""
    phi = data["co_activation"].numpy()
    N = phi.shape[0]

    try:
        from scipy.sparse.csgraph import laplacian
        from scipy.linalg import eigh
        L = laplacian(np.abs(phi) + 1e-10, normed=True)
        _, vecs = eigh(L, subset_by_index=[1, 3])
        order = np.argsort(vecs[:, 0])
        phi_sorted = phi[order][:, order]
    except Exception:
        phi_sorted = phi
        order = np.arange(N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax = max(0.1, np.percentile(np.abs(phi), 95))

    im0 = axes[0].imshow(phi, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title("Co-Activation (phi) — Original Order")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(phi_sorted, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_title("Co-Activation (phi) — Spectral Sorted")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_primitive_pca(data, output_path):
    """PCA of primitive vectors showing neuron specialization."""
    prims = data["primitives"].numpy()
    N, D = prims.shape
    C = data["config"].C
    firing_rate = data["activation"].mean(dim=0).numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(prims)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = ['red' if i < C else 'steelblue' for i in range(N)]
    sizes = [30 if i < C else 5 for i in range(N)]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, alpha=0.6)
    ax.set_title(f"Primitive PCA (red=port, blue=non-port)\n"
                 f"Var explained: {pca.explained_variance_ratio_.sum():.1%}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=firing_rate, s=8,
                    cmap='viridis', alpha=0.7)
    ax.scatter(coords[:C, 0], coords[:C, 1], c='red', s=30,
               marker='*', label='Port neurons', zorder=5)
    plt.colorbar(sc, ax=ax, label="Mean activation")
    ax.set_title("Primitive PCA — colored by activity level")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_signal_flow(data, output_path):
    """Signal magnitude: port vs non-port, plus decay distribution."""
    activation = data["activation"].numpy()
    C = data["config"].C

    port_act = activation[:, :C].mean(axis=1)
    nonport_act = activation[:, C:].mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    ax = axes[0]
    ax.plot(port_act, label=f"Port neurons (0-{C-1})", alpha=0.8, linewidth=0.5)
    ax.plot(nonport_act, label=f"Non-port neurons", alpha=0.8, linewidth=0.5)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean activation norm")
    ax.set_title("Signal Flow: Port vs Non-Port Neurons")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    decay = data["decay"].numpy()
    ax.hist(decay, bins=50, alpha=0.7, color='steelblue')
    ax.set_xlabel("Decay value (sigmoid)")
    ax.set_ylabel("# neurons")
    ax.set_title(f"Decay Distribution (mean={decay.mean():.4f}, std={decay.std():.4f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_connection_weights(data, output_path):
    """Connection weight distribution and structure."""
    cw = data["conn_weights"].numpy()
    N, K = cw.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(cw.flatten(), bins=100, alpha=0.7)
    ax.set_title(f"Connection Weight Distribution\n"
                 f"(mean={cw.mean():.4f}, std={cw.std():.4f})")
    ax.set_xlabel("Weight value")
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    ax = axes[1]
    w_abs = np.abs(cw)
    w_norm = w_abs / (w_abs.sum(axis=1, keepdims=True) + 1e-8)
    entropy = -(w_norm * np.log(w_norm + 1e-8)).sum(axis=1)
    ax.hist(entropy, bins=50, alpha=0.7)
    ax.set_title("Routing Entropy per Neuron\n(high=uniform, low=concentrated)")
    ax.set_xlabel("Entropy")

    ax = axes[2]
    pct = np.percentile(np.abs(cw), 95)
    im = ax.imshow(cw[:64], aspect='auto', cmap='RdBu_r', vmin=-pct, vmax=pct)
    ax.set_title("Weights (neurons 0-63)")
    ax.set_xlabel(f"Connection (K={K})")
    ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Deep analysis of memory graph")
    p.add_argument("checkpoint", help="Path to checkpoint .pt file")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")

    model, config = load_model(args.checkpoint, device)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint), "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("Running one chunk through model...")
    input_ids = torch.randint(0, config.vocab_size, (1, config.T), device=device)
    data = capture_chunk(model, input_ids, device)

    print(f"\nGenerating visualizations to {output_dir}/")
    plot_neuron_activity_heatmap(data, os.path.join(output_dir, "activity_heatmap.png"))
    plot_firing_raster(data, os.path.join(output_dir, "firing_raster.png"))
    plot_co_activation_matrix(data, os.path.join(output_dir, "co_activation.png"))
    plot_primitive_pca(data, os.path.join(output_dir, "primitive_pca.png"))
    plot_signal_flow(data, os.path.join(output_dir, "signal_flow.png"))
    plot_connection_weights(data, os.path.join(output_dir, "connection_weights.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
