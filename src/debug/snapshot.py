"""
Full model state dump from checkpoint for deep debugging.

Usage:
    python -m src.debug.snapshot checkpoints/neuromorphic_a_A_step5000.pt

Outputs to snapshot_dir/:
    config.json         — model configuration
    param_summary.json  — per-parameter name, shape, norm, grad_norm
    gate_weights.pt     — gate_a/gate_b weight matrices per layer
    embedding_norms.pt  — per-token embedding norms
    pm_state/           — pm_K, pm_V, pm_a per block/layer
    em_state/           — em_K, em_V, em_S per block
    wm_state.pt         — wm_K, wm_V, wm_valid, wm_ptr
    hidden_states.pt    — per-layer h tensors
"""

import json
import os
import sys

import torch

# ============================================================================
# Configuration — edit these directly
# ============================================================================
CHECKPOINT_PATH = "checkpoints/neuromorphic_a_A_step5000.pt"
OUTPUT_DIR = "checkpoints/snapshot"
# ============================================================================


def main():
    ckpt_path = CHECKPOINT_PATH
    output_dir = OUTPUT_DIR
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    os.makedirs(output_dir, exist_ok=True)

    state_dict = ckpt.get("model_state_dict", ckpt)
    config = ckpt.get("config", None)

    # 1. Config dump
    if config is not None:
        config_dict = config.__dict__ if hasattr(config, "__dict__") else {}
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({k: v for k, v in config_dict.items()
                       if not k.startswith("_")}, f, indent=2, default=str)
        print(f"  config.json ({len(config_dict)} keys)")

    # 2. Parameter summary
    param_summary = []
    for name, tensor in state_dict.items():
        entry = {
            "name": name,
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "norm": tensor.float().norm().item(),
            "mean": tensor.float().mean().item(),
            "std": tensor.float().std().item(),
            "min": tensor.float().min().item(),
            "max": tensor.float().max().item(),
        }
        param_summary.append(entry)

    summary_path = os.path.join(output_dir, "param_summary.json")
    with open(summary_path, "w") as f:
        json.dump(param_summary, f, indent=2)
    print(f"  param_summary.json ({len(param_summary)} params)")

    # 3. Gate weights
    gate_tensors = {}
    for name, tensor in state_dict.items():
        if "gate_a" in name or "gate_b" in name:
            gate_tensors[name] = tensor
    if gate_tensors:
        gate_path = os.path.join(output_dir, "gate_weights.pt")
        torch.save(gate_tensors, gate_path)
        print(f"  gate_weights.pt ({len(gate_tensors)} tensors)")

    # 4. Embedding norms
    emb_keys = [k for k in state_dict if "embedding" in k and "weight" in k]
    if emb_keys:
        emb = state_dict[emb_keys[0]]
        emb_norms = emb.float().norm(dim=-1)  # [vocab_size]
        emb_path = os.path.join(output_dir, "embedding_norms.pt")
        torch.save({"norms": emb_norms, "shape": list(emb.shape)}, emb_path)
        print(f"  embedding_norms.pt (vocab={emb.shape[0]})")

    # 5. PM state
    pm_dir = os.path.join(output_dir, "pm_state")
    os.makedirs(pm_dir, exist_ok=True)
    pm_tensors = {}
    for name, tensor in state_dict.items():
        if any(k in name for k in ["pm_K", "pm_V", "pm_a", "elig_K", "elig_V"]):
            pm_tensors[name] = tensor
    if pm_tensors:
        torch.save(pm_tensors, os.path.join(pm_dir, "pm_all.pt"))
        print(f"  pm_state/ ({len(pm_tensors)} tensors)")

    # 6. EM state
    em_dir = os.path.join(output_dir, "em_state")
    os.makedirs(em_dir, exist_ok=True)
    em_tensors = {}
    for name, tensor in state_dict.items():
        if any(k in name for k in ["em_K", "em_V", "em_S"]):
            em_tensors[name] = tensor
    if em_tensors:
        torch.save(em_tensors, os.path.join(em_dir, "em_all.pt"))
        print(f"  em_state/ ({len(em_tensors)} tensors)")

    # 7. WM state
    wm_tensors = {}
    for name, tensor in state_dict.items():
        if any(k in name for k in ["wm_K", "wm_V", "wm_valid", "wm_ptr"]):
            wm_tensors[name] = tensor
    if wm_tensors:
        wm_path = os.path.join(output_dir, "wm_state.pt")
        torch.save(wm_tensors, wm_path)
        print(f"  wm_state.pt ({len(wm_tensors)} tensors)")

    # 8. Hidden states
    hidden_tensors = {}
    for name, tensor in state_dict.items():
        # Match layer hidden states (e.g., blocks.0.layers.0.h)
        if name.endswith(".h") and "layers" in name:
            hidden_tensors[name] = tensor
    if hidden_tensors:
        h_path = os.path.join(output_dir, "hidden_states.pt")
        torch.save(hidden_tensors, h_path)
        print(f"  hidden_states.pt ({len(hidden_tensors)} tensors)")

    print(f"\nSnapshot saved to: {output_dir}")


if __name__ == "__main__":
    main()
