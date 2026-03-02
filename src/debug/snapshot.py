"""
Full model state dump from checkpoint for deep debugging.

Usage:
    python -m src.debug.snapshot checkpoints/neuromorphic_a_A_step5000.pt

Outputs to snapshot_dir/:
    config.json         — model configuration
    param_summary.json  — per-parameter name, shape, norm, grad_norm
    embedding_norms.pt  — per-token embedding norms
    pm_state/           — pm_bias
    em_state/           — em_K, em_V, em_S
"""

import json
import os
import sys

import torch

from ..model import NeuromorphicLM
from ..model.state import load_runtime_state

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
    runtime_state = ckpt.get("runtime_state", None)

    model = None
    if config is not None:
        model = NeuromorphicLM(config)
        model.load_state_dict(state_dict, strict=False)
        if runtime_state is not None:
            load_runtime_state(model, runtime_state)

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

    # 3. Scan layer weights (proj_in contains decay/gate parameters)
    scan_tensors = {}
    for name, tensor in state_dict.items():
        if "stage1." in name or "stage3." in name:
            scan_tensors[name] = tensor
    if scan_tensors:
        scan_path = os.path.join(output_dir, "scan_weights.pt")
        torch.save(scan_tensors, scan_path)
        print(f"  scan_weights.pt ({len(scan_tensors)} tensors)")

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
    if model is not None:
        pm = model.pm
        for k in ["pm_bias"]:
            t = getattr(pm, k, None)
            if t is not None:
                pm_tensors[f"pm.{k}"] = t.detach().cpu()
    else:
        for name, tensor in state_dict.items():
            if "pm_bias" in name:
                pm_tensors[name] = tensor
    if pm_tensors:
        torch.save(pm_tensors, os.path.join(pm_dir, "pm_all.pt"))
        print(f"  pm_state/ ({len(pm_tensors)} tensors)")

    # 6. EM state
    em_dir = os.path.join(output_dir, "em_state")
    os.makedirs(em_dir, exist_ok=True)
    em_tensors = {}
    if model is not None:
        em = model.em
        for k in ["em_K", "em_V", "em_S"]:
            t = getattr(em, k, None)
            if t is not None:
                em_tensors[f"em.{k}"] = t.detach().cpu()
    else:
        for name, tensor in state_dict.items():
            if any(k in name for k in ["em_K", "em_V", "em_S"]):
                em_tensors[name] = tensor
    if em_tensors:
        torch.save(em_tensors, os.path.join(em_dir, "em_all.pt"))
        print(f"  em_state/ ({len(em_tensors)} tensors)")

    print(f"\nSnapshot saved to: {output_dir}")


if __name__ == "__main__":
    main()
