"""Optimizer construction with two LR groups.

Per plan §4.4:
- Memory-side params (concept_ids, concept_states, state_init, read+write
  modules, read_attn) at lr_memory (default 3e-4 for Phase 1, 1e-4 for Phase 2).
- Llama-side adapter params (W_in, W_out, scale, MemInjectLayer cross-attn)
  at lr_adapter (default 1e-4 for Phase 1, 5e-5 for Phase 2).
- Llama backbone is frozen — no group needed.
"""

from __future__ import annotations

import torch

from src.trajectory_memory.integrated_lm import IntegratedLM


def build_optimizer(
    model: IntegratedLM,
    *,
    lr_memory: float,
    lr_adapter: float,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with two param groups for the trajectory-memory
    architecture.

    Args:
        model:        IntegratedLM instance.
        lr_memory:    LR for memory-side params (manifold + read + write + read_attn).
        lr_adapter:   LR for Llama-side adapter (W_in / W_out / scale / cross-attn).
        weight_decay: AdamW weight decay (default 0).

    Returns:
        AdamW optimizer. Groups added only if non-empty (adapter group is empty
        in `attach_lm=False` test mode).
    """
    cfg = model.cfg
    memory_params = (
        list(model.manifold.parameters())
        + list(model.read_module.parameters())
        + list(model.write_module.parameters())
        + list(model.read_attn.parameters())
    )
    adapter_params: list = []
    if model.host is not None:
        mem_inject = model.host.layer_list()[cfg.inject_layer]
        adapter_params = [
            p for n, p in mem_inject.named_parameters()
            if p.requires_grad and not n.startswith("orig_layer")
        ]

    groups = []
    if memory_params:
        groups.append({"params": memory_params, "lr": lr_memory})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr_adapter})
    if not groups:
        # Fallback: single group with all trainable params at lr_memory.
        groups.append({
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": lr_memory,
        })

    # N8 — fused AdamW: PyTorch docs note fused implementations are
    # generally faster than foreach, foreach faster than for-loop.
    # AdamW has stable CUDA fused since 2.x. Negligible overhead if
    # `fused` arg not supported on this device (falls back to foreach).
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, weight_decay=weight_decay, fused=fused)
