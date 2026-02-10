"""
Loss utilities for neuromorphic LM training.

Online cross-entropy (never materializes [BS, T, vocab]) and regularizers.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def online_cross_entropy(logits: Tensor, targets: Tensor,
                         loss_mask: Tensor) -> tuple:
    """Per-token CE loss with masking. Operates on single timestep.

    Args:
        logits: [BS, vocab] — model output for one timestep
        targets: [BS] — target token ids
        loss_mask: [BS] bool — True for valid positions

    Returns:
        loss_sum: scalar — sum of CE losses for valid positions
        valid_count: int — number of valid positions
    """
    valid_count = loss_mask.sum().item()
    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0

    loss = F.cross_entropy(
        logits[loss_mask], targets[loss_mask], reduction="sum"
    )
    return loss, int(valid_count)


def batched_cross_entropy(logits_all: Tensor, targets_all: Tensor,
                          loss_mask_all: Tensor) -> tuple:
    """Batched CE loss over a span of P tokens.

    Args:
        logits_all: [BS, P, vocab] — model outputs for all span tokens
        targets_all: [BS, P] — target token ids
        loss_mask_all: [BS, P] bool — True for valid positions

    Returns:
        loss_sum: scalar — sum of CE losses for valid positions
        valid_count: int — number of valid positions
    """
    BS, P, V = logits_all.shape
    flat_logits = logits_all.reshape(BS * P, V)
    flat_targets = targets_all.reshape(BS * P)
    flat_mask = loss_mask_all.reshape(BS * P)

    valid_count = flat_mask.sum().item()
    if valid_count == 0:
        return torch.tensor(0.0, device=logits_all.device, requires_grad=True), 0

    loss = F.cross_entropy(
        flat_logits[flat_mask], flat_targets[flat_mask], reduction="sum"
    )
    return loss, int(valid_count)


def compute_regularizers(model) -> Tensor:
    """Compute PM budget penalty, EM budget penalty, etc.

    Returns scalar regularization loss.
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)

    # PM budget penalty: penalize if sum(pm_a) approaches budget
    if model.config.pm_enabled:
        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm
                if pm.pm_a is not None:
                    usage = pm.pm_a.sum(dim=-1)  # [BS]
                    excess = F.relu(usage - model.config.budget_pm * 0.9)
                    reg = reg + excess.mean() * 0.01

    # EM budget penalty
    if model.config.em_enabled:
        for block in model.blocks:
            em = block.em
            if em.em_S is not None:
                usage = em.em_S.sum(dim=-1)  # [BS]
                excess = F.relu(usage - model.config.budget_em * 0.9)
                reg = reg + excess.mean() * 0.01

    return reg
