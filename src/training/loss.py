"""
Loss utilities for neuromorphic LM training (v5).

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
    """Batched CE loss over a segment of N tokens.

    Args:
        logits_all: [BS, N, vocab] — model outputs
        targets_all: [BS, N] — target token ids
        loss_mask_all: [BS, N] bool — True for valid positions

    Returns:
        loss_sum: scalar — sum of CE losses for valid positions
        valid_count: GPU tensor — number of valid positions
    """
    BS, N, V = logits_all.shape
    mask = loss_mask_all.reshape(BS * N)
    targets = torch.where(mask, targets_all.reshape(BS * N),
                          torch.tensor(-100, device=targets_all.device, dtype=targets_all.dtype))
    loss = F.cross_entropy(
        logits_all.reshape(BS * N, V), targets,
        ignore_index=-100, reduction="sum",
    )
    return loss, mask.sum()


def compute_loss_and_surprise(
    logits_all: Tensor, targets_all: Tensor, loss_mask_all: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Combined loss + surprise via a single fused cross_entropy kernel.

    Returns:
        loss_sum: scalar with grad — sum of NLL over valid positions
        valid_count: GPU tensor — number of valid positions
        token_surprise: [BS, N, 1] detached — per-token surprise
    """
    BS, N, V = logits_all.shape
    per_token_ce = F.cross_entropy(
        logits_all.reshape(BS * N, V),
        targets_all.reshape(BS * N),
        reduction="none",
    ).reshape(BS, N)

    mask_f = loss_mask_all.float()
    valid_count = loss_mask_all.sum()
    loss = (per_token_ce * mask_f).sum()
    token_surprise = (per_token_ce.detach() * mask_f).unsqueeze(-1)

    return loss, valid_count, token_surprise


def compute_regularizers(model) -> Tensor:
    """Compute PM bias norm penalty, EM budget penalty.

    Returns scalar regularization loss.
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)

    if model.config.pm_enabled:
        pm = model.pm
        if pm.pm_bias is not None:
            # Penalize large PM bias norms
            bias_norm = pm.pm_bias.norm(dim=-1)  # [BS, B]
            excess = F.relu(bias_norm - model.config.budget_pm * 0.9)
            reg = reg + excess.mean() * 0.01

    if model.config.em_enabled:
        em = model.em
        if em.em_S is not None:
            usage = em.em_S.sum(dim=-1)  # [BS, B]
            excess = F.relu(usage - model.config.budget_em * 0.9)
            reg = reg + excess.mean() * 0.01

    return reg
