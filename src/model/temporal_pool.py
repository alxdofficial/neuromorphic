"""
Temporal pooling for multi-timescale block processing.

TemporalPooler uses learnable weighted aggregation within non-overlapping
windows for downsampling and repeat_interleave for upsampling.

When a carry mask is provided, tokens from a previous document within a
window are masked out before aggregation (boundary-aware pooling). This
prevents cross-document mixing while still allowing the model to learn
which positions within a window matter most.

Free function carry_min_pool provides min-pooling for carry masks
(any boundary in the pooling window forces a reset at the pooled position).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalPooler(nn.Module):
    """Learnable temporal pooling for multi-timescale blocks.

    Scale=1 is identity (no-op). Scale>1 uses a learned weighted sum within
    non-overlapping windows of s tokens for downsampling, and repeat_interleave
    for upsampling.

    Weights are softmax-normalized and initialized to uniform (1/s), so at
    init this is equivalent to average pooling. The model learns to weight
    positions within each window as training progresses.

    When carry is provided to downsample(), tokens from a previous document
    within a window are masked out: if carry[k]=0 (boundary at position k),
    all positions before k in that window get zero weight.
    """

    def __init__(self, D_h: int, scale: int):
        super().__init__()
        self.scale = scale
        if scale > 1:
            # Learnable per-position weights within each window.
            # Zeros → softmax → uniform (1/s) → starts as average pooling.
            self.pool_logits = nn.Parameter(torch.zeros(scale))

    def downsample(self, x: Tensor, carry: Tensor = None) -> Tensor:
        """Downsample: [BS, P, D] → [BS, P//scale, D].

        Learnable weighted sum within non-overlapping windows. When carry
        is provided, tokens before doc boundaries within a window are
        masked out before aggregation.

        Args:
            x: [BS, P, D] — input tensor
            carry: [BS, P, 1] — optional carry mask (0 at doc boundaries)
        """
        if self.scale == 1:
            return x
        BS, P, D = x.shape
        s = self.scale

        # Reshape into non-overlapping windows: [BS, P_b, s, D]
        x_win = x.reshape(BS, P // s, s, D)

        # Learnable weights, softmax-normalized
        w = F.softmax(self.pool_logits, dim=0)  # [s]

        if carry is not None:
            c_win = carry.reshape(BS, P // s, s, 1)  # [BS, P_b, s, 1]

            # Build boundary mask within each window.
            # mask[i] = 1 iff there's no boundary between position i and the
            # end of the window. If carry[k]=0 (boundary at k), all positions
            # before k get masked out.
            #
            # mask[i] = product(carry[j] for j in range(i+1, s))
            # Last position always has mask=1 (nothing after it).
            carry_shifted = torch.cat(
                [c_win[:, :, 1:], c_win.new_ones(BS, P // s, 1, 1)], dim=2
            )
            mask = carry_shifted.flip(2).cumprod(dim=2).flip(2)  # [BS, P_b, s, 1]

            # Masked weights, renormalized so they still sum to 1
            w_eff = w.view(1, 1, s, 1) * mask
            w_eff = w_eff / w_eff.sum(dim=2, keepdim=True).clamp(min=1e-8)
            return (x_win * w_eff).sum(dim=2)  # [BS, P_b, D]
        else:
            return (x_win * w.view(1, 1, s, 1)).sum(dim=2)  # [BS, P_b, D]

    def upsample(self, x: Tensor, target_len: int) -> Tensor:
        """x: [BS, P//scale, D] → [BS, target_len, D]."""
        if self.scale == 1:
            return x
        return x.repeat_interleave(self.scale, dim=1)[:, :target_len]


def carry_min_pool(carry: Tensor, scale: int) -> Tensor:
    """Min-pool carry mask: any boundary in the window forces reset.

    carry: [BS, P, 1] (0 at boundaries, 1 elsewhere)
    Returns: [BS, P//scale, 1]
    """
    if scale == 1:
        return carry
    # min(x) = -max(-x)
    c_t = carry.transpose(1, 2)                        # [BS, 1, P]
    # Left-pad with 1.0 (no false boundaries from padding)
    c_t = F.pad(c_t, (scale - 1, 0), value=1.0)
    out = -F.max_pool1d(-c_t, kernel_size=scale, stride=scale)
    return out.transpose(1, 2)                         # [BS, P//scale, 1]
