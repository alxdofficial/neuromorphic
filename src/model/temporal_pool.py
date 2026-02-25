"""
Temporal pooling for multi-timescale block processing.

TemporalPooler uses strided sampling (every s-th token) for downsampling
and repeat_interleave for upsampling. This avoids token mixing across
doc boundaries that would occur with causal conv or avg pooling.

Free function carry_min_pool provides min-pooling for carry masks
(any boundary in the pooling window forces a reset at the pooled position).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalPooler(nn.Module):
    """Strided temporal pooling for multi-timescale blocks.

    Scale=1 is identity (no-op). Scale>1 takes every s-th token (strided
    sampling) for downsampling and uses repeat_interleave for upsampling.

    Strided sampling is used instead of causal conv pooling to avoid
    mixing tokens across doc boundaries within the pooling window. This
    also simplifies step()-mode inference (just "update every s tokens").
    """

    def __init__(self, D_h: int, scale: int):
        super().__init__()
        self.scale = scale

    def downsample(self, x: Tensor) -> Tensor:
        """x: [BS, P, D] → [BS, P//scale, D]. Strided sampling."""
        if self.scale == 1:
            return x
        return x[:, ::self.scale]

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
