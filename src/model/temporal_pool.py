"""
Temporal pooling for multi-timescale block processing.

TemporalPooler applies causal depthwise conv1d pooling with stride to reduce
sequence length from P to P//scale. Upsampling uses repeat_interleave.

Causality is ensured by left-padding with (scale-1) zeros before the conv,
so output position j only depends on input positions <= j*scale.

Free functions provide non-parameterized causal average pooling (for
D-dimensional signals like y_wm, x_proj) and min-pooling for carry masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalPooler(nn.Module):
    """Causal depthwise conv1d pooling with learnable weights.

    Scale=1 is identity (no-op). Scale>1 applies a depthwise convolution
    with kernel_size=scale, stride=scale, initialized as average pooling.
    """

    def __init__(self, D_h: int, scale: int):
        super().__init__()
        self.scale = scale
        self.D_h = D_h
        if scale > 1:
            self.pool_conv = nn.Conv1d(
                D_h, D_h, kernel_size=scale, stride=scale,
                groups=D_h, bias=False,
            )
            nn.init.constant_(self.pool_conv.weight, 1.0 / scale)

    def downsample(self, x: Tensor) -> Tensor:
        """x: [BS, P, D_h] → [BS, P//scale, D_h]. Strictly causal."""
        if self.scale == 1:
            return x
        # Transpose to [BS, D_h, P] for conv1d
        x_t = x.transpose(1, 2)
        # Causal left-pad: prepend (scale-1) zeros
        x_t = F.pad(x_t, (self.scale - 1, 0))
        # Conv with stride reduces to P//scale
        out = self.pool_conv(x_t)
        return out.transpose(1, 2)  # [BS, P//scale, D_h]

    def upsample(self, x: Tensor, target_len: int) -> Tensor:
        """x: [BS, P//scale, D_h] → [BS, target_len, D_h]."""
        if self.scale == 1:
            return x
        return x.repeat_interleave(self.scale, dim=1)[:, :target_len]


def causal_avg_pool(x: Tensor, scale: int) -> Tensor:
    """Non-parameterized causal average pooling for arbitrary-dim tensors.

    x: [BS, P, D] → [BS, P//scale, D]. Strictly causal via left-padding.
    """
    if scale == 1:
        return x
    x_t = x.transpose(1, 2)                           # [BS, D, P]
    x_t = F.pad(x_t, (scale - 1, 0))                  # causal left-pad
    out = F.avg_pool1d(x_t, kernel_size=scale, stride=scale)
    return out.transpose(1, 2)                         # [BS, P//scale, D]


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
