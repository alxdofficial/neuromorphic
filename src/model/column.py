"""
Cortical Column Group (v4).

C cortical columns batched via GroupedLinear (einsum over group dim).
Each column: LayerNorm -> FFN with residual, PM/EM projections, PCM.
No Python loop over columns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .predictive_coding import GroupedLinear, GroupedLayerNorm, CrossPassPCM
from .utils import unit_normalize


class CorticalColumnGroup(nn.Module):
    """C cortical columns batched via GroupedLinear.

    Processes all N tokens for all C columns simultaneously.
    Each column: LayerNorm -> FFN (D_col -> 4*D_col -> D_col) with residual.
    PM/EM projections for read and write candidates.
    PCM for cross-pass prediction.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        C = config.C
        D_col = config.D_col
        D_mem = config.D_mem
        D_pcm = config.D_pcm
        self.config = config

        # FFN (two grouped linears + activation)
        self.ffn_norm = GroupedLayerNorm(C, D_col)
        self.ffn_up = GroupedLinear(C, D_col, D_col * config.ffn_expansion)
        self.ffn_down = GroupedLinear(C, D_col * config.ffn_expansion, D_col)

        # PM projections (D_col <-> D_mem)
        self.W_pm_up = GroupedLinear(C, D_col, D_mem)
        self.W_pm_down = GroupedLinear(C, D_mem, D_col)

        # EM projections
        self.W_em_up = GroupedLinear(C, D_col, D_mem)
        self.W_em_down = GroupedLinear(C, D_mem, D_col)

        # PM eligibility candidate projections
        self.W_k_pre = GroupedLinear(C, D_col, D_mem)
        self.W_v_post = GroupedLinear(C, D_col, D_mem)

        # EM candidate projections
        self.W_k_cand = GroupedLinear(C, D_col, D_mem)
        self.W_v_cand = GroupedLinear(C, D_col, D_mem)
        self.W_nov = GroupedLinear(C, D_col, 1)

        # PCM
        self.pcm = CrossPassPCM(C, D_col, D_pcm) if config.pcm_enabled else None

        # FFN gain from PCM (zero-init so gain starts at 1.0)
        if config.pcm_enabled:
            self.W_gain = GroupedLinear(C, D_pcm, D_col)
            nn.init.zeros_(self.W_gain.weight)
            if self.W_gain.bias is not None:
                nn.init.zeros_(self.W_gain.bias)

    def forward(self, x_col, pm_state, em_state, z_hat_prev):
        """
        x_col:      [BS, N, C, D_col]
        pm_state:   ProceduralMemory — block's PM
        em_state:   EpisodicMemory — block's EM
        z_hat_prev: [BS, N, C, D_pcm] or None

        Returns: x_out, z, z_hat, surprise, elig_info, novelty_info
        """
        # 1. PM read (holographic, in D_mem space)
        if self.config.pm_enabled and pm_state.is_initialized():
            q_pm = self.W_pm_up(x_col)                     # [BS,N,C,D_mem]
            y_pm = pm_state.read(q_pm)                      # [BS,N,C,D_mem]
            x_col = x_col + self.W_pm_down(y_pm)            # residual

        # 2. EM read (top-k retrieval)
        if self.config.em_enabled and em_state.is_initialized():
            q_em = self.W_em_up(x_col)                      # [BS,N,C,D_mem]
            y_em = em_state.read(q_em)                       # [BS,N,C,D_mem]
            x_col = x_col + self.W_em_down(y_em)            # residual

        # 3. PCM encode + surprise
        if self.pcm is not None:
            z = self.pcm.encode(x_col)                      # [BS,N,C,D_pcm]
            surprise, delta = self.pcm.compute_surprise(z, z_hat_prev)
            gain = 1.0 + 0.1 * torch.tanh(self.W_gain(delta))  # [BS,N,C,D_col]
        else:
            z = None
            surprise = torch.zeros(
                x_col.shape[:-1], device=x_col.device, dtype=x_col.dtype
            )
            delta = None
            gain = None

        # 4. FFN with gain modulation
        h = self.ffn_norm(x_col)
        if gain is not None:
            h = h * gain
        h = self.ffn_down(F.gelu(self.ffn_up(h)))
        x_out = x_col + h                                   # residual

        # 5. PCM hypothesis (predict next pass)
        if self.pcm is not None:
            z_hat = self.pcm.predict(z)                     # [BS,N,C,D_pcm]
        else:
            z_hat = None

        # 6. PM eligibility accumulation info
        k_cand = unit_normalize(self.W_k_pre(x_out))       # [BS,N,C,D_mem]
        v_cand = self.W_v_post(x_out)                      # [BS,N,C,D_mem]
        gate = (surprise / self.config.surprise_scale).clamp(0, 1)  # [BS,N,C]

        # 7. EM novelty scoring info
        q_nov = unit_normalize(self.W_k_cand(x_out))       # [BS,N,C,D_mem]
        v_nov = self.W_v_cand(x_out)                       # [BS,N,C,D_mem]
        w_nov = torch.sigmoid(self.W_nov(x_out).squeeze(-1))  # [BS,N,C]

        elig_info = (k_cand, v_cand, gate)
        novelty_info = (q_nov, v_nov, w_nov, surprise)

        return x_out, z, z_hat, surprise, elig_info, novelty_info
