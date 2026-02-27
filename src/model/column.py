"""
Cortical Column Group (v4).

G = B_blocks * C cortical columns batched via GroupedLinear (einsum over
group dim). Each column: LayerNorm -> FFN with residual, PM/EM projections,
PCM.  No Python loop over columns or blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .predictive_coding import GroupedLinear, GroupedLayerNorm, CrossPassPCM
from .utils import unit_normalize


class CorticalColumnGroup(nn.Module):
    """G = B*C cortical columns batched via GroupedLinear.

    Processes all N tokens for all G columns simultaneously.
    Each column: LayerNorm -> FFN (D_col -> 4*D_col -> D_col) with residual.
    PM/EM projections for read and write candidates.
    PCM for cross-pass prediction.

    PM/EM reads are reshaped internally:
      column space [BS, N, G, D] <-> memory space [BS*B, N*C, D]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        B = config.B_blocks
        C = config.C
        G = B * C
        D_col = config.D_col
        D_mem = config.D_mem
        D_pcm = config.D_pcm
        self.B = B
        self.C = C
        self.G = G
        self.config = config

        # FFN (two grouped linears + activation)
        self.ffn_norm = GroupedLayerNorm(G, D_col)
        self.ffn_up = GroupedLinear(G, D_col, D_col * config.ffn_expansion)
        self.ffn_down = GroupedLinear(G, D_col * config.ffn_expansion, D_col)

        # PM projections (D_col <-> D_mem)
        self.W_pm_up = GroupedLinear(G, D_col, D_mem)
        self.W_pm_down = GroupedLinear(G, D_mem, D_col)

        # EM projections
        self.W_em_up = GroupedLinear(G, D_col, D_mem)
        self.W_em_down = GroupedLinear(G, D_mem, D_col)

        # PM eligibility candidate projections
        self.W_k_pre = GroupedLinear(G, D_col, D_mem)
        self.W_v_post = GroupedLinear(G, D_col, D_mem)

        # EM candidate projections
        self.W_k_cand = GroupedLinear(G, D_col, D_mem)
        self.W_v_cand = GroupedLinear(G, D_col, D_mem)
        self.W_nov = GroupedLinear(G, D_col, 1)

        # PCM
        self.pcm = CrossPassPCM(G, D_col, D_pcm) if config.pcm_enabled else None

        # FFN gain from PCM (zero-init so gain starts at 1.0)
        if config.pcm_enabled:
            self.W_gain = GroupedLinear(G, D_pcm, D_col)
            nn.init.zeros_(self.W_gain.weight)
            if self.W_gain.bias is not None:
                nn.init.zeros_(self.W_gain.bias)

    # ------------------------------------------------------------------
    # Reshape helpers: column space <-> memory space
    # ------------------------------------------------------------------

    def _to_mem_space(self, x: Tensor) -> Tensor:
        """[BS, N, G, D] -> [BS*B, N*C, D]  (4-D input)
           [BS, N, G]    -> [BS*B, N*C]      (3-D input)"""
        if x.dim() == 4:
            BS, N, _G, D = x.shape
            return (x.view(BS, N, self.B, self.C, D)
                     .permute(0, 2, 1, 3, 4)
                     .reshape(BS * self.B, N * self.C, D))
        # 3-D: no trailing feature dim
        BS, N, _G = x.shape
        return (x.view(BS, N, self.B, self.C)
                 .permute(0, 2, 1, 3)
                 .reshape(BS * self.B, N * self.C))

    def _from_mem_space(self, x: Tensor) -> Tensor:
        """[BS*B, N*C, D] -> [BS, N, G, D]  (3-D input)
           [BS*B, N*C]    -> [BS, N, G]      (2-D input)"""
        if x.dim() == 3:
            BSB, NC, D = x.shape
            BS = BSB // self.B
            N = NC // self.C
            return (x.view(BS, self.B, N, self.C, D)
                     .permute(0, 2, 1, 3, 4)
                     .reshape(BS, N, self.G, D))
        BSB, NC = x.shape
        BS = BSB // self.B
        N = NC // self.C
        return (x.view(BS, self.B, N, self.C)
                 .permute(0, 2, 1, 3)
                 .reshape(BS, N, self.G))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_col, pm_state, em_state, z_hat_prev):
        """
        x_col:      [BS, N, G, D_col]   G = B*C
        pm_state:   ProceduralMemory — state [BS*B, r, D_mem]
        em_state:   EpisodicMemory  — state [BS*B, M, D_mem]
        z_hat_prev: [BS, N, G, D_pcm] or None

        Returns: x_out, z, z_hat, surprise, elig_info, novelty_info
        """
        # 1. PM read (holographic, in D_mem space)
        if self.config.pm_enabled and pm_state.is_initialized():
            q_pm = self.W_pm_up(x_col)                     # [BS,N,G,D_mem]
            q_pm_b = self._to_mem_space(q_pm)               # [BS*B,N*C,D_mem]
            y_pm_b = pm_state.read(q_pm_b)                  # [BS*B,N*C,D_mem]
            y_pm = self._from_mem_space(y_pm_b)              # [BS,N,G,D_mem]
            x_col = x_col + self.W_pm_down(y_pm)            # residual

        # 2. EM read (top-k retrieval)
        if self.config.em_enabled and em_state.is_initialized():
            q_em = self.W_em_up(x_col)                      # [BS,N,G,D_mem]
            q_em_b = self._to_mem_space(q_em)                # [BS*B,N*C,D_mem]
            y_em_b = em_state.read(q_em_b)                   # [BS*B,N*C,D_mem]
            y_em = self._from_mem_space(y_em_b)              # [BS,N,G,D_mem]
            x_col = x_col + self.W_em_down(y_em)            # residual

        # 3. PCM encode + surprise
        if self.pcm is not None:
            z = self.pcm.encode(x_col)                      # [BS,N,G,D_pcm]
            surprise, delta = self.pcm.compute_surprise(z, z_hat_prev)
            gain = 1.0 + 0.1 * torch.tanh(self.W_gain(delta))  # [BS,N,G,D_col]
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
            z_hat = self.pcm.predict(z)                     # [BS,N,G,D_pcm]
        else:
            z_hat = None

        # 6. PM eligibility accumulation info
        k_cand = unit_normalize(self.W_k_pre(x_out))       # [BS,N,G,D_mem]
        v_cand = self.W_v_post(x_out)                      # [BS,N,G,D_mem]
        gate = (surprise / self.config.surprise_scale).clamp(0, 1)  # [BS,N,G]

        # 7. EM novelty scoring info
        q_nov = unit_normalize(self.W_k_cand(x_out))       # [BS,N,G,D_mem]
        v_nov = self.W_v_cand(x_out)                       # [BS,N,G,D_mem]
        w_nov = torch.sigmoid(self.W_nov(x_out).squeeze(-1))  # [BS,N,G]

        elig_info = (k_cand, v_cand, gate)
        novelty_info = (q_nov, v_nov, w_nov, surprise)

        return x_out, z, z_hat, surprise, elig_info, novelty_info
