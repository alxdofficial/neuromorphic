"""
Cortical Column Group (v4).

G = B_blocks * C cortical columns batched via GroupedLinear (einsum over
group dim). Each column: LayerNorm -> FFN with residual, PM/EM projections,
PCM.  No Python loop over columns or blocks.

Layout: column activations are [BS, N, B, C, D_col] internally.
GroupedLinear calls: view(BS, N, G, D) -> GroupedLinear -> view(BS, N, B, C, D_out)
(both free views since B,C,D are contiguous).
PM/EM operate directly in D_col space — no projection needed.
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
    PM/EM read directly in D_col space (no projection).
    PCM for cross-pass prediction.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        B = config.B_blocks
        C = config.C
        G = B * C
        D_col = config.D_col
        D_pcm = config.D_pcm
        self.B = B
        self.C = C
        self.G = G
        self.config = config

        # FFN (two grouped linears + activation)
        self.ffn_norm = GroupedLayerNorm(G, D_col)
        self.ffn_up = GroupedLinear(G, D_col, D_col * config.ffn_expansion)
        self.ffn_down = GroupedLinear(G, D_col * config.ffn_expansion, D_col)

        # Fused post-processing projections:
        # k_pre (D_col) + v_post (D_col) + k_cand (D_col) + v_cand (D_col) + nov (1)
        self.W_post_fused = GroupedLinear(G, D_col, 4 * D_col + 1)

        # PCM
        self.pcm = CrossPassPCM(G, D_col, D_pcm) if config.pcm_enabled else None

        # FFN gain from PCM (zero-init so gain starts at 1.0)
        if config.pcm_enabled:
            self.W_gain = GroupedLinear(G, D_pcm, D_col)
            nn.init.zeros_(self.W_gain.weight)
            if self.W_gain.bias is not None:
                nn.init.zeros_(self.W_gain.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_col, pm_state, em_state, z_hat_prev):
        """
        x_col:      [BS, N, G, D_col]   G = B*C
        pm_state:   ProceduralMemory — state [BS, B, r, D_col]
        em_state:   EpisodicMemory  — state [BS, B, M, D_col]
        z_hat_prev: [BS, N, G, D_pcm] or None

        Returns: x_out, z, z_hat, surprise, elig_info, novelty_info
        """
        BS, N, G, D_col = x_col.shape
        B, C = self.B, self.C

        # Helper: view between [BS, N, G, D] and [BS, N, B, C, D]
        # to_5d is always on GroupedLinear output (contiguous) -> free view
        # to_g uses reshape since einsum outputs may have non-trivial strides
        def to_5d(x):
            return x.view(BS, N, B, C, -1)

        def to_g(x):
            return x.reshape(BS, N, G, -1)

        # 1. PM read (holographic, direct in D_col space)
        if self.config.pm_enabled:
            q_pm = to_5d(x_col)                                    # free view [BS,N,B,C,D_col]
            y_pm = pm_state.read(q_pm)                             # [BS,N,B,C,D_col]
            x_col = x_col + to_g(y_pm)                            # residual

        # 2. EM read (top-k retrieval, direct in D_col space)
        if self.config.em_enabled:
            q_em = to_5d(x_col)                                    # free view [BS,N,B,C,D_col]
            y_em = em_state.read(q_em)                             # [BS,N,B,C,D_col]
            x_col = x_col + to_g(y_em)                            # residual

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

        # 6. Fused post-processing projections
        proj = self.W_post_fused(x_out)                     # [BS,N,G,4*D_col+1]
        proj_5d = to_5d(proj)                                # [BS,N,B,C,4*D_col+1]
        k_pre, v_post, k_cand_raw, v_cand, nov_raw = proj_5d.split(
            [D_col, D_col, D_col, D_col, 1], dim=-1
        )

        # PM eligibility info
        k_cand = unit_normalize(k_pre)                       # [BS,N,B,C,D_col]
        gate = (surprise.view(BS, N, B, C) / self.config.surprise_scale).clamp(0, 1)

        # EM novelty info
        q_nov = unit_normalize(k_cand_raw)                   # [BS,N,B,C,D_col]
        w_nov = torch.sigmoid(nov_raw.squeeze(-1))           # [BS,N,B,C]

        elig_info = (k_cand, v_post, gate)
        novelty_info = (q_nov, v_cand, w_nov, surprise.view(BS, N, B, C))

        return x_out, z, z_hat, surprise, elig_info, novelty_info
