"""
Cortical Column Group (v4).

G = B_blocks * C cortical columns batched via GroupedLinear (einsum over
group dim). Each column: LayerNorm -> FFN with residual, PM/EM projections,
PCM.  No Python loop over columns or blocks.

Layout: column activations are [BS, N, B, C, D_col] internally.
GroupedLinear calls: view(BS, N, G, D) -> GroupedLinear -> view(BS, N, B, C, D_out)
(both free views since B,C,D are contiguous).
PM/EM operate at block level in D space — concat C columns for read, split back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig
from .predictive_coding import GroupedLinear, GroupedLayerNorm, CrossPassPCM
from .utils import unit_normalize


class LateralMixer(nn.Module):
    """L2/3: cross-column attention within each block. Shared across B blocks.

    Single-head attention across C columns within each block.
    Weights are shared (not per-block or per-group).
    """

    def __init__(self, D_col: int):
        super().__init__()
        self.ln = nn.LayerNorm(D_col)
        self.W_q = nn.Linear(D_col, D_col)
        self.W_k = nn.Linear(D_col, D_col)
        self.W_v = nn.Linear(D_col, D_col)
        self.W_out = nn.Linear(D_col, D_col)
        self.scale = D_col ** -0.5
        # Zero-init output projection so mixer starts as identity (residual)
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: [BS, N, B, C, D_col] -> [BS, N, B, C, D_col]"""
        h = self.ln(x)
        q, k, v = self.W_q(h), self.W_k(h), self.W_v(h)
        # Attention across C dim within each block
        attn = torch.einsum("snbcd, snbed -> snbce", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)  # [BS, N, B, C, C]
        out = torch.einsum("snbce, snbed -> snbcd", attn, v)
        return x + self.W_out(out)


class CorticalColumnGroup(nn.Module):
    """G = B*C cortical columns batched via GroupedLinear.

    Processes all N tokens for all G columns simultaneously.
    Each column: LayerNorm -> FFN (D_col -> 4*D_col -> D_col) with residual.
    PM/EM read at block level in D space (concat C columns, split back).
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

        # Lateral mixer: cross-column attention within each block (shared across B)
        self.lateral = LateralMixer(D_col)

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

        self._grad_ckpt = config.gradient_checkpointing

    def _ffn_block(self, x_col: Tensor, gain: Tensor | None) -> Tensor:
        """FFN with optional gain modulation (checkpointable, no side effects)."""
        h = self.ffn_norm(x_col)
        if gain is not None:
            h = h * gain
        return self.ffn_down(F.gelu(self.ffn_up(h)))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_col, pm_state, em_state, z_hat_prev):
        """
        x_col:      [BS, N, G, D_col]   G = B*C
        pm_state:   ProceduralMemory — state [BS, B, r, D]
        em_state:   EpisodicMemory  — state [BS, B, M, D]
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

        # 1. PM read (holographic, block-level in D space)
        if self.config.pm_enabled:
            q_pm = to_5d(x_col).reshape(BS, N, B, -1)             # [BS,N,B,D] (concat C cols)
            y_pm = pm_state.read(q_pm)                             # [BS,N,B,D]
            y_pm_cols = y_pm.view(BS, N, B, C, D_col)              # split back
            x_col = x_col + to_g(y_pm_cols)                       # residual

        # 2. EM read (top-k retrieval, block-level in D space)
        if self.config.em_enabled:
            q_em = to_5d(x_col).reshape(BS, N, B, -1)             # [BS,N,B,D]
            y_em = em_state.read(q_em)                             # [BS,N,B,D]
            y_em_cols = y_em.view(BS, N, B, C, D_col)              # split back
            x_col = x_col + to_g(y_em_cols)                       # residual

        # 3. Lateral mixing (cross-column attention within each block)
        x_col = to_g(self.lateral(to_5d(x_col)))

        # 4. PCM encode + surprise
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

        # 5. FFN with gain modulation (optionally gradient-checkpointed)
        if self._grad_ckpt and x_col.requires_grad:
            h = grad_checkpoint(self._ffn_block, x_col, gain, use_reentrant=False)
        else:
            h = self._ffn_block(x_col, gain)
        x_out = x_col + h                                   # residual

        # 6. PCM hypothesis (predict next pass)
        if self.pcm is not None:
            z_hat = self.pcm.predict(z)                     # [BS,N,G,D_pcm]
        else:
            z_hat = None

        # 7. Fused post-processing projections
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
