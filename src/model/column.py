"""
Cortical Column Group (v4).

G = B_blocks * C cortical columns batched via GroupedLinear (einsum over
group dim). Each column: LayerNorm -> FFN with residual, PM/EM projections,
PCM.  No Python loop over columns or blocks.

Layout: column activations are [BS, N_C, B, C, D_col] internally.
GroupedLinear calls: view(BS, N_C, G, D) -> GroupedLinear -> view(BS, N_C, B, C, D_out)
(both free views since B,C,D are contiguous).
PM/EM reads use per-column D_col slices (read_sliced); writes stay block-level D.
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
    """L2/3: learned linear mixing across C columns within each block.

    A [C, C] mixing matrix applied independently per D_col feature.
    Zero-init so it starts as identity (residual).
    """

    def __init__(self, C: int):
        super().__init__()
        self.C = C
        # [C, C] mixing matrix — zero-init so residual starts as identity
        self.mix = nn.Parameter(torch.zeros(C, C))

    def forward(self, x: Tensor) -> Tensor:
        """x: [BS, N, B, C, D_col] -> [BS, N, B, C, D_col]"""
        # Mix across C dim: out[c] = x[c] + Σ_j mix[c,j] * x[j]
        mixed = torch.einsum("...cd, ec -> ...ed", x, self.mix)
        return x + mixed


class CrossBlockMixer(nn.Module):
    """Learned linear mixing across B blocks for the same column index.

    A [B, B] mixing matrix applied independently per C column and D_col feature.
    Zero-init so it starts as identity (residual).
    """

    def __init__(self, B: int):
        super().__init__()
        self.B = B
        # [B, B] mixing matrix — zero-init so residual starts as identity
        self.cross_mix = nn.Parameter(torch.zeros(B, B))

    def forward(self, x: Tensor) -> Tensor:
        """x: [BS, N, B, C, D_col] -> [BS, N, B, C, D_col]"""
        # Mix across B dim: out[b] = x[b] + Σ_j cross_mix[b,j] * x[j]
        mixed = torch.einsum("...bcd, fb -> ...fcd", x, self.cross_mix)
        return x + mixed


class PositionAttention(nn.Module):
    """Content-adaptive attention across N_C positions per column group.

    G groups independently attend over N_C positions.
    Zero-init out_proj so it starts as identity (residual).
    Input/output: [BS, N_C, G, D_col]
    """

    def __init__(self, G: int, D_col: int, D_attn: int, grad_ckpt: bool = False):
        super().__init__()
        self.G = G
        self.D_col = D_col
        self.D_attn = D_attn
        self.scale = D_attn ** -0.5
        self._grad_ckpt = grad_ckpt

        self.norm = GroupedLayerNorm(G, D_col)
        self.q_proj = GroupedLinear(G, D_col, D_attn)
        self.k_proj = GroupedLinear(G, D_col, D_attn)
        self.v_proj = GroupedLinear(G, D_col, D_col)
        self.out_proj = GroupedLinear(G, D_col, D_col)
        # Zero-init out_proj so residual starts as identity
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _attention(self, x: Tensor) -> Tensor:
        """Core attention computation (checkpointable)."""
        BS, N_C, G, D_col = x.shape
        h = self.norm(x)
        q = self.q_proj(h)   # [BS, N_C, G, D_attn]
        k = self.k_proj(h)   # [BS, N_C, G, D_attn]
        v = self.v_proj(h)   # [BS, N_C, G, D_col]

        # Transpose to [BS*G, N_C, D_*] for bmm
        q = q.permute(0, 2, 1, 3).reshape(BS * G, N_C, self.D_attn)
        k = k.permute(0, 2, 1, 3).reshape(BS * G, N_C, self.D_attn)
        v = v.permute(0, 2, 1, 3).reshape(BS * G, N_C, D_col)

        # scores: [BS*G, N_C, N_C]
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)  # [BS*G, N_C, D_col]

        # Reshape back to [BS, N_C, G, D_col]
        out = out.reshape(BS, G, N_C, D_col).permute(0, 2, 1, 3)
        return self.out_proj(out)

    def forward(self, x: Tensor) -> Tensor:
        """x: [BS, N_C, G, D_col] -> [BS, N_C, G, D_col]"""
        if self._grad_ckpt and x.requires_grad:
            out = grad_checkpoint(self._attention, x, use_reentrant=False)
        else:
            out = self._attention(x)
        return x + out


class FFNStack(nn.Module):
    """L residual FFN layers (LayerNorm -> up -> GELU -> down) with optional gain on first layer."""

    def __init__(self, G: int, D_col: int, D_hidden: int, L: int, grad_ckpt: bool = False):
        super().__init__()
        self.L = L
        self._grad_ckpt = grad_ckpt
        self.norms = nn.ModuleList([GroupedLayerNorm(G, D_col) for _ in range(L)])
        self.ups = nn.ModuleList([GroupedLinear(G, D_col, D_hidden) for _ in range(L)])
        self.downs = nn.ModuleList([GroupedLinear(G, D_hidden, D_col) for _ in range(L)])

    def _layer(self, x: Tensor, idx: int, gain: Tensor | None) -> Tensor:
        """Single FFN layer (checkpointable)."""
        h = self.norms[idx](x)
        if gain is not None:
            h = h * gain
        return self.downs[idx](F.gelu(self.ups[idx](h)))

    def forward(self, x: Tensor, gain: Tensor | None = None) -> Tensor:
        """L residual FFN layers. gain applied to first layer only."""
        for i in range(self.L):
            layer_gain = gain if i == 0 else None
            if self._grad_ckpt and x.requires_grad:
                h = grad_checkpoint(self._layer, x, i, layer_gain, use_reentrant=False)
            else:
                h = self._layer(x, i, layer_gain)
            x = x + h
        return x


class CorticalColumnGroup(nn.Module):
    """G = B*C cortical columns batched via GroupedLinear.

    Processes all N_C tokens for all G columns simultaneously.
    Per R-pass order: PCM -> PM/EM read -> lateral -> pos_attn -> ffn_pre -> cross-block -> ffn_post -> W_post.
    PM/EM reads use per-column D_col slices (read_sliced).
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

        # Two FFN stacks: pre (after lateral) and post (after cross-block)
        D_hidden = D_col * config.ffn_expansion
        L = config.ffn_depth
        self.ffn_pre = FFNStack(G, D_col, D_hidden, L, config.gradient_checkpointing)
        self.ffn_post = FFNStack(G, D_col, D_hidden, L, config.gradient_checkpointing)

        # Cross-block mixer: learned linear mixing across B blocks
        self.cross_block = CrossBlockMixer(B)

        # Lateral mixer: learned linear mixing across C columns within each block
        self.lateral = LateralMixer(C)

        # Position attention: content-adaptive attention across N_C positions
        if config.position_attn_dim > 0:
            self.pos_attn = PositionAttention(G, D_col, config.position_attn_dim,
                                              config.gradient_checkpointing)
        else:
            self.pos_attn = None

        # Fused post-processing projections:
        # k_pre (D_col) + v_post (D_col) + k_cand (D_col) + v_cand (D_col) + nov (1)
        self.W_post_fused = GroupedLinear(G, D_col, 4 * D_col + 1)

        # PCM
        self.pcm = CrossPassPCM(G, D_col, D_pcm) if config.pcm_enabled else None

        # FFN gain from PCM (zero-init so gain starts at 1.0) — applied to first FFN layer only
        if config.pcm_enabled:
            self.W_gain = GroupedLinear(G, D_pcm, D_col)
            nn.init.zeros_(self.W_gain.weight)
            if self.W_gain.bias is not None:
                nn.init.zeros_(self.W_gain.bias)

    def forward(self, x_col, pm_state, em_state, z_hat_prev):
        """Single R-pass: PCM -> PM/EM read -> lateral -> pos_attn -> ffn_pre -> cross-block -> ffn_post -> W_post.

        x_col:      [BS, N, G, D_col]   G = B*C
        pm_state:   ProceduralMemory — state [BS, B, r, D]
        em_state:   EpisodicMemory  — state [BS, B, M, D]
        z_hat_prev: [BS, N, G, D_pcm] or None

        Returns: x_out, z, z_hat, surprise, elig_info, novelty_info
        """
        BS, N, G, D_col = x_col.shape
        B, C = self.B, self.C

        def to_5d(x):
            return x.view(BS, N, B, C, -1)

        def to_g(x):
            return x.reshape(BS, N, G, -1)

        # 1. PCM encode + surprise + gain
        if self.pcm is not None:
            z = self.pcm.encode(x_col)
            surprise, delta = self.pcm.compute_surprise(z, z_hat_prev)
            gain = 1.0 + 0.1 * torch.tanh(self.W_gain(delta))
        else:
            z = None
            surprise = torch.zeros(
                x_col.shape[:-1], device=x_col.device, dtype=x_col.dtype
            )
            delta = None
            gain = None

        # 2. PM read (holographic, per-column D_col slices)
        if self.config.pm_enabled:
            q_pm = to_5d(x_col)                    # [BS, N_C, B, C, D_col]
            y_pm = pm_state.read_sliced(q_pm)      # [BS, N_C, B, C, D_col]
            x_col = x_col + to_g(y_pm)

        # 3. EM read (top-k retrieval, per-column D_col slices)
        if self.config.em_enabled:
            q_em = to_5d(x_col)                    # [BS, N_C, B, C, D_col]
            y_em = em_state.read_sliced(q_em)      # [BS, N_C, B, C, D_col]
            x_col = x_col + to_g(y_em)

        # 4. Lateral mixing (local: across C columns within each block)
        x_col = to_g(self.lateral(to_5d(x_col)))

        # 4b. Position attention (global: across N_C positions, content-adaptive)
        if self.pos_attn is not None:
            x_col = self.pos_attn(x_col)

        # 5. Pre FFN stack (L layers, gain on first only)
        x_col = self.ffn_pre(x_col, gain=gain)

        # 6. Cross-block mixing (global: across B blocks)
        x_col = to_g(self.cross_block(to_5d(x_col)))

        # 7. Post FFN stack (L layers, no gain)
        x_out = self.ffn_post(x_col)

        # 8. PCM hypothesis
        if self.pcm is not None:
            z_hat = self.pcm.predict(z)
        else:
            z_hat = None

        # 9. Fused post-processing projections
        proj = self.W_post_fused(x_out)
        proj_5d = to_5d(proj)
        k_pre, v_post, k_cand_raw, v_cand, nov_raw = proj_5d.split(
            [D_col, D_col, D_col, D_col, 1], dim=-1
        )

        k_cand = unit_normalize(k_pre)
        gate = (surprise.view(BS, N, B, C) / self.config.surprise_scale).clamp(0, 1)

        q_nov = unit_normalize(k_cand_raw)
        w_nov = torch.sigmoid(nov_raw.squeeze(-1))

        elig_info = (k_cand, v_post, gate)
        novelty_info = (q_nov, v_cand, w_nov, surprise.view(BS, N, B, C))

        return x_out, z, z_hat, surprise, elig_info, novelty_info
