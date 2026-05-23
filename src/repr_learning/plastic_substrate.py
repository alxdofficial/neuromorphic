"""Exp 2: plastic substrate — stack of gated-MLP blocks with one plastic
weight per block, updated by a Hebbian + Oja basis modulated by a
learnable per-block controller.

Write: pass context tokens through the stack one window at a time; after
each window, each block's plastic weight receives an update of the form
    ΔW = gain · (post ⊗ pre) − decay · W_fast
where (gain, decay) come from a per-block learnable controller that sees
local features (pre/post norms, cosine, mean surprise, log W_fast norm).
Gradient flows through the write so the controller can be trained from
QA loss; older windows are detached at chunk boundary to bound memory.

Read: pass M conditioning vectors through the stack with weights frozen.
The output is M memory tokens.

The "fast" weight per block is per-batch-element [B, h, h] state, passed
externally as a list of tensors. The "slow" weight is a nn.Parameter that
the optimizer updates as a learned initialization.

Bottleneck accounting (matches A/B/MT/Mamba at v1h scale):
    M × h_sub = 26,100 floats per read call (M=36, h_sub=725).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PlasticBlock(nn.Module):
    """One block: norm → gated plastic linear → residual.

        residual = x
        x_n      = LayerNorm(x)
        gate     = sigmoid(W_gate(x_n))            # static
        pre      = x_n
        post     = GELU(pre @ (W_slow + W_fast).T) # W_slow is Parameter, W_fast is per-batch
        return residual + gate * post,  pre, post
    """

    def __init__(self, h_sub: int):
        super().__init__()
        self.h_sub = h_sub
        self.norm = nn.LayerNorm(h_sub)
        self.W_gate = nn.Linear(h_sub, h_sub, bias=False)
        self.W_slow = nn.Parameter(torch.empty(h_sub, h_sub))
        nn.init.xavier_uniform_(self.W_slow, gain=0.5)
        nn.init.xavier_uniform_(self.W_gate.weight)

    def forward(
        self, x: Tensor, W_fast: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        x:        [B, ..., h_sub]   activations
        W_fast:   [B, h_sub, h_sub] per-batch fast weight, or None

        Returns (out, pre, post):
            out:  same shape as x — residual + gate * post
            pre:  pre-activation (LayerNorm output)
            post: post-activation (GELU output of plastic linear)
        """
        residual = x
        x_n = self.norm(x)
        gate = torch.sigmoid(self.W_gate(x_n))

        # Plastic linear: x_n @ (W_slow + W_fast).T
        slow_out = x_n @ self.W_slow.T  # [B, ..., h_sub]

        if W_fast is None:
            out_lin = slow_out
        else:
            # Per-batch contribution. Flatten the middle dims so we can use bmm-style:
            # x_n: [B, K, h]  W_fast: [B, h, h]  →  x_n @ W_fast.T = [B, K, h]
            B = x_n.shape[0]
            orig_shape = x_n.shape
            x_flat = x_n.reshape(B, -1, self.h_sub)            # [B, K, h]
            fast_flat = x_flat @ W_fast.transpose(-1, -2)      # [B, K, h]
            fast_out = fast_flat.reshape(orig_shape)
            out_lin = slow_out + fast_out

        post = F.gelu(out_lin)
        out = residual + gate * post
        return out, x_n, post


class PlasticityController(nn.Module):
    """Per-block controller: local features → (gain, decay) per batch element.

    Features (5):
        mean ‖pre‖
        mean ‖post‖
        mean cos numerator (pre · post / T)
        mean surprise
        log1p(‖W_fast‖)

    Output: gain ∈ ℝ (modulates Hebbian basis), decay ∈ [0,1) (Oja-style decay).

    Final layer is zero-init: at init, gain=decay=0 so the substrate's fast
    weight stays at zero and the block behaves as pure W_slow. Training
    learns when/how strongly to apply plasticity.
    """

    N_FEATURES = 5

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.feature_norm = nn.LayerNorm(self.N_FEATURES)
        self.net = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        pre: Tensor,       # [B, T_w, h_sub]
        post: Tensor,      # [B, T_w, h_sub]
        surprise: Tensor,  # [B, T_w]
        W_fast: Tensor,    # [B, h_sub, h_sub]
    ) -> tuple[Tensor, Tensor]:
        B = pre.shape[0]
        pre_norm = pre.norm(dim=-1).mean(dim=-1)              # [B]
        post_norm = post.norm(dim=-1).mean(dim=-1)            # [B]
        cos_num = (pre * post).sum(-1).mean(dim=-1)           # [B]
        surp_mean = surprise.mean(dim=-1)                     # [B]
        w_norm = W_fast.flatten(1).norm(dim=-1).log1p()       # [B]
        feats = torch.stack(
            [pre_norm, post_norm, cos_num, surp_mean, w_norm], dim=-1,
        )                                                      # [B, 5]
        feats = self.feature_norm(feats)
        out = self.net(feats)                                  # [B, 2]
        gain = out[..., 0]
        decay = torch.sigmoid(out[..., 1])
        return gain, decay


class ReadHead(nn.Module):
    """Project a Llama-side hidden vector → M conditioning vectors for the substrate.

    Single shared projection (d_in → h_sub) plus M learned bias queries.
    The conditioning vectors all share the same context-dependent direction
    but get diversified by the M orthogonal-ish biases. This is the cheap
    BLIP-2-style design (1.5M params vs the per-query-matrix's 53M).
    """

    def __init__(self, d_in: int, h_sub: int, M: int):
        super().__init__()
        self.M = M
        self.h_sub = h_sub
        # Shared projection of the (pooled) conditioning vector
        self.proj = nn.Linear(d_in, h_sub, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        # M learned base queries, added to the projected vector
        self.queries = nn.Parameter(torch.empty(M, h_sub))
        nn.init.normal_(self.queries, std=h_sub ** -0.5)

    def forward(self, h_in: Tensor) -> Tensor:
        """
        h_in: [B, ..., d_in]
        Returns conditioning vectors: [B, ..., M, h_sub]
        """
        proj = self.proj(h_in)                                 # [B, ..., h_sub]
        proj = proj.unsqueeze(-2)                              # [B, ..., 1, h_sub]
        return self.queries + proj                             # [B, ..., M, h_sub]


class PlasticSubstrate(nn.Module):
    """Stack of D plastic blocks with per-block plasticity controllers.

    State (the "fast weights") lives outside the module as a list of D
    tensors of shape [B, h_sub, h_sub]. This makes batched per-context
    state explicit and lets the trainer detach at chunk boundaries.
    """

    def __init__(self, D: int, h_sub: int):
        super().__init__()
        self.D = D
        self.h_sub = h_sub
        self.blocks = nn.ModuleList([PlasticBlock(h_sub) for _ in range(D)])
        self.controllers = nn.ModuleList([PlasticityController() for _ in range(D)])

    def init_fast_state(
        self, B: int, device: torch.device, dtype: torch.dtype,
    ) -> list[Tensor]:
        """All-zero fast weights per block, per batch element."""
        return [
            torch.zeros(B, self.h_sub, self.h_sub, device=device, dtype=dtype)
            for _ in range(self.D)
        ]

    def write_window(
        self,
        x: Tensor,                              # [B, T_w, h_sub]
        surprise: Tensor,                       # [B, T_w]
        fast_state: list[Tensor],               # D × [B, h_sub, h_sub]
        attention_mask: Optional[Tensor] = None,  # [B, T_w] True=real
    ) -> list[Tensor]:
        """One write pass over a window. Updates fast weights per block;
        returns a new list of fast tensors. Gradient flows through this call.
        """
        B, T_w, _ = x.shape
        new_state = []
        for d in range(self.D):
            block = self.blocks[d]
            controller = self.controllers[d]
            W_fast = fast_state[d]

            out, pre, post = block(x, W_fast=W_fast)
            # pre, post: [B, T_w, h_sub]

            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)                  # [B, T_w, 1]
                T_real = mask.sum(dim=1).clamp(min=1.0).squeeze(-1)         # [B]
                pre_eff = pre * mask
                post_eff = post * mask
                surp_eff = surprise * attention_mask.float()
            else:
                T_real = torch.full(
                    (B,), float(T_w), device=x.device, dtype=x.dtype,
                )
                pre_eff = pre
                post_eff = post
                surp_eff = surprise

            # Hebbian aggregate over time: sum_t post[b,t,:] ⊗ pre[b,t,:], normalized by T_real
            hebbian = torch.einsum("bti,btj->bij", post_eff, pre_eff)        # [B, h, h]
            hebbian = hebbian / T_real.view(B, 1, 1)

            # Plasticity controller → (gain, decay) per batch element
            gain, decay = controller(pre_eff, post_eff, surp_eff, W_fast)    # [B], [B]
            delta = gain.view(B, 1, 1) * hebbian - decay.view(B, 1, 1) * W_fast

            # Rigorous all-padded invariant: if a batch row has zero real
            # tokens, the substrate state for that row must not change. Even
            # though hebbian is already 0 in that case, the decay term would
            # pull a non-zero W_fast toward zero — not what we want.
            if attention_mask is not None:
                has_real = attention_mask.any(dim=1).to(delta.dtype)         # [B]
                delta = delta * has_real.view(B, 1, 1)

            W_fast_new = W_fast + delta
            new_state.append(W_fast_new)

            x = out

        return new_state

    def read(
        self,
        cond: Tensor,                  # [B, M, h_sub] or [B, ..., h_sub]
        fast_state: list[Tensor],      # D × [B, h_sub, h_sub] frozen substrate state
    ) -> Tensor:
        """Forward conditioning vectors through the stack WITHOUT updating
        the fast weights. Returns the final-layer activations. Same shape
        as `cond`. Gradient flows through fast_state (the substrate state
        is a tensor with grad enabled — it's just not mutated here).
        """
        x = cond
        for d in range(self.D):
            block = self.blocks[d]
            W_fast = fast_state[d]
            out, _, _ = block(x, W_fast=W_fast)
            x = out
        return x
