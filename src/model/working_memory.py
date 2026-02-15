"""
Working Memory — sliding window attention with ring-buffer KV cache.

One shared instance across the entire model. Provides transformer-like
precision on short context (copying, binding) via bounded attention
over the last W tokens.

State per stream:
    wm_K: [BS, W, D_wm]   — key cache
    wm_V: [BS, W, D_wm]   — value cache
    wm_valid: [BS, W]      — bool mask for valid entries
    wm_ptr: [BS]           — ring buffer write index
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin


class WorkingMemory(nn.Module, StateMixin):
    _state_tensor_names = ["wm_K", "wm_V", "wm_valid", "wm_ptr"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.W = config.W
        self.D_wm = config.D_wm
        self.n_heads = config.n_heads_wm
        self.head_dim = config.D_wm // config.n_heads_wm
        assert config.D_wm % config.n_heads_wm == 0

        # Projections
        self.W_q = nn.Linear(config.D, config.D_wm)
        self.W_k = nn.Linear(config.D, config.D_wm)
        self.W_v = nn.Linear(config.D, config.D_wm)
        self.W_o = nn.Linear(config.D_wm, config.D)

        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.drop_resid = nn.Dropout(config.dropout)

        # State tensors (lazily initialized on first step)
        self.wm_K: Tensor = None
        self.wm_V: Tensor = None
        self.wm_valid: Tensor = None
        self.wm_ptr: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize state tensors on first forward pass."""
        self.wm_K = torch.zeros(BS, self.W, self.D_wm, device=device)
        self.wm_V = torch.zeros(BS, self.W, self.D_wm, device=device)
        self.wm_valid = torch.zeros(BS, self.W, dtype=torch.bool, device=device)
        self.wm_ptr = torch.zeros(BS, dtype=torch.long, device=device)

    def step(self, x: Tensor, reset_mask: Tensor) -> Tensor:
        """Process one token through working memory.

        Args:
            x: [BS, D] — current token embedding
            reset_mask: [BS] bool — True for streams at doc boundary

        Returns:
            y_wm: [BS, D] — working memory output
        """
        BS, D = x.shape
        device = x.device

        # Lazy init
        if self.wm_K is None:
            self._lazy_init(BS, device)

        # Reset validity for masked streams
        if reset_mask.any():
            self.wm_valid = self.wm_valid.clone()
            self.wm_valid[reset_mask] = False
            self.wm_ptr = self.wm_ptr.clone()
            self.wm_ptr[reset_mask] = 0

        # Project q, k, v
        q = self.W_q(x)    # [BS, D_wm]
        k = self.W_k(x)    # [BS, D_wm]
        v = self.W_v(x)    # [BS, D_wm]

        # Write (k, v) into ring buffer.
        ptr = self.wm_ptr  # [BS]
        batch_idx = torch.arange(BS, device=device)

        if torch.is_grad_enabled():
            # Functional scatter keeps gradients alive for W_k / W_v
            write_mask = torch.nn.functional.one_hot(
                ptr, self.W
            ).to(dtype=self.wm_K.dtype)  # [BS, W]
            write_mask_3d = write_mask.unsqueeze(-1)  # [BS, W, 1]
            self.wm_K = self.wm_K * (1 - write_mask_3d) + k.unsqueeze(1) * write_mask_3d
            self.wm_V = self.wm_V * (1 - write_mask_3d) + v.unsqueeze(1) * write_mask_3d
        else:
            # In-place index write (no allocation, no autograd overhead)
            self.wm_K[batch_idx, ptr] = k
            self.wm_V[batch_idx, ptr] = v

        # Update validity
        self.wm_valid = self.wm_valid.clone()
        self.wm_valid[batch_idx, ptr] = True

        # Multi-head attention over valid entries
        # Reshape q: [BS, n_heads, head_dim]
        q_h = q.view(BS, self.n_heads, self.head_dim)
        # Reshape K, V cache: [BS, W, n_heads, head_dim] -> [BS, n_heads, W, head_dim]
        K_h = self.wm_K.view(BS, self.W, self.n_heads, self.head_dim).transpose(1, 2)
        V_h = self.wm_V.view(BS, self.W, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [BS, n_heads, W]
        attn = torch.einsum("bnd, bnwd -> bnw", q_h, K_h) * self.scale

        # Mask invalid positions
        valid_mask = self.wm_valid.unsqueeze(1)  # [BS, 1, W]
        attn = attn.masked_fill(~valid_mask, float("-inf"))

        # Softmax (handle all-invalid case)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # all-invalid -> zero attention

        # Store for debug metrics before dropout (normalized probs for entropy)
        self._last_attn = attn.detach()  # [BS, n_heads, W]

        attn = self.attn_drop(attn)

        # Weighted sum: [BS, n_heads, head_dim]
        out = torch.einsum("bnw, bnwd -> bnd", attn, V_h)

        # Merge heads and project
        out = out.reshape(BS, self.D_wm)
        y_wm = self.drop_resid(self.W_o(out))  # [BS, D]

        # Advance pointer (ring buffer wrap)
        self.wm_ptr = (ptr + 1) % self.W

        return y_wm

    def forward_span(self, x_all: Tensor, reset_mask_all: Tensor) -> Tensor:
        """Process P tokens in parallel through working memory.

        Dispatches to batched path when no mid-span resets exist (common case),
        falls back to sequential path for mid-span resets (rare).

        Args:
            x_all: [BS, P, D] — token embeddings for all span tokens
            reset_mask_all: [BS, P] bool — True at doc boundaries

        Returns:
            y_wm_all: [BS, P, D] — working memory outputs
        """
        BS, P, D = x_all.shape
        device = x_all.device

        if self.wm_K is None:
            self._lazy_init(BS, device)

        # Batch projections (shared by both paths)
        q_all = self.W_q(x_all)   # [BS, P, D_wm]
        k_all = self.W_k(x_all)   # [BS, P, D_wm]
        v_all = self.W_v(x_all)   # [BS, P, D_wm]

        if reset_mask_all[:, 1:].any():
            # Mid-span reset (rare) — fall back to sequential
            return self._forward_span_sequential(
                q_all, k_all, v_all, reset_mask_all, BS, P, device
            )
        return self._forward_span_batched(
            q_all, k_all, v_all, reset_mask_all, BS, P, device
        )

    def _forward_span_sequential(
        self, q_all: Tensor, k_all: Tensor, v_all: Tensor,
        reset_mask_all: Tensor, BS: int, P: int, device: torch.device,
    ) -> Tensor:
        """Sequential per-token attention (fallback for mid-span resets)."""
        outputs = []
        for t in range(P):
            reset_t = reset_mask_all[:, t]

            # Reset validity for masked streams
            if reset_t.any():
                self.wm_valid = self.wm_valid.clone()
                self.wm_valid[reset_t] = False
                self.wm_ptr = self.wm_ptr.clone()
                self.wm_ptr[reset_t] = 0

            q = q_all[:, t]  # [BS, D_wm]
            k = k_all[:, t]  # [BS, D_wm]
            v = v_all[:, t]  # [BS, D_wm]

            # Write into ring buffer
            ptr = self.wm_ptr
            write_mask = torch.nn.functional.one_hot(
                ptr, self.W
            ).to(dtype=self.wm_K.dtype)
            write_mask_3d = write_mask.unsqueeze(-1)

            self.wm_K = self.wm_K * (1 - write_mask_3d) + k.unsqueeze(1) * write_mask_3d
            self.wm_V = self.wm_V * (1 - write_mask_3d) + v.unsqueeze(1) * write_mask_3d

            # Update validity
            self.wm_valid = self.wm_valid.clone()
            batch_idx = torch.arange(BS, device=device)
            self.wm_valid[batch_idx, ptr] = True

            # Multi-head attention
            q_h = q.view(BS, self.n_heads, self.head_dim)
            K_h = self.wm_K.view(BS, self.W, self.n_heads, self.head_dim).transpose(1, 2)
            V_h = self.wm_V.view(BS, self.W, self.n_heads, self.head_dim).transpose(1, 2)

            attn = torch.einsum("bnd, bnwd -> bnw", q_h, K_h) * self.scale
            valid_mask = self.wm_valid.unsqueeze(1)
            attn = attn.masked_fill(~valid_mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            attn = attn.nan_to_num(0.0)
            attn = self.attn_drop(attn)

            out = torch.einsum("bnw, bnwd -> bnd", attn, V_h)
            out = out.reshape(BS, self.D_wm)
            y_wm = self.drop_resid(self.W_o(out))
            outputs.append(y_wm)

            # Advance pointer
            self.wm_ptr = (ptr + 1) % self.W

        return torch.stack(outputs, dim=1)  # [BS, P, D]

    def _forward_span_batched(
        self, q_all: Tensor, k_all: Tensor, v_all: Tensor,
        reset_mask_all: Tensor, BS: int, P: int, device: torch.device,
    ) -> Tensor:
        """Batched causal attention over [cache + span] (no mid-span resets).

        Handles first-token resets only. Builds a combined KV of
        [wm_K, k_all] and uses a carefully constructed mask for exact
        sequential equivalence.
        """
        W = self.W

        # --- Handle first-token resets (invalidate cache for those streams) ---
        reset_first = reset_mask_all[:, 0]  # [BS]
        if reset_first.any():
            self.wm_valid = self.wm_valid.clone()
            self.wm_valid[reset_first] = False
            self.wm_ptr = self.wm_ptr.clone()
            self.wm_ptr[reset_first] = 0

        wm_ptr = self.wm_ptr  # [BS] — starting write pointer

        # --- Build combined KV: [cache | span] → [BS, W+P, D_wm] ---
        combined_K = torch.cat([self.wm_K, k_all], dim=1)  # [BS, W+P, D_wm]
        combined_V = torch.cat([self.wm_V, v_all], dim=1)  # [BS, W+P, D_wm]

        # --- Build attention mask [BS, P, W+P] ---
        # Cache portion: per-query validity excluding overwritten positions
        offsets = torch.arange(P, device=device)  # [P]
        write_pos = (wm_ptr.unsqueeze(1) + offsets.unsqueeze(0)) % W  # [BS, P]

        # write_onehot[b, t, w] = True if span token t writes to cache position w
        write_onehot = torch.zeros(BS, P, W, dtype=torch.bool, device=device)
        write_onehot.scatter_(2, write_pos.unsqueeze(-1), True)

        # cumulative_overwrite[b, t, w] = True if any token 0..t writes to position w
        # For query at position t, cache positions overwritten by tokens 0..t must
        # be masked — the old values are stale, and the fresh values are already
        # available in the span portion of the combined KV via the causal mask.
        cumulative_overwrite = write_onehot.cumsum(dim=1).bool()  # [BS, P, W]

        # Cache valid per query: original validity minus overwritten positions
        cache_valid_per_query = self.wm_valid.unsqueeze(1).expand(BS, P, W) & ~cumulative_overwrite
        # [BS, P, W]

        # Span portion: lower-triangular causal mask (token t attends to 0..t)
        span_causal = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))
        span_causal = span_causal.unsqueeze(0).expand(BS, P, P)  # [BS, P, P]

        # Combined mask: [BS, P, W+P]
        attn_mask = torch.cat([cache_valid_per_query, span_causal], dim=2)

        # --- Multi-head attention ---
        # Reshape for multi-head: Q [BS, P, n_heads, head_dim]
        q_h = q_all.view(BS, P, self.n_heads, self.head_dim)
        K_h = combined_K.view(BS, W + P, self.n_heads, self.head_dim)
        V_h = combined_V.view(BS, W + P, self.n_heads, self.head_dim)

        # Transpose to [BS, n_heads, seq, head_dim]
        q_h = q_h.transpose(1, 2)    # [BS, n_heads, P, head_dim]
        K_h = K_h.transpose(1, 2)    # [BS, n_heads, W+P, head_dim]
        V_h = V_h.transpose(1, 2)    # [BS, n_heads, W+P, head_dim]

        # Attention scores: [BS, n_heads, P, W+P]
        attn = torch.matmul(q_h, K_h.transpose(-2, -1)) * self.scale

        # Apply mask: [BS, 1, P, W+P] broadcast over heads
        attn_mask_4d = attn_mask.unsqueeze(1)  # [BS, 1, P, W+P]
        attn = attn.masked_fill(~attn_mask_4d, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # all-invalid -> zero attention
        attn = self.attn_drop(attn)

        # Weighted sum: [BS, n_heads, P, head_dim]
        out = torch.matmul(attn, V_h)

        # Merge heads: [BS, P, D_wm]
        out = out.transpose(1, 2).reshape(BS, P, self.D_wm)

        # Output projection: [BS, P, D]
        y_wm_all = self.drop_resid(self.W_o(out))

        # --- Differentiable cache update (write all P tokens at once) ---
        # write_masks[b, t, w] = 1.0 if span token t writes to cache position w
        write_masks = torch.zeros(BS, P, W, dtype=self.wm_K.dtype, device=device)
        write_masks.scatter_(2, write_pos.unsqueeze(-1), 1.0)

        # For positions written by multiple span tokens (P > W edge, but P <= W
        # guaranteed), only the last write matters. Use the last writer's value.
        # any_written[b, w, 1] = max over t of write_masks[b, t, w]
        any_written = write_masks.max(dim=1).values.unsqueeze(-1)  # [BS, W, 1]

        # Scatter span k/v into cache positions via einsum
        new_K = torch.einsum("bpw, bpd -> bwd", write_masks, k_all)  # [BS, W, D_wm]
        new_V = torch.einsum("bpw, bpd -> bwd", write_masks, v_all)  # [BS, W, D_wm]

        # Differentiable blend: keep old where not overwritten, new where overwritten
        self.wm_K = self.wm_K * (1 - any_written) + new_K
        self.wm_V = self.wm_V * (1 - any_written) + new_V

        # --- Update wm_valid and advance wm_ptr ---
        new_valid = self.wm_valid.clone()
        # Mark all written positions as valid
        any_written_bool = any_written.squeeze(-1).bool()  # [BS, W]
        new_valid = new_valid | any_written_bool
        self.wm_valid = new_valid

        self.wm_ptr = (wm_ptr + P) % W

        return y_wm_all
