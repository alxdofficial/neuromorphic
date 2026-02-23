"""
Working Memory — two implementations:

1. WorkingMemory (softmax): sliding window attention with ring-buffer KV cache.
2. GLAWorkingMemory (gla): Gated Linear Attention with recurrent state.

One shared instance across the entire model.

WorkingMemory state per stream:
    wm_K: [BS, W, D_wm]   — key cache
    wm_V: [BS, W, D_wm]   — value cache
    wm_valid: [BS, W]      — bool mask for valid entries
    wm_ptr: [BS]           — ring buffer write index

GLAWorkingMemory state per stream:
    gla_state: [BS, H, head_dim, head_dim]  — recurrent state matrix (bf16 on CUDA, fp32 on CPU)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, runtime_state_dtype


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

        # ALiBi recency bias: fixed slopes per head (not learned)
        slopes = 2.0 ** (-8.0 * torch.arange(1, self.n_heads + 1, dtype=torch.float32) / self.n_heads)
        self.register_buffer("alibi_slopes", slopes)  # [n_heads]

        # Pre-computed causal mask for span attention (avoids per-call allocation)
        self.register_buffer(
            "_span_causal",
            torch.tril(torch.ones(config.P, config.P, dtype=torch.bool)),
        )  # [P, P]

        # State tensors (lazily initialized on first step)
        self.wm_K: Tensor = None
        self.wm_V: Tensor = None
        self.wm_valid: Tensor = None
        self.wm_ptr: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize state tensors on first forward pass."""
        state_dtype = runtime_state_dtype(device)
        self.wm_K = torch.zeros(BS, self.W, self.D_wm, device=device, dtype=state_dtype)
        self.wm_V = torch.zeros(BS, self.W, self.D_wm, device=device, dtype=state_dtype)
        self.wm_valid = torch.zeros(BS, self.W, dtype=torch.bool, device=device)
        self.wm_ptr = torch.zeros(BS, dtype=torch.long, device=device)

    def _alibi_bias_step(self, BS: int) -> Tensor:
        """ALiBi recency bias for single-token attention over cache.

        Age of cache slot j = (ptr - j) % W, where ptr is the current
        write position. Bias = -slope * age (recent slots get less penalty).

        Returns: [BS, n_heads, W]
        """
        slot_idx = torch.arange(self.W, device=self.alibi_slopes.device)  # [W]
        distances = (self.wm_ptr.unsqueeze(1) - slot_idx.unsqueeze(0)) % self.W  # [BS, W]
        # slopes: [n_heads] -> [1, n_heads, 1]; distances: [BS, W] -> [BS, 1, W]
        return -self.alibi_slopes.view(1, -1, 1) * distances.unsqueeze(1).float()

    def _alibi_bias_span(self, BS: int, P: int, write_pos: Tensor) -> Tensor:
        """ALiBi recency bias for span attention over [cache | span].

        Cache portion: distance from query t to cache slot j =
            (write_pos[t] - j) % W, accounting for ring buffer position.
        Span portion: causal distance t - i for i <= t.

        Args:
            write_pos: [BS, P] — ring buffer write positions per span token

        Returns: [BS, n_heads, P, W+P]
        """
        W = self.W
        device = self.alibi_slopes.device

        # Cache distances: [BS, P, W]
        slot_idx = torch.arange(W, device=device)  # [W]
        cache_dist = (write_pos.unsqueeze(-1) - slot_idx.view(1, 1, W)) % W  # [BS, P, W]

        # Span distances: causal [P, P], clamped to 0 for non-causal (masked out)
        # dist[t, i] = t - i (query t, key i); non-causal entries clamped to 0
        span_pos = torch.arange(P, device=device)
        span_dist = (span_pos.unsqueeze(1) - span_pos.unsqueeze(0)).clamp(min=0)  # [P, P]
        span_dist = span_dist.unsqueeze(0).expand(BS, P, P)  # [BS, P, P]

        # Combined: [BS, P, W+P]
        distances = torch.cat([cache_dist, span_dist], dim=-1).float()

        # slopes: [n_heads] -> [1, n_heads, 1, 1]; distances -> [BS, 1, P, W+P]
        return -self.alibi_slopes.view(1, -1, 1, 1) * distances.unsqueeze(1)

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

        # Reset validity for masked streams (unconditional tensor ops)
        keep = (~reset_mask).unsqueeze(1)
        self.wm_valid = self.wm_valid & keep
        self.wm_ptr = self.wm_ptr * (~reset_mask).long()

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
        attn = attn + self._alibi_bias_step(BS)

        # Mask invalid positions
        valid_mask = self.wm_valid.unsqueeze(1)  # [BS, 1, W]
        attn = attn.masked_fill(~valid_mask, float("-inf"))

        # Softmax (handle all-invalid: zero attention instead of NaN)
        all_invalid = (~valid_mask).all(dim=-1, keepdim=True)  # [BS, 1, 1]
        attn = torch.softmax(attn, dim=-1)
        attn = torch.where(all_invalid, torch.zeros_like(attn), attn)

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

        Always uses the batched path (no data-dependent branch).
        Mid-span resets are handled via the carry mask in the recurrence;
        stale WM cache entries are an accepted minor approximation (same
        class as frozen PM/EM within spans).

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

        # Batch projections
        q_all = self.W_q(x_all)   # [BS, P, D_wm]
        k_all = self.W_k(x_all)   # [BS, P, D_wm]
        v_all = self.W_v(x_all)   # [BS, P, D_wm]

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
            keep = (~reset_t).unsqueeze(1)
            self.wm_valid = self.wm_valid & keep
            self.wm_ptr = self.wm_ptr * (~reset_t).long()

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
            attn = attn + self._alibi_bias_step(BS)
            valid_mask = self.wm_valid.unsqueeze(1)
            attn = attn.masked_fill(~valid_mask, float("-inf"))
            all_invalid = (~valid_mask).all(dim=-1, keepdim=True)
            attn = torch.softmax(attn, dim=-1)
            attn = torch.where(all_invalid, torch.zeros_like(attn), attn)
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
        """Batched causal attention over [cache + span].

        Handles first-token resets via unconditional masking. Mid-span
        resets are an accepted approximation (stale cache entries persist
        until the next span). Builds a combined KV of [wm_K, k_all] and
        uses a carefully constructed mask.
        """
        W = self.W

        # --- Handle first-token resets (invalidate cache for those streams) ---
        # Unconditional tensor ops (no .any() sync) — no-op when mask is all-False
        reset_first = reset_mask_all[:, 0]  # [BS]
        keep = (~reset_first).unsqueeze(1)                  # [BS, 1]
        self.wm_valid = self.wm_valid * keep                # zeros validity for reset streams
        self.wm_ptr = self.wm_ptr * (~reset_first).long()   # zeros ptr for reset streams

        wm_ptr = self.wm_ptr  # [BS] — starting write pointer

        # --- Build combined KV: [cache | span] → [BS, W+P, D_wm] ---
        combined_K = torch.cat([self.wm_K, k_all], dim=1)  # [BS, W+P, D_wm]
        combined_V = torch.cat([self.wm_V, v_all], dim=1)  # [BS, W+P, D_wm]

        # --- Build attention mask [BS, P, W+P] ---
        # Cache portion: per-query validity excluding overwritten positions
        offsets = torch.arange(P, device=device)  # [P]
        write_pos = (wm_ptr.unsqueeze(1) + offsets.unsqueeze(0)) % W  # [BS, P]

        # Cache slot overwrite mask by modular position arithmetic:
        # slot_age[b,w] = first timestep index that overwrites slot w.
        slot_idx = torch.arange(W, device=device)  # [W]
        slot_age = (slot_idx.unsqueeze(0) - wm_ptr.unsqueeze(1)) % W  # [BS, W]
        cumulative_overwrite = (
            slot_age.unsqueeze(1) <= offsets.view(1, P, 1)
        )  # [BS, P, W]

        # Cache valid per query: original validity minus overwritten positions
        cache_valid_per_query = self.wm_valid.unsqueeze(1).expand(BS, P, W) & ~cumulative_overwrite
        # [BS, P, W]

        # Span portion: lower-triangular causal mask (token t attends to 0..t)
        span_causal = self._span_causal[:P, :P].unsqueeze(0).expand(BS, P, P)  # [BS, P, P]

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
        attn = attn + self._alibi_bias_span(BS, P, write_pos)

        # Apply mask: [BS, 1, P, W+P] broadcast over heads
        attn_mask_4d = attn_mask.unsqueeze(1)  # [BS, 1, P, W+P]
        attn = attn.masked_fill(~attn_mask_4d, float("-inf"))

        # Handle all-invalid rows (e.g. first token with empty WM cache)
        all_invalid = (~attn_mask_4d).all(dim=-1, keepdim=True)  # [BS, 1, P, 1]
        attn = torch.softmax(attn, dim=-1)
        attn = torch.where(all_invalid, torch.zeros_like(attn), attn)
        attn = self.attn_drop(attn)

        # Weighted sum: [BS, n_heads, P, head_dim]
        out = torch.matmul(attn, V_h)

        # Merge heads: [BS, P, D_wm]
        out = out.transpose(1, 2).reshape(BS, P, self.D_wm)

        # Output projection: [BS, P, D]
        y_wm_all = self.drop_resid(self.W_o(out))

        # --- Differentiable cache update (write all P tokens at once) ---
        # P <= W is guaranteed by config validation, so write_pos is unique per stream.
        scatter_idx = write_pos.unsqueeze(-1).expand(-1, -1, self.D_wm)  # [BS, P, D_wm]
        self.wm_K = self.wm_K.scatter(1, scatter_idx, k_all)
        self.wm_V = self.wm_V.scatter(1, scatter_idx, v_all)

        # --- Update wm_valid and advance wm_ptr ---
        new_valid = self.wm_valid.clone()
        # Mark all written positions as valid
        new_valid.scatter_(1, write_pos, True)
        self.wm_valid = new_valid

        self.wm_ptr = (wm_ptr + P) % W

        return y_wm_all


# ---------------------------------------------------------------------------
# GLA recurrence (pure PyTorch — torch.compile fuses the loop)
# ---------------------------------------------------------------------------

def _gla_recurrence(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, state: Tensor,
    reset_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pure PyTorch GLA recurrence over a span of P tokens.

    Args:
        q: [BS, P, H, K]  — queries
        k: [BS, P, H, K]  — keys
        v: [BS, P, H, V]  — values
        g: [BS, P, H, K]  — log-space decay gates (negative)
        state: [BS, H, K, V] — recurrent state matrix (bf16 on CUDA, fp32 on CPU)
        reset_mask: [BS, P] bool or None — True at doc boundaries;
            state is zeroed *before* computing that token's output.

    Returns:
        output: [BS, P, H, V]
        new_state: [BS, H, K, V]
    """
    BS, P, H, V = v.shape
    # Preallocate output tensor (avoids list + stack)
    outputs = torch.empty(BS, P, H, V, dtype=q.dtype, device=q.device)
    # Hoist exp outside the loop (one fused kernel instead of P separate ones)
    decay_all = torch.exp(g)  # [BS, P, H, K]
    for t in range(P):
        # Mid-span doc boundary: zero state before this token
        if reset_mask is not None:
            keep = (~reset_mask[:, t]).to(state.dtype).view(-1, 1, 1, 1)  # [BS,1,1,1]
            state = state * keep
        # state update: S_t = diag(decay) @ S_{t-1} + k_t^T @ v_t
        state = state * decay_all[:, t].unsqueeze(-1) + \
            k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)   # [BS, H, K, V]
        # output: o_t = q_t @ S_t
        outputs[:, t] = torch.matmul(q[:, t].unsqueeze(-2), state).squeeze(-2)
    return outputs, state  # [BS, P, H, V], [BS, H, K, V]


class GLAWorkingMemory(nn.Module, StateMixin):
    """Gated Linear Attention working memory.

    Replaces softmax attention + ring buffer with a recurrent linear
    attention mechanism. The state is a [H, K, V] matrix per stream
    (fixed size, O(1) per token). Learned log-space gates control
    the decay of old information (replaces ALiBi).

    Pure PyTorch implementation — torch.compile fuses the P=64 recurrence
    loop into a single kernel.
    """
    _state_tensor_names = ["gla_state"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.D_wm = config.D_wm
        self.n_heads = config.n_heads_wm
        self.head_dim = config.D_wm // config.n_heads_wm
        assert config.D_wm % config.n_heads_wm == 0

        # Q/K/V/O projections (same dims as softmax WM)
        self.W_q = nn.Linear(config.D, config.D_wm)
        self.W_k = nn.Linear(config.D, config.D_wm)
        self.W_v = nn.Linear(config.D, config.D_wm)
        self.W_o = nn.Linear(config.D_wm, config.D)

        # Gate projection: learned recency via log-space gates (replaces ALiBi)
        # Low-rank bottleneck keeps param count small
        self.W_g = nn.Sequential(
            nn.Linear(config.D, config.gate_low_rank, bias=False),
            nn.Linear(config.gate_low_rank, config.D_wm, bias=True),
        )
        self.gate_logit_normalizer = 16

        self.drop_resid = nn.Dropout(config.dropout)

        # Recurrent state: [BS, H, head_dim, head_dim]
        self.gla_state: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize recurrent state on first forward pass."""
        self.gla_state = torch.zeros(
            BS, self.n_heads, self.head_dim, self.head_dim,
            dtype=runtime_state_dtype(device), device=device,
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Handle loading old softmax WM checkpoints.

        W_q/W_k/W_v/W_o transfer directly (same shapes).
        Old ring buffer state and ALiBi buffers are ignored.
        W_g initializes randomly (no old equivalent).
        """
        # Remove old softmax-specific keys that don't exist in GLA
        old_keys = [
            "alibi_slopes", "_span_causal",
        ]
        for key in old_keys:
            full_key = prefix + key
            if full_key in state_dict:
                del state_dict[full_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def step(self, x: Tensor, reset_mask: Tensor) -> Tensor:
        """Process one token through GLA working memory.

        Args:
            x: [BS, D] — current token embedding
            reset_mask: [BS] bool — True for streams at doc boundary

        Returns:
            y_wm: [BS, D] — working memory output
        """
        BS = x.shape[0]
        device = x.device

        if self.gla_state is None:
            self._lazy_init(BS, device)

        # Zero state for reset streams (unconditional tensor ops)
        keep = (~reset_mask).to(self.gla_state.dtype).view(BS, 1, 1, 1)
        self.gla_state = self.gla_state * keep

        # Project
        q = self.W_q(x).view(BS, self.n_heads, self.head_dim)
        k = self.W_k(x).view(BS, self.n_heads, self.head_dim)
        v = self.W_v(x).view(BS, self.n_heads, self.head_dim)
        g = F.logsigmoid(
            self.W_g(x).view(BS, self.n_heads, self.head_dim)
        ) / self.gate_logit_normalizer  # [BS, H, K]

        # Single recurrence step
        decay = torch.exp(g)                                # [BS, H, K]
        self.gla_state = self.gla_state * decay.unsqueeze(-1) + \
            k.unsqueeze(-1) * v.unsqueeze(-2)               # [BS, H, K, V]
        out = torch.matmul(q.unsqueeze(-2), self.gla_state).squeeze(-2)  # [BS, H, V]

        # Merge heads and project
        out = out.reshape(BS, self.D_wm)
        y_wm = self.drop_resid(self.W_o(out))  # [BS, D]

        return y_wm

    def forward_span(self, x_all: Tensor, reset_mask_all: Tensor) -> Tensor:
        """Process P tokens in parallel through GLA working memory.

        Args:
            x_all: [BS, P, D] — token embeddings for all span tokens
            reset_mask_all: [BS, P] bool — True at doc boundaries

        Returns:
            y_wm_all: [BS, P, D] — working memory outputs
        """
        BS, P, D = x_all.shape

        if self.gla_state is None:
            self._lazy_init(BS, x_all.device)

        # Batch projections: [BS, P, D_wm] -> [BS, P, H, head_dim]
        q = self.W_q(x_all).view(BS, P, self.n_heads, self.head_dim)
        k = self.W_k(x_all).view(BS, P, self.n_heads, self.head_dim)
        v = self.W_v(x_all).view(BS, P, self.n_heads, self.head_dim)
        g = F.logsigmoid(
            self.W_g(x_all).view(BS, P, self.n_heads, self.head_dim)
        ) / self.gate_logit_normalizer  # [BS, P, H, K]

        # Pure PyTorch recurrence with per-token resets at doc boundaries
        out, self.gla_state = _gla_recurrence(
            q, k, v, g, self.gla_state, reset_mask=reset_mask_all,
        )
        # out: [BS, P, H, V]

        # Merge heads and project: [BS, P, D]
        out = out.reshape(BS, P, self.D_wm)
        y_wm_all = self.drop_resid(self.W_o(out))

        return y_wm_all
