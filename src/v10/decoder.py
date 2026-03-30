"""Causal Transformer Decoder with Sliding-Window Cross-Attention.

Upper decoder ("frontal cortex") for v10-gnn. Runs ONCE after the memory
graph simulates all T steps. Processes all T tokens in parallel using
standard causal self-attention and sliding-window cross-attention to
word_states from the memory graph.

The memory graph is the ONLY path from input to output -- there is no
H_mid skip connection. Short-term context comes from the sliding window
of word_states (last W timesteps visible per query position).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.scan import RMSNorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SiLUGatedFFN(nn.Module):
    """LLaMA-style gated FFN: gate(x) * up(x), then down."""

    def __init__(self, D: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(D, d_ff, bias=False)
        self.w_up = nn.Linear(D, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, D, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class _MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional mask (causal or custom)."""

    def __init__(self, D_q: int, D_kv: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        assert D_q % n_heads == 0
        self.n_heads = n_heads
        self.d_head = D_q // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(D_q, D_q, bias=False)
        self.k_proj = nn.Linear(D_kv, D_q, bias=False)
        self.v_proj = nn.Linear(D_kv, D_q, bias=False)
        self.out_proj = nn.Linear(D_q, D_q, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor,
                mask: Tensor | None = None) -> Tensor:
        """
        Args:
            q:    [BS, Tq, D_q]
            kv:   [BS, Tkv, D_kv]
            mask:  bool tensor, True = **allowed**, False = masked.
                  Shape [Tq, Tkv] or [BS, Tq, Tkv] or [BS, n_heads, Tq, Tkv].

        Returns:
            out: [BS, Tq, D_q]
        """
        BS, Tq, _ = q.shape
        Tkv = kv.size(1)
        H, d = self.n_heads, self.d_head

        Q = self.q_proj(q).view(BS, Tq, H, d).transpose(1, 2)   # [BS,H,Tq,d]
        K = self.k_proj(kv).view(BS, Tkv, H, d).transpose(1, 2)  # [BS,H,Tkv,d]
        V = self.v_proj(kv).view(BS, Tkv, H, d).transpose(1, 2)  # [BS,H,Tkv,d]

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [BS,H,Tq,Tkv]

        if mask is not None:
            # Expand mask to [BS, H, Tq, Tkv] if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)     # [1,1,Tq,Tkv]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)                  # [BS,1,Tq,Tkv]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(BS, Tq, H * d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single decoder layer: causal self-attn + sliding-window cross-attn + FFN."""

    def __init__(self, D_dec: int, D_scan: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0):
        super().__init__()
        # Self-attention (causal)
        self.ln_self = RMSNorm(D_dec)
        self.self_attn = _MultiHeadAttention(D_dec, D_dec, n_heads, dropout)

        # Cross-attention (sliding window over word_states)
        self.ln_cross = RMSNorm(D_dec)
        self.cross_attn = _MultiHeadAttention(D_dec, D_scan, n_heads, dropout)

        # FFN
        self.ln_ffn = RMSNorm(D_dec)
        self.ffn = _SiLUGatedFFN(D_dec, d_ff, dropout)

    def forward(self, x: Tensor, kv_cross: Tensor,
                causal_mask: Tensor, cross_mask: Tensor) -> Tensor:
        """
        Args:
            x:           [BS, T, D_dec]
            kv_cross:    [BS, T*num_words, D_scan]
            causal_mask: [T, T] bool — True = allowed
            cross_mask:  [T, T*num_words] bool — sliding window

        Returns:
            x: [BS, T, D_dec]
        """
        # Causal self-attention
        h = self.ln_self(x)
        x = x + self.self_attn(h, h, mask=causal_mask)

        # Sliding-window cross-attention
        h = self.ln_cross(x)
        x = x + self.cross_attn(h, kv_cross, mask=cross_mask)

        # FFN
        h = self.ln_ffn(x)
        x = x + self.ffn(h)

        return x


# ---------------------------------------------------------------------------
# Sliding Window Decoder
# ---------------------------------------------------------------------------

def _build_causal_mask(T: int, device: torch.device) -> Tensor:
    """Lower-triangular causal mask. True = allowed (can attend)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def _build_sliding_window_cross_mask(
    T: int, num_words: int, W: int, device: torch.device,
) -> Tensor:
    """Sliding-window cross-attention mask.

    Query position t can attend to KV positions from steps
    max(0, t - W + 1) to t (inclusive). Each step has `num_words`
    KV entries (contiguous in the flattened KV sequence).

    Returns:
        mask: [T, T * num_words] bool — True = allowed
    """
    Tkv = T * num_words
    mask = torch.zeros(T, Tkv, dtype=torch.bool, device=device)
    for t in range(T):
        s_start = max(0, t - W + 1)
        kv_start = s_start * num_words
        kv_end = (t + 1) * num_words  # exclusive
        mask[t, kv_start:kv_end] = True
    return mask


class SlidingWindowDecoder(nn.Module):
    """Causal transformer decoder with sliding-window cross-attention.

    Processes all T tokens in parallel. Self-attention uses a standard
    causal mask. Cross-attention to word_states uses a sliding window
    of width W (each query sees the last W timesteps of memory words).

    Args:
        D_dec:       Decoder hidden dimension.
        D_scan:      Dimension of word_states (= neurons_per_word * D_neuron).
        n_heads:     Number of attention heads.
        n_layers:    Number of decoder layers.
        d_ff:        FFN intermediate dimension.
        W_sliding:   Sliding window width for cross-attention.
        vocab_size:  Output vocabulary size.
        D_embed:     Embedding dimension (for tie_embeddings and final proj).
        dropout:     Dropout rate.
        tie_embeddings: If a weight tensor is provided, tie lm_head to it.
    """

    def __init__(
        self,
        D_dec: int,
        D_scan: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        W_sliding: int,
        vocab_size: int,
        D_embed: int,
        dropout: float = 0.0,
        tie_embeddings: Tensor | None = None,
    ):
        super().__init__()
        self.D_dec = D_dec
        self.D_scan = D_scan
        self.W_sliding = W_sliding
        self.vocab_size = vocab_size
        self.D_embed = D_embed

        # Project mean-pooled word_states → decoder dim
        self.query_proj = nn.Linear(D_scan, D_dec, bias=False)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(D_dec, D_scan, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final norm
        self.ln_final = RMSNorm(D_embed)

        # Optional projection from D_dec → D_embed (if they differ)
        if D_dec != D_embed:
            self.proj_down = nn.Linear(D_dec, D_embed, bias=False)
        else:
            self.proj_down = None

        # LM head
        self.lm_head = nn.Linear(D_embed, vocab_size, bias=False)
        if tie_embeddings is not None:
            self.lm_head.weight = tie_embeddings

    def forward(self, word_states: Tensor) -> Tensor:
        """
        Args:
            word_states: [BS, T, num_words, D_scan]
                Memory graph neuron states grouped into words,
                collected at every simulation step.

        Returns:
            logits: [BS, T, vocab_size]
        """
        BS, T, num_words, D_scan = word_states.shape
        device = word_states.device

        # Queries: mean-pool across words at each timestep, then project
        # [BS, T, num_words, D_scan] → [BS, T, D_scan] → [BS, T, D_dec]
        queries = self.query_proj(word_states.mean(dim=2))

        # KV for cross-attention: flatten words into sequence
        # [BS, T, num_words, D_scan] → [BS, T*num_words, D_scan]
        kv_cross = word_states.reshape(BS, T * num_words, D_scan)

        # Build masks (once, reused across layers)
        causal_mask = _build_causal_mask(T, device)
        cross_mask = _build_sliding_window_cross_mask(
            T, num_words, self.W_sliding, device)

        # Run decoder layers
        x = queries
        for layer in self.layers:
            x = layer(x, kv_cross, causal_mask, cross_mask)

        # Final projection
        if self.proj_down is not None:
            x = self.proj_down(x)
        x = self.ln_final(x)

        logits = self.lm_head(x)  # [BS, T, vocab_size]
        return logits
