"""Config dataclass for V2.1 representation learning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ReprConfig:
    """All hyperparameters for V2.1 representation learning.

    Default values match docs/v2.1_repr_learning.md.
    """

    # ── Llama backbone ─────────────────────────────────────────────────────
    llama_model: str = "meta-llama/Llama-3.2-1B"
    d_llama: int = 2048  # Llama-3.2-1B hidden size
    llama_vocab_size: int = 128_256

    # ── Encoder ────────────────────────────────────────────────────────────
    d_enc: int = 512                # encoder transformer hidden size
    enc_n_layers: int = 2
    enc_n_heads: int = 4
    enc_ffn_dim: int = 1024
    enc_dropout: float = 0.0

    # ── Bottleneck (V2.1) ──────────────────────────────────────────────────
    n_nodes: int = 4096             # codebook size
    d_concept: int = 1024           # concept_id vector dim (slow weights)
    d_edge: int = 128               # edge_vec dim (per-window state)
    n_edges: int = 32               # K — edges per window
    edge_token_packing: Literal["triple"] = "triple"
    # "triple" → 3 tokens per edge (src, edge, dst) interleaved → 96 total

    # ── Bottleneck (Baselines) ─────────────────────────────────────────────
    # Baseline A (flat classification): n_flat_codes slots, no edge structure
    n_flat_codes: int = 96          # match V2.1 memory token count
    # Baseline B (continuous): D_cont chosen for matched float budget vs V2.1's edge channel
    d_continuous: int = 48          # 96 × 48 ≈ 32 × 128 (V2.1 edge_vec budget)

    # ── Routing / selection ────────────────────────────────────────────────
    # We use classification-style picking (NOT VQ-VAE quantization).
    # Forward: argmax over scores; backward: Gumbel-softmax + STE.
    selection_temperature: float = 0.5
    load_balance_coef: float = 0.01

    # ── Training data ──────────────────────────────────────────────────────
    window_size_min: int = 128
    window_size_max: int = 384
    window_size_median: int = 256
    # For v0 we use fixed-size windows for simplicity. Variable later.
    fixed_window_size: int = 256

    # ── Masking ────────────────────────────────────────────────────────────
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 0.9
    mask_span_min: int = 5
    mask_span_max: int = 15

    # ── Loss ───────────────────────────────────────────────────────────────
    # Reconstruction loss is per-token CE on masked positions.
    # No VQ commitment loss (we use classification, not VQ-VAE).
    # Just load-balance aux loss to prevent classification collapse.

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 5000
    eval_every: int = 1000

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    # ── Validation ─────────────────────────────────────────────────────────
    def __post_init__(self):
        assert 0.0 <= self.mask_ratio_min <= self.mask_ratio_max <= 1.0
        assert self.mask_span_min <= self.mask_span_max
        assert self.window_size_min <= self.window_size_median <= self.window_size_max
        # For V2.1 with triple packing: memory tokens = 3 × n_edges
        # For baselines: memory tokens = n_flat_codes
        # Both should be the same for fair comparison.
        assert 3 * self.n_edges == self.n_flat_codes, (
            f"V2.1 memory tokens (3 × n_edges = {3*self.n_edges}) must match "
            f"baseline memory tokens (n_flat_codes = {self.n_flat_codes})"
        )

    @property
    def n_memory_tokens(self) -> int:
        """Total memory tokens prepended to Llama input."""
        return self.n_flat_codes  # same as 3 × n_edges for triple packing
