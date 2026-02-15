"""
Spatial Decoder — hierarchical aggregation of NML intermediate outputs.

Three-level architecture inspired by cortical organization:

  Level 1 (Columnar):  Per-block attention across L layer outputs → column summary.
                        Analogy: cortical column integrating across its laminar layers.

  Level 2 (Thalamic):  Cross-block + memory-type integration → K integrated tokens.
                        Analogy: thalamus binding cortical regions with memory systems.

  Level 3 (Decoder):   Multi-layer cross-attention decoder → enhanced hidden state.
                        Analogy: language production area reading from organized memory.

When snapshot_enabled=False, none of this is instantiated.  The model falls back
to the original path: concat(h_blocks) → lm_head.

When snapshot_enabled=True, output_proj is small-initialized (std=0.01) so the
decoder starts near-identity (h_final + noise ≈ h_final) while allowing gradient
flow.  Zero-init would kill all upstream gradients via chain rule.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig


# ============================================================================
# Level 1: Columnar Attention (one per block)
# ============================================================================

class ColumnarAttention(nn.Module):
    """Summarize L layer outputs within a single block.

    A learned summary query cross-attends to all layer outputs (tagged with
    layer-position embeddings), refined through multiple layers.
    Produces one column summary vector per block.

    Input:  layer_outputs [BS, L, D_h]
    Output: column_summary [BS, D_h]
    """

    def __init__(self, D_h: int, L: int, n_heads: int, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.L = L
        self.layer_emb = nn.Embedding(L, D_h)
        self.summary_query = nn.Parameter(torch.zeros(1, 1, D_h))
        nn.init.normal_(self.summary_query, std=0.02)
        self.drop = nn.Dropout(dropout)

        # Multi-layer refinement: each layer is cross-attn + FFN
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm_ca": nn.LayerNorm(D_h),
                "cross_attn": nn.MultiheadAttention(D_h, n_heads,
                                                     batch_first=True,
                                                     dropout=dropout),
                "norm_ff": nn.LayerNorm(D_h),
                "ffn": nn.Sequential(
                    nn.Linear(D_h, D_h * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D_h * 4, D_h),
                ),
            }))
        self.final_norm = nn.LayerNorm(D_h)

    def forward(self, layer_outputs: Tensor) -> Tensor:
        """
        layer_outputs: [BS, L, D_h]
        Returns: [BS, D_h]
        """
        BS = layer_outputs.shape[0]
        device = layer_outputs.device

        # Add layer-position embeddings
        pos = self.layer_emb(torch.arange(self.L, device=device))  # [L, D_h]
        kv = layer_outputs + pos.unsqueeze(0)  # [BS, L, D_h]

        # Iteratively refine the summary query
        q = self.summary_query.expand(BS, -1, -1)  # [BS, 1, D_h]
        for layer in self.layers:
            # Pre-norm cross-attention
            q_norm = layer["norm_ca"](q)
            q = q + self.drop(layer["cross_attn"](q_norm, kv, kv)[0])
            # Pre-norm FFN
            q_norm = layer["norm_ff"](q)
            q = q + self.drop(layer["ffn"](q_norm))

        return self.final_norm(q.squeeze(1))  # [BS, D_h]


# ============================================================================
# Level 2: Thalamic Integrator
# ============================================================================

class ThalamicIntegrator(nn.Module):
    """Bind column summaries + memory readouts into integrated tokens.

    Input tokens (all projected to d_dec):
      - B column summaries   (cortical processing)
      - 1 PM readout         (procedural knowledge)
      - 1 EM readout         (episodic recall)
      - 1 WM readout         (recent context)
    Total: B + 3 input tokens.

    Learned output queries cross-attend to these, producing K integrated
    memory tokens for the decoder.

    Output: [BS, K, d_dec]
    """

    def __init__(self, d_dec: int, K: int, B: int, n_heads: int,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.K = K
        self.B = B

        # Type embeddings: 0=cortical, 1=PM, 2=EM, 3=WM
        self.type_emb = nn.Embedding(4, d_dec)
        # Block position embeddings (for cortical tokens only)
        self.block_emb = nn.Embedding(B, d_dec)

        # Learned output queries
        self.output_queries = nn.Parameter(torch.zeros(1, K, d_dec))
        nn.init.normal_(self.output_queries, std=0.02)
        self.drop = nn.Dropout(dropout)

        # Multi-layer refinement: each layer is cross-attn + FFN
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm_ca": nn.LayerNorm(d_dec),
                "cross_attn": nn.MultiheadAttention(d_dec, n_heads,
                                                     batch_first=True,
                                                     dropout=dropout),
                "norm_ff": nn.LayerNorm(d_dec),
                "ffn": nn.Sequential(
                    nn.Linear(d_dec, d_dec * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_dec * 4, d_dec),
                ),
            }))
        self.final_norm = nn.LayerNorm(d_dec)

    def forward(self, cortical: Tensor, pm: Tensor,
                em: Tensor, wm: Tensor) -> Tensor:
        """
        cortical: [BS, B, d_dec]  — projected column summaries
        pm:       [BS, d_dec]     — projected PM summary
        em:       [BS, d_dec]     — projected EM summary
        wm:       [BS, d_dec]     — projected WM summary

        Returns:  [BS, K, d_dec]  — integrated memory tokens
        """
        BS = cortical.shape[0]
        device = cortical.device

        # Tag cortical tokens with type=0 + block position
        block_ids = torch.arange(self.B, device=device)
        cortical = cortical + self.type_emb(torch.zeros_like(block_ids)) + self.block_emb(block_ids)

        # Tag memory tokens with their type
        pm_tok = pm.unsqueeze(1) + self.type_emb(torch.tensor(1, device=device))   # [BS, 1, d_dec]
        em_tok = em.unsqueeze(1) + self.type_emb(torch.tensor(2, device=device))   # [BS, 1, d_dec]
        wm_tok = wm.unsqueeze(1) + self.type_emb(torch.tensor(3, device=device))   # [BS, 1, d_dec]

        # Assemble all memory tokens
        memory = torch.cat([cortical, pm_tok, em_tok, wm_tok], dim=1)  # [BS, B+3, d_dec]

        # Iteratively refine the output queries
        q = self.output_queries.expand(BS, -1, -1)  # [BS, K, d_dec]
        for layer in self.layers:
            # Pre-norm cross-attention to memory
            q_norm = layer["norm_ca"](q)
            q = q + self.drop(layer["cross_attn"](q_norm, memory, memory)[0])
            # Pre-norm FFN
            q_norm = layer["norm_ff"](q)
            q = q + self.drop(layer["ffn"](q_norm))

        return self.final_norm(q)  # [BS, K, d_dec]


# ============================================================================
# Level 3: Decoder Block (standard pre-norm transformer decoder layer)
# ============================================================================

class DecoderBlock(nn.Module):
    """Single decoder layer: self-attention + cross-attention + FFN."""

    def __init__(self, d_dec: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_dec)
        self.self_attn = nn.MultiheadAttention(
            d_dec, n_heads, batch_first=True, dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_dec)
        self.cross_attn = nn.MultiheadAttention(
            d_dec, n_heads, batch_first=True, dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(d_dec)
        self.ffn = nn.Sequential(
            nn.Linear(d_dec, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_dec),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        x:      [BS, S, d_dec]  — decoder query sequence (S=1 for single-token)
        memory: [BS, K, d_dec]  — integrated memory tokens from thalamic integrator

        Returns: [BS, S, d_dec]
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        x = x + self.drop(self.self_attn(x_norm, x_norm, x_norm)[0])

        # Pre-norm cross-attention to memory
        x_norm = self.norm2(x)
        x = x + self.drop(self.cross_attn(x_norm, memory, memory)[0])

        # Pre-norm FFN
        x_norm = self.norm3(x)
        x = x + self.drop(self.ffn(x_norm))

        return x


# ============================================================================
# SpatialDecoder: composes all three levels
# ============================================================================

class SpatialDecoder(nn.Module):
    """Full spatial decoder pipeline.

    Takes:
      - Per-block layer outputs (from Block.step with return_layers=True)
      - PM/EM/WM state summaries
      - h_final (concatenated block outputs)

    Produces:
      - h_decoded [BS, D] — enhanced hidden state for lm_head

    output_proj is small-initialized so h_decoded ≈ h_final at startup.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        D = config.D
        D_h = config.D_h
        D_em = config.D_em
        d_dec = config.d_dec
        B = config.B
        L = config.L
        K = config.thalamic_tokens
        n_heads = config.n_heads_decoder

        # Level 1: Columnar attention (one per block)
        col_heads = max(1, D_h // 32)  # e.g. 128//32 = 4 heads
        # Ensure D_h is divisible by col_heads for MultiheadAttention
        while col_heads > 1 and D_h % col_heads != 0:
            col_heads -= 1
        drop = config.dropout
        self.columnar = nn.ModuleList([
            ColumnarAttention(D_h, L, n_heads=col_heads,
                              n_layers=config.columnar_layers,
                              dropout=drop)
            for _ in range(B)
        ])

        # Projections to d_dec for Level 2 inputs
        self.col_proj = nn.Linear(D_h, d_dec)
        self.pm_proj = nn.Linear(D_h, d_dec)
        self.em_proj = nn.Linear(D_em, d_dec)
        self.wm_proj = nn.Linear(D, d_dec)

        # Level 2: Thalamic integrator
        self.thalamic = ThalamicIntegrator(d_dec, K, B, n_heads,
                                           n_layers=config.thalamic_layers,
                                           dropout=drop)

        # Level 3: Deep decoder
        self.query_proj = nn.Linear(D, d_dec)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_dec, n_heads, d_dec * 4, dropout=drop)
            for _ in range(config.decoder_layers)
        ])

        # Project back to D for lm_head.
        # Small-init (not zero!) so gradients can flow through the decoder.
        # The residual h_final + context means h_final still dominates at startup.
        self.output_proj = nn.Linear(d_dec, D, bias=False)
        nn.init.normal_(self.output_proj.weight, std=0.01)

    def forward(
        self,
        block_layer_outputs: list[Tensor],
        pm_summary: Tensor,
        em_summary: Tensor,
        wm_output: Tensor,
        h_final: Tensor,
    ) -> Tensor:
        """
        block_layer_outputs: list of B tensors, each [BS, L, D_h]
        pm_summary:  [BS, D_h]  — strength-weighted PM readout (zeros if disabled)
        em_summary:  [BS, D_em] — strength-weighted EM readout (zeros if disabled)
        wm_output:   [BS, D]    — WM step output
        h_final:     [BS, D]    — concatenated final block outputs (existing path)

        Returns: [BS, D] — enhanced hidden state
        """
        # Level 1: Columnar summaries
        col_summaries = []
        for b, columnar in enumerate(self.columnar):
            col_summaries.append(columnar(block_layer_outputs[b]))  # [BS, D_h]
        col_stack = torch.stack(col_summaries, dim=1)  # [BS, B, D_h]

        # Project all inputs to d_dec
        cortical = self.col_proj(col_stack)       # [BS, B, d_dec]
        pm_dec = self.pm_proj(pm_summary)          # [BS, d_dec]
        em_dec = self.em_proj(em_summary)           # [BS, d_dec]
        wm_dec = self.wm_proj(wm_output)            # [BS, d_dec]

        # Level 2: Thalamic integration
        memory = self.thalamic(cortical, pm_dec, em_dec, wm_dec)  # [BS, K, d_dec]

        # Level 3: Deep decoder
        q = self.query_proj(h_final).unsqueeze(1)  # [BS, 1, d_dec]
        for block in self.decoder_blocks:
            q = block(q, memory)                    # [BS, 1, d_dec]

        # Project back to D and add as residual
        context = self.output_proj(q.squeeze(1))    # [BS, D]
        h_decoded = h_final + context               # [BS, D]

        return h_decoded
