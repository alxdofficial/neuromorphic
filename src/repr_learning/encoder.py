"""Encoder modules: V2.1 + two baselines.

All three share a common bidirectional transformer over the input text.
They differ in how the encoder output (text representation) is turned
into memory tokens for the frozen Llama decoder.

V2.1Encoder           : 32 edges = (src_id, dst_id, edge_vec) triples
FlatBaselineEncoder   : 96 independent code picks, no edge structure
ContinuousBaselineEncoder : 96 continuous vectors, no quantization

All three return:
    memory_tokens : [B, n_memory_tokens, d_llama]
    aux_outputs   : dict with auxiliary losses + stats (e.g., load_balance)
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ReprConfig
from .selection import gumbel_argmax_ste, load_balance_loss


class SmallBiTransformer(nn.Module):
    """A small bidirectional transformer encoder over Llama token embeddings.

    Takes [B, T, d_llama] token embeddings as input (Llama's embed_tokens
    output, which we keep frozen). Outputs [B, T, d_enc] contextualized
    representations.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        # Project d_llama → d_enc
        self.in_proj = nn.Linear(cfg.d_llama, cfg.d_enc, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_enc,
            nhead=cfg.enc_n_heads,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.enc_n_layers)
        # Learned positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.window_size_max, cfg.d_enc),
        )
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(self, token_embeds: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # token_embeds: [B, T, d_llama]; cast to encoder dtype (fp32 trainable)
        h = self.in_proj(token_embeds.to(self.in_proj.weight.dtype))   # [B, T, d_enc]
        T = h.shape[1]
        h = h + self.pos_embed[:, :T, :]                   # add position
        if attention_mask is not None:
            # nn.TransformerEncoder expects True = mask (don't attend)
            src_key_padding_mask = ~attention_mask         # True where padded
        else:
            src_key_padding_mask = None
        out = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        return out                                          # [B, T, d_enc]


class _SlotCrossAttn(nn.Module):
    """Cross-attention from learned slot queries to text representations.

    Used by all three encoders to pool text into a fixed number of slots
    (one slot per edge or per code).
    """

    def __init__(self, d_enc: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_enc,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_enc)
        self.norm_kv = nn.LayerNorm(d_enc)

    def forward(
        self, queries: Tensor, kv: Tensor, key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # queries: [B, N_slots, d_enc]   kv: [B, T, d_enc]
        q = self.norm_q(queries)
        k = self.norm_kv(kv)
        out, _ = self.attn(
            q, k, k,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out + queries  # residual


class V21Encoder(nn.Module):
    """V2.1 encoder: 32 edges = (src_id, dst_id, edge_vec) triples.

    Pipeline:
        text_embeds → BiTransformer → text_h
        slot_queries (learned) → cross-attn over text_h → edge_slots
        per-slot heads → (src_q, dst_q, edge_vec)
        classify src_q and dst_q against codebook → (src_id, dst_id)
        package per-edge as 3 memory tokens (src, edge, dst) interleaved
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        # Shared bi-transformer + slot cross-attention
        self.bi_transformer = SmallBiTransformer(cfg)
        self.edge_queries = nn.Parameter(torch.zeros(cfg.n_edges, cfg.d_enc))
        nn.init.normal_(self.edge_queries, std=0.02)
        self.slot_attn = _SlotCrossAttn(cfg.d_enc, cfg.enc_n_heads)

        # Per-slot heads: src_q, dst_q (D_concept), edge_vec (D_edge)
        self.src_head = nn.Sequential(
            nn.Linear(cfg.d_enc, cfg.d_enc), nn.GELU(),
            nn.Linear(cfg.d_enc, cfg.d_concept),
        )
        self.dst_head = nn.Sequential(
            nn.Linear(cfg.d_enc, cfg.d_enc), nn.GELU(),
            nn.Linear(cfg.d_enc, cfg.d_concept),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(cfg.d_enc, cfg.d_enc), nn.GELU(),
            nn.Linear(cfg.d_enc, cfg.d_edge),
        )

        # Concept_id codebook: N × D_concept. Slow weights, learned by backprop.
        self.concept_id = nn.Parameter(torch.zeros(cfg.n_nodes, cfg.d_concept))
        nn.init.normal_(self.concept_id, std=1.0 / cfg.d_concept ** 0.5)

        # Projections to Llama d_model. One projection per token role.
        # src and dst project D_concept → d_llama (separate weights to distinguish roles)
        # edge projects D_edge → d_llama
        # LayerNorm on output keeps memory token magnitudes bounded
        # (Llama expects RMS ~ 1.0 per-token, not arbitrary).
        def _proj(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_out // 2), nn.GELU(),
                nn.Linear(d_out // 2, d_out),
                nn.LayerNorm(d_out),
            )
        self.proj_src = _proj(cfg.d_concept, cfg.d_llama)
        self.proj_dst = _proj(cfg.d_concept, cfg.d_llama)
        self.proj_edge = _proj(cfg.d_edge, cfg.d_llama)

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        cfg = self.cfg
        B = token_embeds.shape[0]

        # 1. Bi-transformer over text
        text_h = self.bi_transformer(token_embeds, attention_mask)   # [B, T, d_enc]

        # 2. Cross-attn from edge_queries to text
        edge_queries = self.edge_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, d_enc]
        kv_mask = ~attention_mask if attention_mask is not None else None
        edge_slots = self.slot_attn(edge_queries, text_h, key_padding_mask=kv_mask)  # [B, K, d_enc]

        # 3. Per-slot heads
        src_q = self.src_head(edge_slots)     # [B, K, D_concept]
        dst_q = self.dst_head(edge_slots)     # [B, K, D_concept]
        edge_vec = self.edge_head(edge_slots) # [B, K, D_edge]

        # 4. Classify against codebook (dot product → softmax)
        src_scores = src_q @ self.concept_id.T       # [B, K, N]
        dst_scores = dst_q @ self.concept_id.T       # [B, K, N]
        src_id, src_onehot = gumbel_argmax_ste(
            src_scores, cfg.selection_temperature, self.training,
        )
        dst_id, dst_onehot = gumbel_argmax_ste(
            dst_scores, cfg.selection_temperature, self.training,
        )

        # 5. Gather chosen concept_id embeddings (differentiable via one_hot)
        src_emb = src_onehot @ self.concept_id        # [B, K, D_concept]
        dst_emb = dst_onehot @ self.concept_id        # [B, K, D_concept]

        # 6. Project to Llama d_model
        src_tok = self.proj_src(src_emb)              # [B, K, d_llama]
        dst_tok = self.proj_dst(dst_emb)              # [B, K, d_llama]
        edge_tok = self.proj_edge(edge_vec)           # [B, K, d_llama]

        # 7. Interleave to 3K memory tokens: [src_0, edge_0, dst_0, src_1, edge_1, dst_1, ...]
        memory = torch.stack([src_tok, edge_tok, dst_tok], dim=2)  # [B, K, 3, d_llama]
        memory = memory.reshape(B, 3 * cfg.n_edges, cfg.d_llama)   # [B, 3K, d_llama]

        # 8. Aux outputs
        aux = {
            "load_balance_loss": (
                load_balance_loss(src_scores) + load_balance_loss(dst_scores)
            ) * 0.5,
            "src_ids": src_id,
            "dst_ids": dst_id,
            "src_scores_max": src_scores.max(dim=-1).values.mean(),
            "src_scores_entropy": -(
                F.softmax(src_scores, dim=-1)
                * F.log_softmax(src_scores, dim=-1)
            ).sum(-1).mean(),
        }
        return memory, aux


class FlatBaselineEncoder(nn.Module):
    """Baseline A: flat classification, 96 independent codes per window.

    No edge structure, no triples. 96 slot queries each pick one of 4096
    nodes. Same total memory token count (96) as V2.1 for fair comparison.

    Tests: does the (src, edge, dst) triple structure provide value over
    flat code selection?
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        self.bi_transformer = SmallBiTransformer(cfg)
        self.code_queries = nn.Parameter(torch.zeros(cfg.n_flat_codes, cfg.d_enc))
        nn.init.normal_(self.code_queries, std=0.02)
        self.slot_attn = _SlotCrossAttn(cfg.d_enc, cfg.enc_n_heads)

        self.code_head = nn.Sequential(
            nn.Linear(cfg.d_enc, cfg.d_enc), nn.GELU(),
            nn.Linear(cfg.d_enc, cfg.d_concept),
        )

        self.concept_id = nn.Parameter(torch.zeros(cfg.n_nodes, cfg.d_concept))
        nn.init.normal_(self.concept_id, std=1.0 / cfg.d_concept ** 0.5)

        self.proj_code = nn.Sequential(
            nn.Linear(cfg.d_concept, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            nn.LayerNorm(cfg.d_llama),
        )

    def forward(
        self, token_embeds: Tensor, attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        cfg = self.cfg
        B = token_embeds.shape[0]

        text_h = self.bi_transformer(token_embeds, attention_mask)
        code_queries = self.code_queries.unsqueeze(0).expand(B, -1, -1)
        kv_mask = ~attention_mask if attention_mask is not None else None
        code_slots = self.slot_attn(code_queries, text_h, key_padding_mask=kv_mask)

        code_q = self.code_head(code_slots)                       # [B, M, D_concept]
        scores = code_q @ self.concept_id.T                        # [B, M, N]
        code_id, onehot = gumbel_argmax_ste(
            scores, cfg.selection_temperature, self.training,
        )
        code_emb = onehot @ self.concept_id                        # [B, M, D_concept]
        memory = self.proj_code(code_emb)                          # [B, M, d_llama]

        aux = {
            "load_balance_loss": load_balance_loss(scores),
            "code_ids": code_id,
            "scores_max": scores.max(dim=-1).values.mean(),
        }
        return memory, aux


class ContinuousBaselineEncoder(nn.Module):
    """Baseline B: continuous bottleneck, no quantization.

    96 slot queries each produce a continuous vector of D_cont dim.
    No codebook, no classification, no discrete picking. Direct
    continuous-to-Llama projection.

    Tests: does the discrete node selection provide value over a
    matched-budget continuous bottleneck?

    D_cont is chosen so 96 × D_cont ≈ V2.1's edge_vec budget
    (32 × D_edge = 4096 floats) → D_cont ≈ 48.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        self.bi_transformer = SmallBiTransformer(cfg)
        self.cont_queries = nn.Parameter(torch.zeros(cfg.n_flat_codes, cfg.d_enc))
        nn.init.normal_(self.cont_queries, std=0.02)
        self.slot_attn = _SlotCrossAttn(cfg.d_enc, cfg.enc_n_heads)

        # Per-slot continuous head: d_enc → D_cont
        self.cont_head = nn.Sequential(
            nn.Linear(cfg.d_enc, cfg.d_enc), nn.GELU(),
            nn.Linear(cfg.d_enc, cfg.d_continuous),
        )

        # Project D_cont → d_llama
        self.proj_cont = nn.Sequential(
            nn.Linear(cfg.d_continuous, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            nn.LayerNorm(cfg.d_llama),
        )

    def forward(
        self, token_embeds: Tensor, attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        B = token_embeds.shape[0]
        text_h = self.bi_transformer(token_embeds, attention_mask)
        cont_queries = self.cont_queries.unsqueeze(0).expand(B, -1, -1)
        kv_mask = ~attention_mask if attention_mask is not None else None
        cont_slots = self.slot_attn(cont_queries, text_h, key_padding_mask=kv_mask)

        cont_vec = self.cont_head(cont_slots)                      # [B, M, D_cont]
        memory = self.proj_cont(cont_vec)                          # [B, M, d_llama]

        aux = {
            "load_balance_loss": torch.zeros((), device=memory.device),  # n/a
            "cont_vec_norm": cont_vec.norm(dim=-1).mean(),
        }
        return memory, aux
