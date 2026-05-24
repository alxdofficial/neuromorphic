"""Encoder modules: V2.1 + four baselines.

All five share a common bidirectional transformer over the input text
(except RecurrentBaselineEncoder which uses Mamba). They differ in how
the encoder output is turned into memory tokens for frozen Llama.

V2.1Encoder              : 32 edges = (src_id, dst_id, edge_vec) triples
FlatBaselineEncoder      : 96 independent code picks, no edge structure
ContinuousBaselineEncoder: 96 continuous vectors, no quantization
MemorizingBaselineEncoder: per-token KV bank, query from unmasked positions,
                           top-96 retrieved (Memorising-Transformers-style)
RecurrentBaselineEncoder : Mamba SSM over 256 tokens, pool to 96

All five return:
    memory_tokens : [B, n_memory_tokens, d_llama]
    aux_outputs   : dict with auxiliary losses + stats (e.g., load_balance)
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ReprConfig
from .selection import gumbel_argmax_ste, load_balance_loss, router_z_loss


@torch.no_grad()
def _sinusoidal_pe(seq_len: int, d_model: int, offset: int = 0,
                    *, device=None, dtype=torch.float32) -> Tensor:
    """Standard sinusoidal positional encoding for write-side pin tokens
    (audit2 #5). Without this, graph/splat/plastic writes are order-
    invariant within a 1024-token window (their pooling/cross-attn over
    pins doesn't carry token position).

    Returns [seq_len, d_model] PE for positions [offset, offset+seq_len).
    """
    if d_model <= 0 or seq_len <= 0:
        return torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    positions = torch.arange(offset, offset + seq_len,
                              device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
    d_half = (d_model + 1) // 2
    div = torch.exp(torch.arange(d_half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))                       # [d/2]
    angles = positions * div                                                # [T, d/2]
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)[:, : pe[:, 0::2].shape[-1]]
    pe[:, 1::2] = torch.cos(angles)[:, : pe[:, 1::2].shape[-1]]
    return pe.to(dtype)


class QFormerAdapter(nn.Module):
    """BLIP-2 / Flamingo-style cross-attention adapter.

    Takes encoder memory tokens (`[B, M_in, d_llama]`) and produces a
    re-formatted set of "Llama-friendly" soft-prompt tokens via:
      1. Project memory tokens d_llama → d_adapter
      2. For each adapter layer:
         a. Cross-attention: learned queries attend to memory KV
         b. Self-attention among queries
         c. FFN
      3. Project queries d_adapter → d_llama
      4. Output `[B, M_out, d_llama]` for prepending to Llama input

    Designed for the case where the encoder's output structure (e.g. V2.1's
    src/edge/dst triplets) doesn't naturally match the input distribution
    Llama was pretrained to consume. The learned queries decouple "what
    Llama needs to see" from "what the encoder produced," and the cross-
    attention is the explicit interpretation layer that v0's direct
    projection lacked.

    Defaults sized to keep V2.1 + adapter under Mamba's 19M trainable
    params: d_adapter=320, 1 layer, 8 heads → ~3M params.
    """

    def __init__(self, cfg: ReprConfig, n_queries: int):
        super().__init__()
        d_in = cfg.d_llama
        d_a = cfg.qformer_d_adapter
        nh = cfg.qformer_n_heads
        d_ffn = cfg.qformer_ffn_mult * d_a

        # Projections into / out of adapter space
        self.in_proj = nn.Linear(d_in, d_a)
        self.out_proj = nn.Linear(d_a, d_in)
        self.out_norm = nn.LayerNorm(d_in)

        # Learned queries: a fresh "soft prompt" learned per query slot.
        # ORTHOGONAL init (not std=0.02) so 96 queries start in mutually
        # orthogonal directions in 320-dim space — gives cross-attention
        # something to differentiate from step 0. v1c showed std=0.02 init
        # collapses (queries near-identical → cross-attn outputs identical
        # → no signal to diversify). Same fix that rescued continuous_baseline.
        self.queries = nn.Parameter(torch.zeros(n_queries, d_a))
        nn.init.orthogonal_(self.queries)

        # Stack of (cross-attn → self-attn → FFN) blocks
        self.layers = nn.ModuleList()
        for _ in range(cfg.qformer_n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_q_norm": nn.LayerNorm(d_a),
                "cross_kv_norm": nn.LayerNorm(d_a),
                "cross_attn": nn.MultiheadAttention(d_a, nh, batch_first=True),
                "self_norm": nn.LayerNorm(d_a),
                "self_attn": nn.MultiheadAttention(d_a, nh, batch_first=True),
                "ffn_norm": nn.LayerNorm(d_a),
                "ffn": nn.Sequential(
                    nn.Linear(d_a, d_ffn), nn.GELU(),
                    nn.Linear(d_ffn, d_a),
                ),
            }))

    def forward(self, memory_in: Tensor) -> tuple[Tensor, Tensor]:
        # memory_in: [B, M_in, d_llama]
        B = memory_in.shape[0]
        kv = self.in_proj(memory_in)                          # [B, M_in, d_a]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)       # [B, M_out, d_a]

        for layer in self.layers:
            # Cross-attention: q attends to kv (memory)
            q_n = layer["cross_q_norm"](q)
            kv_n = layer["cross_kv_norm"](kv)
            cross_out, _ = layer["cross_attn"](q_n, kv_n, kv_n, need_weights=False)
            q = q + cross_out

            # Self-attention among queries
            s_n = layer["self_norm"](q)
            self_out, _ = layer["self_attn"](s_n, s_n, s_n, need_weights=False)
            q = q + self_out

            # FFN
            q = q + layer["ffn"](layer["ffn_norm"](q))

        out = self.out_proj(q)                                # [B, M_out, d_llama]
        out = self.out_norm(out)

        # Diversity loss on outputs (mean squared off-diagonal cosine).
        # v1c showed Q-Former outputs collapsed to ~98% pairwise cosine;
        # this is the same anti-collapse recipe as continuous_baseline.
        with torch.amp.autocast("cuda", enabled=False):
            o = F.normalize(out.float(), dim=-1)
            cos = o @ o.transpose(1, 2)                       # [B, M_out, M_out]
            M = cos.shape[1]
            eye = torch.eye(M, dtype=torch.bool, device=cos.device)
            off_diag = cos[:, ~eye].view(cos.shape[0], -1)
            diversity_loss = off_diag.pow(2).mean()
        return out, diversity_loss


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
        # Learned positional embedding. Sized to the largest expected window:
        # max(legacy window_size_max, v1b max_window_size). New entries are
        # randomly initialized; v0 ckpts loaded with strict=False keep their
        # 384-row subset and the trailing rows are fresh-trained.
        pos_max = max(cfg.window_size_max, cfg.max_window_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_max, cfg.d_enc),
        )
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(self, token_embeds: Tensor, attention_mask: Optional[Tensor] = None,
                position_offset: int = 0) -> Tensor:
        # token_embeds: [B, T, d_llama]; cast to encoder dtype (fp32 trainable)
        h = self.in_proj(token_embeds.to(self.in_proj.weight.dtype))   # [B, T, d_enc]
        T = h.shape[1]
        # position_offset enables window-aware encoding when called repeatedly
        # over windows of a longer chunk. v1g passes `position_offset=w*window_size`
        # per streaming write so token@chunk_pos=1500 gets pos_embed[1500], NOT
        # pos_embed[476] like it would if every window started from index 0.
        h = h + self.pos_embed[:, position_offset:position_offset + T, :]
        if attention_mask is not None:
            # nn.TransformerEncoder expects True = mask (don't attend)
            src_key_padding_mask = ~attention_mask         # True where padded
            # Defensive: when a sample's window is 100% padding (can happen
            # for shorter inputs like HotpotQA contexts that don't fill the
            # 4096-token budget across all four 1024-token windows), the
            # all-True padding mask makes softmax over all-masked keys
            # produce NaN. Unmask position 0 in any such row — that row's
            # output is meaningless anyway (no valid tokens in this window),
            # but it stays finite and the downstream slot state isn't
            # contaminated for the OTHER rows in the batch.
            all_padded_rows = src_key_padding_mask.all(dim=-1)  # [B]
            if all_padded_rows.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_padded_rows, 0] = False
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


class _InvertedSlotAttn(nn.Module):
    """Single-layer Slot-Attention-style inverted cross-attention.

    The canonical anti-collapse mechanism from Locatello et al. 2020
    ("Object-Centric Learning with Slot Attention"). Key difference vs
    standard cross-attention: the softmax is applied OVER SLOTS for each
    input position (dim=1), not over keys for each slot. This creates
    a zero-sum competition — each input token must "vote" for which
    slot(s) get it, so slots compete for content and specialize.

    Without this inversion, identical slot queries produce identical
    outputs (no matter how many self-attention layers we stack after).
    With it, slight differences in slot queries are amplified through
    the assignment competition into distinguishable specializations.

    Simplified vs the full Locatello recipe: no GRU update, no iterative
    refinement (T=1 effectively), and slots are deterministic learned
    parameters rather than samples from a learned distribution. This is
    the minimum-viable port of inverted attention for our setting.
    """

    def __init__(self, d_enc: int, d_ffn: Optional[int] = None):
        super().__init__()
        if d_ffn is None:
            d_ffn = 2 * d_enc
        self.norm_inputs = nn.LayerNorm(d_enc)
        self.norm_slots = nn.LayerNorm(d_enc)
        self.norm_ffn = nn.LayerNorm(d_enc)
        self.to_q = nn.Linear(d_enc, d_enc, bias=False)
        self.to_k = nn.Linear(d_enc, d_enc, bias=False)
        self.to_v = nn.Linear(d_enc, d_enc, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_enc, d_ffn), nn.GELU(),
            nn.Linear(d_ffn, d_enc),
        )
        self.scale = d_enc ** -0.5

    def forward(
        self, slots: Tensor, inputs: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # slots: [B, N_slots, d_enc]    inputs: [B, T, d_enc]
        orig_slots = slots
        all_padded = None
        if key_padding_mask is not None:
            all_padded = key_padding_mask.all(dim=-1)         # [B]

        inputs_n = self.norm_inputs(inputs)
        slots_n = self.norm_slots(slots)

        q = self.to_q(slots_n)          # [B, N, d]
        k = self.to_k(inputs_n)         # [B, T, d]
        v = self.to_v(inputs_n)         # [B, T, d]

        # Attention logits [B, N_slots, T]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # INVERTED softmax: across slots (dim=1) for each key. Each token's
        # mass is split across slots; slots compete for content.
        # NOTE: do NOT mask before softmax_over_slots — for padded keys the
        # entire column would be -inf and softmax→NaN. Zero out padded
        # contributions AFTER softmax instead.
        attn = attn.softmax(dim=1)
        if key_padding_mask is not None:
            # key_padding_mask: [B, T], True = padded → contributes 0
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        # Renormalize across keys (Slot Attention's "weighted mean" step).
        # Without epsilon, a slot that got 0 mass from every valid key would
        # divide 0/0 here. With it, that slot just gets a small uniform
        # share that the FFN then resolves.
        attn = attn + 1e-8
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = attn @ v               # [B, N_slots, d]

        # All-padded rows: every key was masked, so any update is junk from
        # zero-embed positions. Zero the update so slots pass through unchanged.
        if all_padded is not None and all_padded.any():
            updates = updates * (~all_padded).to(updates.dtype).view(-1, 1, 1)

        slots = slots + updates          # residual
        # FFN residual: applied to ALL rows because torch ops don't short-circuit.
        # Even with all-padded inputs, FFN(norm(slots)) is generally nonzero
        # (it sees the prior slot state, not the inputs). For all-padded rows
        # we restore the pre-call state at the end so the streaming-write
        # invariant holds: an all-padded window is a no-op on slot state.
        slots = slots + self.ffn(self.norm_ffn(slots))

        if all_padded is not None and all_padded.any():
            keep = (~all_padded).to(slots.dtype).view(-1, 1, 1)
            slots = slots * keep + orig_slots * (1.0 - keep)

        return slots


class _SlotSelfAttn(nn.Module):
    """Self-attention block among slot queries + FFN — the canonical
    anti-collapse mechanism from Perceiver IO and DETR.

    Without inter-slot communication, slots that start near-identical (or
    that downstream layers homogenize) cannot differentiate. From DETR's
    de-homogenization analysis: "Self-attention layers disperse queries
    from each other in terms of both position and feature distance. When
    self-attention layers are removed from the decoder, DETR becomes
    compromised to duplicate detections" (Carion et al. 2020). Perceiver
    IO alternates cross-attention with a "deep stack of Transformer-style
    self-attention blocks in the latent space" (Jaegle et al. 2021).

    One self-attention layer + FFN, both with pre-LayerNorm and residual.
    """

    def __init__(self, d_enc: int, n_heads: int = 4, d_ffn: Optional[int] = None):
        super().__init__()
        if d_ffn is None:
            d_ffn = 2 * d_enc
        self.norm_attn = nn.LayerNorm(d_enc)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_enc,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_enc)
        self.ffn = nn.Sequential(
            nn.Linear(d_enc, d_ffn), nn.GELU(),
            nn.Linear(d_ffn, d_enc),
        )

    def forward(self, slots: Tensor) -> Tensor:
        # slots: [B, N_slots, d_enc]
        h = self.norm_attn(slots)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        slots = slots + attn_out
        slots = slots + self.ffn(self.norm_ffn(slots))
        return slots


class V21Encoder(nn.Module):
    """V2.1 encoder: 32 edges = (src_id, dst_id, edge_vec) triples.

    Pipeline:
        text_embeds → BiTransformer → text_h
        slot_queries (learned) → cross-attn over text_h → edge_slots
        per-slot heads → (src_q, dst_q, edge_vec)
        classify src_q, dst_q against codebook → (src_id, dst_id)
        modifier MLPs add instance-specific delta to concept[src_id], concept[dst_id]
        package per-edge as 3 memory tokens (src, edge, dst) interleaved

    The codebook is used for *routing* (which concept). The modifier MLP
    produces *content* (the actual value used to construct memory tokens):
    src_value = concept[src_id] + modifier(concat(concept[src_id], src_q)).
    Same for dst. This lets the model encode specific instances ("Alice",
    "42") on top of categorical anchors ("Person", "Number") without
    spending codebook slots on every instance.
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
        # Stays at full d_concept (1024 default) for routing discrimination —
        # 4096 codes need adequate dimensionality to be distinguishable.
        self.concept_id = nn.Parameter(torch.zeros(cfg.n_nodes, cfg.d_concept))
        nn.init.normal_(self.concept_id, std=1.0 / cfg.d_concept ** 0.5)

        # Learnable scoring scale. Init at ln(1)=0 so scoring magnitude at
        # init matches the no-scale baseline (Gumbel-dominated picks =
        # exploration). The model can grow it during training if the
        # codebook becomes discriminative enough that scoring signal
        # deserves amplification. Larger init values (e.g. ln(40)) cause
        # premature collapse via positive-feedback loop between scale and
        # picked-code reinforcement, even with load-balance loss.
        self.score_log_scale = nn.Parameter(torch.tensor(0.0))

        # v1e: optionally compress codebook lookups + queries from d_concept
        # (1024) to d_node_state (128) before the modifier MLP. Decouples
        # "codebook discrimination capacity" (needs to be wide) from "per-
        # edge state carried forward as memory" (can be narrow). Shared
        # down-projections — codebook entries are role-agnostic, queries
        # are not but the role differentiation happens later via role_embed.
        # LayerNorm after each down-proj keeps magnitudes consistent
        # regardless of the projection's init scale.
        d_state = cfg.d_node_carried   # 128 if d_node_state>0 else d_concept
        if d_state != cfg.d_concept:
            self.concept_down = nn.Sequential(
                nn.Linear(cfg.d_concept, d_state),
                nn.LayerNorm(d_state),
            )
            self.query_down = nn.Sequential(
                nn.Linear(cfg.d_concept, d_state),
                nn.LayerNorm(d_state),
            )
        else:
            self.concept_down = None
            self.query_down = None
        self._d_state = d_state

        # Modifier MLPs: input is concat(concept_emb, query) in d_state space,
        # output is residual delta added to concept_emb. Small init keeps
        # initial delta near zero.
        def _modifier():
            mlp = nn.Sequential(
                nn.Linear(2 * d_state, cfg.d_modifier_hidden), nn.GELU(),
                nn.Linear(cfg.d_modifier_hidden, d_state),
            )
            nn.init.normal_(mlp[-1].weight, std=1e-3)
            nn.init.zeros_(mlp[-1].bias)
            return mlp
        self.src_modifier = _modifier()
        self.dst_modifier = _modifier()

        # Slim low-rank projections to Llama d_model. src/dst now start from
        # d_state (128) instead of d_concept (1024) when compression is on.
        def _proj(d_in, d_out, d_hidden):
            return nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_out),
                nn.LayerNorm(d_out),
            )
        # Edge → memory-token projection. Two modes:
        # - "triple" (legacy): emit 3 tokens per edge (src, edge, dst), each
        #   projected separately. 3K total memory tokens for Llama.
        # - "fused" (v1e default): concat (src_value, edge_vec, dst_value)
        #   then project once. 1 token per edge → K total memory tokens.
        #   Llama sees semantic units (whole relations) rather than role atoms;
        #   memory-token attention cost drops by 9× at K=32.
        if cfg.edge_token_packing == "triple":
            self.proj_src = _proj(d_state, cfg.d_llama, cfg.d_proj_hidden_v21_main)
            self.proj_dst = _proj(d_state, cfg.d_llama, cfg.d_proj_hidden_v21_main)
            self.proj_edge = _proj(cfg.d_edge, cfg.d_llama, cfg.d_proj_hidden_v21_edge)
            self.proj_fused = None
        else:   # "fused"
            self.proj_src = None
            self.proj_dst = None
            self.proj_edge = None
            d_fused_in = 2 * d_state + cfg.d_edge
            self.proj_fused = _proj(d_fused_in, cfg.d_llama, cfg.d_proj_hidden_v21_fused)

        # Role embeddings only apply to "triple" packing. In "fused" mode each
        # token represents a whole edge; the src/edge/dst roles are internal
        # to the fused vector and decoded by the projection itself.
        if cfg.use_role_embeddings and cfg.edge_token_packing == "triple":
            self.role_embed = nn.Parameter(torch.zeros(3, cfg.d_llama))
        else:
            self.register_parameter("role_embed", None)

        # Optional BLIP-2-style adapter between projected memory and Llama.
        # Number of queries matches actual memory token count, which depends
        # on edge packing.
        if cfg.use_qformer_adapter:
            n_mem = (3 * cfg.n_edges) if cfg.edge_token_packing == "triple" else cfg.n_edges
            self.qformer = QFormerAdapter(cfg, n_queries=n_mem)
        else:
            self.qformer = None

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        cfg = self.cfg
        B = token_embeds.shape[0]
        del mask_positions  # unused — V2.1 sees full unmasked input

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

        # 4. Classify against codebook (scaled dot product → softmax).
        # The learnable score_log_scale (init at ln(100)) brings raw
        # score magnitude up to ~Gumbel-noise scale so routing learning
        # gets signal from step 0.
        scale = self.score_log_scale.exp()
        src_scores = (src_q @ self.concept_id.T) * scale  # [B, K, N]
        dst_scores = (dst_q @ self.concept_id.T) * scale  # [B, K, N]
        src_id, src_onehot = gumbel_argmax_ste(
            src_scores, cfg.selection_temperature, self.training,
        )
        dst_id, dst_onehot = gumbel_argmax_ste(
            dst_scores, cfg.selection_temperature, self.training,
        )

        # 5. Gather chosen concept_id embeddings (differentiable via one_hot)
        src_emb_wide = src_onehot @ self.concept_id   # [B, K, D_concept]
        dst_emb_wide = dst_onehot @ self.concept_id   # [B, K, D_concept]

        # 5a. (v1e) Compress codebook lookups + queries to d_node_state.
        # Routing happens in the wide d_concept space (for discrimination);
        # everything carried forward is in the narrow d_state space.
        if self.concept_down is not None:
            src_emb = self.concept_down(src_emb_wide)        # [B, K, d_state]
            dst_emb = self.concept_down(dst_emb_wide)        # [B, K, d_state]
            src_q_state = self.query_down(src_q)             # [B, K, d_state]
            dst_q_state = self.query_down(dst_q)             # [B, K, d_state]
        else:
            src_emb = src_emb_wide
            dst_emb = dst_emb_wide
            src_q_state = src_q
            dst_q_state = dst_q

        # 5b. Apply modifier MLPs: src_value = concept[id] + modifier(concat).
        # Zero-init output Linear means initial delta is 0; identity behavior
        # at init, learns instance specialization over training.
        # Soft-clip delta vs ‖concept_emb‖ so the modifier stays a perturbation
        # rather than overwriting the categorical anchor.
        src_delta_raw = self.src_modifier(torch.cat([src_emb, src_q_state], dim=-1))
        dst_delta_raw = self.dst_modifier(torch.cat([dst_emb, dst_q_state], dim=-1))
        clip_ratio = cfg.modifier_delta_clip_ratio
        if clip_ratio > 0:
            with torch.no_grad():
                src_emb_norm = src_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                dst_emb_norm = dst_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            src_delta_norm = src_delta_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dst_delta_norm = dst_delta_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            max_src = src_emb_norm * clip_ratio
            max_dst = dst_emb_norm * clip_ratio
            src_scale = (max_src / src_delta_norm).clamp(max=1.0)
            dst_scale = (max_dst / dst_delta_norm).clamp(max=1.0)
            src_delta = src_delta_raw * src_scale
            dst_delta = dst_delta_raw * dst_scale
            with torch.no_grad():
                clip_frac = (src_scale < 1.0).float().mean() * 0.5 + (dst_scale < 1.0).float().mean() * 0.5
        else:
            src_delta = src_delta_raw
            dst_delta = dst_delta_raw
            clip_frac = torch.zeros((), device=src_delta_raw.device)
        src_value = src_emb + src_delta                # [B, K, D_concept]
        dst_value = dst_emb + dst_delta                # [B, K, D_concept]

        # 6. Project edge components to Llama d_model
        if cfg.edge_token_packing == "triple":
            # Three separate tokens per edge: [src, edge, dst]
            src_tok = self.proj_src(src_value)        # [B, K, d_llama]
            dst_tok = self.proj_dst(dst_value)        # [B, K, d_llama]
            edge_tok = self.proj_edge(edge_vec)       # [B, K, d_llama]
            if self.role_embed is not None:
                src_tok = src_tok + self.role_embed[0]
                edge_tok = edge_tok + self.role_embed[1]
                dst_tok = dst_tok + self.role_embed[2]
            memory = torch.stack([src_tok, edge_tok, dst_tok], dim=2)  # [B, K, 3, d_llama]
            memory = memory.reshape(B, 3 * cfg.n_edges, cfg.d_llama)   # [B, 3K, d_llama]
        else:   # "fused"
            # One token per edge: concat (src, edge, dst) then project once
            fused_in = torch.cat([src_value, edge_vec, dst_value], dim=-1)  # [B, K, 2*d_state + d_edge]
            memory = self.proj_fused(fused_in)        # [B, K, d_llama]

        # 7b. Optional Q-Former adapter (BLIP-2 style): re-format projected
        # memory through cross-attention with learned Llama-side queries so
        # the frozen LM sees its preferred input distribution rather than
        # raw edge-triplet projections.
        qformer_diversity = torch.zeros((), device=memory.device, dtype=memory.dtype)
        if self.qformer is not None:
            memory, qformer_diversity = self.qformer(memory)  # [B, 3K, d_llama]

        # 8. Aux outputs (metrics-ready: detached scalars where possible)
        with torch.no_grad():
            src_ent = -(F.softmax(src_scores, dim=-1)
                        * F.log_softmax(src_scores, dim=-1)).sum(-1).mean()
            dst_ent = -(F.softmax(dst_scores, dim=-1)
                        * F.log_softmax(dst_scores, dim=-1)).sum(-1).mean()
            # Modifier delta magnitudes (size of perturbation to codebook lookups)
            src_delta = (src_value - src_emb).norm(dim=-1).mean()
            dst_delta = (dst_value - dst_emb).norm(dim=-1).mean()
            edge_vec_norm = edge_vec.norm(dim=-1).mean()

        # Codebook orthogonality penalty: encourage distinct codes.
        # Full N×N cosine is too expensive for N=4096; sample a random subset
        # of codebook rows each step. Squared off-diagonal cosine averaged.
        # Gradient flows back into self.concept_id only (not picks).
        n_sub = min(cfg.codebook_orth_subsample, cfg.n_nodes)
        sub_idx = torch.randperm(cfg.n_nodes, device=self.concept_id.device)[:n_sub]
        sub = F.normalize(self.concept_id[sub_idx].float(), dim=-1)  # [n_sub, D]
        cos_mat = sub @ sub.T                                          # [n_sub, n_sub]
        eye = torch.eye(n_sub, dtype=torch.bool, device=cos_mat.device)
        off_diag = cos_mat.masked_select(~eye)
        codebook_orth_loss = (off_diag ** 2).mean()

        # Router z-loss (Mixtral / Switch-v2): caps logit magnitude so the
        # network can't make picks arbitrarily sharp via score_log_scale.
        z_loss = (router_z_loss(src_scores) + router_z_loss(dst_scores)) * 0.5

        # Q-Former diversity gets added to load_balance_loss (which model.py
        # multiplies by load_balance_coef=0.01). Scale by qformer_diversity_scale
        # so the contribution is comparable to recon when collapsed (squared cos
        # near 1 → ~10 nat contribution at scale 1000, coef 0.01).
        lb = (
            load_balance_loss(src_scores, picks=src_id)
            + load_balance_loss(dst_scores, picks=dst_id)
        ) * 0.5
        if self.qformer is not None:
            lb = lb + cfg.qformer_diversity_coef * cfg.qformer_diversity_scale * qformer_diversity

        aux = {
            "load_balance_loss": lb,
            "codebook_orth_loss": codebook_orth_loss,
            "z_loss": z_loss,
            "qformer_diversity_loss": qformer_diversity,
            "picked_ids": torch.cat([src_id, dst_id], dim=-1),  # [B, 2K] for unique counting
            "routing_entropy": (src_ent + dst_ent) * 0.5,
            "modifier_delta_norm": (src_delta + dst_delta) * 0.5,
            "modifier_delta_clip_frac": clip_frac,
            "edge_vec_norm": edge_vec_norm,
            "score_log_scale": self.score_log_scale.detach(),
        }
        return memory, aux


class FlatBaselineEncoder(nn.Module):
    """Baseline A: flat classification, 96 independent codes per window.

    No edge structure, no triples. 96 slot queries each pick one of 4096
    nodes. Same total memory token count (96) as V2.1 for fair comparison.

    Codebook width is d_concept_baseline (separate from V2.1's d_concept)
    so that A's pre-projection float count matches V2.1's:
        A: 96 × d_concept_baseline = 96 × 725 ≈ 69,600
        V2.1: 32 × (1024 + 128 + 1024) = 69,632

    Tests: does the (src, edge, dst) triple structure provide value over
    flat code selection at matched pre-projection budget?
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
            nn.Linear(cfg.d_enc, cfg.d_concept_baseline),
        )

        self.concept_id = nn.Parameter(torch.zeros(cfg.n_nodes, cfg.d_concept_baseline))
        nn.init.normal_(self.concept_id, std=1.0 / cfg.d_concept_baseline ** 0.5)
        # Learnable scale init at ln(1)=0 (no-scale baseline) — see V21Encoder
        # for the reasoning. The model can grow the scale during training
        # if it improves routing signal-to-noise.
        self.score_log_scale = nn.Parameter(torch.tensor(0.0))

        self.proj_code = nn.Sequential(
            nn.Linear(cfg.d_concept_baseline, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            nn.LayerNorm(cfg.d_llama),
        )

    def init_streaming_state(self, batch_size: int, device, dtype):
        """v1g streaming init: per-batch slot queries. State lives in the
        encoder's native dtype (fp32); the `dtype` arg is the Llama-side
        dtype and is ignored here."""
        del dtype
        return self.code_queries.unsqueeze(0).expand(
            batch_size, -1, -1,
        ).contiguous()

    def streaming_write(
        self, state: Tensor, token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None, chunk_offset: int = 0,
    ) -> tuple[Tensor, dict]:
        """One v1g write: refresh slot queries by cross-attending to a new
        1024-token window. Returns updated slot queries; no aux losses fire
        here — routing/diversity get computed once in `finalize_memory`."""
        text_h = self.bi_transformer(token_embeds, attention_mask,
                                      position_offset=chunk_offset)
        kv_mask = ~attention_mask if attention_mask is not None else None
        new_state = self.slot_attn(state, text_h, key_padding_mask=kv_mask)
        return new_state, {}

    def finalize_memory(self, state: Tensor) -> tuple[Tensor, dict]:
        """Project the final slot-query state to memory tokens + aux losses."""
        cfg = self.cfg
        code_q = self.code_head(state)
        scores = (code_q @ self.concept_id.T) * self.score_log_scale.exp()
        code_id, onehot = gumbel_argmax_ste(
            scores, cfg.selection_temperature, self.training,
        )
        code_emb = onehot @ self.concept_id
        memory = self.proj_code(code_emb)
        with torch.no_grad():
            ent = -(F.softmax(scores, dim=-1)
                    * F.log_softmax(scores, dim=-1)).sum(-1).mean()
        aux = {
            "load_balance_loss": load_balance_loss(scores, picks=code_id),
            "picked_ids": code_id,
            "routing_entropy": ent,
        }
        return memory, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Single-window forward (legacy path) — equivalent to one streaming
        write + finalize. Kept so existing HSM/JEPA scripts work unchanged."""
        del mask_positions
        B = token_embeds.shape[0]
        state = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)


class ContinuousBaselineEncoder(nn.Module):
    """Baseline B: continuous bottleneck, no quantization.

    96 slot queries each produce a continuous vector of D_cont dim.
    No codebook, no classification, no discrete picking. Direct
    continuous-to-Llama projection.

    D_cont = 725 matches Baseline A's per-slot dim and is chosen so
    96 × D_cont ≈ V2.1's 69,632 pre-projection floats per chunk. This
    makes B "A without quantization" — same architecture, same width,
    the only difference being that B's vectors live freely in ℝ^725
    while A's are constrained to a 4096-point codebook manifold.

    Tests: does the discrete codebook provide value over a continuous
    bottleneck at matched pre-projection width?
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        self.bi_transformer = SmallBiTransformer(cfg)
        # Orthogonal init: 96 slots start in 96 mutually orthogonal
        # directions, giving the inverted-attention competition below
        # a non-degenerate starting point.
        self.cont_queries = nn.Parameter(torch.zeros(cfg.n_flat_codes, cfg.d_enc))
        nn.init.orthogonal_(self.cont_queries)
        # Slot Attention-style inverted attention (Locatello et al. 2020).
        # Softmax-over-slots creates zero-sum competition: each input
        # token votes for which slot gets it, so slots specialize. This
        # is the canonical anti-collapse mechanism for slot architectures.
        # Standard cross-attention (softmax-over-keys) has no such
        # competition and reliably collapses all 96 slots into one effective
        # vector for this kind of task.
        self.slot_attn = _InvertedSlotAttn(cfg.d_enc)
        # A second inverted-attention layer for refinement (analogous to
        # iterating Slot Attention; cheaper than a full GRU loop).
        self.slot_attn_2 = _InvertedSlotAttn(cfg.d_enc)

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

    def init_streaming_state(self, batch_size: int, device, dtype):
        """v1g streaming init: per-batch slot state. State lives in the
        encoder's native dtype (fp32); the Llama-side `dtype` arg is unused
        here — the encoder casts inputs to its weights' dtype internally."""
        del dtype
        return self.cont_queries.unsqueeze(0).expand(
            batch_size, -1, -1,
        ).contiguous()

    def streaming_write(
        self, state: Tensor, token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None, chunk_offset: int = 0,
    ) -> tuple[Tensor, dict]:
        """One v1g write: two rounds of inverted slot attention over a new
        1024-token window, with `state` as the input slot queries. This is
        the canonical streaming Slot Attention recipe — each write is an
        iteration of refinement using the next window's tokens."""
        text_h = self.bi_transformer(token_embeds, attention_mask,
                                      position_offset=chunk_offset)
        kv_mask = ~attention_mask if attention_mask is not None else None
        state = self.slot_attn(state, text_h, key_padding_mask=kv_mask)
        state = self.slot_attn_2(state, text_h, key_padding_mask=kv_mask)
        return state, {}

    def finalize_memory(self, state: Tensor) -> tuple[Tensor, dict]:
        """Project final slot state to d_llama + compute diversity loss."""
        cont_vec = self.cont_head(state)
        memory = self.proj_cont(cont_vec)
        with torch.amp.autocast("cuda", enabled=False):
            def _diversity(x):
                x_norm = F.normalize(x.float(), dim=-1)
                cos = x_norm @ x_norm.transpose(1, 2)
                M = cos.shape[1]
                eye = torch.eye(M, dtype=torch.bool, device=cos.device)
                off_diag = cos[:, ~eye].view(cos.shape[0], -1)
                return off_diag.pow(2).mean()
            diversity_slots = _diversity(state)
            diversity_mem = _diversity(memory)
            diversity_loss = diversity_slots + diversity_mem
        aux = {
            "load_balance_loss": self.cfg.b_diversity_scale * diversity_loss,
            "diversity_slots_raw": diversity_slots,
            "diversity_mem_raw": diversity_mem,
            "cont_vec_norm": cont_vec.norm(dim=-1).mean(),
        }
        return memory, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        del mask_positions
        B = token_embeds.shape[0]

        text_h = self.bi_transformer(token_embeds, attention_mask)
        cont_queries = self.cont_queries.unsqueeze(0).expand(B, -1, -1)
        kv_mask = ~attention_mask if attention_mask is not None else None
        # Two inverted-attention rounds (Slot Attention-style competition)
        cont_slots = self.slot_attn(cont_queries, text_h, key_padding_mask=kv_mask)
        cont_slots = self.slot_attn_2(cont_slots, text_h, key_padding_mask=kv_mask)

        cont_vec = self.cont_head(cont_slots)                      # [B, M, D_cont]
        memory = self.proj_cont(cont_vec)                          # [B, M, d_llama]

        # Diversity loss on memory tokens. Penalizes squared pairwise cosine.
        # Reconstruction CE has no slot-level supervision (Llama can use any
        # subset of memory tokens), so without an explicit diversity signal
        # the optimal solution is to collapse all 96 slots to one vector.
        # We compute the loss on the cont_slots (pre-projection) AND the
        # memory tokens (post-projection) so both spaces are diversified.
        with torch.amp.autocast("cuda", enabled=False):
            def _diversity(x):
                x_norm = F.normalize(x.float(), dim=-1)
                cos = x_norm @ x_norm.transpose(1, 2)
                M = cos.shape[1]
                eye = torch.eye(M, dtype=torch.bool, device=cos.device)
                off_diag = cos[:, ~eye].view(cos.shape[0], -1)
                return off_diag.pow(2).mean()

            diversity_slots = _diversity(cont_slots)
            diversity_mem = _diversity(memory)
            diversity_loss = diversity_slots + diversity_mem

        # Scale large enough to compete with reconstruction CE. Effective
        # contribution after load_balance_coef=0.01 multiplier is up to
        # ~20 nat at full collapse, dominating recon's ~7 nat.
        aux = {
            "load_balance_loss": self.cfg.b_diversity_scale * diversity_loss,
            "diversity_slots_raw": diversity_slots,
            "diversity_mem_raw": diversity_mem,
            "cont_vec_norm": cont_vec.norm(dim=-1).mean(),
        }
        return memory, aux


class MemorizingBaselineEncoder(nn.Module):
    """Baseline 4: Memorising-Transformers-style retrieval.

    The encoder produces a per-token KV bank of size [T, d_mt_value].
    At read time, a single query is pooled from unmasked positions
    (mimicking what the Llama decoder can see), top-96 entries are
    retrieved by dot-product similarity, projected to d_llama, and
    handed to Llama as memory tokens.

    Pre-projection budget: 96 × d_mt_value = 69,600 floats / chunk
    (matched to V2.1's 69,632).

    Note: hard top-K retrieval blocks gradient through the ranking.
    Keys still learn indirectly via the shared encoder hidden state
    (values and keys come from the same text_h). This is consistent
    with how Memorising Transformers themselves train.

    Tests: does retrieval from a verbatim per-token bank beat
    compression into fixed-N learned slots?
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        self.bi_transformer = SmallBiTransformer(cfg)
        # Per-token KV head — splits into key and value. Widened to 2-layer MLP
        # to give MT comparable per-token transformation capacity to A's codebook
        # routing + B's slot attention. Single Linear was structurally too light
        # for fair comparison; this brings MT trainable params from ~9.7M to ~11.5M.
        self.kv_head = nn.Sequential(
            nn.Linear(cfg.d_enc, 2 * cfg.d_enc), nn.GELU(),
            nn.Linear(2 * cfg.d_enc, 2 * cfg.d_mt_value),
        )
        # Query head — operates on the pooled unmasked hidden state
        self.query_head = nn.Sequential(
            nn.Linear(cfg.d_enc, 2 * cfg.d_enc), nn.GELU(),
            nn.Linear(2 * cfg.d_enc, cfg.d_mt_value),
        )

        self.proj_value = nn.Sequential(
            nn.Linear(cfg.d_mt_value, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            nn.LayerNorm(cfg.d_llama),
        )
        self.n_retrieve = cfg.n_flat_codes  # 96 — match V2.1 memory token count

    def init_streaming_state(self, batch_size: int, device, dtype):
        """v1g streaming init: accumulate per-window kv pairs as we go."""
        return []

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0):
        """Build the KV bank one 1024-token window at a time. Each window's
        bi_transformer only sees its own window (same constraint as A/B), so
        MT no longer has a 4× attention-span advantage over A/B."""
        text_h = self.bi_transformer(token_embeds, attention_mask,
                                      position_offset=chunk_offset)
        kv = self.kv_head(text_h)
        keys, values = kv.chunk(2, dim=-1)
        state.append({
            "keys": keys,
            "values": values,
            "attention_mask": attention_mask,
        })
        return state, {}

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        """Concat per-window kv pairs into the full chunk bank."""
        if not state:
            raise RuntimeError("MT finalize_memory called with no writes")
        keys = torch.cat([s["keys"] for s in state], dim=1)
        values = torch.cat([s["values"] for s in state], dim=1)
        if state[0]["attention_mask"] is not None:
            attention_mask = torch.cat([s["attention_mask"] for s in state], dim=1)
        else:
            attention_mask = None
        B = keys.shape[0]

        placeholder = torch.zeros(
            B, 0, self.cfg.d_llama,
            device=keys.device, dtype=keys.dtype,
        )
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=keys.device, dtype=torch.float32,
            ),
            "mt_bank": {
                "keys": keys,
                "values": values,
                "attention_mask": attention_mask,
            },
        }
        return placeholder, aux

    def retrieve_for_query(
        self,
        bank: dict,
        question_embeds: Tensor,         # [B, T_q, d_llama] raw Llama embeds of the question
        question_mask: Tensor,           # [B, T_q] bool — True at valid question positions
        K: int,                          # retrieval budget
    ) -> tuple[Tensor, dict]:
        """v1h QA retrieval — one query per chunk, pooled from question tokens.

        The query is the mean of `bi_transformer.in_proj(question_embeds)` over
        valid question positions. No bi_transformer attention is applied —
        same recipe as `retrieve_per_sentence` (avoids any contextual leak).
        Returns [B, K, d_llama] retrieved memory tokens, one set per example."""
        keys = bank["keys"]
        values = bank["values"]
        attn_mask = bank["attention_mask"]
        B, T, d_value = keys.shape
        device = keys.device

        # Project question embeds to d_enc via in_proj only (no attention)
        q_proj = self.bi_transformer.in_proj(
            question_embeds.to(self.bi_transformer.in_proj.weight.dtype)
        )                                                       # [B, T_q, d_enc]
        contrib = question_mask.to(q_proj.dtype).unsqueeze(-1)  # [B, T_q, 1]
        denom = contrib.sum(dim=1).clamp(min=1.0)                # [B, 1]
        q_pool = (q_proj * contrib).sum(dim=1) / denom            # [B, d_enc]
        query = self.query_head(q_pool)                          # [B, d_value]

        # Score keys against the single per-example query
        scores = torch.einsum("btd,bd->bt", keys, query)         # [B, T]
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        # Audit2 #7: guard against all-pad query / fewer-than-K valid keys.
        # Without guards, softmax over all -inf returns NaN, contaminating
        # the memory tensor.
        if attn_mask is not None:
            valid_per_row = attn_mask.sum(dim=-1)                # [B]
            any_valid = valid_per_row > 0
        else:
            valid_per_row = torch.full((B,), T, device=device, dtype=torch.long)
            any_valid = torch.ones(B, dtype=torch.bool, device=device)

        K_retrieve = min(K, T)
        top_scores, top_idx = scores.topk(K_retrieve, dim=-1)    # [B, K]
        # Mask top entries that came from padded positions (when row has
        # fewer than K valid keys, some top picks are padded). Use a per-row
        # K_valid = min(K, valid_count) to know how many top picks are real.
        rank = torch.arange(K_retrieve, device=device).unsqueeze(0)  # [1, K]
        per_row_K = valid_per_row.unsqueeze(-1).clamp(max=K_retrieve)  # [B, 1]
        valid_topk = rank < per_row_K                                # [B, K]
        # Sanitize -inf top scores so softmax doesn't see them; replace
        # invalid entries' scores with a finite low value (they're masked
        # out of the soft weights anyway via valid_topk).
        top_scores = torch.where(
            torch.isfinite(top_scores), top_scores,
            torch.full_like(top_scores, -1e4),
        )

        top_idx_exp = top_idx.unsqueeze(-1).expand(B, K_retrieve, d_value)
        retrieved = values.gather(1, top_idx_exp)                 # [B, K, d_value]

        # Zero out invalid (padded) top-K positions in retrieved
        retrieved = retrieved * valid_topk.unsqueeze(-1).to(retrieved.dtype)
        # All-pad-query rows: zero memory entirely
        retrieved = retrieved * any_valid.view(B, 1, 1).to(retrieved.dtype)

        # STE gradient with the same valid-K masking on soft weights
        masked_scores = top_scores.masked_fill(~valid_topk, float("-inf"))
        # If a row has ZERO valid top entries (all-pad), masked_scores is
        # all -inf → softmax NaN. Replace with zero gate for those rows.
        row_has_valid = valid_topk.any(dim=-1, keepdim=True)      # [B, 1]
        safe_masked = torch.where(
            row_has_valid, masked_scores, torch.zeros_like(masked_scores),
        )
        soft_w = F.softmax(safe_masked, dim=-1).unsqueeze(-1)
        gate = 1.0 + (soft_w * K_retrieve - (soft_w * K_retrieve).detach())
        retrieved = retrieved * gate

        memory = self.proj_value(retrieved)                       # [B, K, d_llama]

        # Diversity loss on retrieved memory (same recipe)
        with torch.amp.autocast("cuda", enabled=False):
            mem_norm = F.normalize(memory.float(), dim=-1)
            cos = mem_norm @ mem_norm.transpose(1, 2)
            Mtok = cos.shape[1]
            eye = torch.eye(Mtok, dtype=torch.bool, device=cos.device)
            off_diag = cos[:, ~eye].view(cos.shape[0], -1)
            diversity_loss = off_diag.pow(2).mean()

        aux = {
            "load_balance_loss": self.cfg.mt_diversity_scale * diversity_loss,
            "diversity_raw": diversity_loss,
            "retrieved_score_mean": scores.masked_fill(
                scores == float("-inf"), 0.0
            ).mean(),
        }
        return memory, aux

    def retrieve_per_sentence(
        self,
        bank: dict,
        chunk_embeds: Tensor,            # [B, T, d_llama] raw Llama embeds for query
        query_starts: Tensor,            # [B, K_query] long — sentence start in chunk
        query_lengths: Tensor,           # [B, K_query] long
        mask_positions: Tensor,          # [B, K_query, L_max] bool
        reveal_positions: Tensor,        # [B, K_query, L_max] bool
        K: int,                          # retrieval budget per query
    ) -> tuple[Tensor, dict]:
        """For each queried sentence: pool query from that sentence's visible
        positions, score against the bank, retrieve top-K values, project to
        d_llama. Returns [B*K_query, K, d_llama] memory.

        The query is pooled from `bi_transformer.in_proj(chunk_embeds)` — a
        per-position linear projection of raw Llama embeddings, WITHOUT the
        transformer attention layers. This prevents the bidirectional encoder
        attention from leaking still-masked tokens' content into the visible
        positions used for query pooling."""
        keys = bank["keys"]
        values = bank["values"]
        attn_mask = bank["attention_mask"]
        B, T, d_value = keys.shape
        K_query = query_starts.shape[1]
        L_max = mask_positions.shape[2]
        BK = B * K_query
        device = keys.device

        # Build [B, K_query, T] mask: True where chunk position t is INSIDE
        # sentence k_query AND that sentence position is VISIBLE (unmasked
        # or revealed). This is the set of positions we pool the query from.
        pos_idx = torch.arange(T, device=device).view(1, 1, T)
        q_starts = query_starts.unsqueeze(-1)
        q_lengths = query_lengths.unsqueeze(-1)
        in_sentence = (pos_idx >= q_starts) & (pos_idx < q_starts + q_lengths)

        sent_pos = (pos_idx - q_starts).clamp(min=0, max=L_max - 1)
        still_masked = mask_positions & ~reveal_positions  # [B, K_query, L_max]
        sm_at_pos = still_masked.gather(2, sent_pos)        # [B, K_query, T]
        visible_for_pool = in_sentence & ~sm_at_pos
        if attn_mask is not None:
            visible_for_pool = visible_for_pool & attn_mask.unsqueeze(1)

        # Project raw embeds to d_enc per-position (NO attention). This is
        # the query-side encoder; it cannot encode info from masked positions
        # because there's no cross-position mixing.
        q_proj = self.bi_transformer.in_proj(
            chunk_embeds.to(self.bi_transformer.in_proj.weight.dtype)
        )  # [B, T, d_enc]

        # Pool one query per (b, k_query) from q_proj at visible positions
        contrib = visible_for_pool.to(q_proj.dtype).unsqueeze(-1)  # [B, K_query, T, 1]
        denom = contrib.sum(dim=2).clamp(min=1.0)                  # [B, K_query, 1]
        q_pool = (q_proj.unsqueeze(1) * contrib).sum(dim=2) / denom  # [B, K_query, d_enc]
        query = self.query_head(q_pool)                            # [B, K_query, d_value]

        # Score keys against per-(b, k_query) query
        scores = torch.einsum("btd,bkd->bkt", keys, query)          # [B, K_query, T]
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

        # Top-K per query
        K_retrieve = min(K, T)
        top_scores, top_idx = scores.topk(K_retrieve, dim=-1)        # [B, K_query, K_retrieve]
        # Gather values along T
        values_exp = values.unsqueeze(1).expand(B, K_query, T, d_value)
        top_idx_exp = top_idx.unsqueeze(-1).expand(B, K_query, K_retrieve, d_value)
        retrieved = values_exp.gather(2, top_idx_exp)               # [B, K_query, K, d_value]

        # STE gradient pathway (same recipe as the single-query forward)
        soft_w = F.softmax(top_scores, dim=-1).unsqueeze(-1)
        soft_unit = soft_w * K_retrieve
        gate = 1.0 + (soft_unit - soft_unit.detach())
        retrieved = retrieved * gate

        # Flatten to [BK, K_retrieve, d_value] and project to d_llama
        retrieved_bk = retrieved.reshape(BK, K_retrieve, d_value)
        memory = self.proj_value(retrieved_bk)                       # [BK, K, d_llama]

        # Diversity loss on retrieved memory (same recipe as forward; same
        # known caveat that the loss may not actually reduce diversity).
        with torch.amp.autocast("cuda", enabled=False):
            mem_norm = F.normalize(memory.float(), dim=-1)
            cos = mem_norm @ mem_norm.transpose(1, 2)
            Mtok = cos.shape[1]
            eye = torch.eye(Mtok, dtype=torch.bool, device=cos.device)
            off_diag = cos[:, ~eye].view(cos.shape[0], -1)
            diversity_loss = off_diag.pow(2).mean()

        aux = {
            "load_balance_loss": self.cfg.mt_diversity_scale * diversity_loss,
            "diversity_raw": diversity_loss,
            "retrieved_score_mean": scores.masked_fill(
                scores == float("-inf"), 0.0
            ).mean(),
        }
        return memory, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        B, T, _ = token_embeds.shape

        # 1. Bi-transformer over full (unmasked) input
        text_h = self.bi_transformer(token_embeds, attention_mask)   # [B, T, d_enc]

        # 2. Per-token KV bank
        kv = self.kv_head(text_h)                                    # [B, T, 2*d_mt_value]
        keys, values = kv.chunk(2, dim=-1)                           # [B, T, d_mt_value]

        # 3. Query pooled from unmasked positions
        if mask_positions is not None:
            unmasked = (~mask_positions).float().unsqueeze(-1)       # [B, T, 1]
        else:
            unmasked = torch.ones(B, T, 1, device=text_h.device, dtype=text_h.dtype)
        if attention_mask is not None:
            unmasked = unmasked * attention_mask.float().unsqueeze(-1)
        denom = unmasked.sum(dim=1).clamp(min=1.0)                   # [B, 1]
        q_pool = (text_h * unmasked).sum(dim=1) / denom              # [B, d_enc]
        query = self.query_head(q_pool)                              # [B, d_mt_value]

        # 4. Score keys, mask out padding
        scores = (keys * query.unsqueeze(1)).sum(dim=-1)             # [B, T]
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))

        # 5. Top-K retrieval (hard forward, STE for query/key gradient).
        # k is capped to actual valid (non-padded) positions per-batch so
        # that under variable-length / padded inputs we never gather a
        # padded slot. Pad slots are filled with a zero memory token below.
        if attention_mask is not None:
            valid_per_row = attention_mask.sum(dim=1)                # [B]
            min_valid = int(valid_per_row.min().item())
        else:
            min_valid = T
        k = min(self.n_retrieve, T, max(min_valid, 1))
        top_scores, top_idx = scores.topk(k, dim=-1)                 # [B, k], [B, k]
        retrieved = values.gather(
            1, top_idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
        )                                                            # [B, k, d_mt_value]

        # STE gradient pathway: forward magnitude unchanged, but a soft
        # mean-corrected weighting lets gradient flow back through scores
        # → query and the keys half of kv_head. Without this, hard top-K
        # blocks all gradient through the ranking and query_head never
        # learns. Forward: gate ≡ 1. Backward: gate ≈ softmax(top_scores)·k.
        soft_w = F.softmax(top_scores, dim=-1).unsqueeze(-1)         # [B, k, 1]
        soft_unit = soft_w * k                                       # mean ≈ 1 across k
        gate = 1.0 + (soft_unit - soft_unit.detach())                # forward = 1, gradient flows
        retrieved = retrieved * gate

        # 6. Pad to n_retrieve if k < n_retrieve (variable-length / tiny windows)
        if k < self.n_retrieve:
            pad = torch.zeros(
                B, self.n_retrieve - k, retrieved.shape[-1],
                device=retrieved.device, dtype=retrieved.dtype,
            )
            retrieved = torch.cat([retrieved, pad], dim=1)

        memory = self.proj_value(retrieved)                          # [B, 96, d_llama]

        # Diversity loss on memory tokens (same recipe as B).
        # In the simplified single-prepend MT (no per-layer retrieval, no
        # gating), the reconstruction CE has no slot-level supervision —
        # so without an explicit diversity signal, the bi-transformer
        # converges on similar per-position values and top-K retrieval
        # collapses all 96 memory tokens to one effective vector. This
        # is a known failure mode of slot/retrieval architectures
        # without diversification pressure. The original MT (Wu 2022)
        # avoids this by retrieving per-layer with local-attention
        # gating, which we don't replicate for budget parity.
        with torch.amp.autocast("cuda", enabled=False):
            mem_norm = F.normalize(memory.float(), dim=-1)
            cos = mem_norm @ mem_norm.transpose(1, 2)
            Mtok = cos.shape[1]
            eye = torch.eye(Mtok, dtype=torch.bool, device=cos.device)
            off_diag = cos[:, ~eye].view(cos.shape[0], -1)
            diversity_loss = off_diag.pow(2).mean()

        aux = {
            "load_balance_loss": self.cfg.mt_diversity_scale * diversity_loss,
            "diversity_raw": diversity_loss,
            "retrieved_score_max": scores.masked_fill(
                scores == float("-inf"), 0.0
            ).max(dim=-1).values.mean(),
            "retrieved_score_mean": scores.masked_fill(
                scores == float("-inf"), 0.0
            ).mean(),
        }
        return memory, aux


class NullEncoder(nn.Module):
    """Baseline 6: vanilla Llama — no encoder, no memory tokens.

    Returns an empty memory tensor of shape [B, 0, d_llama]. The decoder's
    forward path then runs Llama purely on the masked text input, with
    only mask_embed as a trainable parameter. This is the LOSS FLOOR for
    "what can Llama do on this task without any side-car module?"

    If V2.1 / A / B / MT / Mamba can't beat this floor, the memory module
    isn't contributing — the task is solvable from local context alone.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

    def init_streaming_state(self, batch_size: int, device, dtype):
        # NullEncoder has no slot state; we stash (B, device, dtype) in a
        # zero-width tensor so finalize_memory can recover B and the device.
        return torch.zeros(batch_size, 0, self.cfg.d_llama, device=device, dtype=dtype)

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0):
        del chunk_offset  # Null encoder has no encoder modules
        return state, {}

    def finalize_memory(self, state: Tensor) -> tuple[Tensor, dict]:
        aux = {"load_balance_loss": torch.zeros((), device=state.device, dtype=torch.float32)}
        return state, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        B = token_embeds.shape[0]
        memory = torch.zeros(
            B, 0, self.cfg.d_llama,
            device=token_embeds.device,
            dtype=token_embeds.dtype,
        )
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=token_embeds.device, dtype=torch.float32,
            ),
        }
        return memory, aux


class RecurrentBaselineEncoder(nn.Module):
    """Baseline 5: Mamba state-space-model bottleneck.

    Mamba processes the 256-token input as a recurrent state-space
    model, producing 256 per-token hidden states. Those are narrowed
    to d_recurrent (725) per token, then adaptively pooled to 96
    memory tokens and projected to d_llama.

    Pre-projection budget: 96 × d_recurrent = 69,600 floats / chunk.

    Tests: does recurrent compression match parallel slot-attention
    compression at matched bottleneck width?

    Note: requires `mamba_ssm` (CUDA-only — no CPU fallback).
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "RecurrentBaselineEncoder requires mamba_ssm. "
                "Install with: pip install mamba-ssm"
            ) from e

        # Project Llama embed → Mamba d_model
        self.in_proj = nn.Linear(cfg.d_llama, cfg.d_mamba, bias=False)

        # Mamba stack
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=cfg.d_mamba,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
            )
            for _ in range(cfg.mamba_n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_mamba)

        # Per-token bottleneck: d_mamba → d_recurrent
        self.bottleneck = nn.Linear(cfg.d_mamba, cfg.d_recurrent)

        # Project d_recurrent → d_llama
        self.proj_to_llama = nn.Sequential(
            nn.Linear(cfg.d_recurrent, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            nn.LayerNorm(cfg.d_llama),
        )

        self.target_len = cfg.n_flat_codes  # 96

    def init_streaming_state(self, batch_size: int, device, dtype):
        """Mamba's SSM hidden state IS naturally streaming, but plumbing the
        mamba_ssm state across multiple forwards is fragile. Equivalent:
        accumulate window tokens, run one forward over the concatenation.
        State is a list of (token_embeds, attention_mask) tuples."""
        return []

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0):
        del chunk_offset  # Mamba SSM doesn't use positional embed
        state.append((token_embeds, attention_mask))
        return state, {}

    def finalize_memory(self, state: list) -> tuple[Tensor, dict]:
        """Concat accumulated windows and run the standard Mamba forward."""
        if not state:
            raise RuntimeError("Mamba finalize_memory called with no writes")
        token_embeds = torch.cat([t for t, _ in state], dim=1)
        if state[0][1] is None:
            attention_mask = None
        else:
            attention_mask = torch.cat([m for _, m in state], dim=1)
        return self.forward(token_embeds, attention_mask)

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        del mask_positions  # Mamba doesn't need it

        # 1. Project to Mamba d_model
        h = self.in_proj(token_embeds.to(self.in_proj.weight.dtype))   # [B, T, d_mamba]

        # 2. Mamba stack with residual
        for block in self.mamba_blocks:
            h = h + block(h)
        h = self.norm(h)                                               # [B, T, d_mamba]

        # 3. Per-token bottleneck
        h = self.bottleneck(h)                                         # [B, T, d_recurrent]

        # 4. Adaptive pool T → target_len. With variable-length inputs we
        # pool only the valid (non-padded) prefix per sample so padding
        # never enters the pooled memory tokens.
        B, T, _ = h.shape
        if attention_mask is not None:
            # Per-sample pool over the valid prefix. Loop over B (small)
            # — cheap relative to Mamba's O(T) recurrence.
            h_t = h.transpose(1, 2).contiguous()                       # [B, d, T]
            pooled = []
            for i in range(B):
                valid = int(attention_mask[i].sum().item())
                if valid < 1:
                    valid = 1
                p = F.adaptive_avg_pool1d(
                    h_t[i:i + 1, :, :valid], self.target_len,
                )                                                       # [1, d, 96]
                pooled.append(p)
            h_pooled = torch.cat(pooled, dim=0).transpose(1, 2)        # [B, 96, d_recurrent]
        else:
            h_t = h.transpose(1, 2)                                    # [B, d_recurrent, T]
            h_pooled = F.adaptive_avg_pool1d(h_t, self.target_len)     # [B, d_recurrent, 96]
            h_pooled = h_pooled.transpose(1, 2)                        # [B, 96, d_recurrent]

        # 5. Project to Llama d_model
        memory = self.proj_to_llama(h_pooled)                          # [B, 96, d_llama]

        aux = {
            "load_balance_loss": torch.zeros(
                (), device=memory.device, dtype=memory.dtype,
            ),
            "h_pooled_norm": h_pooled.norm(dim=-1).mean(),
        }
        return memory, aux


class PlasticBaselineEncoder(nn.Module):
    """Exp 2: plastic substrate — stack of gated-MLP blocks whose weights
    are updated by a Hebbian + Oja basis modulated by a learnable per-block
    controller. The substrate ENCODES the context into its weight state.

    Streaming write (per window):
      tokens [B, T_w, d_llama] → project to h_sub → PlasticSubstrate.write_window
      The substrate's fast-weight state [D × B × h × h] mutates across windows.
      Gradient flows through plasticity steps within the chunk (chunk-bounded
      BPTT: detach at chunk boundary, not at window boundary).

    Read (per-position, via a pre-hook on one Llama layer):
      At Llama layer L's input, each position k has hidden state h_k that
      encodes question + previously-read memory + answer-so-far. That h_k
      is the conditioning:
          h_mem  = W_in(h_k)                     [B, T, h_sub]
          rd     = substrate.read(h_mem, fast)   [B, T, h_sub]
          inj    = scale ⊙ W_out(rd)             [B, T, d_llama]
          h_in'  = h_k + inj                     residual injection
      Then layer L processes h_in'. Each Llama position queries the substrate
      with its own current state — different positions get different memory.

    Substrate "memory" lives in the plastic weight state, not in any
    prepended token tensor. compute_qa_loss skips the prepend entirely for
    this variant and installs the pre-hook.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        self.h_sub = cfg.d_continuous  # 725
        self.D = getattr(cfg, "plastic_depth", 4)
        # Llama layer index at which to inject memory. Mid-depth gets refined
        # hidden states without being too late to influence subsequent layers.
        self.inject_layer_idx = getattr(cfg, "plastic_inject_layer", 8)

        from .plastic_substrate import PlasticSubstrate

        # Project Llama-side d_llama → h_sub for substrate input on the WRITE path.
        self.token_proj = nn.Sequential(
            nn.Linear(cfg.d_llama, self.h_sub, bias=False),
            nn.LayerNorm(self.h_sub),
        )

        self.substrate = PlasticSubstrate(D=self.D, h_sub=self.h_sub)

        # READ-path projections (the MemInject bridge between d_llama and h_sub).
        # Trained end-to-end via gradient from QA loss through the pre-hook.
        # Both xavier_uniform so the injection has *some* signal at step 0 —
        # zero-init W_out would kill gradient flow back to the entire substrate
        # at step 0 (matmul with a zero weight = no gradient on either input).
        # The tanh-bounded `scale_raw` (init small) plays the role of the
        # neutrality knob.
        self.W_in = nn.Linear(cfg.d_llama, self.h_sub, bias=False)
        self.W_out = nn.Linear(self.h_sub, cfg.d_llama, bias=False)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.1)

        # Per-dim scale (tanh-bounded, ReZero-style). scale_raw=atanh(0.05/1)
        # ⇒ effective scale ≈ 0.05 at init — the injection is small but
        # nonzero, so Llama is barely perturbed and gradient can still flow.
        self.scale_max = 1.0
        scale_init = 0.05
        _scale_raw_init = float(torch.atanh(torch.tensor(scale_init / self.scale_max)))
        self.scale_raw = nn.Parameter(
            torch.full((cfg.d_llama,), _scale_raw_init),
        )

    # ── Streaming interface (matches other encoders) ──────────────────────
    def init_streaming_state(self, batch_size: int, device, dtype):
        """State is a dict with the substrate fast weights. The question is
        no longer stashed — reads are per-position via the MemInject hook
        installed by compute_qa_loss, not a one-shot question-conditioned read.
        """
        fast = self.substrate.init_fast_state(batch_size, device, dtype)
        return {"fast_state": fast}

    def streaming_write(
        self, state, token_embeds, attention_mask=None, chunk_offset=0,
        surprise=None,
    ):
        """One write pass over a window.

        token_embeds : [B, T_w, d_llama]
        attention_mask: [B, T_w] True where real (not padding)
        surprise     : [B, T_w] per-token surprise (NLL from frozen Llama on
                       context with no memory). If None, treated as zeros.
        """
        # Substrate runs in bf16 (matches Llama's dtype) for tensor-core speed.
        # Parameter weights are fp32; autocast handles the mixed-precision matmul.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = self.token_proj(token_embeds)                        # [B, T_w, h_sub]
            B, T_w, _ = x.shape
            # Audit2 #5: add sinusoidal PE so within-window order matters.
            # Was order-invariant before (Hebbian write averages over T).
            pe = _sinusoidal_pe(T_w, x.shape[-1], offset=chunk_offset,
                                 device=x.device, dtype=x.dtype)
            x = x + pe.unsqueeze(0)
            if surprise is None:
                surprise = torch.zeros(B, T_w, device=x.device, dtype=x.dtype)
            else:
                surprise = surprise.to(x.dtype)
            fast_state = [w.to(x.dtype) for w in state["fast_state"]]
            new_fast = self.substrate.write_window(
                x, surprise, fast_state, attention_mask=attention_mask,
            )
        state = dict(state)
        state["fast_state"] = new_fast
        return state, {}

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        """Return an empty memory tensor (no prepend) + the fast_state in aux.
        compute_qa_loss reads fast_state from aux and uses it to install the
        per-position MemInject pre-hook during the Llama forward.
        """
        fast_state = state["fast_state"]
        # Determine B, device, dtype from fast_state[0] to construct the
        # empty memory placeholder.
        B = fast_state[0].shape[0]
        device = fast_state[0].device
        dtype = next(self.W_out.parameters()).dtype
        empty_mem = torch.zeros(B, 0, self.cfg.d_llama, device=device, dtype=dtype)

        with torch.no_grad():
            fast_norms = torch.stack(
                [w.flatten(1).norm(dim=-1).mean() for w in fast_state]
            ).mean()
        aux = {
            "load_balance_loss": torch.zeros((), device=device, dtype=dtype),
            "fast_state_norm": fast_norms,
            # compute_qa_loss reads this to wire the per-position hook.
            "plastic_fast_state": fast_state,
        }
        return empty_mem, aux

    def inject(self, hidden_states: Tensor, fast_state: list[Tensor]) -> Tensor:
        """The per-position memory injection. Used by the pre-hook closure
        installed in compute_qa_loss.

        hidden_states: [B, T, d_llama] from the layer ABOUT to receive it
        fast_state   : the substrate state from the write phase

        Returns hidden_states + scale ⊙ W_out(substrate.read(W_in(hidden_states))).
        """
        h_dtype = hidden_states.dtype
        # Substrate runs in bf16; autocast handles fp32-weight × bf16-input.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h_mem = self.W_in(hidden_states)                       # [B, T, h_sub]
            fs = [w.to(h_mem.dtype) for w in fast_state]
            readout = self.substrate.read(h_mem, fs)               # [B, T, h_sub]
            eff_scale = self.scale_max * torch.tanh(self.scale_raw) # [d_llama]
            inj = eff_scale * self.W_out(readout)                  # [B, T, d_llama]

        out = hidden_states + (inj.to(h_dtype) if inj.dtype != h_dtype else inj)
        return out

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Non-streaming forward — only used by older code paths. For QA,
        compute_qa_loss handles the streaming write + per-position read hook.
        Here we just ingest the input as one window and return empty memory.
        """
        del mask_positions
        B = token_embeds.shape[0]
        device = token_embeds.device
        dtype = token_embeds.dtype
        state = self.init_streaming_state(B, device, dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)


class GraphBaselineEncoder(nn.Module):
    """Exp 1: graph baseline. Bounded-budget edge memory with continuous
    (src, dst, state) tuples and soft node-reuse via attention snap.

    Substrate: K_max edges in continuous space (no codebook). Per window,
    the updater transformer reads (encoded pins + current edges) and
    proposes (src, dst, state, snap_gates, keep_gate, saliency_delta) per
    slot. Proposed endpoints are soft-snapped to existing endpoints to
    encourage graph connectivity.

    Read: K_max edge tokens prepended to Llama input. Each token =
    fused MLP of concat(src, dst, state, saliency_emb). Same prepend
    pattern as B/MT — no MemInject hook.

    See docs/exp1_graph_baseline.md.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        self.K_max = cfg.graph_K_max
        self.d_node = cfg.graph_d_node
        self.d_state = cfg.graph_d_state
        self.d_updater = cfg.graph_d_updater
        self.n_updater_layers = cfg.graph_updater_layers

        self.lambda_connect = cfg.graph_lambda_connect
        self.lambda_adjust = cfg.graph_lambda_adjust

        from .graph_substrate import GraphUpdater

        # Pin encoder: project Llama token embeds → updater dim
        self.pin_encoder = nn.Sequential(
            nn.Linear(cfg.d_llama, self.d_updater * 2, bias=False),
            nn.GELU(),
            nn.Linear(self.d_updater * 2, self.d_updater),
            nn.LayerNorm(self.d_updater),
        )

        self.updater = GraphUpdater(
            d=self.d_updater,
            K_max=self.K_max,
            d_node=self.d_node,
            d_state=self.d_state,
            n_layers=self.n_updater_layers,
            n_heads=4,
            ffn_mult=4,
        )

        # Fused edge → memory token projection
        # Input: concat(src, dst, state, saliency_scalar) → d_proj_hidden → d_llama
        edge_concat_dim = 2 * self.d_node + self.d_state + 1
        self.proj_to_llama = nn.Sequential(
            nn.Linear(edge_concat_dim, cfg.graph_d_proj_hidden), nn.GELU(),
            nn.Linear(cfg.graph_d_proj_hidden, cfg.d_llama),
            nn.LayerNorm(cfg.d_llama),
        )

        # Learned deterministic initial edges (audit C1). Without this the
        # substrate would re-randomize every forward, leaving training-time
        # non-determinism AND requiring keep_gate to "overwrite garbage"
        # rather than "update meaningful prior". Random init kept here as
        # the PARAMETER init seed (one-shot at module construction).
        self.init_src = nn.Parameter(
            torch.randn(self.K_max, self.d_node) * (self.d_node ** -0.5)
        )
        self.init_dst = nn.Parameter(
            torch.randn(self.K_max, self.d_node) * (self.d_node ** -0.5)
        )
        self.init_state = nn.Parameter(
            torch.randn(self.K_max, self.d_state) * (self.d_state ** -0.5)
        )
        # Saliency starts low (sigmoid(-4) ≈ 0.018) — slots are
        # uninformative until the updater raises saliency via its delta head.
        self.init_saliency_logit = -4.0

    # ── Streaming interface ───────────────────────────────────────────────
    def init_streaming_state(self, batch_size: int, device, dtype):
        from .graph_substrate import init_graph_state
        edges = init_graph_state(
            batch_size, self.K_max, self.d_node, self.d_state,
            device, dtype=next(self.pin_encoder.parameters()).dtype,
            init_src=self.init_src,
            init_dst=self.init_dst,
            init_state=self.init_state,
            init_saliency_logit=self.init_saliency_logit,
        )
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "edges": edges,
            "aux_accum": zero.clone(),
            "L_connect_accum": zero.clone(),
            "L_adjust_accum": zero.clone(),
            "n_windows": 0,
        }

    def streaming_write(
        self, state, token_embeds, attention_mask=None, chunk_offset=0,
    ):
        """One write pass over a window. Updates all K_max edge slots via
        the transformer-updater + soft-snap on endpoints."""
        from .graph_substrate import soft_snap, loss_connectivity, loss_adjust

        w_dtype = next(self.pin_encoder.parameters()).dtype
        token_embeds = token_embeds.to(w_dtype)

        # All-padded window: substrate unchanged
        if attention_mask is not None and not attention_mask.any():
            return state, {}

        # Encode pins
        pins = self.pin_encoder(token_embeds)                    # [B, T_w, d_updater]
        # Audit2 #5: add sinusoidal PE so within-window token order matters.
        # Without this the updater's cross-attn over pins is order-invariant.
        T_w = pins.shape[1]
        pe = _sinusoidal_pe(T_w, pins.shape[-1], offset=chunk_offset,
                             device=pins.device, dtype=pins.dtype)
        pins = pins + pe.unsqueeze(0)

        if attention_mask is not None:
            pins_pad_mask = ~attention_mask
            all_pad_rows = pins_pad_mask.all(dim=-1)
            if all_pad_rows.any():
                pins_pad_mask = pins_pad_mask.clone()
                pins_pad_mask[all_pad_rows, 0] = False
            has_real = attention_mask.any(dim=-1)
        else:
            pins_pad_mask = None
            has_real = None

        edges_old = state["edges"]
        proposed, gates = self.updater(pins, edges_old, pins_pad_mask=pins_pad_mask)

        # Soft snap: resolve proposed endpoints against the bank of existing
        # endpoints. endpoint_bank = concat all old src+dst across slots.
        endpoint_bank = torch.cat([edges_old["src"], edges_old["dst"]], dim=1)
        # [B, 2*K_max, d_node]
        snapped_src, max_sim_src = soft_snap(
            proposed["src"], endpoint_bank, gates["snap_gate_src"],
        )
        snapped_dst, max_sim_dst = soft_snap(
            proposed["dst"], endpoint_bank, gates["snap_gate_dst"],
        )

        # Apply keep_gate: preserve old vs use snapped proposal. Saliency
        # logit gets the same gate (audit M2) so keep=1 truly preserves
        # the slot rather than letting saliency drift independently.
        kg = gates["keep_gate"].unsqueeze(-1)
        kg_1d = gates["keep_gate"]                                # [B, K]
        edges_new = {
            "src": kg * edges_old["src"] + (1 - kg) * snapped_src,
            "dst": kg * edges_old["dst"] + (1 - kg) * snapped_dst,
            "state": kg * edges_old["state"]
                     + (1 - kg) * proposed["state"],
            "saliency_logit": (kg_1d * edges_old["saliency_logit"]
                               + (1 - kg_1d) * proposed["saliency_logit"]),
        }

        # Per-row all-pad protection on the edges update
        if attention_mask is not None:
            has_real_f = has_real.to(w_dtype)
            keep_mask = has_real_f.view(-1, 1, 1)
            keep_mask_1d = has_real_f.view(-1, 1)
            edges_new = {
                "src": edges_new["src"] * keep_mask + edges_old["src"] * (1 - keep_mask),
                "dst": edges_new["dst"] * keep_mask + edges_old["dst"] * (1 - keep_mask),
                "state": edges_new["state"] * keep_mask + edges_old["state"] * (1 - keep_mask),
                "saliency_logit": edges_new["saliency_logit"] * keep_mask_1d
                                  + edges_old["saliency_logit"] * (1 - keep_mask_1d),
            }

        # Aux losses
        L_connect = loss_connectivity(
            gates["snap_gate_src"], gates["snap_gate_dst"],
            max_sim_src, max_sim_dst,
            has_real=has_real,
        )
        L_adj = loss_adjust(edges_new, edges_old, has_real=has_real)

        weighted_aux = (
            self.lambda_connect * L_connect
            + self.lambda_adjust * L_adj
        )

        new_state = dict(state)
        new_state["edges"] = edges_new
        new_state["aux_accum"] = state["aux_accum"] + weighted_aux
        new_state["L_connect_accum"] = state["L_connect_accum"] + L_connect.detach()
        new_state["L_adjust_accum"] = state["L_adjust_accum"] + L_adj.detach()
        new_state["n_windows"] = state["n_windows"] + 1
        return new_state, {
            "graph_L_connect": L_connect.detach(),
            "graph_L_adjust": L_adj.detach(),
        }

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        """Project K_max edges to d_llama memory tokens (fused per-edge MLP),
        prepended by compute_qa_loss like the other prepend variants."""
        edges = state["edges"]
        B = edges["src"].shape[0]
        device = edges["src"].device
        dtype = next(self.proj_to_llama.parameters()).dtype

        # Pack per-edge features: (src, dst, state, saliency_scalar)
        sal = torch.sigmoid(edges["saliency_logit"]).unsqueeze(-1)    # [B, K_max, 1]
        edge_feats = torch.cat([
            edges["src"], edges["dst"], edges["state"], sal,
        ], dim=-1)                                                     # [B, K_max, 2d_n+d_s+1]
        memory = self.proj_to_llama(edge_feats.to(dtype))              # [B, K_max, d_llama]
        # Audit2 #4: gate the projected memory by saliency AFTER the
        # LayerNorm inside proj_to_llama. Without this, low-saliency
        # slots still emit full-scale Llama tokens (norm ~45) — the
        # saliency feature gets washed by norm. Multiply post-norm:
        # low-saliency edge → low-magnitude memory token, the decoder
        # mostly ignores it.
        memory = memory * sal.to(memory.dtype)

        n_w = max(state["n_windows"], 1)
        graph_aux = state["aux_accum"] / n_w

        with torch.no_grad():
            sal_mean = torch.sigmoid(edges["saliency_logit"]).mean()
            src_norm = edges["src"].norm(dim=-1).mean()
            # Endpoint reuse diagnostic: average cosine between all (src, dst) endpoints
            ep_bank = torch.cat([edges["src"], edges["dst"]], dim=1)   # [B, 2K, d]
            ep_norm = F.normalize(ep_bank, dim=-1, eps=1e-6)
            cos_matrix = ep_norm @ ep_norm.transpose(-1, -2)           # [B, 2K, 2K]
            # Mean off-diagonal cosine — higher = more node reuse
            K2 = ep_bank.shape[1]
            off_diag_mask = ~torch.eye(K2, dtype=torch.bool, device=device)
            ep_reuse = cos_matrix[:, off_diag_mask].mean()

        aux = {
            "load_balance_loss": torch.zeros((), device=device, dtype=dtype),
            "graph_aux": graph_aux,
            "graph_L_connect": state["L_connect_accum"] / n_w,
            "graph_L_adjust": state["L_adjust_accum"] / n_w,
            "graph_saliency_mean": sal_mean,
            "graph_src_norm": src_norm,
            "graph_endpoint_reuse": ep_reuse,
        }
        return memory, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Non-streaming forward — used by older code paths."""
        del mask_positions
        B = token_embeds.shape[0]
        device = token_embeds.device
        dtype = token_embeds.dtype
        state = self.init_streaming_state(B, device, dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)


class SplatBaselineEncoder(nn.Module):
    """Exp 3: Gaussian Splat baseline.

    Memory is a fixed-K signed Gaussian mixture in a shared latent space.
    Writes (per 1024-token window) call a TransformerUpdater that takes
    (encoded pins + current blobs) → outputs target blob parameters. 4
    write calls per 4096-token chunk; gradient flows through the BPTT chain
    of all 4 updater calls.

    Reads (per Llama position) emit K_rays ray directions and one origin
    from h_llama. Each ray's response = closed-form signed line integral
    of the density field. K_rays × 3 features (total, pos-only, neg-only)
    per position, projected to d_llama, injected via the MemInject pre-hook.

    Auxiliary losses (pin satisfaction, mass proportionality, proximal
    stickiness, sign-saturation) are accumulated per-window during write
    and packed into `aux["splat_aux"]` for compute_qa_loss to add.

    See docs/exp3_gaussian_splat_baseline.md.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

        # Substrate dimensions (typed ReprConfig fields — see config.py)
        self.d = cfg.splat_d
        self.K = cfg.splat_K
        self.K_rays = cfg.splat_K_rays
        self.n_updater_layers = cfg.splat_updater_layers
        self.inject_layer_idx = cfg.splat_inject_layer

        # Aux loss coefficients (importance weights; losses are pre-normalized)
        self.alpha_pin = cfg.splat_alpha_pin
        self.beta_prop = cfg.splat_beta_prop
        self.lambda_adj = cfg.splat_lambda_adj
        self.lambda_sat = cfg.splat_lambda_sat

        from .splat_substrate import TransformerUpdater

        # Pin encoder: project Llama token embeds → latent d
        self.pin_encoder = nn.Sequential(
            nn.Linear(cfg.d_llama, self.d * 2, bias=False),
            nn.GELU(),
            nn.Linear(self.d * 2, self.d),
            nn.LayerNorm(self.d),
        )

        # The transformer-updater that writes blob targets
        self.updater = TransformerUpdater(
            d=self.d, K=self.K,
            n_layers=self.n_updater_layers,
            n_heads=4, ffn_mult=4,
        )

        # ── Read path ──────────────────────────────────────────────
        # Origin: per-position projection d_llama → d
        self.origin_head = nn.Sequential(
            nn.Linear(cfg.d_llama, self.d, bias=False),
            nn.LayerNorm(self.d),
        )
        # Directions: per-position projection d_llama → K_rays × d (low-rank for params)
        self.direction_mid = self.d  # rank of the dir-head bottleneck
        self.direction_head = nn.Sequential(
            nn.Linear(cfg.d_llama, self.direction_mid * 2, bias=False),
            nn.GELU(),
            nn.Linear(self.direction_mid * 2, self.K_rays * self.d),
        )
        # Default xavier init — direction-magnitude doesn't matter (we
        # normalize to unit vectors), but the gradient signal back through
        # the unit-normalization needs the pre-norm vector to have
        # non-vanishing magnitude.
        with torch.no_grad():
            for m in self.direction_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        # Read post-projection: K_rays × 4 features → d_llama
        # (channels: I_total, I_pos, I_neg, log_max/d — see ray_features)
        self.ray_feat_per = 4
        ray_feat_dim = self.K_rays * self.ray_feat_per
        self.read_post = nn.Sequential(
            nn.Linear(ray_feat_dim, cfg.d_llama, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_llama, cfg.d_llama),
        )
        # Default xavier; `scale_raw` (init 0.05) provides the small-injection
        # bias. Aggressive gain=0.1 cascading kills grad signal at step 0.
        with torch.no_grad():
            for m in self.read_post:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        # ReZero-style tanh-bounded scale for the residual injection
        self.scale_max = 1.0
        scale_init = 0.05
        _scale_raw_init = float(
            torch.atanh(torch.tensor(scale_init / self.scale_max))
        )
        self.scale_raw = nn.Parameter(
            torch.full((cfg.d_llama,), _scale_raw_init),
        )

        # Learned deterministic initial blobs (audit2 #2). Previously
        # init_splat_state re-randomized per forward → same input gave
        # different memory. Now these nn.Parameter inits are expanded
        # across the batch in init_streaming_state for determinism.
        import math as _math
        self.init_mu = nn.Parameter(
            torch.randn(self.K, self.d) * (self.d ** -0.5)
        )
        self.init_log_diag_sigma = nn.Parameter(
            torch.randn(self.K, self.d) * 0.1
        )
        target_w = 1.0 / self.K
        self.init_w_raw = nn.Parameter(
            torch.full((self.K,), _math.log(_math.exp(target_w) - 1.0))
        )
        # Signs start near 0 (tanh ≈ 0, high-gradient region).
        self.init_s_logit = nn.Parameter(
            torch.randn(self.K) * 0.1
        )

    # ── Streaming interface ───────────────────────────────────────────────
    def init_streaming_state(self, batch_size: int, device, dtype):
        from .splat_substrate import init_splat_state
        # Use the encoder's compute dtype (fp32 for splat — high-D Gaussians
        # are precision-sensitive; cast inputs at the boundary).
        blobs = init_splat_state(
            batch_size, self.K, self.d, device,
            dtype=next(self.pin_encoder.parameters()).dtype,
            init_mu=self.init_mu,
            init_log_diag_sigma=self.init_log_diag_sigma,
            init_w_raw=self.init_w_raw,
            init_s_logit=self.init_s_logit,
        )
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "blobs": blobs,
            "aux_accum": zero.clone(),
            "L_pin_accum": zero.clone(),
            "L_prop_accum": zero.clone(),
            "L_adj_accum": zero.clone(),
            "L_sat_accum": zero.clone(),
            "n_windows": 0,
        }

    def streaming_write(
        self, state, token_embeds, attention_mask=None, chunk_offset=0,
    ):
        """One write pass over a window. Updates blobs via TransformerUpdater.
        Computes per-window aux losses (pin/proportional/adjust/sign-sat)
        and accumulates into state["aux_accum"]."""
        from .splat_substrate import (
            loss_pin, loss_proportional, loss_adjust, loss_sign_saturation,
        )

        w_dtype = next(self.pin_encoder.parameters()).dtype
        token_embeds = token_embeds.to(w_dtype)

        # All-padded window: the substrate must not change. Returning the
        # state untouched satisfies the all-pad streaming-write invariant
        # and avoids cross-attention NaN (softmax over all-masked keys).
        if attention_mask is not None and not attention_mask.any():
            return state, {}

        # Encode pins
        pins = self.pin_encoder(token_embeds)                    # [B, T_w, d]
        # Audit2 #5: sinusoidal PE so within-window token order matters.
        T_w = pins.shape[1]
        pe = _sinusoidal_pe(T_w, pins.shape[-1], offset=chunk_offset,
                             device=pins.device, dtype=pins.dtype)
        pins = pins + pe.unsqueeze(0)
        if attention_mask is not None:
            pins_pad_mask = ~attention_mask                       # True = padded
            # Defensively unmask position 0 in any row that's all-padded
            # (per-row protection — entire batch all-padded was handled above).
            all_pad_rows = pins_pad_mask.all(dim=-1)
            if all_pad_rows.any():
                pins_pad_mask = pins_pad_mask.clone()
                pins_pad_mask[all_pad_rows, 0] = False
        else:
            pins_pad_mask = None

        blobs_old = state["blobs"]
        blobs_new = self.updater(pins, blobs_old, pins_pad_mask=pins_pad_mask)

        # Per-row all-padded protection on the blob update: rows whose pin
        # window is all-padded should not have their blobs changed. We mask
        # the update by gating the new blob params with has_real.
        if attention_mask is not None:
            has_real = attention_mask.any(dim=-1).to(w_dtype)    # [B]
            keep = has_real.view(-1, 1, 1)                        # broadcast over (K, d)
            blobs_new = {
                "mu": blobs_new["mu"] * keep
                      + blobs_old["mu"] * (1 - keep),
                "log_diag_sigma": blobs_new["log_diag_sigma"] * keep
                      + blobs_old["log_diag_sigma"] * (1 - keep),
                "w_raw": blobs_new["w_raw"] * has_real.view(-1, 1)
                      + blobs_old["w_raw"] * (1 - has_real.view(-1, 1)),
                "s_logit": blobs_new["s_logit"] * has_real.view(-1, 1)
                      + blobs_old["s_logit"] * (1 - has_real.view(-1, 1)),
            }

        # Per-window aux losses — masked by has_real so all-padded rows
        # don't contribute spurious pressure to L_prop / L_adj / L_sat.
        pins_real_mask = attention_mask if attention_mask is not None else None
        if attention_mask is not None:
            has_real_rows = attention_mask.any(dim=-1)              # [B]
        else:
            has_real_rows = None
        L_pin = loss_pin(pins, pins_real_mask, blobs_new)
        L_prop = loss_proportional(pins_real_mask, blobs_new)
        L_adj = loss_adjust(blobs_new, blobs_old, has_real=has_real_rows)
        L_sat = loss_sign_saturation(blobs_new, has_real=has_real_rows)

        weighted_aux = (
            self.alpha_pin * L_pin
            + self.beta_prop * L_prop
            + self.lambda_adj * L_adj
            + self.lambda_sat * L_sat
        )

        new_state = dict(state)
        new_state["blobs"] = blobs_new
        new_state["aux_accum"] = state["aux_accum"] + weighted_aux
        # Accumulate sublosses separately for telemetry (averaged in finalize)
        for key, val in [
            ("L_pin_accum", L_pin), ("L_prop_accum", L_prop),
            ("L_adj_accum", L_adj), ("L_sat_accum", L_sat),
        ]:
            new_state[key] = state.get(key, torch.zeros((), device=val.device, dtype=val.dtype)) + val.detach()
        new_state["n_windows"] = state["n_windows"] + 1
        return new_state, {
            # Per-window diagnostics (detached, for jsonl logging)
            "splat_L_pin": L_pin.detach(),
            "splat_L_prop": L_prop.detach(),
            "splat_L_adj": L_adj.detach(),
            "splat_L_sat": L_sat.detach(),
        }

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        """Returns empty memory + blobs in aux (read happens via inject hook
        in compute_qa_loss). The accumulated aux loss is averaged over windows
        and packed under `splat_aux` for direct addition to total loss.

        Individual splat sublosses (L_pin/L_prop/L_adj/L_sat, averaged) are
        also surfaced as detached scalars for trainer logging + audit.
        """
        blobs = state["blobs"]
        B = blobs["mu"].shape[0]
        device = blobs["mu"].device
        dtype = next(self.read_post.parameters()).dtype
        empty_mem = torch.zeros(B, 0, self.cfg.d_llama, device=device, dtype=dtype)

        n_w = max(state["n_windows"], 1)
        splat_aux = state["aux_accum"] / n_w

        with torch.no_grad():
            sign_abs = torch.tanh(blobs["s_logit"]).abs().mean()
            w_mean = F.softplus(blobs["w_raw"]).mean()
            mu_norm = blobs["mu"].norm(dim=-1).mean()
            L_pin_avg = state["L_pin_accum"] / n_w
            L_prop_avg = state["L_prop_accum"] / n_w
            L_adj_avg = state["L_adj_accum"] / n_w
            L_sat_avg = state["L_sat_accum"] / n_w

        aux = {
            "load_balance_loss": torch.zeros((), device=device, dtype=dtype),
            "splat_aux": splat_aux,
            "splat_blobs": blobs,
            # Surfaced for trainer logging + verify_v1h aux dominance check
            "splat_L_pin": L_pin_avg,
            "splat_L_prop": L_prop_avg,
            "splat_L_adj": L_adj_avg,
            "splat_L_sat": L_sat_avg,
            # Diagnostics
            "splat_sign_abs_mean": sign_abs,
            "splat_w_mean": w_mean,
            "splat_mu_norm_mean": mu_norm,
        }
        return empty_mem, aux

    def inject(self, hidden_states: Tensor, blobs: dict) -> Tensor:
        """Per-position memory injection. Used by the pre-hook in
        compute_qa_loss. h_llama → (origin, K_rays directions) → line-integral
        ray features → residual injection.
        """
        from .splat_substrate import ray_features

        h_dtype = hidden_states.dtype
        w_dtype = next(self.origin_head.parameters()).dtype
        h = hidden_states.to(w_dtype) if h_dtype != w_dtype else hidden_states

        # B, T, d_llama
        B, T, _ = h.shape

        # Origin per Llama position
        o = self.origin_head(h)                                  # [B, T, d]

        # K_rays directions, normalized to unit length
        dirs = self.direction_head(h)                            # [B, T, K_rays·d]
        dirs = dirs.reshape(B, T, self.K_rays, self.d)
        dirs = F.normalize(dirs, dim=-1, eps=1e-6)               # unit vectors

        # Per-ray features: [B, T, K_rays, 4]
        # Cast blobs to compute dtype (they live in fp32; w_dtype is fp32 too)
        blobs_cast = {k: v.to(w_dtype) if v.dtype != w_dtype else v
                      for k, v in blobs.items()}
        feats = ray_features(o, dirs, blobs_cast)                # [B, T, K_rays, 4]

        # Flatten to [B, T, K_rays · 4] and project to d_llama
        feats_flat = feats.reshape(B, T, self.K_rays * self.ray_feat_per)
        inj_d = self.read_post(feats_flat)                       # [B, T, d_llama]

        eff_scale = self.scale_max * torch.tanh(self.scale_raw)  # [d_llama]
        inj = eff_scale * inj_d                                  # [B, T, d_llama]

        out = hidden_states + (inj.to(h_dtype) if inj.dtype != h_dtype else inj)
        return out

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Non-streaming forward — only used by older code paths."""
        del mask_positions
        B = token_embeds.shape[0]
        device = token_embeds.device
        dtype = token_embeds.dtype
        state = self.init_streaming_state(B, device, dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)

