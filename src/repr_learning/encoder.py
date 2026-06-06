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
import functools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ReprConfig
from .selection import gumbel_argmax_ste, load_balance_loss, router_z_loss


class _NormMatch(nn.Module):
    """Put projected memory tokens in Llama's token-embedding magnitude region.

    The prepend projections used to end in nn.LayerNorm(d_llama), whose output has L2 norm
    ~sqrt(d_llama) ≈ 45 — ~49× the ~0.9 norm of real Llama token embeddings. Prepended
    unmasked, those 49×-loud tokens act as attention distractors the frozen LM cannot route
    around (the dominant reason the prepend baselines fell below the no-memory floor). Keep
    the LayerNorm (centering/stability) but rescale to ~0.9 (learnable). This is a magnitude
    CORRECTION, not added capacity — it does not change what the baseline is.
    """

    def __init__(self, d: int, target: float = 0.9):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(float(target)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.ln(x), dim=-1) * self.scale


@functools.lru_cache(maxsize=128)
def _sinusoidal_pe(seq_len: int, d_model: int, offset: int = 0,
                    *, device=None, dtype=torch.float32) -> Tensor:
    """Standard sinusoidal positional encoding for write-side pin tokens
    (audit2 #5). Without this, graph/splat/plastic writes are order-
    invariant within a 1024-token window (their pooling/cross-attn over
    pins doesn't carry token position).

    Returns [seq_len, d_model] PE for positions [offset, offset+seq_len).

    Cached across calls — same (seq_len, d_model, offset, device, dtype)
    returns the same tensor, avoiding the tiny arange+sin+cos compute on
    every streaming-write window (4× per chunk at chunk_size=8192).
    """
    if d_model <= 0 or seq_len <= 0:
        return torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        positions = torch.arange(offset, offset + seq_len,
                                  device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        angles = positions * div
        pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)[:, : pe[:, 0::2].shape[-1]]
        pe[:, 1::2] = torch.cos(angles)[:, : pe[:, 1::2].shape[-1]]
        return pe.to(dtype)


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
        # Positional encoding: non-parametric sinusoidal (_sinusoidal_pe), applied
        # per-window with absolute offset in forward(). This used to be a LEARNED
        # table sized [1, max(window_size_max, max_window_size), d_enc]; at the
        # operative chunk_size=8192 (trainer sets max_window_size=chunk_size) that
        # is [1, 8192, 816] = 6,684,672 trainable floats — load-bearing capacity
        # that handed flat/continuous/MT a ~13% memory-param edge over graph_v6
        # (sinusoidal, non-parametric) and mamba (no PE). Sinusoidal matches
        # graph_v6's substrate exactly and preserves the absolute-position design
        # intent (token@chunk_pos=p gets the p-th PE) at zero param cost, so the
        # head-to-head compares matched ~48M memory mechanisms.

    def forward(self, token_embeds: Tensor, attention_mask: Optional[Tensor] = None,
                position_offset: int = 0) -> Tensor:
        # token_embeds: [B, T, d_llama]; cast to encoder dtype (fp32 trainable)
        h = self.in_proj(token_embeds.to(self.in_proj.weight.dtype))   # [B, T, d_enc]
        T = h.shape[1]
        # position_offset enables window-aware encoding when called repeatedly
        # over windows of a longer chunk. v1g passes `position_offset=w*window_size`
        # per streaming write so token@chunk_pos=1500 gets pos_embed[1500], NOT
        # pos_embed[476] like it would if every window started from index 0.
        h = h + _sinusoidal_pe(T, h.shape[-1], offset=position_offset,
                               device=h.device, dtype=h.dtype)
        if attention_mask is not None:
            # nn.TransformerEncoder expects True = mask (don't attend)
            src_key_padding_mask = ~attention_mask         # True where padded
            if not src_key_padding_mask.any():
                # No padding anywhere in this window → drop the mask entirely so
                # SDPA can pick the FlashAttention kernel (O(T) memory; never
                # materializes or saves the T×T scores for backward). Masking
                # nothing is a mathematical no-op, so this is EXACT — not an
                # approximation. ~7 of 8 packed 1024-windows hit this path, which
                # is what lets the encoders fit at chunk=8192/BS=8 without (much)
                # activation checkpointing. The partially-filled tail window keeps
                # its mask (math backend, but one window's O(T²) is cheap).
                src_key_padding_mask = None
            else:
                # Defensive: when a sample's window is 100% padding (can happen
                # for shorter inputs like HotpotQA contexts that don't fill the
                # token budget across all 1024-token windows), the all-True
                # padding mask makes softmax over all-masked keys produce NaN.
                # Unmask position 0 in any such row — that row's output is
                # meaningless anyway (no valid tokens), but it stays finite and
                # the downstream slot state isn't contaminated for OTHER rows.
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

        # Dead-code revival (CVQ-VAE, Zheng et al. 2023): VQ codebooks collapse
        # to a small active subset (~341/4096 observed pre-fix). Track EMA pick
        # usage and periodically reseed never-picked codes from heavy users +
        # noise so the full codebook stays a fair discrete-bottleneck control.
        self.register_buffer("code_usage_ema", torch.zeros(cfg.n_nodes))
        self.register_buffer("_revival_step", torch.zeros((), dtype=torch.long))

        self.proj_code = nn.Sequential(
            nn.Linear(cfg.d_concept_baseline, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            _NormMatch(cfg.d_llama),     # v2.2: norm-match to Llama token scale (was LayerNorm → 49× OOD)
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

    @torch.no_grad()
    def _record_usage(self, code_id: Tensor) -> None:
        """EMA of per-code pick frequency (fraction of picks). Window-decayed."""
        n = self.cfg.n_nodes
        hist = torch.bincount(code_id.reshape(-1), minlength=n).to(
            self.code_usage_ema.dtype
        )
        hist = hist / hist.sum().clamp(min=1.0)
        decay = 1.0 - 1.0 / max(self.cfg.dead_code_revival_window, 1)
        self.code_usage_ema.mul_(decay).add_(hist, alpha=1.0 - decay)
        self._revival_step += 1

    @torch.no_grad()
    def _maybe_revive(self) -> None:
        """Every interval (post-warmup), reseed never-picked codes from heavy
        users + noise. Runs BEFORE the codebook is used this step so the
        in-place edit can't corrupt autograd's saved tensors."""
        cfg = self.cfg
        step = int(self._revival_step)
        if (step < cfg.dead_code_revival_warmup
                or step % cfg.dead_code_revival_interval != 0):
            return
        n = cfg.n_nodes
        w = self.code_usage_ema.clamp(min=0)
        if float(w.sum()) <= 0:
            return
        dead_idx = torch.nonzero(
            self.code_usage_ema < (0.01 / n), as_tuple=False,
        ).squeeze(-1)                                  # <1% of uniform usage
        if dead_idx.numel() == 0:
            return
        max_revive = max(1, n // 10)                   # cap per interval for stability
        if dead_idx.numel() > max_revive:
            perm = torch.randperm(dead_idx.numel(), device=dead_idx.device)
            dead_idx = dead_idx[perm[:max_revive]]
        donors = torch.multinomial(w, dead_idx.numel(), replacement=True)
        noise = torch.randn_like(self.concept_id.data[dead_idx])
        self.concept_id.data[dead_idx] = (
            self.concept_id.data[donors]
            + noise * cfg.dead_code_revival_noise_std
        )
        self.code_usage_ema[dead_idx] = self.code_usage_ema[donors]

    def finalize_memory(self, state: Tensor) -> tuple[Tensor, dict]:
        """Project the final slot-query state to memory tokens + aux losses."""
        cfg = self.cfg
        if self.training:
            self._maybe_revive()  # reseed dead codes BEFORE using the codebook
        code_q = self.code_head(state)
        scores = (code_q @ self.concept_id.T) * self.score_log_scale.exp()
        code_id, onehot = gumbel_argmax_ste(
            scores, cfg.selection_temperature, self.training,
        )
        code_emb = onehot @ self.concept_id
        memory = self.proj_code(code_emb)
        if self.training:
            self._record_usage(code_id)
        with torch.no_grad():
            ent = -(F.softmax(scores, dim=-1)
                    * F.log_softmax(scores, dim=-1)).sum(-1).mean()
            n_active = int((self.code_usage_ema > 0.01 / cfg.n_nodes).sum())
        aux = {
            "load_balance_loss": load_balance_loss(scores, picks=code_id),
            # z-loss caps unbounded growth of the selection logits (via
            # score_log_scale) that would otherwise kill gumbel exploration and
            # collapse the codebook. model.py already applies cfg.z_loss_coef.
            "z_loss": router_z_loss(scores),
            "picked_ids": code_id,
            "routing_entropy": ent,
            "codes_active": torch.tensor(float(n_active)),
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


class _CanonicalSlotAttention(nn.Module):
    """Faithful Slot Attention (Locatello et al. 2020, Algorithm 1).

    Iterative refinement with SHARED weights across iterations:
      - softmax-OVER-SLOTS competition + weighted-mean normalization,
      - a GRU update (input = attention update, hidden = previous slots),
      - a residual MLP after the GRU.
    The GRU + iteration loop (paired with stochastic shared-Gaussian slot
    init in the caller) is the canonical anti-collapse mechanism. No
    diversity/orthogonality loss is used — that is non-canonical.
    """

    def __init__(self, d_enc: int, n_iters: int = 3, d_ffn: Optional[int] = None):
        super().__init__()
        self.n_iters = n_iters
        self.scale = d_enc ** -0.5
        self.norm_inputs = nn.LayerNorm(d_enc)
        self.norm_slots = nn.LayerNorm(d_enc)
        self.norm_mlp = nn.LayerNorm(d_enc)
        self.to_q = nn.Linear(d_enc, d_enc, bias=False)
        self.to_k = nn.Linear(d_enc, d_enc, bias=False)
        self.to_v = nn.Linear(d_enc, d_enc, bias=False)
        self.gru = nn.GRUCell(d_enc, d_enc)
        d_ffn = d_ffn or 2 * d_enc
        self.mlp = nn.Sequential(
            nn.Linear(d_enc, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_enc),
        )

    def forward(
        self, slots: Tensor, inputs: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        n_iters: Optional[int] = None,
    ) -> Tensor:
        # slots: [B, N, d]  inputs: [B, T, d]  key_padding_mask: [B, T] True=pad
        B, N, d = slots.shape
        n = n_iters if n_iters is not None else self.n_iters
        inputs_n = self.norm_inputs(inputs)
        k = self.to_k(inputs_n)                       # [B, T, d]
        v = self.to_v(inputs_n)                       # [B, T, d]
        all_padded = (
            key_padding_mask.all(dim=-1) if key_padding_mask is not None else None
        )
        for _ in range(n):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale    # [B, N, d]
            attn = q @ k.transpose(-2, -1)            # [B, N, T]
            attn = attn.softmax(dim=1)                # compete OVER SLOTS
            if key_padding_mask is not None:
                attn = attn.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            attn = attn + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)          # weighted mean over keys
            updates = attn @ v                        # [B, N, d]
            if all_padded is not None and all_padded.any():
                updates = updates * (~all_padded).to(updates.dtype).view(-1, 1, 1)
            slots = self.gru(
                updates.reshape(B * N, d), slots_prev.reshape(B * N, d),
            ).reshape(B, N, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
            if all_padded is not None and all_padded.any():
                keep = (~all_padded).to(slots.dtype).view(-1, 1, 1)
                slots = slots * keep + slots_prev * (1.0 - keep)
        return slots


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
        N, d = cfg.n_flat_codes, cfg.d_enc
        # Canonical Slot Attention init (Locatello et al. 2020): slots are
        # SAMPLED from a single learned Gaussian N(mu, diag(exp(log_sigma)))
        # SHARED across all slots. Symmetry is broken by the sampled noise +
        # the iterative competition, NOT by per-slot learned embeddings — so
        # slots specialize through the algorithm rather than via hard-coded
        # diversity (the old orthogonal-init + diversity-loss did the latter,
        # which distorted the baseline).
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, d))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)
        # Deterministic eval: with shared mu/sigma all slots are identical
        # without noise; a fixed persistent eval-eps breaks symmetry
        # reproducibly while training samples fresh noise each step.
        self.register_buffer("slot_eval_eps", torch.randn(1, N, d))
        self.slot_attn = _CanonicalSlotAttention(d, n_iters=cfg.slot_iters)

        # Per-slot continuous head: d_enc → D_cont
        self.cont_head = nn.Sequential(
            nn.Linear(cfg.d_enc, cfg.d_enc), nn.GELU(),
            nn.Linear(cfg.d_enc, cfg.d_continuous),
        )

        # Project D_cont → d_llama
        self.proj_cont = nn.Sequential(
            nn.Linear(cfg.d_continuous, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            _NormMatch(cfg.d_llama),     # v2.2: norm-match to Llama token scale (was LayerNorm → 49× OOD)
        )

    def _init_slots(self, B: int, device, dtype) -> Tensor:
        """Sample slots from the learned shared Gaussian. Train: fresh noise.
        Eval: fixed persistent eps (deterministic, still symmetry-broken)."""
        del dtype  # encoder casts to its own param dtype internally
        sigma = torch.exp(self.slot_log_sigma)
        if self.training:
            eps = torch.randn(
                B, self.cfg.n_flat_codes, self.cfg.d_enc,
                device=self.slot_mu.device, dtype=self.slot_mu.dtype,
            )
        else:
            eps = self.slot_eval_eps.to(self.slot_mu.dtype).expand(B, -1, -1)
        return self.slot_mu + sigma * eps

    def _aux(self, slots: Tensor, memory: Tensor, cont_vec: Tensor) -> dict:
        """Diversity is TELEMETRY ONLY for collapse monitoring; it is added to
        the loss only when cfg.b_diversity_scale > 0 (default 0 = faithful Slot
        Attention, which prevents collapse via stochastic init + GRU + iters)."""
        with torch.amp.autocast("cuda", enabled=False):
            def _diversity(x):
                x_norm = F.normalize(x.float(), dim=-1)
                cos = x_norm @ x_norm.transpose(1, 2)
                M = cos.shape[1]
                eye = torch.eye(M, dtype=torch.bool, device=cos.device)
                off_diag = cos[:, ~eye].view(cos.shape[0], -1)
                return off_diag.pow(2).mean()
            diversity_slots = _diversity(slots)
            diversity_mem = _diversity(memory)
        return {
            "load_balance_loss": self.cfg.b_diversity_scale
            * (diversity_slots + diversity_mem),
            "diversity_slots_raw": diversity_slots,
            "diversity_mem_raw": diversity_mem,
            "cont_vec_norm": cont_vec.norm(dim=-1).mean(),
        }

    def init_streaming_state(self, batch_size: int, device, dtype):
        """v1g streaming init: sample per-batch slots from the learned shared
        Gaussian (canonical Slot Attention init)."""
        return self._init_slots(batch_size, device, dtype)

    def streaming_write(
        self, state: Tensor, token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None, chunk_offset: int = 0,
    ) -> tuple[Tensor, dict]:
        """One streaming write: canonical Slot Attention refinement over a new
        1024-token window, with `state` as the current slots. Slots carry
        across windows, so each window adds slot_iters refinement steps."""
        text_h = self.bi_transformer(token_embeds, attention_mask,
                                      position_offset=chunk_offset)
        kv_mask = ~attention_mask if attention_mask is not None else None
        state = self.slot_attn(state, text_h, key_padding_mask=kv_mask)
        return state, {}

    def finalize_memory(self, state: Tensor) -> tuple[Tensor, dict]:
        """Project final slot state to d_llama; diversity is telemetry only."""
        cont_vec = self.cont_head(state)
        memory = self.proj_cont(cont_vec)
        return memory, self._aux(state, memory, cont_vec)

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        del mask_positions
        B = token_embeds.shape[0]
        text_h = self.bi_transformer(token_embeds, attention_mask)
        kv_mask = ~attention_mask if attention_mask is not None else None
        slots = self._init_slots(B, token_embeds.device, token_embeds.dtype)
        slots = self.slot_attn(slots, text_h, key_padding_mask=kv_mask)
        cont_vec = self.cont_head(slots)                           # [B, M, D_cont]
        memory = self.proj_cont(cont_vec)                          # [B, M, d_llama]
        return memory, self._aux(slots, memory, cont_vec)


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
            _NormMatch(cfg.d_llama),     # v2.2: norm-match to Llama token scale (was LayerNorm → 49× OOD)
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
        """v1h QA retrieval — per-question-token retrieval (faithful MT).

        Each question token produces its own query; bank keys are scored
        against all of them and max-pooled, so a key is kept if ANY position
        wants it (MT's per-position mechanism), then top-K are returned. Uses
        in_proj only (no attention) to avoid leaking still-masked tokens.
        Returns [B, K, d_llama] retrieved memory tokens, one set per example."""
        keys = bank["keys"]
        values = bank["values"]
        attn_mask = bank["attention_mask"]
        B, T, d_value = keys.shape
        device = keys.device

        # Per-position retrieval (faithful Memorizing Transformers, Wu et al.
        # 2022): score the bank against EVERY question token, not one pooled
        # query. A bank key's score is the MAX over question tokens — a key is
        # retrieved if ANY decoding position wants it. This restores MT's core
        # mechanism (different facts for different positions) while still
        # emitting a fixed K memory tokens for budget parity. (in_proj only,
        # no attention — avoids leaking still-masked tokens, same as before.)
        q_proj = self.bi_transformer.in_proj(
            question_embeds.to(self.bi_transformer.in_proj.weight.dtype)
        )                                                        # [B, T_q, d_enc]
        per_tok_query = self.query_head(q_proj)                  # [B, T_q, d_value]
        per_tok_scores = torch.einsum("btd,bqd->bqt", keys, per_tok_query)  # [B, T_q, T]
        per_tok_scores = per_tok_scores.masked_fill(
            ~question_mask.bool().unsqueeze(-1), float("-inf"),
        )
        scores = per_tok_scores.max(dim=1).values                # [B, T]
        # Entirely-padded question rows → max over all -inf; finite-ize so the
        # downstream top-K/softmax guards behave (such rows have no valid query
        # and retrieve uniformly, matching the old pooled-query degenerate case).
        scores = torch.where(
            torch.isfinite(scores), scores, torch.zeros_like(scores),
        )
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
        # Re-zero invalid slots AFTER projection: proj_value's bias + LayerNorm
        # map zeroed retrieved vectors to NONZERO memory, injecting garbage
        # tokens into Llama. Mask again here so (a) padded top-K slots, (b)
        # all-pad-BANK rows, and (c) all-pad-QUESTION rows (no valid query
        # token → arbitrary retrieval) all stay exactly zero. (Audit 2026-05-29.)
        query_valid = question_mask.bool().any(dim=-1)            # [B]
        memory = memory * valid_topk.unsqueeze(-1).to(memory.dtype)
        memory = memory * any_valid.view(B, 1, 1).to(memory.dtype)
        memory = memory * query_valid.view(B, 1, 1).to(memory.dtype)

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


class ICAEBaselineEncoder(nn.Module):
    """ICAE — In-Context Autoencoder (Ge et al., ICLR 2024) as a memory encoder.

    Faithful port of ICAE's compressor to the streaming-encoder interface:
      * the encoder is a FROZEN Llama base + a trainable encoder-LoRA (q/v) +
        M learnable memory-slot embeddings;
      * passage embeds are accumulated across streaming windows, then the
        LoRA-Llama runs ONCE over [passage ++ M slots] at finalize;
      * the final hidden states at the M slot positions become the memory.

    Weight-share = option (A) (docs/emat_baselines_plan.md): the encoder owns
    its OWN frozen base copy (identical weights to the decoder's frozen base)
    so its encoder-LoRA cannot collide with the decoder's read-side LoRA. Costs
    one extra base in memory; zero adapter-collision risk; unambiguously
    faithful. Trainable = encoder-LoRA + slots + norm-match.

    Returns memory: [B, M, d_llama] (M = cfg.icae_n_slots or cfg.n_flat_codes).
    Closed-book: the question is NOT seen by the encoder (it goes to the
    decoder); ICAE ignores any question_embeds model.py stashes in the state.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        # Local import avoids any module-load cycle with decoder.py.
        from .decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base,
            rank=cfg.icae_lora_rank,
            alpha=cfg.icae_lora_alpha,
            target_names=tuple(cfg.llama_lora_target_names),
        )
        if getattr(cfg, "grad_checkpoint_llama", False):
            base.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        self.M = cfg.icae_n_slots or cfg.n_flat_codes
        # Slot embeddings: init at the base embedding-table mean (centered in
        # Llama's token region) + symmetry-breaking noise scaled to the
        # embedding std (principled — scales with the distribution, not a bare
        # constant; see feedback-no-magic-numbers).
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(dim=0)
            emb_std = embed.weight.float().std().item()
        slot_init = mean_vec.view(1, cfg.d_llama).repeat(self.M, 1)
        slot_init = slot_init + emb_std * torch.randn(self.M, cfg.d_llama)
        self.slots = nn.Parameter(slot_init)
        self.norm = _NormMatch(cfg.d_llama)
        print(f"[ICAE] encoder-LoRA wrapped {n_wrapped} layers "
              f"(rank={cfg.icae_lora_rank}); M={self.M} slots")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)   # base always in inference mode (no dropout)
        return self

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        """Legacy single-window path (recon/JEPA/HSM). EMAT uses the streaming
        interface directly; this exists so the non-QA loss paths don't crash on
        the ICAE variant. mask_positions is ignored (ICAE is closed-book)."""
        del mask_positions
        B = token_embeds.shape[0]
        state = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)

    def init_streaming_state(self, batch_size: int, device, dtype):
        d = self.cfg.d_llama
        return {
            "emb": torch.zeros(batch_size, 0, d, device=device, dtype=dtype),
            "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
        }

    def streaming_write(self, state, token_embeds, attention_mask=None,
                        chunk_offset=0, **extra):
        if attention_mask is None:
            attention_mask = torch.ones(
                token_embeds.shape[:2], device=token_embeds.device,
                dtype=torch.bool)
        new = {
            "emb": torch.cat([state["emb"], token_embeds], dim=1),
            "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1),
        }
        return new, {}

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        emb = state["emb"]                                   # [B, T, d]
        mask = state["mask"]                                 # [B, T] bool
        B, T, d = emb.shape
        slots = self.slots.to(emb.dtype).unsqueeze(0).expand(B, self.M, d)
        inp = torch.cat([emb, slots], dim=1)                 # [B, T+M, d]
        slot_mask = torch.ones(B, self.M, device=mask.device, dtype=mask.dtype)
        attn = torch.cat([mask, slot_mask], dim=1).long()    # HF: 1=attend
        # Inner LlamaModel → last_hidden_state (skip the lm_head). Causal mask
        # means the appended slots (at the END) attend over all passage tokens.
        # grad_checkpoint_llama: HF's per-layer checkpointing is gated on
        # base.training (forced False here), so we explicitly checkpoint the
        # whole base forward instead — otherwise the flag is a silent no-op and
        # a long-context 1B encoder pass OOMs (adversarial review I2).
        def _run_base(inp_, attn_):
            return self.base.model(inputs_embeds=inp_,
                                   attention_mask=attn_).last_hidden_state
        if (self.training and torch.is_grad_enabled()
                and getattr(self.cfg, "grad_checkpoint_llama", False)):
            import torch.utils.checkpoint as _ckpt
            h = _ckpt.checkpoint(_run_base, inp, attn, use_reentrant=False)
        else:
            h = _run_base(inp, attn)                          # [B, T+M, d]
        mem = self.norm(h[:, -self.M:, :].float())           # [B, M, d_llama]
        return mem, {}


class _CompGatedLoRALinear(nn.Module):
    """LoRA wrapper whose low-rank update fires ONLY at masked (<COMP>) positions.

        out = base(x) + comp_mask · (alpha/rank) · (x Aᵀ) Bᵀ ,  comp_mask ∈ {0,1}

    The base Linear stays frozen; lora_A/lora_B train. The per-forward comp_mask
    [B,T,1] is read from a shared 1-element holder so every wrapped layer sees the
    same position mask without threading it through HF's attention signature. This
    is CCM's signature: text tokens pass through the frozen base unchanged; only
    <COMP> tokens get the adapter, so the model can't bypass memory.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, mask_holder: list):
        super().__init__()
        self.base = base_linear                       # frozen (caller froze it)
        self.lora_A = nn.Parameter(torch.zeros(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # B stays 0 ⇒ Δ starts at 0
        self.scale = alpha / rank
        self._mask = mask_holder

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        m = self._mask[0]
        if m is None:
            return out
        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)
        delta = (x @ A.t()) @ B.t()                   # [B, T, out_f]
        return out + m.to(out.dtype) * self.scale * delta


class CCMBaselineEncoder(nn.Module):
    """CCM — Compressed Context Memory (Kim et al., ICLR 2024) as a memory encoder.

    Faithful port preserving CCM's three signatures: (1) a conditional LoRA gated
    to <COMP> positions ONLY; (2) recurrence — each window's <COMP> tokens attend
    to the running memory of prior <COMP> outputs; (3) a merge (1/t running mean,
    fixed M) or concat (grows) fold. Like the ICAE port, the native per-layer KV
    memory is read out as the <COMP> tokens' LAST-LAYER hidden states → M vectors
    in d_llama space (caveat C1: drops per-layer KV injection; compensate via
    n_comp; report by floats). Own frozen base copy (option A) so the COMP-LoRA
    can't collide with the decoder's read-side LoRA. Trainable = COMP-LoRA +
    <COMP> embeds + norm.

    memory: [B, M, d_llama]  (merge: M=n_comp; concat: M=n_comp×n_windows).
    Closed-book: the question is not seen by the encoder.

    Note: the heavy per-window base forward lives in streaming_write, which the
    trainer wraps in activation-checkpointing — so no internal checkpoint here
    (unlike ICAE, whose forward is in finalize_memory).
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from .decoder import load_frozen_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        self._mask = [None]                           # shared per-forward comp-mask
        targets = set(cfg.ccm_lora_targets)
        self._lora_layers = []
        n_wrapped = 0
        for layer in base.model.layers:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if name in targets and hasattr(attn, name):
                    orig = getattr(attn, name)
                    if not isinstance(orig, nn.Linear):
                        continue
                    w = _CompGatedLoRALinear(orig, cfg.ccm_lora_rank,
                                             cfg.ccm_lora_alpha, self._mask)
                    setattr(attn, name, w)
                    self._lora_layers.append(w)
                    n_wrapped += 1
        self.base = base
        self.n_comp = cfg.ccm_n_comp or cfg.n_flat_codes
        self.fold = cfg.ccm_fold
        if self.fold not in ("merge", "concat"):
            raise ValueError(f"ccm_fold must be 'merge'|'concat', got {self.fold!r}")
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(dim=0)
            emb_std = embed.weight.float().std().item()
        comp_init = mean_vec.view(1, cfg.d_llama).repeat(self.n_comp, 1)
        comp_init = comp_init + emb_std * torch.randn(self.n_comp, cfg.d_llama)
        self.comp_embeds = nn.Parameter(comp_init)
        self.norm = _NormMatch(cfg.d_llama)
        print(f"[CCM] COMP-gated LoRA wrapped {n_wrapped} linears "
              f"(rank={cfg.ccm_lora_rank}); n_comp={self.n_comp}; fold={self.fold}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def _comp_forward(self, prefix, window_emb, window_mask):
        """Run [prefix_mem ++ window ++ COMP] through the base; return the COMP
        tokens' last-layer hiddens [B, n_comp, d]. Only COMP positions are gated."""
        B, _, d = window_emb.shape
        comp = self.comp_embeds.to(window_emb.dtype).unsqueeze(0).expand(B, self.n_comp, d)
        parts, masks = [], []
        if prefix is not None:
            parts.append(prefix.to(window_emb.dtype))
            masks.append(torch.ones(B, prefix.shape[1], device=window_emb.device,
                                    dtype=torch.bool))
        parts.append(window_emb); masks.append(window_mask.bool())
        parts.append(comp)
        masks.append(torch.ones(B, self.n_comp, device=window_emb.device, dtype=torch.bool))
        inp = torch.cat(parts, dim=1)
        attn = torch.cat(masks, dim=1).long()         # HF: 1=attend
        T = inp.shape[1]
        cm = torch.zeros(B, T, 1, device=inp.device, dtype=inp.dtype)
        cm[:, -self.n_comp:, :] = 1.0                 # gate the new COMP tokens only
        self._mask[0] = cm
        try:
            h = self.base.model(inputs_embeds=inp, attention_mask=attn).last_hidden_state
        finally:
            self._mask[0] = None                      # never leak the mask past this call
        return h[:, -self.n_comp:, :]

    def init_streaming_state(self, batch_size: int, device, dtype):
        return {"mem": None, "t": 0, "B": batch_size, "device": device, "dtype": dtype}

    def streaming_write(self, state, token_embeds, attention_mask=None,
                        chunk_offset=0, **extra):
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2],
                                        device=token_embeds.device, dtype=torch.bool)
        t = state["t"] + 1
        prefix = state["mem"]                         # condition on prior memory (None@t=1)
        h_comp = self._comp_forward(prefix, token_embeds, attention_mask)
        if self.fold == "merge":
            new_mem = (h_comp if prefix is None
                       else (1.0 - 1.0 / t) * prefix + (1.0 / t) * h_comp)
        else:  # concat
            new_mem = h_comp if prefix is None else torch.cat([prefix, h_comp], dim=1)
        return {**state, "mem": new_mem, "t": t}, {}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        """Legacy single-window path (non-QA losses). Closed-book: no question."""
        del mask_positions
        B = token_embeds.shape[0]
        state = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        mem = state["mem"]
        if mem is None:                               # empty-passage guard
            mem = torch.zeros(state["B"], self.n_comp, self.cfg.d_llama,
                              device=state["device"], dtype=torch.float32)
        return self.norm(mem.float()), {}


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


class FullContextEncoder(nn.Module):
    """Vanilla full-context — the CEILING reference.

    Passes the raw token embeddings through unchanged as "memory tokens."
    No encoding, no compression — Llama literally sees the full context
    prepended to the question. Establishes the upper bound: what frozen
    Llama achieves when the entire source is visible.

    Companion to NullEncoder (the floor with NO memory). Together they
    bracket every compressed memory variant:
        NullEncoder (floor)  ≤  any memory variant  ≤  FullContextEncoder (ceiling)

    Any variant that doesn't beat NullEncoder has useless memory; any that
    approaches FullContextEncoder has near-perfect compression. The variants
    in between tell us the bits-per-information cost of each architecture.

    Caveat: this variant has zero compression — its "memory" is the raw
    text (M = chunk_size = 4096). Reads via the same prepend pathway as
    other prepend variants. Llama-3.2-1B has a 128k context window so
    prepending 4k + question + answer fits comfortably.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

    def init_streaming_state(self, batch_size: int, device, dtype):
        # Start with an empty memory; streaming_write accumulates token embeds
        # across windows so that finalize_memory returns the full context.
        return {
            "context_embeds": None,
            "context_mask": None,
            "_B": batch_size,
            "_device": device,
            "_dtype": dtype,
        }

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0):
        # Accumulate raw embeddings across the streaming windows. attention_mask
        # is honored: padded positions still occupy slots but are zero-vectored
        # so the decoder's positional encoding tracks correctly.
        del chunk_offset
        if attention_mask is not None:
            mask = attention_mask.to(token_embeds.dtype).unsqueeze(-1)
            token_embeds = token_embeds * mask
        prev = state.get("context_embeds")
        prev_mask = state.get("context_mask")
        if prev is None:
            new = token_embeds
            new_mask = (
                attention_mask if attention_mask is not None
                else torch.ones(token_embeds.shape[0], token_embeds.shape[1],
                                dtype=torch.bool, device=token_embeds.device)
            )
        else:
            new = torch.cat([prev, token_embeds], dim=1)
            window_mask = (
                attention_mask if attention_mask is not None
                else torch.ones(token_embeds.shape[0], token_embeds.shape[1],
                                dtype=torch.bool, device=token_embeds.device)
            )
            new_mask = torch.cat([prev_mask, window_mask], dim=1)
        new_state = dict(state)
        new_state["context_embeds"] = new
        new_state["context_mask"] = new_mask
        return new_state, {}

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        ctx = state.get("context_embeds")
        mem_mask = state.get("context_mask")
        if ctx is None:
            ctx = torch.zeros(
                state["_B"], 0, self.cfg.d_llama,
                device=state["_device"], dtype=state["_dtype"],
            )
            mem_mask = torch.zeros(state["_B"], 0, dtype=torch.bool,
                                    device=state["_device"])
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=ctx.device, dtype=ctx.dtype,
            ),
            # v5.5: surface the real-token mask so model.py can mask out the
            # padded context positions in Llama's attention mask. Previously
            # padded slots were zero-vectored but still attended-to, letting
            # Llama use them as causal scratch space and contaminating the
            # vanilla_full_context "ceiling" reference.
            "memory_mask": mem_mask,
        }
        return ctx, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        del mask_positions
        if attention_mask is not None:
            mask = attention_mask.to(token_embeds.dtype).unsqueeze(-1)
            token_embeds = token_embeds * mask
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=token_embeds.device, dtype=torch.float32,
            ),
        }
        return token_embeds, aux


class RecurrentBaselineEncoder(nn.Module):
    """Baseline 5: Mamba state-space-model bottleneck.

    Mamba processes the full context as a recurrent state-space model,
    producing one hidden state per token. Those are narrowed to
    d_recurrent per token, then adaptively average-pooled to
    n_flat_codes (128) memory tokens and projected to d_llama.

    Pre-projection budget: n_flat_codes × d_recurrent floats / chunk.

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

        # Canonical Mamba stack: pre-norm RMSNorm residual blocks, matching
        # the official mamba_ssm Block (h = h + mixer(RMSNorm(h))). Prior
        # config used the bare mixer with no per-block norm — non-canonical
        # and destabilizing. One RMSNorm per layer + a final RMSNorm.
        self.mamba_norms = nn.ModuleList([
            nn.RMSNorm(cfg.d_mamba) for _ in range(cfg.mamba_n_layers)
        ])
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=cfg.d_mamba,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
            )
            for _ in range(cfg.mamba_n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_mamba)

        # Per-token bottleneck: d_mamba → d_recurrent
        self.bottleneck = nn.Linear(cfg.d_mamba, cfg.d_recurrent)

        # Project d_recurrent → d_llama
        self.proj_to_llama = nn.Sequential(
            nn.Linear(cfg.d_recurrent, cfg.d_llama // 2), nn.GELU(),
            nn.Linear(cfg.d_llama // 2, cfg.d_llama),
            _NormMatch(cfg.d_llama),     # v2.2: norm-match to Llama token scale (was LayerNorm → 49× OOD)
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

        # 2. Canonical pre-norm residual Mamba stack: h = h + mixer(RMSNorm(h)).
        # 4 layers at d=1024 fit the full 8192-token sweep in VRAM at BS=8
        # without activation checkpointing, so no recompute in backward.
        for norm, block in zip(self.mamba_norms, self.mamba_blocks):
            h = h + block(norm(h))
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


class GraphV6BaselineEncoder(nn.Module):
    """Graph v6: soft-pointer graph memory with a no-op-free, per-token read.

    See docs/graph_v6.md + src/repr_learning/graph_substrate_v6.py.

    WRITE (graph_v5 lineage): chunk-fresh (mu,sigma) node bank + soft-pointer edges,
      updated per window by a unified typed-token transformer (GraphV6Updater) with a
      per-token FFN readout + anchor gate. No proposal pool, no competitive write head.
    READ (plastic lineage): finalize_memory builds per-edge FACT-TOKENS (directional
      FiLM-by-state of materialized endpoints) and returns them in aux with an empty
      [B,0,d_llama] memory placeholder (M=0); compute_qa_loss installs a per-position
      pre-hook calling self.inject, which does soft retrieval over the fact-tokens at
      every decode position and fuses the result into that position's hidden state.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        self.K_node = cfg.graph_v6_K_node
        self.K_edge = cfg.graph_v6_K_edge
        self.d_node = cfg.graph_v6_d_node
        self.d_state = cfg.graph_v6_d_state
        d_read = cfg.graph_v6_d_read
        self.inject_layer_idx = getattr(cfg, "graph_v6_inject_layer", 8)

        from .graph_substrate_v5 import SoftPointer
        from .graph_substrate_v6 import (
            GraphV6Updater, GraphV6Gate, GraphV6FactBuilder, GraphV6FactReader,
        )

        d_up = cfg.graph_v6_d_updater
        self.pin_encoder = nn.Sequential(
            nn.Linear(cfg.d_llama, d_up * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_up * 2, d_up),
            nn.LayerNorm(d_up),
        )
        # chunk-fresh init params (no per-slot trained params)
        s = float(cfg.graph_v6_init_log_sigma)
        self.mu_node = nn.Parameter(torch.zeros(self.d_node))
        self.log_sigma_node = nn.Parameter(torch.full((self.d_node,), s))
        self.mu_state = nn.Parameter(torch.zeros(self.d_state))
        self.log_sigma_state = nn.Parameter(torch.full((self.d_state,), s))
        self.mu_q = nn.Parameter(torch.zeros(self.d_node))
        self.log_sigma_q = nn.Parameter(torch.full((self.d_node,), s))

        self.updater = GraphV6Updater(
            d_node=self.d_node, d_state=self.d_state, d=d_up,
            K_node=self.K_node, K_edge=self.K_edge, d_pin=d_up,
            n_layers=cfg.graph_v6_updater_layers, n_heads=cfg.graph_v6_updater_heads,
        )
        self.node_gate = GraphV6Gate(self.d_node, hidden=64,
                                     init_bias=cfg.graph_v6_node_gate_init_bias)
        self.edge_gate = GraphV6Gate(2 * self.d_node + self.d_state, hidden=64,
                                     init_bias=cfg.graph_v6_edge_gate_init_bias)
        self.read_pointer = SoftPointer(
            d_node=self.d_node, init_temperature=float(cfg.graph_v6_read_temperature),
            kv_split=True,
        )
        self.fact_builder = GraphV6FactBuilder(
            d_node=self.d_node, d_state=self.d_state, d_read=d_read,
            film_hidden=cfg.graph_v6_film_hidden, mlp_hidden=cfg.graph_v6_builder_mlp_hidden,
        )
        self.fact_reader = GraphV6FactReader(
            d_llama=cfg.d_llama, d_read=d_read,
            n_heads=cfg.graph_v6_read_heads, ffn_mult=cfg.graph_v6_read_ffn_mult,
        )

    # ── Streaming interface ──────────────────────────────────────────────
    def init_streaming_state(self, batch_size, device, dtype, seed=None):
        from .graph_substrate_v6 import init_graph_v6_state
        w_dtype = next(self.pin_encoder.parameters()).dtype
        gen = None
        # Deterministic-eval guard: the node/edge/q init noise is symmetry-
        # breaking randomness that SHOULD vary per step during training, but at
        # eval it must be FIXED so metrics are reproducible run-to-run (matches
        # the continuous/MT deterministic-eval-noise convention, v5.4 fix #2).
        # Without this, graph_v6 eval drew from the global RNG (gen=None) and
        # graph init changed between eval runs.
        if seed is None and not self.training:
            seed = 1234
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        state = init_graph_v6_state(
            B=batch_size, K_node=self.K_node, K_edge=self.K_edge,
            d_node=self.d_node, d_state=self.d_state,
            mu_node=self.mu_node, log_sigma_node=self.log_sigma_node,
            mu_state=self.mu_state, log_sigma_state=self.log_sigma_state,
            mu_q=self.mu_q, log_sigma_q=self.log_sigma_q,
            device=device, dtype=w_dtype, generator=gen,
        )
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {**state, "n_windows": 0,
                "node_gate_mean_accum": zero.clone(),
                "edge_gate_mean_accum": zero.clone()}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0):
        from .graph_substrate_v5 import _rmsnorm
        from .graph_substrate_v6 import _sinusoidal_pe
        w_dtype = next(self.pin_encoder.parameters()).dtype
        token_embeds = token_embeds.to(w_dtype)
        if attention_mask is not None and not attention_mask.any():
            return state, {}

        pins = self.pin_encoder(token_embeds)
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
            if not pins_pad_mask.any():
                pins_pad_mask = None
            has_real = attention_mask.any(dim=-1)
        else:
            pins_pad_mask = None
            has_real = None

        N_old, q_src_old, q_dst_old, st_old = (
            state["N"], state["q_src"], state["q_dst"], state["state"])
        tgt = self.updater(pins, pins_pad_mask, N_old, q_src_old, q_dst_old, st_old)

        # gate node bank (target = FFN readout)
        g_node = self.node_gate(N_old, tgt["N"])
        N_new = _rmsnorm(N_old + g_node.unsqueeze(-1) * (tgt["N"] - N_old))

        # gate edges (one anchor gate over the concatenated edge fields)
        edge_old = torch.cat([q_src_old, q_dst_old, st_old], dim=-1)
        edge_tgt = torch.cat([tgt["q_src"], tgt["q_dst"], tgt["state"]], dim=-1)
        g_edge = self.edge_gate(edge_old, edge_tgt)                # [B, K_edge]
        ge = g_edge.unsqueeze(-1)
        q_src_new = _rmsnorm(q_src_old + ge * (tgt["q_src"] - q_src_old))
        q_dst_new = _rmsnorm(q_dst_old + ge * (tgt["q_dst"] - q_dst_old))
        st_new = _rmsnorm(st_old + ge * (tgt["state"] - st_old))

        if has_real is not None:
            km = has_real.to(w_dtype).view(-1, 1, 1)
            N_new = N_new * km + N_old * (1 - km)
            q_src_new = q_src_new * km + q_src_old * (1 - km)
            q_dst_new = q_dst_new * km + q_dst_old * (1 - km)
            st_new = st_new * km + st_old * (1 - km)

        with torch.no_grad():
            ngm = g_node.float().mean().to(torch.float32)
            egm = g_edge.float().mean().to(torch.float32)
        new_state = dict(state)
        new_state.update(
            N=N_new, q_src=q_src_new, q_dst=q_dst_new, state=st_new,
            n_windows=state["n_windows"] + 1,
            node_gate_mean_accum=state["node_gate_mean_accum"] + ngm,
            edge_gate_mean_accum=state["edge_gate_mean_accum"] + egm,
        )
        return new_state, {"graph_v6_node_gate_mean": ngm, "graph_v6_edge_gate_mean": egm}

    def _build_facts(self, state, zero_state=False):
        """READ Stage A: materialize endpoints + directional FiLM-by-state fact tokens."""
        N, q_src, q_dst, st = state["N"], state["q_src"], state["q_dst"], state["state"]
        sp_k, sp_v = self.read_pointer.project_kv(N)
        src_ep, _ = self.read_pointer.attend(q_src, sp_k, sp_v)
        dst_ep, _ = self.read_pointer.attend(q_dst, sp_k, sp_v)
        return self.fact_builder(src_ep, dst_ep, st, zero_state=zero_state)

    def finalize_memory(self, state):
        N = state["N"]
        B, device = N.shape[0], N.device
        dtype = next(self.fact_reader.parameters()).dtype
        zero_state = bool(state.get("zero_state", False))   # finer ablation than zero_memory
        fact_value = self._build_facts(state, zero_state=zero_state)
        empty_mem = torch.zeros(B, 0, self.cfg.d_llama, device=device, dtype=dtype)
        n_w = max(state["n_windows"], 1)
        aux = {
            "load_balance_loss": torch.zeros((), device=device, dtype=dtype),
            # graph_aux=0 (no aux loss) — also flips compute_qa_loss's `graph_aux is not
            # None` guard True so the graph_v6_* telemetry pass-through actually fires.
            "graph_aux": torch.zeros((), device=device, dtype=dtype),
            "graph_v6_facts": {"value": fact_value},
            "graph_v6_node_gate_mean_avg": (state["node_gate_mean_accum"] / n_w).to(torch.float32),
            "graph_v6_edge_gate_mean_avg": (state["edge_gate_mean_accum"] / n_w).to(torch.float32),
            "graph_v6_fact_norm": fact_value.detach().float().norm(dim=-1).mean().to(torch.float32),
        }
        # ── Eval-only health telemetry (no grad, zero train-time cost) ────────
        # Diagnose whether the v6 mechanism is alive: dead read (rezero eff≈0),
        # ignored edge-state (state_effect≈0 violates no-op-free), collapsed node
        # bank (collapse_cos→1), degenerate soft-pointer read (entropy→0 over-sharp
        # or →log K diffuse; active_frac→0 = hub collapse).
        if not self.training:
            with torch.no_grad():
                fr = self.fact_reader
                eff = (fr.scale_max * torch.tanh(fr.scale_raw)).abs().mean()
                aux["graph_v6_rezero_scale_eff"] = eff.float().to(torch.float32)
                # v6.1: per-position read gate mean (from the last decode pass) — should
                # drop below 1 if the model learns to suppress the read where unhelpful.
                if getattr(fr, "_last_gate_mean", None) is not None:
                    aux["graph_v6_read_gate_mean"] = fr._last_gate_mean.to(torch.float32)
                fact_zero = self._build_facts(state, zero_state=True)
                aux["graph_v6_state_effect"] = (
                    (fact_value - fact_zero).float().norm(dim=-1).mean().to(torch.float32))
                Nf = N.float()
                Nf = Nf / Nf.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                cos = Nf @ Nf.transpose(1, 2)                          # [B, Kn, Kn]
                Kn = N.shape[1]
                offmask = ~torch.eye(Kn, dtype=torch.bool, device=N.device)
                aux["graph_v6_node_collapse_cos"] = cos[:, offmask].mean().to(torch.float32)
                sp_k, sp_v = self.read_pointer.project_kv(N)
                _, a_src = self.read_pointer.attend(state["q_src"], sp_k, sp_v)
                _, a_dst = self.read_pointer.attend(state["q_dst"], sp_k, sp_v)
                def _ent(a):                                           # mean read-pointer entropy
                    p = a.float().clamp_min(1e-9)
                    return (-(p * p.log()).sum(-1)).mean()
                aux["graph_v6_read_src_entropy"] = _ent(a_src).to(torch.float32)
                aux["graph_v6_read_dst_entropy"] = _ent(a_dst).to(torch.float32)
                picks = torch.cat([a_src.argmax(-1), a_dst.argmax(-1)], dim=1)  # [B, 2*K_edge]
                active = torch.zeros(N.shape[0], Kn, dtype=torch.bool, device=N.device)
                active.scatter_(1, picks, True)
                aux["graph_v6_node_active_frac"] = active.float().mean().to(torch.float32)
        return empty_mem, aux

    def inject(self, hidden_states, facts):
        """READ Stage B: per-position soft retrieval over fact-tokens (installed as a
        forward pre-hook on Llama layer `inject_layer_idx` by compute_qa_loss)."""
        return self.fact_reader(hidden_states, facts["value"])

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        """Non-streaming fallback (recon/HSM paths). The QA path uses streaming_write +
        finalize_memory + the per-position inject hook. Here we return a prepend
        projection of the fact-tokens so non-QA callers get usable memory."""
        del mask_positions
        B, device, dtype = token_embeds.shape[0], token_embeds.device, token_embeds.dtype
        state = self.init_streaming_state(B, device, dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        fact_value = self._build_facts(state)
        memory = self.fact_reader.W_out(fact_value).to(dtype)     # [B, K_edge, d_llama] (query-agnostic fallback)
        aux = {"load_balance_loss": torch.zeros((), device=device, dtype=memory.dtype)}
        return memory, aux

