"""H2O heavy-hitter KV eviction (Zhang et al. 2023, arXiv:2306.14048) — FAITHFUL per-layer port.

TRAINING-FREE eval-only baseline. Now a faithful rendering of H2O's mechanism at our scale:
per LAYER (and per KV-head) we keep the tokens that receive the most attention mass (the "heavy
hitters") plus a most-recent local window, and we index-select those tokens' ORIGINAL contextualized
KEY/VALUE — i.e. the exact contextualized, real-position projections from a single full-context pass —
rather than re-encoding a selected subset. This matches H2O's `H2OKVCache_LayerWise` selection
(heavy-hitter + recent split, per layer/head) and its in-place eviction of the model's OWN cached KV.

DELIBERATE, DOCUMENTED deviations (still "H2O-inspired", not bit-identical):
  * GQA reduction: SmolLM2 is grouped-query (9 query-heads → 3 KV-heads). H2O's released code scores
    per query-head and would shape-crash under GQA; we sum the `group` query-heads that share a KV-head
    into one per-KV-head score (mirrors H2O's own "sum received attention" semantics), then select once
    per KV-head. This head→kv-head fold is our choice; upstream had no precedent (MHA-only models).
  * Single-shot vs incremental: H2O accumulates its heavy-hitter score across autoregressive decode
    steps and evicts a growing cache in place. We are an encoder: one eager pass over the buffered
    context gives the complete received-attention score in one shot — no running total / re-indexing.
  * Position-free injection: like every per-layer-KV arm here (gisting/memoryllm), the selected KV is
    injected position-free through the shared prefix-cache path (decoder.py). We preserve the original
    CONTEXT (no re-encode) but not H2O's original RoPE phase — consistent with the other arms so the
    comparison stays apples-to-apples.

Algorithm:
  1. Each streaming window appends its token embeds to a buffer (streaming_write).
  2. finalize_memory runs the frozen LM ONCE with eager output_attentions, capturing per layer BOTH
     the received-attention score (reduced to KV-heads) AND the original k_proj/v_proj outputs.
  3. Per (layer, KV-head): keep top-(M−r) heavy hitters + r most-recent = M tokens, index-selecting
     the captured ORIGINAL KV (no second forward, no re-encode).
  4. Decoder reads those M per-layer KV pairs via the shared `_prefix_kv_forward` path.

ASTERISKS (results table): training-free / eval-only (trainer skips this arm); one forward per encode;
eviction uses the LM's own causal attention scores; keep-count is layer/head-uniform (M).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...config import ReprConfig


class H2OBaselineEncoder(nn.Module):
    reads_per_layer_kv = True
    is_conditioned_read = False
    wants_surprise = False

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        # Eager attention is REQUIRED for the score pass: sdpa/flash do not return attention weights
        # (output_attentions returns None). Eager is slower but this arm is eval-only.
        base, _ = load_frozen_llama(cfg.llama_model, attn_implementation="eager")
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        _bc = base.config
        self.L = _bc.num_hidden_layers
        self.n_q = _bc.num_attention_heads
        self.n_kv = getattr(_bc, "num_key_value_heads", None) or _bc.num_attention_heads
        self.group = self.n_q // self.n_kv                 # GQA query-heads per kv-head (9//3 = 3)
        self.head_dim = getattr(_bc, "head_dim", None) or (_bc.hidden_size // _bc.num_attention_heads)
        self.d = cfg.d_llama
        self.M = int(getattr(cfg, "h2o_n_budget", 0) or cfg.n_flat_codes)
        self.recent_ratio = float(getattr(cfg, "h2o_recent_ratio", 0.1))
        # No trainable parameters — eval-only. The shared read-LoRA on the decoder is the only
        # thing that trains; this encoder is a pure attention-score KV selector.
        print(f"[H2O] faithful per-layer heavy-hitter eviction; budget M={self.M}, "
              f"recent_ratio={self.recent_ratio:.2f}; GQA fold {self.n_q}q→{self.n_kv}kv; "
              f"eval-only (no trainable encoder params)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def init_streaming_state(self, batch_size, device, dtype):
        return {"emb": torch.zeros(batch_size, 0, self.d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2],
                                        device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    @staticmethod
    def _select(score_vec, valid_mask, M, n_recent, n_heavy):
        """H2O heavy-hitter + recent selection → sorted kept ORIGINAL indices.
        Returns min(M, #valid) sorted positions: the r most-recent valid tokens (always kept) plus the
        top heavy hitters among the rest. When #valid ≤ M, keep every valid position in order."""
        valid_pos = valid_mask.nonzero(as_tuple=False).squeeze(-1)      # [#valid]
        if valid_pos.numel() <= M:
            return valid_pos                                           # keep all valid, in order
        s = score_vec.clone()
        s[~valid_mask] = float("-inf")
        if n_recent > 0:
            recent_pos = valid_pos[-n_recent:]
            s[recent_pos] = float("-inf")                              # protect from heavy competition
            heavy = s.topk(n_heavy, largest=True).indices             # #valid−r ≥ n_heavy here
            return torch.cat([heavy, recent_pos]).sort().values
        return s.topk(M, largest=True).indices.sort().values

    @torch.no_grad()
    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                       # [B,T,d], [B,T] bool
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("H2O.finalize_memory: empty context (T=0)")
        M = min(self.M, T)
        L, n_kv, hd, group = self.L, self.n_kv, self.head_dim, self.group
        base_dtype = next(self.base.parameters()).dtype

        # ── ONE eager pass: per layer capture the received-attention SCORE (folded to KV-heads) AND the
        #    original contextualized k_proj/v_proj outputs (real context + positions preserved). ──
        # attn hook computes the per-kv-head score immediately then DROPS the [B,H,T,T] matrix, so only
        # one layer's attention matrix is ever live (peak ~1GB, not ×L). k/v hooks snapshot the projections.
        score = [None] * L                                            # each [B, n_kv, T]
        kbuf = [None] * L                                             # each [B, T, n_kv, hd]
        vbuf = [None] * L

        def _attn_hook(li):
            def hook(module, inp, out):
                if not (isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[1])):
                    return out                                        # no weights to score
                aw = out[1]                                           # [B, n_q, T_q, T_k]
                s = aw.float().sum(dim=2)                             # [B, n_q, T_k] received-attention
                score[li] = s.view(B, n_kv, group, T).sum(dim=2)     # GQA fold → [B, n_kv, T]
                return (out[0], None) + tuple(out[2:])                # drop the matrix → free memory
            return hook

        def _proj_hook(buf, li):
            def hook(module, inp, out):
                buf[li] = out.view(B, T, n_kv, hd)                   # pre-RoPE, original-context K/V
            return hook

        handles = []
        for li, layer in enumerate(self.base.model.layers):
            handles.append(layer.self_attn.register_forward_hook(_attn_hook(li)))
            handles.append(layer.self_attn.k_proj.register_forward_hook(_proj_hook(kbuf, li)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(_proj_hook(vbuf, li)))
        try:
            self.base.model(inputs_embeds=emb.to(base_dtype), attention_mask=mask.long(),
                            output_attentions=True, use_cache=False)
        finally:
            for hh in handles:
                hh.remove()
        if score[0] is None or float(sum(s.abs().sum() for s in score)) == 0.0:
            raise RuntimeError("H2O score pass captured no attention weights — the base is not returning "
                               "eager attention weights (check attn_implementation='eager').")

        # ── per (layer, KV-head, batch): heavy-hitter + recent selection on ORIGINAL positions ──
        n_recent = max(0, min(int(self.recent_ratio * M), M - 1))
        n_heavy = M - n_recent
        # keep-COUNT per batch row is layer/head-invariant (depends only on #valid and M) → one shared mask.
        keep_cnt = torch.clamp(mask.sum(dim=1), max=M)                # [B]
        max_keep = int(keep_cnt.max().item())

        Ks, Vs = [], []
        for li in range(L):
            Kl = torch.zeros(B, n_kv, max_keep, hd, device=emb.device, dtype=base_dtype)
            Vl = torch.zeros(B, n_kv, max_keep, hd, device=emb.device, dtype=base_dtype)
            sc_l = score[li]                                          # [B, n_kv, T]
            for j in range(n_kv):
                for b in range(B):
                    idx = self._select(sc_l[b, j], mask[b], M, n_recent, n_heavy)   # sorted, ≤ max_keep
                    n = idx.numel()
                    Kl[b, j, :n] = kbuf[li][b, idx, j, :]
                    Vl[b, j, :n] = vbuf[li][b, idx, j, :]
            Ks.append(Kl)
            Vs.append(Vl)
            kbuf[li] = None
            vbuf[li] = None                                          # free the full-T buffers as we go

        # one shared per-batch mask: first keep_cnt[b] slots are real (uniform across layers/kv-heads).
        mm = (torch.arange(max_keep, device=emb.device)[None, :] < keep_cnt[:, None]).float()  # [B, max_keep]
        empty = torch.zeros(B, 0, d, device=emb.device, dtype=torch.float32)
        return empty, {"past_kv": (Ks, Vs), "memory_mask": mm, "read_mode": "per_layer_kv"}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        st = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
