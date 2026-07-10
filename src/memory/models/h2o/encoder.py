"""H2O — Heavy-Hitter Oracle KV eviction (Zhang et al. 2023, arXiv:2306.14048).

TRAINING-FREE eval-only baseline. H2O keeps the M tokens that receive the most total attention
mass across all layers and heads (the "heavy hitters"), using the observation that a small subset
of tokens accumulates most of the attention in autoregressive LMs. This is the natural LM-intrinsic
competitor: no encoder training, no added params — just select M positions by attention score and
carry their per-layer KV as the memory.

Algorithm (our streaming adaptation):
  1. For each streaming window run the frozen LM with output_attentions=True.
  2. Accumulate per-token attention-received score: score[t] = Σ_{l,h} Σ_s attn[l,h,s,t]
     (how much total attention position t receives from all query positions, across all layers/heads).
  3. After all windows, keep the top-M positions by accumulated score.
  4. Run a SECOND frozen forward over only those M token embeds (no causal context — just the
     surviving tokens as a set) to get their per-layer K,V → `past_kv`.
  5. Decoder reads those M per-layer KV pairs via the shared `_prefix_kv_forward` path.

ASTERISKS (for the results table):
  * Training-free — no learnable encoder; only the shared read-LoRA on the decoder is trained.
  * Two forward passes per encode (score pass + KV pass).
  * Eviction uses attention scores from the LM's OWN causal context, not the decoder's.
  * Does NOT implement H2O's optional "recent" window bias (our M budget is already fixed).
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
        # Eager attention is REQUIRED for the score pass: sdpa/flash do not
        # return attention weights (output_attentions returns None). Eager is
        # slower but this arm is eval-only so the cost is acceptable.
        base, _ = load_frozen_llama(cfg.llama_model, attn_implementation="eager")
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        _bc = base.config
        self.L = _bc.num_hidden_layers
        self.n_kv = getattr(_bc, "num_key_value_heads", None) or _bc.num_attention_heads
        self.head_dim = getattr(_bc, "head_dim", None) or (_bc.hidden_size // _bc.num_attention_heads)
        self.d = cfg.d_llama
        self.M = int(getattr(cfg, "h2o_n_budget", 0) or cfg.n_flat_codes)
        self.recent_ratio = float(getattr(cfg, "h2o_recent_ratio", 0.1))
        # No trainable parameters — eval-only. The shared read-LoRA on the decoder is the only
        # thing that trains; this encoder is a pure attention-score KV selector.
        print(f"[H2O] training-free KV eviction; budget M={self.M}, recent_ratio={self.recent_ratio:.2f}; "
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

    @torch.no_grad()
    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("H2O.finalize_memory: empty context (T=0)")
        M = min(self.M, T)

        # ── Pass 1: score pass — run frozen LM with output_attentions to get heavy hitters ──
        # Each layer returns attn [B, n_heads, T, T]; score[t] = total attention received at t.
        # Use the full causal mask so the scores reflect the LM's natural attention pattern.
        base_dtype = next(self.base.parameters()).dtype
        # Sum-and-drop hooks: eager attention returns (attn_output, attn_weights)
        # per layer. Each hook accumulates the received-attention score (sum over
        # the query axis → how much attention each key position receives) then
        # REPLACES the weights with None, so only ONE layer's [B,H,T,T] matrix is
        # ever live (peak ~1.2GB vs ~90GB if output_attentions retained all layers).
        score = torch.zeros(B, T, device=emb.device, dtype=torch.float32)
        def _sd_hook(module, inp, out):
            if not (isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[1])):
                return out                                       # nothing to accumulate/drop
            aw = out[1]                                          # [B, n_heads, T_q, T_k]
            score.add_(aw.float().sum(dim=(1, 2)))               # received attention per key
            return (out[0], None) + tuple(out[2:])               # drop weights → free the matrix
        handles = [layer.self_attn.register_forward_hook(_sd_hook)
                   for layer in self.base.model.layers]
        try:
            self.base.model(inputs_embeds=emb.to(base_dtype), attention_mask=mask.long(),
                            output_attentions=True, use_cache=False)
        finally:
            for hh in handles:
                hh.remove()
        if float(score.abs().sum()) == 0.0:
            raise RuntimeError("H2O score pass captured no attention weights — the base is not "
                               "returning eager attention weights (check attn_implementation='eager').")
        # Mask out pad positions so they are never selected.
        score = score.masked_fill(~mask, float("-inf"))

        # H2O selection: optionally protect the most-recent r = recent_ratio×M tokens as "local"
        # (always kept regardless of score) — mirrors H2O's original local+heavy-hitter split.
        n_recent = max(0, min(int(self.recent_ratio * M), M - 1))
        n_heavy = M - n_recent
        keep_idx_list = []
        for b in range(B):
            s = score[b].clone()
            if n_recent > 0:
                # Protect the last n_recent valid positions: find their indices, zero their score
                # so they don't compete with the heavy hitter budget.
                valid_pos = mask[b].nonzero(as_tuple=False).squeeze(-1)
                recent_pos = valid_pos[-n_recent:] if len(valid_pos) >= n_recent else valid_pos
                s[recent_pos] = float("-inf")                              # exclude from heavy budget
                heavy_pos = s.topk(min(n_heavy, int(mask[b].sum())), largest=True).indices
                idx = torch.cat([heavy_pos, recent_pos]).unique()
            else:
                idx = s.topk(min(M, int(mask[b].sum())), largest=True).indices
            keep_idx_list.append(idx.sort().values)                        # sorted for KV coherence

        # Pad to the same length across the batch (use max, fill with zeros for all-pad rows).
        max_keep = max(len(i) for i in keep_idx_list)
        keep_mask = torch.zeros(B, max_keep, device=emb.device, dtype=torch.bool)
        for b, idx in enumerate(keep_idx_list):
            keep_mask[b, :len(idx)] = True

        # ── Pass 2: KV pass — run a SECOND frozen forward over only the selected M tokens ──
        # Each example has a DIFFERENT selection set, so we can't batch them naively. We build a
        # padded [B, max_keep, d] sub-sequence for each example and run one batched forward.
        sel_emb = torch.zeros(B, max_keep, d, device=emb.device, dtype=base_dtype)
        for b, idx in enumerate(keep_idx_list):
            sel_emb[b, :len(idx)] = emb[b, idx].to(base_dtype)

        # Capture per-layer K,V via hooks (same pattern as Gisting/Beacon).
        kbuf = [None] * self.L
        vbuf = [None] * self.L
        handles = []
        for li, layer in enumerate(self.base.model.layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(
                (lambda i: lambda m, ip, o: kbuf.__setitem__(i, o))(li)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(
                (lambda i: lambda m, ip, o: vbuf.__setitem__(i, o))(li)))
        try:
            self.base.model(
                inputs_embeds=sel_emb,
                attention_mask=keep_mask.long(),
                use_cache=False,
            )
        finally:
            for hh in handles:
                hh.remove()

        def _to_kv(t):
            # raw proj output [B, max_keep, n_kv*head_dim] → [B, n_kv, max_keep, head_dim]
            return t.view(B, max_keep, self.n_kv, self.head_dim).permute(0, 2, 1, 3)

        Ks = [_to_kv(kbuf[i]) for i in range(self.L)]
        Vs = [_to_kv(vbuf[i]) for i in range(self.L)]

        empty = torch.zeros(B, 0, d, device=emb.device, dtype=torch.float32)
        return empty, {"past_kv": (Ks, Vs), "memory_mask": keep_mask.float(),
                       "read_mode": "per_layer_kv"}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        st = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
