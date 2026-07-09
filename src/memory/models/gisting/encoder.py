"""Gisting (Mu, Li & Goodman 2023, arXiv:2304.08467) — learnable GIST tokens compress the
preceding context into their per-layer KEY/VALUE activations ("gist caching"), which is the
NATIVE per-layer-KV read (not an input prepend). Own frozen SmolLM2 copy + encoder-LoRA (q/v,
same targets as ICAE for param-match); the gist-token embeddings + LoRA are trainable.

Faithful mechanism: append κ gist tokens after the context; a causal forward lets them attend
over the context; their per-layer K,V are the compressed memory. The paper's "tokens after the
gist may attend ONLY to the gist" is realized here by the DECODER reading only the gist KV (never
the raw context) — exactly the per-layer-KV read. Internal seg_len chunking (like AutoCompressor)
emits κ = M/n_segments gist tokens per segment so the single-shot MAE path and the multi-window
path both accumulate to the M budget. ASTERISK: per-layer KV byte footprint (~L× a prepend arm).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...config import ReprConfig


class GistingBaselineEncoder(nn.Module):
    reads_per_layer_kv = True

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base,
            rank=int(getattr(cfg, "gisting_lora_rank", 104)),
            alpha=int(getattr(cfg, "gisting_lora_alpha", 208)),
            target_names=tuple(getattr(cfg, "gisting_lora_target_names",
                                       cfg.llama_lora_target_names)))
        base.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        _bc = base.config
        self.L = _bc.num_hidden_layers
        self.n_kv = getattr(_bc, "num_key_value_heads", None) or _bc.num_attention_heads
        self.head_dim = getattr(_bc, "head_dim", None) or (_bc.hidden_size // _bc.num_attention_heads)
        self.d = cfg.d_llama
        self.M = int(getattr(cfg, "gisting_n_gist", 0) or cfg.n_flat_codes)       # total gist budget
        self.seg_len = int(getattr(cfg, "gisting_segment_len", 0) or 256)
        self.k = int(getattr(cfg, "gisting_gist_per_seg", 0) or max(1, self.M // 8))  # κ gist / segment
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean = embed.weight.float().mean(0)
            std = embed.weight.float().std().item()
        self.gist = nn.Parameter(mean.view(1, self.d).repeat(self.k, 1) + std * torch.randn(self.k, self.d))
        print(f"[Gisting] encoder-LoRA {n_wrapped} layers; κ={self.k} gist/seg "
              f"(seg_len={self.seg_len}) → per-layer-KV read")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def init_streaming_state(self, batch_size, device, dtype):
        return {"K": None, "V": None, "B": batch_size, "device": device, "dtype": dtype}

    def _capture_seg(self, seg, seg_mask):
        """Append κ gist tokens after the segment, run the frozen base, capture the gist tokens'
        per-layer K,V (via k_proj/v_proj hooks) → each layer [B, n_kv, κ, head_dim]."""
        B, W, d = seg.shape
        g = self.gist.to(seg.dtype).unsqueeze(0).expand(B, self.k, d)
        inp = torch.cat([seg, g], dim=1)
        attn = torch.cat([seg_mask.long(),
                          torch.ones(B, self.k, device=seg.device, dtype=torch.long)], dim=1)
        kbuf, vbuf, handles = [None] * self.L, [None] * self.L, []
        for _li, _layer in enumerate(self.base.model.layers):
            handles.append(_layer.self_attn.k_proj.register_forward_hook(
                (lambda i: (lambda m, ip, o: kbuf.__setitem__(i, o)))(_li)))
            handles.append(_layer.self_attn.v_proj.register_forward_hook(
                (lambda i: (lambda m, ip, o: vbuf.__setitem__(i, o)))(_li)))
        try:
            self.base.model(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        finally:
            for _hh in handles:
                _hh.remove()
        def _kv(t):     # gist positions → [B, n_kv, κ, head_dim]
            return t[:, -self.k:, :].view(B, self.k, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        return [_kv(kbuf[i]) for i in range(self.L)], [_kv(vbuf[i]) for i in range(self.L)]

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        B, W, d = token_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        Kacc, Vacc = state["K"], state["V"]
        n = 0 if Kacc is None else Kacc[0].shape[2]
        for s in range(0, W, self.seg_len):
            if n >= self.M:
                break
            sm = attention_mask[:, s:s + self.seg_len]
            if not sm.any():                                     # skip an all-pad segment
                continue
            Kn, Vn = self._capture_seg(token_embeds[:, s:s + self.seg_len], sm)
            if Kacc is None:
                Kacc, Vacc = Kn, Vn
            else:
                Kacc = [torch.cat([Kacc[i], Kn[i]], dim=2) for i in range(self.L)]
                Vacc = [torch.cat([Vacc[i], Vn[i]], dim=2) for i in range(self.L)]
            n += self.k
        return {**state, "K": Kacc, "V": Vacc}, {}

    def finalize_memory(self, state):
        K, V = state["K"], state["V"]
        if K is None:                                            # degenerate all-pad episode
            B, dev = state["B"], state["device"]
            z = [torch.zeros(B, self.n_kv, 1, self.head_dim, device=dev) for _ in range(self.L)]
            empty = torch.zeros(B, 0, self.d, device=dev, dtype=torch.float32)
            return empty, {"past_kv": (z, list(z)), "memory_mask": torch.zeros(B, 1, device=dev),
                           "read_mode": "per_layer_kv"}
        B, n = K[0].shape[0], K[0].shape[2]
        empty = torch.zeros(B, 0, self.d, device=K[0].device, dtype=torch.float32)   # M=0 prepend
        mm = torch.ones(B, n, device=K[0].device)
        return empty, {"past_kv": (K, V), "memory_mask": mm, "read_mode": "per_layer_kv"}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        st = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
