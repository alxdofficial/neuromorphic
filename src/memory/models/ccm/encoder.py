"""CCM — Compressed Context Memory (Kim et al., ICLR 2024) as a memory encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


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
        # Compute the low-rank update in fp32 (matching the decoder's LoRALinear)
        # so CCM and ICAE differ only in mechanism, not adapter precision.
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return out + m.to(out.dtype) * self.scale * delta.to(out.dtype)


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
        from ...decoder import load_frozen_llama
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
        with torch.no_grad():   # seed norm-match scale to the backbone embed norm
            self.norm.scale.data.fill_(   # (0.9 default is ~3x too quiet on SmolLM2; match hlvocab)
                base.get_input_embeddings().weight.float().norm(dim=-1).mean().item())
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
        B = token_embeds.shape[0]
        has_real = attention_mask.any(dim=1)          # [B]: this window has ≥1 real token
        prefix = state["mem"]                         # condition on prior memory (None@first window)
        h_comp = self._comp_forward(prefix, token_embeds, attention_mask)
        if self.fold == "merge":
            # Per-row running mean over REAL windows only. t counts each row's real
            # windows, so the 1/t weight tracks that row's own update count; an all-pad
            # window (short row, late in a mixed-length batch) must carry the prefix
            # UNCHANGED rather than blend in padding-derived COMP and dilute the mean.
            prev_t = state["t"]
            if not torch.is_tensor(prev_t):           # promote scalar init → per-row counter
                prev_t = torch.zeros(B, device=token_embeds.device, dtype=torch.float32)
            t = prev_t + has_real.to(prev_t.dtype)    # [B]
            if prefix is None:
                new_mem = h_comp                      # first window: nothing to carry
            else:
                # weight in prefix dtype (bf16) so blended/prefix match for torch.where;
                # the t counter itself stays float32 for exact per-row window counts.
                tt = t.clamp(min=1).to(prefix.dtype).view(-1, 1, 1)
                blended = (1.0 - 1.0 / tt) * prefix + (1.0 / tt) * h_comp
                new_mem = torch.where(has_real.view(-1, 1, 1), blended, prefix)
        else:  # concat: grows memory; all-pad rows append a prefix-only COMP summary
            t = state["t"] + 1
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
