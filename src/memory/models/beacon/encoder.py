"""Activation Beacon (Zhang et al., BAAI, arXiv:2401.03462) as a memory encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


class _BeaconRoutedLinear(nn.Module):
    """Routes <beacon> positions through a SEPARATE trainable projection
    (cloned-init from the frozen base), ordinary positions through the frozen
    base:  out = where(beacon_mask==1, beacon_proj(x), base(x)).

    This is Beacon's signature — separate full per-layer projections (NOT a LoRA
    delta), which is what distinguishes it from CCM's gated LoRA. The per-forward
    beacon_mask [B,T,1] is read from a shared 1-element holder.
    """

    def __init__(self, base_linear: nn.Linear, mask_holder: list):
        super().__init__()
        self.base = base_linear                       # frozen
        self.beacon = nn.Linear(base_linear.in_features, base_linear.out_features,
                                bias=base_linear.bias is not None)
        with torch.no_grad():                         # clone-init from the base (paper §3.1)
            self.beacon.weight.copy_(base_linear.weight)
            if base_linear.bias is not None:
                self.beacon.bias.copy_(base_linear.bias)
        self._mask = mask_holder

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        m = self._mask[0]
        if m is None:
            return base_out
        # Cast the fp32 beacon weight to x's dtype so this works BOTH under bf16
        # autocast (training) and WITHOUT it (eval / AR-decode) — keeps fp32
        # master weights, mirrors the LoRA cast. Avoids the fp32-weight×bf16-input
        # crash on the non-autocast eval path (review BUG 1).
        bias = self.beacon.bias.to(x.dtype) if self.beacon.bias is not None else None
        beacon_out = F.linear(x, self.beacon.weight.to(x.dtype), bias)
        return torch.where(m.to(torch.bool), beacon_out, base_out)


class BeaconBaselineEncoder(nn.Module):
    """Activation Beacon (Zhang et al., BAAI, arXiv:2401.03462) as a memory encoder.

    Faithful port preserving Beacon's distinctive axes vs ICAE/CCM: SEPARATE full
    beacon q/k/v projections per layer (clone-init, routed to beacon positions —
    NOT LoRA), INTERLEAVED beacon tokens (one per α-unit), and streaming with the
    accumulated beacons carried forward (attend_prev). Like ICAE/CCM, the native
    per-layer KV memory is read out as the beacons' LAST-LAYER hidden states → M
    vectors in d_llama (caveat: discards the per-layer-KV axis, ~8× capacity).
    Own frozen base copy. Trainable = separate beacon projections + beacon embed +
    norm (heavy — ~100M with q,k,v on 1B; that is faithful, report it).

    Simplifications vs the paper (documented): causal masking (not the custom
    full-coverage mask), and clone-init beacon embed from the embedding mean
    (paper inits from EOS). memory: [B, M, d_llama], M = Σ_windows ceil(W/α).
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        base.model.gradient_checkpointing_enable(   # cap the 1B-forward activation peak (~100M beacon proj)
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self._mask = [None]
        # cfg.beacon_param uses the paper's short names ("q","k","v"); normalize
        # to HF projection names so the wrap loop matches.
        targets = {t if t.endswith("_proj") else f"{t}_proj" for t in cfg.beacon_param}
        wrap_layers = set(getattr(cfg, "beacon_wrap_layers", ()) or ())   # () = all (capacity knob)
        self._beacon_layers = []
        n_wrapped = 0
        for li, layer in enumerate(base.model.layers):
            if wrap_layers and li not in wrap_layers:
                continue
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if name in targets and hasattr(attn, name):
                    orig = getattr(attn, name)
                    if not isinstance(orig, nn.Linear):
                        continue
                    w = _BeaconRoutedLinear(orig, self._mask)
                    setattr(attn, name, w)
                    self._beacon_layers.append(w)
                    n_wrapped += 1
        self.base = base
        _bc = base.config
        self.n_kv_heads = getattr(_bc, "num_key_value_heads", None) or _bc.num_attention_heads
        self.head_dim = getattr(_bc, "head_dim", None) or (_bc.hidden_size // _bc.num_attention_heads)
        # NATIVE Beacon read = per-layer KV (the beacon key/value activations at EVERY layer),
        # restored via the shared per-layer-KV path. beacon_read='prepend' keeps the legacy
        # last-hidden prepend as an A/B control.
        self.reads_per_layer_kv = (getattr(cfg, "beacon_read", "kv") == "kv")
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(dim=0)
            emb_std = embed.weight.float().std().item()
        self.beacon_embed = nn.Parameter(
            mean_vec.view(1, cfg.d_llama) + emb_std * torch.randn(1, cfg.d_llama))
        self.norm = _NormMatch(cfg.d_llama)
        with torch.no_grad():   # seed norm-match scale to the backbone embed norm
            self.norm.scale.data.fill_(   # (0.9 default is ~3x too quiet on SmolLM2; match hlvocab)
                base.get_input_embeddings().weight.float().norm(dim=-1).mean().item())
        print(f"[Beacon] separate {sorted(targets)} projections on {n_wrapped} "
              f"linears; α={cfg.beacon_ratio or 32}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def _interleave(self, window_emb, window_mask, alpha):
        """Split the window into α-units, append one <beacon> after each →
        [B, n_units*(α+1), d] + a boolean beacon mask. Pads to a multiple of α."""
        B, W, d = window_emb.shape
        n_units = -(-W // alpha)                      # ceil(W/α)
        pad = n_units * alpha - W
        if pad:
            window_emb = F.pad(window_emb, (0, 0, 0, pad))
            window_mask = F.pad(window_mask, (0, pad), value=False)
        we = window_emb.view(B, n_units, alpha, d)
        wm = window_mask.view(B, n_units, alpha)
        beac = self.beacon_embed.to(window_emb.dtype).view(1, 1, 1, d).expand(B, n_units, 1, d)
        # A beacon is attendable iff its α-unit holds ≥1 real token. An all-pad unit
        # (a short row's tail, or the final-window pad) must NOT contribute an attendable
        # beacon — marking it valid (the old `ones`) let real positions attend to a
        # padding-summary AND extracted it as a memory slot summarizing pure padding.
        bmk = wm.any(dim=2, keepdim=True)             # [B, n_units, 1] bool
        seq = torch.cat([we, beac], dim=2).reshape(B, n_units * (alpha + 1), d)
        msk = torch.cat([wm, bmk], dim=2).reshape(B, n_units * (alpha + 1))
        is_beacon = torch.zeros(B, n_units * (alpha + 1), dtype=torch.bool,
                                device=window_emb.device)
        is_beacon[:, alpha::alpha + 1] = True         # beacon after each α-unit
        return seq, msk, is_beacon

    def init_streaming_state(self, batch_size: int, device, dtype):
        return {"mem": None, "B": batch_size, "device": device, "dtype": dtype}

    def streaming_write(self, state, token_embeds, attention_mask=None,
                        chunk_offset=0, **extra):
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2],
                                        device=token_embeds.device, dtype=torch.bool)
        alpha = self.cfg.beacon_ratio or 32
        seq, msk, is_beacon = self._interleave(token_embeds, attention_mask.bool(), alpha)
        prefix = state["mem"]                         # accumulated beacons (attend_prev)
        prev_valid = state.get("valid")               # [B, n_prev] per-beacon validity (None@first)
        if prefix is not None:
            B = prefix.shape[0]
            # Carry the accumulated per-beacon validity as the prefix attention mask so a
            # padding beacon from an earlier window stays non-attendable here too (the old
            # all-ones would re-admit it as context). prev_valid aligns 1:1 with prefix beacons.
            pm = (prev_valid if prev_valid is not None
                  else torch.ones(B, prefix.shape[1], device=prefix.device, dtype=torch.bool))
            # is_beacon does double duty: it selects the beacon q/k/v projections AND
            # marks which positions are EXTRACTED as new beacons (`pos` below). Carried
            # prefix beacons are intentionally marked NOT-beacon: they are read as
            # condensed CONTEXT by this window (base projections) and must be excluded
            # from extraction so new_mem grows by only the fresh beacons (marking them
            # beacon would re-route AND re-extract them, double-counting memory).
            pb = torch.zeros(B, prefix.shape[1], device=prefix.device, dtype=torch.bool)
            seq = torch.cat([prefix.to(seq.dtype), seq], dim=1)
            msk = torch.cat([pm.bool(), msk], dim=1)
            is_beacon = torch.cat([pb, is_beacon], dim=1)
        # Capture per-layer beacon K,V (the NATIVE Beacon memory = keys/values at EVERY layer)
        # via forward-hooks on each layer's k_proj/v_proj; the NEW beacons' K,V accumulate across
        # windows exactly like the last-hidden readout below.
        Ln = len(self.base.model.layers)
        kbuf, vbuf, handles = [None] * Ln, [None] * Ln, []
        if self.reads_per_layer_kv:
            for _li, _layer in enumerate(self.base.model.layers):
                handles.append(_layer.self_attn.k_proj.register_forward_hook(
                    (lambda i: (lambda m, inp, o: kbuf.__setitem__(i, o)))(_li)))
                handles.append(_layer.self_attn.v_proj.register_forward_hook(
                    (lambda i: (lambda m, inp, o: vbuf.__setitem__(i, o)))(_li)))
        self._mask[0] = is_beacon.unsqueeze(-1).to(seq.dtype)
        try:
            h = self.base.model(inputs_embeds=seq, attention_mask=msk.long()).last_hidden_state
        finally:
            self._mask[0] = None
            for _hh in handles:
                _hh.remove()
        pos = is_beacon[0].nonzero(as_tuple=False).squeeze(-1)   # same across batch
        new_beacons = h[:, pos, :]                    # [B, n_beacon, d]  (telemetry / prepend mode)
        new_valid = msk[:, pos]                       # [B, n_beacon] bool: beacon's unit had real tokens
        new_mem = new_beacons if prefix is None else torch.cat([prefix, new_beacons], dim=1)
        all_valid = (new_valid if prev_valid is None
                     else torch.cat([prev_valid, new_valid], dim=1))
        out_state = {**state, "mem": new_mem, "valid": all_valid}
        if self.reads_per_layer_kv:
            Bn, nnew = seq.shape[0], int(pos.numel())
            def _to_kv(t):    # [B, seq, n_kv*hd] → beacon positions → [B, n_kv_heads, n_new, head_dim]
                return t[:, pos, :].view(Bn, nnew, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            Knew = [_to_kv(kbuf[i]) for i in range(Ln)]
            Vnew = [_to_kv(vbuf[i]) for i in range(Ln)]
            pk = state.get("past_kv")
            if pk is None:
                out_state["past_kv"] = (Knew, Vnew)
            else:
                Kacc, Vacc = pk
                out_state["past_kv"] = ([torch.cat([Kacc[i], Knew[i]], dim=2) for i in range(Ln)],
                                        [torch.cat([Vacc[i], Vnew[i]], dim=2) for i in range(Ln)])
        return out_state, {}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        """Legacy single-window path (non-QA losses). Closed-book: no question."""
        del mask_positions
        B = token_embeds.shape[0]
        state = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        mem = state["mem"]
        valid = state.get("valid")
        if mem is None:
            mem = torch.zeros(state["B"], 1, self.cfg.d_llama,
                              device=state["device"], dtype=torch.float32)
            valid = torch.ones(state["B"], 1, device=state["device"], dtype=torch.bool)
        if self.reads_per_layer_kv and state.get("past_kv") is not None:
            # NATIVE per-layer-KV read: empty prepend memory (M=0 in the decoder layout); the
            # beacon K,V ride in aux['past_kv'] and are injected as a non-causal cache prefix.
            empty = torch.zeros(mem.shape[0], 0, self.cfg.d_llama,
                                device=mem.device, dtype=torch.float32)
            return empty, {"past_kv": state["past_kv"], "memory_mask": valid,
                           "read_mode": "per_layer_kv"}
        # Legacy prepend (beacon_read='prepend'): per-beacon validity → memory_mask so the decoder
        # never attends to a beacon that summarized only padding.
        aux = {} if valid is None else {"memory_mask": valid}
        return self.norm(mem.float()), aux
