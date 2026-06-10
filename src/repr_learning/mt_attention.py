"""Faithful Memorizing Transformers attention wrapper (Wu et al., ICLR 2022).

This replaces the broken `memorizing_baseline` (which did a one-shot
content-free-trigger prepend, a port artifact — see MEMORY.md
`project_mt_baseline_invalid`). Here we implement the REAL mechanism:

  * a non-differentiable kNN datastore of per-token (key, value) pairs gathered
    from the context, and
  * a single decoder layer whose self-attention is augmented with a kNN read:
    each query retrieves its top-k nearest memory keys, attends over their
    values, and BLENDS the memory readout with the ordinary local attention via
    a learned per-head sigmoid gate.

The kNN + gate math (L2-normalized dot-product retrieval, top-k, memory-softmax,
per-head learnable sigmoid gate blending local vs. memory attention output) is
adapted from lucidrains/memorizing-transformers-pytorch (MIT), specifically the
`KNNAttention` / `KNNMemory` gate:
    out = local_out * (1 - g) + memory_out * g ,   g = sigmoid(gate_bias)
ported onto HF's GQA LlamaAttention (Llama-3.2-1B: 32 Q heads / 8 KV heads,
head_dim 64). We keep the gate per KV-head (8 scalars) and broadcast it to the
4 Q-heads in each GQA group.

Faithfulness notes:
  * Keys/queries used for the datastore + scoring are taken BEFORE RoPE
    (pre-RoPE), removing the positional confound: a memory written at context
    position p must be retrievable from a decode position q regardless of the
    |p - q| RoPE phase. The LOCAL attention path still uses RoPE'd q/k as
    normal (that path is unchanged from base Llama).
  * The base LlamaAttention's projections (q/k/v/o) are reused verbatim and stay
    frozen; the ONLY trainable parameter this baseline adds is the per-head gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# repeat_kv / apply_rotary_pos_emb / ALL_ATTENTION_FUNCTIONS are reused verbatim
# from the installed transformers LlamaAttention so the local-attention path is
# byte-for-byte the base model's behavior.
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class MTKNNGate(nn.Module):
    """Per-KV-head learnable blend gate (the only trainable param MT adds).

    `gate_bias` is one scalar per KV head (8 for Llama-3.2-1B). The blend weight
    is g = sigmoid(gate_bias); at init_bias=0 → g=0.5 (equal local/memory mix).
    Faithful to lucidrains' per-head `nn.Parameter` gate.
    """

    def __init__(self, n_kv_heads: int, init_bias: float = 0.0):
        super().__init__()
        self.gate_bias = nn.Parameter(torch.full([n_kv_heads], float(init_bias)))


class MTLlamaAttention(nn.Module):
    """Wraps an existing LlamaAttention, adding a kNN memory read at this layer.

    NOT a subclass — `__getattr__` delegates q_proj/k_proj/v_proj/o_proj/config
    (and any other base attribute) to `self._base_attn`, so the wrapper holds no
    duplicate of the frozen projections.

    Three modes:
      * capture  (`_capture_mode=True`)        : stash PRE-RoPE keys + values for
        the datastore; return the plain base output (no blend).
      * read     (`_datastore is not None`)    : retrieve top-k memory values per
        query and blend into the local attention output.
      * off      (datastore None, not capture) : pure base LlamaAttention output
        (must equal vanilla numerically — OFF parity).
    """

    def __init__(self, base_attn: nn.Module, gate: MTKNNGate, topk: int = 32):
        super().__init__()
        # Register as a child so .to(device)/.parameters() include it (frozen).
        self._base_attn = base_attn
        self._gate = gate
        self._topk = int(topk)
        # Read-time state (set by the model around the decoder forward):
        self._datastore = None        # dict(keys=[B,n_ctx,H_kv,d], values=..., ctx_mask=[B,n_ctx] bool)
        self._capture_mode = False
        self._captured_kv = None      # tuple(keys, values) each [B, n_ctx, H_kv, d] (pre-RoPE keys)

    # Delegate unknown attribute lookups (q_proj, k_proj, v_proj, o_proj,
    # config, num_key_value_groups, head_dim, scaling, layer_idx, …) to the base.
    def __getattr__(self, name):
        # nn.Module stores its own children/params/buffers in __dict__ via
        # _parameters/_modules; defer to nn.Module.__getattr__ first, then base.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("_base_attn")
            if base is None:
                raise
            return getattr(base, name)

    # ---- datastore lifecycle ------------------------------------------------
    def set_datastore(self, keys: Tensor, values: Tensor, ctx_mask: Tensor) -> None:
        """keys/values: [B, n_ctx, H_kv, d_head] (pre-RoPE keys). ctx_mask: [B, n_ctx] bool."""
        self._datastore = {"keys": keys, "values": values, "ctx_mask": ctx_mask}

    def clear_datastore(self) -> None:
        self._datastore = None

    # ---- forward ------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        base = self._base_attn
        input_shape = hidden_states.shape[:-1]                  # [B, T]
        hidden_shape = (*input_shape, -1, base.head_dim)

        # --- q/k/v proj + view/transpose (verbatim from LlamaAttention.forward) ---
        query_states = base.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hq,T,d]
        key_states = base.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # [B,Hkv,T,d]
        value_states = base.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hkv,T,d]

        # Pre-RoPE copies for the datastore / kNN scoring (Risk 3: removes the
        # positional confound). [B, T, Hkv, d] / [B, Hq, T, d] layouts kept handy.
        pre_rope_keys = key_states                              # [B,Hkv,T,d]
        pre_rope_queries = query_states                         # [B,Hq,T,d]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, base.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            base,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not base.training else base.attention_dropout,
            scaling=base.scaling,
            **kwargs,
        )
        # attn_output: [B, T, Hq, d_head] (interface transposes back to (1,2)).
        # This is V_c (local-attention readout) — leave it untouched in OFF mode.

        # --- capture mode: stash PRE-RoPE keys + values; no blend ---
        if self._capture_mode:
            B, Hkv, T, d = pre_rope_keys.shape
            self._captured_kv = (
                pre_rope_keys.transpose(1, 2).detach().contiguous(),    # [B,T,Hkv,d]
                value_states.transpose(1, 2).detach().contiguous(),     # [B,T,Hkv,d]
            )
            # fall through to base reshape + o_proj (return plain base output)

        # --- read mode: blend kNN memory readout into the local output ---
        elif self._datastore is not None:
            attn_output = self._blend_memory(attn_output, pre_rope_queries)

        # --- reshape + o_proj (verbatim from LlamaAttention.forward) ---
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = base.o_proj(attn_output)
        return attn_output, attn_weights

    def _blend_memory(self, attn_output: Tensor, pre_rope_queries: Tensor) -> Tensor:
        """attn_output (V_c): [B, T, Hq, d_head]. pre_rope_queries: [B, Hq, T, d_head].

        Per KV-head h (group of 4 Q-heads): mean the group's pre-RoPE queries,
        L2-normalize q and the stored keys in fp32, score against the datastore,
        top-k + memory-softmax, gather top values, and blend per-head:
            attn_output[:, :, 4h:4h+4] = g_h * V_m_h + (1 - g_h) * V_c .
        """
        base = self._base_attn
        ds = self._datastore
        keys = ds["keys"]            # [B, n_ctx, Hkv, d]
        values = ds["values"]        # [B, n_ctx, Hkv, d]
        ctx_mask = ds["ctx_mask"]    # [B, n_ctx] bool
        out_dtype = attn_output.dtype
        B, T, Hq, d = attn_output.shape
        Hkv = keys.shape[2]
        group = base.num_key_value_groups        # = Hq // Hkv = 4
        n_ctx = keys.shape[1]
        topk = min(self._topk, n_ctx)
        gate = torch.sigmoid(self._gate.gate_bias)   # [Hkv], fp32

        # padded-context mask additive bias [B, 1, n_ctx]
        neg = torch.finfo(torch.float32).min
        ctx_bias = torch.where(
            ctx_mask.bool(), torch.zeros_like(ctx_mask, dtype=torch.float32),
            torch.full_like(ctx_mask, neg, dtype=torch.float32),
        ).unsqueeze(1)                                # [B, 1, n_ctx]

        # work per-group in fp32 (autocast disabled) for the normalize + scores.
        with torch.autocast(device_type=attn_output.device.type, enabled=False):
            q_pr = pre_rope_queries.float().view(B, Hkv, group, T, d)  # [B,Hkv,group,T,d]
            q_grp = q_pr.mean(dim=2)                                   # [B,Hkv,T,d]
            q_n = F.normalize(q_grp, dim=-1)                          # [B,Hkv,T,d]
            k_n = F.normalize(keys.float(), dim=-1)                    # [B,n_ctx,Hkv,d]
            k_n = k_n.permute(0, 2, 1, 3)                              # [B,Hkv,n_ctx,d]
            # scores [B,Hkv,T,n_ctx]
            scores = torch.einsum("bhtd,bhnd->bhtn", q_n, k_n)
            scores = scores + ctx_bias.unsqueeze(1)                   # broadcast [B,1,1,n_ctx]
            top_scores, top_idx = scores.topk(topk, dim=-1)           # [B,Hkv,T,topk]
            soft = F.softmax(top_scores, dim=-1)                      # memory-softmax [B,Hkv,T,topk]

            # gather top values: values [B,n_ctx,Hkv,d] -> [B,Hkv,n_ctx,d]
            v_h = values.float().permute(0, 2, 1, 3)                  # [B,Hkv,n_ctx,d]
            idx = top_idx.unsqueeze(-1).expand(B, Hkv, T, topk, d)    # [B,Hkv,T,topk,d]
            v_gathered = torch.gather(
                v_h.unsqueeze(2).expand(B, Hkv, T, n_ctx, d), 3, idx)  # [B,Hkv,T,topk,d]
            V_m = torch.einsum("bhtk,bhtkd->bhtd", soft, v_gathered)  # [B,Hkv,T,d]

        # blend per group; attn_output is [B,T,Hq,d] -> view group dim
        V_c = attn_output.view(B, T, Hkv, group, d)                   # [B,T,Hkv,group,d]
        V_m_b = V_m.permute(0, 2, 1, 3).unsqueeze(3).to(out_dtype)    # [B,T,Hkv,1,d]
        g = gate.view(1, 1, Hkv, 1, 1).to(out_dtype)                  # [1,1,Hkv,1,1]
        blended = g * V_m_b + (1.0 - g) * V_c                         # [B,T,Hkv,group,d]
        return blended.reshape(B, T, Hq, d)


def install_mt_wrapper(llama_model: nn.Module, layer_idx: int,
                       gate: MTKNNGate, topk: int) -> MTLlamaAttention:
    """Replace layers[layer_idx].self_attn with an MTLlamaAttention wrapper.

    `llama_model` is a LlamaForCausalLM; the decoder layers live at
    `llama_model.model.layers`. Raises if the layer is already wrapped
    (no double-wrap — Risk 6).
    """
    layers = llama_model.model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"mt_layer={layer_idx} out of range for {len(layers)}-layer Llama")
    orig = layers[layer_idx].self_attn
    if isinstance(orig, MTLlamaAttention):
        raise RuntimeError(
            f"layer {layer_idx} already wrapped with MTLlamaAttention (double-wrap)")
    wrapper = MTLlamaAttention(orig, gate, topk)
    setattr(layers[layer_idx], "self_attn", wrapper)
    return wrapper
