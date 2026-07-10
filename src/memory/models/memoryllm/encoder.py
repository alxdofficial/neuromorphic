"""MemoryLLM (Wang et al. 2024, arXiv:2402.04624) — fixed-size per-layer latent memory pool
with a compress-then-RANDOM-DROP self-update, read as per-layer KV.

Faithful to MemoryLLM's defining axes: (1) a FIXED-SIZE pool of N latent vectors PER LAYER,
(2) self-update = compress the new text into K candidate vectors + RANDOMLY DROP K old slots +
insert the candidates (so old memory decays stochastically), (3) the memory is read as per-layer
KEY/VALUE — the NATIVE per-layer-KV read (not an input prepend), via the shared prefix-cache path.
Own frozen SmolLM2 copy + encoder-LoRA (q/k/v/o); the pool is the main trainable memory.

Candidate generation uses the DECOUPLED fallback (a parameterless SDPA of the pool's last-K slots
over this window's per-layer context hiddens) rather than MemoryLLM's exact in-model per-layer
update where text and memory co-attend — a documented tractable approximation (see the blueprint).
ASTERISKS for the results doc: per-layer KV byte footprint (L·N·d, ~60× a prepend arm); stochastic
random-drop state (non-deterministic — seed for eval); native objective is loss-neutral CE.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...config import ReprConfig


class MemoryLLMBaselineEncoder(nn.Module):
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
            rank=int(getattr(cfg, "memoryllm_lora_rank", 46)),
            alpha=int(getattr(cfg, "memoryllm_lora_alpha", 92)),
            target_names=tuple(getattr(cfg, "memoryllm_lora_target_names",
                                       ("q_proj", "k_proj", "v_proj", "o_proj"))))
        base.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        _bc = base.config
        self.L = _bc.num_hidden_layers
        self.n_kv = getattr(_bc, "num_key_value_heads", None) or _bc.num_attention_heads
        self.head_dim = getattr(_bc, "head_dim", None) or (_bc.hidden_size // _bc.num_attention_heads)
        self.d = cfg.d_llama
        self.N = int(getattr(cfg, "memoryllm_n_mem", 0) or cfg.n_flat_codes)   # slots per layer
        self.K = max(1, min(int(getattr(cfg, "memoryllm_k_new", 8)), self.N))  # dropped+inserted / window
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean = embed.weight.float().mean(0)
            std = embed.weight.float().std().item()
        # per-layer learnable pool init: embed-table mean + noise, one pool of N vectors per layer.
        self.pool0 = nn.Parameter(mean.view(1, 1, self.d).repeat(self.L, self.N, 1)
                                  + std * torch.randn(self.L, self.N, self.d))
        print(f"[MemoryLLM] encoder-LoRA {n_wrapped} layers; per-layer pool "
              f"L={self.L}×N={self.N}×d={self.d}; random-drop K={self.K}/window (per-layer-KV read)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def init_streaming_state(self, batch_size, device, dtype):
        return {"pool": self.pool0.to(dtype).unsqueeze(1)
                .expand(self.L, batch_size, self.N, self.d).contiguous()}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        B, W, d = token_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        active = attention_mask.bool().any(dim=1)
        if not active.any():
            return state, {}
        # per-layer context hiddens for THIS window (hs[l] = input to layer l)
        out = self.base.model(inputs_embeds=token_embeds, attention_mask=attention_mask.long(),
                              output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                                   # tuple length L+1
        pool = state["pool"]                                     # [L, B, N, d]
        # BATCHED candidate SDPA across the L context depths (was a 30-iteration Python loop of tiny
        # GEMMs+softmaxes). Under the trainer's bf16 autocast the matmuls run in bf16 regardless, so the
        # old per-layer .float() upcasts were dead (autocast recasts) — dropped. softmax stays fp32-promoted
        # by autocast, as before. Result bit-identical to the per-layer loop.
        H = torch.stack([hs[l] for l in range(self.L)], dim=0)   # [L, B, W, d] input to each layer
        Q = pool[:, :, -self.K:, :]                              # [L, B, K, d] last-K slots compress the window
        scores = torch.matmul(Q, H.transpose(-1, -2)) / (d ** 0.5)          # [L, B, K, W]
        pad = (~attention_mask.bool()).view(1, B, 1, W)          # True where pad → mask out as keys
        scores = scores.masked_fill(pad, float("-inf"))
        attn_w = torch.softmax(scores, dim=-1).nan_to_num(0.0)              # all-pad → NaN → 0
        cand = torch.matmul(attn_w, H).to(pool.dtype)                       # [L, B, K, d]
        # RANDOM DROP K slots per layer (kept as a loop so the per-layer randperm RNG stream is
        # bit-identical to the pre-batch version). keep N-K survivors, insert candidates; idle-row freeze.
        new_layers = []
        for l in range(self.L):
            p_l = pool[l]                                        # [B, N, d]
            perm = torch.randperm(self.N, device=p_l.device)
            survivors = p_l[:, perm[:self.N - self.K], :]        # [B, N-K, d]
            proposed = torch.cat([survivors, cand[l]], dim=1)    # [B, N, d]
            proposed = torch.where(active.view(B, 1, 1), proposed, p_l)
            new_layers.append(proposed)
        return {"pool": torch.stack(new_layers, dim=0)}, {}

    def finalize_memory(self, state):
        pool = state["pool"]                                     # [L, B, N, d]
        B = pool.shape[1]
        Ks, Vs = [], []
        for l in range(self.L):
            attn = self.base.model.layers[l].self_attn
            p = pool[l]                                          # [B, N, d]
            k = attn.k_proj(p).view(B, self.N, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
            v = attn.v_proj(p).view(B, self.N, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
            Ks.append(k)                                         # [B, n_kv, N, head_dim]
            Vs.append(v)
        empty = torch.zeros(B, 0, self.d, device=pool.device, dtype=torch.float32)   # M=0 prepend
        mm = torch.ones(B, self.N, device=pool.device)
        return empty, {"past_kv": (Ks, Vs), "memory_mask": mm, "read_mode": "per_layer_kv"}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        st = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
