"""MemoryLLM (Wang et al. 2024, arXiv:2402.04624) — fixed-size per-layer latent memory pool
with a compress-then-RANDOM-DROP self-update, read as per-layer KV.

Faithful to MemoryLLM's defining axes: (1) a FIXED-SIZE pool of N latent vectors PER LAYER,
(2) self-update = compress the new text into K candidate vectors + RANDOMLY DROP K old slots +
insert the candidates (so old memory decays stochastically), (3) the memory is read as per-layer
KEY/VALUE — the NATIVE per-layer-KV read (not an input prepend), via the shared prefix-cache path.
Own frozen SmolLM2 copy + encoder-LoRA (q/k/v/o); the pool is the main trainable memory.

CANDIDATE GENERATION (faithful port, replacing the old parameterless-SDPA approximation): the window
CO-ATTENDS to the pool through the REAL frozen+LoRA self-attention layers. Per layer l, that layer's
last-K pool slots (`memory[l][-K:]`, MemoryLLM's recency window) are injected as a NON-CAUSAL KV prefix
(each layer sees its OWN pool slice — matching upstream's per-layer memory), the window is run through
the base with that prefix, and layer l's OUTPUT at the window's trailing-K valid positions becomes
layer l's new candidate. A shared learned `new_memory_positional_emb` cue marks those carrier positions
(upstream's `new_memory_positional_emb`; the per-position variant was "lethal" upstream → shared+zero-init).
Drop/insert uses ONE shared random permutation across all layers (upstream default drop_memory_per_layer
=False → same physical slot dropped at every depth).

DELIBERATE deviations (documented for the results doc): (a) the positional cue is added at the input
embeds rather than re-injected at every layer's residual (dominant effect; avoids per-layer hooks that
fight gradient-checkpointing); (b) the KV prefix is position-free (unrotated), consistent with our read
path and the other per-layer-KV arms, vs upstream's low real positions on the memory block; (c) frozen
backbone + LoRA (upstream trains the whole backbone) — this port fixes the write MECHANISM, it is not a
guarantee of an empirical win under the frozen-backbone constraint.
ASTERISKS: per-layer KV byte footprint (L·N·d, ~60× a prepend arm); stochastic random-drop state
(non-deterministic — seed for eval); native objective is loss-neutral CE.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...common import _NormMatch
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
        self.seg_len = int(getattr(cfg, "memoryllm_segment_len", 0) or 256)     # self-chunk single-shot MAE
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean = embed.weight.float().mean(0)
            std = embed.weight.float().std().item()
        # per-layer learnable pool init: embed-table mean + noise, one pool of N vectors per layer.
        self.pool0 = nn.Parameter(mean.view(1, 1, self.d).repeat(self.L, self.N, 1)
                                  + std * torch.randn(self.L, self.N, self.d))
        # MemoryLLM's new-memory cue: a single shared d-vector (zero-init) added at the trailing-K
        # "will-become-memory" carrier positions during the compression forward. Shared (NOT per-position
        # / per-K) — upstream found the per-position variant "lethal". Trains via requires_grad.
        self.new_memory_positional_emb = nn.Parameter(torch.zeros(self.d))
        # Candidates are RAW layer-output hiddens (deep residual-stream norms are large). Upstream's
        # backbone is TRAINED to keep memory-token norms controlled; ours is FROZEN, so unnormalized
        # candidates blow up the pool → huge KV → gnorm in the millions. Match them to the backbone token
        # scale (out_norm auto-calibrated to ~3.2 in ReprModel.__init__), matching pool0's embed-scale init.
        self.out_norm = _NormMatch(self.d)
        print(f"[MemoryLLM] encoder-LoRA {n_wrapped} layers; per-layer pool "
              f"L={self.L}×N={self.N}×d={self.d}; random-drop K={self.K}/window (per-layer-KV read); "
              f"co-attention compress (faithful)")

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
        if W <= self.seg_len:
            return self._write_window(state, token_embeds, attention_mask), {}
        # MAE hands the WHOLE span in one call → self-chunk into seg_len sub-windows so memoryllm runs its
        # real streaming self-update (reaching ~50% episode content over 8 windows) instead of ONE write
        # (only K=8/N=96 episode content, 88 slots left as pool0). Same self-chunking as AutoCompressor;
        # per-forward memory is bounded by the base's HF gradient-checkpointing (enabled in __init__).
        for s in range(0, W, self.seg_len):
            e = min(s + self.seg_len, W)
            state = self._write_window(state, token_embeds[:, s:e], attention_mask[:, s:e])
        return state, {}

    def _write_window(self, state, token_embeds, attention_mask):
        from ...decoder import build_prefix_cache
        B, W, d = token_embeds.shape
        K = self.K
        active = attention_mask.bool().any(dim=1)
        if not active.any():
            return state
        pool = state["pool"]                                     # [L, B, N, d]

        # ── FAITHFUL co-attention compress. The window co-attends to the pool's last-K slots (injected
        #    as a per-layer NON-CAUSAL KV prefix — each layer sees its OWN pool[l] slice), and layer l's
        #    OUTPUT at the window's trailing-K valid positions becomes layer l's new candidate. ──
        # per-layer KV prefix from pool[l][:, -K:] via that layer's own (LoRA) k/v_proj (matches finalize).
        prefixK, prefixV = [], []
        for l in range(self.L):
            attn = self.base.model.layers[l].self_attn
            p = pool[l][:, -K:, :]                               # [B, K, d]  (MemoryLLM recency window)
            prefixK.append(attn.k_proj(p).view(B, K, self.n_kv, self.head_dim).permute(0, 2, 1, 3))
            prefixV.append(attn.v_proj(p).view(B, K, self.n_kv, self.head_dim).permute(0, 2, 1, 3))

        # trailing-K VALID positions per example (right-padding aware) = the new-memory carriers.
        pos = torch.arange(W, device=token_embeds.device).expand(B, W)
        vscore = torch.where(attention_mask.bool(), pos, torch.full_like(pos, -1))
        car_idx = vscore.topk(K, dim=1).values.clamp_min(0).sort(dim=1).values     # [B, K] last-K valid
        trailing = torch.zeros(B, W, 1, device=token_embeds.device, dtype=token_embeds.dtype)
        trailing.scatter_(1, car_idx.unsqueeze(-1), 1.0)                            # [B, W, 1] cue mask
        # add the new-memory positional cue at the carriers (input-embed level; see docstring deviation).
        cue = self.new_memory_positional_emb.to(token_embeds.dtype)
        emb_in = token_embeds + trailing * cue

        # forward the window with the per-layer pool prefix; read per-layer OUTPUT hiddens.
        cache = build_prefix_cache((prefixK, prefixV))
        full_mask = torch.cat([torch.ones(B, K, device=token_embeds.device, dtype=torch.long),
                               attention_mask.long()], dim=1)                       # prefix always attended
        out = self.base.model(inputs_embeds=emb_in, attention_mask=full_mask,
                              past_key_values=cache, use_cache=True, output_hidden_states=True)
        hs = out.hidden_states                                   # tuple len L+1; hs[l+1] = layer-l output [B,W,d]
        gather_idx = car_idx[:, :, None].expand(B, K, d)
        cand = torch.stack([hs[l + 1].gather(1, gather_idx) for l in range(self.L)], dim=0)  # [L,B,K,d]
        cand = self.out_norm(cand.float()).to(pool.dtype)        # match backbone token scale (frozen base → tame norms)

        # RANDOM DROP K slots + insert candidates. ONE shared randperm across all layers (MemoryLLM
        # default drop_memory_per_layer=False → the same physical slot is dropped/kept at every depth).
        perm = torch.randperm(self.N, device=pool.device)
        survivors = pool[:, :, perm[:self.N - K], :]             # [L, B, N-K, d]
        dropped = pool[:, :, perm[self.N - K:], :]               # [L, B, K, d] the slots perm would drop
        # A window with FEWER than K valid tokens can't supply K genuine carriers: car_idx clamped the
        # surplus picks to a DUPLICATE of token 0. car_idx is sorted ascending, so the (K−V) clamped
        # duplicates sit at the FRONT → the V genuine carriers are the LAST V. For the surplus positions
        # keep the OLD (dropped) slot instead of inserting a duplicate — so only V real slots turn over.
        V = attention_mask.bool().sum(dim=1, keepdim=True)       # [B,1]
        real_car = torch.arange(K, device=pool.device).unsqueeze(0) >= (K - V)      # [B,K] True = genuine
        cand = torch.where(real_car.view(1, B, K, 1), cand, dropped)
        proposed = torch.cat([survivors, cand], dim=2)          # [L, B, N, d]
        proposed = torch.where(active.view(1, B, 1, 1), proposed, pool)             # idle-row freeze
        return {"pool": proposed}

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
