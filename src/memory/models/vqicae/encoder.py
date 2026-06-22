"""vqicae — ICAE, but each memory slot must be a DISCRETE code from a large VQ-VAE codebook.

Exactly the ICAE write (own frozen base + encoder-LoRA, M learnable slots appended to the passage,
run through the LM's own layers, read the slots' final hiddens) — except the slot vectors are NOT
free: each is quantized to its nearest entry in a large learned codebook (VQ-VAE, van den Oord 2017,
EMA variant + dead-code reinit + commitment loss). The discretized slots are projected back to
d_llama and PREPENDED. This tests the discreteness thesis directly: forcing per-slot content to be a
code (log₂K bits) instead of a free vector should push example-specific information out of the
per-cell content and into WHICH codes are chosen — or collapse the codebook (the project's prior VQ
worry). Usage canaries (perplexity, active-code count) make either outcome visible.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


class _VectorQuantizerEMA(nn.Module):
    """VQ-VAE quantizer with an EMA-updated codebook (buffer, not gradient-trained), a commitment
    loss, dead-code reinitialization, and usage canaries. Straight-through on the forward."""
    def __init__(self, K: int, d: int, decay: float = 0.99, eps: float = 1e-5,
                 beta: float = 0.25, reinit_thresh: float = 1.0):
        super().__init__()
        self.K, self.d, self.decay, self.eps, self.beta, self.reinit_thresh = K, d, decay, eps, beta, reinit_thresh
        gen = torch.Generator().manual_seed(0)
        cb = torch.randn(K, d, generator=gen) / math.sqrt(d)
        self.register_buffer("codebook", cb, persistent=True)
        self.register_buffer("cluster_size", torch.ones(K), persistent=True)
        self.register_buffer("embed_avg", cb.clone(), persistent=True)

    def forward(self, z: Tensor):
        """z [N,d] → (z_q_st [N,d] straight-through quantized, commit_loss, canaries)."""
        z = z.float()
        cb = self.codebook
        d2 = z.pow(2).sum(1, keepdim=True) - 2 * z @ cb.t() + cb.pow(2).sum(1)   # [N,K] ‖z-e‖²
        idx = d2.argmin(1)                                                       # [N] nearest code
        z_q = cb[idx]                                                            # [N,d]
        commit = self.beta * (z - z_q.detach()).pow(2).mean()                   # pull encoder→code
        z_q_st = z + (z_q - z).detach()                                         # straight-through

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.K).type_as(z)                      # [N,K]
                n = onehot.sum(0)                                               # [K] batch counts
                embed_sum = onehot.t() @ z                                      # [K,d]
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total   # laplace
                self.codebook.copy_(self.embed_avg / cs.unsqueeze(1))
                # dead-code reinit: revive codes that fell out of use → a random current-batch vector
                dead = self.cluster_size < self.reinit_thresh
                n_dead = int(dead.sum())
                if n_dead > 0 and z.shape[0] > 0:
                    pick = torch.randint(0, z.shape[0], (n_dead,), device=z.device)
                    self.codebook[dead] = z[pick]
                    self.embed_avg[dead] = z[pick]
                    self.cluster_size[dead] = 1.0

        with torch.no_grad():
            probs = F.one_hot(idx, self.K).type_as(z).mean(0)                   # [K] usage
            perplexity = (-(probs.clamp_min(1e-10).log() * probs).sum()).exp()  # effective #codes used
            active = (self.cluster_size > self.reinit_thresh).float().sum()
        return z_q_st, commit, {"vqicae_perplexity": perplexity,
                                "vqicae_active_codes": active, "vqicae_idx": idx}


class VQICAEEncoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the M (discretized) slots

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base, rank=cfg.vqicae_lora_rank, alpha=cfg.vqicae_lora_alpha,
            target_names=tuple(cfg.llama_lora_target_names))
        base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        d = cfg.d_llama
        self.M = cfg.vqicae_n_slots or cfg.n_flat_codes
        d_code = cfg.vqicae_d_code

        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0); emb_std = embed.weight.float().std().item()
        self.slots = nn.Parameter(mean_vec.view(1, d).repeat(self.M, 1) + emb_std * torch.randn(self.M, d))

        self.down = nn.Linear(d, d_code)                    # slot hidden → codebook space
        self.up = nn.Linear(d_code, d)                      # quantized code → d_llama (prepend)
        self.vq = _VectorQuantizerEMA(cfg.vqicae_codebook_size, d_code,
                                      decay=cfg.vqicae_ema_decay, beta=cfg.vqicae_commit_beta,
                                      reinit_thresh=cfg.vqicae_reinit_thresh)
        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(embed.weight.float().norm(dim=-1).mean().item())
        print(f"[vqicae] ICAE write + VQ slots: M={self.M}, encoder-LoRA r{cfg.vqicae_lora_rank} "
              f"({n_wrapped} layers), codebook K={cfg.vqicae_codebook_size}×{d_code} (EMA), "
              f"beta={cfg.vqicae_commit_beta}")

    def train(self, mode: bool = True):
        super().train(mode); self.base.train(False); return self

    def init_streaming_state(self, batch_size, device, dtype):
        d = self.cfg.d_llama
        return {"emb": torch.zeros(batch_size, 0, d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]
        B, T, d = emb.shape
        slots = self.slots.to(emb.dtype).unsqueeze(0).expand(B, self.M, d)
        inp = torch.cat([emb, slots], dim=1)
        attn = torch.cat([mask, torch.ones(B, self.M, device=mask.device, dtype=mask.dtype)], dim=1).long()

        def _run(inp_, attn_):
            return self.base.model(inputs_embeds=inp_, attention_mask=attn_).last_hidden_state
        # Only gradient-checkpoint the base forward for LONG sequences (the QA ~8k-chunk regime);
        # at the mixed ctx (~1k+M tokens) the forward fits comfortably, so run it directly (faster).
        if self.training and torch.is_grad_enabled() and inp.shape[1] > 2048:
            import torch.utils.checkpoint as _ckpt
            h = _ckpt.checkpoint(_run, inp, attn, use_reentrant=False)
        else:
            h = _run(inp, attn)
        slot_final = h[:, -self.M:]                                  # [B,M,d]

        z = self.down(slot_final).reshape(B * self.M, -1)           # [B·M, d_code]
        z_q, commit, vq_aux = self.vq(z)                            # discretize (straight-through)
        memory = self.norm(self.up(z_q).reshape(B, self.M, d).float())   # [B,M,d_llama]

        aux = {"vq_loss": commit,
               "vqicae_perplexity": vq_aux["vqicae_perplexity"].detach(),
               "vqicae_active_codes": vq_aux["vqicae_active_codes"].detach()}
        with torch.no_grad():
            # fraction of the codebook used IN THIS batch (distinct nearest codes / K)
            aux["vqicae_batch_used"] = torch.tensor(
                float(vq_aux["vqicae_idx"].unique().numel()) / self.vq.K, device=emb.device)
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
