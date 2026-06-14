"""ICAE — In-Context Autoencoder (Ge et al., ICLR 2024) as a memory encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


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
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base,
            rank=cfg.icae_lora_rank,
            alpha=cfg.icae_lora_alpha,
            target_names=tuple(cfg.llama_lora_target_names),
        )
        # Per-layer gradient checkpointing on the encoder base: finalize runs ONE 1B forward
        # over the whole chunk (T+M ~8320 tokens) -> an uncheckpointed peak OOMs at
        # chunk=8192/BS>=8. Unconditional (the old grad_checkpoint_llama flag was never set).
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
        # ALWAYS activation-checkpoint the heavy 1B base forward under training
        # (auto-skips under eval no_grad). HF's per-layer ckpt is gated on
        # base.training=False here, and the trainer only checkpoints
        # streaming_write (which for ICAE is just a cat), so without this the
        # whole-passage finalize forward is un-checkpointed and OOMs at chunk
        # 8192 (dual-review HIGH). No longer gated on the dead grad_checkpoint
        # _llama flag — the previous gating made it a silent no-op.
        def _run_base(inp_, attn_):
            return self.base.model(inputs_embeds=inp_,
                                   attention_mask=attn_).last_hidden_state
        if self.training and torch.is_grad_enabled():
            import torch.utils.checkpoint as _ckpt
            h = _ckpt.checkpoint(_run_base, inp, attn, use_reentrant=False)
        else:
            h = _run_base(inp, attn)                          # [B, T+M, d]
        mem = self.norm(h[:, -self.M:, :].float())           # [B, M, d_llama]
        return mem, {}
