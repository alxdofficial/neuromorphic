"""ICAE — In-Context Autoencoder (Ge et al., ICLR 2024) as a memory encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


class ICAEBaselineEncoder(nn.Module):
    """ICAE — In-Context Autoencoder (Ge et al., ICLR 2024) as a memory encoder.

    Faithful port of ICAE's compressor to the streaming-encoder interface, with an
    RMT/AutoCompressor recurrence so the memory PERSISTS across streaming windows:
      * the encoder is a FROZEN Llama base + a trainable encoder-LoRA (q/v) +
        M learnable memory-slot embeddings;
      * per window the LoRA-Llama runs over [prev_memory ++ window ++ M slots] and the
        M slot hiddens become the new memory — the slots (causal, at the end) attend over
        BOTH the carried memory and the new tokens, so the frozen LM reconciles old vs new
        (fixed M slots forever);
      * window-0 has empty prev_memory → [window ++ slots] = single-shot ICAE, so
        published-ICAE behavior is exactly the n_windows=1 special case (baseline fidelity).

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
        # HF per-layer checkpointing on the encoder base. Now mostly vestigial: the compress
        # forward runs per-window inside streaming_write, which the trainer already
        # activation-checkpoints (grad_checkpoint_stream), and base.training=False makes HF's
        # own checkpointing inert. Kept as a harmless belt-and-suspenders for any un-wrapped path.
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
        with torch.no_grad():   # seed norm-match scale to the backbone embed norm
            self.norm.scale.data.fill_(   # (0.9 default is ~3x too quiet on SmolLM2; match hlvocab)
                base.get_input_embeddings().weight.float().norm(dim=-1).mean().item())
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
        # RMT/AutoCompressor-style persistent memory: the M compressed slots carried across
        # windows. Empty at window 0 so the first window is exactly single-shot ICAE
        # ([window ++ slots]) → published-ICAE behavior is the n_windows=1 special case.
        return {"mem": torch.zeros(batch_size, 0, d, device=device, dtype=dtype)}

    def streaming_write(self, state, token_embeds, attention_mask=None,
                        chunk_offset=0, **extra):
        """Recurrent compress (RMT/AutoCompressor idiom over ICAE's op): read fresh memory
        slots from [prev_memory ++ new_window ++ M slots]. The slots (causal, at the end)
        attend over BOTH the carried memory and the new window, so the frozen LoRA-LM decides
        how to reconcile old and new. Fixed M slots forever; window-0 (empty prev) reduces to
        single-shot ICAE. The trainer activation-checkpoints this call per window, so the heavy
        base forward stays plain here (matches AutoCompressor)."""
        del chunk_offset, extra
        prev = state["mem"]                                  # [B, M, d] (or [B, 0, d] first window)
        B, W, d = token_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        slots = self.slots.to(token_embeds.dtype).unsqueeze(0).expand(B, self.M, d)
        inp = torch.cat([prev, token_embeds, slots], dim=1)  # [B, M_prev + W + M, d]
        _ones = lambda n: torch.ones(B, n, device=attention_mask.device, dtype=torch.long)
        attn = torch.cat([_ones(prev.shape[1]), attention_mask.long(), _ones(self.M)], dim=1)
        # Inner LlamaModel → last_hidden_state (skip lm_head). Causal: the slots at the END read
        # over prev-memory + window. base.training=False → HF per-layer ckpt is inert; the trainer's
        # per-window activation-checkpoint (grad_checkpoint_stream) covers this whole forward.
        h = self.base.model(inputs_embeds=inp, attention_mask=attn).last_hidden_state
        mem = self.norm(h[:, -self.M:, :].float())           # [B, M, d_llama]
        return {"mem": mem}, {}

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        # Memory is already the compressed M slots after the last window's recurrent write
        # (streaming_write). n_windows >= 1 always, so state["mem"] is [B, M, d].
        return state["mem"], {}
