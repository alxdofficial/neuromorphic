"""Top-level ReprLearningModel — wires encoder + decoder together.

The model takes an encoder (V21Encoder, FlatBaselineEncoder, or
ContinuousBaselineEncoder) and the frozen Llama decoder, and produces
the reconstruction loss for training.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ReprConfig
from .encoder import (
    ContinuousBaselineEncoder,
    FlatBaselineEncoder,
    MemorizingBaselineEncoder,
    NullEncoder,
    RecurrentBaselineEncoder,
    V21Encoder,
)
from .decoder import FrozenLlamaDecoder


class ReprLearningModel(nn.Module):
    """Encoder + frozen Llama decoder, end-to-end.

    Args:
        cfg: ReprConfig instance
        variant: one of "v21", "flat_baseline", "continuous_baseline",
                 "memorizing_baseline", "recurrent_baseline", "vanilla_llama"
        llama_model: optional pre-loaded Llama (for sharing across models)
    """

    VARIANTS = {
        "v21": V21Encoder,
        "flat_baseline": FlatBaselineEncoder,
        "continuous_baseline": ContinuousBaselineEncoder,
        "memorizing_baseline": MemorizingBaselineEncoder,
        "recurrent_baseline": RecurrentBaselineEncoder,
        "vanilla_llama": NullEncoder,
    }

    def __init__(
        self,
        cfg: ReprConfig,
        variant: str = "v21",
        llama_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.variant = variant

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Must be one of {list(self.VARIANTS)}."
            )

        self.encoder = self.VARIANTS[variant](cfg)
        self.decoder = FrozenLlamaDecoder(cfg, llama_model=llama_model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        mask_positions: Tensor,
    ) -> dict:
        """
        Args:
            input_ids: [B, T] Llama token ids
            attention_mask: [B, T] bool — True where real (not padding)
            mask_positions: [B, T] bool — True at positions to mask + predict

        Returns:
            dict with:
                loss        : total loss (recon + aux)
                loss_recon  : reconstruction CE on masked positions
                loss_aux    : auxiliary loss (e.g., load-balance)
                memory      : [B, M, d_llama] memory tokens (for inspection)
                **aux       : any extras from the encoder
        """
        # 1. Get Llama's frozen token embeddings as encoder input
        with torch.no_grad():
            embed_layer = self.decoder.llama.get_input_embeddings()
            token_embeds = embed_layer(input_ids)             # [B, T, d_llama]

        # 2. Encoder → memory tokens
        # mask_positions is forwarded so MT-style encoders can derive
        # the retrieval query from unmasked positions. Other encoders
        # ignore it.
        memory, aux = self.encoder(
            token_embeds, attention_mask, mask_positions=mask_positions,
        )

        # 3. Decoder → loss. Forward token_embeds + attention_mask so the
        # decoder reuses the already-computed embeddings (no second lookup)
        # and applies a proper attention mask under variable-length inputs.
        _, loss_recon = self.decoder(
            input_ids, mask_positions, memory,
            attention_mask=attention_mask,
            token_embeds=token_embeds,
        )

        # 4. Combine losses
        loss_aux = aux.get(
            "load_balance_loss",
            torch.zeros((), device=loss_recon.device, dtype=loss_recon.dtype),
        )
        loss_orth = aux.get(
            "codebook_orth_loss",
            torch.zeros((), device=loss_recon.device, dtype=loss_recon.dtype),
        )
        loss_z = aux.get(
            "z_loss",
            torch.zeros((), device=loss_recon.device, dtype=loss_recon.dtype),
        )
        loss = (
            loss_recon
            + self.cfg.load_balance_coef * loss_aux
            + self.cfg.codebook_orth_coef * loss_orth
            + self.cfg.z_loss_coef * loss_z
        )

        out = {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_aux": loss_aux.detach() if isinstance(loss_aux, Tensor) else loss_aux,
            "loss_orth": loss_orth.detach() if isinstance(loss_orth, Tensor) else loss_orth,
            "loss_z": loss_z.detach() if isinstance(loss_z, Tensor) else loss_z,
            "memory_shape": tuple(memory.shape),
            # Full memory + aux preserved for metrics computation.
            # memory is kept attached (used by the caller's metrics pass before
            # optimizer.step()); aux contains detached diagnostic scalars.
            "memory": memory,
            "aux": aux,
        }
        # Forward any extra encoder outputs (non-tensor or already detached)
        for k, v in aux.items():
            if k != "load_balance_loss":
                out[k] = v
        return out

    def trainable_parameters(self):
        """Yield only the trainable parameters (excludes frozen Llama)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def compute_hsm_loss(
        self,
        chunk_1_ids: Tensor,             # [B, T1] long
        chunk_2_ids: Tensor,             # [B, T2] long
        mask_positions: Tensor,          # [B, T2] bool — True where chunk_2 is masked
        match_layer: int = -1,           # which Llama layer to match (-1 = final)
    ) -> dict:
        """Hidden-state-matching loss for v1e cross-chunk training.

        Teacher: frozen Llama on `[chunk_1, chunk_2_masked]` → hidden states
                 at chunk_2 positions. Sees chunk_1 verbatim. No grad.
        Student: frozen Llama on `[encoder(chunk_1), chunk_2_masked]` →
                 hidden states at chunk_2 positions. Has only the encoder's
                 compressed memory of chunk_1.
        Loss:    MSE between student and teacher hidden states at all
                 chunk_2 positions.

        Both teacher and student see chunk_2 with the SAME mask applied.
        The only difference between the two forwards is chunk_1 verbatim vs
        encoder(chunk_1). The encoder must compress chunk_1 so Llama's
        downstream chunk_2 processing is preserved.
        """
        B, T1 = chunk_1_ids.shape
        _, T2 = chunk_2_ids.shape

        embed = self.decoder.llama.get_input_embeddings()

        # 1. Get input embeddings for both chunks
        with torch.no_grad():
            chunk_1_embeds = embed(chunk_1_ids)             # [B, T1, d_llama]
            chunk_2_embeds_raw = embed(chunk_2_ids)         # [B, T2, d_llama]

        # 2. Replace masked positions in chunk_2 with mask_embed
        # Both teacher and student see the same masked chunk_2.
        mask_vec = self.decoder.mask_embed.to(chunk_2_embeds_raw.dtype)
        d_llama = chunk_2_embeds_raw.shape[-1]
        chunk_2_embeds = torch.where(
            mask_positions.unsqueeze(-1),
            mask_vec.view(1, 1, d_llama).expand_as(chunk_2_embeds_raw),
            chunk_2_embeds_raw,
        )

        # 3. TEACHER forward: full chunk_1 verbatim + masked chunk_2.
        # Frozen Llama, no grad. Use llama.model (the LlamaModel body) NOT
        # llama (LlamaForCausalLM), which unconditionally runs lm_head and
        # produces logits we never use — ~40% wasted compute per forward.
        with torch.no_grad():
            teacher_inputs_embeds = torch.cat([chunk_1_embeds, chunk_2_embeds], dim=1)
            teacher_out = self.decoder.llama.model(
                inputs_embeds=teacher_inputs_embeds,
                output_hidden_states=True,
                use_cache=False,
            )
            # hidden_states is a tuple: [embeddings, layer_0_out, ..., layer_N_out]
            # match_layer=-1 takes the final layer; positive int indexes from start.
            teacher_h_all = teacher_out.hidden_states[match_layer]
            teacher_h = teacher_h_all[:, T1:, :]            # [B, T2, d_llama]

        # 4. STUDENT forward: encoder(chunk_1) + masked chunk_2.
        # Gradient flows through encoder + projection. Llama base weights
        # frozen but gradient passes through.
        attention_mask = torch.ones(B, T1, dtype=torch.bool, device=chunk_1_ids.device)
        memory, aux = self.encoder(
            chunk_1_embeds, attention_mask, mask_positions=None,
        )                                                    # [B, M, d_llama]
        M = memory.shape[1]
        student_inputs_embeds = torch.cat(
            [memory.to(chunk_2_embeds.dtype), chunk_2_embeds], dim=1,
        )
        # Same: call llama.model directly to skip lm_head.
        student_out = self.decoder.llama.model(
            inputs_embeds=student_inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
        )
        student_h_all = student_out.hidden_states[match_layer]
        student_h = student_h_all[:, M:, :]                  # [B, T2, d_llama]

        # 5. Loss: MSE at every chunk_2 position, in float32 for stability.
        loss_hsm = F.mse_loss(student_h.float(), teacher_h.float())

        # 6. Combine with encoder aux losses (load_balance, codebook_orth, z_loss)
        loss_aux = aux.get(
            "load_balance_loss",
            torch.zeros((), device=loss_hsm.device, dtype=loss_hsm.dtype),
        )
        loss_orth = aux.get(
            "codebook_orth_loss",
            torch.zeros((), device=loss_hsm.device, dtype=loss_hsm.dtype),
        )
        loss_z = aux.get(
            "z_loss",
            torch.zeros((), device=loss_hsm.device, dtype=loss_hsm.dtype),
        )
        loss = (
            loss_hsm
            + self.cfg.load_balance_coef * loss_aux
            + self.cfg.codebook_orth_coef * loss_orth
            + self.cfg.z_loss_coef * loss_z
        )

        return {
            "loss": loss,
            "loss_hsm": loss_hsm.detach(),
            "loss_aux": loss_aux.detach() if isinstance(loss_aux, Tensor) else loss_aux,
            "loss_orth": loss_orth.detach() if isinstance(loss_orth, Tensor) else loss_orth,
            "loss_z": loss_z.detach() if isinstance(loss_z, Tensor) else loss_z,
            "memory": memory,
            "aux": aux,
            "memory_shape": tuple(memory.shape),
        }
