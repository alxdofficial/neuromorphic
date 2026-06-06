"""Top-level ReprLearningModel — wires encoder + decoder together.

The model takes an encoder (V21Encoder, VQVAEBaselineEncoder, or
SlotAttentionBaselineEncoder) and the frozen Llama decoder, and produces
the reconstruction loss for training.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from .chat_template import ChatTemplate, build_chat_template
from .config import ReprConfig
from .encoder import (
    SlotAttentionBaselineEncoder,
    VQVAEBaselineEncoder,
    FullContextEncoder,
    GraphV6BaselineEncoder,
    MemorizingTransformerBaselineEncoder,
    NullEncoder,
    MambaBaselineEncoder,
)
from .decoder import FrozenLlamaDecoder
from .jepa import (
    JEPAPredictor, init_ema_target, update_ema_target,
    vicreg_variance_loss, vicreg_covariance_loss,
)


class ReprLearningModel(nn.Module):
    """Encoder + frozen Llama decoder, end-to-end.

    Args:
        cfg: ReprConfig instance
        variant: one of "v21", "flat_baseline", "continuous_baseline",
                 "memorizing_baseline", "recurrent_baseline", "vanilla_llama"
        llama_model: optional pre-loaded Llama (for sharing across models)
    """

    VARIANTS = {
        # canonical names = the published model each baseline is based on
        "vqvae_baseline": VQVAEBaselineEncoder,                  # van den Oord 2017
        "slot_attention_baseline": SlotAttentionBaselineEncoder,  # Locatello 2020
        "memorizing_transformer_baseline": MemorizingTransformerBaselineEncoder,  # Wu 2022
        "mamba_baseline": MambaBaselineEncoder,                  # Gu & Dao 2023
        "graph_v6_baseline": GraphV6BaselineEncoder,             # ours
        "vanilla_llama": NullEncoder,         # loss floor — Llama with no memory
        "vanilla_full_context": FullContextEncoder,  # loss ceiling — Llama sees full evidence
        # back-compat aliases (old variant strings in checkpoints / legacy scripts → same class)
        "flat_baseline": VQVAEBaselineEncoder,
        "continuous_baseline": SlotAttentionBaselineEncoder,
        "memorizing_baseline": MemorizingTransformerBaselineEncoder,
        "recurrent_baseline": MambaBaselineEncoder,
    }

    def __init__(
        self,
        cfg: ReprConfig,
        variant: str = "graph_v6_baseline",
        llama_model: Optional[nn.Module] = None,
        chat_template: Optional[ChatTemplate] = None,
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
        # Optional HF per-layer gradient checkpointing on the Llama base model
        # (recompute in backward, use_reentrant=False; HF gates on self.training so
        # eval is unaffected). Flag-controlled, off by default. vanilla_full_context
        # is NOT trained — it re-forwards the full ~8192-token context per question
        # (~40x the decoder tokens of the memory arms), so backward OOMs; it runs
        # frozen/eval-only instead (a frozen full-context Llama is a valid ceiling).
        if getattr(cfg, "grad_checkpoint_llama", False):
            self.decoder.llama.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        # Chat template: caller can pass a pre-built one to avoid reloading
        # the tokenizer per model instance. Otherwise build it here. When
        # the backbone has no chat template (base Llama), self.chat_template
        # stays None and compute_qa_loss falls back to legacy raw concat.
        if chat_template is not None:
            self.chat_template = chat_template
            print(f"[chat-template] (shared) {self.chat_template.summary()}")
        else:
            self.chat_template = self._maybe_build_chat_template(cfg)

    @staticmethod
    def _maybe_build_chat_template(cfg: ReprConfig) -> Optional[ChatTemplate]:
        """Build a chat template for cfg.llama_model if the tokenizer has one.

        Narrow exception handling: ImportError on transformers is the only
        thing we swallow. Tokenizer-without-chat-template returns None
        explicitly. Any OTHER failure (build error, file missing) raises —
        we don't silently fall back to raw concat when the user clearly
        chose an Instruct backbone.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            print(f"[chat-template] disabled — transformers missing: {e}")
            return None
        _tok = AutoTokenizer.from_pretrained(cfg.llama_model)
        if getattr(_tok, "chat_template", None) is None:
            print(f"[chat-template] disabled — {cfg.llama_model} has no chat template")
            return None
        ct = build_chat_template(_tok, cfg.system_intro_for_memory)
        print(f"[chat-template] {ct.summary()}")
        return ct

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
        # Variant-specific aux (graph_baseline / splat_baseline) — pre-weighted
        # by the encoder. Same pattern as compute_qa_loss; audit M1 required
        # this here too so non-QA training paths actually train these losses.
        splat_aux = aux.get("splat_aux", None)
        if splat_aux is not None:
            loss = loss + splat_aux.to(loss.dtype)
        graph_aux = aux.get("graph_aux", None)
        if graph_aux is not None:
            loss = loss + graph_aux.to(loss.dtype)

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

    def init_jepa(self, predictor_d_hidden: int = 512,
                  predictor_n_layers: int = 2, predictor_n_heads: int = 8):
        """Initialize JEPA components: predictor + EMA target encoder.

        Called once before JEPA training. Predictor maps online memory →
        predicted target memory. Target encoder is an EMA copy of self.encoder
        with no gradients (updated externally via update_jepa_target).
        """
        device = next(self.encoder.parameters()).device
        self.jepa_predictor = JEPAPredictor(
            d_in=self.cfg.d_llama,
            d_hidden=predictor_d_hidden,
            n_layers=predictor_n_layers,
            n_heads=predictor_n_heads,
        ).to(device)
        self.jepa_target_encoder = init_ema_target(self.encoder).to(device)
        self._jepa_active = True

    @torch.no_grad()
    def update_jepa_target(self, tau: float = 0.996):
        """EMA update of the JEPA target encoder. Call after each optimizer.step."""
        update_ema_target(self.encoder, self.jepa_target_encoder, tau=tau)

    def compute_jepa_loss(
        self,
        chunk_1_ids: Tensor,
        chunk_2_ids: Tensor,
        var_coef: float = 5.0,
        cov_coef: float = 0.5,
    ) -> dict:
        """JEPA cross-chunk encoding prediction loss.

        Online: encoder(chunk_1) → memory_1; predictor(memory_1) → memory_2_pred
        Target: target_encoder(chunk_2) → memory_2 [stop-grad, EMA weights]
        Loss: MSE(memory_2_pred, memory_2) + VicReg variance + covariance

        No Llama in the loop (only its embedding layer for input lookup).
        Gradients flow: loss → predictor → memory_1 → online encoder.
        Target encoder receives no gradient; updates via update_jepa_target.
        """
        if not hasattr(self, "jepa_predictor"):
            raise RuntimeError("Call model.init_jepa() before compute_jepa_loss")

        B, T1 = chunk_1_ids.shape
        _, T2 = chunk_2_ids.shape
        device = chunk_1_ids.device
        embed = self.decoder.llama.get_input_embeddings()

        # 1. Embed both chunks (no grad — Llama embed is frozen)
        with torch.no_grad():
            chunk_1_embeds = embed(chunk_1_ids)
            chunk_2_embeds = embed(chunk_2_ids)

        # 2. Online: encode chunk_1, then predict chunk_2's encoding
        am1 = torch.ones(B, T1, dtype=torch.bool, device=device)
        memory_1, aux = self.encoder(
            chunk_1_embeds, am1, mask_positions=None,
        )
        memory_2_pred = self.jepa_predictor(memory_1.float())

        # 3. Target: encode chunk_2 with EMA encoder, no grad
        with torch.no_grad():
            am2 = torch.ones(B, T2, dtype=torch.bool, device=device)
            memory_2_target, _ = self.jepa_target_encoder(
                chunk_2_embeds, am2, mask_positions=None,
            )
            memory_2_target = memory_2_target.float()

        # 4. Prediction loss (MSE)
        loss_pred = F.mse_loss(memory_2_pred, memory_2_target)

        # 5. VicReg anti-collapse — applied to BOTH online memory_1 and
        # target memory_2. Online VicReg gives the encoder gradient pressure;
        # target VicReg is a diagnostic (target has no grad, so the value
        # tells us whether the EMA-tracked encoder is collapsing on val
        # data even though no gradient pushes it directly).
        loss_var = vicreg_variance_loss(memory_1)
        loss_cov = vicreg_covariance_loss(memory_1)
        with torch.no_grad():
            target_var_diag = vicreg_variance_loss(memory_2_target)
            target_cov_diag = vicreg_covariance_loss(memory_2_target)

        # 6. Encoder aux losses (load_balance, codebook_orth, z_loss — V2.1-specific)
        loss_aux = aux.get(
            "load_balance_loss",
            torch.zeros((), device=device, dtype=loss_pred.dtype),
        )
        loss_orth = aux.get(
            "codebook_orth_loss",
            torch.zeros((), device=device, dtype=loss_pred.dtype),
        )
        loss_z = aux.get(
            "z_loss",
            torch.zeros((), device=device, dtype=loss_pred.dtype),
        )
        loss = (
            loss_pred
            + var_coef * loss_var
            + cov_coef * loss_cov
            + self.cfg.load_balance_coef * loss_aux
            + self.cfg.codebook_orth_coef * loss_orth
            + self.cfg.z_loss_coef * loss_z
        )
        # Variant-specific aux (graph / splat) — audit M1.
        splat_aux = aux.get("splat_aux", None)
        if splat_aux is not None:
            loss = loss + splat_aux.to(loss.dtype)
        graph_aux = aux.get("graph_aux", None)
        if graph_aux is not None:
            loss = loss + graph_aux.to(loss.dtype)

        return {
            "loss": loss,
            "loss_jepa": loss_pred.detach(),
            "loss_var": loss_var.detach(),
            "loss_cov": loss_cov.detach(),
            "loss_var_target": target_var_diag.detach(),
            "loss_cov_target": target_cov_diag.detach(),
            "loss_aux": loss_aux.detach() if isinstance(loss_aux, Tensor) else loss_aux,
            "loss_orth": loss_orth.detach() if isinstance(loss_orth, Tensor) else loss_orth,
            "loss_z": loss_z.detach() if isinstance(loss_z, Tensor) else loss_z,
            "memory": memory_1,
            "memory_target": memory_2_target,
            "aux": aux,
            "memory_shape": tuple(memory_1.shape),
        }

    def compute_sentence_recon_loss(
        self,
        batch,                          # SentenceChunkBatch
        window_size: int = 1024,
    ) -> dict:
        """v1g sentence-level shuffled-retrieval loss with restricted attention.

        Pipeline:
          1. Encoder ingests the full chunk via 4 × `window_size` streaming
             writes (memory state persistent across writes).
          2. For each (batch element, queried sentence), build a decoder
             input where unmasked + revealed positions carry GT embeddings
             and still-masked positions carry mask_embed.
          3. Build a custom 4D attention mask: query i attends to key j iff
             (j is a visible position, i.e., memory token OR unmasked OR
             revealed) OR (i == j, self-attention). Still-masked positions
             can therefore predict from visible context but cannot share
             info with each other.
          4. One Llama forward over [BK, M + L_max, d_llama] with that mask.
          5. CE loss on still-masked positions only.
        """
        device = batch.input_ids.device
        B, T = batch.input_ids.shape
        K = batch.query_input_ids.shape[1]
        L_max = batch.query_input_ids.shape[2]

        # ---- 1. Encode chunk via streaming writes ----
        embed = self.decoder.llama.get_input_embeddings()
        with torch.no_grad():
            chunk_embeds = embed(batch.input_ids)
        n_windows = (T + window_size - 1) // window_size
        state = self.encoder.init_streaming_state(B, device, chunk_embeds.dtype)
        # Activation-checkpoint each window during training so we hold ~one
        # window of encoder activations instead of all n_windows at once (the
        # chunk=8192 OOM for the windowed encoders). Exact gradients; recompute
        # in backward. Skipped under no_grad (eval) — nothing to save for backward.
        ckpt_stream = (getattr(self.cfg, "grad_checkpoint_stream", True)
                       and self.training and torch.is_grad_enabled())
        for w in range(n_windows):
            s = w * window_size
            e = min(s + window_size, T)
            win_emb = chunk_embeds[:, s:e, :]
            win_mask = batch.attention_mask[:, s:e]
            if ckpt_stream:
                def _write(st, em, mk, off=s):
                    new_st, _ = self.encoder.streaming_write(st, em, mk, chunk_offset=off)
                    return new_st
                state = torch.utils.checkpoint.checkpoint(
                    _write, state, win_emb, win_mask, use_reentrant=False)
            else:
                state, _ = self.encoder.streaming_write(
                    state, win_emb, win_mask, chunk_offset=s)
        memory, finalize_aux = self.encoder.finalize_memory(state)
        # memory: [B, M, d_llama] — placeholder [B, 0, d_llama] for MT and Vanilla.

        # ---- 2. Flatten (B, K) -> BK ----
        BK = B * K
        query_ids_flat = batch.query_input_ids.reshape(BK, L_max).to(device)
        mask_flat = batch.mask_positions.reshape(BK, L_max).to(device)
        reveal_flat = batch.reveal_positions.reshape(BK, L_max).to(device)
        lengths_flat = batch.query_lengths.reshape(BK).to(device)

        # ---- 2b. MT special path: retrieve per-sentence ----
        # MT's finalize_memory returns a placeholder + the KV bank in aux.
        # Each queried sentence gets its own top-K retrieval from the bank.
        mt_bank = finalize_aux.get("mt_bank")
        if mt_bank is not None:
            K_retrieve = self.cfg.n_flat_codes  # match baseline memory tok count
            mt_memory_bk, mt_aux = self.encoder.retrieve_per_sentence(
                mt_bank,
                chunk_embeds,                          # raw embeds for query (no leak)
                batch.query_starts.to(device),
                batch.query_lengths.to(device),
                batch.mask_positions.to(device),
                batch.reveal_positions.to(device),
                K=K_retrieve,
            )
            # mt_memory_bk: [BK, K_retrieve, d_llama]
            # Replace finalize_aux's mt_bank with the actual aux losses + diagnostics
            finalize_aux = {k: v for k, v in finalize_aux.items() if k != "mt_bank"}
            finalize_aux.update(mt_aux)
            M = K_retrieve
        else:
            M = memory.shape[1]
            mt_memory_bk = None

        # ---- 3. Per-position predicates ----
        pos_idx = torch.arange(L_max, device=device).unsqueeze(0)
        valid_pos = pos_idx < lengths_flat.unsqueeze(1)         # [BK, L_max]
        still_masked = mask_flat & ~reveal_flat & valid_pos     # predict here
        is_visible_sent = (~still_masked) & valid_pos           # unmasked OR revealed (non-pad)

        # ---- 4. Decoder input embeddings: GT at visible, mask_vec at still-masked ----
        with torch.no_grad():
            sent_embeds = embed(query_ids_flat)                  # [BK, L_max, d_llama]
        sent_embeds = sent_embeds.clone()
        mask_vec = self.decoder.mask_embed.to(sent_embeds.dtype)
        sent_embeds = torch.where(
            still_masked.unsqueeze(-1),
            mask_vec.view(1, 1, -1).expand_as(sent_embeds),
            sent_embeds,
        )
        # Pad positions: zero them so they contribute nothing through any
        # subsequent attention. (They're also masked out as attention keys.)
        sent_embeds = sent_embeds * valid_pos.unsqueeze(-1).to(sent_embeds.dtype)

        # ---- 5. Per-query memory tokens; prepend ----
        # Three cases:
        #   - MT: per-sentence retrieval already produced mt_memory_bk [BK, K_retr, d_llama]
        #   - Standard memory variants: replicate single [B, M, d] memory across K queries
        #   - Vanilla (M=0): skip memory entirely (reshape with -1 fails on 0 elements)
        if mt_memory_bk is not None:
            memory_bk = mt_memory_bk.to(sent_embeds.dtype)
            full_embeds = torch.cat([memory_bk, sent_embeds], dim=1)
        elif M > 0:
            d_mem = memory.shape[-1]
            memory_bk = (
                memory.unsqueeze(1).expand(B, K, M, d_mem).reshape(BK, M, d_mem)
                .to(sent_embeds.dtype)
            )
            full_embeds = torch.cat([memory_bk, sent_embeds], dim=1)
        else:
            full_embeds = sent_embeds
        T_total = full_embeds.shape[1]

        # ---- 6. Custom 4D attention mask ----
        # attend[i, j] = is_visible[j] OR (i == j). Memory tokens always visible.
        is_visible_full = torch.cat([
            torch.ones(BK, M, dtype=torch.bool, device=device),
            is_visible_sent,
        ], dim=1)                                                # [BK, T_total]
        eye = torch.eye(T_total, dtype=torch.bool, device=device)
        attn_mask_2d = is_visible_full.unsqueeze(1) | eye.unsqueeze(0)
        # Additive bias form for Llama / SDPA: 0 = attend, large negative = ignore.
        # Use Llama dtype so the cast inside SDPA doesn't drop precision unexpectedly.
        attn_dtype = full_embeds.dtype
        zero = torch.zeros((), device=device, dtype=attn_dtype)
        neg_inf = torch.finfo(attn_dtype).min
        ninf = torch.full((), neg_inf, device=device, dtype=attn_dtype)
        additive_mask = torch.where(attn_mask_2d, zero, ninf).unsqueeze(1)
        # additive_mask: [BK, 1, T_total, T_total]

        # ---- 7. Llama forward (bidirectional via custom mask) ----
        # `logits_to_keep=L_max` makes HF compute lm_head only on the last
        # L_max positions (= sentence positions, since memory is prepended).
        # Saves vocab × M = 128k × 36 unnecessary logit computations per BK item.
        out = self.decoder.llama(
            inputs_embeds=full_embeds,
            attention_mask=additive_mask,
            logits_to_keep=L_max,
        )
        sent_logits = out.logits                                # [BK, L_max, vocab]

        # ---- 8. CE loss on still-masked positions ----
        if still_masked.any():
            loss_recon = F.cross_entropy(
                sent_logits[still_masked].float(),
                query_ids_flat[still_masked],
                reduction="mean",
            )
        else:
            loss_recon = (memory.float().sum() * 0.0
                          + self.decoder.mask_embed.float().sum() * 0.0)

        # ---- 9. Aux loss aggregation ----
        loss_aux = finalize_aux.get(
            "load_balance_loss",
            torch.zeros((), device=device, dtype=loss_recon.dtype),
        )
        loss_orth = finalize_aux.get(
            "codebook_orth_loss",
            torch.zeros((), device=device, dtype=loss_recon.dtype),
        )
        loss_z = finalize_aux.get(
            "z_loss",
            torch.zeros((), device=device, dtype=loss_recon.dtype),
        )
        loss = (
            loss_recon
            + self.cfg.load_balance_coef * loss_aux
            + self.cfg.codebook_orth_coef * loss_orth
            + self.cfg.z_loss_coef * loss_z
        )
        # Variant-specific aux (graph / splat) — audit M1.
        splat_aux = finalize_aux.get("splat_aux", None)
        if splat_aux is not None:
            loss = loss + splat_aux.to(loss.dtype)
        graph_aux = finalize_aux.get("graph_aux", None)
        if graph_aux is not None:
            loss = loss + graph_aux.to(loss.dtype)

        return {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_aux": loss_aux.detach() if isinstance(loss_aux, Tensor) else loss_aux,
            "loss_orth": loss_orth.detach() if isinstance(loss_orth, Tensor) else loss_orth,
            "loss_z": loss_z.detach() if isinstance(loss_z, Tensor) else loss_z,
            "memory": memory,
            "memory_shape": tuple(memory.shape),
            "n_writes": n_windows,
            "n_still_masked": still_masked.sum().detach(),
            "n_revealed": (mask_flat & reveal_flat & valid_pos).sum().detach(),
            "n_visible_in_query": is_visible_sent.sum().detach(),
            "aux": finalize_aux,
            # Diagnostic-only (no grad needed); used by inspect_v1g.py
            "sent_logits": sent_logits.detach(),
            "still_masked": still_masked,
            "query_ids_flat": query_ids_flat,
            # MT's actual per-sentence memory [BK, K_retrieve, d_llama] when applicable
            "mt_memory_bk": mt_memory_bk,
        }

    def compute_qa_loss(
        self,
        batch,                          # QABatch from data_qa.py
        window_size: int = 1024,
        zero_memory: bool = False,
        shuffle_memory: bool = False,
        oracle_memory: bool = False,
    ) -> dict:
        """v1h composite-QA loss.

        Pipeline (matches the memory paradigm — decoder never sees raw context):
          1. Encoder ingests context via streaming writes → memory tokens.
             Original context tokens are dropped after this step.
          2. Decoder forward on `[memory, question, answer]` only.
          3. Teacher-forced CE on answer-content positions: at answer slot t
             the model uses (memory, question, GT_answer[:t]) to predict
             GT_answer[t]; loss only on the load-bearing content tokens
             (`answer_content_mask`), not on padding or filler answer tokens.

        zero_memory: diagnostic ablation. When True the decoder runs with
            NO memory contribution — prepend variants get their memory
            tensor zeroed AND those positions masked OUT of attention;
            MemInject variants (plastic, splat) skip the forward_pre_hook
            so Llama runs unmodified. Useful for measuring how much each
            architecture's memory actually contributes vs the parametric
            Llama floor (vanilla_llama).
        """
        device = batch.context_ids.device
        B, T_ctx = batch.context_ids.shape
        T_q = batch.question_ids.shape[1]
        T_a = batch.answer_ids.shape[1]

        embed = self.decoder.llama.get_input_embeddings()

        # ---- 1. Encode context (no_grad embed lookup) ----
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)
            ctx_surprise = None   # was the plastic_baseline neuromod signal (retired)
        n_windows = (T_ctx + window_size - 1) // window_size
        state = self.encoder.init_streaming_state(B, device, ctx_embeds.dtype)
        # Activation-checkpoint each window during training so we hold ~one
        # window of encoder activations instead of all n_windows at once (the
        # chunk=8192 OOM for the windowed encoders flat/continuous/MT). Exact
        # gradients; recompute in backward. Skipped under no_grad (eval) and when
        # per-window `extra` (plastic surprise) is present.
        ckpt_stream = (getattr(self.cfg, "grad_checkpoint_stream", True)
                       and self.training and torch.is_grad_enabled())
        for w in range(n_windows):
            s = w * window_size
            e = min(s + window_size, T_ctx)
            extra = {}
            if ctx_surprise is not None:
                extra["surprise"] = ctx_surprise[:, s:e]
            win_emb = ctx_embeds[:, s:e, :]
            win_mask = batch.context_mask[:, s:e]
            if ckpt_stream and not extra:
                def _write(st, em, mk, off=s):
                    new_st, _ = self.encoder.streaming_write(st, em, mk, chunk_offset=off)
                    return new_st
                state = torch.utils.checkpoint.checkpoint(
                    _write, state, win_emb, win_mask, use_reentrant=False)
            else:
                state, _ = self.encoder.streaming_write(
                    state, win_emb, win_mask, chunk_offset=s, **extra)
        # v5.6: hand the question to the encoder (graph_v5 reads it for the
        # question-conditioned readout; dict-state variants ignore the keys).
        # NullEncoder (Tensor state) and Mamba (list state) are non-dict — guard.
        if isinstance(state, dict):
            state["question_embeds"] = embed(batch.question_ids)
            state["question_mask"] = batch.question_mask
        memory, finalize_aux = self.encoder.finalize_memory(state)
        # memory: [B, M, d_llama] or [B, 0, d_llama] for MT/Vanilla
        M = memory.shape[1]

        # ---- 2. MT branch: retrieve per-chunk using question as query ----
        mt_bank = finalize_aux.get("mt_bank")
        if mt_bank is not None:
            with torch.no_grad():
                q_embeds_for_query = embed(batch.question_ids)
            K_retrieve = self.cfg.n_flat_codes
            memory, mt_aux = self.encoder.retrieve_for_query(
                mt_bank, q_embeds_for_query, batch.question_mask, K=K_retrieve,
            )
            finalize_aux = {k: v for k, v in finalize_aux.items() if k != "mt_bank"}
            finalize_aux.update(mt_aux)
            M = memory.shape[1]    # retrieve_for_query returns min(K, bank_len); a passage
                                   # shorter than n_flat_codes yields <K tokens (Stage-A short passages)

        # ORACLE ceiling (Stage-A): bypass the encoder entirely — the RAW passage
        # embeddings ARE the memory (lossless, M = T_ctx prepended tokens, padded
        # positions attended-out). Isolates the FROZEN decoder's capability to
        # locate-and-read the answer from a faithful memory from any encoder
        # write/read loss. If oracle ALSO ≈ shuffle, the frozen decoder can't route
        # content even from a perfect memory → the decoder (not the memory) is the wall.
        if oracle_memory:
            with torch.no_grad():
                memory = embed(batch.context_ids)
            finalize_aux = {"memory_mask": batch.context_mask}
            M = memory.shape[1]

        # SHUFFLE control (Stage-A load-bearing test): give each row a DIFFERENT
        # passage's memory. graph_v6 reads facts from finalize_aux (NOT `memory`),
        # so roll the right object per arm — rolling `memory` would be a silent
        # no-op for graph_v6 and falsely look like REAL.
        if shuffle_memory:
            assert memory.shape[0] >= 2, "shuffle_memory control needs batch >= 2"
            if (self.variant == "graph_v6_baseline" and not self.cfg.graph_v6_prepend_read
                    and finalize_aux.get("graph_v6_facts") is not None):
                fv = finalize_aux["graph_v6_facts"]
                fv["value"] = fv["value"].roll(1, 0)
            else:
                memory = memory.roll(1, 0)
                if finalize_aux.get("memory_mask") is not None:   # roll the per-slot mask WITH the memory
                    finalize_aux = {**finalize_aux,               # (oracle: variable passage lengths)
                                    "memory_mask": finalize_aux["memory_mask"].roll(1, 0)}

        # zero_memory ablation: for prepend variants we DROP the memory
        # slots entirely (M=0) so the question starts at RoPE position 0,
        # matching vanilla_llama's positional layout. Previously we kept
        # the M slots but attended-out: that left the question at
        # position M, biasing zero-mem reports by memory length.
        # For MemInject variants (plastic/splat) M is already 0 here;
        # zero_memory just skips hook installation below.
        if zero_memory:
            M = 0

        # ---- 3. Per-row packing of [memory, real_q, real_a] + trailing pad ----
        # AVOIDS the alignment bug where right-padded questions caused
        # answer[0] to be predicted from a pad-embedding position when
        # batched examples had different question lengths. With per-row
        # packing, the last real-question token is always immediately
        # followed by the first real-answer token, regardless of batch
        # variation.
        #
        # When self.chat_template is set (Instruct/chat-tuned backbones), we
        # additionally splice in fixed scaffold tokens around memory/Q/A and
        # append an <|eot_id|> to each answer. Layout per row:
        #
        #   [pre_mem][memory][post_mem][question][post_q][answer + eot]
        #
        # All scaffold spans are constant per template. Memory + question +
        # answer keep their existing per-row packing semantics.
        with torch.no_grad():
            q_embeds = embed(batch.question_ids)
            a_embeds = embed(batch.answer_ids)

        q_lens = batch.question_mask.sum(dim=1)               # [B]
        a_lens = batch.answer_mask.sum(dim=1)                 # [B]

        # Chat-template path: splice scaffold tokens around memory/Q/A. If
        # cfg.append_answer_eot is True (default), also append the backbone's
        # end-of-turn token after each answer's last real token so TF training
        # learns to emit it (and AR decode can then stop cleanly on it).
        ct = self.chat_template
        if ct is not None:
            append_eot = bool(getattr(self.cfg, "append_answer_eot", True))
            # Pre-embed scaffold token ids (no_grad — embedding lookup)
            with torch.no_grad():
                pre_mem_ids = ct.pre_memory_ids.to(device)
                post_mem_ids = ct.post_memory_ids.to(device)
                post_q_ids = ct.post_question_ids.to(device)
                pre_mem_embeds = embed(pre_mem_ids)            # [L_pre, d]
                post_mem_embeds = embed(post_mem_ids)          # [L_post_mem, d]
                post_q_embeds = embed(post_q_ids)              # [L_post_q, d]
            L_pre = pre_mem_embeds.shape[0]
            L_post_mem = post_mem_embeds.shape[0]
            L_post_q = post_q_embeds.shape[0]

            if append_eot:
                eot = ct.eot_id
                # Append eot to answer tensors. Extend by 1 column, place eot
                # at each row's current end (position a_lens[i]); mark mask +
                # content mask True at that position so loss is computed on it.
                B_, T_a_old = batch.answer_ids.shape
                row_idx = torch.arange(B_, device=device)
                a_lens_long = a_lens.long()
                ans_pad = int(self.cfg.pad_token_id)
                new_a_ids = torch.full((B_, T_a_old + 1), ans_pad,
                                        dtype=torch.long, device=device)
                new_a_ids[:, :T_a_old] = batch.answer_ids
                new_a_ids[row_idx, a_lens_long] = eot
                new_a_mask = torch.zeros((B_, T_a_old + 1), dtype=torch.bool, device=device)
                new_a_mask[:, :T_a_old] = batch.answer_mask
                new_a_mask[row_idx, a_lens_long] = True
                new_content_mask = torch.zeros((B_, T_a_old + 1), dtype=torch.bool, device=device)
                new_content_mask[:, :T_a_old] = batch.answer_content_mask
                new_content_mask[row_idx, a_lens_long] = True
                # Re-embed answer with appended eot, update per-row lengths.
                with torch.no_grad():
                    a_embeds = embed(new_a_ids)                     # [B, T_a+1, d]
                a_lens = new_a_mask.sum(dim=1)
                answer_ids_for_loss = new_a_ids
                answer_content_for_loss = new_content_mask
                T_a = T_a_old + 1
            else:
                # Chat scaffold spliced in, but no eot appended to answer —
                # for ablating "does EOT supervision actually help?" or for
                # apples-to-apples vs tranche-3 numbers.
                answer_ids_for_loss = batch.answer_ids
                answer_content_for_loss = batch.answer_content_mask
        else:
            L_pre = L_post_mem = L_post_q = 0
            pre_mem_embeds = post_mem_embeds = post_q_embeds = None
            answer_ids_for_loss = batch.answer_ids
            answer_content_for_loss = batch.answer_content_mask

        T_total = L_pre + M + L_post_mem + T_q + L_post_q + T_a   # upper bound

        # Determine dtype for full_embeds. memory might be empty (vanilla)
        # in which case use q_embeds dtype.
        if M > 0:
            embed_dtype = memory.dtype
            memory_dec = memory.to(q_embeds.dtype)
        else:
            embed_dtype = q_embeds.dtype
            memory_dec = None

        full_embeds = torch.zeros(B, T_total, q_embeds.shape[-1],
                                   device=device, dtype=q_embeds.dtype)
        attn_mask_full = torch.zeros(B, T_total, dtype=torch.bool, device=device)
        # Per-row alignment: at prediction position (offset+k-1) predict answer[k]
        pred_mask = torch.zeros(B, T_total - 1, dtype=torch.bool, device=device)
        pred_targets = torch.zeros(B, T_total - 1, dtype=torch.long, device=device)

        # v5.5: optional per-slot memory mask from encoder. Used by
        # vanilla_full_context to mark padded context positions as
        # not-attendable (previously Llama could attend to all M slots
        # including zero-vector pads, contaminating the ceiling reference).
        # Compressed-memory variants don't set this — every memory slot is
        # real content and all M positions should be attended.
        mem_mask_from_enc = finalize_aux.get("memory_mask")
        for i in range(B):
            t_q = int(q_lens[i].item())
            t_a = int(a_lens[i].item())
            col = 0
            # Scaffold: pre-memory header (chat-template path only; no-op when ct=None)
            if L_pre > 0:
                full_embeds[i, col:col + L_pre] = pre_mem_embeds.to(q_embeds.dtype)
                attn_mask_full[i, col:col + L_pre] = True
                col += L_pre
            # Memory (zero_memory: leave embeds as zeros AND keep attn False)
            if M > 0 and not zero_memory:
                full_embeds[i, col:col + M] = memory_dec[i]
                if mem_mask_from_enc is not None:
                    attn_mask_full[i, col:col + M] = mem_mask_from_enc[i, :M].to(torch.bool)
                else:
                    attn_mask_full[i, col:col + M] = True
            col += M
            # Scaffold: post-memory transition (<|eot|><|user_header|>)
            if L_post_mem > 0:
                full_embeds[i, col:col + L_post_mem] = post_mem_embeds.to(q_embeds.dtype)
                attn_mask_full[i, col:col + L_post_mem] = True
                col += L_post_mem
            # Real question
            if t_q > 0:
                full_embeds[i, col:col + t_q] = q_embeds[i, :t_q]
                attn_mask_full[i, col:col + t_q] = True
            col += t_q
            # Scaffold: post-question transition (<|eot|><|asst_header|>)
            if L_post_q > 0:
                full_embeds[i, col:col + L_post_q] = post_q_embeds.to(q_embeds.dtype)
                attn_mask_full[i, col:col + L_post_q] = True
                col += L_post_q
            # Real answer (with eot appended if chat-template path)
            ans_start = col
            if t_a > 0:
                full_embeds[i, col:col + t_a] = a_embeds[i, :t_a]
                attn_mask_full[i, col:col + t_a] = True
            # Per-row prediction alignment: logit at position p predicts token at p+1.
            # To predict answer[k] (real token at position ans_start+k), we use
            # the logit at position ans_start+k-1. Loss only at content positions.
            for k in range(t_a):
                if not answer_content_for_loss[i, k]:
                    continue
                pred_pos = ans_start + k - 1
                if 0 <= pred_pos < T_total - 1:
                    pred_mask[i, pred_pos] = True
                    pred_targets[i, pred_pos] = answer_ids_for_loss[i, k]

        # ---- 4. Llama forward (causal mask is native; padding via attn_mask) ----
        # graph_v6: install a forward_pre_hook on the chosen Llama decoder layer
        # that calls encoder.inject(hidden_states, facts) at every position. The
        # hook is removed in finally so the shared Llama module is unmodified
        # across variants.
        hook_handle = None
        if zero_memory:
            # No memory: Llama forward runs unmodified (no per-token read).
            pass
        elif (self.variant == "graph_v6_baseline" and not oracle_memory
                and not self.cfg.graph_v6_prepend_read):
            # Per-decode-token read (docs/graph_v6.md READ Stage B): facts built in
            # finalize_memory, soft-retrieved + fused at every position by the hook.
            facts_for_hook = finalize_aux["graph_v6_facts"]
            inject_layer_idx = self.encoder.inject_layer_idx
            encoder_ref = self.encoder

            def pre_hook(module, args, kwargs):
                if not args:
                    return None
                hidden_states = args[0]
                injected = encoder_ref.inject(hidden_states, facts_for_hook)
                new_args = (injected,) + args[1:]
                return new_args, kwargs

            hook_handle = (
                self.decoder.llama.model.layers[inject_layer_idx]
                .register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

        try:
            # Selective lm_head: run base model for hidden states, then only
            # apply lm_head to prediction positions. Profiled saving at B=12,
            # T=224: 120→102 ms (15%), 7603→6209 MiB (1.4 GiB). Avoids the
            # [B, T, vocab=128K] logits tensor entirely.
            # v5.5 perf: use_cache=False — QA is teacher-forced + single-pass,
            # there is no autoregressive decode that reuses KV. Building the
            # cache adds a noticeable per-step cost (esp. for vanilla_full_ctx
            # where T_ctx ~= 8K).
            base_out = self.decoder.llama.model(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask_full.to(torch.long),
                use_cache=False,
            )
            hidden = base_out.last_hidden_state            # [B, T_total, d_llama]
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        pred_hidden_all = hidden[:, :-1, :]                # [B, T_total-1, d_llama]

        # ---- 5. CE on content positions only ----
        if pred_mask.any():
            # Apply lm_head only at masked positions (avoids materializing
            # [B, T, vocab] for memory/question/pad positions).
            sel_hidden = pred_hidden_all[pred_mask]        # [N_pred, d_llama]
            sel_logits = self.decoder.llama.lm_head(sel_hidden)  # [N_pred, vocab]
            sel_targets = pred_targets[pred_mask]          # [N_pred]
            # v5.5 perf: compute CE ONCE with reduction='none' (a full vocab
            # scan), then derive both the scalar loss (.mean()) and the
            # per-example aggregation (scatter_add) from the same tensor.
            # Previously we did two CE passes — one for mean, one for none.
            per_token_nll_train = F.cross_entropy(
                sel_logits.float(), sel_targets, reduction="none",
            )                                              # [N_pred]
            loss_recon = per_token_nll_train.mean()
            # Per-example CE for telemetry — detached to avoid extra grad path
            per_example_loss = torch.zeros(B, device=device, dtype=loss_recon.dtype)
            with torch.no_grad():
                row_idx = pred_mask.nonzero(as_tuple=False)[:, 0]  # [N_pred]
                per_token_nll = per_token_nll_train.detach()       # [N_pred]
                row_sum = torch.zeros(B, device=device, dtype=per_token_nll.dtype)
                row_cnt = torch.zeros(B, device=device, dtype=per_token_nll.dtype)
                row_sum.scatter_add_(0, row_idx, per_token_nll)
                row_cnt.scatter_add_(0, row_idx, torch.ones_like(per_token_nll))
                per_example_loss = (row_sum / row_cnt.clamp_min(1)).to(loss_recon.dtype)
        else:
            loss_recon = (memory.float().sum() * 0.0
                          + self.decoder.mask_embed.float().sum() * 0.0)
            per_example_loss = torch.zeros(B, device=device, dtype=loss_recon.dtype)

        # ---- 6. Aux loss aggregation ----
        loss_aux = finalize_aux.get(
            "load_balance_loss",
            torch.zeros((), device=device, dtype=loss_recon.dtype),
        )
        loss_orth = finalize_aux.get(
            "codebook_orth_loss",
            torch.zeros((), device=device, dtype=loss_recon.dtype),
        )
        loss_z = finalize_aux.get(
            "z_loss",
            torch.zeros((), device=device, dtype=loss_recon.dtype),
        )
        loss = (
            loss_recon
            + self.cfg.load_balance_coef * loss_aux
            + self.cfg.codebook_orth_coef * loss_orth
            + self.cfg.z_loss_coef * loss_z
        )
        # Splat variant: pre-weighted aux total (alpha·L_pin + beta·L_prop +
        # lambda·L_adj + lambda_sat·L_sat) added directly; coefficients are
        # already baked in by the encoder. We also stash the individual
        # sublosses on `splat_telemetry` for trainer logging + audit.
        splat_aux = finalize_aux.get("splat_aux", None)
        splat_telemetry = None
        if splat_aux is not None:
            loss = loss + splat_aux.to(loss.dtype)
            splat_telemetry = {
                "splat_aux": splat_aux.detach(),
                "splat_L_pin": finalize_aux.get("splat_L_pin"),
                "splat_L_prop": finalize_aux.get("splat_L_prop"),
                "splat_L_adj": finalize_aux.get("splat_L_adj"),
                "splat_L_sat": finalize_aux.get("splat_L_sat"),
            }

        # Graph variant: v3 has no aux loss (graph_aux is zero); telemetry
        # keys are derived signals (u, age, pick strength, overwrite rate).
        # Legacy keys (L_connect, L_adjust, saliency_mean) kept as None for
        # back-compat with downstream plotting code that .get()s them.
        graph_aux = finalize_aux.get("graph_aux", None)
        graph_telemetry = None
        if graph_aux is not None:
            loss = loss + graph_aux.to(loss.dtype)
            graph_telemetry = {
                "graph_aux": graph_aux.detach(),
                "graph_endpoint_reuse": finalize_aux.get("graph_endpoint_reuse"),
                "graph_u_mean": finalize_aux.get("graph_u_mean"),
                "graph_age_mean": finalize_aux.get("graph_age_mean"),
                "graph_src_norm": finalize_aux.get("graph_src_norm"),
                # v4 telemetry: per-slot gate distribution + routing diagnostics
                "graph_pick_affinity_avg": finalize_aux.get("graph_pick_affinity_avg"),
                "graph_gate_mean_avg": finalize_aux.get("graph_gate_mean_avg"),
                "graph_frac_anchor_avg": finalize_aux.get("graph_frac_anchor_avg"),
                "graph_frac_loadbearer_avg": finalize_aux.get("graph_frac_loadbearer_avg"),
                "graph_frac_jumpedship_avg": finalize_aux.get("graph_frac_jumpedship_avg"),
                "graph_frac_selfpick_avg": finalize_aux.get("graph_frac_selfpick_avg"),
                # Per-slot specialization
                "graph_g_slot_std": finalize_aux.get("graph_g_slot_std"),
                "graph_g_slot_range": finalize_aux.get("graph_g_slot_range"),
                # Hub-and-spoke topology
                "graph_frac_hubs": finalize_aux.get("graph_frac_hubs"),
                "graph_degree_max": finalize_aux.get("graph_degree_max"),
                "graph_degree_mean": finalize_aux.get("graph_degree_mean"),
            }
            # Pass through per-window breakdown keys (graph_g_mean_w0, graph_frac_anchor_w0, etc.)
            for k, v in finalize_aux.items():
                if k.startswith("graph_g_mean_w") or k.startswith("graph_frac_"):
                    if k not in graph_telemetry:   # don't clobber the _avg keys
                        graph_telemetry[k] = v
            # v5.1 telemetry pass-through — all graph_v5_* keys from finalize_aux
            # (node/edge gates, pick affinity/entropy, soft-pointer sharpness +
            # reuse, cross-role overlap). Lifted to top-level out so the trainer
            # can log them via out.get(key) without unpacking aux dict.
            for k, v in finalize_aux.items():
                if k.startswith("graph_v5_"):
                    graph_telemetry[k] = v
                elif k.startswith("graph_v6_") and k != "graph_v6_facts":
                    graph_telemetry[k] = v
        # Vanilla has no trainable params in the QA loss path (Llama is frozen
        # and mask_embed isn't used without a [MASK] token in the input). Add
        # a zero-weighted mask_embed term so backward has a grad to compute;
        # mask_embed itself receives zero gradient.
        loss = loss + 0.0 * self.decoder.mask_embed.float().sum()

        # ---- 7. Diagnostics: top-1 accuracy on content positions ----
        # Reuse the already-computed sel_logits from the selective lm_head
        # path; no need to re-materialize [B, T, vocab] just for argmax.
        with torch.no_grad():
            n_content_total = pred_mask.float().sum().clamp(min=1.0)
            if pred_mask.any():
                sel_preds = sel_logits.argmax(dim=-1)          # [N_pred]
                top1_acc = (sel_preds == sel_targets).float().sum() / n_content_total
            else:
                top1_acc = torch.zeros((), device=device)

        out = {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_aux": loss_aux.detach() if isinstance(loss_aux, Tensor) else loss_aux,
            "loss_orth": loss_orth.detach() if isinstance(loss_orth, Tensor) else loss_orth,
            "loss_z": loss_z.detach() if isinstance(loss_z, Tensor) else loss_z,
            "memory": memory,
            "memory_shape": tuple(memory.shape),
            "n_writes": n_windows,
            "n_content_positions": int(pred_mask.sum().item()),
            "top1_acc": top1_acc.detach(),
            "per_example_loss": per_example_loss,             # [B] for per-family aggregation
            "aux": finalize_aux,
        }
        if splat_telemetry is not None:
            out.update(splat_telemetry)
        if graph_telemetry is not None:
            out.update(graph_telemetry)
        # flat_baseline codebook health → top-level so the trainer logs it to
        # jsonl (codes_active = #live codes; collapse = the flat analogue of
        # graph routing collapse).
        for _k in ("codes_active", "routing_entropy"):
            if _k in finalize_aux:
                out[_k] = finalize_aux[_k]
        return out

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
        # Variant-specific aux (graph / splat) — audit M1.
        splat_aux = aux.get("splat_aux", None)
        if splat_aux is not None:
            loss = loss + splat_aux.to(loss.dtype)
        graph_aux = aux.get("graph_aux", None)
        if graph_aux is not None:
            loss = loss + graph_aux.to(loss.dtype)

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
