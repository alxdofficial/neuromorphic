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
    PlasticBaselineEncoder,
    RecurrentBaselineEncoder,
    SplatBaselineEncoder,
    V21Encoder,
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
        "v21": V21Encoder,
        "flat_baseline": FlatBaselineEncoder,
        "continuous_baseline": ContinuousBaselineEncoder,
        "memorizing_baseline": MemorizingBaselineEncoder,
        "recurrent_baseline": RecurrentBaselineEncoder,
        "plastic_baseline": PlasticBaselineEncoder,
        "splat_baseline": SplatBaselineEncoder,
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
        for w in range(n_windows):
            s = w * window_size
            e = min(s + window_size, T)
            state, _ = self.encoder.streaming_write(
                state, chunk_embeds[:, s:e, :], batch.attention_mask[:, s:e],
                chunk_offset=s,
            )
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
        """
        device = batch.context_ids.device
        B, T_ctx = batch.context_ids.shape
        T_q = batch.question_ids.shape[1]
        T_a = batch.answer_ids.shape[1]

        embed = self.decoder.llama.get_input_embeddings()

        # ---- 1. Encode context (no_grad embed lookup) ----
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)
            # Plastic variant + surprise enabled: extra frozen-Llama forward
            # over the full context to compute per-token NLL as the
            # neuromodulator signal. Off by default — the surprise pass on
            # 4096 tokens is the dominant cost (6× slowdown). Re-enable via
            # cfg.plastic_use_surprise for ablation.
            use_surprise = (
                self.variant == "plastic_baseline"
                and getattr(self.cfg, "plastic_use_surprise", False)
            )
            if use_surprise:
                ctx_logits = self.decoder.llama(
                    input_ids=batch.context_ids,
                    attention_mask=batch.context_mask.to(torch.long),
                ).logits                                                  # [B, T_ctx, V]
                log_probs = F.log_softmax(ctx_logits[:, :-1, :].float(), dim=-1)
                tgt = batch.context_ids[:, 1:]
                nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T_ctx-1]
                ctx_surprise = F.pad(nll, (1, 0), value=0.0)              # [B, T_ctx]
                ctx_surprise = ctx_surprise * batch.context_mask.to(ctx_surprise.dtype)
            else:
                ctx_surprise = None
        n_windows = (T_ctx + window_size - 1) // window_size
        state = self.encoder.init_streaming_state(B, device, ctx_embeds.dtype)
        for w in range(n_windows):
            s = w * window_size
            e = min(s + window_size, T_ctx)
            extra = {}
            if ctx_surprise is not None:
                extra["surprise"] = ctx_surprise[:, s:e]
            state, _ = self.encoder.streaming_write(
                state, ctx_embeds[:, s:e, :], batch.context_mask[:, s:e],
                chunk_offset=s, **extra,
            )
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
            M = K_retrieve

        # ---- 3. Per-row packing of [memory, real_q, real_a] + trailing pad ----
        # AVOIDS the alignment bug where right-padded questions caused
        # answer[0] to be predicted from a pad-embedding position when
        # batched examples had different question lengths. With per-row
        # packing, the last real-question token is always immediately
        # followed by the first real-answer token, regardless of batch
        # variation.
        with torch.no_grad():
            q_embeds = embed(batch.question_ids)
            a_embeds = embed(batch.answer_ids)

        q_lens = batch.question_mask.sum(dim=1)               # [B]
        a_lens = batch.answer_mask.sum(dim=1)                 # [B]
        T_total = M + T_q + T_a                                # upper bound

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
        # Per-row alignment: at prediction position M+t_q+k-1 predict answer[k]
        pred_mask = torch.zeros(B, T_total - 1, dtype=torch.bool, device=device)
        pred_targets = torch.zeros(B, T_total - 1, dtype=torch.long, device=device)

        for i in range(B):
            t_q = int(q_lens[i].item())
            t_a = int(a_lens[i].item())
            # Place memory
            if M > 0:
                full_embeds[i, :M] = memory_dec[i]
                attn_mask_full[i, :M] = True
            # Place real question
            if t_q > 0:
                full_embeds[i, M:M + t_q] = q_embeds[i, :t_q]
                attn_mask_full[i, M:M + t_q] = True
            # Place real answer
            if t_a > 0:
                full_embeds[i, M + t_q:M + t_q + t_a] = a_embeds[i, :t_a]
                attn_mask_full[i, M + t_q:M + t_q + t_a] = True
            # Per-row prediction alignment: logit at position p predicts token at p+1
            # To predict answer[k] (real token at position M+t_q+k), we use
            # the logit at position M+t_q+k-1. Loss only at content positions.
            for k in range(t_a):
                if not batch.answer_content_mask[i, k]:
                    continue
                pred_pos = M + t_q + k - 1
                if 0 <= pred_pos < T_total - 1:
                    pred_mask[i, pred_pos] = True
                    pred_targets[i, pred_pos] = batch.answer_ids[i, k]

        # ---- 4. Llama forward (causal mask is native; padding via attn_mask) ----
        # Plastic/Splat variants: install a forward_pre_hook on the chosen
        # Llama decoder layer that calls encoder.inject(hidden_states, state).
        # The hook is removed in finally so the shared Llama module is
        # unmodified across variants.
        hook_handle = None
        if self.variant == "plastic_baseline":
            state_for_hook = finalize_aux["plastic_fast_state"]
            inject_layer_idx = self.encoder.inject_layer_idx
            encoder_ref = self.encoder

            def pre_hook(module, args, kwargs):
                if not args:
                    return None
                hidden_states = args[0]
                injected = encoder_ref.inject(hidden_states, state_for_hook)
                new_args = (injected,) + args[1:]
                return new_args, kwargs

            hook_handle = (
                self.decoder.llama.model.layers[inject_layer_idx]
                .register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
        elif self.variant == "splat_baseline":
            blobs_for_hook = finalize_aux["splat_blobs"]
            inject_layer_idx = self.encoder.inject_layer_idx
            encoder_ref = self.encoder

            def pre_hook(module, args, kwargs):
                if not args:
                    return None
                hidden_states = args[0]
                injected = encoder_ref.inject(hidden_states, blobs_for_hook)
                new_args = (injected,) + args[1:]
                return new_args, kwargs

            hook_handle = (
                self.decoder.llama.model.layers[inject_layer_idx]
                .register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

        try:
            out = self.decoder.llama(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask_full.to(torch.long),
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        logits = out.logits                                    # [B, T_total, vocab]
        pred_logits_all = logits[:, :-1, :]                    # [B, T_total-1, vocab]

        # ---- 5. CE on content positions only ----
        if pred_mask.any():
            loss_recon = F.cross_entropy(
                pred_logits_all[pred_mask].float(),
                pred_targets[pred_mask],
                reduction="mean",
            )
            # Per-example CE for telemetry (per-family aggregation in trainer)
            per_example_loss = torch.zeros(B, device=device, dtype=loss_recon.dtype)
            with torch.no_grad():
                for i in range(B):
                    if pred_mask[i].any():
                        per_example_loss[i] = F.cross_entropy(
                            pred_logits_all[i, pred_mask[i]].float(),
                            pred_targets[i, pred_mask[i]],
                            reduction="mean",
                        ).detach()
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
        # already baked in by the encoder.
        splat_aux = finalize_aux.get("splat_aux", None)
        if splat_aux is not None:
            loss = loss + splat_aux.to(loss.dtype)
        # Vanilla has no trainable params in the QA loss path (Llama is frozen
        # and mask_embed isn't used without a [MASK] token in the input). Add
        # a zero-weighted mask_embed term so backward has a grad to compute;
        # mask_embed itself receives zero gradient.
        loss = loss + 0.0 * self.decoder.mask_embed.float().sum()

        # ---- 7. Diagnostics: top-1 accuracy on content positions ----
        with torch.no_grad():
            preds_full = pred_logits_all.argmax(dim=-1)        # [B, T_total-1]
            n_content_total = pred_mask.float().sum().clamp(min=1.0)
            top1_acc = ((preds_full == pred_targets) & pred_mask).float().sum() / n_content_total

        return {
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
