"""Top-level ReprLearningModel — wires encoder + decoder together.

The model takes one of the encoder variants (see VARIANTS) and the frozen
Llama decoder, and produces the reconstruction loss for training.
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
from .models.autocompressor import AutoCompressorBaselineEncoder
from .models.beacon import BeaconBaselineEncoder
from .models.ccm import CCMBaselineEncoder
from .models.graph import GraphEncoder
from .models.hierarchical_learned_vocab import HLVocabEncoder
from .models.icae import ICAEBaselineEncoder
from .models.soft_pointer_graph import SoftPointerGraphEncoder
from .models.vanilla import FullContextEncoder, NullEncoder
from .decoder import FrozenLlamaDecoder


def _participation_ratio(x: Tensor) -> float:
    """Effective rank (participation ratio of squared singular values) of the rows
    of x — `(Σσ²)² / Σσ⁴`. Computed via the covariance-trace identity (no SVD):
    `(tr C)² / ‖C‖_F²` on the mean-centered gram `C = Xcᵀ Xc`. Cheap (a d×d gram),
    exact. PR≈1 ⇒ rank-1 collapse; PR≈d ⇒ isotropic. The collapse canary for the
    graph's write routing / node values / endpoint codes / injected read."""
    x = x.detach().float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(0, keepdim=True)
    C = xc.t() @ xc                                   # [d,d]
    tr = torch.diagonal(C).sum()
    fro2 = (C * C).sum()
    return float((tr * tr / fro2.clamp_min(1e-12)).item())


class ReprLearningModel(nn.Module):
    """Encoder + frozen Llama decoder, end-to-end.

    Args:
        cfg: ReprConfig instance
        variant: one of the keys in VARIANTS (e.g. "soft_pointer_graph_baseline",
                 "hlvocab_baseline", "icae_baseline", "vanilla_llama")
        llama_model: optional pre-loaded Llama (for sharing across models)
    """

    VARIANTS = {
        # ── ABANDONED (2026-06-15) — kept loadable for reproducing prior results,
        # NOT in the active suite. Both hit the rank-1 read/membership wall; the
        # line moved to the VQ-VAE→graph+TokenGT model. See project_mae_4k_collapse_result.
        "soft_pointer_graph_baseline": SoftPointerGraphEncoder,   # ABANDONED (was graph_v6, free-endpoint)
        "hlvocab_baseline": HLVocabEncoder,                       # ABANDONED (was graph_v9, compression-by-vocab)
        # the current line: VQ-codebook graph + TokenGT controller + inject reader
        "graph_baseline": GraphEncoder,
        "icae_baseline": ICAEBaselineEncoder,  # ICAE (ICLR'24) compressor, EMAT-retrained
        "ccm_baseline": CCMBaselineEncoder,    # CCM (ICLR'24) recurrent compressor, EMAT-retrained
        "beacon_baseline": BeaconBaselineEncoder,  # Activation Beacon (BAAI) per-layer beacon attn
        "autocompressor_baseline": AutoCompressorBaselineEncoder,  # AutoCompressors/RMT recurrent summary
        "vanilla_llama": NullEncoder,         # loss floor — Llama with no memory
        "vanilla_full_context": FullContextEncoder,  # loss ceiling — Llama sees full evidence
    }

    def __init__(
        self,
        cfg: ReprConfig,
        variant: str = "soft_pointer_graph_baseline",
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

        # soft_pointer_graph reads via the prepend path but (unlike hlvocab and the
        # ports) has no own base copy at init to size its norm-match from — calibrate
        # its prepend_norm scale to the decoder's embedding norm here so its memory
        # tokens start at the backbone's token scale, not the quiet 0.9 default.
        _ppn = getattr(self.encoder, "prepend_norm", None)
        if _ppn is not None:
            with torch.no_grad():
                _emb = self.decoder.llama.get_input_embeddings().weight.float()
                _ppn.scale.data.fill_(_emb.norm(dim=-1).mean().item())

        # hlvocab and soft_pointer_graph both read via the PREPEND path (memory
        # tokens prepended to the decode sequence) — no dedicated cross-attn
        # reader module. The v7/v8 readers (GraphCrossAttnReader / GraphV8SymReader
        # / GraphV7UnbindReader) were retired with their lineages.

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
        # stays None and compute_loss falls back to legacy raw concat.
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
        # by the encoder. Same pattern as compute_loss; audit M1 required
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

    # compressor variants whose memory is a [B, M, d] prepend (capacity-relative
    # slicing applies to these; vanillas pass through at M=0 / M=T).
    _MASKED_RECON_COMPRESSORS = ("icae_baseline", "ccm_baseline", "hlvocab_baseline",
                        "autocompressor_baseline", "beacon_baseline",
                        "soft_pointer_graph_baseline")  # slice to k too if selected (capacity-fair)

    def compute_masked_reconstruction_loss(
        self,
        batch,                          # sentence-pair batch (data_masked_reconstruction.py)
        window_size: int = None,        # unused (single-window sentences); kept for dispatch parity
        zero_memory: bool = False,
        shuffle_memory: bool = False,
        mask_ratio: float = 0.85,
    ) -> dict:
        """True MAE compression objective (docs/compression_objective.md).

        Compress the sentence pair → k-slot memory; mask ~mask_ratio of the span
        with the learned mask_embed; decode [memory ; masked_span] causally and
        predict the TRUE token at masked positions (position p from hidden p-1,
        whose causal prefix is masks/anchors + memory — NOT true tokens, so the
        teacher-forcing local-prior cheat is removed). REAL/SHUF/OFF and the
        no-memory floor share this path. Capacity-relative: memory sliced to
        batch.k_slots for the compressor variants; vanilla_llama → M=0 (floor),
        vanilla_full_context → M=T (ceiling).
        """
        # graph reads by INJECT (forward hook), not prepend — its own path.
        if self.variant == "graph_baseline":
            return self._graph_masked_reconstruction_loss(
                batch, zero_memory=zero_memory, shuffle_memory=shuffle_memory,
                mask_ratio=mask_ratio)
        device = batch.context_ids.device
        B, T = batch.context_ids.shape
        embed = self.decoder.llama.get_input_embeddings()
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)                       # [B,T,d]

        # ---- 1. encode the span → memory [B, M, d] (single window) ----
        state = self.encoder.init_streaming_state(B, device, ctx_embeds.dtype)
        state, _ = self.encoder.streaming_write(state, ctx_embeds, batch.context_mask)
        memory, mem_aux = self.encoder.finalize_memory(state)          # [B, M, d]
        memory = memory.to(ctx_embeds.dtype)

        # capacity-relative: use the first k slots (prefix/Matryoshka code)
        k = getattr(batch, "k_slots", None)
        if k is not None and self.variant in self._MASKED_RECON_COMPRESSORS:
            memory = memory[:, :int(k)]
        # memory attention mask: real for compressor slots; context_mask for the
        # full-context ceiling (its M=T memory has padded positions)
        if self.variant == "vanilla_full_context":
            mem_mask = batch.context_mask.float()
        else:
            mem_mask = torch.ones(B, memory.shape[1], device=device)

        # ---- 2. REAL / SHUF / OFF ----
        if zero_memory:
            memory = memory[:, :0]
            mem_mask = mem_mask[:, :0]
        elif shuffle_memory:
            if B == 1:   # mirror the QA path: SHUF is a no-op at B==1, fail loudly [merge #6]
                raise ValueError("shuffle_memory requires batch size > 1 (B==1 would leave REAL memory).")
            memory = torch.roll(memory, shifts=1, dims=0)
            mem_mask = torch.roll(mem_mask, shifts=1, dims=0)
        M = memory.shape[1]

        # ---- 3. masked decoder input ----
        mask_vec = self.decoder.mask_embed.to(ctx_embeds.dtype)
        rnd = torch.rand(B, T, device=device)                          # eval: seeded in run_val
        masked = (rnd < mask_ratio) & batch.context_mask
        dec_in = torch.where(masked.unsqueeze(-1), mask_vec.view(1, 1, -1), ctx_embeds)
        full = torch.cat([memory, dec_in], dim=1)                      # [B, M+T, d]
        attn = torch.cat([mem_mask, batch.context_mask.float()], dim=1).long()

        base_out = self.decoder.llama.model(
            inputs_embeds=full, attention_mask=attn, use_cache=False)
        span_hidden = base_out.last_hidden_state[:, M:]                 # [B,T,d]

        # ---- 4. CE on masked positions (predict t_p from hidden p-1) ----
        pred_hidden = span_hidden[:, :-1]                              # predicts t_1..t_{T-1}
        targets = batch.context_ids[:, 1:]
        loss_mask = masked[:, 1:] & batch.context_mask[:, 1:]
        if loss_mask.sum() == 0:                                        # degenerate batch
            loss_mask[:, 0] = batch.context_mask[:, 1]
        sel_hidden = pred_hidden[loss_mask]                            # [N,d]
        sel_targets = targets[loss_mask]                              # [N]
        logits = self.decoder.llama.lm_head(sel_hidden).float()        # [N,V]
        per_tok = F.cross_entropy(logits, sel_targets, reduction="none")
        loss_recon = per_tok.mean()

        with torch.no_grad():
            top1 = (logits.argmax(-1) == sel_targets).float().mean()
            rows = loss_mask.nonzero(as_tuple=False)[:, 0]
            per_ex = torch.zeros(B, device=device)
            cnt = torch.zeros(B, device=device)
            per_ex.scatter_add_(0, rows, per_tok.detach())
            cnt.scatter_add_(0, rows, torch.ones_like(per_tok))
            per_ex = per_ex / cnt.clamp_min(1.0)

        # keep mask_embed in-graph even when no positions select it
        loss = loss_recon + 0.0 * self.decoder.mask_embed.float().sum()
        # anti-collapse: load-balance on every competitive softmax (emitted by the
        # encoder as load_balance_loss; e.g. hlvocab routing+emit). Weighted by the
        # (previously inert) load_balance_coef — the one uniform mechanism.
        loss_aux = mem_aux.get("load_balance_loss") if mem_aux else None
        if loss_aux is not None:
            loss = loss + self.cfg.load_balance_coef * loss_aux
        out = {
            "loss": loss, "loss_recon": loss_recon.detach(),
            "top1_acc": top1, "per_example_loss": per_ex,
            "loss_aux": loss_aux.detach() if torch.is_tensor(loss_aux) else torch.zeros((), device=device),
            "n_content_positions": int(loss_mask.sum()),   # masked positions = the CE targets
            "memory_shape": (B, M),                        # (batch, code size) for the logger
            "mae_n_masked": float(loss_mask.sum()), "mae_M": float(M),
        }
        # surface encoder telemetry (hlvocab_* collapse/presence canaries) so the
        # trainer's hlvocab_ glob logs it to JSONL [fix C]. Scalars only.
        for _k, _v in (mem_aux or {}).items():
            if torch.is_tensor(_v) and _v.numel() == 1:
                out[_k] = _v.detach()
            elif isinstance(_v, (int, float)):
                out[_k] = _v
        return out

    def _graph_masked_reconstruction_loss(
        self,
        batch,
        zero_memory: bool = False,
        shuffle_memory: bool = False,
        mask_ratio: float = 0.85,
    ) -> dict:
        """MAE loss for the graph model — read by INJECT, not prepend.

        Write: tap the frozen backbone for the observation → relational PARSER →
        graph (E edges; endpoints POINTER-SELECTED from the learnable node bank +
        regressed edge states). Read: a forward hook on the mid-late decoder layer
        binds each edge op(src,dst,edge) and adds the reader's (RMS-matched, gated)
        output to the residual stream — NO prepend, so the decode sequence is just
        the masked span (M=0). REAL/OFF/SHUF: graph present / hook absent / rolled.
        See docs/graph_model.md (source of truth).
        """
        device = batch.context_ids.device
        B, T = batch.context_ids.shape
        enc = self.encoder
        embed = self.decoder.llama.get_input_embeddings()
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)                       # [B,T,d]

        # ---- 1. parse the observation into the graph ----
        state = enc.init_streaming_state(B, device, ctx_embeds.dtype)
        state, _ = enc.streaming_write(state, ctx_embeds, batch.context_mask)
        _, mem_aux = enc.finalize_memory(state)
        graph = mem_aux["graph"]

        # ---- 2. REAL / OFF / SHUF ----
        inject = True
        if zero_memory:                                                # OFF: no hook
            inject = False
        elif shuffle_memory:                                           # SHUF: roll graph along batch
            if B == 1:
                raise ValueError("shuffle_memory requires batch size > 1 (B==1 leaves REAL memory).")
            graph = {k: (torch.roll(v, shifts=1, dims=0) if torch.is_tensor(v) and v.dim() >= 1 else v)
                     for k, v in graph.items()}

        # ---- 3. masked decoder input (no prepend; M=0) ----
        mask_vec = self.decoder.mask_embed.to(ctx_embeds.dtype)
        rnd = torch.rand(B, T, device=device)                          # eval: seeded in run_val
        masked = (rnd < mask_ratio) & batch.context_mask
        dec_in = torch.where(masked.unsqueeze(-1), mask_vec.view(1, 1, -1), ctx_embeds)
        attn = batch.context_mask.long()

        # ---- 4. inject hook on the mid-late decoder layer ----
        handle = None
        cap = {}                                        # stashes the injected vector for read-eff-rank
        if inject:
            reader = enc.reader
            layer = self.decoder.llama.model.layers[enc.inject_layer]

            def _inject_hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                inj = reader(hidden, graph).to(hidden.dtype)
                cap["inj"] = inj
                hidden = hidden + inj
                if isinstance(output, tuple):
                    return (hidden,) + tuple(output[1:])
                return hidden

            handle = layer.register_forward_hook(_inject_hook)
        try:
            base_out = self.decoder.llama.model(
                inputs_embeds=dec_in, attention_mask=attn, use_cache=False)
            span_hidden = base_out.last_hidden_state                    # [B,T,d] (M=0)
        finally:
            if handle is not None:
                handle.remove()

        # ---- 5. CE on masked positions (predict t_p from hidden p-1) ----
        pred_hidden = span_hidden[:, :-1]
        targets = batch.context_ids[:, 1:]
        loss_mask = masked[:, 1:] & batch.context_mask[:, 1:]
        if loss_mask.sum() == 0:                                        # degenerate batch
            loss_mask[:, 0] = batch.context_mask[:, 1]
        sel_hidden = pred_hidden[loss_mask]
        sel_targets = targets[loss_mask]
        logits = self.decoder.llama.lm_head(sel_hidden).float()
        per_tok = F.cross_entropy(logits, sel_targets, reduction="none")
        loss_recon = per_tok.mean()

        with torch.no_grad():
            top1 = (logits.argmax(-1) == sel_targets).float().mean()
            rows = loss_mask.nonzero(as_tuple=False)[:, 0]
            per_ex = torch.zeros(B, device=device)
            cnt = torch.zeros(B, device=device)
            per_ex.scatter_add_(0, rows, per_tok.detach())
            cnt.scatter_add_(0, rows, torch.ones_like(per_tok))
            per_ex = per_ex / cnt.clamp_min(1.0)

        # No aux loss — anti-collapse is architectural (pointer-select + tags + fixed
        # bank). Holistic recon only; keep mask_embed in-graph.
        loss = loss_recon + 0.0 * self.decoder.mask_embed.float().sum()
        out = {
            "loss": loss, "loss_recon": loss_recon.detach(),
            "top1_acc": top1, "per_example_loss": per_ex,
            "loss_aux": torch.zeros((), device=device),
            "n_content_positions": int(loss_mask.sum()),
            "memory_shape": (B, enc.gcfg.n_edges),          # E edges for the logger
            "mae_n_masked": float(loss_mask.sum()), "mae_M": 0.0,
            "graph_read_gate": float(torch.tanh(enc.reader.gate).abs().mean().item()),
        }
        # ---- 6. holistic anti-collapse telemetry (every step; cheap — PR via cov) ----
        # Canaries at every stage of the pipeline (docs/graph_model.md §7):
        #   SELECTION → ptr_entropy (sharp→near-one-hot good; high→blending bad) +
        #               nodes_used (distinct nodes pointed to = reuse / coverage).
        #   VOCABULARY→ bank_effrank (node-bank collapse).
        #   RELATION  → edge_effrank (edge-state rank across edges).
        #   READ      → read_effrank (injected signal across positions; prior models
        #               collapsed this to ~1 = membership-not-binding).
        with torch.no_grad():
            d_g = enc.gcfg.d_graph
            src_ptr, dst_ptr = graph["src_ptr"], graph["dst_ptr"]       # [B,E,N]
            ent = -(src_ptr * src_ptr.clamp_min(1e-12).log()).sum(-1).mean() \
                  - (dst_ptr * dst_ptr.clamp_min(1e-12).log()).sum(-1).mean()
            sel = torch.cat([src_ptr.argmax(-1).reshape(-1), dst_ptr.argmax(-1).reshape(-1)])
            out["graph_ptr_entropy"] = float((ent / 2).item())
            out["graph_nodes_used"] = float(sel.unique().numel())
            out["graph_bank_effrank"] = _participation_ratio(enc.parser.node_bank)
            out["graph_edge_effrank"] = _participation_ratio(
                graph["edge_state"].reshape(-1, d_g))
            if "inj" in cap:                                            # REAL path only
                out["graph_read_effrank"] = _participation_ratio(
                    cap["inj"][batch.context_mask.bool()])
        return out

    def compute_loss(
        self,
        batch,                          # QABatch from data_qa.py
        window_size: int = 1024,
        zero_memory: bool = False,
        shuffle_memory: bool = False,
    ) -> dict:
        """v1h composite-QA loss.

        (Dispatches to compute_masked_reconstruction_loss when task_mode == "masked_reconstruction" so the
        trainer + run_val call sites stay unchanged.)

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
            Llama floor (vanilla_llama). This is the OFF control.

        shuffle_memory: the SHUF control. Roll the finalized memory along the
            batch dim by 1 so each example is decoded with a DIFFERENT
            example's memory (right question, wrong memory). If the decoder is
            genuinely USING memory, REAL ≪ SHUF; if REAL ≈ SHUF the memory is
            being ignored — the MQAR/binding-failure signature. The EMAT gate
            is REAL ≫ SHUF ≫ OFF. Mutually exclusive with zero_memory (zero
            wins). No-op when M == 0 (vanilla/MT-before-retrieve).
        """
        if getattr(self, "task_mode", None) == "masked_reconstruction":
            return self.compute_masked_reconstruction_loss(
                batch, zero_memory=zero_memory, shuffle_memory=shuffle_memory,
                mask_ratio=self.cfg.mae_mask_ratio)
        device = batch.context_ids.device
        B, T_ctx = batch.context_ids.shape
        T_q = batch.question_ids.shape[1]
        T_a = batch.answer_ids.shape[1]

        embed = self.decoder.llama.get_input_embeddings()

        # ---- 1. Encode context (no_grad embed lookup) ----
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)
            ctx_surprise = None   # was the plastic_baseline neuromod signal (retired)

        enc_input = ctx_embeds

        n_windows = (T_ctx + window_size - 1) // window_size
        state = self.encoder.init_streaming_state(B, device, ctx_embeds.dtype)
        # Activation-checkpoint each window during training so we hold ~one
        # window of encoder activations instead of all n_windows at once (the
        # chunk=8192 OOM for the windowed encoders flat/continuous/MT). Exact
        # gradients; recompute in backward. Skipped under no_grad (eval).
        ckpt_stream = (getattr(self.cfg, "grad_checkpoint_stream", True)
                       and self.training and torch.is_grad_enabled())
        del ctx_surprise   # retired surprise/contextualization plumbing
        for w in range(n_windows):
            s = w * window_size
            e = min(s + window_size, T_ctx)
            win_emb = enc_input[:, s:e, :]
            win_mask = batch.context_mask[:, s:e]
            if ckpt_stream:
                def _write(st, em, mk, off=s):
                    new_st, _ = self.encoder.streaming_write(st, em, mk, chunk_offset=off)
                    return new_st
                state = torch.utils.checkpoint.checkpoint(
                    _write, state, win_emb, win_mask, use_reentrant=False)
            else:
                state, _ = self.encoder.streaming_write(
                    state, win_emb, win_mask, chunk_offset=s)
        # Hand the question to the encoder (dict-state variants may read it;
        # NullEncoder (Tensor state) and Mamba (list state) are non-dict — guard).
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
            M = K_retrieve

        # ---- 2b. Faithful MT branch (retired baseline; inert — no current
        # encoder produces mt_faithful_bank). M STAYS 0 (NO prepend) — the read
        # happens inside attention at a fixed decoder layer, not by prepending.
        #   OFF  (zero_memory)               : leave datastore None → vanilla Llama.
        #   SHUF (shuffle_memory, not zero)  : roll keys/values/ctx_mask by 1 on
        #                                       dim 0 so each row reads another
        #                                       row's datastore (right Q, wrong mem).
        #   REAL (neither)                   : set the row-aligned datastore.
        mt_faithful_bank = finalize_aux.pop("mt_faithful_bank", None)
        if mt_faithful_bank is not None and not zero_memory:
            ds_keys = mt_faithful_bank["keys"]
            ds_values = mt_faithful_bank["values"]
            ds_ctx_mask = mt_faithful_bank["ctx_mask"]
            if shuffle_memory:
                if B <= 1:
                    raise ValueError(
                        "shuffle_memory (SHUF control) needs batch size > 1; "
                        "roll-by-1 at B==1 is a no-op (SHUF would equal REAL).")
                ds_keys = torch.roll(ds_keys, shifts=1, dims=0)
                ds_values = torch.roll(ds_values, shifts=1, dims=0)
                ds_ctx_mask = torch.roll(ds_ctx_mask, shifts=1, dims=0)
            self.encoder.mt_attn_wrapper.set_datastore(ds_keys, ds_values, ds_ctx_mask)

        # shuffle_memory (SHUF control): roll memory along the batch so each
        # row is decoded with another row's memory. Applied AFTER the MT
        # retrieve so the retrieved bank is mismatched too. REAL ≫ SHUF ⇒ the
        # memory is actually used; REAL ≈ SHUF ⇒ ignored. zero_memory wins.
        # Guard B>1: roll-by-1 at B==1 is identity → SHUF==REAL silently (the
        # control gives zero signal). Require batched eval for a valid SHUF.
        if shuffle_memory and not zero_memory and M > 0:
            if B > 1:
                memory = torch.roll(memory, shifts=1, dims=0)
            else:
                raise ValueError(
                    "shuffle_memory (SHUF control) needs batch size > 1; "
                    "roll-by-1 at B==1 is a no-op (SHUF would equal REAL).")

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
        # soft_pointer_graph reads via the SAME prepend path as the baselines: its
        # finalize_memory returns [B, K_edge, d_llama] memory that is prepended
        # like every other arm (no privileged per-position inject hook), so the
        # comparison isolates the write mechanism and the REAL/SHUF/OFF binding
        # gate applies to the graph too.
        hook_handle = None

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
            # MT-faithful: always clear the kNN datastore so the shared decoder
            # Llama is left in OFF/vanilla state for the next variant or call.
            if hasattr(self.encoder, "mt_attn_wrapper"):
                self.encoder.mt_attn_wrapper.clear_datastore()

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
            # soft_pointer_graph telemetry pass-through — all spg_* keys from
            # finalize_aux (node/edge gates, fact norm, read-side health probes).
            # Lifted to top-level out so the trainer can log them via out.get(key).
            for k, v in finalize_aux.items():
                if k.startswith("spg_") and k != "spg_facts":
                    graph_telemetry[k] = v
        # hlvocab telemetry: encoder finalize aux (state geometry + write-path
        # dynamics + collapse/presence canaries).
        hlvocab_telemetry = {
            k: v for k, v in finalize_aux.items() if k.startswith("hlvocab_")
        }
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
        if hlvocab_telemetry:
            out.update(hlvocab_telemetry)
        # flat_baseline codebook health → top-level so the trainer logs it to
        # jsonl (codes_active = #live codes; collapse = the flat analogue of
        # graph routing collapse).
        for _k in ("codes_active", "routing_entropy"):
            if _k in finalize_aux:
                out[_k] = finalize_aux[_k]
        return out
