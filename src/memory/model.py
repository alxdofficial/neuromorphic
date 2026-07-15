"""Top-level ReprLearningModel — wires encoder + decoder together.

The model takes one of the encoder variants (see VARIANTS) and the frozen
Llama decoder, and produces the reconstruction loss for training.
"""
from __future__ import annotations
import copy
import math
from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from .chat_template import ChatTemplate, build_chat_template
from .config import ReprConfig
from .models.autocompressor import AutoCompressorBaselineEncoder
from .models.icae import ICAEBaselineEncoder
from .models.slotgraph import SlotGraphEncoder
from .models.memoryllm import MemoryLLMBaselineEncoder
from .models.gisting import GistingBaselineEncoder
from .models.titans import TitansEncoder
from .models.h2o import H2OBaselineEncoder
from .models.vanilla import FullContextEncoder, NullEncoder
from .decoder import FrozenLlamaDecoder, disable_lora


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
        "icae_baseline": ICAEBaselineEncoder,  # ICAE (ICLR'24) compressor
        "autocompressor_baseline": AutoCompressorBaselineEncoder,  # AutoCompressors/RMT recurrent summary
        # THE slotgraph — 96 node slots, NO edge tokens; separate frozen write/read LM copies + LoRAs;
        # persistent unit relation + scalar confidence per pair on the attention VALUE path,
        # harvested from the LM's per-layer attention; prepend+bidir read shaped by confidence-scaled
        # relation state (models/slotgraph/; docs/design/slotgraph_design.md).
        "slotgraph_baseline": SlotGraphEncoder,
        # slotgraph v1 read-experiment: SAME graph encoder, but the read is a PER-LAYER-KV prefix
        # (nodes run through the LM node-block with the write's C·R value-path injection, per-layer k/v
        # captured) instead of prepend+bidir. Isolates the read-surface axis vs gisting/memoryllm.
        "slotgraph_kv_baseline": SlotGraphEncoder,
        # slotgraph Option B (faithful live read): SAME graph encoder, read = PREPEND node tokens whose
        # node↔node attention is edge-MODULATED live in the decoder's last-K self-attention (edges re-injected
        # fresh from the store, never smeared). The only read that exercises "edges modify inter-node attention"
        # in the decoder's real attention at read time. ≈ prepend cost.
        "slotgraph_liveread_baseline": SlotGraphEncoder,
        # MemoryLLM (arXiv:2402.04624): fixed per-layer latent pool + compress-then-RANDOM-DROP
        # self-update, read as per-layer KV (native per-layer-KV read) (models/memoryllm/).
        "memoryllm_baseline": MemoryLLMBaselineEncoder,
        # Gisting (arXiv:2304.08467): learnable gist tokens compress context into their per-layer
        # KV ("gist caching") — native per-layer-KV read (models/gisting/).
        "gisting_baseline": GistingBaselineEncoder,
        # Titans-inspired (arXiv:2501.00663): deep-MLP neural memory updated by a TEST-TIME gradient step
        # (learns to memorize at test time); SIMPLIFIED prepend read, not faithful MAC (models/titans/).
        "titans_baseline": TitansEncoder,
        # H2O — Heavy-Hitter Oracle training-free KV eviction (Zhang 2023, arXiv:2306.14048).
        # Keeps the M tokens with the highest cumulative attention-received score; no encoder
        # training (eval-only: only the shared decoder read-LoRA trains). Two forward passes.
        "h2o_baseline": H2OBaselineEncoder,
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
        self.variant = variant

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Must be one of {list(self.VARIANTS)}."
            )
        if variant in {"slotgraph_baseline", "slotgraph_kv_baseline",
                       "slotgraph_liveread_baseline", "h2o_baseline"}:
            cfg = copy.copy(cfg)
        if variant == "slotgraph_baseline":
            # THE slotgraph read geometry is prepend + Set-LLM bidirectional at a uniform memory position
            # (the relational read needs intra-memory attention; see docs/design/slotgraph_design.md §6).
            if getattr(cfg, "rect_prepend_mask", False):
                raise ValueError("slotgraph_baseline requires bidir_mem_attn; rect_prepend_mask is incompatible")
            cfg.bidir_mem_attn = True
            cfg.uniform_mem_pos = True
        if variant == "slotgraph_kv_baseline":
            # v1 experiment: per-layer-KV read (passive prefix) — NO prepend geometry (bidir/uniform stay
            # off; the model routes this arm through _prefix_kv_forward because reads_per_layer_kv is True).
            cfg.slotgraph_kv_read = True
        if variant == "slotgraph_liveread_baseline":
            # Option B: prepend node tokens + edge-modulated decoder attention (live_read installs the hooks
            # and forces bidir + uniform memory geometry in compute_loss).
            cfg.slotgraph_live_read = True
        if variant == "h2o_baseline":
            cfg.use_llama_lora = False
        self.cfg = cfg

        self.encoder = self.VARIANTS[variant](cfg)
        self.decoder = FrozenLlamaDecoder(cfg, llama_model=llama_model)
        if variant == "h2o_baseline":
            for p in self.decoder.parameters():
                p.requires_grad_(False)

        # soft_pointer_graph / biomem read via the prepend path but (unlike hlvocab, slotgraph and the
        # ports) have no own base copy at init to size their norm-match from — calibrate the prepend
        # norm-match scale to the decoder's embedding norm here so the memory tokens start at the
        # backbone's actual token scale (SmolLM2 ≈ 3.2), not the quiet 0.9 _NormMatch default.
        for _nm_attr in ("prepend_norm", "out_norm"):
            _nm = getattr(self.encoder, _nm_attr, None)
            if _nm is not None and hasattr(_nm, "scale"):
                with torch.no_grad():
                    _emb = self.decoder.llama.get_input_embeddings().weight.float()
                    _nm.scale.data.fill_(_emb.norm(dim=-1).mean().item())

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
        # forward() runs a PLAIN prepend via FrozenLlamaDecoder — it does NOT install the slotgraph
        # live-read edge hooks / KV prefix / bidir-uniform geometry (those live in compute_loss and
        # compute_masked_reconstruction_loss). Refuse rather than silently evaluate a different architecture
        # (audit #10). The training + eval harness uses compute_loss, so this only guards ad-hoc model(...).
        if getattr(self.encoder, "live_read", False) or getattr(self.encoder, "reads_per_layer_kv", False):
            raise NotImplementedError(
                "ReprLearningModel.forward() does not implement the slotgraph live-read / per-layer-KV read "
                "(it runs a plain prepend). Use compute_loss / compute_masked_reconstruction_loss for these "
                "variants.")

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
    _MASKED_RECON_COMPRESSORS = ("icae_baseline",
                        "autocompressor_baseline",
                        "slotgraph_baseline", "slotgraph_kv_baseline", "slotgraph_liveread_baseline",
                        "memoryllm_baseline", "gisting_baseline", "titans_baseline",
                        "h2o_baseline")

    def _chunked_lm_head_ce(self, sel_hidden, sel_targets, chunk: int = 2048):
        """fp32 ``lm_head`` + per-token CE over the selected positions, computed in
        position-chunks with backward recompute (``checkpoint``) so the ``[chunk, V]``
        fp32 logits never materialize for the whole selection at once — in EITHER the
        forward (allocation) or the backward (softmax-grad twin). Returns
        ``(per_tok [N] with grad, top1 scalar)``, math-identical to a dense
        ``lm_head → cross_entropy``; only the peak memory differs.

        Motivation: the reconstruct (MAE) task predicts the full ~2048-token passage, so
        at B=8/ctx=2048 it selects N≈14k positions and the dense ``[N, 49152]`` fp32
        logits alone are ~2.5GB — which tips a 24GB pod over at this line. Chunking to
        2048 positions caps the CE working set at ~0.4GB regardless of N."""
        lm_head = self.decoder.llama.lm_head
        N = sel_hidden.shape[0]
        if N <= chunk:
            logits = lm_head(sel_hidden).float()                       # [N,V]
            per_tok = F.cross_entropy(logits, sel_targets, reduction="none")
            with torch.no_grad():
                top1 = (logits.argmax(-1) == sel_targets).float().mean()
            return per_tok, top1

        def _chunk_ce(h, t):
            lg = lm_head(h).float()                                    # [chunk,V] (recomputed in backward)
            return F.cross_entropy(lg, t, reduction="none"), (lg.argmax(-1) == t)

        pt_parts, correct_parts = [], []
        for h, t in zip(sel_hidden.split(chunk), sel_targets.split(chunk)):
            pt, correct = torch.utils.checkpoint.checkpoint(_chunk_ce, h, t, use_reentrant=False)
            pt_parts.append(pt)
            correct_parts.append(correct)
        per_tok = torch.cat(pt_parts)                                  # [N] with grad
        top1 = torch.cat(correct_parts).float().mean()
        return per_tok, top1

    def compute_masked_reconstruction_loss(
        self,
        batch,                          # sentence-pair batch (data_masked_reconstruction.py)
        window_size: int = None,        # unused (single-window sentences); kept for dispatch parity
        zero_memory: bool = False,
        shuffle_memory: bool = False,
        mask_ratio: float = 0.85,
        memory_override=None,           # (memory, mem_aux): SKIP the encoder, decode with this memory —
                                        # the objective-mode rolled reads (1 encoder run, B decoder reads)
        shuffle_roll: int = 1,          # SHUF roll amount (r>0 pairs example i with memory i−r; the
                                        # in-batch InfoNCE sweeps r=1..B−1 to score ALL negatives)
        return_memory: bool = False,    # stash the (pre-roll) memory+aux on the out dict (objective modes)
        encoder_only: bool = False,     # build memory and return WITHOUT decoding (GradCache cut point)
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
        device = batch.context_ids.device
        B, T = batch.context_ids.shape
        embed = self.decoder.llama.get_input_embeddings()
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)                       # [B,T,d]

        # ---- 1. encode the span → memory [B, M, d] (single window) ----
        # biomem/arrivalmem ingest the frozen LM's FINAL hidden (not raw embeds); others raw.
        if memory_override is not None:
            memory, mem_aux = memory_override
            mem_aux = dict(mem_aux)     # SHUF below rolls memory_mask — never mutate the caller's aux
        else:
            enc_in = self._encode_for_memory(ctx_embeds, batch.context_mask)
            surprise = (self._token_surprise(enc_in, getattr(batch, "context_ids", None), batch.context_mask)
                        if getattr(self.encoder, "wants_surprise", False) else None)
            state = self.encoder.init_streaming_state(B, device, ctx_embeds.dtype)
            # MAE is INTENTIONALLY a single-shot write over the whole span (NOT the windowed cadence the
            # generic continuation/qa/bio path uses): it is the pure compression-FIDELITY reference (the
            # floor/ceiling band), so it measures storage capacity, not streaming write. The streaming-
            # write path is exercised by the other 4 tasks. Stream MAE too only if you want a windowed
            # fidelity variant.
            state, _ = self.encoder.streaming_write(state, enc_in, batch.context_mask, surprise=surprise)
            memory, mem_aux = self.encoder.finalize_memory(state)      # [B, M, d]
        memory = memory.to(ctx_embeds.dtype)

        # capacity-relative: use the first k slots (prefix/Matryoshka code). GATED on memory already being
        # the budget size — slotgraph3 emits an EXPANDED read (K·read_topk=128 edge tokens, a re-representation
        # of its matched 32-latent state), and a naive memory[:, :32] there keeps only nodes 0-3 (node-major
        # reshape) → guts the MAE read. The Matryoshka slice only applies when M == k (a genuine prefix code).
        k = getattr(batch, "k_slots", None)
        if k is not None and self.variant in self._MASKED_RECON_COMPRESSORS and memory.shape[1] == int(k):
            memory = memory[:, :int(k)]
        # memory attention mask: real for compressor slots; context_mask for the
        # full-context ceiling (its M=T memory has padded positions); a per-slot mask
        # from the encoder (e.g. Beacon padding-beacons) when one is supplied.
        if self.variant == "vanilla_full_context":
            mem_mask = batch.context_mask.float()
        elif mem_aux.get("memory_mask") is not None:
            mem_mask = mem_aux["memory_mask"][:, :memory.shape[1]].float()   # k-sliced to match
        else:
            mem_mask = torch.ones(B, memory.shape[1], device=device)

        # GradCache cut point / objective-mode stash: the pre-roll REAL memory
        if encoder_only or return_memory:
            _mem_ret = (memory, dict(mem_aux))
        if encoder_only:
            z = torch.zeros((), device=device)
            return {"loss": z, "loss_recon": z, "top1_acc": z, "memory_shape": (B, memory.shape[1]),
                    "_memory": _mem_ret[0], "_mem_aux": _mem_ret[1]}

        # ---- 2. REAL / SHUF / OFF ----
        if zero_memory:
            memory = memory[:, :0]
            mem_mask = mem_mask[:, :0]
        elif shuffle_memory:
            if B == 1:   # mirror the QA path: SHUF is a no-op at B==1, fail loudly [merge #6]
                raise ValueError("shuffle_memory requires batch size > 1 (B==1 would leave REAL memory).")
            memory = torch.roll(memory, shifts=int(shuffle_roll), dims=0)
            mem_mask = torch.roll(mem_mask, shifts=int(shuffle_roll), dims=0)
        M = memory.shape[1]

        # ---- 3. masked decoder input ----
        mask_vec = self.decoder.mask_embed.to(ctx_embeds.dtype)
        rnd = torch.rand(B, T, device=device)                          # eval: seeded in run_val
        masked = (rnd < mask_ratio) & batch.context_mask
        dec_in = torch.where(masked.unsqueeze(-1), mask_vec.view(1, 1, -1), ctx_embeds)
        if getattr(self.encoder, "reads_per_layer_kv", False) and not zero_memory and mem_aux.get("past_kv") is not None:
            # per-layer-KV native read (Beacon/MemoryLLM): memory[:, :M] is empty; inject the
            # encoder's per-layer (K,V) as a prefix cache. dec_in stays memory-free (text-only out).
            span_hidden = self._prefix_kv_forward(
                dec_in, batch.context_mask, mem_aux["past_kv"], mem_aux.get("memory_mask"),
                shuffle_memory=shuffle_memory, shuffle_roll=shuffle_roll)
        else:
            full = torch.cat([memory, dec_in], dim=1)                  # [B, M+T, d]
            attn = torch.cat([mem_mask, batch.context_mask.float()], dim=1).long()
            # live read (Option B) forces bidir memory + uniform node positions, like slotgraph_baseline.
            _live = getattr(self.encoder, "live_read", False) and M > 0
            _bidir = getattr(self.cfg, "bidir_mem_attn", False) or _live
            if getattr(self.cfg, "rect_prepend_mask", False) and _bidir:
                raise ValueError("rect_prepend_mask and bidir_mem_attn are mutually exclusive read geometries")
            if getattr(self.cfg, "rect_prepend_mask", False) and M > 0:
                attn = self._rect_prepend_mask(attn, M, full.dtype)    # [B,1,L,L] 4-D additive
            elif _bidir and M > 0:
                attn = self._bidir_prepend_mask(attn, M, full.dtype)   # Set-LLM: bidirectional memory block
            pos_ids = (self._uniform_mem_position_ids(B, M, full.shape[1], device)
                       if (getattr(self.cfg, "uniform_mem_pos", False) or _live) and M > 0 else None)

            # biomem: query-conditioned READ — a zero-init pre-hook at the tap layer reads
            # every position's hidden through the frozen written edges and fuses the recall
            # back. REAL = read; OFF (zero_memory) = no read; SHUF = read wrong-example edges.
            hook_handle = self._install_conditioned_read_hook(zero_memory, shuffle_memory, B)
            # per-layer prepend re-injection hook (legacy): a no-op unless an encoder sets
            # reinforce_prepend_each_layer + emits 'prepend_struct'. The current slotgraph does NEITHER
            # (its structure lives entirely in the post-LM message-passing read), so this returns [].
            reinforce = self._install_prepend_reinforce_hooks(mem_aux, M, 0, shuffle_memory)
            # biomem-prepend: re-read W at every layer with the current slot hiddens (no-op for others).
            refresh = self._install_prepend_refresh_hooks(M, 0, zero_memory, shuffle_memory)
            # slotgraph Option B: edge-modulate the decoder's last-K node↔node attention (OFF drops it).
            live_inject = (self._install_live_inject_hooks(mem_aux, M, 0, shuffle_memory, shuffle_roll)
                           if not zero_memory else [])
            try:
                base_out = self.decoder.llama.model(
                    inputs_embeds=full, attention_mask=attn, position_ids=pos_ids, use_cache=False)
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
                for hh in live_inject:
                    hh.remove()
                for hh in reinforce:
                    hh.remove()
                for hh in refresh:
                    hh.remove()
            span_hidden = base_out.last_hidden_state[:, M:]             # [B,T,d]

        # ---- 4. CE on masked positions (predict t_p from hidden p-1) ----
        pred_hidden = span_hidden[:, :-1]                              # predicts t_1..t_{T-1}
        targets = batch.context_ids[:, 1:]
        loss_mask = masked[:, 1:] & batch.context_mask[:, 1:]
        if loss_mask.sum() == 0:                                        # degenerate batch
            loss_mask[:, 0] = batch.context_mask[:, 1]
        sel_hidden = pred_hidden[loss_mask]                            # [N,d]
        sel_targets = targets[loss_mask]                              # [N]
        # chunked + backward-recomputed lm_head→CE (peak ~one chunk of [chunk,V], not the
        # full ~14k-position reconstruct span) so B=8/ctx=2048 fits 24GB pods; math-identical.
        per_tok, top1 = self._chunked_lm_head_ce(sel_hidden, sel_targets)
        loss_recon = per_tok.mean()
        # differentiable per-example CE (mean per-token NLL per row) — the objective modes' InfoNCE
        # logits are built from these across memory rolls; the detached copy keeps telemetry unchanged.
        rows = loss_mask.nonzero(as_tuple=False)[:, 0]
        pe_sum = torch.zeros(B, device=device, dtype=per_tok.dtype)
        pe_cnt = torch.zeros(B, device=device, dtype=per_tok.dtype)
        pe_sum = pe_sum.scatter_add(0, rows, per_tok)
        pe_cnt = pe_cnt.scatter_add(0, rows, torch.ones_like(per_tok))
        loss_per_example = pe_sum / pe_cnt.clamp_min(1.0)              # [B], WITH grad

        with torch.no_grad():
            per_ex = loss_per_example.detach()

        # keep mask_embed in-graph even when no positions select it
        loss = loss_recon + 0.0 * self.decoder.mask_embed.float().sum()
        # anti-collapse: load-balance on every competitive softmax (emitted by the
        # encoder as load_balance_loss; e.g. hlvocab routing+emit). Weighted by the
        # (previously inert) load_balance_coef — the one uniform mechanism.
        loss_aux = mem_aux.get("load_balance_loss") if mem_aux else None
        if loss_aux is not None:
            loss = loss + self.cfg.load_balance_coef * loss_aux
        _vq = mem_aux.get("vq_loss") if mem_aux else None      # vqicae commitment (pre-weighted by beta)
        if _vq is not None:
            loss = loss + _vq.to(loss.dtype)
        out = {
            "loss": loss, "loss_recon": loss_recon.detach(),
            "top1_acc": top1, "per_example_loss": per_ex,
            "loss_per_example": loss_per_example,          # [B] WITH grad (objective-mode InfoNCE logits)
            "loss_aux": loss_aux.detach() if torch.is_tensor(loss_aux) else torch.zeros((), device=device),
            "n_content_positions": int(loss_mask.sum()),   # masked positions = the CE targets
            "memory_shape": (B, M),                        # (batch, code size) for the logger
            "mae_n_masked": float(loss_mask.sum()), "mae_M": float(M),
        }
        if return_memory:
            out["_memory"], out["_mem_aux"] = _mem_ret
        # surface encoder telemetry (hlvocab_* collapse/presence canaries) so the
        # trainer's hlvocab_ glob logs it to JSONL [fix C]. Scalars only.
        for _k, _v in (mem_aux or {}).items():
            if torch.is_tensor(_v) and _v.numel() == 1:
                out[_k] = _v.detach()
            elif isinstance(_v, (int, float)):
                out[_k] = _v
        if self.variant == "graph_baseline" and mem_aux.get("graph") is not None:
            out.update(self._graph_canaries(mem_aux["graph"], memory))
        return out

    def _graph_canaries(self, graph, memory) -> dict:
        """Anti-collapse canaries for the graph model — shared by the MAE and generic
        (QA/conditioned/continuation) loss paths (docs/graph_model.md §7). SELECTION →
        ptr_entropy + nodes_used; VOCABULARY → bank_effrank; RELATION → edge_effrank;
        READ → mem_effrank, the rank of the PREPENDED memory tokens across edges×batch
        (the old inject read collapsed this to ≈1; prepend should restore it)."""
        enc = self.encoder
        out = {}
        with torch.no_grad():
            d_g = enc.gcfg.d_graph
            if "src_ptr" in graph:                                      # discrete bank (no ptr for free endpoints)
                src_ptr, dst_ptr = graph["src_ptr"], graph["dst_ptr"]   # [B,E,N]
                ent = -(src_ptr * src_ptr.clamp_min(1e-12).log()).sum(-1).mean() \
                      - (dst_ptr * dst_ptr.clamp_min(1e-12).log()).sum(-1).mean()
                sel = torch.cat([src_ptr.argmax(-1).reshape(-1), dst_ptr.argmax(-1).reshape(-1)])
                out["graph_ptr_entropy"] = float((ent / 2).item())
                out["graph_nodes_used"] = float(sel.unique().numel())
                out["graph_bank_effrank"] = _participation_ratio(enc.parser.node_bank)
            out["graph_edge_effrank"] = _participation_ratio(
                graph["edge_state"].reshape(-1, d_g))
            # WRITE-SIDE collapse canaries (docs/graph_neighborhood_read.md): the parser
            # can collapse INDEPENDENTLY of the read — all E edges → the same vector, and/or
            # the output ignoring the observation. Watch these live (project_graph_write_collapse).
            es = graph["edge_state"].float()                        # [B,E,d]
            if es.shape[1] > 1:                                     # within-sample edge SIMILARITY (→1 = all edges identical)
                esn = torch.nn.functional.normalize(es, dim=-1)
                cos = esn @ esn.transpose(-1, -2)                  # [B,E,E]
                E_ = es.shape[1]
                off = cos.sum(dim=(-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(-1)
                out["graph_edge_cos"] = float((off / (E_ * (E_ - 1))).mean())
            if es.shape[0] > 1:                                     # eff-rank across INPUTS (→1 = parser ignores the obs)
                out["graph_input_sens"] = _participation_ratio(es.mean(dim=1))
            if memory is not None and memory.shape[1] > 0:
                out["graph_mem_effrank"] = _participation_ratio(
                    memory.reshape(-1, memory.shape[-1]))
        return out

    def _rect_prepend_mask(self, attn2d, M, dtype):
        """KBLaM-style rectangular decoder mask for a memory PREPEND at positions [0:M).
        Memory tokens attend only to THEMSELVES (diagonal — keeps SDPA rows finite; stops
        memory↔memory mixing/blurring through the decoder's layers); all later tokens attend
        causally to every valid position (text→memory retrieval unchanged). attn2d: [B,L]
        validity (long/bool). Returns a [B,1,L,L] float additive mask (0 / -inf)."""
        B, L = attn2d.shape
        valid = attn2d.bool()
        causal = torch.tril(torch.ones(L, L, device=attn2d.device, dtype=torch.bool))
        allow = causal.unsqueeze(0) & valid.unsqueeze(1)                 # [B,L,L]
        if M > 0:
            allow[:, :M, :] = False                                      # memory rows: self only
            eye = torch.eye(L, device=attn2d.device, dtype=torch.bool)
            allow[:, :M, :] |= eye[:M, :].unsqueeze(0)
        m4 = torch.zeros(B, 1, L, L, device=attn2d.device, dtype=dtype)
        m4.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)
        return m4

    def _bidir_prepend_mask(self, attn2d, M, dtype):
        """Set-LLM decoder mask for a memory PREPEND at [0:M): the memory block attends to itself
        BIDIRECTIONALLY (an edge token composes with BOTH its endpoint node tokens regardless of
        emission order — plain causal lets memory token i see only j<i, imposing a spurious
        order-through-visibility on an unordered set, the one geometry the set-invariance
        literature uniformly warns against); text tokens stay causal (they already see all memory —
        earlier positions). attn2d: [B,L] validity (long/bool). Returns [B,1,L,L] additive mask."""
        B, L = attn2d.shape
        valid = attn2d.bool()
        allow2d = torch.tril(torch.ones(L, L, device=attn2d.device, dtype=torch.bool))
        if M > 0:
            allow2d = allow2d.clone()
            allow2d[:M, :M] = True                                       # memory block: full bidirectional
        allow = allow2d.unsqueeze(0) & valid.unsqueeze(1)                # [B,L,L]
        m4 = torch.zeros(B, 1, L, L, device=attn2d.device, dtype=dtype)
        m4.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)
        return m4

    def _uniform_mem_position_ids(self, B, M, L, device):
        """RoPE position ids for a prepend of M memory tokens + (L-M) text tokens: memory all at
        position 0 (an unordered SET, mutually equidistant from text — no intra-memory order, no
        differential distance bias), text at 1..L-M (normal relative structure preserved). [B,L] long."""
        rest = torch.arange(1, L - M + 1, device=device)
        pos = torch.cat([torch.zeros(M, device=device, dtype=torch.long), rest], dim=0)
        return pos.unsqueeze(0).expand(B, -1)

    def _prefix_kv_forward(self, inputs_embeds, base_mask, past_kv, memory_mask,
                           *, shuffle_memory=False, shuffle_roll=1):
        """Per-layer-KV READ (Beacon / MemoryLLM native): inject the encoder's per-layer (K, V)
        as a NON-CAUSAL DynamicCache prefix and forward the frozen decoder over ``inputs_embeds``
        alone. Returns TEXT-ONLY hidden [B, T, d] — the memory keys are attended-to only (no
        residual, no ``[:, M:]`` slice needed). SHUF rolls the KV + mask along the batch in lockstep
        (the caller's M==0 packing means the prepend-path SHUF at compute_loss L890 is skipped for
        these arms, so the roll happens here). OFF is handled by the caller (never calls this)."""
        from .decoder import build_prefix_cache
        B = inputs_embeds.shape[0]
        K, V = past_kv                                        # each: L × [B, n_kv_heads, M, head_dim]
        mm = memory_mask
        if shuffle_memory:
            if B == 1:
                raise ValueError("shuffle_memory (SHUF) requires batch size > 1.")
            K = [torch.roll(k, shifts=int(shuffle_roll), dims=0) for k in K]
            V = [torch.roll(v, shifts=int(shuffle_roll), dims=0) for v in V]
            if mm is not None:
                mm = torch.roll(mm, shifts=int(shuffle_roll), dims=0)
        Mmem = K[0].shape[2]
        if mm is None:
            mm = torch.ones(B, Mmem, device=inputs_embeds.device)
        # width Mmem + T: the mask keeps M memory columns even though inputs_embeds does not,
        # because the DynamicCache prefix adds Mmem keys. cache_position defaults to arange(T)+Mmem.
        attn = torch.cat([mm[:, :Mmem].to(base_mask.dtype), base_mask.to(base_mask.dtype)], dim=1).long()
        L = len(K)

        def _run(emb, at, *kv):
            # rebuild the DynamicCache inside so a checkpoint recompute re-injects the prefix
            cache = build_prefix_cache((list(kv[:L]), list(kv[L:])))
            o = self.decoder.llama.model(inputs_embeds=emb, attention_mask=at,
                                         past_key_values=cache, use_cache=True)
            return o.last_hidden_state

        # Activation-checkpoint the decode (recompute in backward): the reconstruct task decodes the
        # whole ~2048-token passage with a 30-layer KV prefix and retains it for backward → the
        # per-layer-KV arms (gisting/memoryllm) OOM at B=8 on 24GB. This path installs NO forward
        # hooks, so the recompute is exact (math-identical). K/V carry grad, so checkpoint is active.
        if self.training and getattr(self.cfg, "grad_checkpoint_decode", True):
            return torch.utils.checkpoint.checkpoint(
                _run, inputs_embeds, attn, *K, *V, use_reentrant=False)
        return _run(inputs_embeds, attn, *K, *V)

    def _install_conditioned_read_hook(self, zero_memory: bool, shuffle_memory: bool, B=None):
        """Query-conditioned READ (biomem): register a pre-hook at the encoder's
        tap layer that reads every position's hidden state through the written memory and
        fuses the recall back into the residual stream. The read is the SOLE recall path
        for these arms (finalize_memory returns an empty prepend). Returns a hook handle
        (caller removes in finally), or None for non-conditioned-read arms / the OFF gate.
        REAL = read; OFF (zero_memory) = no read; SHUF = read wrong-example memory state."""
        enc = self.encoder
        if not getattr(enc, "is_conditioned_read", False) or zero_memory:
            return None
        if not enc.has_read_state():
            return None
        if shuffle_memory:                                   # SHUF: roll the memory across the batch
            if (B if B is not None else 2) == 1:
                raise ValueError("shuffle_memory requires batch size > 1.")
            enc.roll_read_state()
        layers = self.decoder.llama.model.layers
        L = min(enc.read_tap_layer + 1, len(layers) - 1)     # fuse into the input of layer L
        def _hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            h = h + enc.conditioned_read(h)                  # gated read → gentle at step 0
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = h
            return args, kwargs
        return layers[L].register_forward_pre_hook(_hook, with_kwargs=True)

    def _install_prepend_reinforce_hooks(self, mem_aux, M: int, offset: int, shuffle_memory: bool):
        """slotgraph: re-inject the structural embedding into the M prepend positions BEFORE every
        frozen-LM layer (the scaffold can't wash out through depth). The struct rides in
        mem_aux['prepend_struct'] [B,M,d_llama]; gradient flows back through it to the structure
        heads, so this is also a training signal. Returns hook handles (caller removes in finally).
        Assumes the memory occupies positions [offset:offset+M] (true for the no-chat-template
        prepend layout used by the active runs). No-op for non-slotgraph arms / M==0."""
        enc = self.encoder
        if not getattr(enc, "reinforce_prepend_each_layer", False) or M <= 0:
            return []
        struct = mem_aux.get("prepend_struct") if mem_aux else None
        if struct is None:
            return []
        struct = struct[:, :M]
        if shuffle_memory and struct.shape[0] > 1:        # match the rolled memory (SHUF control)
            struct = torch.roll(struct, shifts=1, dims=0)

        def _mk():
            def _hook(module, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                if h is None or h.shape[1] < offset + M:
                    return None
                new = h.clone()
                new[:, offset:offset + M] = new[:, offset:offset + M] + struct.to(h.dtype)
                if args:
                    return (new,) + tuple(args[1:]), kwargs
                kw = dict(kwargs); kw["hidden_states"] = new
                return args, kw
            return _hook
        return [layer.register_forward_pre_hook(_mk(), with_kwargs=True)
                for layer in self.decoder.llama.model.layers]

    def _install_live_inject_hooks(self, mem_aux, M: int, offset: int, shuffle_memory: bool,
                                   shuffle_roll: int = 1):
        """OPTION B (faithful live read): on the DECODER's late self-attn layers, edge-MODULATE the node↔
        node attention over the M prepend positions [offset:offset+M] — the write's value-path injection
        reused at read (out_i += U·Σ_j a_nn_ij·C·R). The edge state R/C is read FRESH from mem_aux each layer,
        so the relational structure is never smeared through depth. SHUF rolls the edge state by the SAME
        shuffle_roll as the node memory (else nodes and edges would be from different wrong examples); OFF is
        handled by zero_memory upstream (no prepend → hook no-ops). Returns hook handles.

        The FINAL layer is EXCLUDED: its injection is added post-attention to node rows that no later layer
        reads (loss is on text logits), so it would be a structurally-dead tap (zero gradient)."""
        enc = self.encoder
        if not getattr(enc, "live_read", False) or M <= 0 or not mem_aux:
            return []
        if mem_aux.get("read_mode") != "live_inject":
            return []
        R, C = mem_aux.get("edge_R"), mem_aux.get("edge_C")
        if R is None or C is None:
            return []
        if shuffle_memory and R.shape[0] > 1:            # SHUF: modulate with the WRONG example's edges
            R = torch.roll(R, shifts=int(shuffle_roll), dims=0)
            C = torch.roll(C, shifts=int(shuffle_roll), dims=0)
        E_de = enc._effective_edge(R, C)                 # [B,M,M,d_e] — computed once, re-used every layer
        L = len(self.decoder.llama.model.layers)
        wl = int(getattr(enc, "write_layers", 0))
        # last write_layers EXCLUDING the final layer (dead); aligns with the harvest taps minus layer L-1.
        inject_layers = set(range(L - wl, L - 1)) if wl > 0 else set(range(L - 1))

        def _mk(li):
            def _hook(module, args, kwargs, out):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.shape[1] < offset + M:
                    return None
                cos, sin = kwargs["position_embeddings"]
                node_h = hidden[:, offset:offset + M]
                resid = enc._edge_resid(module, node_h, cos[:, offset:offset + M],
                                        sin[:, offset:offset + M], E_de)
                hs = out[0] if isinstance(out, tuple) else out
                hs = torch.cat([hs[:, :offset], hs[:, offset:offset + M] + resid.to(hs.dtype),
                                hs[:, offset + M:]], dim=1)
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            return _hook
        return [self.decoder.llama.model.layers[li].self_attn.register_forward_hook(_mk(li), with_kwargs=True)
                for li in sorted(inject_layers)]

    def _install_prepend_refresh_hooks(self, M: int, offset: int, zero_memory: bool,
                                       shuffle_memory: bool):
        """biomem-prepend PER-LAYER REFRESH: before every decoder layer, re-read the written edges
        with the current (attention-mixed, query-aware) slot hiddens and ADD a zero-init-gated recall
        to the M prepend positions [offset:offset+M]. The slots become a working memory refined through
        depth — recovering the generation-time conditioning a static prepend gives up, and letting each
        slot see the others (dedup). No-op unless the encoder sets wants_prepend_refresh.
        REAL = refresh; OFF (zero_memory) = none; SHUF = re-read the WRONG-example edges (rolled W)."""
        enc = self.encoder
        if not getattr(enc, "wants_prepend_refresh", False) or zero_memory or M <= 0:
            return []
        if not getattr(enc, "has_read_state", lambda: False)():
            return []
        if shuffle_memory:                                   # roll W to match the rolled prepend memory
            enc.roll_read_state()                            # (harness already enforces B>1 for SHUF)
        def _hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is None or h.shape[1] < offset + M:
                return None
            delta = enc.refresh_prepend(h[:, offset:offset + M])      # [B,M,d] gated recall
            new = h.clone()
            new[:, offset:offset + M] = new[:, offset:offset + M] + delta.to(h.dtype)
            if args:
                return (new,) + tuple(args[1:]), kwargs
            kw = dict(kwargs); kw["hidden_states"] = new
            return args, kw
        return [layer.register_forward_pre_hook(_hook, with_kwargs=True)
                for layer in self.decoder.llama.model.layers]

    def _encode_for_memory(self, ctx_embeds, ctx_mask):
        """What the memory WRITE ingests. Arms that set `ingest_lm_final_hidden` (biomem,
        arrivalmem) ingest the FROZEN LM's FINAL-layer hidden state — run the full backbone
        over the context — NOT raw token embeddings, so they encode with the pretrained LM's
        depth like the baselines (no_grad: a fixed pretrained encoder; the memory's own
        modules do the learning). Other arms keep their existing input (raw embeds)."""
        if not getattr(self.encoder, "ingest_lm_final_hidden", False):
            return ctx_embeds
        # FROZEN contextualizer: bypass the trainable read-side LoRA (disable_lora) so the write
        # input is the PURE pretrained backbone, not a moving target shaped by an adapter that
        # receives NO gradient from this no_grad path. Without it every ingest_lm_final_hidden arm
        # (arrivalmem / biomem) silently encoded against a drifting feature distribution.
        with torch.no_grad(), disable_lora():
            out = self.decoder.llama.model(
                inputs_embeds=ctx_embeds, attention_mask=ctx_mask.long(), use_cache=False)
        return out.last_hidden_state

    def _token_surprise(self, hidden, ctx_ids, ctx_mask, chunk: int = 256):
        """Per-token next-token SURPRISE from the FROZEN LM: surprise_t = −log p_LM(ctx_{t+1} | ctx_≤t),
        normalized by log(vocab) → ~[0,1]. A FREE, pretrained 'this token was unexpected' signal — the LM
        is already a next-token predictor, so its prediction error is a principled surprise (no extra net,
        unlike Titans which constructs one). no_grad + disable_lora (pure backbone); chunked over T to bound
        the [B,chunk,V] logits; detached (a fixed write-time conditioning feature). Returns [B,T] or None."""
        if ctx_ids is None:
            return None
        B, T, _ = hidden.shape
        V = self.cfg.llama_vocab_size
        surprise = hidden.new_zeros(B, T)
        with torch.no_grad(), disable_lora():
            for s in range(0, max(T - 1, 1), chunk):
                e = min(s + chunk, T - 1)
                if e <= s:
                    break
                logits = self.decoder.llama.lm_head(hidden[:, s:e]).float()         # [B, blk, V]
                tgt = ctx_ids[:, s + 1:e + 1]                                        # next-token ids
                ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                     reduction="none").view(B, e - s)
                surprise[:, s:e] = ce
        return (surprise / math.log(V) * ctx_mask.float()).detach()                  # ~[0,1], padding→0

    def compute_streaming_continuation_loss(
        self, batch, *, window_size: int, n_horizons: int,
        zero_memory: bool = False, shuffle_memory: bool = False, shuffle_roll: int = 1,
    ) -> dict:
        """Multi-horizon streaming continuation. At each streaming-window boundary b (256, 512, …,
        T_ctx) compress the prefix [0:b] into memory and predict the next ``predict_len`` block from
        that memory alone; average CE across horizons. The intermediate target blocks live INSIDE
        ``context_ids`` (the token right after boundary b), and the final boundary (b == T_ctx)
        predicts ``answer_ids`` (the block just past the compressed span — the classic single-shot
        continuation). Reuses the generic decode per horizon via ``_force_generic`` sub-batches, so
        every arm works with no encoder change and REAL/SHUF/OFF flow through per horizon.

        n_horizons caps how many of the DEEPEST boundaries to score (the full-context one is always
        included, so this stays comparable to single-shot eval). Cost ≈ n_horizons prefix re-streams;
        acceptable for a 4-window streaming episode (continuation is ~1/5 of the mix)."""
        B, T_ctx = batch.context_ids.shape
        predict_len = batch.answer_ids.shape[1]
        # intermediate-horizon targets are sliced predict_len tokens after each boundary; if that exceeds
        # window_size, adjacent horizons' targets OVERLAP and those tokens get double-scored. Guard it.
        assert predict_len <= window_size, (
            f"multi-horizon continuation needs predict_len ({predict_len}) <= window_size "
            f"({window_size}); otherwise horizon targets overlap and are double-counted")
        n_windows = (T_ctx + window_size - 1) // window_size
        boundaries = [min((i + 1) * window_size, T_ctx) for i in range(n_windows)]  # increasing; last == T_ctx
        boundaries = boundaries[-n_horizons:]                                       # keep the deepest (incl. full-context)
        # which boundaries actually decode a horizon (full-context always; intermediate only if the
        # predict_len target block fits inside the span) — snapshot memory ONLY at these.
        keep = {b for b in boundaries if b >= T_ctx or b + predict_len <= T_ctx}

        # ── Stream the encoder ONCE and SNAPSHOT (memory, aux) at each scored boundary ──
        # (was: re-encode the prefix [0:b] from scratch per horizon → O(n_horizons^2) window-forwards.)
        # Numerically identical: the memory at boundary b is a deterministic function of the prefix, and
        # streaming_write sees the same window slices in the same order. For accumulate-in-finalize arms
        # (slotgraph4/h2o) the state after window w already holds emb[:, :b], so finalize there equals the
        # old per-horizon re-encode; for incremental arms (icae/ac/titans/gisting/memoryllm) finalize is a
        # cheap pure read, so this is the O(n) win. One shared encoder graph → correct summed gradients.
        snapshots = {}
        if keep:
            device = batch.context_ids.device
            embed = self.decoder.llama.get_input_embeddings()
            with torch.no_grad():
                ctx_embeds_full = embed(batch.context_ids)
            enc_input = self._encode_for_memory(ctx_embeds_full, batch.context_mask)
            surprise_full = (self._token_surprise(enc_input, batch.context_ids, batch.context_mask)
                             if getattr(self.encoder, "wants_surprise", False) else None)
            with torch.no_grad():
                q_embeds_const = embed(batch.question_ids)
            state = self.encoder.init_streaming_state(B, device, ctx_embeds_full.dtype)
            ckpt_stream = (getattr(self.cfg, "grad_checkpoint_stream", True)
                           and self.training and torch.is_grad_enabled())
            for w in range(n_windows):
                s = w * window_size
                e = min(s + window_size, T_ctx)
                win_emb = enc_input[:, s:e, :]
                win_mask = batch.context_mask[:, s:e]
                win_sur = surprise_full[:, s:e] if surprise_full is not None else None
                if ckpt_stream:
                    def _write(st, em, mk, su=win_sur, off=s):
                        new_st, _ = self.encoder.streaming_write(st, em, mk, chunk_offset=off, surprise=su)
                        return new_st
                    state = torch.utils.checkpoint.checkpoint(
                        _write, state, win_emb, win_mask, use_reentrant=False)
                else:
                    state, _ = self.encoder.streaming_write(
                        state, win_emb, win_mask, chunk_offset=s, surprise=win_sur)
                if e in keep:                                 # snapshot memory at this boundary
                    snap = dict(state) if isinstance(state, dict) else state
                    if isinstance(snap, dict):                # question is constant across horizons
                        snap["question_embeds"] = q_embeds_const
                        snap["question_mask"] = batch.question_mask
                    snapshots[e] = self.encoder.finalize_memory(snap)

        losses, accs, last_out = [], [], None
        for b in boundaries:
            if b not in keep:                                 # intermediate target would overflow → skip
                continue
            if b >= T_ctx:                                    # full-context horizon → predict answer_ids
                sub = replace(batch, answer_ids=batch.answer_ids, answer_mask=batch.answer_mask,
                              answer_content_mask=batch.answer_content_mask)
            else:                                             # intermediate horizon → predict the in-context next block
                tgt = batch.context_ids[:, b:b + predict_len]
                cm = torch.ones_like(tgt, dtype=torch.bool)
                sub = replace(batch, answer_ids=tgt, answer_mask=cm.clone(), answer_content_mask=cm)
            # decode via the snapshot memory (memory_override SKIPS the encoder); SHUF/OFF roll the given
            # memory exactly as they would the freshly-encoded memory, so REAL/SHUF/OFF parity holds.
            out = self.compute_loss(sub, window_size=window_size, _force_generic=True,
                                    memory_override=snapshots[b],
                                    zero_memory=zero_memory, shuffle_memory=shuffle_memory,
                                    shuffle_roll=shuffle_roll)
            losses.append(out["loss"]); accs.append(out["top1_acc"]); last_out = out

        if not losses:                                        # degenerate (predict_len > window_size etc.) → single-shot
            return self.compute_loss(batch, window_size=window_size, _force_generic=True,
                                     zero_memory=zero_memory, shuffle_memory=shuffle_memory,
                                     shuffle_roll=shuffle_roll)
        loss = torch.stack(losses).mean()
        out = dict(last_out)
        out["loss"] = loss
        out["loss_recon"] = loss
        out["top1_acc"] = torch.stack(accs).mean()
        out["n_horizons"] = len(losses)
        return out

    def compute_loss(
        self,
        batch,                          # QABatch from data/tasks/base.py::_collate
        window_size: int = 1024,
        zero_memory: bool = False,
        shuffle_memory: bool = False,
        memory_override=None,           # (memory, aux): SKIP the encoder, decode with this memory
        shuffle_roll: int = 1,          # SHUF roll amount (in-batch InfoNCE sweeps r=1..B−1)
        return_memory: bool = False,    # stash pre-roll memory+aux on the out dict
        encoder_only: bool = False,     # build memory, return WITHOUT decoding (GradCache cut point)
        return_logits: bool = False,    # add answer-position sel_logits/sel_targets (behavioral-KL distillation)
        _force_generic: bool = False,   # internal: bypass the continuation multi-horizon dispatch (per-horizon calls)
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
                mask_ratio=(getattr(batch, "mask_ratio", None)
                            if getattr(batch, "mask_ratio", None) is not None else self.cfg.mae_mask_ratio),
                memory_override=memory_override, shuffle_roll=shuffle_roll,
                return_memory=return_memory, encoder_only=encoder_only)
        # ---- streaming continuation: predict the next block at each window boundary ----
        # Only for the plain CE / REAL-SHUF-OFF path (objective-mode + GradCache stay single-shot).
        if (getattr(self, "task_mode", None) == "continuation" and not _force_generic
                and getattr(self.cfg, "continuation_multi_horizon", True)
                and memory_override is None and not encoder_only
                and not return_memory and not return_logits):
            T_ctx = batch.context_ids.shape[1]
            n_windows = (T_ctx + window_size - 1) // window_size
            n_h = getattr(batch, "n_horizons", None)
            n_h = n_windows if n_h is None else min(int(n_h), n_windows)
            if n_h > 1:
                return self.compute_streaming_continuation_loss(
                    batch, window_size=window_size, n_horizons=n_h,
                    zero_memory=zero_memory, shuffle_memory=shuffle_memory, shuffle_roll=shuffle_roll)
        device = batch.context_ids.device
        B, T_ctx = batch.context_ids.shape
        T_q = batch.question_ids.shape[1]
        T_a = batch.answer_ids.shape[1]

        embed = self.decoder.llama.get_input_embeddings()

        # ---- 1. Encode context (no_grad embed lookup) ----
        with torch.no_grad():
            ctx_embeds = embed(batch.context_ids)
            ctx_surprise = None   # was the plastic_baseline neuromod signal (retired)

        if memory_override is not None:
            # objective-mode rolled read: SKIP the encoder entirely — decode with the given memory
            # (the caller ran the encoder once; SHUF below rolls a COPY of the aux, never the caller's)
            memory, finalize_aux = memory_override
            finalize_aux = dict(finalize_aux)
            n_windows = 0
        else:
            # biomem/arrivalmem ingest the frozen LM's FINAL hidden (not raw embeds); others raw.
            enc_input = self._encode_for_memory(ctx_embeds, batch.context_mask)
            # biomem next-token SURPRISE (write-time conditioning); None for arms that don't want it.
            surprise_full = (self._token_surprise(enc_input, getattr(batch, "context_ids", None),
                                                  batch.context_mask)
                             if getattr(self.encoder, "wants_surprise", False) else None)

            n_windows = (T_ctx + window_size - 1) // window_size
            state = self.encoder.init_streaming_state(B, device, ctx_embeds.dtype)
            # Activation-checkpoint each window during training so we hold ~one
            # window of encoder activations instead of all n_windows at once (the
            # chunk=8192 OOM for the windowed encoders flat/continuous/MT). Exact
            # gradients; recompute in backward. Skipped under no_grad (eval).
            # BUT: if the encoder's base LM ALREADY self-checkpoints (icae/autocompressor/gisting/
            # memoryllm/slotgraph call base.model.gradient_checkpointing_enable in __init__), this outer
            # checkpoint is REDUNDANT — it recomputes the whole windowed forward AGAIN on top of the
            # inner per-layer recompute (2-3× the base-LM forward per step for the same memory goal). The
            # inner HF checkpointing already bounds each window's working set to ~one layer; skip the
            # outer level for those arms (~20-30% step compute). (~15 lines of audit; verified VRAM-safe.)
            _bm = getattr(getattr(self.encoder, "base", None), "model", None)
            _enc_self_ckpt = bool(getattr(_bm, "is_gradient_checkpointing", False))
            ckpt_stream = (getattr(self.cfg, "grad_checkpoint_stream", True)
                           and self.training and torch.is_grad_enabled()
                           and not _enc_self_ckpt)
            del ctx_surprise   # retired surprise/contextualization plumbing
            for w in range(n_windows):
                s = w * window_size
                e = min(s + window_size, T_ctx)
                win_emb = enc_input[:, s:e, :]
                win_mask = batch.context_mask[:, s:e]
                win_sur = surprise_full[:, s:e] if surprise_full is not None else None
                if ckpt_stream:
                    def _write(st, em, mk, su=win_sur, off=s):
                        new_st, _ = self.encoder.streaming_write(st, em, mk, chunk_offset=off, surprise=su)
                        return new_st
                    state = torch.utils.checkpoint.checkpoint(
                        _write, state, win_emb, win_mask, use_reentrant=False)
                else:
                    state, _ = self.encoder.streaming_write(
                        state, win_emb, win_mask, chunk_offset=s, surprise=win_sur)
            # Hand the question to the encoder (dict-state variants may read it;
            # NullEncoder (Tensor state) and Mamba (list state) are non-dict — guard).
            if isinstance(state, dict):
                state["question_embeds"] = embed(batch.question_ids)
                state["question_mask"] = batch.question_mask
            memory, finalize_aux = self.encoder.finalize_memory(state)
        # memory: [B, M, d_llama] or [B, 0, d_llama] for MT/Vanilla/graph
        M = memory.shape[1]
        # GradCache cut point / objective-mode stash: the pre-roll REAL memory
        if encoder_only or return_memory:
            _mem_ret = (memory, dict(finalize_aux))
        if encoder_only:
            z = torch.zeros((), device=device)
            return {"loss": z, "loss_recon": z, "top1_acc": z, "memory_shape": (B, M),
                    "_memory": _mem_ret[0], "_mem_aux": _mem_ret[1]}

        # graph_baseline now PREPENDS its E memory tokens like any compressor (finalize
        # returns [B,E,d_llama]); REAL/OFF/SHUF are handled by the shared prepend logic
        # below. The parsed graph dict rides in finalize_aux["graph"] for the canaries.

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
                memory = torch.roll(memory, shifts=int(shuffle_roll), dims=0)
                # Roll the per-slot memory mask in lockstep (vanilla_full_context sets it):
                # rolling memory but not its mask would pair row i's rolled memory with
                # row i's ORIGINAL pad mask, contaminating the SHUF control.
                _mm = finalize_aux.get("memory_mask")
                if _mm is not None:
                    finalize_aux["memory_mask"] = torch.roll(_mm, shifts=int(shuffle_roll), dims=0)
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
        # Hoist per-row lengths to CPU in ONE sync each (was 2·B .item() syncs inside the loop).
        q_lens_l = q_lens.tolist()
        a_lens_l = a_lens.tolist()
        # Vectorized answer→prediction alignment (replaces the per-answer-token `for k in range(t_a)`
        # inner loop, which did a GPU→CPU sync per token via `if not answer_content_for_loss[i,k]`).
        # ans_start[i] = header offset + memory + post-mem scaffold + question + post-q scaffold.
        ans_start_vec = (L_pre + M + L_post_mem) + q_lens.long() + L_post_q       # [B]
        T_a = answer_content_for_loss.shape[1]
        _k = torch.arange(T_a, device=device)
        pred_pos = ans_start_vec[:, None] + _k[None, :] - 1                       # [B, T_a]
        valid_km = (answer_content_for_loss.bool()
                    & (_k[None, :] < a_lens[:, None])
                    & (pred_pos >= 0) & (pred_pos < T_total - 1))                 # [B, T_a]
        _rows = torch.arange(B, device=device)[:, None].expand(B, T_a)[valid_km]
        _cols = pred_pos[valid_km]
        pred_mask[_rows, _cols] = True
        pred_targets[_rows, _cols] = answer_ids_for_loss[valid_km]
        for i in range(B):
            t_q = q_lens_l[i]
            t_a = a_lens_l[i]
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
            # Real answer (with eot appended if chat-template path). The answer→prediction alignment
            # (pred_mask/pred_targets) is done vectorially ABOVE the loop; col here must match the
            # ans_start_vec formula (header+M+post_mem+question+post_q) exactly.
            if t_a > 0:
                full_embeds[i, col:col + t_a] = a_embeds[i, :t_a]
                attn_mask_full[i, col:col + t_a] = True

        # ---- 4. Llama forward (causal mask is native; padding via attn_mask) ----
        # soft_pointer_graph reads via the SAME prepend path as the baselines: its
        # finalize_memory returns [B, K_edge, d_llama] memory that is prepended
        # like every other arm (no privileged per-position inject hook), so the
        # comparison isolates the write mechanism and the REAL/SHUF/OFF binding
        # gate applies to the graph too. biomem is the exception: its read is
        # query-conditioned via a tap-layer hook (installed below), not a prepend.
        hook_handle = self._install_conditioned_read_hook(zero_memory, shuffle_memory, B)
        # slotgraph: re-inject the structural embed at each LM layer. Memory is at positions [0:M]
        # only when there is NO chat scaffold (L_pre=0); guard on that (the active backbone).
        reinforce = (self._install_prepend_reinforce_hooks(finalize_aux, M, 0, shuffle_memory)
                     if self.chat_template is None else [])
        # biomem-prepend per-layer refresh (memory at [0:M] only with no chat scaffold).
        refresh = (self._install_prepend_refresh_hooks(M, 0, zero_memory, shuffle_memory)
                   if self.chat_template is None else [])
        # slotgraph Option B (live read): edge-modulate the decoder's last-K node↔node attention. Gated on
        # not-zero_memory so the OFF band-gate control (memory suppressed) also drops the live edge injection.
        live_inject = (self._install_live_inject_hooks(finalize_aux, M, 0, shuffle_memory, shuffle_roll)
                       if (self.chat_template is None and not zero_memory) else [])
        if (self.chat_template is not None and not zero_memory
                and getattr(self.encoder, "wants_prepend_refresh", False)
                and not getattr(self, "_warned_refresh_suppressed", False)):
            print("[WARN] biomem per-layer refresh SUPPRESSED in the QA path: a chat template is set, "
                  "so the memory is not at offset 0. The prepend refresh (a core part of the read) is OFF. "
                  "Use a no-chat-template backbone (e.g. SmolLM2-135M) for the full biomem read.", flush=True)
            self._warned_refresh_suppressed = True

        try:
            # Selective lm_head: run base model for hidden states, then only
            # apply lm_head to prediction positions. Profiled saving at B=12,
            # T=224: 120→102 ms (15%), 7603→6209 MiB (1.4 GiB). Avoids the
            # [B, T, vocab=128K] logits tensor entirely.
            # v5.5 perf: use_cache=False — QA is teacher-forced + single-pass,
            # there is no autoregressive decode that reuses KV. Building the
            # cache adds a noticeable per-step cost (esp. for vanilla_full_ctx
            # where T_ctx ~= 8K).
            _attn_qa = attn_mask_full.to(torch.long)
            _rect_qa = (getattr(self.cfg, "rect_prepend_mask", False) and M > 0 and not zero_memory
                        and self.chat_template is None)      # memory sits at [0:M) only without a chat scaffold
            # live read (Option B) forces BIDIR memory attention (dense node↔node, matching the write's graph)
            # and UNIFORM node positions (permutation-symmetric SET, matching the write's node block).
            _live = bool(live_inject)
            _bidir_qa = ((getattr(self.cfg, "bidir_mem_attn", False) or _live) and M > 0 and not zero_memory
                         and self.chat_template is None)
            if _rect_qa and _bidir_qa:
                raise ValueError("rect_prepend_mask and bidir_mem_attn are mutually exclusive read geometries")
            if _rect_qa:
                _attn_qa = self._rect_prepend_mask(_attn_qa, M, full_embeds.dtype)
            elif _bidir_qa:
                _attn_qa = self._bidir_prepend_mask(_attn_qa, M, full_embeds.dtype)
            _pos_qa = (self._uniform_mem_position_ids(B, M, T_total, device)
                       if ((getattr(self.cfg, "uniform_mem_pos", False) or _live) and M > 0 and not zero_memory
                           and self.chat_template is None) else None)
            if _pos_qa is None:
                # COMPACT position_ids from the attention mask so INTERNAL right-padding does not open a
                # RoPE gap before the question. Critical for the full-context CEILING (M=T padded memory)
                # and the behavioral_kl TEACHER (reuses this path with M=T, uniform_mem_pos forced off):
                # with default arange positions the question lands at absolute pos T (past 40–370 pad slots),
                # distorting its hidden states / the KL target. cumsum → real tokens 0..valid-1, padding
                # pinned (masked anyway), question at `valid`. No-op for dense-memory compressors
                # (cumsum of all-ones == arange), so it does not perturb icae/ac/titans.
                _pos_qa = (attn_mask_full.long().cumsum(dim=1) - 1).clamp_min(0)
            if (getattr(self.encoder, "reads_per_layer_kv", False) and not zero_memory
                    and finalize_aux.get("past_kv") is not None):
                # per-layer-KV native read: M==0 above so full_embeds is memory-free ([pre,q,a]);
                # inject the encoder's per-layer (K,V) as a prefix cache (memory attended as keys only).
                hidden = self._prefix_kv_forward(
                    full_embeds, attn_mask_full, finalize_aux["past_kv"], finalize_aux.get("memory_mask"),
                    shuffle_memory=shuffle_memory, shuffle_roll=shuffle_roll)
            else:
                base_out = self.decoder.llama.model(
                    inputs_embeds=full_embeds,
                    attention_mask=_attn_qa,
                    position_ids=_pos_qa,
                    use_cache=False,
                )
                hidden = base_out.last_hidden_state        # [B, T_total, d_llama]
        finally:
            if hook_handle is not None:
                hook_handle.remove()
            for hh in live_inject:
                hh.remove()
            for hh in reinforce:
                hh.remove()
            for hh in refresh:
                hh.remove()
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
            # Per-example CE (mean per-token NLL per row) WITH grad — the objective modes' InfoNCE
            # logits are built from these across memory rolls; detached copy keeps telemetry unchanged.
            row_idx = pred_mask.nonzero(as_tuple=False)[:, 0]      # [N_pred]
            row_sum = torch.zeros(B, device=device, dtype=per_token_nll_train.dtype)
            row_cnt = torch.zeros(B, device=device, dtype=per_token_nll_train.dtype)
            row_sum = row_sum.scatter_add(0, row_idx, per_token_nll_train)
            row_cnt = row_cnt.scatter_add(0, row_idx, torch.ones_like(per_token_nll_train))
            loss_per_example = (row_sum / row_cnt.clamp_min(1)).to(loss_recon.dtype)   # [B] grad
            per_example_loss = loss_per_example.detach()
        else:
            loss_recon = (memory.float().sum() * 0.0
                          + self.decoder.mask_embed.float().sum() * 0.0)
            loss_per_example = torch.zeros(B, device=device, dtype=loss_recon.dtype) + loss_recon * 0.0
            per_example_loss = loss_per_example.detach()

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
        _vq = finalize_aux.get("vq_loss", None)                # vqicae commitment (pre-weighted by beta)
        if _vq is not None:
            loss = loss + _vq.to(loss.dtype)
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
                tok_correct = (sel_preds == sel_targets)       # [N_pred] bool
                top1_acc = tok_correct.float().sum() / n_content_total
                # Per-example teacher-forced exact match: a row is correct iff
                # ALL its content-position argmax predictions match the gold
                # answer tokens. With all-content answers (bAbI/EMAT) this is the
                # answer-span EM under teacher forcing. Mirrors per_example_loss's
                # scatter aggregation. Rows with no content position score 0.
                row_idx = pred_mask.nonzero(as_tuple=False)[:, 0]   # [N_pred]
                row_cnt = torch.zeros(B, device=device)
                row_hit = torch.zeros(B, device=device)
                row_cnt.scatter_add_(0, row_idx, torch.ones_like(tok_correct, dtype=torch.float))
                row_hit.scatter_add_(0, row_idx, tok_correct.float())
                per_example_em = ((row_hit == row_cnt) & (row_cnt > 0)).float()  # [B]
            else:
                top1_acc = torch.zeros((), device=device)
                per_example_em = torch.zeros(B, device=device)

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
            "loss_per_example": loss_per_example,             # [B] WITH grad (objective-mode InfoNCE logits)
            "per_example_em": per_example_em.detach(),        # [B] teacher-forced exact match (bAbI EM)
            "aux": finalize_aux,
        }
        if return_memory:
            out["_memory"], out["_mem_aux"] = _mem_ret
        if return_logits:
            # answer-position logits for behavioral-KL distillation (teacher = full-context memory
            # override, student = encoder memory). pred_mask.nonzero() is row-major over identical
            # answer packing, so student/teacher sel_logits align 1:1 by (row, answer-token). sel_targets
            # returned so the caller can assert the alignment. None when no content positions.
            if pred_mask.any():
                out["sel_logits"] = sel_logits            # [N_pred, V] (grad on the student call)
                out["sel_targets"] = sel_targets          # [N_pred]
            else:
                out["sel_logits"] = None
                out["sel_targets"] = None
        if splat_telemetry is not None:
            out.update(splat_telemetry)
        if graph_telemetry is not None:
            out.update(graph_telemetry)
        # graph_baseline anti-collapse canaries (shared with the MAE path). The parsed
        # graph rides in finalize_aux["graph"]; mem_effrank uses the prepended memory.
        _parsed_graph = finalize_aux.get("graph")
        if _parsed_graph is not None and self.variant == "graph_baseline":
            out.update(self._graph_canaries(_parsed_graph, memory))
        if hlvocab_telemetry:
            out.update(hlvocab_telemetry)
        # biomem / slotgraph / arrival structure canaries — surface every scalar so the
        # trainer's biomem_/slotgraph_/arrival_ globs log them on EVERY mixed subtask, not just
        # MAE (the masked-recon path merges these at its own return; the generic path must too
        # or babi/continuation drop them).
        for _k, _v in (finalize_aux or {}).items():
            if not (_k.startswith("biomem_") or _k.startswith("slotgraph_")
                    or _k.startswith("h2o_") or _k.startswith("vqicae_")):
                continue
            if torch.is_tensor(_v) and _v.numel() == 1:
                out[_k] = _v.detach()
            elif isinstance(_v, (int, float)):
                out[_k] = _v
        # flat_baseline codebook health → top-level so the trainer logs it to
        # jsonl (codes_active = #live codes; collapse = the flat analogue of
        # graph routing collapse).
        for _k in ("codes_active", "routing_entropy"):
            if _k in finalize_aux:
                out[_k] = finalize_aux[_k]
        return out
