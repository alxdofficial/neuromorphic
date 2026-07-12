"""AutoCompressors — recurrent soft-prompt summary compressor with summary ACCUMULATION."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from ...common import _NormMatch
from ...config import ReprConfig


class AutoCompressorBaselineEncoder(nn.Module):
    """AutoCompressors (Chevalier et al. EMNLP'23) on a FROZEN Llama + encoder-LoRA.

    The DEFINING mechanism vs plain RMT is **summary accumulation**: the summary vectors
    σ_1..σ_{i-1} of ALL prior windows are CONCATENATED and prepended to window i (a direct
    information pathway to every preceding segment — no re-compression of already-stored
    summaries). Per window we append κ learnable <Sum> tokens after [accumulated σ ++ window
    tokens], run the LoRA-Llama, and read the κ <Sum> output hiddens as this window's summary
    σ_i, which is APPENDED to the accumulation (not overwritten). finalize returns the full
    accumulated κ·n_windows ≈ M vectors as prepend memory. κ = M / n_windows keeps the final
    budget matched to the other fixed-M baselines (12 slots × 8 windows = 96 at ctx2048/win256).

    (An earlier version REPLACED a fixed single-summary carry each window — that is RMT, the
    weaker predecessor AutoCompressor beats; restored to faithful accumulation 2026-07-08.)

    ASTERISKS (results footnotes) — deviations from official AutoCompressors: (1) frozen backbone + LoRA
    (paper fine-tunes the whole LM) — the shared fair-comparison adaptation; (2) FULL BPTT across windows
    (paper uses randomized-substep STOP-GRADIENT); (3) FIXED seg_len segmentation (paper randomizes segment
    boundaries); (4) final NormMatch on the emitted memory; (5) separate encoder/decoder LoRA adapters.
    Summary accumulation itself is faithful. Per-window κ from cfg.autocompressor_summary_per_window."""

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base,
            rank=int(getattr(cfg, "autocompressor_lora_rank", cfg.icae_lora_rank)),
            alpha=int(getattr(cfg, "autocompressor_lora_alpha", cfg.icae_lora_alpha)),
            target_names=tuple(getattr(cfg, "autocompressor_lora_targets",
                                       cfg.llama_lora_target_names)),
        )
        base.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        self.M = int(getattr(cfg, "autocompressor_n_slots", 0) or cfg.n_flat_codes)  # final budget cap
        # κ = summary vectors emitted PER segment; κ·n_segments accumulates to ≈ M. Fallback assumes
        # ctx/segment ≈ 8 when the preset didn't set it (non-mixed paths).
        self.k = int(getattr(cfg, "autocompressor_summary_per_window", 0) or max(1, self.M // 8))
        # AutoCompressor defines its OWN segment length (independent of the harness streaming window):
        # streaming_write chunks whatever token_embeds it is handed into seg_len sub-segments and emits
        # κ per sub-segment. This makes the emitted budget ≈ M in BOTH the single-shot MAE path (one
        # 2048 window → 8 sub-segments → 96) and the multi-window path (eight 256 windows → 96) — so
        # autocompressor is budget-matched to the fixed-M baselines on every task, not just streamed ones.
        self.seg_len = int(getattr(cfg, "autocompressor_segment_len", 0) or 256)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(dim=0)
            emb_std = embed.weight.float().std().item()
        slot_init = (mean_vec.view(1, cfg.d_llama).repeat(self.k, 1)
                     + emb_std * torch.randn(self.k, cfg.d_llama))
        self.slots = nn.Parameter(slot_init)                  # κ learnable <Sum> tokens (reused/window)
        self.norm = _NormMatch(cfg.d_llama)
        with torch.no_grad():   # seed norm-match scale to the backbone embed norm
            self.norm.scale.data.fill_(   # (0.9 default is ~3x too quiet on SmolLM2; match hlvocab)
                base.get_input_embeddings().weight.float().norm(dim=-1).mean().item())
        print(f"[AutoCompressor] encoder-LoRA wrapped {n_wrapped} layers; "
              f"κ={self.k}/window → accumulate to M≤{self.M} (summary accumulation)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def init_streaming_state(self, batch_size: int, device, dtype):
        del batch_size, device, dtype
        return {"acc": None, "valid": None}                   # empty accumulation (grows per window)

    def _compress_segment(self, acc, valid, seg, seg_mask):
        """Emit κ summaries for one segment, appended to the accumulation (σ_{<i} prepended, read-only)."""
        B, Ws, d = seg.shape
        slots = self.slots.to(seg.dtype).unsqueeze(0).expand(B, self.k, d)
        _ones = lambda n: torch.ones(B, n, device=seg.device, dtype=torch.long)
        if acc is None:                                       # first segment: no prior summaries
            inp = torch.cat([seg, slots], dim=1)
            attn = torch.cat([seg_mask.long(), _ones(self.k)], dim=1)
        else:                                                 # prepend ALL prior summaries (accumulation)
            inp = torch.cat([acc, seg, slots], dim=1)
            attn = torch.cat([valid.long(), seg_mask.long(), _ones(self.k)], dim=1)
        # COMPACTED positions (fidelity): trailing segment-pad must not push the κ summary slots to a
        # later RoPE position than the real tokens warrant — cumsum(attn)-1 places the slots right after
        # the last real token, so the emitted summaries are invariant to the segment's pad count.
        pos = torch.clamp(attn.cumsum(dim=1) - 1, min=0)
        h = self.base.model(inputs_embeds=inp, attention_mask=attn, position_ids=pos).last_hidden_state
        sigma = h[:, -self.k:, :].to(seg.dtype)               # this segment's κ summaries
        # An all-pad segment contributes no real content: keep its κ slots (uniform length across the
        # batch → clean batching) but mark them invalid so the decoder never attends to them.
        chunk_valid = seg_mask.bool().any(dim=1).view(B, 1).expand(B, self.k)
        if acc is None:
            acc, valid = sigma, chunk_valid
        else:
            acc, valid = torch.cat([acc, sigma], dim=1), torch.cat([valid, chunk_valid], dim=1)
        if acc.shape[1] > self.M:                             # defensive: never exceed the read budget
            acc, valid = acc[:, :self.M], valid[:, :self.M]
        return acc, valid

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        B, W, d = token_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        acc, valid = state["acc"], state["valid"]             # [B, k_so_far, d] / [B, k_so_far] or None
        # Chunk this call's tokens into AutoCompressor's own seg_len segments and accumulate κ each —
        # so single-shot (W=2048) and per-window (W=256) calls both accumulate to ≈ M.
        # The MAE single-shot path passes W=2048 → 8 segments in ONE call with NO outer per-window
        # checkpoint, so all 8 segment forwards' activations stay live (the ~4-6GB B=8 peak driver, and
        # unique to this arm). Checkpoint each segment when the loop runs >1 seg under grad, so only one
        # segment's activations are held (frozen base, train(False) → no dropout/RNG → recompute is exact).
        n_seg = -(-W // self.seg_len)
        ckpt_seg = (n_seg > 1 and self.training and torch.is_grad_enabled()
                    and getattr(self.cfg, "grad_checkpoint_stream", True))
        for s in range(0, W, self.seg_len):
            if acc is not None and acc.shape[1] >= self.M:
                break
            seg = token_embeds[:, s:s + self.seg_len]
            sm = attention_mask[:, s:s + self.seg_len]
            # checkpoint only once acc/valid are real tensors (checkpoint can't take None args); the
            # first segment (acc=None) is cheap to hold anyway — nothing accumulated yet.
            if ckpt_seg and acc is not None:
                acc, valid = _ckpt.checkpoint(
                    self._compress_segment, acc, valid, seg, sm, use_reentrant=False)
            else:
                acc, valid = self._compress_segment(acc, valid, seg, sm)
        return {"acc": acc, "valid": valid}, {}

    def finalize_memory(self, state):
        acc, valid = state["acc"], state["valid"]
        mem = self.norm(acc.float())                          # [B, k_so_far, d_llama]
        return mem, {"memory_mask": valid}                    # mask out all-pad-window summaries

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        state = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)
