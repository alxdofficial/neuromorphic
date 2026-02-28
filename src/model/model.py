"""
NeuromorphicLM (v4) — iterative refinement with cortical columns.

Processes N-token segments through R iterative passes.  All B_blocks * C
columns process N_C = N//C tokens each via a single CorticalColumnGroup
with G = B*C groups.  PM/EM state has explicit B dim: [BS, B, ...].
No Python loop over blocks — only a loop over R refinement passes.

Fan-out: interleaved token partitioning — column c gets tokens c, c+C, c+2C, ...
         with feature slice c*D_col:(c+1)*D_col.
Fan-in: mean across blocks + skip connection with small-scale-init D_col→D projection.
PM/EM reads use per-column D_col slices; writes stay at block level D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .column import CorticalColumnGroup
from .procedural_memory import ProceduralMemory, PMNeuromodulator
from .episodic_memory import EpisodicMemory, EMNeuromodulator
from .utils import runtime_state_dtype, unit_normalize


class NeuromorphicLM(nn.Module):
    """v4: Iterative refinement with cortical columns (block-batched)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        B = config.B_blocks
        C = config.C
        self.B_C = B * C  # G

        D = config.D
        D_embed = config.D_embed

        # Embedding + position
        self.embedding = nn.Embedding(config.vocab_size, D_embed)
        self.pos_embedding = nn.Embedding(config.N, D)

        # D_embed / D_model projection (None when D_embed == D for zero overhead)
        if D_embed != D:
            self.proj_up = nn.Linear(D_embed, D)
            self.proj_down = nn.Linear(D, D_embed)
        else:
            self.proj_up = None
            self.proj_down = None

        # Fan-out: free reshape (MHA-style broadcast + split, no learned params)

        # Single merged column group (G = B*C groups)
        self.columns = CorticalColumnGroup(config)

        # Single PM + EM (state: [BS, B, ...], D-dim block level)
        self.pm = ProceduralMemory(config.D, config.r, config)
        self.em = EpisodicMemory(config.D, config.M, config)

        # Single neuromodulator pair (shared across blocks)
        self.pm_neuromod = PMNeuromodulator(config.D, config)
        self.em_neuromod = EMNeuromodulator(config.D, config)

        # Fan-in: mean across blocks + D_col -> D skip-connection projection
        # Small-scale init (not zero): gradient must flow through fan_in to column
        # processing. Zero-init would kill all upstream gradients since
        # d(loss)/d(x_flat) = W.T @ d(loss)/d(output) = 0 when W = 0.
        D_col = config.D_col
        self.fan_in = nn.Linear(D_col, D)
        nn.init.normal_(self.fan_in.weight, std=0.02)
        nn.init.zeros_(self.fan_in.bias)
        self.ln_final = nn.LayerNorm(D)

        # LM head (optionally tied to embedding)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Damped mixing parameter (sigmoid -> lambda_mix init)
        # logit(0.5) = 0.0
        init_logit = torch.log(torch.tensor(config.lambda_mix / (1 - config.lambda_mix + 1e-8)))
        self.lambda_logit = nn.Parameter(torch.tensor(float(init_logit)))

    def _to_cols(self, x: Tensor) -> Tensor:
        """[BS, N, D] -> [BS, N_C, G, D_col] via interleaved token partitioning.

        Column c gets tokens c, c+C, c+2C, ... with feature slice c*D_col:(c+1)*D_col.
        Implemented via diagonal extraction: view [BS, N_C, C, C, D_col] then take
        diagonal(dim1=2, dim2=3) to select matching token-position/feature-slice pairs.
        """
        BS, N, D = x.shape
        B, C, D_col = self.config.B_blocks, self.config.C, self.config.D_col
        N_C = N // C
        # [BS, N_C, tok_group=C, feat_slice=C, D_col]
        x = x.view(BS, N_C, C, C, D_col)
        # diagonal selects elements where tok_group == feat_slice -> [BS, N_C, D_col, C]
        x_diag = torch.diagonal(x, dim1=2, dim2=3)
        # [BS, N_C, C, D_col] — column c has tokens c,c+C,... with feature slice c
        x_part = x_diag.permute(0, 1, 3, 2).contiguous()
        # Replicate across B blocks: [BS, N_C, 1, C, D_col] -> [BS, N_C, B, C, D_col] -> [BS, N_C, G, D_col]
        return x_part.unsqueeze(2).expand(-1, -1, B, -1, -1).reshape(BS, N_C, B * C, D_col)

    def _from_cols(self, x_cols: Tensor, x_input: Tensor) -> Tensor:
        """[BS, N_C, G, D_col] -> [BS, N, D] via mean over blocks + skip connection.

        x_cols: column outputs [BS, N_C, G, D_col]
        x_input: D-space skip connection [BS, N, D]

        Mean across B blocks -> [BS, N_C, C, D_col], reshape restores original token
        order (N_C*C = N), then D_col->D projection + skip connection.
        At init fan_in is small-scale, so output ≈ x_input.
        """
        BS = x_cols.shape[0]
        N_C = x_cols.shape[1]
        B, C, D_col = self.config.B_blocks, self.config.C, self.config.D_col
        N = N_C * C
        # [BS, N_C, B, C, D_col] -> mean across blocks -> [BS, N_C, C, D_col]
        x_mean = x_cols.view(BS, N_C, B, C, D_col).mean(dim=2)
        # Reshape restores original token order: (N_C, C) -> N
        x_flat = x_mean.reshape(BS, N, D_col)
        return x_input + self.fan_in(x_flat)

    def forward_segment(
        self, input_ids: Tensor, reset_mask: Tensor | None = None,
        fitb_mask: Tensor | None = None, target_ids: Tensor | None = None,
        _override_R: int | None = None,
    ):
        """Process N tokens through R refinement passes.

        input_ids: [BS, N]
        reset_mask: [BS] bool — streams to reset PM/EM before processing
        fitb_mask: [BS, N] bool — FITB masked positions (None = NTP path)
        target_ids: [BS, N] — original unmasked IDs for inline FITB loss (optional)

        Returns:
            NTP path  (fitb_mask is None):
                (logits [BS, N, vocab], aux_loss)
            FITB path (target_ids provided):
                (fitb_loss scalar, aux_loss, num_valid tensor)
            FITB path (target_ids is None):
                (per_pass_logits list[R × [BS,N,vocab]], aux_loss)
        """
        if reset_mask is not None:
            self._reset_memory(reset_mask)

        BS, N = input_ids.shape
        B, C, G = self.config.B_blocks, self.config.C, self.B_C
        device = input_ids.device

        is_fitb = fitb_mask is not None
        inline_loss = is_fitb and target_ids is not None
        R_passes = _override_R if _override_R is not None else self.config.R

        # Initial embedding (D-space, full N tokens)
        sequence = input_ids  # may have <FITB> at masked positions
        original_ids = input_ids
        x = self.embedding(sequence)  # [BS, N, D_embed]
        if self.proj_up is not None:
            x = self.proj_up(x)       # [BS, N, D]
        positions = torch.arange(N, device=device)
        x = x + self.pos_embedding(positions)

        # Save D-space representation for skip connection
        x_input = x

        # Fan-out to column space (interleaved token partitioning)
        x_cols = self._to_cols(x)  # [BS, N_C, G, D_col]

        z_hat_prev = None
        aux_loss = torch.tensor(0.0, device=device)
        lam = torch.sigmoid(self.lambda_logit)

        # Pre-compute static-shape targets for inline FITB loss (Bug 9)
        if inline_loss:
            fitb_loss = torch.tensor(0.0, device=device)
            fitb_valid = torch.tensor(0, device=device, dtype=torch.long)
            # Build targets with ignore_index=-100 at non-masked positions
            # Use torch.where for compile-friendly static shapes (no boolean indexing)
            static_targets = torch.where(fitb_mask, target_ids,
                                         torch.tensor(-100, device=device, dtype=target_ids.dtype))
        elif is_fitb:
            per_pass_logits = []

        for r in range(R_passes):
            # FITB: re-embed from updated sequence (r > 0)
            if is_fitb and r > 0:
                x_fresh = self.embedding(sequence)
                if self.proj_up is not None:
                    x_fresh = self.proj_up(x_fresh)
                x_fresh = x_fresh + self.pos_embedding(positions)
                x_input = x_fresh  # update skip base for re-embedded sequence
                x_cols_fresh = self._to_cols(x_fresh)
                x_cols = (1 - lam) * x_out + lam * x_cols_fresh

            # Single column pass: PCM → PM/EM read → lateral → ffn_pre → cross-block → ffn_post → W_post
            x_out, z, z_hat, surprise, elig_info, nov_info = \
                self.columns(x_cols, self.pm, self.em, z_hat_prev)

            # PCM prediction loss
            if self.config.pcm_enabled and z_hat_prev is not None and z is not None:
                aux_loss = aux_loss + self.columns.pcm.prediction_loss(z_hat_prev, z) * self.config.pcm_pred_weight

            # Single commit per R pass
            self._process_and_commit(elig_info, nov_info, x_out)

            if inline_loss:
                # Static-shape FITB loss via skip-connection readout
                x_readout = self._from_cols(x_out, x_input)           # [BS, N, D]
                x_readout = self.ln_final(x_readout)
                if self.proj_down is not None:
                    x_readout = self.proj_down(x_readout)              # [BS, N, D_embed]
                logits_r = self.lm_head(x_readout)                     # [BS, N, vocab]

                V = logits_r.shape[-1]
                loss_r = F.cross_entropy(
                    logits_r.reshape(-1, V), static_targets.reshape(-1),
                    ignore_index=-100, reduction='sum',
                )
                fitb_loss = fitb_loss + loss_r
                fitb_valid = fitb_valid + fitb_mask.sum()

                # Re-embed predictions for next pass (static torch.where)
                if r < R_passes - 1:
                    with torch.no_grad():
                        predicted_ids = logits_r.argmax(dim=-1)  # [BS, N]
                        sequence = torch.where(fitb_mask, predicted_ids, original_ids)

            elif is_fitb:
                # Legacy FITB path (no target_ids): full logits via skip-connection
                x_readout = self._from_cols(x_out, x_input)
                x_readout = self.ln_final(x_readout)
                if self.proj_down is not None:
                    x_readout = self.proj_down(x_readout)
                logits_r = self.lm_head(x_readout)
                per_pass_logits.append(logits_r)

                if r < R_passes - 1:
                    with torch.no_grad():
                        predicted_ids = logits_r.argmax(dim=-1)
                        sequence = torch.where(fitb_mask, predicted_ids, original_ids)
            else:
                # NTP path: damped mixing (column space)
                if r > 0:
                    x_cols = (1 - lam) * x_cols + lam * x_out
                else:
                    x_cols = x_out

            z_hat_prev = z_hat

        # Per-segment memory maintenance (once, not per pass).
        # Passes are refinement at the same time index — decay/aging should
        # reflect elapsed *segments*, not refinement depth.
        if self.config.pm_enabled:
            self.pm.base_decay()
        if self.config.em_enabled:
            if hasattr(self, '_em_decay_last'):
                self.em.base_decay(self._em_decay_last)
            self.em.age_tick(N)

        if inline_loss:
            return fitb_loss, aux_loss, fitb_valid
        if is_fitb:
            return per_pass_logits, aux_loss

        # NTP path: fan-in via skip connection + LM head (once)
        x = self._from_cols(x_cols, x_input)  # [BS, N, D]
        x = self.ln_final(x)
        if self.proj_down is not None:
            x = self.proj_down(x)       # [BS, N, D_embed]
        logits = self.lm_head(x)        # [BS, N, vocab]

        return logits, aux_loss

    # ------------------------------------------------------------------
    # PM eligibility routing + commit, EM novelty + write
    # ------------------------------------------------------------------

    def _process_and_commit(self, elig_info, nov_info, x_out):
        """Route eligibility to PM slots, score EM novelty, then commit/write.

        elig_info: (k_cand, v_cand, gate) — [BS, N_C, B, C, D_col] or [BS, N_C, B, C]
        nov_info: (q_nov, v_nov, w_nov, surprise) — same shapes
        x_out: [BS, N_C, G, D_col] (unused directly, info extracted in column)
        """
        k_cand, v_cand, gate = elig_info
        BS, N, B, C, D_col = k_cand.shape
        D = C * D_col  # = config.D

        # Concat columns -> block level (free reshape)
        k_block = k_cand.reshape(BS, N, B, D)
        v_block = v_cand.reshape(BS, N, B, D)
        gate_block = gate.mean(dim=3)  # [BS, N, B] (average surprise across columns)

        # --- PM eligibility routing (batched matmul, block level) ---
        r = self.config.r
        if self.config.pm_enabled:
            k_norm = unit_normalize(k_block)
            # route_scores via batched matmul (strided views, no copy)
            k_bn = k_norm.transpose(1, 2)                                       # [BS, B, N, D]
            route_scores = torch.matmul(k_bn, self.pm.pm_K.transpose(-1, -2))   # [BS, B, N, r]
            route_scores = route_scores.transpose(1, 2)                          # [BS, N, B, r]

            route_w = torch.softmax(
                route_scores / self.config.tau_route_pm, dim=-1
            )  # [BS, N, B, r]
            gated_routed = gate_block.unsqueeze(-1) * route_w  # [BS, N, B, r]

            # elig_K/V via batched matmul (strided views, no copy)
            gr_bn = gated_routed.permute(0, 2, 3, 1)                            # [BS, B, r, N]
            k_bn = k_block.transpose(1, 2)                                      # [BS, B, N, D]
            v_bn = v_block.transpose(1, 2)                                      # [BS, B, N, D]
            elig_K = torch.matmul(gr_bn, k_bn)                                  # [BS, B, r, D]
            elig_V = torch.matmul(gr_bn, v_bn)                                  # [BS, B, r, D]
        else:
            elig_K = torch.zeros(
                BS, B, r, D,
                device=k_block.device, dtype=k_block.dtype,
            )
            elig_V = torch.zeros_like(elig_K)

        # --- EM novelty scoring + candidate selection (block level) ---
        q_nov_5d, v_nov_5d, w_nov, surp = nov_info
        # Concat to block level
        q_nov = q_nov_5d.reshape(BS, N, B, D)
        v_nov = v_nov_5d.reshape(BS, N, B, D)
        surp_block = surp.mean(dim=3)  # [BS, N, B]
        w_block = w_nov.mean(dim=3)    # [BS, N, B]

        if self.config.em_enabled:
            novelty = self.em.score_novelty(q_nov, surp_block, w_block)
            em_cands = self.em.select_top_candidates(
                q_nov, v_nov, novelty, self.config.C_em
            )
        else:
            em_cands = (
                torch.zeros(BS, B, self.config.C_em, D, device=k_block.device, dtype=k_block.dtype),
                torch.zeros(BS, B, self.config.C_em, D, device=k_block.device, dtype=k_block.dtype),
                torch.zeros(BS, B, self.config.C_em, device=k_block.device, dtype=k_block.dtype),
            )

        # --- PM commit ---
        if self.config.pm_enabled:
            # NOTE: base_decay() is called once per segment (after R loop),
            # not per pass. See forward_segment().
            # Neuromod: flatten [BS, B] -> [BS*B] for MLP, reshape back
            BSB = BS * B
            elig_summary = elig_K.norm(dim=-1).mean(dim=-1)  # [BS, B]
            pm_usage = self.pm.pm_a.sum(dim=-1)              # [BS, B]
            content_emb = elig_K.mean(dim=2)                 # [BS, B, D]
            g, slot_logits, tau, ww_pm = self.pm_neuromod(
                elig_summary.reshape(BSB).detach(),
                pm_usage.reshape(BSB).detach(),
                content_emb.reshape(BSB, D).detach(),
            )
            # Reshape neuromod outputs back to [BS, B, ...]
            g = g.reshape(BS, B)
            slot_logits = slot_logits.reshape(BS, B, -1)
            tau = tau.reshape(BS, B)
            ww_pm = ww_pm.reshape(BS, B)

            # Bug 2: scale g by eligibility magnitude so pass-0 (surprise=0) barely writes
            # Element-wise normalization avoids global .max() reduction (sync barrier)
            elig_mag = (elig_summary / (elig_summary.detach() + 1e-6)).clamp(0, 1)  # [BS, B]
            g = g * elig_mag.detach()

            # Bug 3: learned weakness bias on slot_logits (prefer weaker slots)
            weakness_bias = ww_pm.unsqueeze(-1) * self.pm.pm_a.detach()  # [BS, B, r]
            slot_logits = slot_logits - weakness_bias

            self.pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # --- EM write ---
        cand_K, cand_V, cand_scores = em_cands
        if self.config.em_enabled:
            em_usage = self.em.em_S.sum(dim=-1)              # [BS, B]
            novelty_mean = cand_scores.mean(dim=-1)          # [BS, B]
            content_emb = cand_K.mean(dim=2)                 # [BS, B, D]
            BSB = BS * B
            g_em, tau_em, decay, ww_em = self.em_neuromod(
                novelty_mean.reshape(BSB).detach(),
                em_usage.reshape(BSB).detach(),
                content_emb.reshape(BSB, D).detach(),
            )
            # Reshape back to [BS, B]
            g_em = g_em.reshape(BS, B)
            tau_em = tau_em.reshape(BS, B)
            decay = decay.reshape(BS, B)
            ww_em = ww_em.reshape(BS, B)
            self.em.write(cand_K, cand_V, cand_scores, g_em, tau_em, ww=ww_em)
            # Stash decay for once-per-segment application (after R loop).
            # Only the last pass's decay is used — consistent with PM's base_decay().
            self._em_decay_last = decay

    # ------------------------------------------------------------------
    # Embedding resize (for FITB token registration)
    # ------------------------------------------------------------------

    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize embedding + LM head for new vocab (e.g. after adding FITB tokens).

        Preserves existing weights; new rows are initialized from N(0, 0.02).
        """
        old_vocab = self.embedding.num_embeddings
        if new_vocab_size == old_vocab:
            return

        D_embed = self.config.D_embed
        old_weight = self.embedding.weight.data

        new_emb = nn.Embedding(new_vocab_size, D_embed)
        copy_n = min(old_vocab, new_vocab_size)
        new_emb.weight.data[:copy_n] = old_weight[:copy_n]
        if new_vocab_size > old_vocab:
            nn.init.normal_(new_emb.weight.data[old_vocab:], mean=0.0, std=0.02)
        self.embedding = new_emb

        if self.config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        else:
            old_lm = self.lm_head.weight.data
            new_lm = nn.Linear(D_embed, new_vocab_size, bias=False)
            copy_n = min(old_vocab, new_vocab_size)
            new_lm.weight.data[:copy_n] = old_lm[:copy_n]
            if new_vocab_size > old_vocab:
                nn.init.normal_(new_lm.weight.data[old_vocab:], mean=0.0, std=0.02)
            self.lm_head = new_lm

        self.config.vocab_size = new_vocab_size

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_memory(self, mask: Tensor):
        """Reset PM/EM for masked streams at doc boundary.

        mask: [BS] bool.  PM/EM handle expansion internally.
        """
        if not self.config.lifelong_mode:
            self.pm.reset_content(mask)
            self.em.reset_states(mask)

    def initialize_states(self, BS: int, device: torch.device):
        """Pre-allocate runtime state tensors (with explicit B dim)."""
        dtype = runtime_state_dtype(device)
        self.pm.initialize(BS, device, dtype)
        self.em.initialize(BS, device, dtype)

    def detach_states(self):
        """TBPTT boundary: detach all PM/EM state."""
        self.pm.detach_states()
        self.em.detach_states()

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> Tensor:
        """Simple autoregressive generation using N-token windows.

        prompt_ids: [BS, P] — prompt token ids
        Returns: [BS, P + max_new_tokens]
        """
        N = self.config.N
        BS = prompt_ids.shape[0]
        device = prompt_ids.device
        sequence = prompt_ids

        # Ensure states are initialized
        if not self.pm.is_initialized() or not self.em.is_initialized():
            self.initialize_states(BS, device)

        for _ in range(max_new_tokens):
            # Take last N tokens as context window
            ctx = sequence[:, -N:]
            pad_len = N - ctx.shape[1]
            if pad_len > 0:
                ctx = F.pad(ctx, (pad_len, 0), value=0)

            logits, _ = self.forward_segment(ctx)  # [BS, N, vocab]
            next_logits = logits[:, -1, :]  # [BS, vocab]

            # Temperature + top-k sampling
            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, _ = next_logits.topk(top_k, dim=-1)
                next_logits[next_logits < topk_vals[:, -1:]] = -float('inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)  # [BS, 1]
            sequence = torch.cat([sequence, next_id], dim=1)

        return sequence

    @torch.no_grad()
    def generate_segments(self, prompt_ids: Tensor, max_new_tokens: int = 512,
                          temperature: float = 1.0, top_k: int = 50,
                          refine_passes: int | None = None) -> Tensor:
        """Non-autoregressive segment generation via FITB iterative refinement.

        Phase 1: Process prompt through NTP segments to warm PM/EM.
        Phase 2: Generate N-token chunks with all positions as <FITB>,
                 R passes refine, sample from last pass's logits.

        prompt_ids: [BS, P] — prompt token ids
        max_new_tokens: number of tokens to generate (rounded up to multiple of N)
        temperature: sampling temperature
        top_k: top-k sampling (0 to disable)
        refine_passes: override R for generation (None = use config.R)

        Returns: [BS, P + generated_tokens]
        """
        if self.config.fitb_id < 0:
            raise ValueError(
                "generate_segments requires fitb_id to be set (>= 0). "
                "Register a <FITB> token in your tokenizer first."
            )

        N = self.config.N
        BS = prompt_ids.shape[0]
        device = prompt_ids.device
        R = refine_passes if refine_passes is not None else self.config.R

        # Ensure states are initialized
        if not self.pm.is_initialized() or not self.em.is_initialized():
            self.initialize_states(BS, device)

        # Phase 1: Process prompt through NTP segments to warm PM/EM
        P = prompt_ids.shape[1]
        if P > 0:
            # Pad prompt to multiple of N
            pad_len = (N - P % N) % N
            if pad_len > 0:
                prompt_padded = F.pad(prompt_ids, (pad_len, 0), value=0)
            else:
                prompt_padded = prompt_ids

            n_prompt_segs = prompt_padded.shape[1] // N
            for seg_idx in range(n_prompt_segs):
                seg = prompt_padded[:, seg_idx * N : (seg_idx + 1) * N]
                self.forward_segment(seg)  # NTP path, warms PM/EM

        # Phase 2: Generate N-token chunks via FITB refinement
        generated = []
        tokens_remaining = max_new_tokens

        while tokens_remaining > 0:
            chunk_len = min(N, tokens_remaining)
            # Start with all <FITB> tokens
            seg_ids = torch.full((BS, N), self.config.fitb_id,
                                 device=device, dtype=prompt_ids.dtype)
            fitb_mask = torch.ones(BS, N, dtype=torch.bool, device=device)

            # If chunk_len < N, only the first chunk_len positions are generated
            if chunk_len < N:
                fitb_mask[:, chunk_len:] = False
                seg_ids[:, chunk_len:] = 0  # padding

            # R refinement passes
            per_pass_logits, _ = self.forward_segment(
                seg_ids, fitb_mask=fitb_mask, _override_R=R
            )

            # Sample from last pass's logits
            last_logits = per_pass_logits[-1]  # [BS, N, vocab]
            if temperature != 1.0:
                last_logits = last_logits / temperature
            if top_k > 0:
                topk_vals, _ = last_logits.topk(top_k, dim=-1)
                last_logits[last_logits < topk_vals[:, :, -1:]] = -float('inf')
            probs = F.softmax(last_logits, dim=-1)
            sampled = torch.multinomial(
                probs.reshape(-1, probs.shape[-1]), 1
            ).reshape(BS, N)

            generated.append(sampled[:, :chunk_len])
            tokens_remaining -= chunk_len

        result = torch.cat([prompt_ids] + generated, dim=1)
        return result

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        return self.train(False)
