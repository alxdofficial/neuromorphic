"""Phase1RetrievalTrainerV2 — Wave 1 retrieval-pretraining trainer for v2.

Same protocol as v1's Phase1RetrievalTrainer:
- For each batch of M chunks (each chunk has 8 facts + 1 question + answer):
  - Write 8 passage windows (writes go to manifold's edge buffer)
  - Run a zero-memory Llama forward over the question to get q_hiddens
  - Run a Q+A window with read trajectory conditioned on q_hiddens
  - Loss = answer NTP CE + load_balance + z_loss + entry_contrastive + per_step_contrastive

Differences from v1:
- Uses VocabularyManifold instead of mutable Manifold
- No prev_states / new_states plumbing (manifold owns edge buffer)
- Per-step contrastive is on step_queries between matched read/write trajectories
- Edge-level diagnostics instead of cell-level
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2.manifold import rms_norm_last


@dataclass
class V2RetrievalMetrics:
    """Per-step metrics for v2 retrieval trainer."""
    loss: float = 0.0
    answer_loss: float = 0.0
    answer_token_count: int = 0
    aux_load_balance: float = 0.0
    aux_z_loss: float = 0.0
    grad_norm: float = 0.0
    answer_acc: float = 0.0

    # Contrastive losses
    l_contrast_entry: float = 0.0
    l_contrast_per_step: float = 0.0

    # Edge memory health
    n_active_edges: int = 0
    edge_active_fraction: float = 0.0
    mean_fan_out: float = 0.0
    mean_edge_state_norm: float = 0.0
    mean_edge_specificity: float = 0.0
    mean_visit_count: float = 0.0
    mean_edge_age: float = 0.0

    # Walker diagnostics
    entry_logits_max: float = 0.0
    edge_score_active_frac: float = 0.0

    # R↔W overlap variants (the metric family we obsessed over in v1)
    # All are Jaccard ratios |R∩W| / |R∪W| over visited cell-ids per chunk,
    # averaged across the batch.
    rw_overlap: float = 0.0          # alias for rw_overlap_target (back-compat)
    rw_overlap_target: float = 0.0   # read trajectory vs target fact's write
    rw_overlap_all: float = 0.0      # read trajectory vs ALL 8 writes
    rw_overlap_entry: float = 0.0    # entry node only (step 0)
    rw_overlap_hop: float = 0.0      # hops only (steps 1..K-1)

    # Per-window unique cells (mean across batch & J)
    r_unique_per_traj: float = 0.0   # unique cells per individual read trajectory
    w_unique_per_traj: float = 0.0
    r_unique_per_window: float = 0.0  # unique cells per window (across J trajectories)
    w_unique_per_window: float = 0.0

    # Routing entropy at entry (read + write), averaged over J × batch
    # 0 = collapsed to one cell; log(N) = perfectly uniform.
    read_entry_entropy: float = 0.0
    write_entry_entropy: float = 0.0

    # Per-module gradient norms (so we know which side is starving)
    grad_norm_read: float = 0.0
    grad_norm_write: float = 0.0
    grad_norm_entry_proj: float = 0.0
    grad_norm_lambda_edge: float = 0.0
    grad_norm_concept_ids: float = 0.0
    grad_norm_mem_inject: float = 0.0
    grad_norm_read_attn: float = 0.0

    # Vocabulary-bank health (canary for collapse / runaway)
    concept_ids_norm_mean: float = 0.0
    concept_ids_norm_cv: float = 0.0           # CV = std/mean — flat if collapsed
    concept_ids_pairwise_cos: float = 0.0       # mean off-diagonal cos sim

    # Per-key breakdown (val only)
    per_key_loss: dict[str, float] = field(default_factory=dict)
    per_key_acc: dict[str, float] = field(default_factory=dict)
    per_key_n: dict[str, int] = field(default_factory=dict)


class Phase1RetrievalTrainerV2:
    """Mirror of v1 Phase1RetrievalTrainer for the v2 architecture."""

    def __init__(
        self,
        model: IntegratedLMV2,
        optimizer: Optimizer,
        *,
        pad_token_id: int,
        scheduler: Any | None = None,
        grad_clip: float | None = 1.0,
        load_balance_coef: float | None = None,
        z_loss_coef: float | None = None,
        contrast_coef: float | None = None,
        contrast_temperature: float | None = None,
        per_step_contrast_coef: float | None = None,
        per_step_contrast_temperature: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.pad_token_id = pad_token_id
        # Coefs/temperatures: kwarg overrides cfg; otherwise read from cfg.
        cfg = model.cfg
        self.load_balance_coef = load_balance_coef if load_balance_coef is not None else cfg.load_balance_coef
        self.z_loss_coef = z_loss_coef if z_loss_coef is not None else cfg.z_loss_coef
        self.contrast_coef = contrast_coef if contrast_coef is not None else cfg.contrast_coef
        self.contrast_temperature = contrast_temperature if contrast_temperature is not None else cfg.contrast_temperature
        self.per_step_contrast_coef = per_step_contrast_coef if per_step_contrast_coef is not None else cfg.per_step_contrast_coef
        self.per_step_contrast_temperature = per_step_contrast_temperature if per_step_contrast_temperature is not None else cfg.per_step_contrast_temperature
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

    # ── tensor building ──

    @staticmethod
    def _pad_or_trunc(ids: list[int], target_len: int, pad_id: int) -> list[int]:
        if len(ids) >= target_len:
            return ids[:target_len]
        return ids + [pad_id] * (target_len - len(ids))

    def _build_tensors(self, batch: list[dict], device: torch.device) -> dict:
        """Build batched tensors for one batch of M chunks.

        Each chunk has: fact_passages_token_ids (list of 8), question_token_ids,
        answer_token_ids, target_idx.
        """
        cfg = self.model.cfg
        T = cfg.T_window
        M = len(batch)
        pad_id = self.pad_token_id

        # Passages: [M, n_facts, T]. Init with pad_id (not 0) so any
        # unfilled rows are masked out by `(passage != pad_id)`
        # downstream — if a chunk ever returns < n_facts, the missing
        # rows look like pure padding instead of all-zero "real" tokens.
        n_facts = cfg.n_facts_per_chunk
        passages = torch.full((M, n_facts, T), pad_id, dtype=torch.long, device=device)
        for m, chunk in enumerate(batch):
            passages_in = chunk["fact_passages_token_ids"]
            assert len(passages_in) == n_facts, (
                f"expected {n_facts} passages per chunk, got {len(passages_in)}"
            )
            for i, p in enumerate(passages_in):
                passages[m, i] = torch.tensor(
                    self._pad_or_trunc(p, T, pad_id), device=device,
                )

        # Question + answer concatenated. The answer mask gates which
        # positions contribute to the CE loss.
        #
        # When the sampler provides `answer_content_token_positions` (offsets
        # within `answer_token_ids` that carry the actual memory-load-bearing
        # content — e.g. "doctor" inside the templated "X works as a doctor"),
        # we mask ONLY those positions. Otherwise the loss trains the
        # template scaffold via plain LM, letting the model reach high
        # accuracy without ever consulting memory.
        #
        # Fall back to the full answer span when content positions are
        # missing or empty (older datasets, ambiguous answers).
        qa_seqs = []
        answer_masks = []
        question_lens = []
        for chunk in batch:
            q = chunk["question_token_ids"]
            a = chunk["answer_token_ids"]
            qa = q + a
            qa = self._pad_or_trunc(qa, T, pad_id)
            qa_seqs.append(qa)

            content_positions = chunk.get("answer_content_token_positions") or []
            if content_positions:
                mask = [False] * T
                for offset in content_positions:
                    qa_pos = len(q) + offset
                    if 0 <= qa_pos < T:
                        mask[qa_pos] = True
            else:
                mask = (
                    [False] * len(q)
                    + [True] * len(a)
                    + [False] * max(0, T - len(q) - len(a))
                )[:T]
            answer_masks.append(mask)
            question_lens.append(min(len(q), T))

        qa = torch.tensor(qa_seqs, dtype=torch.long, device=device)
        answer_mask = torch.tensor(answer_masks, dtype=torch.bool, device=device)

        # Question-only token ids for the zero-memory Llama forward
        max_q = max(question_lens) if question_lens else 1
        max_q = max(max_q, 1)
        question_ids = torch.full(
            (M, max_q), pad_id, dtype=torch.long, device=device,
        )
        for m, chunk in enumerate(batch):
            ql = question_lens[m]
            if ql > 0:
                question_ids[m, :ql] = torch.tensor(
                    chunk["question_token_ids"][:ql], device=device,
                )

        target_idxs = torch.tensor(
            [c["target_idx"] for c in batch], dtype=torch.long, device=device,
        )

        return {
            "passages": passages,
            "qa": qa,
            "answer_mask": answer_mask,
            "question_ids": question_ids,
            "question_lens": question_lens,
            "target_idxs": target_idxs,
        }

    # ── question-conditioning forward (zero-memory) ──

    def _compute_question_hiddens(
        self, question_ids: Tensor, question_lens: list[int],
    ) -> Tensor:
        """Run Llama over question tokens with memory disabled, return hiddens."""
        if self.model.llama is None:
            BS, T = question_ids.shape
            return torch.randn(
                BS, T, self.model.cfg.d_lm, device=question_ids.device,
            )
        mem_inject = self.model._mem_inject_layer()
        saved_fn = mem_inject.memory_fn
        # Use zero-readout closure
        d_mem = self.model.cfg.D_concept
        def zero_readout(h_mem: Tensor) -> Tensor:
            return torch.zeros(
                h_mem.shape[0], h_mem.shape[1], d_mem,
                device=h_mem.device, dtype=h_mem.dtype,
            )
        mem_inject.memory_fn = zero_readout
        try:
            q_mask = (question_ids != self.pad_token_id)
            base_out = self.model.llama.model(
                input_ids=question_ids,
                attention_mask=q_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            q_hiddens = base_out.last_hidden_state.float()
        finally:
            mem_inject.memory_fn = saved_fn
        return q_hiddens

    # ── contrastive losses ──

    def _entry_contrastive(
        self,
        passage_hiddens_per_window: list[Tensor],   # 8 × [M, T, d_lm]
        passage_masks_per_window: list[Tensor],     # 8 × [M, T] bool
        q_hiddens: Tensor,                           # [M, q_len, d_lm]
        q_mask: Tensor,                              # [M, q_len] bool
        target_idxs: Tensor,                         # [M] long
    ) -> Tensor:
        """InfoNCE on entry-projector outputs between matched (Q, P) pairs.

        Pool is MASKED — unmasked mean would average in pad-token hidden
        states (~80% of the pool for a 50-token passage padded to T=256),
        which would teach the projector to align padding noise.
        """
        M = target_idxs.shape[0]

        def _masked_mean(h: Tensor, m: Tensor) -> Tensor:
            mf = m.to(h.dtype).unsqueeze(-1)               # [M, T, 1]
            return (h * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)

        p_pool_list = [
            _masked_mean(h, m)
            for h, m in zip(passage_hiddens_per_window, passage_masks_per_window)
        ]
        p_pools = torch.stack(p_pool_list, dim=1)          # [M, 8, d_lm]
        q_pool = _masked_mean(q_hiddens, q_mask)           # [M, d_lm]

        # Project via shared entry_proj to D_concept; mean over J
        d_lm = p_pools.shape[-1]
        n_facts = self.model.cfg.n_facts_per_chunk
        p_qentry = self.model.entry_proj(p_pools.reshape(M * n_facts, d_lm))  # [M*N, J, D]
        p_pool_D = p_qentry.mean(dim=1)                                       # [M*N, D]
        q_qentry = self.model.entry_proj(q_pool)                              # [M, J, D]
        q_pool_D = q_qentry.mean(dim=1)                                       # [M, D]

        # L2-normalize for cosine-based InfoNCE. Routing uses RMS-norm
        # internally, which differs from L2 only by a uniform magnitude
        # factor (sqrt(D)) — the *direction* of the projector's output is
        # what both losses ultimately compare, so cosine similarity here
        # is the right pairing geometry. (Using rms_norm here would
        # produce O(D)-magnitude logits that saturate the softmax.)
        q_n = F.normalize(q_pool_D, dim=-1)
        p_n = F.normalize(p_pool_D, dim=-1)
        S = (q_n @ p_n.T) / self.contrast_temperature                    # [M, M*N]
        labels = torch.arange(M, device=S.device) * n_facts + target_idxs
        return F.cross_entropy(S, labels)

    def _per_step_contrastive(
        self,
        read_step_queries: Tensor,                  # [M, J, K_read, D]
        target_write_step_queries: Tensor,          # [M, J, K_write, D]
    ) -> Tensor:
        """InfoNCE at each hop k between read and target's write step_queries.
        Mean of per-hop losses."""
        M, J, K_r, D = read_step_queries.shape
        K_w = target_write_step_queries.shape[2]
        K = min(K_r, K_w)
        if K == 0:
            return torch.zeros((), device=read_step_queries.device,
                               dtype=read_step_queries.dtype)
        losses = []
        for k in range(K):
            r_pool = read_step_queries[:, :, k, :].mean(dim=1)           # [M, D]
            p_pool = target_write_step_queries[:, :, k, :].mean(dim=1)   # [M, D]
            r_n = F.normalize(r_pool, dim=-1)
            p_n = F.normalize(p_pool, dim=-1)
            S = (r_n @ p_n.T) / self.per_step_contrast_temperature       # [M, M]
            labels = torch.arange(M, device=S.device)
            losses.append(F.cross_entropy(S, labels))
        return torch.stack(losses).mean()

    @staticmethod
    @torch.no_grad()
    def _rw_overlap(read_ids: Tensor, target_write_ids: Tensor) -> float:
        """Per-chunk overlap, mean across chunks."""
        M = read_ids.shape[0]
        accum = 0.0
        n = 0
        for b in range(M):
            r_set = set(read_ids[b].flatten().tolist())
            w_set = set(target_write_ids[b].flatten().tolist())
            if r_set:
                accum += len(r_set & w_set) / len(r_set)
                n += 1
        return accum / max(n, 1)

    @staticmethod
    def _unique_metrics(visited_ids: Tensor) -> tuple[float, float]:
        """visited_ids: [M, J, K]. Returns (per_traj_mean, per_window_mean) — count of unique cells."""
        M, J, K = visited_ids.shape
        per_traj = 0.0
        per_window = 0.0
        for b in range(M):
            window_set: set = set()
            for j in range(J):
                traj_set = set(visited_ids[b, j].tolist())
                per_traj += len(traj_set)
                window_set |= traj_set
            per_window += len(window_set)
        return per_traj / max(M * J, 1), per_window / max(M, 1)

    @staticmethod
    def _visited_entropy(visited_ids: Tensor, N: int) -> float:
        """Empirical entropy of the entry-cell distribution (step 0) across batch × J.
        0 = collapsed to one cell; log(N) = perfectly uniform.
        """
        entry = visited_ids[:, :, 0].flatten()
        counts = torch.bincount(entry, minlength=N).float()
        p = counts / counts.sum().clamp_min(1.0)
        nz = p > 0
        return float(-(p[nz] * p[nz].log()).sum())

    @staticmethod
    def _bank_health(concept_ids: Tensor, sample: int = 128) -> tuple[float, float, float]:
        """Returns (norm_mean, norm_cv, pairwise_cos_mean).
        Subsamples `sample` concept_ids to keep pairwise cost bounded.
        """
        N, D = concept_ids.shape
        with torch.no_grad():
            norms = concept_ids.norm(dim=-1)
            norm_mean = float(norms.mean())
            norm_cv = float(norms.std() / norms.mean().clamp_min(1e-6))
            idx = torch.randperm(N, device=concept_ids.device)[:min(sample, N)]
            sub = concept_ids[idx]
            sub_n = sub / sub.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cos = sub_n @ sub_n.t()
            mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
            pairwise = float(cos[mask].mean())
        return norm_mean, norm_cv, pairwise

    def _per_module_grad_norms(self) -> dict[str, float]:
        """Group grad norms by module. Call AFTER backward, BEFORE optimizer.step.

        Iteration order matters — more-specific groups precede less-specific.
        `lambda_edge` lives under both `read_module.walker` and
        `write_module.walker`; we want it counted in its own bucket, not
        absorbed into the read/write totals.

        `mem_inject` params live at `llama.model.layers.{inject_layer}.{W_in,
        W_out,scale_raw}` (MemInjectLayer replaced a base Llama layer in
        place). Vanilla Llama layers don't have those attribute names, so
        substring match is safe.
        """
        # Order matters: first match wins.
        groups = (
            ("lambda_edge",  ("walker.lambda_edge",)),
            ("entry_proj",   ("entry_proj.",)),
            ("concept_ids",  ("manifold.id_basis", "manifold.id_proj")),
            ("mem_inject",   (".W_in.", ".W_out.", ".scale_raw")),
            ("read_attn",    ("read_attn.",)),
            ("read",         ("read_module.",)),
            ("write",        ("write_module.",)),
        )
        sums: dict[str, float] = {k: 0.0 for k, _ in groups}
        for n, p in self.model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            g2 = float(p.grad.detach().pow(2).sum())
            for grp, prefixes in groups:
                if any(pf in n for pf in prefixes):
                    sums[grp] += g2
                    break
        return {k: math.sqrt(v) for k, v in sums.items()}

    # ── training step ──

    def step(self, batch: list[dict]) -> V2RetrievalMetrics:
        """One gradient update over a batch of M chunks."""
        cfg = self.model.cfg
        M = len(batch)
        device = next(self.model.parameters()).device
        T = cfg.T_window

        tensors = self._build_tensors(batch, device)
        passages = tensors["passages"]                              # [M, 8, T]
        qa = tensors["qa"]                                          # [M, T]
        answer_mask = tensors["answer_mask"]                        # [M, T]
        question_ids = tensors["question_ids"]
        question_lens = tensors["question_lens"]
        target_idxs = tensors["target_idxs"]                        # [M]

        self.optimizer.zero_grad(set_to_none=True)

        # Increment global step counter (used by manifold's MIN_AGE protection)
        self.model.manifold.advance_step()

        # ── 1. Run 8 passage windows (writes) ────────────────────
        passage_hiddens_per_window: list[Tensor] = []
        passage_masks_per_window: list[Tensor] = []
        write_visited_per_fact: list[Tensor] = []
        write_step_queries_per_fact: list[Tensor] = []
        aux_lb_acc = torch.zeros((), device=device)
        aux_z_acc = torch.zeros((), device=device)

        # Wave 1 writes n_facts INDEPENDENT facts; each passage's write
        # must NOT see the prior fact's memory. Pass
        # prev_window_hiddens=None so mem_inject is zero-readout for
        # fact-write windows. (Streaming Wave 2 keeps the carry-over by
        # design — there the prior context is *what we're trying to
        # retrieve from*.)
        n_facts = cfg.n_facts_per_chunk
        for i in range(n_facts):
            passage_ids_i = passages[:, i, :]
            passage_mask_i = (passage_ids_i != self.pad_token_id)
            out = self.model.forward_window(
                lm_input_ids=passage_ids_i,
                prev_window_hiddens=None,
                attention_mask=passage_mask_i,
                prev_attention_mask=None,
                hard_routing=True,
                write_mode="passage",
            )
            passage_hiddens_per_window.append(out["current_hiddens"])
            passage_masks_per_window.append(passage_mask_i)
            write_visited_per_fact.append(out["write_visited_ids"].detach())
            write_step_queries_per_fact.append(out["write_step_queries"])
            aux_lb_acc = aux_lb_acc + out["aux_load_balance"]
            aux_z_acc = aux_z_acc + out["aux_z_loss"]

        # ── 2. Question-conditioned read forward ─────────────────
        q_hiddens = self._compute_question_hiddens(question_ids, question_lens)
        q_mask = (question_ids != self.pad_token_id)
        qa_mask = (qa != self.pad_token_id)

        out_qa = self.model.forward_window(
            lm_input_ids=qa,
            prev_window_hiddens=None,           # read_conditioning_hiddens wins
            attention_mask=qa_mask,
            prev_attention_mask=None,
            read_conditioning_hiddens=q_hiddens,
            read_conditioning_mask=q_mask,
            hard_routing=True,
            write_mode="qa",
        )
        aux_lb_acc = aux_lb_acc + out_qa["aux_load_balance"]
        aux_z_acc = aux_z_acc + out_qa["aux_z_loss"]
        # Average aux over n_facts writes + 1 read
        n_walker_calls = n_facts + 1
        aux_lb_acc = aux_lb_acc / float(n_walker_calls)
        aux_z_acc = aux_z_acc / float(n_walker_calls)

        # ── 3. Answer loss (NTP on answer tokens) ─────────────────
        logits = out_qa["logits"]                                   # [M, T, V]
        V = logits.shape[-1]
        shift_logits = logits[:, :-1, :]
        shift_targets = qa[:, 1:]
        shift_mask = answer_mask[:, 1:]
        per_tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_targets.reshape(-1),
            reduction="none",
        ).reshape(M, T - 1)
        n_answer = shift_mask.float().sum().clamp_min(1.0)
        answer_loss = (per_tok_ce * shift_mask.float()).sum() / n_answer

        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_targets) & shift_mask
            answer_acc = float(correct.sum().float() / n_answer)

        # ── 4. Contrastive losses ────────────────────────────────
        l_contrast_entry = self._entry_contrastive(
            passage_hiddens_per_window,
            passage_masks_per_window,
            q_hiddens,
            q_mask,
            target_idxs,
        )
        # Per-step contrastive on step_queries (matches read at QA window
        # with the TARGET fact's write step_queries)
        write_sq_stack = torch.stack(write_step_queries_per_fact, dim=1)  # [M, 8, J, K_w, D]
        target_write_sq = write_sq_stack[
            torch.arange(M, device=device), target_idxs,
        ]  # [M, J, K_w, D]
        l_contrast_per_step = self._per_step_contrastive(
            out_qa["read_step_queries"], target_write_sq,
        )

        # ── 5. Total loss ────────────────────────────────────────
        total_loss = (
            answer_loss
            + self.load_balance_coef * aux_lb_acc
            + self.z_loss_coef * aux_z_acc
            + self.contrast_coef * l_contrast_entry
            + self.per_step_contrast_coef * l_contrast_per_step
        )

        # ── 6. Backward ──────────────────────────────────────────
        total_loss.backward()

        # Per-module grad norms BEFORE clip (clip scales, doesn't erase, but
        # we want the unclipped magnitude for diagnostics).
        per_module_gn = self._per_module_grad_norms()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
        else:
            with torch.no_grad():
                grad_norm = torch.norm(torch.stack(
                    [p.grad.norm() for p in trainable_params if p.grad is not None]
                ))

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._step_count += 1

        # ── 7. Diagnostics ───────────────────────────────────────
        edge_stats = self.model.manifold.edge_stats()

        read_visited = out_qa["read_visited_ids"]                    # [M, J, K_r]
        wv_stack = torch.stack(write_visited_per_fact, dim=1)        # [M, 8, J, K_w]
        target_wv = wv_stack[torch.arange(M, device=device), target_idxs]  # [M, J, K_w]

        # R∩W overlap variants
        rw_target = self._rw_overlap(read_visited, target_wv)
        rw_all    = self._rw_overlap(read_visited, wv_stack.flatten(1, 2))
        rw_entry  = self._rw_overlap(read_visited[:, :, :1], target_wv[:, :, :1])
        rw_hop    = self._rw_overlap(read_visited[:, :, 1:], target_wv[:, :, 1:])

        # Per-window unique cells
        r_per_traj, r_per_win = self._unique_metrics(read_visited)
        # Use the TARGET fact's writes for write-side unique (most relevant)
        w_per_traj, w_per_win = self._unique_metrics(target_wv)

        # Entry routing entropy
        N = cfg.N
        r_ent_entropy = self._visited_entropy(read_visited, N)
        w_ent_entropy = self._visited_entropy(target_wv, N)

        # Bank health on concept_ids (project id_basis through id_proj)
        with torch.no_grad():
            concept_ids = self.model.manifold.id_proj(self.model.manifold.id_basis)
        bank_norm_mean, bank_norm_cv, bank_pairwise = self._bank_health(concept_ids)

        return V2RetrievalMetrics(
            loss=float(total_loss.detach()),
            answer_loss=float(answer_loss.detach()),
            answer_token_count=int(n_answer.item()),
            aux_load_balance=float(aux_lb_acc.detach()),
            aux_z_loss=float(aux_z_acc.detach()),
            grad_norm=float(grad_norm.detach()) if isinstance(grad_norm, Tensor) else float(grad_norm),
            answer_acc=answer_acc,
            l_contrast_entry=float(l_contrast_entry.detach()),
            l_contrast_per_step=float(l_contrast_per_step.detach()),
            n_active_edges=edge_stats["n_active_edges"],
            edge_active_fraction=edge_stats["active_fraction"],
            mean_fan_out=edge_stats["mean_fan_out"],
            mean_edge_state_norm=edge_stats["mean_state_norm"],
            mean_edge_specificity=edge_stats["mean_specificity"],
            mean_visit_count=edge_stats["mean_visit_count"],
            mean_edge_age=edge_stats["mean_age"],
            entry_logits_max=float(out_qa.get("read_entry_logits_max", 0.0)),
            edge_score_active_frac=0.0,  # TODO: surface from walker
            rw_overlap=rw_target,
            rw_overlap_target=rw_target,
            rw_overlap_all=rw_all,
            rw_overlap_entry=rw_entry,
            rw_overlap_hop=rw_hop,
            r_unique_per_traj=r_per_traj,
            w_unique_per_traj=w_per_traj,
            r_unique_per_window=r_per_win,
            w_unique_per_window=w_per_win,
            read_entry_entropy=r_ent_entropy,
            write_entry_entropy=w_ent_entropy,
            grad_norm_read=per_module_gn["read"],
            grad_norm_write=per_module_gn["write"],
            grad_norm_entry_proj=per_module_gn["entry_proj"],
            grad_norm_lambda_edge=per_module_gn["lambda_edge"],
            grad_norm_concept_ids=per_module_gn["concept_ids"],
            grad_norm_mem_inject=per_module_gn["mem_inject"],
            grad_norm_read_attn=per_module_gn["read_attn"],
            concept_ids_norm_mean=bank_norm_mean,
            concept_ids_norm_cv=bank_norm_cv,
            concept_ids_pairwise_cos=bank_pairwise,
        )

    @torch.no_grad()
    def eval_step(self, batch: list[dict]) -> V2RetrievalMetrics:
        """Validation step. Same write+read protocol as train, but:
        - No backprop, no optimizer step, no step_counter advance.
        - hard_routing=False (argmax — STE not needed in eval).
        - Computes per-task loss/acc breakdown via the chunk's metadata.
        - Snapshot/restore manifold edge buffers around the call so val
          writes don't contaminate training memory.
        - Passes the same attention masks as the train step.
        """
        cfg = self.model.cfg
        M = len(batch)
        device = next(self.model.parameters()).device
        T = cfg.T_window

        tensors = self._build_tensors(batch, device)
        passages = tensors["passages"]
        qa = tensors["qa"]
        answer_mask = tensors["answer_mask"]
        question_ids = tensors["question_ids"]
        question_lens = tensors["question_lens"]
        target_idxs = tensors["target_idxs"]

        # Snapshot training memory — val writes mutate buffers in-place.
        snap = self.model.manifold.snapshot_edge_state()
        try:
            # Run n_facts INDEPENDENT writes (same as train)
            write_visited_per_fact: list[Tensor] = []
            for i in range(cfg.n_facts_per_chunk):
                passage_ids_i = passages[:, i, :]
                passage_mask_i = (passage_ids_i != self.pad_token_id)
                out = self.model.forward_window(
                    lm_input_ids=passage_ids_i,
                    prev_window_hiddens=None,
                    attention_mask=passage_mask_i,
                    prev_attention_mask=None,
                    hard_routing=False,
                    write_mode="passage",
                )
                write_visited_per_fact.append(out["write_visited_ids"])

            # Question-conditioned read forward
            q_hiddens = self._compute_question_hiddens(question_ids, question_lens)
            q_mask = (question_ids != self.pad_token_id)
            qa_mask = (qa != self.pad_token_id)
            out_qa = self.model.forward_window(
                lm_input_ids=qa,
                prev_window_hiddens=None,
                attention_mask=qa_mask,
                prev_attention_mask=None,
                read_conditioning_hiddens=q_hiddens,
                read_conditioning_mask=q_mask,
                hard_routing=False,
                write_mode="qa",
            )
        finally:
            # Restore training memory regardless of any error inside the try.
            self.model.manifold.restore_edge_state(snap)

        # Answer loss + accuracy
        logits = out_qa["logits"]
        V = logits.shape[-1]
        shift_logits = logits[:, :-1, :]
        shift_targets = qa[:, 1:]
        shift_mask = answer_mask[:, 1:]
        per_tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_targets.reshape(-1),
            reduction="none",
        ).reshape(M, T - 1)
        per_chunk_tok_count = shift_mask.float().sum(dim=1).clamp_min(1.0)
        per_chunk_loss = (per_tok_ce * shift_mask.float()).sum(dim=1) / per_chunk_tok_count
        n_answer = shift_mask.float().sum().clamp_min(1.0)
        answer_loss = (per_tok_ce * shift_mask.float()).sum() / n_answer

        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_targets) & shift_mask
        per_chunk_acc = correct.float().sum(dim=1) / per_chunk_tok_count
        answer_acc = float(correct.sum().float() / n_answer)

        # Per-task breakdown by (entity_class, attribute)
        per_key_loss: dict[str, float] = {}
        per_key_acc: dict[str, float] = {}
        per_key_n: dict[str, int] = {}
        for m, chunk in enumerate(batch):
            meta = chunk.get("metadata", {})
            key = f"{meta.get('target_entity_class', '?')}.{meta.get('target_attribute', '?')}"
            n = per_key_n.get(key, 0)
            ploss = per_key_loss.get(key, 0.0)
            pacc = per_key_acc.get(key, 0.0)
            per_key_loss[key] = (ploss * n + float(per_chunk_loss[m])) / (n + 1)
            per_key_acc[key] = (pacc * n + float(per_chunk_acc[m])) / (n + 1)
            per_key_n[key] = n + 1

        # RW overlap
        read_visited = out_qa["read_visited_ids"]
        wv_stack = torch.stack(write_visited_per_fact, dim=1)
        target_wv = wv_stack[torch.arange(M, device=device), target_idxs]
        rw_overlap = self._rw_overlap(read_visited, target_wv)

        edge_stats = self.model.manifold.edge_stats()

        return V2RetrievalMetrics(
            loss=float(answer_loss),
            answer_loss=float(answer_loss),
            answer_token_count=int(n_answer.item()),
            answer_acc=answer_acc,
            n_active_edges=edge_stats["n_active_edges"],
            edge_active_fraction=edge_stats["active_fraction"],
            mean_fan_out=edge_stats["mean_fan_out"],
            mean_edge_state_norm=edge_stats["mean_state_norm"],
            mean_edge_specificity=edge_stats["mean_specificity"],
            mean_visit_count=edge_stats["mean_visit_count"],
            mean_edge_age=edge_stats["mean_age"],
            rw_overlap=rw_overlap,
            per_key_loss=per_key_loss,
            per_key_acc=per_key_acc,
            per_key_n=per_key_n,
        )
