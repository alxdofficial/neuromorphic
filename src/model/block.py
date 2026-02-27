"""
ColumnBlock (v4) — block of C cortical columns with shared PM + EM.

One ColumnBlock per memory block (B_blocks total). Each block owns:
- CorticalColumnGroup (C columns, batched via GroupedLinear)
- ProceduralMemory (shared across columns)
- EpisodicMemory (shared across columns)
- PM/EM neuromodulators
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .column import CorticalColumnGroup
from .procedural_memory import ProceduralMemory, PMNeuromodulator
from .episodic_memory import EpisodicMemory, EMNeuromodulator
from .utils import unit_normalize


class ColumnBlock(nn.Module):
    """Block of C cortical columns with shared PM + EM."""

    def __init__(self, block_idx: int, config: ModelConfig):
        super().__init__()
        self.block_idx = block_idx
        self.config = config

        self.columns = CorticalColumnGroup(config)
        self.pm = ProceduralMemory(config.D_mem, config.r, config)
        self.em = EpisodicMemory(config.D_mem, config.M, config)
        self.pm_neuromod = PMNeuromodulator(config)
        self.em_neuromod = EMNeuromodulator(config)

    def forward_pass(self, x_block: Tensor, z_hat_prev: Tensor | None):
        """One refinement pass: all C columns * all N tokens.

        x_block: [BS, N, C, D_col]
        z_hat_prev: [BS, N, C, D_pcm] or None

        Returns: x_out, z, z_hat, pcm_loss, pm_elig, em_cands
        """
        x_out, z, z_hat, surprise, elig_info, nov_info = \
            self.columns.forward(x_block, self.pm, self.em, z_hat_prev)

        # Aggregate PM eligibility across N tokens and C columns
        k_cand, v_cand, gate = elig_info

        if self.config.pm_enabled and self.pm.is_initialized():
            # Route eligibility to PM slots
            k_norm = unit_normalize(k_cand)
            # pm_K: [BS, r, D_mem], k_norm: [BS, N, C, D_mem]
            route_scores = torch.einsum(
                "bncd, brd -> bncr", k_norm, self.pm.pm_K
            )  # [BS, N, C, r]
            route_w = torch.softmax(
                route_scores / self.config.tau_route_pm, dim=-1
            )  # [BS, N, C, r]

            # Gate and aggregate
            gated_routed = gate.unsqueeze(-1) * route_w  # [BS, N, C, r]
            elig_K = torch.einsum("bncr, bncd -> brd", gated_routed, k_cand)
            elig_V = torch.einsum("bncr, bncd -> brd", gated_routed, v_cand)
        else:
            elig_K = torch.zeros(
                x_block.shape[0], self.config.r, self.config.D_mem,
                device=x_block.device, dtype=x_block.dtype,
            )
            elig_V = torch.zeros_like(elig_K)

        # Score EM novelty and select top candidates
        q_nov, v_nov, w_nov, surp = nov_info
        if self.config.em_enabled and self.em.is_initialized():
            novelty = self.em.score_novelty(q_nov, surp, w_nov)
            em_cands = self.em.select_top_candidates(
                q_nov, v_nov, novelty, self.config.C_em
            )
        else:
            C_em = self.config.C_em
            BS = x_block.shape[0]
            em_cands = (
                torch.zeros(BS, C_em, self.config.D_mem, device=x_block.device, dtype=x_block.dtype),
                torch.zeros(BS, C_em, self.config.D_mem, device=x_block.device, dtype=x_block.dtype),
                torch.zeros(BS, C_em, device=x_block.device, dtype=x_block.dtype),
            )

        # PCM prediction loss
        pcm_loss = torch.tensor(0.0, device=x_block.device)
        if self.config.pcm_enabled and z_hat_prev is not None and z is not None:
            pcm_loss = self.columns.pcm.prediction_loss(z_hat_prev, z)

        return x_out, z, z_hat, pcm_loss, (elig_K, elig_V), em_cands

    def commit_and_write(self, elig, em_cands):
        """PM commit + EM write. Called between passes."""
        elig_K, elig_V = elig
        cand_K, cand_V, cand_scores = em_cands
        BS = elig_K.shape[0]
        device = elig_K.device

        # PM commit
        if self.config.pm_enabled and self.pm.is_initialized():
            self.pm.base_decay()
            elig_summary = elig_K.norm(dim=-1).mean(dim=-1)  # [BS]
            pm_usage = self.pm.pm_a.sum(dim=-1)  # [BS]
            content_emb = elig_K.mean(dim=1)  # [BS, D_mem]
            g, slot_logits, tau = self.pm_neuromod(
                elig_summary.detach(), pm_usage.detach(), content_emb.detach()
            )
            self.pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # EM write
        if self.config.em_enabled and self.em.is_initialized():
            em_usage = self.em.em_S.sum(dim=-1)  # [BS]
            novelty_mean = cand_scores.mean(dim=-1)  # [BS]
            content_emb = cand_K.mean(dim=1)  # [BS, D_mem]
            g_em, tau_em, decay = self.em_neuromod(
                novelty_mean.detach(), em_usage.detach(), content_emb.detach()
            )
            self.em.write(cand_K, cand_V, cand_scores, g_em, tau_em, decay)
            self.em.age_tick(self.config.N)

    def initialize_states(self, BS: int, device: torch.device, dtype: torch.dtype):
        self.pm.initialize(BS, device, dtype)
        self.em.initialize(BS, device, dtype)
