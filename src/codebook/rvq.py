"""Residual VQ-VAE for discretizing the neuromodulator action space.

Used between phase 1 and phase 2 of the two-phase training strategy:
- Phase 1 collects continuous modulator actions into an action database.
- This module fits a codebook on that database.
- Phase 2 uses the codebook to turn the modulator's continuous output into a
  categorical policy over codes, trained with GRPO.

Design choices (see docs/training_strategy.md):
- Residual VQ (not flat VQ) — each level models the residual left by previous
  levels. Avoids the single-quantizer collapse that kills large flat codebooks.
- Small per-level codebooks (16 codes × 4 levels = 65,536 effective vocab).
- Low latent dim (32) — avoids degenerate nearest-neighbor in high dim.
- EMA codebook updates (decay 0.99), dead-code resampling from current batch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualVQ(nn.Module):
    """Residual Vector Quantizer with EMA codebook updates.

    Each level operates on the residual left by previous levels. All
    codebooks live in the same low-dim latent space as the encoder output.

    Attributes
    ----------
    codebooks : [num_levels, codes_per_level, code_dim]  buffer
        The current codebook entries. Updated via EMA during training, not
        via gradient descent.
    cluster_size : [num_levels, codes_per_level]  buffer
        Running count of assignments per code, for EMA normalization.
    embed_avg : [num_levels, codes_per_level, code_dim]  buffer
        Running sum of encoder outputs per code, for EMA normalization.
    """

    def __init__(
        self,
        num_levels: int = 4,
        codes_per_level: int = 16,
        code_dim: int = 32,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 0.01,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codes_per_level = codes_per_level
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        # Init with unit-norm random vectors (per-code L2 norm ~1 in
        # expectation). Principled dimension-scaled init: std = 1/sqrt(d).
        codebooks = torch.randn(
            num_levels, codes_per_level, code_dim) * (code_dim ** -0.5)
        self.register_buffer("codebooks", codebooks)
        self.register_buffer(
            "cluster_size", torch.zeros(num_levels, codes_per_level))
        self.register_buffer("embed_avg", codebooks.clone())

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize z through all residual levels (no grad, no EMA update).

        z: [B, code_dim]
        Returns:
            z_q: [B, code_dim] — reconstructed sum of selected codes
            codes: [B, num_levels] int — selected code index per level
        """
        residual = z
        codes_list = []
        z_q = torch.zeros_like(z)
        for lvl in range(self.num_levels):
            cb = self.codebooks[lvl]  # [K, D]
            dists = (residual.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)  # [B, K]
            codes_lvl = dists.argmin(dim=1)  # [B]
            codes_list.append(codes_lvl)
            selected = cb[codes_lvl]  # [B, D]
            z_q = z_q + selected
            residual = residual - selected
        codes = torch.stack(codes_list, dim=1)  # [B, num_levels]
        return z_q, codes

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass with straight-through gradient.

        z: [B, code_dim]
        Returns:
            z_q_st: [B, code_dim] — quantized with straight-through gradient
            codes: [B, num_levels]
            commit_loss: scalar — commitment loss for encoder training
        """
        z_q, codes = self.quantize(z)
        # Straight-through: forward uses quantized, backward passes through.
        z_q_st = z + (z_q - z).detach()
        commit_loss = F.mse_loss(z_q.detach(), z)
        return z_q_st, codes, commit_loss

    @torch.no_grad()
    def ema_update(self, z: Tensor, codes: Tensor):
        """Update codebooks via EMA.

        Should be called during training, after `forward`, with the encoder
        output z and the selected codes. Each level's codebook is updated
        based on the residual seen at that level.

        Important: residuals must advance using the *pre-update* codebook
        entries (those that the codes were originally assigned against). We
        cache the selected entries before updating to preserve this.
        """
        residual = z.detach()
        for lvl in range(self.num_levels):
            codes_lvl = codes[:, lvl]
            # Cache the pre-update codebook entries for residual advancement.
            old_selected = self.codebooks[lvl][codes_lvl].clone()

            one_hot = F.one_hot(codes_lvl, self.codes_per_level).to(residual.dtype)
            cluster_size_new = one_hot.sum(0)
            embed_sum_new = one_hot.t() @ residual

            self.cluster_size[lvl].mul_(self.decay).add_(
                cluster_size_new, alpha=1 - self.decay)
            self.embed_avg[lvl].mul_(self.decay).add_(
                embed_sum_new, alpha=1 - self.decay)

            n = self.cluster_size[lvl].sum()
            smoothed = (self.cluster_size[lvl] + self.eps) / (
                n + self.codes_per_level * self.eps) * n
            new_codes = self.embed_avg[lvl] / smoothed.unsqueeze(1)
            self.codebooks[lvl].copy_(new_codes)

            # Advance residual using the PRE-update entry, not the post-update one.
            residual = residual - old_selected

    @torch.no_grad()
    def resample_dead_codes(self, z: Tensor):
        """Replace codes with < dead_code_threshold usage fraction by random
        encoder outputs from the current batch. Called periodically during
        training, not every step.

        For multi-level RVQ, each level sees the residual after subtracting
        previous levels' contributions, so dead codes at level L are
        resampled from level-L residuals (not raw z).
        """
        B = z.shape[0]
        residual = z.clone()
        for lvl in range(self.num_levels):
            total = self.cluster_size[lvl].sum().clamp(min=1.0)
            usage_frac = self.cluster_size[lvl] / total
            dead = (usage_frac < self.dead_code_threshold).nonzero(as_tuple=True)[0]
            if dead.numel() > 0:
                # Sample from current-level residuals, not raw z.
                idx = torch.randint(0, B, (dead.numel(),), device=z.device)
                self.codebooks[lvl, dead] = residual[idx].detach()
                self.embed_avg[lvl, dead] = residual[idx].detach()
                self.cluster_size[lvl, dead] = 1.0
            # Advance residual to next level using nearest-code assignment.
            dists = (residual.unsqueeze(1) - self.codebooks[lvl].unsqueeze(0)).pow(2).sum(-1)
            nearest = dists.argmin(dim=1)
            residual = residual - self.codebooks[lvl][nearest]

    @torch.no_grad()
    def usage_histogram(self) -> Tensor:
        """Return per-level code usage fractions, [num_levels, codes_per_level]."""
        totals = self.cluster_size.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.cluster_size / totals

    @torch.no_grad()
    def sample_codes(
        self, z: Tensor, tau: float = 1.0, sample: bool = True,
    ) -> Tensor:
        """Sample per-level codes via distance-based categorical.

        At each level, compute logits = -|residual - codebook|^2 / tau,
        sample a code from the categorical, and subtract the selected code
        from the residual before moving to the next level.

        Args:
            z: [B, code_dim] encoder output
            tau: categorical temperature (tau -> 0 = argmax, tau -> inf = uniform)
            sample: if True, multinomial sample; if False, argmax (deterministic)

        Returns:
            codes: [B, num_levels] long tensor of selected code indices
        """
        residual = z
        codes_list = []
        for lvl in range(self.num_levels):
            cb = self.codebooks[lvl]
            dists = (residual.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)
            logits = -dists / tau
            if sample:
                probs = F.softmax(logits, dim=-1)
                codes_lvl = torch.multinomial(probs, 1).squeeze(-1)
            else:
                codes_lvl = logits.argmax(dim=-1)
            codes_list.append(codes_lvl)
            selected = cb[codes_lvl]
            residual = residual - selected
        return torch.stack(codes_list, dim=1)

    def log_prob(
        self, z: Tensor, codes: Tensor, tau: float = 1.0,
    ) -> Tensor:
        """Compute log pi(codes | z) under the per-level distance-based categorical.

        This is the policy-gradient objective's log-pi term. Gradients flow
        through z (not through codes). Codebook buffers are frozen at call
        time — phase 2 does not update the codebook.

        Args:
            z: [B, code_dim] encoder output (requires_grad if upstream)
            codes: [B, num_levels] long — the sampled codes to score
            tau: categorical temperature (same as used during sampling)

        Returns:
            log_pi: [B] — sum of per-level log categorical probs
        """
        B = z.shape[0]
        residual = z
        log_pi_total = torch.zeros(B, device=z.device, dtype=z.dtype)
        for lvl in range(self.num_levels):
            cb = self.codebooks[lvl].detach()  # codebook is frozen in phase 2
            dists = (residual.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)  # [B, K]
            logits = -dists / tau
            log_probs = F.log_softmax(logits, dim=-1)
            codes_lvl = codes[:, lvl]
            log_pi_lvl = log_probs.gather(1, codes_lvl.unsqueeze(-1)).squeeze(-1)
            log_pi_total = log_pi_total + log_pi_lvl
            # Advance residual using the selected code (detached — residual path
            # is a fixed unrolling for this log-prob computation).
            selected = cb[codes_lvl]
            residual = residual - selected
        return log_pi_total

    @torch.no_grad()
    def diagnostics(
        self, z: Tensor, codes: Tensor, tau: float = 1.0,
    ) -> dict:
        """Return per-level diagnostic scalars for the current (z, codes).

        All computations are no-grad, no EMA updates. Intended for
        logging during phase 2 to probe VQ bottleneck health.

        Args:
            z: [B, code_dim] encoder output
            codes: [B, num_levels] long — the codes actually sampled this rollout
            tau: categorical temperature (same as used during sampling)

        Returns dict with keys:
            rvq_argmax_agree: float — fraction of cells where sampled code == argmax code
            rvq_nearest_ratio_mean: float — mean of |z - 2nd_nearest| / |z - 1st_nearest|
                                   (ratio > 1; close to 1 = ambiguous argmax)
            rvq_cat_entropy_lvl{L}: float — mean per-level entropy of the categorical at z
            rvq_usage_entropy_lvl{L}: float — entropy of the cumulative per-code usage histogram
            rvq_effective_codes_lvl{L}: float — exp(usage_entropy), "effective vocab" per level
            rvq_code_spacing_mean_lvl{L}: float — mean pairwise L2 between codes at this level
            rvq_logpi_lvl{L}: float — mean log_pi contribution from this level
            rvq_code_unique_tuples: int — distinct (code_0, ..., code_L) tuples in this batch
        """
        out = {}
        B = z.shape[0]
        residual = z.detach()
        residual_argmax = z.detach()
        total_agree = 0
        total_ratio = 0.0
        ratio_count = 0

        for lvl in range(self.num_levels):
            cb = self.codebooks[lvl].detach()  # [K, D]

            # Categorical at this residual
            dists = (residual.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)  # [B, K]
            logits = -dists / tau
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            # Per-level entropy of the categorical at z
            lvl_H = -(probs * log_probs).sum(dim=-1).mean().item()
            out[f"rvq_cat_entropy_lvl{lvl}"] = lvl_H

            # Per-level log_pi (contribution to sampled code)
            codes_lvl = codes[:, lvl]
            log_pi_lvl = log_probs.gather(1, codes_lvl.unsqueeze(-1)).squeeze(-1)
            out[f"rvq_logpi_lvl{lvl}"] = log_pi_lvl.mean().item()

            # 1st vs 2nd nearest distance ratio (argmax ambiguity proxy)
            if cb.shape[0] >= 2:
                top2, _ = torch.topk(dists, k=2, dim=-1, largest=False)
                nearest = top2[:, 0].clamp(min=1e-8)
                second = top2[:, 1]
                ratio = (second / nearest).sqrt()  # L2-dist ratio, not sq
                total_ratio += ratio.sum().item()
                ratio_count += ratio.numel()

            # Argmax vs sampled-code agreement
            argmax_lvl = logits.argmax(dim=-1)
            total_agree += (argmax_lvl == codes_lvl).sum().item()

            # Per-level usage histogram (cumulative from EMA cluster_size)
            totals = self.cluster_size[lvl].sum().clamp(min=1.0)
            usage_p = self.cluster_size[lvl] / totals
            usage_p = usage_p.clamp(min=1e-12)
            usage_H = -(usage_p * usage_p.log()).sum().item()
            out[f"rvq_usage_entropy_lvl{lvl}"] = usage_H
            out[f"rvq_effective_codes_lvl{lvl}"] = math.exp(usage_H)

            # Pairwise code spacing (mean L2 between distinct pairs)
            if cb.shape[0] >= 2:
                pdists = (cb.unsqueeze(0) - cb.unsqueeze(1)).pow(2).sum(-1).sqrt()
                # exclude diagonal
                mask = ~torch.eye(cb.shape[0], dtype=torch.bool, device=cb.device)
                out[f"rvq_code_spacing_mean_lvl{lvl}"] = pdists[mask].mean().item()

            # Advance residuals using the sampled code (for the categorical path)
            # and the argmax code (for the argmax path). Needed because later
            # levels see the residual from previous level's selection.
            residual = residual - cb[codes_lvl]

        total_cells = B * self.num_levels
        out["rvq_argmax_agree"] = total_agree / max(total_cells, 1)
        if ratio_count > 0:
            out["rvq_nearest_ratio_mean"] = total_ratio / ratio_count
        # Distinct code tuples in this batch
        out["rvq_code_unique_tuples"] = int(torch.unique(codes, dim=0).shape[0])
        return out

    def entropy(self, z: Tensor, tau: float = 1.0) -> Tensor:
        """Compute entropy H(pi) of the distance-based categorical, summed across levels.

        z: [B, code_dim], returns [B] entropy values. Gradients flow through z.
        """
        B = z.shape[0]
        residual = z
        H_total = torch.zeros(B, device=z.device, dtype=z.dtype)
        for lvl in range(self.num_levels):
            cb = self.codebooks[lvl].detach()
            dists = (residual.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)
            logits = -dists / tau
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            H_total = H_total - (probs * log_probs).sum(dim=-1)
            # Advance residual using nearest code (detached)
            codes_lvl = logits.argmax(dim=-1)
            selected = cb[codes_lvl]
            residual = residual - selected
        return H_total


class ActionVQVAE(nn.Module):
    """Encoder → ResidualVQ → Decoder for modulator action vectors.

    Input: action vectors of shape [B, action_dim]. Output: reconstructed
    action + codes + loss terms.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 32,
        hidden: int = 512,
        num_levels: int = 4,
        codes_per_level: int = 16,
        beta: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        self.rvq = ResidualVQ(
            num_levels=num_levels,
            codes_per_level=codes_per_level,
            code_dim=latent_dim,
            decay=decay,
        )
        self.beta = beta
        # Normalization stats (populated from the codebook checkpoint or by
        # `set_normalization`). As buffers so they move with the module.
        self.register_buffer("action_mean", torch.zeros(1, action_dim))
        self.register_buffer("action_std", torch.ones(1, action_dim))

    def set_normalization(self, mean: Tensor, std: Tensor):
        """Populate action normalization stats after loading a checkpoint."""
        self.action_mean.copy_(mean.to(self.action_mean.device).reshape_as(self.action_mean))
        self.action_std.copy_(std.to(self.action_std.device).reshape_as(self.action_std))

    def normalize(self, action: Tensor) -> Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize(self, action: Tensor) -> Tensor:
        return action * self.action_std + self.action_mean

    def encode(self, action: Tensor) -> Tensor:
        return self.encoder(action)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, action: Tensor) -> dict:
        """Full training forward.

        action: [B, action_dim]
        Returns dict with recon, codes, z, z_q_st, loss, recon_loss, commit_loss.
        """
        z = self.encode(action)
        z_q_st, codes, commit_loss = self.rvq(z)
        recon = self.decode(z_q_st)
        recon_loss = F.mse_loss(recon, action)
        loss = recon_loss + self.beta * commit_loss
        return {
            "recon": recon,
            "codes": codes,
            "z": z,
            "z_q": z_q_st,
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
        }

    @torch.no_grad()
    def encode_quantize(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """No-grad encode + quantize for inference / phase-2 rollouts.

        Returns:
            z: [B, latent_dim] — encoder output (before quantization)
            z_q: [B, latent_dim] — nearest-neighbor quantization
            codes: [B, num_levels] — selected code indices per level
        """
        z = self.encode(action)
        z_q, codes = self.rvq.quantize(z)
        return z, z_q, codes
