"""
Per-block Predictive Coding Module (PCM).

Implements a hypothesis-evidence-surprise loop:
- Evidence encoder: compresses block input to a D_pc latent (per token)
- Hypothesis predictor: predicts next span's latent from current state +
  memory summaries (per span boundary)
- Surprise signal: δ = evidence - hypothesis (per token, vectorial)

The surprise vector δ enters the model in two places:
1. Gate inputs (replaces scalar surprise): cat([x, y_pm, y_wm, y_em, δ])
2. FFN gain modulation: ffn_input *= (1 + 0.1 * tanh(W_gain(δ)))

Auxiliary losses (L_pred, L_recon) train the PCM without affecting the
main model's gradient flow (targets are stop-gradiented).

Surprise is suppressed (returns zeros) until the first boundary_update
sets a valid hypothesis via z_hat_valid.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, runtime_state_dtype


class PredictiveCodingModule(nn.Module, StateMixin):
    """Per-block predictive coding: encode evidence, predict hypothesis, compute surprise."""

    # z_hat is NOT in _state_tensor_names because we override detach_states
    # to preserve its computation graph across TBPTT boundaries.
    _state_tensor_names = []

    def __init__(self, config: ModelConfig):
        super().__init__()
        D_h = config.D_h
        D_pc = config.D_pc
        D_em = config.D_em

        # Evidence encoder: block input → latent
        self.encoder_norm = nn.LayerNorm(D_h)
        self.encoder = nn.Linear(D_h, D_pc)

        # Reconstruction decoder: latent → block input (aux loss only)
        self.decoder = nn.Linear(D_pc, D_h)

        # Hypothesis predictor: (z_end, ctx_b, pm_summary_b, em_summary_b) → z_hat
        pred_input = D_pc + D_h + D_h + D_em
        self.predictor = nn.Sequential(
            nn.Linear(pred_input, D_pc * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(D_pc * 2, D_pc),
        )

        # FFN gain modulation: δ → bounded multiplicative gain for FFN input
        # Zero-init so gain starts at exactly 1.0 (identity modulation)
        self.W_gain = nn.Linear(D_pc, D_h)
        nn.init.zeros_(self.W_gain.weight)
        nn.init.zeros_(self.W_gain.bias)

        self.D_pc = D_pc

        # State: frozen hypothesis from previous span boundary
        self.z_hat: Tensor | None = None
        # Per-stream validity: True after first boundary_update.
        # Prevents garbage surprise from z_hat=0 before any hypothesis exists.
        self.z_hat_valid: Tensor | None = None

    def _lazy_init(self, BS: int, device: torch.device):
        self.z_hat = torch.zeros(
            BS, self.D_pc, device=device, dtype=runtime_state_dtype(device),
        )
        self.z_hat_valid = torch.zeros(
            BS, dtype=torch.bool, device=device,
        )

    def encode(self, x_block: Tensor) -> Tensor:
        """Encode block input to latent evidence.

        Args:
            x_block: [BS, P, D_h] — block input

        Returns:
            z: [BS, P, D_pc]
        """
        return self.encoder(self.encoder_norm(x_block))

    def compute_surprise(self, z: Tensor) -> Tensor:
        """Compute prediction error: evidence minus hypothesis.

        z_hat is detached here so the LM loss does NOT backprop into the
        predictor through the δ → gate → layer → logits path. The predictor
        is trained only by L_pred (prediction loss at span boundaries).

        Returns zeros if z_hat has not been set by boundary_update yet
        (z_hat_valid is False), preventing garbage surprise from the
        initial zero hypothesis.

        Args:
            z: [BS, P, D_pc] — encoded evidence

        Returns:
            delta: [BS, P, D_pc] — surprise vector
        """
        if self.z_hat is None:
            return torch.zeros_like(z)
        # z_hat is [BS, D_pc], broadcast over P
        # Detach: predictor learns from L_pred only, not from LM loss
        delta = z - self.z_hat.detach().unsqueeze(1)
        # Suppress surprise for streams without a valid hypothesis
        if self.z_hat_valid is not None:
            # [BS] → [BS, 1, 1] for broadcast over [BS, P, D_pc]
            valid_mask = self.z_hat_valid.unsqueeze(-1).unsqueeze(-1)
            delta = delta * valid_mask.to(delta.dtype)
        return delta

    def compute_ffn_gain(self, delta: Tensor) -> Tensor:
        """Compute bounded multiplicative gain for FFN input.

        Gain is bounded to [0.9, 1.1] by tanh, preventing sign flips
        and unbounded magnitudes later in training.

        Args:
            delta: [BS, P, D_pc] — surprise vector

        Returns:
            gain: [BS, P, D_h] — multiply with FFN input
        """
        return 1.0 + 0.1 * torch.tanh(self.W_gain(delta))

    def compute_recon_loss(self, z: Tensor, x_block: Tensor) -> Tensor:
        """Reconstruction loss: train encoder+decoder to preserve information.

        Args:
            z: [BS, P, D_pc] — encoded evidence
            x_block: [BS, P, D_h] — original block input (target, detached)

        Returns:
            L_recon: scalar
        """
        x_recon = self.decoder(z)
        return (x_recon - x_block.detach()).pow(2).mean()

    def boundary_update(
        self,
        z_end: Tensor,
        ctx_b: Tensor,
        pm_summary_b: Tensor,
        em_summary_b: Tensor,
    ) -> Tensor:
        """Update hypothesis at span boundary. Returns prediction loss.

        Called after forward pass. Computes L_pred from the old hypothesis
        (only if valid), then generates a new hypothesis for the next span
        and marks it as valid.

        Args:
            z_end: [BS, D_pc] — evidence at last position
            ctx_b: [BS, D_h] — block output at last position
            pm_summary_b: [BS, D_h] — PM strength-weighted readout for this block
            em_summary_b: [BS, D_em] — EM strength-weighted readout for this block

        Returns:
            L_pred: scalar — prediction loss (stop-grad on target)
        """
        # Prediction loss: how wrong was the hypothesis?
        # Only compute if z_hat has a valid hypothesis (set by a previous boundary)
        if self.z_hat is not None and self.z_hat_valid is not None \
                and self.z_hat_valid.any():
            # Mask loss to only count valid streams
            err = (self.z_hat - z_end.detach()).pow(2)  # [BS, D_pc]
            valid_f = self.z_hat_valid.to(err.dtype).unsqueeze(-1)  # [BS, 1]
            L_pred = (err * valid_f).sum() / valid_f.sum().clamp(min=1) / self.D_pc
        else:
            # No valid hypothesis yet (z_hat=None, or z_hat_valid all-False)
            L_pred = z_end.new_tensor(0.0)

        # Generate new hypothesis for next span
        # All inputs detached: predictor learns to predict, main model unaffected
        pred_input = torch.cat([
            z_end.detach(),
            ctx_b.detach(),
            pm_summary_b.detach(),
            em_summary_b.detach(),
        ], dim=-1)
        # z_hat is NOT detached: its graph (predictor params only) persists
        # for next span's L_pred computation
        self.z_hat = self.predictor(pred_input)

        # Mark hypothesis as valid for all streams
        if self.z_hat_valid is not None:
            self.z_hat_valid = torch.ones_like(self.z_hat_valid)

        return L_pred

    def detach_states(self):
        """Override: do NOT detach z_hat.

        z_hat's computation graph (just the predictor's forward pass on
        detached inputs) must persist across TBPTT boundaries so that
        L_pred in the next span can backprop through the predictor weights.
        The graph is tiny (predictor params are leaf tensors).
        """
        pass  # intentionally empty

    def reset_states(self, mask: Tensor):
        """Zero z_hat and invalidate hypothesis for streams at doc boundaries."""
        if self.z_hat is not None:
            keep = (~mask).to(self.z_hat.dtype).unsqueeze(-1)  # [BS, 1]
            self.z_hat = self.z_hat * keep
        if self.z_hat_valid is not None:
            self.z_hat_valid = self.z_hat_valid & (~mask)
