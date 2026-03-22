"""
Procedural Memory (v6) — Hebbian fast-weight network.

Per-bank weight matrix as runtime state. Read: input→matmul→output.
Write: surprise-gated Hebbian correlation (pre⊗post reinforcement).
Commit: efficient right-multiply factorization (1 batched matmul).

Key design: PM is a learned TRANSFORM, not a prediction loop (that's PCM).
Banks differ by plasticity rate (beta_b), giving multiple timescales.
Read is bank-summed via W.sum(B) — no [BS,N,B,D] intermediate.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import StateMixin


def _logit(p: float) -> float:
    """Inverse sigmoid: logit(p) = log(p / (1-p))."""
    return math.log(p / (1.0 - p))


class ProceduralMemory(nn.Module, StateMixin):
    """Hebbian fast-weight network with per-bank weight matrices.

    State:
        W_pm: [BS, B, D_pm, D_pm] — per-bank fast weight matrix

    Read:
        pre = proj_in(H)                   # [BS, N, D_pm]
        post = (Σ_b W_b) @ pre             # [BS, N, D_pm] — bank sum fused
        pm_read = proj_out(post)            # [BS, N, D]

    Write (Hebbian, surprise-gated):
        G = (1/N) · Σ_t σ(‖surp_t‖) · pre_t ⊗ pre_t^T   # gated autocorrelation
        W_b ← W_b @ (decay·I + β_b·G)                     # 1 batched matmul
        clip Frobenius norm to budget
    """

    _state_tensor_names = ["W_pm"]

    def __init__(self, B: int, D: int, D_pm: int = 64, decay: float = 0.999):
        super().__init__()
        self.B = B
        self.D = D
        self.D_pm = D_pm

        # Fixed projections: D ↔ D_pm
        self.proj_in = nn.Linear(D, D_pm)
        self.proj_out = nn.Linear(D_pm, D)
        # Zero-init so PM starts silent (residual branch = 0 at init)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # Per-bank plasticity rate: softplus(raw) → always positive
        self.raw_beta = nn.Parameter(torch.full([B], -3.0))

        # Per-bank decay rate: sigmoid(raw) → in (0, 1), init to target decay
        self.raw_decay = nn.Parameter(torch.full([B], _logit(decay)))

        # Surprise gate sensitivity: sigmoid(scale * ||surp|| + bias)
        # Init: scale=1, bias=0 → recovers original sigmoid(||surp||)
        self.surp_scale = nn.Parameter(torch.ones(1))
        self.surp_bias = nn.Parameter(torch.zeros(1))

        # State (lazily allocated)
        self.W_pm: Tensor | None = None

    @property
    def beta(self) -> Tensor:
        """Per-bank plasticity rate [B], always positive."""
        return F.softplus(self.raw_beta)

    @property
    def decay(self) -> Tensor:
        """Per-bank decay rate [B], in (0, 1)."""
        return torch.sigmoid(self.raw_decay)

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        """Pre-allocate W_pm as (1/B)·I + small noise.

        This ensures the initial read is approximately identity (passes
        information through), and each bank starts with slightly different
        weights so they can specialize.
        """
        D_pm = self.D_pm
        eye = torch.eye(D_pm, device=device, dtype=dtype)
        noise = torch.randn(BS, self.B, D_pm, D_pm, device=device, dtype=dtype) * 0.01
        self.W_pm = eye * (1.0 / self.B) + noise

    def is_initialized(self) -> bool:
        return self.W_pm is not None

    def read(self, H: Tensor) -> tuple[Tensor, Tensor]:
        """Read from PM: project H through bank-summed fast weights.

        Args:
            H: [BS, N, D] — scan output

        Returns:
            pm_read: [BS, N, D] — PM contribution to integration
            pre: [BS, N, D_pm] — projected input (saved for commit)
        """
        pre = self.proj_in(H)                            # [BS, N, D_pm]
        W_sum = self.W_pm.sum(dim=1)                     # [BS, D_pm, D_pm]
        post = torch.matmul(pre, W_sum.transpose(-1, -2))  # [BS, N, D_pm]
        pm_read = self.proj_out(post)                    # [BS, N, D]
        return pm_read, pre

    def commit(self, pre: Tensor, surprise: Tensor, budget: float = 16.0):
        """Segment-end Hebbian update of W_pm.

        Efficient factorization:
            post_bt = W_b @ pre_t  →  ΔW_b = β_b · Σ_t s_t · post_bt ⊗ pre_t^T
            = β_b · W_b @ (Σ_t s_t · pre_t ⊗ pre_t^T)
            = β_b · W_b @ G

        So: W_b_new = decay · W_b + β_b · W_b @ G = W_b @ (decay·I + β_b·G)
        Single batched matmul for all banks.

        Args:
            pre: [BS, N, D_pm] — projected inputs (from read)
            surprise: [BS, N, D] — vector surprise from PCM
            budget: max Frobenius norm per bank
        """
        BS, N, D_pm = pre.shape

        # Surprise gate: σ(scale · ‖surprise‖ + bias) ∈ [0, 1] per token
        # Learnable scale/bias let the model control sensitivity threshold
        surp_mag = surprise.norm(dim=-1)                  # [BS, N]
        s = torch.sigmoid(self.surp_scale * surp_mag + self.surp_bias)  # [BS, N]

        # Gated autocorrelation: G = (1/N) · Σ_t s_t · pre_t ⊗ pre_t^T
        # Efficient: G = (1/N) · (√s · pre)^T @ (√s · pre)
        s_sqrt = s.unsqueeze(-1).sqrt()                   # [BS, N, 1]
        weighted = s_sqrt * pre                           # [BS, N, D_pm]
        G = torch.matmul(
            weighted.transpose(-1, -2), weighted          # [BS, D_pm, D_pm]
        ) * (1.0 / N)

        # Per-bank transform: T_b = decay_b·I + β_b·G
        beta = self.beta                                  # [B]
        decay = self.decay                                # [B]
        eye = torch.eye(D_pm, device=pre.device, dtype=pre.dtype)
        # decay: [1, B, 1, 1], G: [BS, 1, D_pm, D_pm], beta: [1, B, 1, 1]
        T = decay[None, :, None, None] * eye + beta[None, :, None, None] * G.unsqueeze(1)

        # Update: W_b = W_b @ T_b  (batched matmul)
        self.W_pm = torch.matmul(self.W_pm, T)           # [BS, B, D_pm, D_pm]

        # Budget enforcement: Frobenius norm per bank
        # norm over last two dims (the D_pm × D_pm matrix)
        frob = self.W_pm.flatten(-2).norm(dim=-1, keepdim=True)  # [BS, B, 1]
        frob = frob.unsqueeze(-1).clamp(min=1e-8)        # [BS, B, 1, 1]
        scale = (budget / frob).clamp(max=1.0)
        self.W_pm = self.W_pm * scale

    def reset_states(self, mask: Tensor):
        """Reset W_pm to (1/B)·I for masked streams.

        mask: [BS] bool — True for streams to reset.
        """
        if self.W_pm is None:
            return
        D_pm = self.D_pm
        eye = torch.eye(D_pm, device=self.W_pm.device, dtype=self.W_pm.dtype)
        fresh = eye * (1.0 / self.B)
        # mask: [BS] → [BS, 1, 1, 1]
        expanded = mask[:, None, None, None]
        self.W_pm = torch.where(expanded, fresh, self.W_pm)
