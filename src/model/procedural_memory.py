"""
Procedural Memory — holographic modulation with eligibility traces.

One PM instance per (block, layer). B*L total instances.

Read mechanism (holographic):
    Each slot stores a (key, value, strength) triple. Rather than retrieving
    values directly, the input flows *through* the stored modulation patterns:

        scores = normalize(x) @ K^T                  # [BS, r]
        y = sum_i(a_i * score_i * x * v_i)           # [BS, D_h]

    Mathematically: y_d = x_d * [W @ x]_d where W = K^T diag(a) V.
    This is quadratic in x — the effective transformation is input-dependent,
    giving each slot the expressiveness of a learned transformation rather
    than a fixed vector retrieval. Analogous to dendritic computation: the
    stored pattern (v) modulates the incoming signal (x) per-dimension.

Eligibility (Hebbian):
    v_cand = W_v_post(x * h) — stores the input-output *interaction*,
    not just the output. The element-wise product x * h captures which
    dimensions co-activated between pre-synaptic input and post-synaptic
    output, naturally pairing with the holographic read.

State per stream:
    pm_K: [BS, r, D_h]     — key bank (unit-normalized)
    pm_V: [BS, r, D_h]     — value bank (unit-normalized modulation patterns)
    pm_a: [BS, r]           — slot strengths (bounded)
    elig_K: [BS, r, D_h]   — eligibility trace for keys
    elig_V: [BS, r, D_h]   — eligibility trace for values (interaction patterns)

pm_K, pm_V, pm_a are plain tensors (NOT parameters). They are updated at
span boundaries and may carry computation graphs within a TBPTT chunk so
eligibility projections can receive LM gradients from downstream reads.
Eligibility projections (W_k_pre, W_v_post) are trained by backprop.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, unit_normalize, budget_enforce, runtime_state_dtype


class ProceduralMemory(nn.Module, StateMixin):
    _state_tensor_names = ["pm_K", "pm_V", "pm_a", "elig_K", "elig_V"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        D_h = config.D_h
        self.r = config.r
        self.D_h = D_h
        self.rho = config.rho
        self.a_max = config.a_max
        self.budget = config.budget_pm
        self.decay = config.decay_pm
        self.weakness_weight = config.weakness_weight_pm

        # Eligibility projection layers (learned parameters)
        self.W_k_pre = nn.Linear(D_h, D_h)
        self.W_v_post = nn.Linear(D_h, D_h)

        # Post-readout MLP: processes the linear lookup result
        if config.pm_readout_ffn:
            self.readout_norm = nn.LayerNorm(D_h)
            self.readout_ffn = nn.Sequential(
                nn.Linear(D_h, D_h * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(D_h * 4, D_h),
            )
        else:
            self.readout_norm = None
            self.readout_ffn = None

        # State tensors (lazily initialized)
        self.pm_K: Tensor = None
        self.pm_V: Tensor = None
        self.pm_a: Tensor = None
        self.elig_K: Tensor = None
        self.elig_V: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize state on first forward pass."""
        state_dtype = runtime_state_dtype(device)
        self.pm_K = unit_normalize(
            torch.randn(BS, self.r, self.D_h, device=device, dtype=state_dtype)
        )
        self.pm_V = unit_normalize(
            torch.randn(BS, self.r, self.D_h, device=device, dtype=state_dtype)
        )
        self.pm_a = torch.zeros(BS, self.r, device=device, dtype=state_dtype)
        self.elig_K = torch.zeros(BS, self.r, self.D_h, device=device, dtype=state_dtype)
        self.elig_V = torch.zeros(BS, self.r, self.D_h, device=device, dtype=state_dtype)

    def apply(self, x_block: Tensor) -> Tensor:
        """Holographic PM read: input modulated by stored patterns.

        y_d = x_d * sum_i(a_i * (x · k_i) * v_{id}) — quadratic in x.

        Args:
            x_block: [BS, D_h] — block input

        Returns:
            y_pm: [BS, D_h]
        """
        if self.pm_K is None:
            self._lazy_init(x_block.shape[0], x_block.device)

        x_q = unit_normalize(x_block)          # [BS, D_h]
        # Cast to state dtype (bf16 on CUDA) for matmul with pm_K/pm_V
        x_q = x_q.to(self.pm_K.dtype)
        # scores = pm_K @ x_q -> [BS, r]
        scores = torch.matmul(x_q.unsqueeze(1), self.pm_K.transpose(-1, -2)).squeeze(1)
        # Holographic read: input flows through stored modulation patterns
        weighted = (self.pm_a * scores).unsqueeze(-1)           # [BS, r, 1]
        y_pm = (weighted * x_q.unsqueeze(1) * self.pm_V).sum(1)  # [BS, D_h]
        y_pm = y_pm.float()  # back to param dtype for LayerNorm/FFN

        # Post-readout processing
        if self.readout_ffn is not None:
            y_pm = y_pm + self.readout_ffn(self.readout_norm(y_pm))

        return y_pm

    def _apply_batch_core(self, x_block_all: Tensor) -> Tensor:
        """Compiled holographic PM read for P tokens. No lazy init.

        Safe for torch.compile(fullgraph=True).
        """
        x_q = unit_normalize(x_block_all)                                        # [BS, P, D_h]
        # Cast to state dtype (bf16 on CUDA) for matmul with pm_K/pm_V
        x_q = x_q.to(self.pm_K.dtype)
        scores = torch.matmul(x_q, self.pm_K.transpose(-1, -2))                  # [BS, P, r]
        # Holographic read: input flows through stored modulation patterns
        weighted = (self.pm_a.unsqueeze(1) * scores).unsqueeze(-1)               # [BS, P, r, 1]
        y_pm = (weighted * x_q.unsqueeze(2) * self.pm_V.unsqueeze(1)).sum(2)     # [BS, P, D_h]
        y_pm = y_pm.float()  # back to param dtype for LayerNorm/FFN

        if self.readout_ffn is not None:
            y_pm = y_pm + self.readout_ffn(self.readout_norm(y_pm))

        return y_pm

    def apply_batch(self, x_block_all: Tensor) -> Tensor:
        """Read-only PM lookup for P tokens in parallel.

        Thin wrapper that handles lazy init, then delegates to the
        compilable core.

        Args:
            x_block_all: [BS, P, D_h] — block input for all tokens

        Returns:
            y_pm_all: [BS, P, D_h]
        """
        return self._apply_batch_core(x_block_all)

    def update_eligibility(self, x: Tensor, h: Tensor, surprise: Tensor):
        """Differentiable per-token eligibility accumulation.

        Uses Hebbian interaction: v_cand = W_v_post(x * h), capturing the
        per-dimension correlation between pre- and post-synaptic signals.
        This pairs with the holographic read where V modulates the input.

        Eligibility is gated by surprise: only tokens the model doesn't
        predict well contribute to the trace. This prevents the trace norm
        from saturating (which would make the commit gate always fire).

        Args:
            x: [BS, D_h] — layer input (pre-synaptic)
            h: [BS, D_h] — layer output (post-synaptic)
            surprise: [BS, 1] — current surprise signal
        """
        if self.elig_K is None:
            self._lazy_init(x.shape[0], x.device)

        k_cand = unit_normalize(self.W_k_pre(x))   # [BS, D_h]
        v_cand = self.W_v_post(x * h)               # [BS, D_h] — Hebbian interaction

        # Cast to state dtype (bf16 on CUDA) for matmul with pm_K
        state_dtype = self.pm_K.dtype
        k_cand = k_cand.to(state_dtype)
        v_cand = v_cand.to(state_dtype)

        # Gate by surprise: low surprise → near-zero accumulation
        gate = (surprise.squeeze(-1) / self.config.surprise_scale).clamp(0.0, 1.0)  # [BS]
        gate = gate.to(state_dtype).unsqueeze(1).unsqueeze(2)  # [BS, 1, 1]

        # Slot-specific routing: weight candidate contribution per slot
        route_logits = torch.matmul(k_cand.unsqueeze(1), self.pm_K.transpose(-1, -2)).squeeze(1) / self.config.tau_route_pm
        route_w = torch.softmax(route_logits, dim=-1).unsqueeze(-1)  # [BS, r, 1]

        # Accumulate with per-slot routing weights
        # elig_K: [BS, r, D_h], k_cand: [BS, 1, D_h], route_w: [BS, r, 1]
        self.elig_K = self.rho * self.elig_K + gate * route_w * k_cand.unsqueeze(1)
        self.elig_V = self.rho * self.elig_V + gate * route_w * v_cand.unsqueeze(1)

    def _update_eligibility_core(self, x_all: Tensor, h_all: Tensor,
                                 surprise_all: Tensor,
                                 reset_mask: Tensor):
        """Compiled inner loop: projections + gating + fused K/V recurrence.

        No lazy init — safe for torch.compile(fullgraph=True).
        """
        BS, P, D_h = x_all.shape
        r = self.r

        # Batched projections (the expensive part — 2 matmuls instead of 2*P)
        k_cand_all = unit_normalize(self.W_k_pre(x_all))  # [BS, P, D_h]
        v_cand_all = self.W_v_post(x_all * h_all)          # [BS, P, D_h] — Hebbian interaction

        # Cast to state dtype (bf16 on CUDA) for matmul with pm_K and elig state
        state_dtype = self.pm_K.dtype
        k_cand_all = k_cand_all.to(state_dtype)
        v_cand_all = v_cand_all.to(state_dtype)

        # Surprise gating: [BS, P] → [BS, P, 1, 1] for broadcast over [r, D_h]
        gate = (surprise_all.squeeze(-1) / self.config.surprise_scale).clamp(0.0, 1.0)  # [BS, P]
        gate = gate.to(state_dtype).unsqueeze(-1).unsqueeze(-1)  # [BS, P, 1, 1]

        # Slot-specific routing: weight candidate contribution per slot
        route_logits = torch.matmul(k_cand_all, self.pm_K.transpose(-1, -2)) / self.config.tau_route_pm
        route_w = torch.softmax(route_logits, dim=-1)  # [BS, P, r]

        # b terms: gate * route_w * cand → [BS, P, r, D_h]
        b_K = gate * route_w.unsqueeze(-1) * k_cand_all.unsqueeze(2)
        b_V = gate * route_w.unsqueeze(-1) * v_cand_all.unsqueeze(2)

        # Carry mask: 0 at reset positions (zeros previous elig), 1 elsewhere.
        carry = (~reset_mask).to(k_cand_all.dtype)  # [BS, P]

        # Flatten [r, D_h] → [r*D_h] (contiguous for reshape)
        b_K_flat = b_K.contiguous().reshape(BS, P, r * D_h)
        b_V_flat = b_V.contiguous().reshape(BS, P, r * D_h)
        h_K_init = self.elig_K.reshape(BS, r * D_h)
        h_V_init = self.elig_V.reshape(BS, r * D_h)

        # Fused K+V recurrence: update only final state (t=P-1).
        # This avoids materializing [BS, P, ...] intermediates that are unused.
        b_KV = torch.cat([b_K_flat, b_V_flat], dim=-1)       # [BS, P, 2*r*D_h]
        h_KV_init = torch.cat([h_K_init, h_V_init], dim=-1)  # [BS, 2*r*D_h]

        h_KV = h_KV_init
        for t in range(P):
            # Scalar carry per stream/timestep, broadcast over flattened state.
            a_t = (self.rho * carry[:, t]).unsqueeze(-1)   # [BS, 1]
            h_KV = a_t * h_KV + b_KV[:, t]
        elig_K_final, elig_V_final = h_KV.chunk(2, dim=-1)

        # Update state to last token, preserving configured runtime precision.
        self.elig_K = elig_K_final.reshape(BS, r, D_h).to(self.elig_K.dtype)
        self.elig_V = elig_V_final.reshape(BS, r, D_h).to(self.elig_V.dtype)

    def update_eligibility_batch(self, x_all: Tensor, h_all: Tensor,
                                 surprise_all: Tensor,
                                 reset_mask: Tensor):
        """Batched eligibility accumulation over P tokens via final-state recurrence.

        Thin wrapper that handles lazy init, then delegates to the
        compilable core.
        """
        if self.elig_K is None:
            self._lazy_init(x_all.shape[0], x_all.device)
        self._update_eligibility_core(x_all, h_all, surprise_all, reset_mask)

    def base_decay(self):
        """Per-span strength decay applied to ALL streams.

        Called at every span boundary before commit decisions. This ensures
        non-committing streams gradually lose pm_a over time.
        """
        if self.pm_a is not None:
            with torch.no_grad():
                self.pm_a = self.pm_a * self.decay

    def reset_content(self, mask: Tensor):
        """Zero committed PM state for masked streams.

        Clears pm_K, pm_V, pm_a but NOT eligibility traces (elig_K, elig_V).
        Used for mid-span doc-boundary resets in the parallel path where
        eligibility is already handled by the carry mask in the scan.
        """
        for name in ("pm_K", "pm_V", "pm_a"):
            t = getattr(self, name, None)
            if t is not None:
                expanded = mask
                for _ in range(t.dim() - 1):
                    expanded = expanded.unsqueeze(-1)
                setattr(self, name, t * (~expanded).to(t.dtype))

    def reset_eligibility(self, mask: Tensor):
        """Zero only eligibility traces for masked streams.

        Used in lifelong mode: committed PM weights persist,
        but in-progress eligibility clears at doc boundaries.
        """
        if self.elig_K is not None:
            expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
            self.elig_K = self.elig_K * (~expanded).to(self.elig_K.dtype)
            self.elig_V = self.elig_V * (~expanded).to(self.elig_V.dtype)

    def commit(self, p_commit: Tensor, lambda_vals: Tensor,
               g: Tensor, slot_logits: Tensor, tau: Tensor):
        """Span-boundary commit. Fully differentiable soft commit.

        All arguments are continuous tensors — no binary masks. The commit
        strength is controlled by p_commit (learned gate) * g (learned strength).

        Args:
            p_commit: [BS] — commit probability (continuous 0-1, from gate_head)
            lambda_vals: [BS] — commit-time decay per stream
            g: [BS] — write strength per stream
            slot_logits: [BS, r] — slot selection bias from neuromod
            tau: [BS] — softmax temperature for slot selection
        """
        if self.pm_K is None:
            return

        # Normalized eligibility as commit candidates
        elig_K_norm = unit_normalize(self.elig_K)  # [BS, r, D_h]
        elig_V_norm = self.elig_V                   # [BS, r, D_h]

        # Eligibility magnitude gate: when eligibility traces are near-zero,
        # the commit should be proportionally suppressed to avoid strengthening
        # PM slots with uninformative (noise) candidates.
        # elig_mag: [BS] in (0, 1], saturates at 1.0 for non-trivial eligibility.
        elig_norm = self.elig_K.norm(dim=-1).mean(dim=-1)  # [BS] mean L2 across slots
        elig_mag = (elig_norm / (elig_norm + 1.0)).detach()  # [BS] smooth 0→1 gate

        # Slot selection: similarity between current keys and eligibility
        scores = (self.pm_K * elig_K_norm).sum(-1)  # [BS, r]

        # Weakness bias: prefer overwriting weak slots
        scores = scores - self.weakness_weight * self.pm_a

        # Add controller-provided slot bias
        if slot_logits is not None:
            scores = scores + slot_logits

        # Softmax slot selection (replaces hard top-k)
        tau_expanded = tau.unsqueeze(-1)  # [BS, 1]
        weights = torch.softmax(scores / tau_expanded, dim=-1)  # [BS, r]

        # Soft commit: p_commit * g * elig_mag controls total write strength.
        # elig_mag suppresses commits when eligibility traces are near-zero.
        alpha = weights * g.unsqueeze(-1) * p_commit.unsqueeze(-1) * elig_mag.unsqueeze(-1)  # [BS, r]

        # Soft commit-time decay (proportional to p_commit)
        lambda_expanded = lambda_vals.unsqueeze(-1)  # [BS, 1]
        p_expanded = p_commit.unsqueeze(-1)  # [BS, 1]
        self.pm_a = self.pm_a * (1.0 - p_expanded * (1.0 - lambda_expanded))

        # EMA update of keys and values
        alpha_3d = alpha.unsqueeze(-1)  # [BS, r, 1]
        self.pm_K = unit_normalize((1 - alpha_3d) * self.pm_K + alpha_3d * elig_K_norm)
        self.pm_V = unit_normalize((1 - alpha_3d) * self.pm_V + alpha_3d * elig_V_norm)

        # Strength update
        self.pm_a = (self.pm_a + alpha).clamp(0.0, self.a_max)

        # Budget enforcement
        self.pm_a = budget_enforce(self.pm_a, self.budget)

        # Soft eligibility reset (proportional to commit strength)
        reset_3d = p_commit.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
        self.elig_K = self.elig_K * (1.0 - reset_3d)
        self.elig_V = self.elig_V * (1.0 - reset_3d)


class PMNeuromodulator(nn.Module):
    """Neuromodulator for PM commit decisions.

    Produces fully differentiable outputs for all commit parameters.
    All outputs are continuous — trained by main-loss gradient through
    the PM write → read chain.

    When pm_enabled: learned backbone with content-aware features.
    When not pm_enabled: heuristic defaults (shouldn't happen — PM always on).

    Returns:
        p_commit: [BS] — commit strength (continuous 0-1)
        lambda_vals: [BS] — commit-time decay
        g: [BS] — write strength
        slot_logits: [BS, r] — slot selection bias
        tau: [BS] — softmax temperature for slot selection
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pm_enabled = config.pm_enabled
        self.default_g = config.g_pm_default
        self.default_decay = config.decay_pm
        self.tau_floor = config.tau_pm_floor
        self.tau_ceil = config.tau_pm_ceil
        self.default_tau = config.tau_pm

        if self.pm_enabled:
            H = config.neuromod_hidden
            n_scalar = 3  # elig_norm, pm_usage, span_surprise
            n_content = config.content_proj_dim
            n_features = n_scalar + n_content

            # Content projection: elig_K mean [D_h] → [content_proj_dim]
            self.content_proj = nn.Linear(config.D_h, n_content)

            self.backbone = nn.Sequential(
                nn.Linear(n_features, H),
                nn.ReLU(),
            )

            self.gate_head = nn.Linear(H, 1)
            self.lambda_head = nn.Linear(H, 1)
            self.g_head = nn.Linear(H, 1)
            self.slot_head = nn.Linear(H, config.r)
            self.tau_head = nn.Linear(H, 1)

            # Zero backbone bias: zero input → zero backbone → head biases
            nn.init.zeros_(self.backbone[0].bias)

            # Content proj: small init so content features start near zero
            nn.init.normal_(self.content_proj.weight, std=0.01)
            nn.init.zeros_(self.content_proj.bias)

            # gate_head: init to p_commit ≈ 0.5 (sigmoid(0))
            nn.init.normal_(self.gate_head.weight, std=0.01)
            nn.init.zeros_(self.gate_head.bias)

            # g_head: sigmoid(bias) = default_g
            g_frac = max(min(config.g_pm_default, 0.999), 0.001)
            nn.init.normal_(self.g_head.weight, std=0.01)
            nn.init.constant_(self.g_head.bias, math.log(g_frac / (1.0 - g_frac)))

            # lambda: near-zero raw → lambda ≈ default_decay
            nn.init.normal_(self.lambda_head.weight, std=0.01)
            nn.init.constant_(self.lambda_head.bias, math.log(0.001 / 0.999))

            # slot_logits: small init → no slot preference at init
            nn.init.normal_(self.slot_head.weight, std=0.01)
            nn.init.zeros_(self.slot_head.bias)

            # tau: init to default_tau within [floor, ceil]
            tau_frac = (config.tau_pm - config.tau_pm_floor) / max(config.tau_pm_ceil - config.tau_pm_floor, 1e-8)
            tau_frac = max(min(tau_frac, 0.999), 0.001)
            nn.init.normal_(self.tau_head.weight, std=0.01)
            nn.init.constant_(self.tau_head.bias, math.log(tau_frac / (1.0 - tau_frac)))

    def forward(self, elig_norm: Tensor, pm_usage: Tensor,
                span_surprise: Tensor,
                content_emb: Tensor = None) -> tuple:
        """Forward pass.

        Args:
            elig_norm: [BS] — mean eligibility trace norm
            pm_usage: [BS] — normalized PM usage (sum(pm_a) / budget)
            span_surprise: [BS] — mean surprise over span
            content_emb: [BS, D_h] or None — mean eligibility key embedding

        Returns:
            (p_commit, lambda_vals, g, slot_logits, tau)
        """
        if self.pm_enabled:
            return self._forward_learned(elig_norm, pm_usage, span_surprise, content_emb)
        return self._forward_heuristic(elig_norm)

    def _forward_heuristic(self, elig_norm):
        """Fallback with constant defaults (should rarely be used — PM always on)."""
        BS = elig_norm.shape[0]
        device = elig_norm.device
        p_commit = torch.full((BS,), 0.5, device=device)
        lambda_vals = torch.full((BS,), self.default_decay, device=device)
        g = torch.full((BS,), self.default_g, device=device)
        tau = torch.full((BS,), self.default_tau, device=device)
        return p_commit, lambda_vals, g, None, tau

    def _forward_learned(self, elig_norm, pm_usage, span_surprise, content_emb):
        """Fully differentiable learned mode — all outputs continuous."""
        feat_dtype = self.backbone[0].weight.dtype
        scalar_features = torch.stack([elig_norm, pm_usage, span_surprise], dim=-1)  # [BS, 3]
        scalar_features = scalar_features.to(feat_dtype)

        if content_emb is not None:
            content_features = self.content_proj(content_emb.to(feat_dtype))  # [BS, content_proj_dim]
            features = torch.cat([scalar_features, content_features], dim=-1)
        else:
            features = torch.cat([
                scalar_features,
                torch.zeros(elig_norm.shape[0], self.content_proj.out_features,
                            device=elig_norm.device, dtype=feat_dtype)
            ], dim=-1)

        h = self.backbone(features)

        p_commit = torch.sigmoid(self.gate_head(h)).squeeze(-1)

        raw_lambda = torch.sigmoid(self.lambda_head(h)).squeeze(-1)
        lambda_vals = self.default_decay + (1.0 - self.default_decay) * raw_lambda

        g = torch.sigmoid(self.g_head(h)).squeeze(-1)
        slot_logits = self.slot_head(h)

        raw_tau = torch.sigmoid(self.tau_head(h)).squeeze(-1)
        tau = self.tau_floor + (self.tau_ceil - self.tau_floor) * raw_tau

        return p_commit, lambda_vals, g, slot_logits, tau
