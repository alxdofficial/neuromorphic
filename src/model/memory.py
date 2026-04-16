"""Single-cell memory graph with conv-grid neuromodulator.

N_total neurons in a single connectivity pool. Virtual I/O pools (NC_pools =
D / D_n) preserve the LM interface. bf16 throughout; γ clamped to
config.gamma_max for bf16-safe EMA.

See `docs/design_conv_modulator.md` for the full design rationale.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config
from .discrete_policy import DiscreteActionPolicy
from .grid_modulator import ConvGridModulator, ConvTransposeDecoder, port_layout


class MemoryGraph(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.D_n = config.D_n
        self.N = config.N_total
        self.alpha = config.alpha
        self.NC_pools = config.NC_pools

        # Per-neuron identity vector (learned "fingerprint").
        self.neuron_id = nn.Parameter(
            torch.randn(self.N, self.D_n) * (self.D_n ** -0.5))

        # Shared state/msg MLPs (2-layer tanh MLPs).
        Hs = config.state_mlp_hidden
        Hm = config.msg_mlp_hidden
        # state MLP input: cat(received, h) = 2*D_n
        self.state_w1 = nn.Parameter(torch.empty(Hs, 2 * self.D_n))
        self.state_b1 = nn.Parameter(torch.zeros(Hs))
        self.state_w2 = nn.Parameter(torch.empty(self.D_n, Hs))
        self.state_b2 = nn.Parameter(torch.zeros(self.D_n))
        # msg MLP input: h = D_n
        self.msg_w1 = nn.Parameter(torch.empty(Hm, self.D_n))
        self.msg_b1 = nn.Parameter(torch.zeros(Hm))
        self.msg_w2 = nn.Parameter(torch.empty(self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(self.D_n))

        # Per-pool inject projection. H_mid[pool_p] is a D_n slice;
        # project it to alpha * D_n and drop into input-port neurons.
        self.inject_w = nn.Parameter(
            torch.empty(self.NC_pools, self.alpha * self.D_n, self.D_n))
        self.inject_b = nn.Parameter(
            torch.zeros(self.NC_pools, self.alpha * self.D_n))

        # Per-neuron learnable plasticity logits. γ = gamma_max · sigmoid(logit).
        # Init at -3 (γ ≈ 0.046) for W and decay; +2 (γ ≈ 0.85) for hebbian.
        self.W_decay_logit = nn.Parameter(torch.full((self.N,), -3.0))
        self.decay_gamma_logit = nn.Parameter(torch.full((self.N,), -3.0))
        self.hebbian_decay_logit = nn.Parameter(torch.full((self.N,), 2.0))
        self.gamma_max = config.gamma_max

        # Conv-grid modulator (encoder) and conv-transpose decoder.
        self.modulator = ConvGridModulator(config)
        self.decoder = ConvTransposeDecoder(config)
        self.discrete_policy = DiscreteActionPolicy(
            num_codes=config.num_codes, code_dim=config.code_dim)

        # Gumbel τ (mutated by train.py step_callback; anneals 1.0 → 0.3).
        self.gumbel_tau = 1.0

        # Port-layout buffers: input_port_idx [NC_pools, alpha],
        # output_port_idx [NC_pools, alpha], role_id [N].
        layout = port_layout(config)
        self.register_buffer("input_port_idx", layout["input_port_idx"])
        self.register_buffer("output_port_idx", layout["output_port_idx"])
        self.register_buffer("role_id", layout["role_id"])

        # Xavier init for shared MLPs (tanh-gained).
        TANH_GAIN = 5.0 / 3.0
        nn.init.xavier_uniform_(self.state_w1, gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.state_w2, gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.msg_w1, gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.msg_w2, gain=TANH_GAIN)

        # Xavier for per-pool inject (einsum-semantics; manual bound).
        bound = 1.0 * math.sqrt(6.0 / (self.D_n + self.alpha * self.D_n))
        with torch.no_grad():
            self.inject_w.uniform_(-bound, bound)

        self._initialized = False
        self._compiled = False

    # ================================================================
    # State management
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int, device: torch.device):
        dt = torch.bfloat16
        N, D_n = self.N, self.D_n

        # h, msg: per-neuron state and outgoing message (zero cold-start).
        self.h = torch.zeros(BS, N, D_n, device=device, dtype=dt)
        self.msg = torch.zeros(BS, N, D_n, device=device, dtype=dt)
        # decay stored in [0,1]; init mid-range.
        self.decay = torch.full((BS, N), 0.5, device=device, dtype=dt)
        # Hebbian co-activation trace (N × N matrix).
        self.hebbian = torch.zeros(BS, N, N, device=device, dtype=dt)
        # W: sparse init with ~1/sqrt(N) magnitude, RMS-normed per row.
        # Each neuron gets ~8 random nonzero connections at init.
        K_init = min(8, N - 1)
        W = torch.zeros(BS, N, N, device=device, dtype=dt)
        for neuron in range(N):
            neighbors = torch.randperm(N, device=device)[:K_init]
            W[:, neuron, neighbors] = 1.0
        W.diagonal(dim1=-2, dim2=-1).zero_()
        self.W = F.rms_norm(W, normalized_shape=(N,))

        # Surprise state (per-batch scalar).
        self.s_mem_live = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(BS, device=device, dtype=dt)
        # Per-pool readout drift signal.
        self.readout_drift = torch.zeros(
            BS, self.NC_pools, 1, device=device, dtype=dt)
        # Previous readout (full D-dim for memory head prediction).
        self.prev_readout = torch.zeros(
            BS, self.config.D, device=device, dtype=dt)

        self._initialized = True

    def detach_states(self):
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.msg = self.msg.detach()
        self.W = self.W.detach()
        self.decay = self.decay.detach()
        self.hebbian = self.hebbian.detach()
        self.prev_readout = self.prev_readout.detach()
        self.readout_drift = self.readout_drift.detach()
        # s_mem_* are already detached (no_grad EMAs).

    @torch.no_grad()
    def resize_to_bs(self, new_bs: int):
        """Resize runtime state to new_bs. Long-term state is sampled/tiled;
        transient state is reset to zero."""
        if not self._initialized:
            return
        device = self.W.device
        dt = self.W.dtype
        old_bs = self.W.shape[0]
        if new_bs == old_bs:
            return

        if new_bs < old_bs:
            idx = torch.randperm(old_bs, device=device)[:new_bs]
        else:
            idx = torch.arange(new_bs, device=device) % old_bs

        self.W = self.W[idx].clone()
        self.decay = self.decay[idx].clone()
        self.hebbian = self.hebbian[idx].clone()

        self.h = torch.zeros(new_bs, self.N, self.D_n, device=device, dtype=dt)
        self.msg = torch.zeros(new_bs, self.N, self.D_n, device=device, dtype=dt)
        self.s_mem_live = torch.zeros(new_bs, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(new_bs, device=device, dtype=dt)
        self.prev_readout = torch.zeros(
            new_bs, self.config.D, device=device, dtype=dt)
        self.readout_drift = torch.zeros(
            new_bs, self.NC_pools, 1, device=device, dtype=dt)

    def runtime_state_dict(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized": True,
            "h": self.h.clone(),
            "msg": self.msg.clone(),
            "W": self.W.clone(),
            "decay": self.decay.clone(),
            "hebbian": self.hebbian.clone(),
            "s_mem_live": self.s_mem_live.clone(),
            "s_mem_ema_fast": self.s_mem_ema_fast.clone(),
            "prev_readout": self.prev_readout.clone(),
            "readout_drift": self.readout_drift.clone(),
        }

    def load_runtime_state(self, state: dict):
        if not state or not state.get("initialized", False):
            self._initialized = False
            return
        # Reject old-format checkpoints (4-dim h with NC axis).
        loaded_h = state.get("h")
        if loaded_h is None or loaded_h.ndim != 3:
            raise RuntimeError(
                "Checkpoint runtime state is incompatible with conv-grid-modulator "
                "design (expected 3-D h [BS, N, D_n]; got shape "
                f"{tuple(loaded_h.shape) if loaded_h is not None else None}). "
                "Old multi-cell checkpoints cannot be loaded on this branch.")
        device = self.neuron_id.device
        self.h = state["h"].to(device)
        self.msg = state["msg"].to(device)
        self.W = state["W"].to(device)
        self.decay = state["decay"].to(device)
        self.hebbian = state["hebbian"].to(device)
        BS = self.h.shape[0]
        dt = self.h.dtype
        zero_b = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_live = state.get("s_mem_live", zero_b).to(device)
        self.s_mem_ema_fast = state.get("s_mem_ema_fast", zero_b).to(device)
        self.prev_readout = state.get(
            "prev_readout",
            torch.zeros(BS, self.config.D, device=device, dtype=dt)).to(device)
        self.readout_drift = state.get(
            "readout_drift",
            torch.zeros(BS, self.NC_pools, 1, device=device, dtype=dt)).to(device)
        self._initialized = True

    # ================================================================
    # Telemetry
    # ================================================================

    @torch.no_grad()
    def compute_lane_divergence(self) -> dict:
        if not self._initialized:
            return {}
        BS = self.W.shape[0]
        if BS <= 1:
            return {}
        W_mean = self.W.mean(dim=0, keepdim=True)
        heb_mean = self.hebbian.mean(dim=0, keepdim=True)
        W_div = (self.W - W_mean).float().norm(dim=(1, 2)).mean().item()
        W_norm = self.W.float().norm(dim=(1, 2)).mean().item()
        heb_div = (self.hebbian - heb_mean).float().norm(dim=(1, 2)).mean().item()
        heb_norm = self.hebbian.float().norm(dim=(1, 2)).mean().item()
        return {
            "lane_W_divergence": W_div,
            "lane_W_relative_div": W_div / max(W_norm, 1e-8),
            "lane_hebbian_divergence": heb_div,
            "lane_hebbian_relative_div": heb_div / max(heb_norm, 1e-8),
        }

    @torch.no_grad()
    def compute_memory_health(self) -> dict:
        if not self._initialized:
            return {}
        h = self.h.float()
        msg = self.msg.float()
        W = self.W.float()
        decay = self.decay.float()
        heb = self.hebbian.float()

        out = {
            "h_norm": h.norm().item() / max(h.numel() ** 0.5, 1.0),
            "msg_norm": msg.norm().item() / max(msg.numel() ** 0.5, 1.0),
            "h_max": h.abs().max().item(),
            "msg_max": msg.abs().max().item(),
            "decay_mean": decay.mean().item(),
            "decay_std": decay.std().item(),
            "s_mem_live": self.s_mem_live.float().mean().item(),
            "s_mem_ema_fast": self.s_mem_ema_fast.float().mean().item(),
            "readout_drift_mean": self.readout_drift.float().mean().item(),
        }
        N = W.shape[-1]
        diag_mask = torch.eye(N, device=W.device, dtype=torch.bool)
        W_off = W.masked_fill(diag_mask, 0.0)
        out["W_offdiag_norm"] = W_off.norm().item() / max(W_off.numel() ** 0.5, 1.0)
        out["W_offdiag_max"] = W_off.abs().max().item()
        heb_off = heb.masked_fill(diag_mask, 0.0)
        out["hebbian_offdiag_norm"] = (
            heb_off.norm().item() / max(heb_off.numel() ** 0.5, 1.0))
        wf = W_off.reshape(W.shape[0], -1)
        hf = heb_off.reshape(W.shape[0], -1)
        cos = F.cosine_similarity(wf, hf, dim=-1)
        out["W_hebbian_offdiag_cos"] = cos.mean().item()
        return out

    @torch.no_grad()
    def compute_plasticity_rates(self) -> dict:
        gm = self.gamma_max
        W_gamma = gm * torch.sigmoid(self.W_decay_logit).float()
        d_gamma = gm * torch.sigmoid(self.decay_gamma_logit).float()
        h_gamma = gm * torch.sigmoid(self.hebbian_decay_logit).float()

        def _hl(gamma: Tensor) -> Tensor:
            g = gamma.clamp(min=1e-6, max=1.0 - 1e-6)
            return (math.log(0.5) / torch.log(1.0 - g))

        return {
            "W_gamma_mean": W_gamma.mean().item(),
            "W_gamma_max": W_gamma.max().item(),
            "W_gamma_min": W_gamma.min().item(),
            "W_gamma_std": W_gamma.std().item(),
            "W_half_life": _hl(W_gamma).mean().item(),
            "decay_gamma_mean": d_gamma.mean().item(),
            "decay_gamma_max": d_gamma.max().item(),
            "decay_half_life": _hl(d_gamma).mean().item(),
            "hebbian_gamma_mean": h_gamma.mean().item(),
            "hebbian_half_life": _hl(h_gamma).mean().item(),
        }

    def compute_mod_grad_norm(self) -> float:
        """Mean L2 norm of modulator (conv encoder) gradients."""
        norms = []
        for p in self.modulator.parameters():
            if p.grad is None:
                continue
            norms.append(p.grad.detach().float().norm())
        if not norms:
            return 0.0
        return torch.stack(norms).mean().item()

    def compute_component_grad_norms(self) -> dict:
        out = {}
        for name, p in (
            ("grad_conv_stem", self.modulator.stem.weight),
            ("grad_logit_head", self.modulator.logit_head.weight),
            ("grad_codebook", self.discrete_policy.codebook),
            ("grad_dec_init_proj", self.decoder.init_proj.weight),
            ("grad_dW_head", self.decoder.dW_head.weight),
            ("grad_state_w1", self.state_w1),
            ("grad_msg_w1", self.msg_w1),
            ("grad_inject_w", self.inject_w),
            ("grad_neuron_id", self.neuron_id),
        ):
            out[name] = (
                0.0 if p.grad is None
                else p.grad.detach().float().norm().item())
        return out

    def compute_param_norms(self) -> dict:
        return {
            "conv_stem_norm": self.modulator.stem.weight.detach().float().norm().item(),
            "logit_head_norm": self.modulator.logit_head.weight.detach().float().norm().item(),
            "codebook_norm": self.discrete_policy.codebook.detach().float().norm().item(),
            "dec_init_norm": self.decoder.init_proj.weight.detach().float().norm().item(),
            "dW_head_norm": self.decoder.dW_head.weight.detach().float().norm().item(),
            "state_w1_norm": self.state_w1.detach().float().norm().item(),
            "msg_w1_norm": self.msg_w1.detach().float().norm().item(),
            "inject_w_norm": self.inject_w.detach().float().norm().item(),
            "neuron_id_norm": self.neuron_id.detach().float().norm().item(),
        }

    # ================================================================
    # Per-step components
    # ================================================================

    def _receive(self, msg: Tensor, W: Tensor) -> Tensor:
        """Dense message passing: [BS, N, N] @ [BS, N, D_n] → [BS, N, D_n]."""
        return torch.matmul(W, msg)

    def _inject(self, received: Tensor, H_mid_t: Tensor,
                inject_w: Tensor, inject_b: Tensor) -> Tensor:
        """Project H_mid[:, D_n*p : D_n*(p+1)] into pool p's input ports."""
        BS = H_mid_t.shape[0]
        # H_mid_t reshaped as [BS, NC_pools, D_n].
        slices = H_mid_t.reshape(BS, self.NC_pools, self.D_n)
        # Project per pool: [BS, NC_pools, D_n] × [NC_pools, alpha*D_n, D_n]
        # → [BS, NC_pools, alpha*D_n].
        inject = torch.einsum("bpi,poi->bpo", slices, inject_w)
        inject = inject + inject_b.unsqueeze(0)
        # Reshape to [BS, NC_pools, alpha, D_n].
        inject = inject.reshape(BS, self.NC_pools, self.alpha, self.D_n)
        # Scatter into input-port neurons. input_port_idx: [NC_pools, alpha].
        idx = self.input_port_idx.to(received.device)
        # Flatten pool+alpha axes for indexing: [NC_pools*alpha]
        flat_idx = idx.reshape(-1)
        flat_inject = inject.reshape(BS, self.NC_pools * self.alpha, self.D_n)
        received.index_add_(1, flat_idx, flat_inject.to(received.dtype))
        return received

    def _mlp2(self, x: Tensor, w1: Tensor, b1: Tensor,
              w2: Tensor, b2: Tensor) -> Tensor:
        """2-layer tanh MLP."""
        flat = x.reshape(-1, x.shape[-1])
        hidden = torch.tanh(F.linear(flat, w1, b1))
        out = torch.tanh(F.linear(hidden, w2, b2))
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    def _state_update(self, received: Tensor, h: Tensor,
                      decay: Tensor, one_minus_decay: Tensor,
                      w1: Tensor, b1: Tensor,
                      w2: Tensor, b2: Tensor) -> Tensor:
        """state MLP with decay-gated integration: cat(received, h) → MLP → blend."""
        inp = torch.cat([received, h], dim=-1)  # [BS, N, 2*D_n]
        candidate = self._mlp2(inp, w1, b1, w2, b2)
        return decay * h + one_minus_decay * candidate

    def _emit_message(self, h: Tensor, w1: Tensor, b1: Tensor,
                      w2: Tensor, b2: Tensor) -> Tensor:
        msg_new = self._mlp2(h, w1, b1, w2, b2)
        return msg_new + self.neuron_id.to(h.dtype).unsqueeze(0)

    def _readout(self, msg: Tensor) -> Tensor:
        """Pool output ports per pool, concat across pools → [BS, D]."""
        BS = msg.shape[0]
        out_idx = self.output_port_idx.to(msg.device)  # [NC_pools, alpha]
        # Gather output-port msgs: [BS, NC_pools, alpha, D_n]
        flat_idx = out_idx.reshape(-1)
        gathered = msg.index_select(1, flat_idx)
        gathered = gathered.reshape(
            BS, self.NC_pools, self.alpha, self.D_n)
        # Sum over alpha, normalize by sqrt(alpha), concat across pools.
        pooled = gathered.sum(dim=2) * (self.alpha ** -0.5)  # [BS, NC_pools, D_n]
        return pooled.reshape(BS, self.NC_pools * self.D_n)  # [BS, D]

    def _modulate(self, h: Tensor, msg: Tensor, received: Tensor,
                  W: Tensor, hebbian: Tensor, decay: Tensor,
                  s_live: Tensor, s_ema: Tensor,
                  W_gamma: Tensor, decay_gamma: Tensor,
                  phase: str) -> tuple[Tensor, Tensor]:
        """Conv-grid encoder → discrete policy → decoder → apply EMA blend."""
        # Encoder: observation → code logits.
        logits = self.modulator(
            h, msg, received, W, hebbian, decay,
            s_live, s_ema, self.role_id.to(h.device))

        # Sample code and look up embedding.
        if self.training and phase == "phase1":
            soft, codes = self.discrete_policy.sample_gumbel_soft(
                logits, tau=self.gumbel_tau, hard=True)
            emb = self.discrete_policy.lookup_soft(soft)
        elif not self.training:
            # Deterministic argmax in eval for reproducibility.
            codes = logits.argmax(dim=-1)
            emb = self.discrete_policy.lookup(codes)
        else:
            # phase2 / rollouts: hard categorical.
            codes, _ = self.discrete_policy.sample_discrete(
                logits, tau=self.gumbel_tau)
            emb = self.discrete_policy.lookup(codes)

        if self.training and phase == "phase1":
            self.discrete_policy.update_usage(codes)

        # Decoder: embedding → (ΔW_normed, Δdecay_raw).
        if self.training and self.config.checkpoint_decoder:
            dW_normed, dDecay_raw = torch.utils.checkpoint.checkpoint(
                self.decoder, emb, use_reentrant=False)
        else:
            dW_normed, dDecay_raw = self.decoder(emb)

        # EMA blend in bf16. γ clamped to gamma_max to keep (1-γ) ≥ 0.03 safe.
        # [N] → broadcast to [BS, N, 1] for W and [BS, N] for decay.
        W_new = ((1 - W_gamma.unsqueeze(0).unsqueeze(-1)) * W
                 + W_gamma.unsqueeze(0).unsqueeze(-1) * dW_normed.to(W.dtype))
        target_decay = torch.sigmoid(dDecay_raw).to(decay.dtype)
        decay_new = ((1 - decay_gamma.unsqueeze(0)) * decay
                     + decay_gamma.unsqueeze(0) * target_decay)
        return W_new, decay_new

    @staticmethod
    def _hebbian_update(hebbian: Tensor, msg: Tensor, gamma: Tensor) -> Tensor:
        """Convex EMA on msg @ msgᵀ. All bf16 (γ clamped in caller)."""
        coactiv = torch.matmul(msg, msg.transpose(-1, -2))  # [BS, N, N]
        g = gamma.view(1, -1, 1)   # [1, N, 1] — per-neuron (receiver) γ
        return (1 - g) * hebbian + g * coactiv

    # ================================================================
    # Per-block loop
    # ================================================================

    def _run_block(self, h, msg, W, decay, hebbian,
                   s_mem_live, s_mem_ema_fast,
                   prev_readout, readout_drift,
                   block_H_mid, block_input_ids,
                   start_t,
                   inject_w, inject_b,
                   st_w1, st_b1, st_w2, st_b2,
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                   hebbian_gamma, W_gamma, decay_gamma, gain_fast):
        """Run one TBPTT block of memory steps with inline modulation + surprise."""
        block_T = block_H_mid.shape[1]
        BS = h.shape[0]
        D = self.config.D
        D_n = self.D_n
        NC_pools = self.NC_pools
        readouts = torch.empty(BS, block_T, D,
                                device=block_H_mid.device, dtype=h.dtype)

        decay_gate = decay.unsqueeze(-1)             # [BS, N, 1]
        one_minus_gate = 1.0 - decay_gate

        for offset in range(block_T):
            t = start_t + offset
            H_mid_t = block_H_mid[:, offset]           # [BS, D]
            tok_t = block_input_ids[:, offset]

            # TBPTT detach at the 0-th position of a block (except the very
            # first block which isn't preceded by a detach).
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                W = W.detach()
                decay = decay.detach()
                hebbian = hebbian.detach()
                prev_readout = prev_readout.detach()
                decay_gate = decay.unsqueeze(-1)
                one_minus_gate = 1.0 - decay_gate

            # Live surprise (no_grad observation).
            with torch.no_grad():
                x = prev_readout
                if proj_down_w is not None:
                    x = F.linear(x, proj_down_w, proj_down_b)
                x = F.layer_norm(x, (x.shape[-1],), ln_final_w, ln_final_b)
                logits_full = F.linear(x, lm_head_w)
                lse = torch.logsumexp(logits_full.float(), dim=-1)
                target_logit = logits_full.gather(
                    1, tok_t.unsqueeze(1)).squeeze(1).float()
                s_mem_live = (lse - target_logit).to(h.dtype)
                s_mem_ema_fast = (
                    (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live)

            # Modulator fires every modulation_interval tokens.
            if t % self.config.modulation_interval == 0:
                # Compute received for modulator observation (same W @ msg we
                # need for the step below; share the work).
                received = self._receive(msg, W)
                W, decay = self._modulate(
                    h, msg, received, W, hebbian, decay,
                    s_mem_live, s_mem_ema_fast,
                    W_gamma, decay_gamma, phase="phase1")
                decay_gate = decay.unsqueeze(-1)
                one_minus_gate = 1.0 - decay_gate

            # Memory step.
            received = self._receive(msg, W)
            received = self._inject(received, H_mid_t, inject_w, inject_b)
            h = self._state_update(received, h,
                                    decay_gate, one_minus_gate,
                                    st_w1, st_b1, st_w2, st_b2)
            msg = self._emit_message(h, mg_w1, mg_b1, mg_w2, mg_b2)
            readout = self._readout(msg)               # [BS, D]
            readouts[:, offset] = readout

            # Hebbian trace update (on autograd graph).
            hebbian = self._hebbian_update(hebbian, msg, hebbian_gamma)

            # Per-pool readout drift (no_grad).
            with torch.no_grad():
                new_pool = readout.reshape(BS, NC_pools, D_n)
                prev_pool = prev_readout.reshape(BS, NC_pools, D_n)
                readout_drift = (new_pool - prev_pool).abs().mean(
                    dim=-1, keepdim=True).to(h.dtype)

            prev_readout = readout.detach()

        return (h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout, readout_drift, readouts)

    # ================================================================
    # Segment forward
    # ================================================================

    def forward_segment(
        self,
        H_mid: Tensor,
        input_ids: Tensor,
        lm,
        prev_token: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process T tokens through memory. Returns (readouts, mem_pred_loss)."""
        BS, T, _ = H_mid.shape
        if not self._initialized:
            self.initialize_states(BS, H_mid.device)

        # Lazily torch.compile the block loop on CUDA (skip if disabled).
        if (not self._compiled and H_mid.is_cuda
                and not os.environ.get("NEUROMORPHIC_NO_COMPILE")):
            self._run_block_uncompiled = self._run_block
            self._run_block = torch.compile(
                self._run_block, mode="default", fullgraph=False)
            self._compiled = True

        # Valid mask for mem_pred_loss target (cross-document transitions masked).
        eot = self.config.eot_id
        valid_mask = torch.ones(BS, T, device=H_mid.device, dtype=H_mid.dtype)
        if T > 1:
            valid_mask[:, 1:] = (input_ids[:, :-1] != eot).to(H_mid.dtype)
        if prev_token is not None:
            valid_mask[:, 0] = (prev_token.to(input_ids.device) != eot).to(H_mid.dtype)
        else:
            valid_mask[:, 0] = 0.0

        h = self.h
        msg = self.msg
        W = self.W
        decay = self.decay
        hebbian = self.hebbian
        s_mem_live = self.s_mem_live
        s_mem_ema_fast = self.s_mem_ema_fast
        prev_readout = self.prev_readout
        readout_drift = self.readout_drift
        segment_start_prev_readout = prev_readout.detach().clone()
        H_mid = H_mid.to(h.dtype)

        dt = h.dtype
        inject_w = self.inject_w.to(dt)
        inject_b = self.inject_b.to(dt)
        st_w1 = self.state_w1.to(dt)
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)
        lm_head_w = lm.lm_head.weight.to(dt)
        if lm.proj_down is not None:
            proj_down_w = lm.proj_down.weight.to(dt)
            proj_down_b = lm.proj_down.bias.to(dt)
        else:
            proj_down_w = None
            proj_down_b = None
        ln_final_w = lm.ln_final.weight.to(dt)
        ln_final_b = lm.ln_final.bias.to(dt)

        # Per-neuron learnable γ, clamped to [0, gamma_max].
        gm = self.gamma_max
        hebbian_gamma = gm * torch.sigmoid(self.hebbian_decay_logit).to(dt)
        W_gamma = gm * torch.sigmoid(self.W_decay_logit).to(dt)
        decay_gamma = gm * torch.sigmoid(self.decay_gamma_logit).to(dt)

        readouts = torch.empty(BS, T, self.config.D,
                                device=H_mid.device, dtype=dt)
        block_size = max(1, self.config.tbptt_block)
        gain_fast = self.config.gain_ema_fast

        use_ckpt = self.training and self.config.checkpoint_memory
        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_mid = H_mid[:, start_t:end_t]
            block_input_ids = input_ids[:, start_t:end_t]

            block_args = (
                h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout, readout_drift,
                block_H_mid, block_input_ids,
                start_t,
                inject_w, inject_b,
                st_w1, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                hebbian_gamma, W_gamma, decay_gamma, gain_fast)

            if use_ckpt:
                result = torch.utils.checkpoint.checkpoint(
                    self._run_block, *block_args, use_reentrant=False)
            else:
                result = self._run_block(*block_args)

            (h, msg, W, decay, hebbian,
             s_mem_live, s_mem_ema_fast,
             prev_readout, readout_drift, block_out) = result

            readouts[:, start_t:end_t] = block_out

        # Save state (detach defensively).
        self.h = h.detach()
        self.msg = msg.detach()
        self.W = W.detach()
        self.decay = decay.detach()
        self.hebbian = hebbian.detach()
        self.s_mem_live = s_mem_live.detach()
        self.s_mem_ema_fast = s_mem_ema_fast.detach()
        self.prev_readout = prev_readout.detach()
        self.readout_drift = readout_drift.detach()

        # Segment-level mem_pred_loss (memory head predicts token[t] from readout[t-1]).
        shifted_all = torch.cat([
            segment_start_prev_readout.unsqueeze(1).to(readouts.dtype),
            readouts[:, :-1],
        ], dim=1)  # [BS, T, D]

        loss_sum = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        valid_total = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        chunk = block_size
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_readout = shifted_all[:, s:e]
            sub_target = input_ids[:, s:e]
            sub_valid = valid_mask[:, s:e].float()
            sub_logits = lm.mem_head_logits(sub_readout)
            per_tok_loss = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                sub_target.reshape(-1),
                reduction="none",
            ).reshape(sub_target.shape)
            loss_sum = loss_sum + (per_tok_loss * sub_valid).sum()
            valid_total = valid_total + sub_valid.sum()
        mem_pred_loss = loss_sum / valid_total.clamp(min=1.0)

        return readouts, mem_pred_loss
