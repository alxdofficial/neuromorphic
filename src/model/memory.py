"""Multi-cell memory graph with attention neuromodulator.

N_cells connectivity pools × neurons_per_cell each. State tensors carry
explicit NC dim: h, msg are [BS, NC, Nc, D_n]; W, hebbian are
[BS, NC, Nc, Nc]. Each cell has its own block-diagonal W (no cross-cell
edges at W level). Cross-cell mixing happens via:
  - the modulator (attention can see all cells jointly, if configured)
  - LM readout / inject through its scan layers

bf16 throughout; γ clamped to config.gamma_max.
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
from .attention_modulator import AttentionModulator, DirectDecoder, port_layout
from .triton_memory_step import fused_memory_step


class MemoryGraph(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.D_n = config.D_n
        self.NC = config.N_cells                  # number of cells
        self.Nc = config.neurons_per_cell         # neurons per cell
        self.N = self.NC * self.Nc                # total neurons
        self.alpha = config.alpha
        self.NC_pools = config.NC_pools           # = N_cells for default

        # Sanity: for simplicity require NC_pools == N_cells (enforced in
        # Config.validate for the default path).
        assert self.NC_pools == self.NC or self.NC % self.NC_pools == 0

        # Per-neuron identity vector, per cell.
        self.neuron_id = nn.Parameter(
            torch.randn(self.NC, self.Nc, self.D_n) * (self.D_n ** -0.5))

        # Msg MLP (2-layer tanh, shared) — fires event-driven at msg_interval.
        # State update itself has NO learned MLP; it's a LIF-style leaky
        # integrator: h = tanh(decay*h + (1-decay)*received). Plasticity on
        # the state timescale lives in `decay_gamma_logit` per-neuron.
        Hm = config.msg_mlp_hidden
        self.msg_w1 = nn.Parameter(torch.empty(Hm, self.D_n))
        self.msg_b1 = nn.Parameter(torch.zeros(Hm))
        self.msg_w2 = nn.Parameter(torch.empty(self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(self.D_n))

        # Per-cell inject projection. H_mid is split across NC_pools slices;
        # each slice maps to one cell (assuming NC_pools == NC for default).
        self.inject_w = nn.Parameter(
            torch.empty(self.NC, self.alpha * self.D_n, self.D_n))
        self.inject_b = nn.Parameter(
            torch.zeros(self.NC, self.alpha * self.D_n))

        # Per-cell per-neuron plasticity logits: [NC, Nc].
        self.W_decay_logit = nn.Parameter(torch.full((self.NC, self.Nc), -3.0))
        self.decay_gamma_logit = nn.Parameter(torch.full((self.NC, self.Nc), -3.0))
        self.hebbian_decay_logit = nn.Parameter(torch.full((self.NC, self.Nc), 2.0))
        self.gamma_max = config.gamma_max

        # Modulator + decoder + codebook.
        self.modulator = AttentionModulator(config)
        self.decoder = DirectDecoder(config)
        self.discrete_policy = DiscreteActionPolicy(
            num_codes=config.num_codes, code_dim=config.code_dim)

        self.gumbel_tau = 1.0

        # Port-layout buffers (per-cell indices).
        layout = port_layout(config)
        self.register_buffer("input_port_idx", layout["input_port_idx"])
        self.register_buffer("output_port_idx", layout["output_port_idx"])
        self.register_buffer("role_id", layout["role_id"].view(self.NC, self.Nc))
        # Pre-built dense output-port mask for the fused step.
        # mask[c, n] = 1 if neuron n in cell c is an output port, else 0.
        out_mask = torch.zeros(self.NC, self.Nc)
        for c in range(self.NC):
            out_mask[c, layout["output_port_idx"][c]] = 1.0
        self.register_buffer("out_port_mask", out_mask)
        self.readout_scale = self.alpha ** -0.5

        # Init (Xavier with tanh gain where appropriate).
        TANH_GAIN = 5.0 / 3.0
        nn.init.xavier_uniform_(self.msg_w1, gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.msg_w2, gain=TANH_GAIN)

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
        dt = torch.bfloat16 if device.type == "cuda" else torch.float32
        NC, Nc, D_n = self.NC, self.Nc, self.D_n

        self.h = torch.zeros(BS, NC, Nc, D_n, device=device, dtype=dt)
        self.msg = torch.zeros(BS, NC, Nc, D_n, device=device, dtype=dt)
        self.decay = torch.full((BS, NC, Nc), 0.5, device=device, dtype=dt)
        self.hebbian = torch.zeros(BS, NC, Nc, Nc, device=device, dtype=dt)

        # Sparse init for W: ~min(8, Nc-1) random nonzeros per row, RMS-normed.
        K_init = min(8, Nc - 1)
        W = torch.zeros(BS, NC, Nc, Nc, device=device, dtype=dt)
        for c in range(NC):
            for neuron in range(Nc):
                neighbors = torch.randperm(Nc, device=device)[:K_init]
                W[:, c, neuron, neighbors] = 1.0
        W.diagonal(dim1=-2, dim2=-1).zero_()
        self.W = F.rms_norm(W, normalized_shape=(Nc,))

        self.s_mem_live = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(BS, device=device, dtype=dt)
        self.readout_drift = torch.zeros(BS, NC, 1, device=device, dtype=dt)
        self.prev_readout = torch.zeros(
            BS, self.config.D, device=device, dtype=dt)

        self._initialized = True

    def detach_states(self):
        if not self._initialized:
            return
        for attr in ("h", "msg", "W", "decay", "hebbian",
                     "prev_readout", "readout_drift"):
            setattr(self, attr, getattr(self, attr).detach())

    @torch.no_grad()
    def resize_to_bs(self, new_bs: int):
        if not self._initialized:
            return
        device = self.W.device
        dt = self.W.dtype
        old_bs = self.W.shape[0]
        if new_bs == old_bs:
            return
        idx = (torch.randperm(old_bs, device=device)[:new_bs] if new_bs < old_bs
               else torch.arange(new_bs, device=device) % old_bs)
        self.W = self.W[idx].clone()
        self.decay = self.decay[idx].clone()
        self.hebbian = self.hebbian[idx].clone()
        NC, Nc, D_n = self.NC, self.Nc, self.D_n
        self.h = torch.zeros(new_bs, NC, Nc, D_n, device=device, dtype=dt)
        self.msg = torch.zeros(new_bs, NC, Nc, D_n, device=device, dtype=dt)
        self.s_mem_live = torch.zeros(new_bs, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(new_bs, device=device, dtype=dt)
        self.prev_readout = torch.zeros(
            new_bs, self.config.D, device=device, dtype=dt)
        self.readout_drift = torch.zeros(
            new_bs, NC, 1, device=device, dtype=dt)

    def runtime_state_dict(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized": True,
            **{k: getattr(self, k).clone() for k in (
                "h", "msg", "W", "decay", "hebbian",
                "s_mem_live", "s_mem_ema_fast", "prev_readout", "readout_drift")},
        }

    def load_runtime_state(self, state: dict):
        if not state or not state.get("initialized", False):
            self._initialized = False
            return
        loaded_h = state.get("h")
        if loaded_h is None or loaded_h.ndim != 4:
            raise RuntimeError(
                f"Runtime state incompatible: expected 4-D h [BS, NC, Nc, D_n]; "
                f"got shape {tuple(loaded_h.shape) if loaded_h is not None else None}")
        device = self.neuron_id.device
        for k in ("h", "msg", "W", "decay", "hebbian"):
            setattr(self, k, state[k].to(device))
        BS = self.h.shape[0]
        dt = self.h.dtype
        zero_b = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_live = state.get("s_mem_live", zero_b).to(device)
        self.s_mem_ema_fast = state.get("s_mem_ema_fast", zero_b).to(device)
        self.prev_readout = state.get("prev_readout",
            torch.zeros(BS, self.config.D, device=device, dtype=dt)).to(device)
        self.readout_drift = state.get("readout_drift",
            torch.zeros(BS, self.NC, 1, device=device, dtype=dt)).to(device)
        self._initialized = True

    # ================================================================
    # Telemetry (reduced for speed; core metrics only)
    # ================================================================

    @torch.no_grad()
    def compute_lane_divergence(self) -> dict:
        if not self._initialized or self.W.shape[0] <= 1:
            return {}
        # Frobenius norm per-batch-element over (NC, Nc, Nc) via flatten.
        W_mean = self.W.mean(dim=0, keepdim=True)
        W_div = (self.W - W_mean).float().flatten(1).norm(dim=-1).mean().item()
        W_norm = self.W.float().flatten(1).norm(dim=-1).mean().item()
        return {"lane_W_divergence": W_div,
                "lane_W_relative_div": W_div / max(W_norm, 1e-8)}

    @torch.no_grad()
    def compute_memory_health(self) -> dict:
        if not self._initialized:
            return {}
        h, msg, W, decay = (t.float() for t in (self.h, self.msg, self.W, self.decay))
        hebbian = self.hebbian.float()

        # Compute honest off-diagonal stats. The decoder already zeroes W's
        # diagonal, so W_offdiag ≈ W, but the distinction is still meaningful
        # for hebbian (which is msg·msgᵀ including the diagonal).
        Nc = W.shape[-1]
        off_mask = (1.0 - torch.eye(Nc, device=W.device, dtype=W.dtype)).unsqueeze(0).unsqueeze(0)
        W_off = W * off_mask
        hebbian_off = hebbian * off_mask

        # Cosine(W, hebbian) on the off-diagonal flattened vector — diagnostic
        # of whether the modulator's learned W aligns with the Hebbian trace.
        wf = W_off.flatten()
        hf = hebbian_off.flatten()
        denom = (wf.norm() * hf.norm()).clamp(min=1e-8)
        W_hebbian_off_cos = (wf @ hf / denom).item()

        return {
            "h_norm": h.norm().item() / max(h.numel() ** 0.5, 1.0),
            "msg_norm": msg.norm().item() / max(msg.numel() ** 0.5, 1.0),
            "h_max": h.abs().max().item(),
            "msg_max": msg.abs().max().item(),
            "decay_mean": decay.mean().item(),
            "decay_std": decay.std().item(),
            "s_mem_live": self.s_mem_live.float().mean().item(),
            "s_mem_ema_fast": self.s_mem_ema_fast.float().mean().item(),
            "readout_drift_mean": self.readout_drift.float().mean().item(),
            "W_norm": W.norm().item() / max(W.numel() ** 0.5, 1.0),
            "W_max": W.abs().max().item(),
            "W_offdiag_norm": W_off.norm().item() / max(W_off.numel() ** 0.5, 1.0),
            "W_offdiag_max": W_off.abs().max().item(),
            "hebbian_norm": hebbian.norm().item() / max(hebbian.numel() ** 0.5, 1.0),
            "hebbian_offdiag_norm": hebbian_off.norm().item()
                                    / max(hebbian_off.numel() ** 0.5, 1.0),
            "W_hebbian_offdiag_cos": W_hebbian_off_cos,
        }

    @torch.no_grad()
    def compute_plasticity_rates(self) -> dict:
        gm = self.gamma_max
        Wg = gm * torch.sigmoid(self.W_decay_logit).float()
        Dg = gm * torch.sigmoid(self.decay_gamma_logit).float()
        Hg = gm * torch.sigmoid(self.hebbian_decay_logit).float()
        return {
            "W_gamma_mean": Wg.mean().item(),
            "W_gamma_max": Wg.max().item(),
            "W_gamma_min": Wg.min().item(),
            "W_gamma_std": Wg.std().item(),
            "W_half_life": 0.0,  # skip for speed
            "decay_gamma_mean": Dg.mean().item(),
            "decay_gamma_max": Dg.max().item(),
            "decay_half_life": 0.0,
            "hebbian_gamma_mean": Hg.mean().item(),
            "hebbian_half_life": 0.0,
        }

    def compute_mod_grad_norm(self) -> float:
        norms = [p.grad.detach().float().norm()
                 for p in self.modulator.parameters() if p.grad is not None]
        return torch.stack(norms).mean().item() if norms else 0.0

    def compute_component_grad_norms(self) -> dict:
        out = {}
        for name, p in (
            ("grad_tok_proj", self.modulator.tok_proj[0].weight),
            ("grad_logit_head", self.modulator.logit_head.weight),
            ("grad_cell_emb", self.modulator.cell_emb),
            ("grad_codebook", self.discrete_policy.codebook),
            ("grad_decoder", self.decoder.mlp[-1].weight),
            ("grad_decay_gamma_logit", self.decay_gamma_logit),
            ("grad_msg_w1", self.msg_w1),
            ("grad_inject_w", self.inject_w),
            ("grad_neuron_id", self.neuron_id),
        ):
            out[name] = 0.0 if p.grad is None else p.grad.detach().float().norm().item()
        return out

    def compute_param_norms(self) -> dict:
        return {
            "tok_proj_norm": self.modulator.tok_proj[0].weight.detach().float().norm().item(),
            "logit_head_norm": self.modulator.logit_head.weight.detach().float().norm().item(),
            "cell_emb_norm": self.modulator.cell_emb.detach().float().norm().item(),
            "codebook_norm": self.discrete_policy.codebook.detach().float().norm().item(),
            "decoder_norm": self.decoder.mlp[-1].weight.detach().float().norm().item(),
            "decay_gamma_logit_norm": self.decay_gamma_logit.detach().float().norm().item(),
            "msg_w1_norm": self.msg_w1.detach().float().norm().item(),
            "inject_w_norm": self.inject_w.detach().float().norm().item(),
            "neuron_id_norm": self.neuron_id.detach().float().norm().item(),
        }

    # ================================================================
    # Per-step components
    # ================================================================

    def _receive(self, msg: Tensor, W: Tensor) -> Tensor:
        """Block-diagonal message passing: [BS, NC, Nc, Nc] @ [BS, NC, Nc, D_n]."""
        return torch.matmul(W, msg)

    def _inject(self, received: Tensor, H_mid_t: Tensor,
                inject_w: Tensor, inject_b: Tensor) -> Tensor:
        """H_mid_t [BS, D] → per-cell D_n slice → project → cell's input ports."""
        BS = H_mid_t.shape[0]
        NC = self.NC_pools
        # H_mid reshapes to [BS, NC_pools, D_n]. For default NC_pools == NC.
        slices = H_mid_t.reshape(BS, NC, self.D_n)
        # Per-cell inject projection [NC, alpha*D_n, D_n].
        inject = torch.einsum("bci,coi->bco", slices, inject_w)
        inject = inject + inject_b.unsqueeze(0)
        inject = inject.reshape(BS, NC, self.alpha, self.D_n)
        # Scatter into input port neurons. input_port_idx: [NC, alpha] local indices.
        idx = self.input_port_idx.to(received.device)     # [NC, alpha] (local)
        # received: [BS, NC, Nc, D_n]. For each cell c, add inject[:, c, :, :] at
        # positions idx[c, :].
        # Use gather-style: create a padded add tensor [BS, NC, Nc, D_n] and add.
        # Simpler: loop over alpha port positions (alpha is small, usually 4).
        for a in range(self.alpha):
            # idx[:, a] is [NC] long — port index within each cell
            # received[b, c, idx[c, a], :] += inject[b, c, a, :]
            # Batched via scatter_add_:
            # First expand idx to [BS, NC, 1, D_n]
            port_idx = idx[:, a].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                BS, NC, 1, self.D_n)
            received.scatter_add_(
                2, port_idx,
                inject[:, :, a, :].unsqueeze(2).to(received.dtype))
        return received

    @staticmethod
    def _state_update(received: Tensor, h: Tensor,
                      decay_gate: Tensor, one_minus_gate: Tensor) -> Tensor:
        """LIF-style leaky integrator: h = tanh(decay*h + (1-decay)*received).

        Pure elementwise. No state MLP — per-neuron learned knobs live in
        `decay_gamma_logit` (per-cell, per-neuron timescales). The saturating
        tanh serves as the soma activation / spike threshold surrogate.
        """
        return torch.tanh(decay_gate * h + one_minus_gate * received)

    def _emit_message(self, h: Tensor, w1: Tensor, b1: Tensor,
                      w2: Tensor, b2: Tensor) -> Tensor:
        flat = h.reshape(-1, h.shape[-1])
        hid = torch.tanh(F.linear(flat, w1, b1))
        msg_new = torch.tanh(F.linear(hid, w2, b2)).reshape(h.shape)
        # neuron_id is [NC, Nc, D_n]; broadcast over BS.
        return msg_new + self.neuron_id.to(h.dtype).unsqueeze(0)

    def _readout(self, h: Tensor) -> Tensor:
        """Per-cell output-port pool of h (membrane potential) → [BS, D].

        Pooling from h (not msg) so the LM sees the fresh per-token membrane
        state. msg is event-driven and would be piecewise-constant across
        the token interval between spike events.
        """
        BS, NC, _, D_n = h.shape
        out_idx = self.output_port_idx.to(h.device)      # [NC, alpha] local
        idx_exp = out_idx.unsqueeze(-1).expand(NC, self.alpha, D_n)
        idx_exp = idx_exp.unsqueeze(0).expand(BS, NC, self.alpha, D_n).contiguous()
        gathered = h.gather(2, idx_exp)                  # [BS, NC, alpha, D_n]
        pooled = gathered.sum(dim=2) * (self.alpha ** -0.5)
        return pooled.reshape(BS, NC * D_n)

    def _modulate(self, h: Tensor, msg: Tensor, received: Tensor,
                  W: Tensor, hebbian: Tensor, decay: Tensor,
                  s_live: Tensor, s_ema: Tensor,
                  W_gamma: Tensor, decay_gamma: Tensor,
                  phase: str) -> tuple[Tensor, Tensor]:
        """Per-cell attention encoder → per-cell codes → per-cell ΔW."""
        # logits: [BS, NC, K]
        logits, _tokens = self.modulator(
            h, msg, received, W, hebbian, decay,
            s_live, s_ema, self.role_id.to(h.device))

        # Per-cell sampling. Flatten [BS*NC] for discrete_policy operations,
        # reshape back after.
        BS, NC, K = logits.shape
        logits_flat = logits.reshape(BS * NC, K)
        if self.training and phase == "phase1":
            soft, codes = self.discrete_policy.sample_gumbel_soft(
                logits_flat, tau=self.gumbel_tau, hard=True)
            emb = self.discrete_policy.lookup_soft(soft)       # [BS*NC, D_code]
        elif not self.training:
            codes = logits_flat.argmax(dim=-1)
            emb = self.discrete_policy.lookup(codes)
        else:
            codes, _ = self.discrete_policy.sample_discrete(
                logits_flat, tau=self.gumbel_tau)
            emb = self.discrete_policy.lookup(codes)

        if self.training and phase == "phase1":
            self.discrete_policy.update_usage(codes)

        # Decoder is a shared trunk conditioned on the modulator's cell_emb.
        # emb [BS*NC, D_code] → reshape [BS, NC, D_code] → returns (ΔW [BS, NC, Nc, Nc], Δdecay [BS, NC, Nc]).
        emb_3d = emb.reshape(BS, NC, -1)
        dW_normed, dDecay_raw = self.decoder(emb_3d, self.modulator.cell_emb)

        # EMA blend. W_gamma is [NC, Nc] — broadcast to [BS, NC, Nc, Nc] via unsqueeze(-1).
        W_new = ((1 - W_gamma.unsqueeze(0).unsqueeze(-1)) * W
                 + W_gamma.unsqueeze(0).unsqueeze(-1) * dW_normed.to(W.dtype))
        target_decay = torch.sigmoid(dDecay_raw).to(decay.dtype)
        decay_new = ((1 - decay_gamma.unsqueeze(0)) * decay
                     + decay_gamma.unsqueeze(0) * target_decay)
        return W_new, decay_new

    @staticmethod
    def _hebbian_update(hebbian: Tensor, msg: Tensor, gamma: Tensor) -> Tensor:
        """msg @ msgᵀ per cell; γ is [NC, Nc] per-receiver. All bf16."""
        coactiv = torch.matmul(msg, msg.transpose(-1, -2))  # [BS, NC, Nc, Nc]
        g = gamma.unsqueeze(0).unsqueeze(-1)  # [1, NC, Nc, 1]
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
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                   hebbian_gamma, W_gamma, decay_gamma, gain_fast):
        """Multi-timescale memory loop.

        Per-token (fast clock): W @ msg message passing + inject + LIF state
        update + readout from h. Pure elementwise + one bmm.
        Event (msg_interval, default 4): msg = MLP(h), hebbian EMA update.
        Slow event (modulation_interval, default 16): neuromodulator fires,
        emits ΔW and Δdecay via codebook+decoder.
        """
        block_T = block_H_mid.shape[1]
        BS = h.shape[0]
        D = self.config.D
        readouts = torch.empty(BS, block_T, D,
                                device=block_H_mid.device, dtype=h.dtype)

        msg_interval = self.config.msg_interval
        mod_interval = self.config.modulation_interval

        for offset in range(block_T):
            t = start_t + offset
            H_mid_t = block_H_mid[:, offset]
            tok_t = block_input_ids[:, offset]

            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach(); msg = msg.detach(); W = W.detach()
                decay = decay.detach(); hebbian = hebbian.detach()
                prev_readout = prev_readout.detach()

            # Surprise signal from previous readout — always per-token so
            # the modulator sees the freshest EMA whenever it fires.
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
                s_mem_ema_fast = ((1 - gain_fast) * s_mem_ema_fast
                                   + gain_fast * s_mem_live)

            # Per-token hot path: FUSED in one Triton kernel (forward+backward).
            # 1) Compute inject projection from H_mid slice (small, per-cell).
            slices = H_mid_t.reshape(BS, self.NC, self.D_n)
            inject_proj = torch.einsum("bci,coi->bco", slices, inject_w)
            inject_proj = inject_proj + inject_b.unsqueeze(0)
            inject_proj = inject_proj.reshape(BS, self.NC, self.alpha, self.D_n)
            # 2) Fused step: W@msg + in-kernel inject add + LIF + readout pool.
            #    Inject is added to the first α rows inside the kernel — no
            #    dense materialization/scatter on the Python side. Requires
            #    port_layout to put input ports at local indices [0, α).
            h, readout_3d = fused_memory_step(
                h, msg, W, decay, inject_proj, self.out_port_mask,
                self.readout_scale)
            readout = readout_3d.reshape(BS, self.NC * self.D_n)
            readouts[:, offset] = readout

            # Event — end of msg_interval block: refresh msg (spike emission),
            # update Hebbian co-activation trace.
            if (t + 1) % msg_interval == 0:
                msg = self._emit_message(h, mg_w1, mg_b1, mg_w2, mg_b2)
                hebbian = self._hebbian_update(hebbian, msg, hebbian_gamma)

            # Slow event — end of modulation_interval block: neuromodulator
            # writes plasticity updates to W and decay.
            if (t + 1) % mod_interval == 0:
                received_for_mod = self._receive(msg, W)
                W, decay = self._modulate(
                    h, msg, received_for_mod, W, hebbian, decay,
                    s_mem_live, s_mem_ema_fast,
                    W_gamma, decay_gamma, phase="phase1")

            with torch.no_grad():
                new_pool = readout.reshape(BS, self.NC_pools, self.D_n)
                prev_pool = prev_readout.reshape(BS, self.NC_pools, self.D_n)
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
        BS, T, _ = H_mid.shape
        if not self._initialized:
            self.initialize_states(BS, H_mid.device)

        if (not self._compiled and H_mid.is_cuda
                and not os.environ.get("NEUROMORPHIC_NO_COMPILE")):
            self._run_block_uncompiled = self._run_block
            self._run_block = torch.compile(
                self._run_block, mode="default", fullgraph=False)
            self._compiled = True

        eot = self.config.eot_id
        valid_mask = torch.ones(BS, T, device=H_mid.device, dtype=H_mid.dtype)
        if T > 1:
            valid_mask[:, 1:] = (input_ids[:, :-1] != eot).to(H_mid.dtype)
        if prev_token is not None:
            valid_mask[:, 0] = (prev_token.to(input_ids.device) != eot).to(H_mid.dtype)
        else:
            valid_mask[:, 0] = 0.0

        h = self.h; msg = self.msg; W = self.W
        decay = self.decay; hebbian = self.hebbian
        s_mem_live = self.s_mem_live; s_mem_ema_fast = self.s_mem_ema_fast
        prev_readout = self.prev_readout; readout_drift = self.readout_drift
        segment_start_prev_readout = prev_readout.detach().clone()
        H_mid = H_mid.to(h.dtype)

        dt = h.dtype
        inject_w = self.inject_w.to(dt); inject_b = self.inject_b.to(dt)
        mg_w1 = self.msg_w1.to(dt); mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt); mg_b2 = self.msg_b2.to(dt)
        lm_head_w = lm.lm_head.weight.to(dt)
        proj_down_w = lm.proj_down.weight.to(dt) if lm.proj_down else None
        proj_down_b = lm.proj_down.bias.to(dt) if lm.proj_down else None
        ln_final_w = lm.ln_final.weight.to(dt)
        ln_final_b = lm.ln_final.bias.to(dt)

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
            block_args = (
                h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout, readout_drift,
                H_mid[:, start_t:end_t], input_ids[:, start_t:end_t],
                start_t,
                inject_w, inject_b,
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

        self.h = h.detach(); self.msg = msg.detach()
        self.W = W.detach(); self.decay = decay.detach()
        self.hebbian = hebbian.detach()
        self.s_mem_live = s_mem_live.detach()
        self.s_mem_ema_fast = s_mem_ema_fast.detach()
        self.prev_readout = prev_readout.detach()
        self.readout_drift = readout_drift.detach()

        shifted_all = torch.cat([
            segment_start_prev_readout.unsqueeze(1).to(readouts.dtype),
            readouts[:, :-1],
        ], dim=1)

        # Aux-loss chunking is decoupled from tbptt_block. Memory-head CE is
        # a no-grad-through-state logits+CE over (readout, vocab); larger
        # chunks mean fewer kernel launches. Default aux_loss_chunk=T (one
        # pass over the whole segment) which profile measurements show is
        # ~25% faster than chunk=tbptt_block.
        aux_chunk = max(1, self.config.aux_loss_chunk)
        loss_sum = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        valid_total = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        for s in range(0, T, aux_chunk):
            e = min(s + aux_chunk, T)
            sub_logits = lm.mem_head_logits(shifted_all[:, s:e])
            sub_valid = valid_mask[:, s:e].float()
            per_tok = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                input_ids[:, s:e].reshape(-1),
                reduction="none",
            ).reshape(sub_valid.shape)
            loss_sum = loss_sum + (per_tok * sub_valid).sum()
            valid_total = valid_total + sub_valid.sum()
        mem_pred_loss = loss_sum / valid_total.clamp(min=1.0)

        return readouts, mem_pred_loss
