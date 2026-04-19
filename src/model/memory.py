"""Multi-cell memory graph with attention neuromodulator.

N_cells connectivity pools × neurons_per_cell each. State tensors carry
explicit NC dim: h, msg are [BS, NC, N, D_n]; W, hebbian are
[BS, NC, N, N]. Each cell has its own block-diagonal W (no cross-cell
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
        self.N = config.neurons_per_cell         # neurons per cell
        self.N_total = self.NC * self.N                # total neurons
        self.alpha = config.alpha
        self.NC_pools = config.NC_pools           # = N_cells for default

        # Sanity: for simplicity require NC_pools == N_cells (enforced in
        # Config.validate for the default path).
        assert self.NC_pools == self.NC or self.NC % self.NC_pools == 0

        # Per-neuron identity vector, per cell.
        self.neuron_id = nn.Parameter(
            torch.randn(self.NC, self.N, self.D_n) * (self.D_n ** -0.5))

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

        # Per-cell per-neuron plasticity logits: [NC, N].
        self.W_decay_logit = nn.Parameter(torch.full((self.NC, self.N), -3.0))
        self.decay_gamma_logit = nn.Parameter(torch.full((self.NC, self.N), -3.0))
        self.hebbian_decay_logit = nn.Parameter(torch.full((self.NC, self.N), 2.0))
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
        self.register_buffer("role_id", layout["role_id"].view(self.NC, self.N))
        # Pre-built dense output-port mask for the fused step.
        # mask[c, n] = 1 if neuron n in cell c is an output port, else 0.
        out_mask = torch.zeros(self.NC, self.N)
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
        # Sampling-mode override. Decouples "use hard-Categorical phase-2
        # sampling" from `self.training`, which previously also gated
        # modulator dropout back on and made phase-2 rollouts
        # non-deterministic (dropout noise not represented in log_pi).
        # Set to True via `wrapper.rollout_mode()` during phase-2 prefix
        # passes; `_modulate` then picks the hard-sampling branch even
        # when the memory graph is in eval mode (dropout off).
        self._force_phase2_sampling: bool = False

    # ================================================================
    # State management
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int, device: torch.device,
                          dtype: torch.dtype | None = None):
        # Explicit dtype takes precedence — used by PretrainedLMWithMemory
        # to match memory state to the LM's load dtype (bf16 in production,
        # fp32 for bit-exact unit tests). Fallback: bf16 on CUDA, fp32 on
        # CPU.
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        dt = dtype
        NC, N, D_n = self.NC, self.N, self.D_n

        self.h = torch.zeros(BS, NC, N, D_n, device=device, dtype=dt)
        self.msg = torch.zeros(BS, NC, N, D_n, device=device, dtype=dt)
        self.decay = torch.full((BS, NC, N), 0.5, device=device, dtype=dt)
        self.hebbian = torch.zeros(BS, NC, N, N, device=device, dtype=dt)

        # Sparse init for W: ~min(8, N-1) random nonzeros per row, RMS-normed.
        K_init = min(8, N - 1)
        W = torch.zeros(BS, NC, N, N, device=device, dtype=dt)
        for c in range(NC):
            for neuron in range(N):
                neighbors = torch.randperm(N, device=device)[:K_init]
                W[:, c, neuron, neighbors] = 1.0
        W.diagonal(dim1=-2, dim2=-1).zero_()
        self.W = F.rms_norm(W, normalized_shape=(N,))

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
                     "prev_readout", "readout_drift",
                     "s_mem_live", "s_mem_ema_fast"):
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
        NC, N, D_n = self.NC, self.N, self.D_n
        self.h = torch.zeros(new_bs, NC, N, D_n, device=device, dtype=dt)
        self.msg = torch.zeros(new_bs, NC, N, D_n, device=device, dtype=dt)
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
                f"Runtime state incompatible: expected 4-D h [BS, NC, N, D_n]; "
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
        W_mean = self.W.mean(dim=0, keepdim=True)
        W_div = (self.W - W_mean).float().norm(dim=(1, 2, 3)).mean().item()
        W_norm = self.W.float().norm(dim=(1, 2, 3)).mean().item()
        return {"lane_W_divergence": W_div,
                "lane_W_relative_div": W_div / max(W_norm, 1e-8)}

    @torch.no_grad()
    def compute_memory_health(self) -> dict:
        if not self._initialized:
            return {}
        h, msg, W, decay = (t.float() for t in (self.h, self.msg, self.W, self.decay))
        hebbian = self.hebbian.float()
        W_off = W - torch.diag_embed(W.diagonal(dim1=-2, dim2=-1))
        heb_off = hebbian - torch.diag_embed(hebbian.diagonal(dim1=-2, dim2=-1))
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
            "hebbian_offdiag_norm": heb_off.norm().item()
                                    / max(heb_off.numel() ** 0.5, 1.0),
            "W_hebbian_offdiag_cos": 0.0,  # skip for speed
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
        """Block-diagonal message passing: [BS, NC, N, N] @ [BS, NC, N, D_n]."""
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
        # received: [BS, NC, N, D_n]. For each cell c, add inject[:, c, :, :] at
        # positions idx[c, :].
        # Use gather-style: create a padded add tensor [BS, NC, N, D_n] and add.
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
        # neuron_id is [NC, N, D_n]; broadcast over BS.
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
                  phase: str) -> tuple[Tensor, Tensor, Tensor]:
        """Per-cell attention encoder → per-cell codes → per-cell ΔW.

        Returns (W_new, decay_new, log_pi) where log_pi is [BS, NC]: the
        log-probability of the sampled code under the current policy. In
        phase 1 (Gumbel-soft) and eval (argmax) log_pi is zeroed — it is
        only a meaningful REINFORCE signal in phase 2 hard sampling.
        """
        # logits: [BS, NC, K]
        logits, _tokens = self.modulator(
            h, msg, received, W, hebbian, decay,
            s_live, s_ema, self.role_id.to(h.device))

        # Per-cell sampling. Flatten [BS*NC] for discrete_policy operations,
        # reshape back after.
        BS, NC, K = logits.shape
        logits_flat = logits.reshape(BS * NC, K)
        log_pi_flat = torch.zeros(BS * NC, device=logits.device, dtype=logits.dtype)
        # `_force_phase2_sampling` lets callers (phase-2 rollout / grpo_step)
        # request hard Categorical sampling with log_pi even while the
        # modulator is in eval mode — that keeps dropout off, which is
        # required for determinism (seeded Generator only controls
        # multinomial, not dropout).
        use_hard_sampling = (self._force_phase2_sampling or
                              (self.training and phase == "phase2"))
        if use_hard_sampling:
            # Phase 2: hard Categorical sampling. Preserve log_pi so the
            # training loop can compute the REINFORCE / GRPO policy-gradient
            # update. Gradient flows through log_pi back to the modulator
            # logits even though `codes` itself is a non-differentiable
            # sampled index.
            codes, log_pi_flat = self.discrete_policy.sample_discrete(
                logits_flat, tau=self.gumbel_tau)
            emb = self.discrete_policy.lookup(codes)
        elif self.training and phase == "phase1":
            soft, codes = self.discrete_policy.sample_gumbel_soft(
                logits_flat, tau=self.gumbel_tau, hard=True)
            emb = self.discrete_policy.lookup_soft(soft)       # [BS*NC, D_code]
            self.discrete_policy.update_usage(codes)
        else:
            # Eval / deterministic argmax.
            codes = logits_flat.argmax(dim=-1)
            emb = self.discrete_policy.lookup(codes)

        # Decoder is a shared trunk conditioned on the modulator's cell_emb.
        # emb [BS*NC, D_code] → reshape [BS, NC, D_code] → returns (ΔW [BS, NC, N, N], Δdecay [BS, NC, N]).
        emb_3d = emb.reshape(BS, NC, -1)
        dW_normed, dDecay_raw = self.decoder(emb_3d, self.modulator.cell_emb)

        # EMA blend. W_gamma is [NC, N] — broadcast to [BS, NC, N, N] via unsqueeze(-1).
        W_new = ((1 - W_gamma.unsqueeze(0).unsqueeze(-1)) * W
                 + W_gamma.unsqueeze(0).unsqueeze(-1) * dW_normed.to(W.dtype))
        target_decay = torch.sigmoid(dDecay_raw).to(decay.dtype)
        decay_new = ((1 - decay_gamma.unsqueeze(0)) * decay
                     + decay_gamma.unsqueeze(0) * target_decay)
        return W_new, decay_new, log_pi_flat.reshape(BS, NC)

    @staticmethod
    def _hebbian_update(hebbian: Tensor, msg: Tensor, gamma: Tensor) -> Tensor:
        """msg @ msgᵀ per cell; γ is [NC, N] per-receiver. All bf16."""
        coactiv = torch.matmul(msg, msg.transpose(-1, -2))  # [BS, NC, N, N]
        g = gamma.unsqueeze(0).unsqueeze(-1)  # [1, NC, N, 1]
        return (1 - g) * hebbian + g * coactiv

    # ================================================================
    # Per-block loop
    # ================================================================

    def _run_block(self, h, msg, W, decay, hebbian,
                   s_mem_live, s_mem_ema_fast,
                   prev_readout, readout_drift,
                   log_pi_sum,
                   block_H_mid, block_input_ids,
                   start_t,
                   inject_w, inject_b,
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                   hebbian_gamma, W_gamma, decay_gamma, gain_fast,
                   use_rmsnorm: bool = False, rms_eps: float = 1e-5,
                   phase: str = "phase1",
                   preserve_graph: bool = False):
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

            if (not preserve_graph) and t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach(); msg = msg.detach(); W = W.detach()
                decay = decay.detach(); hebbian = hebbian.detach()
                prev_readout = prev_readout.detach()

            # Surprise signal from previous readout — always per-token so
            # the modulator sees the freshest EMA whenever it fires.
            with torch.no_grad():
                x = prev_readout
                if proj_down_w is not None:
                    x = F.linear(x, proj_down_w, proj_down_b)
                if use_rmsnorm:
                    # Llama-style RMSNorm (no bias, no mean-centering).
                    x = F.rms_norm(x, (x.shape[-1],), ln_final_w, rms_eps)
                else:
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
                W, decay, log_pi_step = self._modulate(
                    h, msg, received_for_mod, W, hebbian, decay,
                    s_mem_live, s_mem_ema_fast,
                    W_gamma, decay_gamma, phase=phase)
                # Accumulate over cells into a per-batch scalar — the policy-
                # gradient signal is one log π per rollout. In phase 1 (Gumbel
                # soft) and eval (argmax) log_pi_step is zero, so this is a
                # no-op for those paths.
                log_pi_sum = log_pi_sum + log_pi_step.sum(dim=-1)

            with torch.no_grad():
                new_pool = readout.reshape(BS, self.NC_pools, self.D_n)
                prev_pool = prev_readout.reshape(BS, self.NC_pools, self.D_n)
                readout_drift = (new_pool - prev_pool).abs().mean(
                    dim=-1, keepdim=True).to(h.dtype)
            prev_readout = readout.detach()

        return (h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout, readout_drift, readouts, log_pi_sum)

    # ================================================================
    # Segment forward
    # ================================================================

    def forward_segment(
        self,
        H_mid: Tensor,
        input_ids: Tensor,
        lm,
        prev_token: Tensor | None = None,
        use_rmsnorm: bool = False,
        rms_eps: float = 1e-5,
        phase: str = "phase1",
        preserve_graph: bool = False,
        compute_aux_loss: bool = True,
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
        # Cast LM weights to memory dtype only when dtypes actually differ.
        # Same-dtype .to() is a no-op fast path. When callers load Llama
        # directly in bf16, this becomes free and F.rms_norm hits the fused
        # bf16×bf16 kernel. The old unconditional .to(dt) allocated a 200MB
        # bf16 copy of lm_head on every call — biggest single rollout cost.
        lm_head_w = lm.lm_head.weight
        if lm_head_w.dtype != dt:
            lm_head_w = lm_head_w.to(dt)
        proj_down_w = lm.proj_down.weight if lm.proj_down is not None else None
        if proj_down_w is not None and proj_down_w.dtype != dt:
            proj_down_w = proj_down_w.to(dt)
        proj_down_b = (lm.proj_down.bias
                       if lm.proj_down is not None and lm.proj_down.bias is not None
                       else None)
        if proj_down_b is not None and proj_down_b.dtype != dt:
            proj_down_b = proj_down_b.to(dt)
        ln_final_w = lm.ln_final.weight
        if ln_final_w.dtype != dt:
            ln_final_w = ln_final_w.to(dt)
        # RMSNorm has no bias; pass a zero tensor in the weight's dtype so
        # the LayerNorm fallback path in `_run_block` still type-checks.
        ln_final_b = (lm.ln_final.bias if lm.ln_final.bias is not None
                      else torch.zeros_like(ln_final_w))
        if ln_final_b is not None and ln_final_b.dtype != dt:
            ln_final_b = ln_final_b.to(dt)

        gm = self.gamma_max
        hebbian_gamma = gm * torch.sigmoid(self.hebbian_decay_logit).to(dt)
        W_gamma = gm * torch.sigmoid(self.W_decay_logit).to(dt)
        decay_gamma = gm * torch.sigmoid(self.decay_gamma_logit).to(dt)

        readouts = torch.empty(BS, T, self.config.D,
                                device=H_mid.device, dtype=dt)
        block_size = max(1, self.config.tbptt_block)
        gain_fast = self.config.gain_ema_fast
        # Per-rollout running sum of log π(sampled_code) across all modulator
        # fires in the segment. Zero in phase 1 / eval (log_pi_step stays at
        # zero there). Stored on self._log_pi_sum after the loop so phase-2
        # training can read it.
        log_pi_sum = torch.zeros(BS, device=H_mid.device, dtype=dt)

        use_ckpt = self.training and self.config.checkpoint_memory
        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_args = (
                h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout, readout_drift,
                log_pi_sum,
                H_mid[:, start_t:end_t], input_ids[:, start_t:end_t],
                start_t,
                inject_w, inject_b,
                mg_w1, mg_b1, mg_w2, mg_b2,
                lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                hebbian_gamma, W_gamma, decay_gamma, gain_fast,
                use_rmsnorm, rms_eps, phase, preserve_graph)

            if use_ckpt:
                result = torch.utils.checkpoint.checkpoint(
                    self._run_block, *block_args, use_reentrant=False)
            else:
                result = self._run_block(*block_args)

            (h, msg, W, decay, hebbian,
             s_mem_live, s_mem_ema_fast,
             prev_readout, readout_drift, block_out, log_pi_sum) = result
            readouts[:, start_t:end_t] = block_out

        # Persist state across segments. `preserve_graph=True` keeps the
        # autograd graph alive across forward_segment calls — used by
        # autoregressive phase-1 unroll where the continuation steps need
        # gradient to reach the prefix's modulator fires. Caller is
        # responsible for calling `detach_states()` after backward.
        if preserve_graph:
            self.h = h; self.msg = msg
            self.W = W; self.decay = decay
            self.hebbian = hebbian
            self.s_mem_live = s_mem_live
            self.s_mem_ema_fast = s_mem_ema_fast
            self.prev_readout = prev_readout
            self.readout_drift = readout_drift
        else:
            self.h = h.detach(); self.msg = msg.detach()
            self.W = W.detach(); self.decay = decay.detach()
            self.hebbian = hebbian.detach()
            self.s_mem_live = s_mem_live.detach()
            self.s_mem_ema_fast = s_mem_ema_fast.detach()
            self.prev_readout = prev_readout.detach()
            self.readout_drift = readout_drift.detach()
        # Expose the segment-total log π sum for phase-2 GRPO readers. Keep
        # graph-connected (do NOT detach) — the policy gradient needs
        # backward to flow through log_pi_sum into the modulator logits.
        # ONLY written on phase-2 training calls. Phase-1 / eval calls
        # would stomp the meaningful tensor with a detached-zero one and
        # silently break any downstream reader that missed the capture
        # window. `None` sentinel between phase-2 writes makes stale reads
        # loud instead of silent.
        # Match the sampling-branch gate in _modulate: log_pi is meaningful
        # whenever hard Categorical sampling ran (either phase-2 + train
        # mode, or the explicit _force_phase2_sampling override used by
        # rollout code paths that keep the modulator in eval to silence
        # dropout noise).
        ran_hard_sample = (self._force_phase2_sampling or
                            (phase == "phase2" and self.training))
        if ran_hard_sample:
            self._last_log_pi_sum = log_pi_sum
        else:
            self._last_log_pi_sum = None

        if compute_aux_loss:
            shifted_all = torch.cat([
                segment_start_prev_readout.unsqueeze(1).to(readouts.dtype),
                readouts[:, :-1],
            ], dim=1)

            # Aux-loss chunking is decoupled from tbptt_block. Memory-head CE
            # is a no-grad-through-state logits+CE over (readout, vocab);
            # larger chunks mean fewer kernel launches. Default
            # aux_loss_chunk=T (one pass over the whole segment) which profile
            # measurements show is ~25% faster than chunk=tbptt_block.
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
        else:
            # Skip aux loss entirely — for phase-2 rollouts + eval where the
            # mem_pred_loss isn't in the training loss. Avoids the 128K-vocab
            # lm_head matmul per gen step (biggest chunk of rollout cost).
            mem_pred_loss = torch.zeros((), device=H_mid.device,
                                         dtype=torch.float32)

        return readouts, mem_pred_loss
