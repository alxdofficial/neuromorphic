"""Dense-W cell-grid memory graph.

Connectivity within each cell is a dense N×N weight matrix W, updated by
a neuromodulator with low-rank deltas. Message passing is a single batched
matmul per step. No sparse gather, no Triton kernels in the hot path.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config
from .discrete_policy import DiscreteActionPolicy


class MemoryGraph(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.D_n = config.D_n
        self.N_cells = config.N_cells
        self.C_n = config.neurons_per_cell
        self.alpha = config.alpha

        self.input_lo = 0
        self.input_hi = self.alpha
        self.output_lo = self.input_hi
        self.output_hi = self.output_lo + self.alpha

        # Per-neuron identity. Initialized so each neuron's identity vector
        # has unit expected L2 norm — a principled "fingerprint" scale that
        # depends only on D_n, no magic constant.
        self.neuron_id = nn.Parameter(
            torch.randn(self.N_cells, self.C_n, self.D_n) * (self.D_n ** -0.5))

        Hs = config.state_mlp_hidden
        Hm = config.msg_mlp_hidden
        Hmod = config.cell_mod_hidden
        N = self.C_n

        # Shared state MLP: input = cat(received, h) = 2*D_n
        self.state_w1 = nn.Parameter(torch.empty(Hs, config.state_in))
        self.state_b1 = nn.Parameter(torch.zeros(Hs))
        self.state_w2 = nn.Parameter(torch.empty(self.D_n, Hs))
        self.state_b2 = nn.Parameter(torch.zeros(self.D_n))

        # Shared message MLP: input = h = D_n
        self.msg_w1 = nn.Parameter(torch.empty(Hm, config.msg_in))
        self.msg_b1 = nn.Parameter(torch.zeros(Hm))
        self.msg_w2 = nn.Parameter(torch.empty(self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(self.D_n))

        # Inject projection (per-cell)
        self.inject_w = nn.Parameter(
            torch.empty(self.N_cells, self.alpha * self.D_n, self.D_n))
        self.inject_b = nn.Parameter(
            torch.zeros(self.N_cells, self.alpha * self.D_n))

        # Neuromodulator: produces per-cell discrete code + continuous action.
        # See discrete_policy.py for architecture details. Input layout matches
        # old modulator (config.mod_in). Per-cell logit head, shared codebook
        # + decoder across cells. K=256, D=32 by default.
        self.discrete_policy = DiscreteActionPolicy(
            n_cells=self.N_cells,
            mod_in_dim=config.mod_in,
            action_dim=config.mod_out,
            num_codes=getattr(config, "num_codes", 256),
            code_dim=getattr(config, "code_dim", 32),
            logit_hidden=Hmod,
            decoder_hidden=getattr(config, "decoder_hidden", 128),
        )
        # Gumbel-softmax temperature for phase 1 backprop. Set externally
        # (train.py can anneal 1.0 → 0.3 over bootstrap). Unused in phase 2.
        self.gumbel_tau = 1.0

        # Learnable per-cell decay rate for the Hebbian co-activation trace.
        # Stored as a logit; sigmoid(logit) gives the running-average rate γ
        # in `hebbian = (1-γ) * hebbian + γ * msg @ msg.T`. Init at sigmoid(2)
        # ≈ 0.88 means the running average integrates over ~9 recent steps.
        # Each cell can learn its own timescale for "fire-together" memory.
        self.hebbian_decay_logit = nn.Parameter(torch.full((self.N_cells,), 2.0))

        # Learnable per-cell plasticity rate for W. Same convex-EMA pattern as
        # the Hebbian trace: `W = (1-γ_W) * W + γ_W * rms_norm(delta_W)`.
        # Because delta_W is RMSNormed per row before the update, W stays at
        # ~unit per-row RMS by construction — no unbounded accumulator, no
        # bf16 overflow, no need to renormalize at use time.
        #
        # Init at sigmoid(-3) ≈ 0.047 → W integrates over ~20 modulator calls
        # ≈ 80 tokens. W is a long-term state (the synaptic weight matrix),
        # so we bias toward slow plasticity rather than 50-50 tracking. The
        # cell can still learn a faster rate if the loss rewards it. This
        # also matches the effective plasticity of the (retired) 0.01 init
        # scale on mod_w2 without reintroducing that magic number.
        self.W_decay_logit = nn.Parameter(torch.full((self.N_cells,), -3.0))

        # Learnable per-cell plasticity rate for the neuron decay (persistence)
        # gate. Same convex-EMA pattern: `decay = (1-γ_d) * decay + γ_d *
        # sigmoid(delta_decay_raw)`. The modulator proposes a target
        # persistence in [0,1] via sigmoid, and the runtime decay state EMAs
        # toward it, so decay is a convex combination of [0,1] values and
        # stays in [0,1] by construction.
        #
        # Init at sigmoid(-3) ≈ 0.047 for the same reason as W: decay should
        # reflect a cell's long-run persistence regime, not flip every few
        # tokens. Learnable — the cell can pick up a faster rate if helpful.
        self.decay_gamma_logit = nn.Parameter(torch.full((self.N_cells,), -3.0))

        # Init weights — one uniform rule: Xavier/Glorot with the gain that
        # matches the downstream activation. Tanh layers use 5/3 (Glorot 2010),
        # linear-output layers use 1.0. PyTorch's `nn.init.xavier_uniform_`
        # infers fan_in/fan_out correctly for 2D tensors; for 3D per-cell
        # tensors we compute the bound manually using the true einsum fans.
        TANH_GAIN = 5.0 / 3.0

        # Shared 2D MLPs — every layer feeds a tanh downstream
        nn.init.xavier_uniform_(self.state_w1, gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.state_w2, gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.msg_w1,   gain=TANH_GAIN)
        nn.init.xavier_uniform_(self.msg_w2,   gain=TANH_GAIN)

        # Per-cell 3D tensors — PyTorch's fan calculation is wrong for these
        # (they're einsum-semantics, not conv-semantics). Compute directly.
        def _xavier_einsum(w: Tensor, fan_in: int, fan_out: int, gain: float):
            bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
            with torch.no_grad():
                w.uniform_(-bound, bound)

        # Discrete policy initializes its own weights (see discrete_policy.py).
        # inject_w: [NC, alpha*D_n, D_n], einsum("bni,noi->bno"), linear
        _xavier_einsum(self.inject_w, fan_in=self.D_n,
                       fan_out=self.alpha * self.D_n, gain=1.0)

        # Biases already zero by construction (torch.zeros above). No hand-
        # scaling of output layers — the uniform Xavier rule is the only
        # magnitude management the module needs.

        self._initialized = False
        self._compiled = False
        self._collecting_actions = False
        self._action_buffer: list[Tensor] = []

    # ================================================================
    # State management
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _phase2_substep(self, h, msg, W, gate, one_minus_gate, hebbian,
                        H_aug_t, identity, inject_w, inject_b,
                        st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                        mg_w1, mg_b1, mg_w2, mg_b2,
                        hebbian_gamma):
        """Fused phase 2 per-token step: memory step + hebbian update.

        Kept as a single function so torch.compile can fuse the whole
        thing into one CUDA graph-friendly subroutine.
        """
        h, msg, readout = self._step(
            h, msg, W, gate, one_minus_gate,
            H_aug_t, identity, inject_w, inject_b,
            st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
            mg_w1, mg_b1, mg_w2, mg_b2)
        hebbian = self._hebbian_update(hebbian, msg, hebbian_gamma)
        return h, msg, readout, hebbian

    def _identity(self, BS, dtype, device):
        return self.neuron_id.to(device=device, dtype=dtype).unsqueeze(0).expand(BS, -1, -1, -1)

    def initialize_states(self, BS: int, device: torch.device):
        dt = torch.bfloat16
        NC, N, D_n = self.N_cells, self.C_n, self.D_n
        K_init = self.config.K  # initial sparse connections

        self.h = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.01
        self.msg = torch.zeros(BS, NC, N, D_n, device=device, dtype=dt)
        # Per-neuron decay gate, stored directly in [0,1]. Init at 0.5
        # (midrange, unbiased). Updated via convex EMA toward a sigmoid'd
        # modulator target — see `_modulate_cells`. Because the update is
        # a convex combination of [0,1] values, decay stays in [0,1] by
        # construction, with no logit/sigmoid gymnastics at use time.
        self.decay = torch.full((BS, NC, N), 0.5, device=device, dtype=dt)
        # Hebbian co-activation trace: running EMA of `msg @ msg.T` per cell.
        # Same shape as W. Updated in no_grad inside _step. Provides the
        # modulator with biologically principled "fire-together" structure.
        self.hebbian = torch.zeros(BS, NC, N, N, device=device, dtype=dt)

        # Initialize W as sparse: K random nonzero connections per neuron,
        # then RMSNorm each row so W starts at the unit per-row RMS regime
        # the modulator EMA update maintains at steady state. The nonzero
        # value is arbitrary (1.0) since RMSNorm rescales it anyway.
        W = torch.zeros(BS, NC, N, N, device=device, dtype=dt)
        for cell in range(NC):
            for neuron in range(N):
                neighbors = torch.randperm(N, device=device)[:K_init]
                W[:, cell, neuron, neighbors] = 1.0
            # Zero out self-connections
            W[:, cell].diagonal(dim1=-2, dim2=-1).zero_()
        self.W = F.rms_norm(W, normalized_shape=(N,))

        # Surprise state (scalar per batch — memory-head prediction signal).
        # s_mem_live is the per-token CE at the arriving token under the
        # memory head (computed inside _run_block as logsumexp(logits) -
        # target_logit). s_mem_ema_fast is its short-horizon EMA.
        self.s_mem_live = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(BS, device=device, dtype=dt)
        # Per-cell drift signal — carried across blocks so the modulator at the
        # start of a new block sees the drift from the end of the previous block,
        # not a fresh zero. Note: the per-cell "previous readout" needed to
        # compute drift is derived on-the-fly from `prev_readout` via
        # `.view(BS, NC, D_n)` — no separate state tensor.
        self.readout_drift = torch.zeros(BS, NC, 1, device=device, dtype=dt)
        # Full D-dim previous readout — fed to the memory head for predicting
        # the next token (memory head uses readout[t-1] to predict x_t), and
        # reshaped on demand for the per-cell drift computation.
        self.prev_readout = torch.zeros(BS, self.config.D, device=device, dtype=dt)

        self._initialized = True

    def detach_states(self):
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.msg = self.msg.detach()
        self.W = self.W.detach()
        self.decay = self.decay.detach()
        self.prev_readout = self.prev_readout.detach()
        self.readout_drift = self.readout_drift.detach()
        self.hebbian = self.hebbian.detach()
        # s_mem_* are already detached (no_grad EMAs)

    @torch.no_grad()
    def compute_lane_divergence(self) -> dict:
        """Diagnostic: measure how much BS lanes have diverged.

        Each lane's W/decay/hebbian reflects the modulator's autonomous
        response to that lane's content stream. Divergence is expected and
        healthy — it means the shared modulator policy is producing
        content-appropriate structure in each lane.

        Returns stats for logging (does NOT modify state).
        """
        if not self._initialized:
            return {}
        BS = self.W.shape[0]
        if BS <= 1:
            return {}

        W_mean = self.W.mean(dim=0, keepdim=True)
        decay_mean = self.decay.mean(dim=0, keepdim=True)
        heb_mean = self.hebbian.mean(dim=0, keepdim=True)

        W_div = (self.W - W_mean).float().norm(dim=(2, 3)).mean().item()
        W_norm = self.W.float().norm(dim=(2, 3)).mean().item()
        decay_div = (self.decay - decay_mean).float().norm(dim=-1).mean().item()
        heb_div = (self.hebbian - heb_mean).float().norm(dim=(2, 3)).mean().item()
        heb_norm = self.hebbian.float().norm(dim=(2, 3)).mean().item()

        return {
            "lane_W_divergence": W_div,
            "lane_W_relative_div": W_div / max(W_norm, 1e-8),
            "lane_decay_divergence": decay_div,
            "lane_hebbian_divergence": heb_div,
            "lane_hebbian_relative_div": heb_div / max(heb_norm, 1e-8),
        }

    @torch.no_grad()
    def resize_to_bs(self, new_bs: int):
        """Resize all runtime state tensors to a new batch size.

        Each lane's W/decay/hebbian is a valid memory state produced by
        the shared modulator on that lane's content stream.

        When shrinking (new_bs < old_bs): randomly samples new_bs lanes
        from the old_bs pool so no lane is systematically favored. This
        does lose (old_bs - new_bs) lane states — unavoidable when
        reducing parallelism. The surviving lanes are real memory states,
        not averaged approximations.

        When growing (new_bs > old_bs): tiles the existing lanes
        cyclically. Duplicated lanes start identical but diverge within
        a few steps as they see different content.

        Transient state (h, msg, etc.) is reset to zero because it's
        input-dependent and doesn't transfer across different data streams.
        """
        if not self._initialized:
            return
        device = self.W.device
        dt = self.W.dtype
        old_bs = self.W.shape[0]
        if new_bs == old_bs:
            return

        if new_bs < old_bs:
            # Randomly sample new_bs lanes — no lane is special.
            idx = torch.randperm(old_bs, device=device)[:new_bs]
        else:
            # Tile existing lanes cyclically to fill new_bs.
            idx = torch.arange(new_bs, device=device) % old_bs

        # Long-term state: index from existing lanes.
        self.W = self.W[idx].clone()
        self.decay = self.decay[idx].clone()
        self.hebbian = self.hebbian[idx].clone()

        # Transient state: reinit to zero (will warm up on next forward).
        self.h = torch.zeros(
            new_bs, self.N_cells, self.C_n, self.D_n, device=device, dtype=dt)
        self.msg = torch.zeros(
            new_bs, self.N_cells, self.C_n, self.D_n, device=device, dtype=dt)
        self.s_mem_live = torch.zeros(new_bs, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(new_bs, device=device, dtype=dt)
        self.prev_readout = torch.zeros(
            new_bs, self.config.D, device=device, dtype=dt)
        self.readout_drift = torch.zeros(
            new_bs, self.N_cells, 1, device=device, dtype=dt)

    def runtime_state_dict(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized": True,
            "h": self.h.clone(),
            "msg": self.msg.clone(),
            "W": self.W.clone(),
            "decay": self.decay.clone(),
            "s_mem_live": self.s_mem_live.clone(),
            "s_mem_ema_fast": self.s_mem_ema_fast.clone(),
            "prev_readout": self.prev_readout.clone(),
            "readout_drift": self.readout_drift.clone(),
            "hebbian": self.hebbian.clone(),
        }

    def load_runtime_state(self, state: dict):
        if not state or not state.get("initialized", False):
            self._initialized = False
            return
        device = self.neuron_id.device
        self.h = state["h"].to(device)
        self.msg = state["msg"].to(device)
        self.W = state["W"].to(device)
        BS = self.h.shape[0]
        dt = self.h.dtype
        # Back-compat: older checkpoints stored `decay_logit` (unbounded logit
        # space) instead of `decay` (value in [0,1]). If loading an old
        # checkpoint, apply sigmoid; otherwise use the new field directly.
        if "decay" in state:
            self.decay = state["decay"].to(device)
        elif "decay_logit" in state:
            self.decay = torch.sigmoid(
                state["decay_logit"].to(device).float()).to(dt)
        else:
            self.decay = torch.full(
                (BS, self.N_cells, self.C_n), 0.5, device=device, dtype=dt)
        zero_b = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_live = state.get("s_mem_live", zero_b).to(device)
        self.s_mem_ema_fast = state.get("s_mem_ema_fast", zero_b).to(device)
        self.prev_readout = state.get("prev_readout", torch.zeros(
            BS, self.config.D, device=device, dtype=dt)).to(device)
        # Older checkpoints may have `prev_readout_cell` — ignored now
        # (per-cell view is derived on-the-fly from prev_readout).
        self.readout_drift = state.get("readout_drift", torch.zeros(
            BS, self.N_cells, 1, device=device, dtype=dt)).to(device)
        self.hebbian = state.get("hebbian", torch.zeros(
            BS, self.N_cells, self.C_n, self.C_n, device=device, dtype=dt)).to(device)
        self._initialized = True

    # ================================================================
    # Telemetry (phase-1 plateau detection)
    # ================================================================

    def _build_mod_input(self) -> Tensor:
        """Construct modulator input tensor from current runtime state.

        Returns: [BS, NC, mod_in] in the runtime dtype.
        Used by both `compute_modulator_stats` and `collect_modulator_action`.

        Biologically principled inputs only: rates (per-neuron magnitudes),
        correlations (Hebbian trace), and global neuromodulatory signals
        (surprise, drift). No feature-space "content" peek into the cell.
        See Config.mod_in for the layout.
        """
        BS = self.h.shape[0]
        NC, N = self.N_cells, self.C_n
        dt = self.h.dtype

        h_norms = self.h.float().norm(dim=-1).to(dt)                 # [BS, NC, N]
        msg_norms = self.msg.float().norm(dim=-1).to(dt)             # [BS, NC, N]
        decay_mean = self.decay.mean(dim=2, keepdim=True)            # [BS, NC, 1] in [0,1]
        hebbian_flat = self.hebbian.reshape(BS, NC, N * N)           # [BS, NC, N*N]

        s1 = self.s_mem_live.view(BS, 1, 1).expand(BS, NC, 1).to(dt)
        s2 = self.s_mem_ema_fast.view(BS, 1, 1).expand(BS, NC, 1).to(dt)

        return torch.cat([
            h_norms, msg_norms,         # 2*N     (per-neuron firing rates)
            decay_mean,                  # 1       (per-cell average leakiness)
            self.readout_drift,         # 1       (per-cell volatility)
            s1, s2,                      # 2       (global surprise)
            hebbian_flat,                # N*N     (per-pair coactivation history)
        ], dim=-1)

    def _modulator_forward(
        self, mod_input: Tensor, phase: str = "phase1",
    ) -> dict:
        """Run the neuromodulator policy.

        phase="phase1": Gumbel-softmax sampling (differentiable for backprop)
        phase="phase2": hard Categorical (for RL rollouts)

        Returns dict with keys: logits, codes, action, log_pi.
        The 'action' tensor has the same shape & semantics as the old
        modulator's continuous output — [BS, NC, mod_out].
        """
        return self.discrete_policy.forward(
            mod_input, phase=phase,
            tau=self.gumbel_tau, hard_gumbel=True,
        )

    @torch.no_grad()
    def compute_modulator_stats(self) -> dict:
        """Snapshot modulator stats measuring ACTUAL applied plasticity.

        After the bounded-W redesign, raw modulator output norms are
        misleading — the applied update is row-RMS normalized and EMA-gated
        for W, and sigmoid-plus-EMA for decay. This method simulates one
        modulation step and reports the actual L2 magnitude of the
        applied W/decay changes.

        Also reports the raw action norm for comparison with older runs.
        """
        if not self._initialized:
            return {}

        BS = self.h.shape[0]
        NC, N = self.N_cells, self.C_n

        mod_input = self._build_mod_input()
        out = self._modulator_forward(mod_input, phase="phase1")
        output = out["action"]

        # Raw (pre-gate) modulator output — the driving signal.
        delta_W_raw = output[..., :N * N].reshape(BS, NC, N, N).float()
        delta_decay_raw = output[..., N * N:N * N + N].reshape(BS, NC, N).float()

        raw_action_norm = ((delta_W_raw.abs().mean(dim=(2, 3))
                            + delta_decay_raw.abs().mean(dim=2)) * 0.5).mean().item()

        # Simulate the actual applied update (matches _modulate_cells).
        W_f = self.W.float()
        decay_f = self.decay.float()

        delta_W_norm = F.rms_norm(delta_W_raw, normalized_shape=(N,))
        W_gamma = torch.sigmoid(self.W_decay_logit).float().view(1, -1, 1, 1)
        W_new = (1.0 - W_gamma) * W_f + W_gamma * delta_W_norm

        target_decay = torch.sigmoid(delta_decay_raw)
        d_gamma = torch.sigmoid(self.decay_gamma_logit).float().view(1, -1, 1)
        decay_new = (1.0 - d_gamma) * decay_f + d_gamma * target_decay

        # Actual applied change — this is the signal that matters.
        dW_applied = (W_new - W_f).norm(dim=(2, 3))  # [BS, NC]
        dDecay_applied = (decay_new - decay_f).norm(dim=-1)  # [BS, NC]

        # Variance across batch (per-cell) — measures input-dependent variability
        # of the actual applied update.
        dW_batch_var = (W_new - W_f).var(dim=0, unbiased=False).mean().item()
        dDecay_batch_var = (decay_new - decay_f).var(dim=0, unbiased=False).mean().item()

        return {
            "mod_action_norm": raw_action_norm,  # pre-gate, legacy panel signal
            "applied_dW_norm": dW_applied.mean().item(),
            "applied_dDecay_norm": dDecay_applied.mean().item(),
            "applied_dW_batch_var": dW_batch_var,
            "applied_dDecay_batch_var": dDecay_batch_var,
        }

    @torch.no_grad()
    def compute_plasticity_rates(self) -> dict:
        """Learnable plasticity rate traces (sigmoid of the *_decay_logit params).

        These control EMA rates (gamma in the convex update). Half-life in
        modulation steps: log(0.5) / log(1 - gamma).
        """
        import math
        W_gamma = torch.sigmoid(self.W_decay_logit).float()
        d_gamma = torch.sigmoid(self.decay_gamma_logit).float()
        h_gamma = torch.sigmoid(self.hebbian_decay_logit).float()

        def _hl(gamma: Tensor) -> Tensor:
            # log(0.5) / log(1 - gamma); clamp to avoid inf for gamma~0
            g = gamma.clamp(min=1e-6, max=1.0 - 1e-6)
            return (math.log(0.5) / torch.log(1.0 - g))

        return {
            "W_gamma_mean": W_gamma.mean().item(),
            "W_gamma_min": W_gamma.min().item(),
            "W_gamma_max": W_gamma.max().item(),
            "W_half_life": _hl(W_gamma).mean().item(),
            "decay_gamma_mean": d_gamma.mean().item(),
            "decay_gamma_min": d_gamma.min().item(),
            "decay_gamma_max": d_gamma.max().item(),
            "decay_half_life": _hl(d_gamma).mean().item(),
            "hebbian_gamma_mean": h_gamma.mean().item(),
            "hebbian_half_life": _hl(h_gamma).mean().item(),
        }

    def start_action_collection(self):
        """Enable per-modulation-event action collection inside _run_block.

        Only used during the short action collection sub-phase (~155 steps).
        Forward passes use the uncompiled path to avoid torch.compile
        recompilation from list-append side effects in _modulate_cells.
        """
        self._collecting_actions = True
        self._action_buffer = []

    def stop_action_collection(self):
        """Disable action collection."""
        self._collecting_actions = False

    @torch.no_grad()
    def collect_modulator_action(self) -> Tensor | None:
        """Return collected actions and clear the buffer.

        When _collecting_actions is True, returns all actions collected during
        the chunk (one per modulation event, ~T/modulation_interval per chunk).
        When False, falls back to a single end-of-chunk snapshot.

        Returns: [N_actions, BS, NC, mod_out] tensor, or None.
        """
        if not self._initialized:
            return None
        if self._action_buffer:
            actions = torch.stack(self._action_buffer, dim=0)  # [n, BS, NC, mod_out]
            self._action_buffer = []
            return actions
        # Fallback: single snapshot at current state
        mod_input = self._build_mod_input()
        out = self._modulator_forward(mod_input, phase="phase1")
        return out["action"].float().cpu().unsqueeze(0)

    def compute_mod_grad_norm(self) -> float:
        """Mean per-cell L2 norm of modulator logit-head gradients.

        Read from `.grad` after backward but before zero_grad. Returns 0 if
        any of the logit-head params has no grad yet.
        """
        norms = []
        for p in (self.discrete_policy.logit_w1, self.discrete_policy.logit_w2):
            if p.grad is None:
                return 0.0
            flat = p.grad.reshape(self.N_cells, -1).float()
            norms.append(flat.norm(dim=1))   # [NC]
        return torch.stack(norms, dim=0).mean().item()

    @torch.no_grad()
    def compute_memory_health(self) -> dict:
        """Snapshot of memory runtime state health. Cheap: scalars only.

        Returns a flat dict suitable for jsonl logging.
        """
        if not self._initialized:
            return {}
        out = {}
        h = self.h.float()
        msg = self.msg.float()
        W = self.W.float()
        decay = self.decay.float()

        out["h_norm"] = h.norm().item() / max(h.numel() ** 0.5, 1.0)
        out["msg_norm"] = msg.norm().item() / max(msg.numel() ** 0.5, 1.0)
        out["h_max"] = h.abs().max().item()
        out["msg_max"] = msg.abs().max().item()
        out["decay_mean"] = decay.mean().item()
        out["decay_std"] = decay.std().item()
        out["s_mem_live"] = self.s_mem_live.float().mean().item()
        out["s_mem_ema_fast"] = self.s_mem_ema_fast.float().mean().item()
        out["readout_drift_mean"] = self.readout_drift.float().mean().item()

        # Off-diagonal W structure — after bounded W, on-diagonal is close
        # to identity, so off-diagonal carries the actual structure.
        N = W.shape[-1]
        diag_mask = torch.eye(N, device=W.device, dtype=torch.bool)
        W_off = W.masked_fill(diag_mask, 0.0)
        out["W_offdiag_norm"] = W_off.norm().item() / max(W_off.numel() ** 0.5, 1.0)
        out["W_offdiag_max"] = W_off.abs().max().item()

        # Off-diagonal Hebbian — the actual co-activation signal (diagonal
        # is just self-self, which dominates a full-matrix norm).
        heb = self.hebbian.float()
        heb_off = heb.masked_fill(diag_mask, 0.0)
        out["hebbian_offdiag_norm"] = heb_off.norm().item() / max(heb_off.numel() ** 0.5, 1.0)
        # Cosine between W_off and heb_off (flattened per batch element, mean'd).
        wf = W_off.reshape(W.shape[0], -1)
        hf = heb_off.reshape(W.shape[0], -1)
        cos = F.cosine_similarity(wf, hf, dim=-1)
        out["W_hebbian_offdiag_cos"] = cos.mean().item()
        return out

    def compute_param_norms(self) -> dict:
        """L2 norms of key weight tensors (cheap; cpu-synced scalars)."""
        dp = self.discrete_policy
        return {
            "mod_w1_norm": dp.logit_w1.detach().float().norm().item(),
            "mod_w2_norm": dp.logit_w2.detach().float().norm().item(),
            "codebook_norm": dp.codebook.detach().float().norm().item(),
            "dec_w1_norm": dp.dec_w1.detach().float().norm().item(),
            "dec_w2_norm": dp.dec_w2.detach().float().norm().item(),
            "state_w1_norm": self.state_w1.detach().float().norm().item(),
            "state_w2_norm": self.state_w2.detach().float().norm().item(),
            "msg_w1_norm": self.msg_w1.detach().float().norm().item(),
            "msg_w2_norm": self.msg_w2.detach().float().norm().item(),
            "inject_w_norm": self.inject_w.detach().float().norm().item(),
            "neuron_id_norm": self.neuron_id.detach().float().norm().item(),
        }

    def compute_component_grad_norms(self) -> dict:
        """Per-component grad L2 norms for the memory subnetwork.

        Read after backward, before zero_grad. Returns 0 for components with
        no grad (e.g. when frozen).
        """
        out = {}
        dp = self.discrete_policy
        for name, p in (
            ("grad_mod_w1", dp.logit_w1),
            ("grad_mod_w2", dp.logit_w2),
            ("grad_codebook", dp.codebook),
            ("grad_dec_w1", dp.dec_w1),
            ("grad_dec_w2", dp.dec_w2),
            ("grad_state_w1", self.state_w1),
            ("grad_state_w2", self.state_w2),
            ("grad_msg_w1", self.msg_w1),
            ("grad_msg_w2", self.msg_w2),
            ("grad_inject_w", self.inject_w),
            ("grad_neuron_id", self.neuron_id),
        ):
            out[name] = 0.0 if p.grad is None else p.grad.detach().float().norm().item()
        return out

    # ================================================================
    # Per-step components
    # ================================================================

    @staticmethod
    def _rmsnorm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        """Parameter-free RMSNorm along ``dim`` (last dim only for now).

        Uses torch's native F.rms_norm which is memory-optimized (fused kernel,
        no full-precision intermediate activations). Only supports normalizing
        over the last dim; if ``dim != -1`` we move the dim to last, normalize,
        and move back — but in practice all callers pass dim=-1.
        """
        if dim == -1 or dim == x.ndim - 1:
            return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)
        x_moved = x.transpose(dim, -1).contiguous()
        normed = F.rms_norm(x_moved, normalized_shape=(x_moved.shape[-1],), eps=eps)
        return normed.transpose(dim, -1).contiguous()

    def _receive(self, msg: Tensor, W: Tensor) -> Tensor:
        """Dense matmul message passing.

        W is maintained at ~unit per-row RMS by the modulator's convex-EMA
        update (`_modulate_cells`), so the matmul output is bounded without
        any per-step renormalization here.
        """
        return torch.matmul(W, msg)         # [BS, NC, N, D_n]

    def _inject(self, received, H_aug_t, inject_w, inject_b):
        cell_slice = H_aug_t.reshape(H_aug_t.shape[0], self.N_cells, self.D_n)
        inject = torch.einsum("bni,noi->bno", cell_slice, inject_w)
        inject = inject + inject_b.unsqueeze(0)
        inject = inject.reshape(
            H_aug_t.shape[0], self.N_cells, self.alpha, self.D_n)
        received[:, :, self.input_lo:self.input_hi].add_(inject.to(received.dtype))
        return received

    def _mlp2(self, x, w1, b1, w2, b2):
        """Two-layer MLP: Linear → tanh → Linear → tanh."""
        flat = x.reshape(-1, x.shape[-1])
        hidden = torch.tanh(F.linear(flat, w1, b1))
        out = torch.tanh(F.linear(hidden, w2, b2))
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    def _state_update(self, received, h, decay, one_minus_decay,
                      w1_recv, w1_h, b1, w2, b2):
        """State MLP with split first layer — avoids cat([received, h])."""
        flat_recv = received.reshape(-1, received.shape[-1])
        flat_h = h.reshape(-1, h.shape[-1])
        hidden = torch.tanh(
            F.linear(flat_recv, w1_recv) + F.linear(flat_h, w1_h, b1))
        candidate = torch.tanh(F.linear(hidden, w2, b2))
        candidate = candidate.reshape(h.shape)
        return decay * h + one_minus_decay * candidate

    def _emit_message(self, h, identity, w1, b1, w2, b2):
        msg_new = self._mlp2(h, w1, b1, w2, b2)
        return msg_new + identity

    def _readout(self, msg):
        out_ports = msg[:, :, self.output_lo:self.output_hi]
        readout = out_ports.sum(dim=2) * (self.alpha ** -0.5)
        return readout.reshape(msg.shape[0], -1)

    def _modulate_cells(self, h, msg, W, decay, hebbian,
                        readout_drift, s_mem_live, s_mem_ema_fast,
                        W_gamma, decay_gamma, phase: str = "phase1"):
        """Per-cell neuromodulator step.

        Biologically principled inputs only — rates, correlations, and
        global neuromodulators. No content peek into the cell.
          - h_norms, msg_norms        : per-neuron firing magnitudes (rates)
          - decay_mean                : average per-cell leakiness in [0,1]
          - readout_drift             : per-cell volatility
          - s_mem_live, s_mem_ema_fast : global surprise (broadcast)
          - hebbian_flat              : per-pair coactivation history

        Runs the DiscreteActionPolicy: obs → logits → Gumbel-softmax (phase 1)
        or Categorical (phase 2) → code → decoder → continuous action. The
        continuous action is then consumed identically to the old design:
        parsed into delta_W/delta_decay_raw and applied via convex EMA.
        """
        NC, N = self.N_cells, self.C_n
        BS = h.shape[0]
        dt = h.dtype

        h_norms = h.float().norm(dim=-1).to(dt)                              # [BS, NC, N]
        msg_norms = msg.float().norm(dim=-1).to(dt)                          # [BS, NC, N]
        decay_mean = decay.mean(dim=2, keepdim=True)                         # [BS, NC, 1]
        hebbian_flat = hebbian.reshape(BS, NC, N * N)                        # [BS, NC, N*N]

        s1 = s_mem_live.view(BS, 1, 1).expand(BS, NC, 1).to(dt)
        s2 = s_mem_ema_fast.view(BS, 1, 1).expand(BS, NC, 1).to(dt)

        mod_input = torch.cat([
            h_norms, msg_norms,                # 2*N
            decay_mean,                         # 1
            readout_drift,                      # 1
            s1, s2,                             # 2
            hebbian_flat,                       # N*N
        ], dim=-1)

        out = self._modulator_forward(mod_input, phase=phase)
        output = out["action"]  # [BS, NC, mod_out]
        codes = out["codes"]    # [BS, NC] long

        # Track code usage during phase-1 for dead-code resampling diagnostics
        if phase == "phase1" and self.training:
            self.discrete_policy.update_usage(codes.detach())

        delta_W_raw = output[..., :N * N].reshape(BS, NC, N, N)
        delta_decay_raw = output[..., N * N:N * N + N].reshape(BS, NC, N)

        delta_W = F.rms_norm(delta_W_raw, normalized_shape=(N,))
        g_W_f32 = W_gamma.float().view(1, -1, 1, 1)
        W_new = ((1.0 - g_W_f32) * W.float() + g_W_f32 * delta_W.float()).to(W.dtype)

        target_decay_f32 = torch.sigmoid(delta_decay_raw.float())
        g_d_f32 = decay_gamma.float().view(1, -1, 1)
        decay_new = ((1.0 - g_d_f32) * decay.float() + g_d_f32 * target_decay_f32).to(decay.dtype)

        if self._collecting_actions:
            # Record the continuous decoded action (legacy format). The codes
            # themselves are also available via discrete_policy.update_usage tracking.
            self._action_buffer.append(output.detach().float().cpu())

        return W_new, decay_new

    def _step(self, h, msg, W, decay, one_minus_decay,
              H_aug_t, identity, inject_w, inject_b,
              st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
              mg_w1, mg_b1, mg_w2, mg_b2):
        received = self._receive(msg, W)
        received = self._inject(received, H_aug_t, inject_w, inject_b)

        h = self._state_update(received, h, decay, one_minus_decay,
                               st_w1_recv, st_w1_h, st_b1, st_w2, st_b2)
        msg = self._emit_message(h, identity, mg_w1, mg_b1, mg_w2, mg_b2)
        readout = self._readout(msg)
        return h, msg, readout

    @staticmethod
    def _hebbian_update(hebbian: Tensor, msg: Tensor, gamma: Tensor) -> Tensor:
        """EMA update of the Hebbian co-activation trace.

        hebbian: [BS, NC, N, N]
        msg:     [BS, NC, N, D_n]
        gamma:   [NC] in (0, 1) — per-cell running-average rate

        Returns the updated hebbian. Intentionally ON the autograd graph
        so that downstream use of hebbian in the modulator provides
        gradient back to hebbian_decay_logit (which gamma derives from).

        Convex EMA computed in f32: at γ approaching 1, bf16 representation
        of (1-γ) collapses to zero (100% rel error at logit=+7, 16% at
        logit=+5, 1.6% at logit=+2 = hebbian init). We compute in f32 then
        cast back to the runtime dtype. See audit claim #4.
        """
        # Pairwise dot product across neurons within each cell.
        # [BS, NC, N, D_n] @ [BS, NC, D_n, N] -> [BS, NC, N, N]
        coactiv = torch.matmul(msg.float(), msg.float().transpose(-1, -2))
        # Per-cell gamma broadcast across BS, N, N — f32 for precision.
        g = gamma.float().view(1, -1, 1, 1)
        return ((1.0 - g) * hebbian.float() + g * coactiv).to(hebbian.dtype)

    def _run_block(self, h, msg, W, decay, hebbian,
                   s_mem_live, s_mem_ema_fast,
                   prev_readout_full, readout_drift,
                   block_H_mid, block_input_ids,
                   start_t, identity, inject_w, inject_b,
                   st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                   hebbian_gamma, W_gamma, decay_gamma, gain_fast):
        """Run one block of memory steps with live in-block surprise signals.

        Live surprise: at each step t, use prev_readout_full (readout at t-1) to
        predict input_ids[t] via the weight-tied memory head. The negative
        target logit becomes s_mem_live(t); EMAs update per-token. The modulator
        at the next modulation step sees the fresh signal.
        """
        block_T = block_H_mid.shape[1]
        BS = h.shape[0]
        D = self.config.D
        NC, D_n = self.N_cells, self.D_n
        readouts = torch.empty(BS, block_T, D, device=block_H_mid.device, dtype=h.dtype)

        # `decay` is stored directly in [0,1]. Broadcast to [BS,NC,N,1] for
        # the per-neuron state-update gate.
        gate = decay.unsqueeze(-1)
        one_minus_gate = 1.0 - gate

        for offset in range(block_T):
            t = start_t + offset
            H_mid_t = block_H_mid[:, offset]  # [BS, D]
            tok_t = block_input_ids[:, offset]  # [BS]

            # TBPTT detach
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                W = W.detach()
                decay = decay.detach()
                hebbian = hebbian.detach()
                prev_readout_full = prev_readout_full.detach()
                gate = decay.unsqueeze(-1)
                one_minus_gate = 1.0 - gate

            # --- Live memory-head surprise (no_grad — detached signal) ---
            # Computes proper per-token CE = logsumexp(logits) - target_logit.
            # This is an OBSERVATION fed to the modulator: "how predictable
            # was the token that just came in?" It fires at every position
            # — including cross-document boundaries — because a surprise
            # spike at a document start is the legitimate "context just
            # changed, adapt" signal the modulator needs to react to.
            # The training-target masks (mem_pred_loss, phase-2 reward)
            # are separate and still mask out EOT boundaries.
            with torch.no_grad():
                x = prev_readout_full
                if proj_down_w is not None:
                    x = F.linear(x, proj_down_w, proj_down_b)
                x = F.layer_norm(x, (x.shape[-1],), ln_final_w, ln_final_b)
                logits_full = F.linear(x, lm_head_w)                    # [BS, V]
                lse = torch.logsumexp(logits_full.float(), dim=-1)      # [BS]
                target_logit = logits_full.gather(
                    1, tok_t.unsqueeze(1)).squeeze(1).float()           # [BS]
                s_mem_live = (lse - target_logit).to(h.dtype)
                s_mem_ema_fast = (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live

            # --- Modulate every M tokens using the FRESH surprise ---
            if t % self.config.modulation_interval == 0:
                W, decay = self._modulate_cells(
                    h, msg, W, decay, hebbian,
                    readout_drift, s_mem_live, s_mem_ema_fast,
                    W_gamma, decay_gamma, phase="phase1")
                gate = decay.unsqueeze(-1)
                one_minus_gate = 1.0 - gate

            # --- Memory step ---
            h, msg, readout = self._step(
                h, msg, W, gate, one_minus_gate,
                H_mid_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2)

            readouts[:, offset] = readout

            # Update Hebbian co-activation trace ON the autograd graph so the
            # learnable per-cell `hebbian_decay_logit` (which becomes
            # `hebbian_gamma` after sigmoid) receives gradient via downstream
            # use of `hebbian` in the modulator input.
            hebbian = self._hebbian_update(hebbian, msg, hebbian_gamma)

            # Update per-cell drift signal (used as a separate feature).
            # `prev_readout_full` is still the readout from t-1 at this
            # point — we haven't advanced it yet below — so view it as
            # per-cell and diff against the current cell readout.
            with torch.no_grad():
                new_cell = readout.reshape(BS, NC, D_n)
                prev_cell = prev_readout_full.view(BS, NC, D_n)
                readout_drift = (new_cell - prev_cell).abs().mean(
                    dim=-1, keepdim=True).to(h.dtype)

            # Advance prev_readout_full for next-token memory-head prediction.
            prev_readout_full = readout.detach()

        return (h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout_full, readout_drift, readouts)

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_mid: Tensor, input_ids: Tensor, lm,
                        prev_token: Tensor | None = None):
        """Process T tokens through the memory graph.

        Args:
            H_mid: [BS, T, D] — lower-scan output (detached from LM graph)
            input_ids: [BS, T] — input tokens; used to compute the per-token
                memory-head CE surprise signal (logsumexp(logits) -
                target_logit), where the memory head uses readout[t-1] to
                predict the token at position t.
            lm: LM module — provides mem_head_logits (weight-tied to the
                main lm_head) for the segment-level mem_pred_loss.
            prev_token: [BS] or None — the last token of the previous chunk,
                used to decide if position 0 is a valid prediction target.
                If None, position 0 is treated as valid.

        Returns:
            readouts:      [BS, T, D] — memory readout per token
            mem_pred_loss: scalar      — CE of memory head against input_ids
                                          (used as auxiliary training loss)
        """
        BS, T, _ = H_mid.shape
        if not self._initialized:
            self.initialize_states(BS, H_mid.device)
        # torch.compile the memory loop for speed on CUDA. Disabled during
        # action collection (list-append side effects) or when the
        # NEUROMORPHIC_NO_COMPILE env var is set (for debugging / equivalence
        # testing against the eager path).
        if (not self._compiled and H_mid.is_cuda
                and not self._collecting_actions
                and not os.environ.get("NEUROMORPHIC_NO_COMPILE")):
            self._run_block_uncompiled = self._run_block
            self._run_block = torch.compile(
                self._run_block, mode="default", fullgraph=False)
            self._compiled = True

        # Valid mask for the TRAINING TARGET (mem_pred_loss only).
        # Position t is a valid prediction target iff the "previous"
        # token (input_ids[t-1] or prev_token for t=0) is not EOT.
        # At a cross-document boundary the memory head would be
        # predicting the first token of a new document from the previous
        # document's context — there's no causal relationship, so
        # supervising the memory head there is just noise.
        # NOTE: this mask is NOT applied to the live surprise observation
        # signal — that fires at every position, including EOT boundaries,
        # because a surprise spike at a document start is a legitimate
        # "context just changed, adapt now" signal for the modulator.
        eot = self.config.eot_id
        valid_mask = torch.ones(BS, T, device=H_mid.device, dtype=H_mid.dtype)
        if T > 1:
            valid_mask[:, 1:] = (input_ids[:, :-1] != eot).to(H_mid.dtype)
        if prev_token is not None:
            valid_mask[:, 0] = (prev_token.to(input_ids.device) != eot).to(H_mid.dtype)

        h = self.h
        msg = self.msg
        W = self.W
        decay = self.decay
        hebbian = self.hebbian
        s_mem_live = self.s_mem_live
        s_mem_ema_fast = self.s_mem_ema_fast
        prev_readout_full = self.prev_readout  # full-D; reshaped per-cell inside _run_block for drift
        readout_drift = self.readout_drift
        segment_start_prev_readout = prev_readout_full.detach().clone()
        H_mid = H_mid.to(h.dtype)

        identity = self._identity(BS, h.dtype, h.device)

        dt = h.dtype
        st_w1_full = self.state_w1.to(dt)
        st_w1_recv = st_w1_full[:, :self.D_n].contiguous()
        st_w1_h = st_w1_full[:, self.D_n:].contiguous()
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)
        inject_w = self.inject_w.to(dt)
        inject_b = self.inject_b.to(dt)
        # Discrete policy params are accessed via self.discrete_policy inside
        # _modulate_cells — no longer plumbed through _run_block args.
        # LM head weights for the in-block memory head (bf16 compute copies)
        lm_head_w = lm.lm_head.weight.to(dt)
        if lm.proj_down is not None:
            proj_down_w = lm.proj_down.weight.to(dt)
            proj_down_b = lm.proj_down.bias.to(dt)
        else:
            proj_down_w = None
            proj_down_b = None
        ln_final_w = lm.ln_final.weight.to(dt)
        ln_final_b = lm.ln_final.bias.to(dt)

        readouts = torch.empty(BS, T, self.config.D, device=H_mid.device, dtype=dt)
        block_size = max(1, self.config.tbptt_block)
        gain_fast = self.config.gain_ema_fast
        # Per-cell learnable Hebbian decay rate, sigmoid'd at use. NOT detached
        # so backward through the on-graph Hebbian updates propagates gradient
        # back to hebbian_decay_logit.
        hebbian_gamma = torch.sigmoid(self.hebbian_decay_logit).to(dt)
        W_gamma = torch.sigmoid(self.W_decay_logit).to(dt)
        decay_gamma = torch.sigmoid(self.decay_gamma_logit).to(dt)

        use_ckpt = self.training and self.config.checkpoint_memory
        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_mid = H_mid[:, start_t:end_t]
            block_input_ids = input_ids[:, start_t:end_t]

            block_args = (
                h, msg, W, decay, hebbian,
                s_mem_live, s_mem_ema_fast,
                prev_readout_full, readout_drift,
                block_H_mid, block_input_ids,
                start_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                hebbian_gamma, W_gamma, decay_gamma, gain_fast)
            # Use uncompiled path during action collection to avoid
            # torch.compile recompilation from list-append side effects.
            run_fn = (self._run_block_uncompiled
                      if self._collecting_actions and hasattr(self, '_run_block_uncompiled')
                      else self._run_block)
            if use_ckpt:
                result = torch.utils.checkpoint.checkpoint(
                    run_fn, *block_args, use_reentrant=False)
            else:
                result = run_fn(*block_args)

            (h, msg, W, decay, hebbian,
             s_mem_live, s_mem_ema_fast,
             prev_readout_full, readout_drift, block_out) = result

            readouts[:, start_t:end_t] = block_out

        # Save state
        self.h = h
        self.msg = msg
        self.W = W
        self.decay = decay
        self.hebbian = hebbian
        self.s_mem_live = s_mem_live.detach()
        self.s_mem_ema_fast = s_mem_ema_fast.detach()
        self.prev_readout = prev_readout_full.detach()
        self.readout_drift = readout_drift.detach()

        # --- Segment-level mem_pred_loss, chunked for VRAM ---
        # Memory head uses readout[t-1] to predict token at position t.
        # Mask out positions where the previous token was EOT (cross-doc)
        # so the memory head isn't trained on arbitrary transitions that
        # the main LM loss also ignores.
        shifted_all = torch.cat([
            segment_start_prev_readout.unsqueeze(1).to(readouts.dtype),  # [BS, 1, D]
            readouts[:, :-1],                                             # [BS, T-1, D]
        ], dim=1)  # [BS, T, D]

        loss_sum = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        valid_total = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        chunk = block_size  # reuse the same chunk size (8)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_readout = shifted_all[:, s:e]
            sub_target = input_ids[:, s:e]
            sub_valid = valid_mask[:, s:e].float()                 # [BS, chunk]
            sub_logits = lm.mem_head_logits(sub_readout)  # [BS, chunk, V]
            per_tok_loss = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                sub_target.reshape(-1),
                reduction="none",
            ).reshape(sub_target.shape)                              # [BS, chunk]
            loss_sum = loss_sum + (per_tok_loss * sub_valid).sum()
            valid_total = valid_total + sub_valid.sum()
        mem_pred_loss = loss_sum / valid_total.clamp(min=1.0)

        return readouts, mem_pred_loss

    # ================================================================
    # Phase 2: VQ-sampled rollout (no_grad)
    # ================================================================

    @torch.no_grad()
    def forward_segment_phase2(
        self,
        H_mid: Tensor,
        input_ids: Tensor,
        lm,
        tau: float = 1.0,
        sample: bool = True,
        prev_token: Tensor | None = None,
        h_mid_batch_map: Tensor | None = None,
    ) -> dict:
        """Phase 2 rollout: direct categorical sampling from discrete policy.

        At each modulator call:
            1. Build mod_input as in phase 1
            2. Run the neuromod to produce per-cell logits over K codes
            3. Sample a code per cell (multinomial at temperature tau or
               argmax if sample=False)
            4. Decode the sampled codes to continuous actions via the
               frozen codebook + decoder
            5. Apply delta_W / delta_decay via convex EMA (same as phase 1)
            6. Record (mod_input, codes, t) for the GRPO gradient pass

        Args:
            H_mid: [H_BS, T, D] — lower-scan output from (frozen) LM.
            input_ids: [BS, T] — at the memory's batch size (K*BS_orig)
            lm: (frozen) LM module — provides lm_head / mem_head weights
                for the in-rollout surprise signal.
            tau: temperature for the categorical (softmax(logits/τ))
            sample: True for multinomial, False for argmax (deployment path)
            h_mid_batch_map: [BS] long or None — maps each sample in the
                memory batch to its row in H_mid. Saves VRAM by avoiding
                K-fold duplication of H_mid.

        Returns:
            dict with:
                readouts: [BS, T, D]
                mod_inputs: [n_calls, BS, NC, mod_in]
                codes: [n_calls, BS, NC]  (long)
                call_positions: [n_calls]
        """
        assert self._initialized, "memory must be initialized before phase2 rollout"

        # Lazily compile the phase 2 substep (memory step + hebbian update)
        # on first CUDA call. The substep is pure PyTorch with no side
        # effects, so torch.compile can fuse it into efficient kernels.
        # Gated by env var for equivalence testing.
        if (not hasattr(self, "_phase2_substep_compiled")
                and self.h.is_cuda
                and not os.environ.get("NEUROMORPHIC_NO_COMPILE")):
            self._phase2_substep_compiled = torch.compile(
                self._phase2_substep, mode="default", fullgraph=False)
        substep = getattr(self, "_phase2_substep_compiled", self._phase2_substep)

        BS = self.h.shape[0]  # memory batch size (K*BS_orig when expanded)
        _, T, D = H_mid.shape
        NC, N = self.N_cells, self.C_n
        D_n = self.D_n
        dt = self.h.dtype

        # Per-step LM weights for mem-head surprise (mirrors forward_segment)
        lm_head_w = lm.lm_head.weight.to(dt)
        if lm.proj_down is not None:
            proj_down_w = lm.proj_down.weight.to(dt)
            proj_down_b = lm.proj_down.bias.to(dt)
        else:
            proj_down_w = None
            proj_down_b = None
        ln_final_w = lm.ln_final.weight.to(dt)
        ln_final_b = lm.ln_final.bias.to(dt)

        # Dynamics MLP weight caches
        st_w1_full = self.state_w1.to(dt)
        st_w1_recv = st_w1_full[:, :D_n].contiguous()
        st_w1_h = st_w1_full[:, D_n:].contiguous()
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)
        inject_w = self.inject_w.to(dt)
        inject_b = self.inject_b.to(dt)

        identity = self._identity(BS, dt, H_mid.device)
        H_mid = H_mid.to(dt)
        # Batch map: when H_mid is smaller than BS (e.g. K-expansion),
        # h_mid_batch_map[i] gives the row in H_mid for sample i.
        _bmap = h_mid_batch_map  # [BS] long or None

        # Live runtime state (cloned so we don't clobber phase-1 state)
        h = self.h.clone()
        msg = self.msg.clone()
        W = self.W.clone()
        decay = self.decay.clone()
        hebbian = self.hebbian.clone()
        s_mem_live = self.s_mem_live.clone()
        s_mem_ema_fast = self.s_mem_ema_fast.clone()
        prev_readout_full = self.prev_readout.clone()
        readout_drift = self.readout_drift.clone()
        hebbian_gamma = torch.sigmoid(self.hebbian_decay_logit).to(dt)
        W_gamma = torch.sigmoid(self.W_decay_logit).to(dt)
        decay_gamma = torch.sigmoid(self.decay_gamma_logit).to(dt)

        gate = decay.unsqueeze(-1)
        one_minus_gate = 1.0 - gate

        gain_fast = self.config.gain_ema_fast
        mod_interval = self.config.modulation_interval

        readouts = torch.empty(BS, T, D, device=H_mid.device, dtype=dt)
        mod_input_records: list[Tensor] = []
        codes_records: list[Tensor] = []
        call_positions: list[int] = []

        # Batched surprise: we buffer the prev_readouts used for surprise
        # across a modulation interval (mod_interval tokens) and do ONE
        # lm_head call at [BS, mod_interval, V] instead of mod_interval
        # separate [BS, V] calls. GEMM efficiency gain is 3-4x on matmul
        # throughput. The EMA scheduling is identical to per-token updates
        # at the end of the window (we replay the N updates sequentially
        # from the batched live values).
        prev_readout_buf: list[Tensor] = []  # prev_readouts used for next batched surprise
        tok_buf: list[Tensor] = []            # target tokens for each slot

        def _flush_surprise():
            """Compute surprise for the buffered prev_readouts/tokens in one
            batched lm_head call and apply EMA sequentially.

            Updates s_mem_live and s_mem_ema_fast in the outer scope.
            """
            nonlocal s_mem_live, s_mem_ema_fast
            if not prev_readout_buf:
                return
            # Stack to [BS, n, D]
            pr_stack = torch.stack(prev_readout_buf, dim=1)
            tok_stack = torch.stack(tok_buf, dim=1)  # [BS, n]
            x = pr_stack
            if proj_down_w is not None:
                x = F.linear(x, proj_down_w, proj_down_b)
            x = F.layer_norm(x, (x.shape[-1],), ln_final_w, ln_final_b)
            logits_block = F.linear(x, lm_head_w)  # [BS, n, V]
            lse_block = torch.logsumexp(logits_block.float(), dim=-1)  # [BS, n]
            target_logit_block = logits_block.gather(
                2, tok_stack.unsqueeze(-1)).squeeze(-1).float()  # [BS, n]
            live_block = (lse_block - target_logit_block).to(dt)  # [BS, n]
            # Replay EMA updates sequentially — same math as per-token loop.
            for i in range(live_block.shape[1]):
                s_mem_live = live_block[:, i]
                s_mem_ema_fast = (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live
            prev_readout_buf.clear()
            tok_buf.clear()

        for t in range(T):
            H_mid_t = H_mid[_bmap, t] if _bmap is not None else H_mid[:, t]
            tok_t = input_ids[:, t]

            # Buffer the surprise inputs for batched lm_head. The actual
            # surprise computation is deferred until we need s_mem_live or
            # s_mem_ema_fast (i.e. right before a modulation event).
            prev_readout_buf.append(prev_readout_full)
            tok_buf.append(tok_t)

            # Modulate via VQ sampling every M tokens. Flush buffered surprise
            # first so s_mem_live/ema reflect the tokens seen up to t.
            if t % mod_interval == 0:
                _flush_surprise()
                # Build mod_input in f32 for numerical consistency with the
                # gradient pass (which also runs the modulator in f32). Layout
                # mirrors phase 1's `_build_mod_input`:
                #   h_norms, msg_norms, decay_mean, readout_drift,
                #   s_live, s_fast, hebbian_flat
                h_norms = h.float().norm(dim=-1)              # [BS, NC, N]
                msg_norms = msg.float().norm(dim=-1)          # [BS, NC, N]
                decay_mean = decay.float().mean(dim=2, keepdim=True)
                hebbian_flat = hebbian.float().reshape(BS, NC, N * N)
                s1 = s_mem_live.float().view(BS, 1, 1).expand(BS, NC, 1)
                s2 = s_mem_ema_fast.float().view(BS, 1, 1).expand(BS, NC, 1)
                mod_input_f32 = torch.cat([
                    h_norms, msg_norms,
                    decay_mean,
                    readout_drift.float(),
                    s1, s2,
                    hebbian_flat,
                ], dim=-1)  # [BS, NC, mod_in] in f32

                # Discrete policy: logits → sample code → decode
                logits = self.discrete_policy.compute_logits(mod_input_f32)  # [BS, NC, K]
                if sample:
                    codes, _ = self.discrete_policy.sample_discrete(logits, tau=tau)
                else:
                    codes = logits.argmax(dim=-1)
                action = self.discrete_policy.decode(codes)          # [BS, NC, action_dim] f32
                quantized = action.to(dt)

                # Unpack delta_W, delta_decay_raw and apply the same convex-EMA
                # updates as phase 1 (so the frozen-module determinism matches).
                # F32 compute for (1-γ) precision — same fix as phase 1.
                delta_W_raw = quantized[..., :N * N].reshape(BS, NC, N, N)
                delta_decay_raw = quantized[..., N * N:N * N + N].reshape(BS, NC, N)

                delta_W = F.rms_norm(delta_W_raw, normalized_shape=(N,))
                g_w_f32 = W_gamma.float().view(1, -1, 1, 1)
                W = ((1.0 - g_w_f32) * W.float() + g_w_f32 * delta_W.float()).to(W.dtype)

                target_decay_f32 = torch.sigmoid(delta_decay_raw.float())
                g_d_f32 = decay_gamma.float().view(1, -1, 1)
                decay = ((1.0 - g_d_f32) * decay.float() + g_d_f32 * target_decay_f32).to(decay.dtype)
                gate = decay.unsqueeze(-1)
                one_minus_gate = 1.0 - gate

                # Record on GPU; we stack+return on GPU. The peak VRAM cost
                # of mod_inputs accumulation (~4.6 GB at W=4096) is the
                # price we pay for keeping grpo_step fast (no PCIe transfer).
                mod_input_records.append(mod_input_f32.detach())
                codes_records.append(codes.detach())              # [BS, NC] long
                call_positions.append(t)

            # Fused memory step + hebbian update (torch.compile'd on CUDA)
            h, msg, readout, hebbian = substep(
                h, msg, W, gate, one_minus_gate, hebbian,
                H_mid_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                hebbian_gamma)
            readouts[:, t] = readout

            # prev_readout_full is still readout[t-1] at this point —
            # view as per-cell to compute the drift before advancing it.
            new_cell = readout.reshape(BS, NC, D_n)
            prev_cell = prev_readout_full.view(BS, NC, D_n)
            readout_drift = (new_cell - prev_cell).abs().mean(
                dim=-1, keepdim=True).to(dt)
            prev_readout_full = readout
            # W stays at ~unit per-row RMS via convex-EMA in _modulate_cells.

        # Persist end-of-rollout memory state so the next rollout continues
        # from here (memory is lifelong across rollouts too).
        self.h = h
        self.msg = msg
        self.W = W
        self.decay = decay
        self.hebbian = hebbian.detach()
        self.s_mem_live = s_mem_live
        self.s_mem_ema_fast = s_mem_ema_fast
        self.prev_readout = prev_readout_full
        self.readout_drift = readout_drift

        return {
            "readouts": readouts,
            "mod_inputs": torch.stack(mod_input_records, dim=0),   # [n_calls, BS, NC, mod_in]
            "codes": torch.stack(codes_records, dim=0),             # [n_calls, BS, NC] long
            "call_positions": torch.tensor(call_positions, dtype=torch.long),
        }
