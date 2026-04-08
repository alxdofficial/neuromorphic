"""Dense-W cell-grid memory graph.

Connectivity within each cell is a dense N×N weight matrix W, updated by
a neuromodulator with low-rank deltas. Message passing is a single batched
matmul per step. No sparse gather, no Triton kernels in the hot path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


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

        # Per-neuron identity
        self.neuron_id = nn.Parameter(
            torch.randn(self.N_cells, self.C_n, self.D_n) * 0.02)

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

        # Per-cell modulator
        # Input: h_mean + msg_mean + ctx + W_row_norms + decay_mean
        self.mod_w1 = nn.Parameter(torch.empty(self.N_cells, config.mod_in, Hmod))
        self.mod_b1 = nn.Parameter(torch.zeros(self.N_cells, Hmod))
        self.mod_w2 = nn.Parameter(torch.empty(self.N_cells, Hmod, config.mod_out))
        self.mod_b2 = nn.Parameter(torch.zeros(self.N_cells, config.mod_out))

        # Init weights
        for weight in (
            self.state_w1, self.state_w2, self.msg_w1, self.msg_w2,
            self.mod_w1, self.mod_w2, self.inject_w,
        ):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        with torch.no_grad():
            self.inject_b.zero_()
            self.state_w2.mul_(0.1)
            self.msg_w2.mul_(0.1)
            self.mod_w2.mul_(0.01)

        self._initialized = False
        self._compiled = False

    # ================================================================
    # State management
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _identity(self, BS, dtype, device):
        return self.neuron_id.to(device=device, dtype=dtype).unsqueeze(0).expand(BS, -1, -1, -1)

    def initialize_states(self, BS: int, device: torch.device):
        dt = torch.bfloat16
        NC, N, D_n = self.N_cells, self.C_n, self.D_n
        K_init = self.config.K  # initial sparse connections

        self.h = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.01
        self.msg = torch.zeros(BS, NC, N, D_n, device=device, dtype=dt)
        self.decay_logit = torch.zeros(BS, NC, N, device=device, dtype=dt)

        # Initialize W as sparse: K random nonzero connections per neuron
        W = torch.zeros(BS, NC, N, N, device=device, dtype=dt)
        for cell in range(NC):
            for neuron in range(N):
                neighbors = torch.randperm(N, device=device)[:K_init]
                W[:, cell, neuron, neighbors] = 0.1
            # Zero out self-connections
            W[:, cell].diagonal(dim1=-2, dim2=-1).zero_()
        self.W = W

        # Surprise state (scalar per batch — memory-head prediction signal).
        # s_mem(t) is the NEGATIVE target-logit under the memory head (lower =
        # less surprised). s_mem_live is the instant value the modulator reads
        # inside the loop; s_mem_ema_fast is its short-horizon smoothed track.
        self.s_mem_live = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(BS, device=device, dtype=dt)
        # Per-cell previous readout — for computing readout_drift (local surprise).
        self.prev_readout_cell = torch.zeros(BS, NC, D_n, device=device, dtype=dt)
        # Per-cell drift signal — carried across blocks so the modulator at the
        # start of a new block sees the drift from the end of the previous block,
        # not a fresh zero.
        self.readout_drift = torch.zeros(BS, NC, 1, device=device, dtype=dt)
        # Full D-dim previous readout — fed to the memory head for predicting
        # the next token (memory head uses readout[t-1] to predict x_t).
        self.prev_readout = torch.zeros(BS, self.config.D, device=device, dtype=dt)

        self._initialized = True

    def detach_states(self):
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.msg = self.msg.detach()
        self.W = self.W.detach()
        self.decay_logit = self.decay_logit.detach()
        self.prev_readout = self.prev_readout.detach()
        self.prev_readout_cell = self.prev_readout_cell.detach()
        self.readout_drift = self.readout_drift.detach()
        # s_mem_* are already detached (no_grad EMAs)

    def reset_states(self, mask: Tensor):
        """Retained for API compatibility; memory is intended to be lifelong."""
        pass

    def runtime_state_dict(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized": True,
            "h": self.h.clone(),
            "msg": self.msg.clone(),
            "W": self.W.clone(),
            "decay_logit": self.decay_logit.clone(),
            "s_mem_live": self.s_mem_live.clone(),
            "s_mem_ema_fast": self.s_mem_ema_fast.clone(),
            "prev_readout": self.prev_readout.clone(),
            "prev_readout_cell": self.prev_readout_cell.clone(),
            "readout_drift": self.readout_drift.clone(),
        }

    def load_runtime_state(self, state: dict):
        if not state or not state.get("initialized", False):
            self._initialized = False
            return
        device = self.neuron_id.device
        self.h = state["h"].to(device)
        self.msg = state["msg"].to(device)
        self.W = state["W"].to(device)
        self.decay_logit = state["decay_logit"].to(device)
        BS = self.h.shape[0]
        dt = self.h.dtype
        zero_b = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_live = state.get("s_mem_live", zero_b).to(device)
        self.s_mem_ema_fast = state.get("s_mem_ema_fast", zero_b).to(device)
        self.prev_readout = state.get("prev_readout", torch.zeros(
            BS, self.config.D, device=device, dtype=dt)).to(device)
        self.prev_readout_cell = state.get("prev_readout_cell", torch.zeros(
            BS, self.N_cells, self.D_n, device=device, dtype=dt)).to(device)
        self.readout_drift = state.get("readout_drift", torch.zeros(
            BS, self.N_cells, 1, device=device, dtype=dt)).to(device)
        self._initialized = True

    # ================================================================
    # Telemetry (phase-1 plateau detection)
    # ================================================================

    def _build_mod_input(self) -> Tensor:
        """Construct modulator input tensor from current runtime state.

        Returns: [BS, NC, mod_in] in the runtime dtype.
        Used by both `compute_modulator_stats` and `collect_modulator_action`.
        """
        BS = self.h.shape[0]
        NC = self.N_cells
        dt = self.h.dtype

        h_mean = self.h.mean(dim=2)
        msg_mean = self.msg.mean(dim=2)
        W_stats = self.W.abs().mean(dim=-1).mean(dim=2, keepdim=True)
        decay_mean = self.decay_logit.mean(dim=2, keepdim=True)

        s1 = self.s_mem_live.view(BS, 1, 1).expand(BS, NC, 1).to(dt)
        s2 = self.s_mem_ema_fast.view(BS, 1, 1).expand(BS, NC, 1).to(dt)

        return torch.cat([
            h_mean, msg_mean,
            W_stats, decay_mean,
            self.readout_drift,
            s1, s2,
        ], dim=-1)

    def _modulator_forward(self, mod_input: Tensor) -> Tensor:
        """Eager modulator MLP forward, returns full output [BS, NC, mod_out]."""
        dt = mod_input.dtype
        mod_w1 = self.mod_w1.to(dt)
        mod_b1 = self.mod_b1.to(dt)
        mod_w2 = self.mod_w2.to(dt)
        mod_b2 = self.mod_b2.to(dt)
        hidden = torch.tanh(
            torch.einsum("bni,nih->bnh", mod_input, mod_w1) + mod_b1.unsqueeze(0))
        output = torch.einsum("bnh,nho->bno", hidden, mod_w2) + mod_b2.unsqueeze(0)
        return output

    @torch.no_grad()
    def compute_modulator_stats(self) -> dict:
        """Snapshot modulator action stats at current state. See class docstring."""
        if not self._initialized:
            return {"mod_action_norm": 0.0, "mod_action_var": 0.0}

        BS = self.h.shape[0]
        NC, N = self.N_cells, self.C_n

        mod_input = self._build_mod_input()
        output = self._modulator_forward(mod_input)

        delta_W = output[..., :N * N].reshape(BS, NC, N, N).float()
        delta_decay = output[..., N * N:N * N + N].reshape(BS, NC, N).float()

        dW_mag = delta_W.abs().mean(dim=(2, 3))
        dD_mag = delta_decay.abs().mean(dim=2)
        action_norm = ((dW_mag + dD_mag) * 0.5).mean()

        dW_var = delta_W.var(dim=0, unbiased=False).mean(dim=(1, 2))
        dD_var = delta_decay.var(dim=0, unbiased=False).mean(dim=1)
        action_var = ((dW_var + dD_var) * 0.5).mean()

        return {
            "mod_action_norm": action_norm.item(),
            "mod_action_var": action_var.item(),
        }

    @torch.no_grad()
    def collect_modulator_action(self) -> Tensor | None:
        """Snapshot the full modulator action vector at current runtime state.

        Returns: [BS, NC, mod_out] tensor (concatenation of delta_W and
        delta_decay flattened per cell), or None if not yet initialized.
        Used during action database collection between phase 1 and codebook fit.
        """
        if not self._initialized:
            return None
        mod_input = self._build_mod_input()
        return self._modulator_forward(mod_input).float().cpu()

    def compute_mod_grad_norm(self) -> float:
        """Mean per-cell L2 norm of modulator weight gradients.

        Read from `.grad` after backward but before zero_grad. Returns 0 if
        any of the modulator params has no grad yet.
        """
        norms = []
        for p in (self.mod_w1, self.mod_w2):
            if p.grad is None:
                return 0.0
            flat = p.grad.reshape(self.N_cells, -1).float()
            norms.append(flat.norm(dim=1))   # [NC]
        return torch.stack(norms, dim=0).mean().item()

    # ================================================================
    # Per-step components
    # ================================================================

    def _receive(self, msg: Tensor, W: Tensor) -> Tensor:
        """Dense matmul message passing: received = W @ msg."""
        return torch.matmul(W, msg)  # [BS, NC, N, D_n]

    def _inject(self, received, H_aug_t, inject_w, inject_b):
        cell_slice = H_aug_t.reshape(H_aug_t.shape[0], self.N_cells, self.D_n)
        inject = torch.einsum("bni,noi->bno", cell_slice, inject_w)
        inject = inject + inject_b.unsqueeze(0)
        inject = inject.reshape(
            H_aug_t.shape[0], self.N_cells, self.alpha, self.D_n)
        received[:, :, self.input_lo:self.input_hi].add_(inject.to(received.dtype))
        return received

    def _mlp2(self, x, w1, b1, w2, b2):
        """Two-layer MLP: F.linear → tanh → F.linear → tanh."""
        flat = x.reshape(-1, x.shape[-1])
        hidden = torch.tanh(F.linear(flat, w1, b1))
        out = torch.tanh(F.linear(hidden, w2, b2))
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    def _state_update(self, received, h, decay, one_minus_decay, identity,
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

    def _modulate_cells(self, h, msg, W, decay_logit,
                        readout_drift, s_mem_live, s_mem_ema_fast,
                        mod_w1, mod_b1, mod_w2, mod_b2):
        """Per-cell neuromodulator step.

        Surprise inputs:
          - readout_drift   : [BS, NC, 1] — per-cell local state churn
          - s_mem_live      : [BS]        — current memory-head negative target-logit
          - s_mem_ema_fast  : [BS]        — fast EMA of s_mem_live
        The two scalars are broadcast to all cells.
        """
        NC, N = self.N_cells, self.C_n
        BS = h.shape[0]

        h_mean = h.mean(dim=2)
        msg_mean = msg.mean(dim=2)
        W_stats = W.abs().mean(dim=-1).mean(dim=2, keepdim=True)  # [BS, NC, 1]
        decay_mean = decay_logit.mean(dim=2, keepdim=True)  # [BS, NC, 1]

        # Broadcast global scalars to every cell.
        s1 = s_mem_live.view(BS, 1, 1).expand(BS, NC, 1).to(h_mean.dtype)
        s2 = s_mem_ema_fast.view(BS, 1, 1).expand(BS, NC, 1).to(h_mean.dtype)

        mod_input = torch.cat([
            h_mean, msg_mean,                  # 2*D_n
            W_stats, decay_mean,               # 2
            readout_drift,                      # 1
            s1, s2,                             # 2
        ], dim=-1)
        hidden = torch.tanh(
            torch.einsum("bni,nih->bnh", mod_input, mod_w1) + mod_b1.unsqueeze(0))
        output = torch.einsum("bnh,nho->bno", hidden, mod_w2) + mod_b2.unsqueeze(0)

        # Unpack: direct delta_W, delta_decay
        delta_W = output[..., :N * N].reshape(BS, NC, N, N)
        delta_decay = output[..., N * N:N * N + N].reshape(BS, NC, N)

        return (
            W + delta_W,
            decay_logit + delta_decay,
        )

    def _step(self, h, msg, W, decay, one_minus_decay,
              H_aug_t, identity, inject_w, inject_b,
              st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
              mg_w1, mg_b1, mg_w2, mg_b2):
        received = self._receive(msg, W)
        received = self._inject(received, H_aug_t, inject_w, inject_b)

        h = self._state_update(received, h, decay, one_minus_decay, identity,
                               st_w1_recv, st_w1_h, st_b1, st_w2, st_b2)
        msg = self._emit_message(h, identity, mg_w1, mg_b1, mg_w2, mg_b2)
        readout = self._readout(msg)
        return h, msg, readout

    def _run_block(self, h, msg, W, decay_logit,
                   s_mem_live, s_mem_ema_fast,
                   prev_readout_cell, prev_readout_full, readout_drift,
                   block_H_mid, block_input_ids,
                   start_t, identity, inject_w, inject_b,
                   st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   mod_w1, mod_b1, mod_w2, mod_b2,
                   lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                   w_decay_rate, gain_fast):
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

        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        one_minus_decay = 1.0 - decay

        for offset in range(block_T):
            t = start_t + offset
            H_mid_t = block_H_mid[:, offset]  # [BS, D]
            tok_t = block_input_ids[:, offset]  # [BS]

            # TBPTT detach
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                W = W.detach()
                decay_logit = decay_logit.detach()
                prev_readout_full = prev_readout_full.detach()
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay

            # --- Live memory-head surprise (no_grad — detached signal) ---
            with torch.no_grad():
                x = prev_readout_full
                if proj_down_w is not None:
                    x = F.linear(x, proj_down_w, proj_down_b)
                x = F.layer_norm(x, (x.shape[-1],), ln_final_w, ln_final_b)
                target_emb = lm_head_w[tok_t]  # [BS, D_embed]
                target_logit = (x * target_emb).sum(dim=-1)  # [BS]
                s_mem_live = (-target_logit).to(h.dtype)
                s_mem_ema_fast = (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live

            # --- Modulate every M tokens using the FRESH surprise ---
            if t % self.config.modulation_interval == 0:
                W, decay_logit = self._modulate_cells(
                    h, msg, W, decay_logit,
                    readout_drift, s_mem_live, s_mem_ema_fast,
                    mod_w1, mod_b1, mod_w2, mod_b2)
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay

            # --- Memory step ---
            h, msg, readout = self._step(
                h, msg, W, decay, one_minus_decay,
                H_mid_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2)

            readouts[:, offset] = readout

            # Update per-cell drift (for next modulation).
            with torch.no_grad():
                new_cell = readout.reshape(BS, NC, D_n)
                readout_drift = (new_cell - prev_readout_cell).abs().mean(
                    dim=-1, keepdim=True).to(h.dtype)
                prev_readout_cell = new_cell.detach()

            # Advance prev_readout_full for next-token memory-head prediction.
            prev_readout_full = readout.detach()

            # Soft W decay (on-graph so later-token loss can train modulator).
            W = W * (1.0 - w_decay_rate)

        return (h, msg, W, decay_logit,
                s_mem_live, s_mem_ema_fast,
                prev_readout_cell, prev_readout_full, readout_drift, readouts)

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_mid: Tensor, input_ids: Tensor, lm):
        """Process T tokens through the memory graph.

        Args:
            H_mid: [BS, T, D] — lower-scan output (detached from LM graph)
            input_ids: [BS, T] — input tokens; used to compute the per-token
                memory-head target-logit surprise signal, where the memory
                head uses readout[t-1] to predict the token at position t.
            lm: LM module — provides mem_head_target_logit and mem_head_logits
                (weight-tied to the main lm_head).

        Returns:
            readouts:      [BS, T, D] — memory readout per token
            mem_pred_loss: scalar      — CE of memory head against input_ids
                                          (used as auxiliary training loss)
        """
        BS, T, _ = H_mid.shape
        if not self._initialized:
            self.initialize_states(BS, H_mid.device)
        if not self._compiled and H_mid.is_cuda:
            self._run_block = torch.compile(
                self._run_block, mode="default", fullgraph=False)
            self._compiled = True

        h = self.h
        msg = self.msg
        W = self.W
        decay_logit = self.decay_logit
        s_mem_live = self.s_mem_live
        s_mem_ema_fast = self.s_mem_ema_fast
        prev_readout_cell = self.prev_readout_cell
        prev_readout_full = self.prev_readout  # full-D, for memory head at block start
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
        mod_w1 = self.mod_w1.to(dt)
        mod_b1 = self.mod_b1.to(dt)
        mod_w2 = self.mod_w2.to(dt)
        mod_b2 = self.mod_b2.to(dt)
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
        w_decay_rate = self.config.w_decay_rate
        gain_fast = self.config.gain_ema_fast

        use_ckpt = self.training and self.config.checkpoint_memory
        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_mid = H_mid[:, start_t:end_t]
            block_input_ids = input_ids[:, start_t:end_t]

            block_args = (
                h, msg, W, decay_logit,
                s_mem_live, s_mem_ema_fast,
                prev_readout_cell, prev_readout_full, readout_drift,
                block_H_mid, block_input_ids,
                start_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                mod_w1, mod_b1, mod_w2, mod_b2,
                lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                w_decay_rate, gain_fast)
            if use_ckpt:
                result = torch.utils.checkpoint.checkpoint(
                    self._run_block, *block_args, use_reentrant=False)
            else:
                result = self._run_block(*block_args)

            (h, msg, W, decay_logit,
             s_mem_live, s_mem_ema_fast,
             prev_readout_cell, prev_readout_full, readout_drift, block_out) = result

            readouts[:, start_t:end_t] = block_out

        # Save state
        self.h = h
        self.msg = msg
        self.W = W
        self.decay_logit = decay_logit
        self.s_mem_live = s_mem_live.detach()
        self.s_mem_ema_fast = s_mem_ema_fast.detach()
        self.prev_readout = prev_readout_full.detach()
        self.prev_readout_cell = prev_readout_cell.detach()
        self.readout_drift = readout_drift.detach()

        # --- Segment-level mem_pred_loss, chunked for VRAM ---
        # Memory head uses readout[t-1] to predict token at position t.
        # Chunk the time axis to avoid materializing [BS, T, V] all at once.
        shifted_all = torch.cat([
            segment_start_prev_readout.unsqueeze(1).to(readouts.dtype),  # [BS, 1, D]
            readouts[:, :-1],                                             # [BS, T-1, D]
        ], dim=1)  # [BS, T, D]

        loss_sum = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        count = 0
        chunk = block_size  # reuse the same chunk size (8)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_readout = shifted_all[:, s:e]
            sub_target = input_ids[:, s:e]
            sub_logits = lm.mem_head_logits(sub_readout)  # [BS, chunk, V]
            loss_sum = loss_sum + F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                sub_target.reshape(-1),
                reduction="sum",
            )
            count += sub_target.numel()
        mem_pred_loss = loss_sum / max(count, 1)

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
        vqvae,
        tau: float = 1.0,
        sample: bool = True,
    ) -> dict:
        """Phase 2 rollout: run the memory loop with VQ-sampled discrete actions.

        At each modulator call:
            1. Build mod_input as in phase 1
            2. Run the modulator to produce raw continuous output
            3. Encode via vqvae.encoder -> low-dim latent z
            4. Sample codes from the per-level distance-based categorical
            5. Reconstruct quantized latent z_q from codes
            6. Decode z_q via vqvae.decoder -> quantized action
            7. Unpack into delta_W, delta_decay and apply to memory state
            8. Record (mod_input, codes, t) for the GRPO gradient pass

        Args:
            H_mid: [BS, T, D] — lower-scan output from (frozen) LM
            input_ids: [BS, T]
            lm: (frozen) LM module — provides lm_head / mem_head weights
            vqvae: (frozen) ActionVQVAE — provides encoder, rvq, decoder
            tau: temperature for the distance-based categorical
            sample: True for multinomial sampling, False for argmax

        Returns:
            dict with:
                readouts: [BS, T, D]
                mod_inputs: [n_calls, BS, NC, mod_in] — state at each modulator call
                codes: [n_calls, BS, NC, num_levels] — sampled codes
                call_positions: [n_calls] — token position `t` of each call
        """
        assert self._initialized, "memory must be initialized before phase2 rollout"

        BS, T, D = H_mid.shape
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

        # Live runtime state (cloned so we don't clobber phase-1 state)
        h = self.h.clone()
        msg = self.msg.clone()
        W = self.W.clone()
        decay_logit = self.decay_logit.clone()
        s_mem_live = self.s_mem_live.clone()
        s_mem_ema_fast = self.s_mem_ema_fast.clone()
        prev_readout_cell = self.prev_readout_cell.clone()
        prev_readout_full = self.prev_readout.clone()
        readout_drift = self.readout_drift.clone()

        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        one_minus_decay = 1.0 - decay

        gain_fast = self.config.gain_ema_fast
        w_decay_rate = self.config.w_decay_rate
        mod_interval = self.config.modulation_interval

        readouts = torch.empty(BS, T, D, device=H_mid.device, dtype=dt)
        mod_input_records: list[Tensor] = []
        codes_records: list[Tensor] = []
        call_positions: list[int] = []

        for t in range(T):
            H_mid_t = H_mid[:, t]
            tok_t = input_ids[:, t]

            # Live surprise (same as phase 1)
            x = prev_readout_full
            if proj_down_w is not None:
                x = F.linear(x, proj_down_w, proj_down_b)
            x = F.layer_norm(x, (x.shape[-1],), ln_final_w, ln_final_b)
            target_emb = lm_head_w[tok_t]
            target_logit = (x * target_emb).sum(dim=-1)
            s_mem_live = (-target_logit).to(dt)
            s_mem_ema_fast = (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live

            # Modulate via VQ sampling every M tokens
            if t % mod_interval == 0:
                # Build mod_input in f32 for numerical consistency with the
                # gradient pass (which also runs the modulator in f32).
                h_mean = h.float().mean(dim=2)
                msg_mean = msg.float().mean(dim=2)
                W_stats = W.float().abs().mean(dim=-1).mean(dim=2, keepdim=True)
                decay_mean = decay_logit.float().mean(dim=2, keepdim=True)
                s1 = s_mem_live.float().view(BS, 1, 1).expand(BS, NC, 1)
                s2 = s_mem_ema_fast.float().view(BS, 1, 1).expand(BS, NC, 1)
                mod_input_f32 = torch.cat([
                    h_mean, msg_mean,
                    W_stats, decay_mean,
                    readout_drift.float(),
                    s1, s2,
                ], dim=-1)  # [BS, NC, mod_in] in f32

                raw_action = self._modulator_forward(mod_input_f32)  # f32 out

                # Normalize, encode, sample codes, decode, un-normalize (all f32)
                action_flat = raw_action.reshape(BS * NC, -1)
                action_norm = vqvae.normalize(action_flat)
                z = vqvae.encoder(action_norm)                   # [BS*NC, latent]
                codes = vqvae.rvq.sample_codes(z, tau=tau, sample=sample)  # [BS*NC, L]
                # Reconstruct z_q from codes
                z_q = torch.zeros_like(z)
                for lvl in range(vqvae.rvq.num_levels):
                    z_q = z_q + vqvae.rvq.codebooks[lvl][codes[:, lvl]]
                quantized_norm = vqvae.decoder(z_q)              # [BS*NC, action_dim]
                quantized = vqvae.denormalize(quantized_norm)
                quantized = quantized.reshape(BS, NC, -1).to(dt)

                # Unpack delta_W, delta_decay and apply (cast to bf16 to match state)
                delta_W = quantized[..., :N * N].reshape(BS, NC, N, N)
                delta_decay = quantized[..., N * N:N * N + N].reshape(BS, NC, N)
                W = W + delta_W
                decay_logit = decay_logit + delta_decay
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay

                # Record on GPU (moving to CPU per-call causes sync overhead).
                # Caller is responsible for memory pressure; at phase-2 scale
                # (BS ~8) the total records fit comfortably.
                mod_input_records.append(mod_input_f32.detach())
                codes_records.append(
                    codes.reshape(BS, NC, -1).detach())
                call_positions.append(t)

            # Memory step (same as phase 1)
            h, msg, readout = self._step(
                h, msg, W, decay, one_minus_decay,
                H_mid_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2)
            readouts[:, t] = readout

            new_cell = readout.reshape(BS, NC, D_n)
            readout_drift = (new_cell - prev_readout_cell).abs().mean(
                dim=-1, keepdim=True).to(dt)
            prev_readout_cell = new_cell
            prev_readout_full = readout
            W = W * (1.0 - w_decay_rate)

        # Persist end-of-rollout memory state so the next rollout continues
        # from here (memory is lifelong across rollouts too).
        self.h = h
        self.msg = msg
        self.W = W
        self.decay_logit = decay_logit
        self.s_mem_live = s_mem_live
        self.s_mem_ema_fast = s_mem_ema_fast
        self.prev_readout_cell = prev_readout_cell
        self.prev_readout = prev_readout_full
        self.readout_drift = readout_drift

        return {
            "readouts": readouts,
            "mod_inputs": torch.stack(mod_input_records, dim=0),   # [n_calls, BS, NC, mod_in]
            "codes": torch.stack(codes_records, dim=0),             # [n_calls, BS, NC, L]
            "call_positions": torch.tensor(call_positions, dtype=torch.long),
        }
