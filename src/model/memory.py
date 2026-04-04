"""Memory Graph — vectorized sparse neuron simulation.

N neurons, each with D_n-dim hidden state and K sparse presynaptic connections.
One neuron step per token, all neurons processed in parallel via bulk tensor ops.

Per-token step:
  0. Neuromodulate: per-neuron MLP predicts delta to w_conn, decay, identity
  1. Receive: gather K neighbor messages, weight by sigmoid(w_conn), sum
  2. Inject: add LM signal to input port neurons
  3. State update: shared MLP + structural decay blend
  4. Emit message: shared MLP + identity residual
  5. Readout: output port neuron messages → LM dim
  6. Hebbian trace update (no grad)
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
        N = config.N
        K = config.K
        D_n = config.D_n
        self.N = N
        self.K = K
        self.D_n = D_n
        self.C_mem = config.C_mem
        self.N_port = config.N_port
        self.alpha = config.alpha

        # --- Connectivity (buffer, not learned) ---
        scores = torch.rand(N, N)
        scores[torch.arange(N), torch.arange(N)] = -float('inf')
        _, top_k = scores.topk(K, dim=1)
        conn_idx, _ = top_k.sort(dim=-1)
        self.register_buffer('conn_idx', conn_idx)

        # --- Per-neuron identity embedding ---
        self.neuron_id = nn.Parameter(torch.randn(N, D_n) * 0.02)

        # --- Shared state MLP: F.linear convention [out, in] ---
        self.state_w1 = nn.Parameter(torch.empty(config.state_mlp_hidden, config.state_in))
        self.state_b1 = nn.Parameter(torch.zeros(config.state_mlp_hidden))
        self.state_w2 = nn.Parameter(torch.empty(D_n, config.state_mlp_hidden))
        self.state_b2 = nn.Parameter(torch.zeros(D_n))

        nn.init.kaiming_uniform_(self.state_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.state_w2, a=math.sqrt(5))
        with torch.no_grad():
            self.state_w2.mul_(0.1)

        # --- Shared message MLP ---
        self.msg_w1 = nn.Parameter(torch.empty(config.msg_mlp_hidden, config.msg_in))
        self.msg_b1 = nn.Parameter(torch.zeros(config.msg_mlp_hidden))
        self.msg_w2 = nn.Parameter(torch.empty(D_n, config.msg_mlp_hidden))
        self.msg_b2 = nn.Parameter(torch.zeros(D_n))

        nn.init.kaiming_uniform_(self.msg_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.msg_w2, a=math.sqrt(5))
        with torch.no_grad():
            self.msg_w2.mul_(0.1)

        # --- Per-neuron neuromodulator ---
        H_mod = config.neuromod_hidden
        self.mod_w1 = nn.Parameter(torch.empty(N, config.mod_in, H_mod))
        self.mod_b1 = nn.Parameter(torch.zeros(N, 1, H_mod))
        self.mod_w2 = nn.Parameter(torch.empty(N, H_mod, config.mod_out))
        self.mod_b2 = nn.Parameter(torch.zeros(N, 1, config.mod_out))

        nn.init.kaiming_uniform_(self.mod_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mod_w2, a=math.sqrt(5))
        with torch.no_grad():
            self.mod_w2.mul_(0.01)

        self._initialized = False

    # ================================================================
    # State management
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int, device: torch.device):
        N, D_n, K = self.N, self.D_n, self.K
        dt = torch.bfloat16

        self.h = torch.randn(BS, N, D_n, device=device, dtype=dt) * 0.01
        self.msg = torch.zeros(BS, N, D_n, device=device, dtype=dt)
        self.w_conn = torch.zeros(BS, N, K, device=device, dtype=dt)
        self.identity = self.neuron_id.unsqueeze(0).expand(BS, -1, -1).clone().to(dt)
        self.decay_logit = torch.zeros(BS, N, device=device, dtype=dt)
        self.hebbian_traces = torch.zeros(BS, N, K, device=device, dtype=dt)
        self._initialized = True

    def detach_states(self):
        self.h = self.h.detach()
        self.msg = self.msg.detach()
        self.w_conn = self.w_conn.detach()
        self.identity = self.identity.detach()
        self.decay_logit = self.decay_logit.detach()

    def reset_states(self, mask: Tensor):
        """Reset state for batch elements where mask is True. mask: [BS] bool."""
        if not self._initialized:
            return
        keep = (~mask).to(self.h.dtype)
        k3 = keep[:, None, None]
        k2 = keep[:, None]
        with torch.no_grad():
            self.h = self.h * k3
            self.msg = self.msg * k3
            self.w_conn = self.w_conn * k3
            self.decay_logit = self.decay_logit * k2
            self.hebbian_traces = self.hebbian_traces * k3
            reset_id = self.neuron_id.unsqueeze(0).to(self.identity.dtype)
            r3 = mask.to(self.identity.dtype)[:, None, None]
            self.identity = self.identity * k3 + reset_id * r3

    # ================================================================
    # Per-token step components (all vectorized over N neurons)
    # ================================================================

    def _modulate(self, identity: Tensor, hebbian: Tensor,
                  w_conn: Tensor, decay_logit: Tensor,
                  mod_w1: Tensor, mod_b1: Tensor,
                  mod_w2: Tensor, mod_b2: Tensor,
                  ) -> tuple[Tensor, Tensor, Tensor]:
        K, D_n = self.K, self.D_n

        mod_input = torch.cat([
            identity,
            hebbian,
            w_conn,
            decay_logit.unsqueeze(-1),
        ], dim=-1)

        x = mod_input.permute(1, 0, 2)
        hidden = torch.tanh(torch.bmm(x, mod_w1) + mod_b1)
        output = torch.bmm(hidden, mod_w2) + mod_b2
        output = output.permute(1, 0, 2)

        dw = output[..., :K]
        ddecay = output[..., K]
        didentity = output[..., K + 1:]

        return w_conn + dw, decay_logit + ddecay, identity + didentity

    def _receive(self, msg: Tensor, w_conn: Tensor
                 ) -> tuple[Tensor, Tensor]:
        gathered = msg[:, self.conn_idx]
        w = torch.sigmoid(w_conn).unsqueeze(-1)
        received = (gathered * w).sum(dim=2)
        return received, gathered

    def _inject(self, received: Tensor, H_aug_t: Tensor) -> Tensor:
        BS = H_aug_t.shape[0]
        inject = H_aug_t.reshape(BS, self.C_mem, self.D_n)
        inject = inject.unsqueeze(2).expand(-1, -1, self.alpha, -1)
        inject = inject.reshape(BS, self.N_port, self.D_n)

        inject_full = torch.zeros(
            BS, self.N, self.D_n, device=received.device, dtype=received.dtype)
        inject_full[:, :self.N_port] = inject.to(received.dtype)
        return received + inject_full

    def _state_update(self, received: Tensor, h: Tensor,
                      identity: Tensor, decay_logit: Tensor,
                      w1: Tensor, b1: Tensor,
                      w2: Tensor, b2: Tensor) -> Tensor:
        BS = received.shape[0]

        state_input = torch.cat([
            received, h, identity, decay_logit.unsqueeze(-1),
        ], dim=-1)

        flat = state_input.reshape(-1, state_input.shape[-1])
        hidden = torch.tanh(F.linear(flat, w1, b1))
        candidate = torch.tanh(F.linear(hidden, w2, b2))
        candidate = candidate.reshape(BS, self.N, self.D_n)

        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        return decay * h + (1.0 - decay) * candidate

    def _emit_message(self, h: Tensor, identity: Tensor,
                      w1: Tensor, b1: Tensor,
                      w2: Tensor, b2: Tensor) -> Tensor:
        BS = h.shape[0]

        msg_input = torch.cat([h, identity], dim=-1)
        flat = msg_input.reshape(-1, msg_input.shape[-1])
        hidden = torch.tanh(F.linear(flat, w1, b1))
        msg_raw = torch.tanh(F.linear(hidden, w2, b2))
        msg_new = msg_raw.reshape(BS, self.N, self.D_n)
        return msg_new + identity

    def _readout(self, msg: Tensor) -> Tensor:
        BS = msg.shape[0]
        port_msg = msg[:, self.N_port:2 * self.N_port]
        port_msg = port_msg.reshape(BS, self.C_mem, self.alpha, self.D_n)
        readout = port_msg.sum(dim=2) * (self.alpha ** -0.5)
        return readout.reshape(BS, -1)

    @staticmethod
    def _update_hebbian(gathered: Tensor, msg_new: Tensor,
                        hebbian: Tensor, decay: float = 0.995):
        with torch.no_grad():
            post = msg_new.detach().unsqueeze(2)
            pre = gathered.detach()
            correlation = (pre * post).sum(dim=-1)
            hebbian.mul_(decay).add_(correlation, alpha=1.0 - decay)

    # ================================================================
    # Main forward
    # ================================================================

    def _step(self, h, msg, w_conn, decay_logit, identity, hebbian,
              H_aug_t, mod_w1, mod_b1, mod_w2, mod_b2,
              st_w1, st_b1, st_w2, st_b2,
              mg_w1, mg_b1, mg_w2, mg_b2):
        """Single neuron step. Checkpointable — pure function of inputs."""
        w_conn, decay_logit, identity = self._modulate(
            identity, hebbian, w_conn, decay_logit,
            mod_w1, mod_b1, mod_w2, mod_b2)

        received, gathered = self._receive(msg, w_conn)
        received = self._inject(received, H_aug_t)

        h = self._state_update(received, h, identity, decay_logit,
                               st_w1, st_b1, st_w2, st_b2)
        msg = self._emit_message(h, identity,
                                 mg_w1, mg_b1, mg_w2, mg_b2)

        readout = self._readout(msg)
        return h, msg, w_conn, decay_logit, identity, readout, gathered

    def forward_segment(self, H_aug: Tensor) -> Tensor:
        """Process T tokens. H_aug: [BS, T, D] (detached from LM).

        Per-step detach: each token step detaches recurrent state so the
        autograd graph for each step is independent. Combined with
        gradient checkpointing, only one step's intermediates exist at a
        time during backward, giving constant memory regardless of T.
        """
        BS, T, D = H_aug.shape
        device = H_aug.device

        h = self.h.detach()
        msg = self.msg.detach()
        w_conn = self.w_conn.detach()
        identity = self.identity.detach()
        decay_logit = self.decay_logit.detach()
        hebbian = self.hebbian_traces

        H_aug = H_aug.to(h.dtype)

        # Pre-cast all weights once
        dt = h.dtype
        mod_w1 = self.mod_w1.to(dt)
        mod_b1 = self.mod_b1.to(dt)
        mod_w2 = self.mod_w2.to(dt)
        mod_b2 = self.mod_b2.to(dt)
        st_w1 = self.state_w1.to(dt)
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)

        ema_decay = self.config.hebbian_ema_decay
        readouts = []

        for t in range(T):
            H_aug_t = H_aug[:, t]

            # Per-step detach: state carries info, not gradient
            h = h.detach()
            msg = msg.detach()
            w_conn = w_conn.detach()
            decay_logit = decay_logit.detach()
            identity = identity.detach()

            # Snapshot hebbian for this step (checkpoint will recompute from
            # saved inputs, so hebbian must not be modified in-place before
            # recomputation). Clone is cheap: [BS, N, K] bf16 ≈ 66 MB at BS=48.
            hebb_snapshot = hebbian.clone()

            if self.training:
                h, msg, w_conn, decay_logit, identity, readout, gathered = \
                    torch.utils.checkpoint.checkpoint(
                        self._step,
                        h, msg, w_conn, decay_logit, identity, hebb_snapshot,
                        H_aug_t, mod_w1, mod_b1, mod_w2, mod_b2,
                        st_w1, st_b1, st_w2, st_b2,
                        mg_w1, mg_b1, mg_w2, mg_b2,
                        use_reentrant=False,
                    )
            else:
                h, msg, w_conn, decay_logit, identity, readout, gathered = \
                    self._step(
                        h, msg, w_conn, decay_logit, identity, hebb_snapshot,
                        H_aug_t, mod_w1, mod_b1, mod_w2, mod_b2,
                        st_w1, st_b1, st_w2, st_b2,
                        mg_w1, mg_b1, mg_w2, mg_b2)

            readouts.append(readout)
            # Update hebbian in-place AFTER checkpoint has saved its snapshot
            self._update_hebbian(gathered, msg, hebbian, ema_decay)

        self.h = h
        self.msg = msg
        self.w_conn = w_conn
        self.identity = identity
        self.decay_logit = decay_logit
        self.hebbian_traces = hebbian

        return torch.stack(readouts, dim=1)  # [BS, T, D]

    # ================================================================
    # Structural plasticity
    # ================================================================

    def rewire_connections(self):
        if not self.config.structural_plasticity:
            return
        if not self._initialized:
            return

        N, K = self.N, self.K
        n_swap = max(1, int(N * K * self.config.plasticity_pct))
        explore_frac = self.config.plasticity_exploration_frac
        device = self.conn_idx.device

        with torch.no_grad():
            conn = self.conn_idx

            hebb = self.hebbian_traces.mean(dim=0)
            flat_hebb = hebb.reshape(-1)
            _, prune_flat = flat_hebb.topk(n_swap, largest=False)
            prune_n = prune_flat // K
            prune_k = prune_flat % K

            exists = torch.zeros(N, N, dtype=torch.bool, device=device)
            row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(conn)
            exists[row_idx, conn] = True

            msg_mag = self.msg.detach().float().norm(dim=-1).mean(dim=0)

            n_exploit = n_swap - int(n_swap * explore_frac)
            new_targets = torch.empty(n_swap, dtype=torch.long, device=device)

            for i in range(n_swap):
                n = prune_n[i].item()

                if i < n_exploit:
                    scores = msg_mag.clone()
                    scores[exists[n]] = -float('inf')
                    scores[n] = -float('inf')
                    best = scores.argmax().item()
                    new_targets[i] = best
                else:
                    candidates = (~exists[n]).nonzero(as_tuple=True)[0]
                    candidates = candidates[candidates != n]
                    if len(candidates) == 0:
                        new_targets[i] = conn[n, prune_k[i]]
                        continue
                    idx = torch.randint(len(candidates), (1,), device=device)
                    new_targets[i] = candidates[idx]

                exists[n, new_targets[i]] = True
                exists[n, conn[n, prune_k[i]]] = False

            conn[prune_n, prune_k] = new_targets

            modified_neurons = prune_n.unique()
            for n_idx in modified_neurons:
                conn[n_idx], _ = conn[n_idx].sort()

            self.w_conn[:, prune_n, prune_k] = 0
            self.hebbian_traces[:, prune_n, prune_k] = 0
