"""Memory Graph — lightweight hippocampal neurons, trained by backprop.

v9.1: Many simple neurons (N=8192, D=16) instead of few complex ones.
Each neuron is a tiny agent with ~4K per-neuron params:
  - Scalar connection weights w_conn [K] (synaptic strengths)
  - Decay logit [1] (temporal persistence)
  - Per-neuron MLP: cat(h, received, trace_h, trace_recv) → hidden
      → msg head + modulator head

The modulator sees eligibility traces (EMAs of h and received) giving
it temporal context: what the neuron has been experiencing, not just
the current instant. This lets it make strategic plasticity decisions.

The modulator head produces plasticity signals (three-factor learning):
  - plasticity_gate: consolidate (+1) vs reverse (-1) Hebbian update
  - plasticity_lr: how much to learn right now
  - decay_mod: adjust forgetting rate

Hebbian plasticity on w_conn runs during training AND inference.
All params requires_grad=True — trained end-to-end by backprop.
Interface to LM via learned projections (D_cc ↔ D_mem).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config


class MemoryGraph(nn.Module):
    """Hippocampal memory graph with lightweight per-neuron MLPs.

    Per-neuron params (~4K each, requires_grad=True):
        w_conn [K] — scalar synaptic weights per connection
        decay_logit [1] — base temporal persistence
        W1 [4D, H], b1 [H] — hidden layer (input: h, received, trace_h, trace_recv)
        W_msg [H, D], b_msg [D] — message head
        W_mod [H, 3], b_mod [3] — modulator head

    State tensors (per-batch, not learned):
        h [BS, N, D] — neuron hidden states
        prev_messages [BS, N, D] — last outgoing messages
        trace_h [BS, N, D] — EMA of h (what neuron has been encoding)
        trace_received [BS, N, D] — EMA of received (what neuron has been hearing)
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K = config.K_connections
        D = config.D_mem
        H = config.neuron_hidden
        C_mem = config.D // config.D_mem  # 2048 // 16 = 128 memory channels
        self.C_mem = C_mem

        # ============================================================
        # Fixed sparse topology
        # ============================================================
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')
        K_actual = min(K, N - 1)
        conn_indices = torch.zeros(N, K, dtype=torch.long, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        sorted_idx, _ = conn_indices.sort(dim=-1)
        self.register_buffer('conn_indices', sorted_idx)

        # Broadcast inject/readout: every neuron participates in I/O
        # inject_w [N, C_mem]: how much each neuron listens to each memory channel
        self.inject_w = nn.Parameter(torch.zeros(N, C_mem))
        nn.init.uniform_(self.inject_w, -1.0, 1.0)
        # readout_w [C_mem, N]: how each memory channel reads from neurons
        self.readout_w = nn.Parameter(torch.zeros(C_mem, N))
        nn.init.uniform_(self.readout_w, -1.0, 1.0)

        # ============================================================
        # Per-neuron parameters (~2.5K params each)
        # ============================================================

        # Scalar synaptic weights [N, K]
        self.w_conn = nn.Parameter(torch.zeros(N, K))
        nn.init.uniform_(self.w_conn, -0.5, 0.5)

        # Base decay [N]
        self.decay_logit = nn.Parameter(torch.zeros(N))

        # Per-neuron MLP: hidden layer [4D → H]
        # Input: cat(h, received, trace_h, trace_received)
        self.W1 = nn.Parameter(torch.empty(N, 4 * D, H))
        self.b1 = nn.Parameter(torch.zeros(N, H))
        nn.init.kaiming_uniform_(self.W1, nonlinearity='tanh')

        # Message head [H → D]
        self.W_msg = nn.Parameter(torch.empty(N, H, D))
        self.b_msg = nn.Parameter(torch.zeros(N, D))
        # Small init so messages start near zero
        nn.init.kaiming_uniform_(self.W_msg, nonlinearity='tanh')
        self.W_msg.data.mul_(0.1)

        # Modulator head [H → 3] (gate, lr, decay_mod)
        self.W_mod = nn.Parameter(torch.zeros(N, H, 3))
        self.b_mod = nn.Parameter(torch.zeros(N, 3))
        # Zero-init: modulator starts as no-op

        # Hebbian learning rate (meta-parameter, learned by backprop)
        self.hebbian_lr_logit = nn.Parameter(
            torch.tensor(-4.0))  # sigmoid(-4) ≈ 0.018

        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        device = self.w_conn.device
        N = self.config.N_neurons
        D = self.config.D_mem
        dtype = self.dtype

        self.h = torch.randn(BS, N, D, device=device, dtype=dtype) * 0.01
        self.prev_messages = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.trace_h = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.trace_received = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.msg_magnitude = torch.zeros(BS, N, device=device, dtype=dtype)

        self._initialized = True

    # ================================================================
    # Forward segment (differentiable, TBPTT at segment boundaries)
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment of tokens. Fully differentiable.

        Args:
            cc_signals: [BS, T_seg, C_mem, D_mem] — H_mid reshaped to memory channels

        Returns:
            output: [BS, T_seg, C_mem, D_mem] memory signals for upper scan
        """
        BS, T_seg, C_mem, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.K_connections
        stride = self.config.memory_update_stride

        cc_mem = cc_signals.to(self.dtype)

        # Detach h at segment boundary (TBPTT)
        h = self.h.detach()
        prev_msg = self.prev_messages.detach()

        output = torch.empty(BS, T_seg, C_mem, D, device=cc_signals.device,
                             dtype=cc_signals.dtype)

        for t in range(T_seg):
            if t % stride == 0:
                # 1. Gather neighbor messages
                neighbor_msgs = prev_msg[:, self.conn_indices]  # [BS, N, K, D]

                # 2. Scalar-weighted receive
                routing = torch.sigmoid(self.w_conn)  # [N, K]
                weighted = routing.unsqueeze(0).unsqueeze(-1) * neighbor_msgs
                received = weighted.sum(dim=2)  # [BS, N, D]

                # 3. Broadcast CC injection to all neurons
                inject_weights = torch.sigmoid(self.inject_w)  # [N, C]
                broadcast = torch.einsum(
                    'nc,bcd->bnd', inject_weights,
                    cc_mem[:, t].float()).to(self.dtype)
                received = received + broadcast

                # 4. Per-neuron MLP: cat(h, received, trace_h, trace_recv) → hidden
                mlp_in = torch.cat([
                    h.float(), received.float(),
                    self.trace_h.float(), self.trace_received.float(),
                ], dim=-1)  # [BS, N, 4D]
                hidden = torch.tanh(
                    torch.einsum('bnd,ndh->bnh', mlp_in,
                                 self.W1.float()) + self.b1.float())

                # 5. Message head
                msg = torch.tanh(
                    torch.einsum('bnh,nhd->bnd', hidden,
                                 self.W_msg.float()) + self.b_msg.float())
                msg = msg.to(self.dtype)

                # 6. Modulator head
                mod_out = (torch.einsum('bnh,nho->bno', hidden,
                                        self.W_mod.float()) + self.b_mod.float())
                plasticity_gate = torch.tanh(mod_out[..., 0])      # [BS, N]
                plasticity_lr = torch.sigmoid(mod_out[..., 1])      # [BS, N]
                decay_mod = mod_out[..., 2].to(self.dtype)          # [BS, N]

                # 7. Integrate with modulated decay
                eff_decay = torch.sigmoid(
                    self.decay_logit.unsqueeze(0) + decay_mod
                ).unsqueeze(-1)  # [BS, N, 1]
                h = eff_decay * h + (1 - eff_decay) * received

                # 8. Hebbian plasticity on w_conn (detached — side effect)
                with torch.no_grad():
                    eta = torch.sigmoid(self.hebbian_lr_logit)
                    pre = neighbor_msgs.mean(dim=-1)  # [BS, N, K]
                    post = msg.norm(dim=-1, keepdim=True)  # [BS, N, 1]
                    hebb = pre * post  # [BS, N, K]
                    gate = (plasticity_gate * plasticity_lr).unsqueeze(-1)
                    delta_w = eta * gate * hebb  # [BS, N, K]
                    self.w_conn.data += delta_w.mean(dim=0).float()

                # 9. Update eligibility traces (detached — temporal context)
                with torch.no_grad():
                    td = 0.95
                    self.trace_h = (td * self.trace_h + (1 - td) * h.detach()
                                    ).to(self.dtype)
                    self.trace_received = (td * self.trace_received
                                           + (1 - td) * received.detach()
                                           ).to(self.dtype)

                prev_msg = msg

            # 10. Weighted readout from all neurons
            readout_weights = torch.sigmoid(self.readout_w)  # [C_mem, N]
            read_msg = torch.einsum(
                'cn,bnd->bcd', readout_weights,
                prev_msg.float()).to(cc_signals.dtype)  # [BS, C_mem, D]
            output[:, t] = read_msg

        # Update persistent state
        self.h = h.detach()
        self.prev_messages = prev_msg.detach()

        # Track message magnitude for diagnostics
        with torch.no_grad():
            alpha = 0.05
            seg_mag = prev_msg.detach().norm(dim=-1)
            self.msg_magnitude = (1 - alpha) * self.msg_magnitude + alpha * seg_mag

        return output

    # ================================================================
    # Utilities
    # ================================================================

    def detach_states(self):
        """Detach persistent states from compute graph."""
        self.h = self.h.detach()
        self.prev_messages = self.prev_messages.detach()

    def runtime_state_dict(self) -> dict:
        return {
            'h': self.h,
            'prev_messages': self.prev_messages,
            'trace_h': self.trace_h,
            'trace_received': self.trace_received,
            'msg_magnitude': self.msg_magnitude,
        }

    def load_runtime_state(self, state: dict):
        buffer_names = {name for name, _ in self.named_buffers()}
        param_names = {name for name, _ in self.named_parameters()}
        for key, val in state.items():
            if key not in buffer_names and key not in param_names and hasattr(self, key):
                setattr(self, key, val)
        self._initialized = True
