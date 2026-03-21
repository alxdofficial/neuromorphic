"""Memory Graph — diagonal scan + sparse graph message passing.

Per segment (action_every tokens):
  1. Diagonal parallel scan: h[t] = decay * h[t-1] + (1-decay) * gate[t] * u[t]
  2. Sparse graph message passing: x = silu(A_graph @ x)

All temporal operations are parallelized via FLA HGRN kernel (CUDA)
or associative scan (CPU fallback).

State (persistent across chunks, outside autograd):
  primitives: [BS, N, D_mem] — per-neuron gating values
  decay_logit: [BS, N] — per-neuron decay (pre-sigmoid)
  conn_weights: [BS, N, K_conn] — connection weights
  scan_carries: [BS, 1, N, D_mem] — scan carry

Plasticity metrics:
  flow_ema: [BS, N, K_conn] — EMA of signal flow magnitude
  corr_ema: [BS, N, K_conn] — EMA of co-activation correlation
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config

try:
    from fla.ops.hgrn.fused_recurrent import fused_recurrent_hgrn
    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


def _fla_scan(decay_logit: Tensor, x: Tensor,
              h0: Tensor | None = None) -> Tensor:
    """Fused scan via FLA HGRN kernel.

    HGRN computes: h[t] = sigmoid(g) * h[t-1] + (1 - sigmoid(g)) * x[t]
    So we pass x = gate * u (WITHOUT the (1-decay) pre-scaling).
    The kernel applies (1-sigmoid(g)) internally.

    Args:
        decay_logit: [B, D] — constant across T
        x: [B, T, D] — scan input (NOT pre-scaled by 1-decay)
        h0: [B, D] or None

    Returns:
        h_all: [B, T, D]
    """
    B, T, D = x.shape
    g = F.logsigmoid(decay_logit).unsqueeze(1).expand(B, T, D)
    out, _ = fused_recurrent_hgrn(x, g, initial_state=h0)
    return out


def _cpu_scan(decay_logit: Tensor, b: Tensor,
              h0: Tensor | None = None) -> Tensor:
    """CPU fallback parallel scan: h[t] = sigmoid(decay_logit) * h[t-1] + b[t].

    Args:
        decay_logit: [B, D] — constant across T
        b: [B, T, D] — scan input (pre-scaled by 1-decay)
        h0: [B, D] or None

    Returns:
        h_all: [B, T, D]
    """
    import math
    B, T, D = b.shape
    a = torch.sigmoid(decay_logit).unsqueeze(1).expand(B, T, D)

    aa = a
    if h0 is not None:
        bb = b.clone()
        bb[:, 0] = aa[:, 0] * h0 + bb[:, 0]
    else:
        bb = b.clone()

    num_steps = math.ceil(math.log2(T)) if T > 1 else 0
    for d in range(num_steps):
        stride = 1 << d
        a_prev = aa[:, :-stride]
        b_prev = bb[:, :-stride]
        a_cur = aa[:, stride:]
        b_cur = bb[:, stride:]
        new_a = a_cur * a_prev
        new_b = a_cur * b_prev + b_cur
        aa = torch.cat([aa[:, :stride], new_a], dim=1)
        bb = torch.cat([bb[:, :stride], new_b], dim=1)

    return bb


class MemoryGraph:
    """Diagonal-scan memory graph with sparse message passing.

    Architecture per segment:
        CC signals → inject into port neurons
        → diagonal scan (parallel via FLA HGRN kernel)
        → silu(sparse_graph_message_pass)
        → read port neurons → mem_signals
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.device = device
        self.dtype = dtype

        N = config.N_neurons
        K_conn = config.K_connections

        # Fixed topology: random sparse connectivity
        # conn_indices[j, k] = index of k-th neuron connected to neuron j
        conn_indices = torch.zeros(N, K_conn, dtype=torch.long, device=device)
        conn_mask = torch.ones(N, K_conn, dtype=torch.bool, device=device)
        for j in range(N):
            candidates = [i for i in range(N) if i != j]
            n_actual = min(K_conn, len(candidates))
            perm = torch.randperm(len(candidates), device=device)[:n_actual]
            for k in range(n_actual):
                conn_indices[j, k] = candidates[perm[k]]
            if n_actual < K_conn:
                conn_mask[j, n_actual:] = False

        self.conn_indices = conn_indices  # [N, K_conn]
        self.conn_mask = conn_mask        # [N, K_conn]

        # CC port assignment: first C neurons are CC ports
        self.cc_port_idx = torch.arange(config.C, device=device)

        # State tensors (set in initialize())
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, BS: int):
        """Initialize persistent state for batch size BS."""
        N = self.config.N_neurons
        D = self.config.D_mem
        K_conn = self.config.K_connections

        # Per-neuron parameters (neuromodulator-controlled)
        # Primitives gate CC input: u = cc_signals * primitives
        # Init near 1.0 so signals pass through at full strength
        self.primitives = 1.0 + torch.randn(
            BS, N, D, device=self.device, dtype=self.dtype) * 0.02
        # Decay: sigmoid(0) = 0.5 — neutral starting point (50% carry)
        self.decay_logit = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Connection weights (neuromodulator-controlled)
        # Positive init so message passing has net signal (not zero-mean cancellation)
        self.conn_weights = torch.rand(
            BS, N, K_conn, device=self.device, dtype=self.dtype) * 0.2

        # Scan carry (single block)
        self.scan_carries = torch.zeros(
            BS, 1, N, D, device=self.device, dtype=self.dtype)

        # Running stats
        self.mean_input = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.mean_output = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.usage_count = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Plasticity metrics
        self.flow_ema = torch.zeros(
            BS, N, K_conn, device=self.device, dtype=self.dtype)
        self.corr_ema = torch.zeros(
            BS, N, K_conn, device=self.device, dtype=self.dtype)

        # Cached adjacency matrix (rebuilt when conn_weights change)
        self._adjacency_dirty = True
        self._adjacency_cache = None

        self._initialized = True

    @torch.no_grad()
    def forward_segment(self, cc_signals: Tensor,
                        eot_mask: Tensor | None = None) -> Tensor:
        """Process a segment of tokens: one diagonal scan + message passing.

        Single block: scan over T_seg tokens (parallel via FLA), then one
        round of graph message passing for inter-neuron communication.

        Args:
            cc_signals: [BS, T_seg, C, D_mem] — CC signals for this segment
            eot_mask: [BS, T_seg] bool — True at positions where previous token
                      was EOT. Scan carry is killed at these positions.

        Returns:
            mem_signals: [BS, T_seg, C, D_mem] — memory output signals
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons

        # Build input: inject CC signals into port neurons (first C neurons)
        u = torch.zeros(BS, T_seg, N, D, device=self.device, dtype=self.dtype)
        u[:, :, :C] = cc_signals

        # Modulate with primitives
        u = u * self.primitives.unsqueeze(1)

        # Update mean_input stat
        alpha = 0.05
        self.mean_input = (1 - alpha) * self.mean_input + alpha * u.mean(dim=1)

        # Build per-token decay_logit for EOT boundaries
        if eot_mask is not None and eot_mask.any():
            dl = self.decay_logit.unsqueeze(1).expand(BS, T_seg, N).clone()
            eot_expanded = eot_mask.unsqueeze(-1).expand(BS, T_seg, N)
            dl.masked_fill_(eot_expanded, -30.0)
            decay_logit_t = dl
        else:
            decay_logit_t = None

        # 1. Compute scan input
        use_fla = _HAS_FLA and u.is_cuda
        scan_input = torch.sigmoid(u) * u  # gate * u
        if not use_fla:
            decay = torch.sigmoid(self.decay_logit)
            one_minus_decay = (1.0 - decay).unsqueeze(1).unsqueeze(-1)
            scan_input = one_minus_decay * scan_input
        del u

        # 2. Diagonal parallel scan (single block)
        carry = self.scan_carries[:, 0]
        dl = decay_logit_t if decay_logit_t is not None else self.decay_logit
        x, new_carry = self._chunked_scan(
            dl, scan_input, carry, chunk_size=N, use_fla=use_fla)
        self.scan_carries[:, 0] = new_carry
        del scan_input

        # 3. Message passing for inter-neuron communication
        x = self._message_pass(x)
        x = F.silu(x)
        self._update_plasticity_metrics(x)

        # Update running stats
        self.mean_output = (1 - alpha) * self.mean_output + alpha * x.mean(dim=1)
        self.usage_count = (1 - alpha) * self.usage_count + alpha * (
            x.mean(dim=1).norm(dim=-1) > 0.01).to(dtype=self.dtype)

        # Read port neurons
        mem_signals = x[:, :, self.cc_port_idx]
        return mem_signals

    def _chunked_scan(self, decay_logit: Tensor, scan_input: Tensor,
                      carry: Tensor, chunk_size: int = 256,
                      use_fla: bool = False,
                      ) -> tuple[Tensor, Tensor]:
        """Run diagonal scan chunked over neurons.

        Args:
            decay_logit: [BS, N] (constant) or [BS, T, N] (time-varying for EOT)
            scan_input: [BS, T, N, D] — gate*u for FLA, (1-decay)*gate*u for CPU
            carry: [BS, N, D]
            use_fla: if True, use FLA HGRN kernel (CUDA only)

        Returns:
            output: [BS, T, N, D]
            new_carry: [BS, N, D]
        """
        BS, T, N, D = scan_input.shape
        time_varying = decay_logit.dim() == 3  # [BS, T, N] vs [BS, N]
        outputs = []
        carries = []

        for n0 in range(0, N, chunk_size):
            n1 = min(n0 + chunk_size, N)
            nc = n1 - n0

            # Reshape input: [BS, T, nc, D] → [BS*nc, T, D]
            chunk = scan_input[:, :, n0:n1].permute(0, 2, 1, 3).reshape(BS * nc, T, D)
            h0_chunk = carry[:, n0:n1].reshape(BS * nc, D)

            if time_varying:
                # [BS, T, nc] → [BS*nc, T] → expand to [BS*nc, T, D]
                dl_chunk = decay_logit[:, :, n0:n1].permute(0, 2, 1).reshape(BS * nc, T)
                dl_chunk = dl_chunk.unsqueeze(-1).expand(BS * nc, T, D)

                if use_fla:
                    # Time-varying decay: FLA needs [B, T, D] gate
                    g = F.logsigmoid(dl_chunk)
                    out, _ = fused_recurrent_hgrn(chunk, g, initial_state=h0_chunk)
                else:
                    # CPU: pre-scale already done, use time-varying a
                    a = torch.sigmoid(dl_chunk)
                    # Sequential fallback for time-varying (parallel scan needs constant a)
                    h = h0_chunk
                    outs = []
                    for t in range(T):
                        h = a[:, t] * h + chunk[:, t]
                        outs.append(h)
                    out = torch.stack(outs, dim=1)
            else:
                # Constant decay — fast path
                dl_chunk = decay_logit[:, n0:n1].reshape(BS * nc)
                dl_chunk = dl_chunk.unsqueeze(-1).expand(BS * nc, D)

                if use_fla:
                    out = _fla_scan(dl_chunk, chunk, h0_chunk)
                else:
                    out = _cpu_scan(dl_chunk, chunk, h0_chunk)

            # Reshape back: [BS*nc, T, D] → [BS, T, nc, D]
            out = out.reshape(BS, nc, T, D).permute(0, 2, 1, 3)
            outputs.append(out)
            carries.append(out[:, -1])

        return torch.cat(outputs, dim=2), torch.cat(carries, dim=1)

    def _build_adjacency(self) -> Tensor:
        """Build dense adjacency matrix from sparse connectivity.

        Uses a dirty flag to avoid rebuilding when conn_weights haven't changed.

        Returns:
            A: [BS, N, N] — dense weighted adjacency, mean-normalized
        """
        if not self._adjacency_dirty and self._adjacency_cache is not None:
            return self._adjacency_cache

        BS = self.conn_weights.shape[0]
        N = self.config.N_neurons
        K_conn = self.config.K_connections

        w_masked = self.conn_weights * self.conn_mask.unsqueeze(0).to(dtype=self.dtype)
        n_active = self.conn_mask.sum(dim=-1).clamp(min=1).to(dtype=self.dtype)  # [N]
        w_norm = w_masked / n_active.unsqueeze(0).unsqueeze(-1)  # [BS, N, K_conn]

        A = torch.zeros(BS, N, N, device=self.device, dtype=self.dtype)
        idx = self.conn_indices.unsqueeze(0).expand(BS, N, K_conn)
        A.scatter_add_(2, idx, w_norm)

        self._adjacency_cache = A
        self._adjacency_dirty = False
        return A

    def _message_pass(self, x: Tensor) -> Tensor:
        """Sparse graph message passing via dense matmul.

        Builds a [BS, N, N] adjacency matrix and uses bmm for efficiency.
        One matmul replaces 256 gather operations.

        Args:
            x: [BS, T, N, D]

        Returns:
            messages: [BS, T, N, D]
        """
        BS, T, N, D = x.shape

        A = self._build_adjacency()  # [BS, N, N] — ~2MB, tiny

        # Reshape for batched matmul: [BS, N, T*D]
        x_flat = x.permute(0, 2, 1, 3).reshape(BS, N, T * D)

        # Single bmm: [BS, N, N] @ [BS, N, T*D] → [BS, N, T*D]
        # Use float32 for bmm on CPU (bf16 bmm not supported on CPU)
        if not x_flat.is_cuda and x_flat.dtype == torch.bfloat16:
            y_flat = torch.bmm(A.float(), x_flat.float()).to(self.dtype)
        else:
            y_flat = torch.bmm(A, x_flat)

        # Reshape back: [BS, T, N, D]
        return y_flat.reshape(BS, N, T, D).permute(0, 2, 1, 3)

    def _update_plasticity_metrics(self, x: Tensor):
        """Update flow and correlation EMAs from message passing.

        Uses last timestep only — cheap single-step gather on [BS, N, D].
        """
        ema_decay = self.config.plasticity_ema_decay
        x_last = x[:, -1]  # [BS, N, D]

        # Gather neighbor activations for last timestep only
        x_neighbors = x_last[:, self.conn_indices]  # [BS, N, K_conn, D]
        w = self.conn_weights.unsqueeze(-1)  # [BS, N, K_conn, 1]

        flow = (w * x_neighbors).abs().mean(dim=-1)  # [BS, N, K_conn]
        self.flow_ema = ema_decay * self.flow_ema + (1 - ema_decay) * flow

        x_target = x_last.unsqueeze(2).expand_as(x_neighbors)  # [BS, N, K_conn, D]
        corr = (x_target * x_neighbors).mean(dim=-1)  # [BS, N, K_conn]
        self.corr_ema = ema_decay * self.corr_ema + (1 - ema_decay) * corr

    @torch.no_grad()
    def apply_actions(self, delta_primitives: Tensor,
                      delta_conn_weights: Tensor,
                      delta_decay: Tensor):
        """Apply neuromodulator actions to neuron/connection state.

        Args:
            delta_primitives: [BS, N, D_mem]
            delta_conn_weights: [BS, N, K_conn]
            delta_decay: [BS, N]
        """
        self.primitives = self.primitives + delta_primitives
        self.conn_weights = self.conn_weights + delta_conn_weights
        self.decay_logit = self.decay_logit + delta_decay
        self._adjacency_dirty = True

    @torch.no_grad()
    def structural_plasticity(self):
        """Prune dead connections and regrow random new ones.

        Connections with |weight| < threshold are pruned and randomly rewired.
        """
        N = self.config.N_neurons
        threshold = self.config.prune_threshold

        # Find weak connections (averaged across batch)
        mean_weight = self.conn_weights.abs().mean(dim=0)  # [N, K_conn]
        weak = mean_weight < threshold  # [N, K_conn]

        if not weak.any():
            return

        self._adjacency_dirty = True
        for j in range(N):
            weak_j = weak[j].nonzero(as_tuple=True)[0]
            if len(weak_j) == 0:
                continue

            existing = set(self.conn_indices[j].tolist())
            existing.add(j)

            for k in weak_j:
                for _ in range(10):
                    new_target = torch.randint(0, N, (1,),
                                               device=self.device).item()
                    if new_target not in existing:
                        self.conn_indices[j, k] = new_target
                        existing.add(new_target)
                        self.conn_weights[:, j, k] = torch.randn(
                            self.conn_weights.shape[0],
                            device=self.device, dtype=self.dtype) * 0.1
                        self.flow_ema[:, j, k] = 0
                        self.corr_ema[:, j, k] = 0
                        break

    @torch.no_grad()
    def get_neuron_obs(self) -> Tensor:
        """Build observation tensor for neuromodulator.

        Returns:
            obs: [BS, N, obs_dim]
        """
        parts = [
            self.primitives,                              # [BS, N, D_mem]
            self.mean_input,                              # [BS, N, D_mem]
            self.mean_output,                             # [BS, N, D_mem]
            self.usage_count.unsqueeze(-1),               # [BS, N, 1]
            torch.sigmoid(self.decay_logit).unsqueeze(-1),# [BS, N, 1]
        ]

        # Routing entropy from connection weights
        w_abs = self.conn_weights.abs()
        w_norm = w_abs / w_abs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        eps = torch.tensor(1e-4, dtype=self.dtype, device=self.device)
        entropy = -(w_norm * (w_norm + eps).log()).sum(dim=-1, keepdim=True)
        parts.append(entropy)  # [BS, N, 1]

        # Plasticity metrics (use correction=0 for population std, avoids NaN)
        parts.append(self.flow_ema.mean(dim=-1, keepdim=True))                     # [BS, N, 1]
        parts.append(self.flow_ema.std(dim=-1, keepdim=True, correction=0))        # [BS, N, 1]
        parts.append(self.corr_ema.mean(dim=-1, keepdim=True))                     # [BS, N, 1]
        parts.append(self.flow_ema.min(dim=-1, keepdim=True)[0])                   # [BS, N, 1]

        return torch.cat(parts, dim=-1)

    @property
    def obs_dim(self) -> int:
        """Observation dimension for neuromodulator."""
        # D_mem*3 + 2 (usage,decay) + 1 (entropy) + 4 (plasticity metrics)
        return self.config.D_mem * 3 + 7

    @torch.no_grad()
    def reset_streams(self, mask: Tensor):
        """Reset memory state for specific batch elements (at EOT boundaries).

        Args:
            mask: [BS] bool — True for elements to reset
        """
        if not mask.any():
            return

        self._adjacency_dirty = True
        m = mask.unsqueeze(-1)    # [BS, 1]
        m2 = m.unsqueeze(-1)      # [BS, 1, 1]
        m3 = m2.unsqueeze(-1)     # [BS, 1, 1, 1]

        self.primitives = self.primitives * (~m2).to(dtype=self.dtype) + \
            (1.0 + torch.randn_like(self.primitives) * 0.02) * m2.to(dtype=self.dtype)
        self.decay_logit = self.decay_logit * (~m).to(dtype=self.dtype)
        self.conn_weights = self.conn_weights * (~m2).to(dtype=self.dtype) + \
            torch.rand_like(self.conn_weights) * 0.2 * m2.to(dtype=self.dtype)
        self.scan_carries = self.scan_carries * (~m3).to(dtype=self.dtype)
        self.mean_input = self.mean_input * (~m2).to(dtype=self.dtype)
        self.mean_output = self.mean_output * (~m2).to(dtype=self.dtype)
        self.usage_count = self.usage_count * (~m).to(dtype=self.dtype)
        self.flow_ema = self.flow_ema * (~m2).to(dtype=self.dtype)
        self.corr_ema = self.corr_ema * (~m2).to(dtype=self.dtype)
