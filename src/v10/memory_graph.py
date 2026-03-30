"""GNN Memory Graph -- shared-weight message passing over N=4096 neurons.

Uses PyG's MessagePassing for efficient sparse gather/scatter.
All neurons share three MLPs (state, message, modulator) conditioned
on a per-neuron learnable identity embedding. Neurons differentiate
through connectivity, inject signal, and state history.

Sequential simulation: one step per token, T steps per segment.
Collects word_states at every step for the decoder's cross-attention.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


# ------------------------------------------------------------------
# Shared-weight neuron step (PyG MessagePassing)
# ------------------------------------------------------------------

class NeuronStep(MessagePassing):
    """Single neuron simulation step using shared MLPs.

    Implements the four-phase step: modulate, gather, update, message.
    Uses PyG's propagate() for sparse weighted message aggregation.

    All three MLPs are shared across all neurons. Identity embedding
    conditions each MLP so neurons can differentiate.
    """

    def __init__(self, D: int, D_id: int, K: int,
                 H_state: int, H_msg: int, H_mod: int):
        super().__init__(aggr='add')
        self.D = D
        self.D_id = D_id
        self.K = K

        # Shared state MLP: (received + inject + h + identity) -> update
        # Input dim: D (received) + D (inject) + D (h) + D_id (identity)
        state_in = 3 * D + D_id
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in, H_state),
            nn.SiLU(),
            nn.Linear(H_state, D),
            nn.Tanh(),
        )

        # Shared message MLP: (h + identity) -> msg
        msg_in = D + D_id
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, H_msg),
            nn.SiLU(),
            nn.Linear(H_msg, D),
            nn.Tanh(),
        )

        # Shared modulator MLP: (hebbian + h + identity + received + inject) -> (w_conn, decay, id_delta)
        # Runs AFTER gather so it can see what the neuron is receiving
        # Input dim: K (hebbian) + D (h) + D_id (identity) + D (received) + D (inject)
        mod_in = K + 3 * D + D_id
        # Output dim: K (w_conn) + 1 (decay) + D_id (identity_delta)
        mod_out = K + 1 + D_id
        self.mod_mlp = nn.Sequential(
            nn.Linear(mod_in, H_mod),
            nn.SiLU(),
            nn.Linear(H_mod, mod_out),
            # No tanh -- sigmoid applied externally for w_conn and decay
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for all MLP weights, zeros for biases."""
        for module in [self.state_mlp, self.msg_mlp, self.mod_mlp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """PyG message function: weight neighbor messages by edge_weight.

        Args:
            x_j: [E, D] -- messages from source neurons
            edge_weight: [E] -- sigmoid(w_conn) for each edge

        Returns:
            weighted: [E, D] -- weighted messages
        """
        return edge_weight.unsqueeze(-1) * x_j

    def forward(self, h: Tensor, msgs: Tensor, inject: Tensor,
                identity: Tensor, edge_index: Tensor,
                w_conn: Tensor, hebbian: Tensor,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One simulation step for all neurons (batched).

        All inputs have a leading batch dimension that is flattened
        into the node dimension for PyG, then reshaped back.

        Args:
            h: [BS, N, D] -- neuron hidden states
            msgs: [BS, N, D] -- current messages (from previous step)
            inject: [BS, N, D] -- per-step inject from LM
            identity: [N, D_id] -- shared identity (expanded per batch)
            edge_index: [2, N*K] -- PyG COO edge index
            w_conn: [BS, N, K] -- connection weights (pre-sigmoid logits)
            hebbian: [BS, N, K] -- hebbian traces

        Returns:
            h_new: [BS, N, D]
            msgs_new: [BS, N, D]
            w_conn_new: [BS, N, K]
            decay: [BS, N, 1]
            identity_new: [N, D_id]
        """
        BS, N, D = h.shape
        K = self.K
        D_id = self.D_id

        # Expand identity for batch: [N, D_id] -> [BS, N, D_id]
        id_expanded = identity.unsqueeze(0).expand(BS, -1, -1)

        # ----------------------------------------------------------
        # Step 1: GATHER (sparse weighted via PyG propagate)
        # Runs FIRST so the modulator can see what the neuron receives.
        # Uses PREVIOUS step's w_conn for edge weights.
        # ----------------------------------------------------------
        w_conn_sig = torch.sigmoid(w_conn)  # [BS, N, K]
        msgs_flat = msgs.reshape(BS * N, D)

        # Build batched edge index: offset each batch element by n*N
        if BS > 1:
            offsets = torch.arange(BS, device=edge_index.device).unsqueeze(1) * N
            batched_src = (edge_index[0].unsqueeze(0) + offsets).reshape(-1)
            batched_tgt = (edge_index[1].unsqueeze(0) + offsets).reshape(-1)
            batched_edge_index = torch.stack([batched_src, batched_tgt])
        else:
            batched_edge_index = edge_index

        edge_weight_flat = w_conn_sig.reshape(-1)

        received_flat = self.propagate(
            batched_edge_index, x=msgs_flat,
            edge_weight=edge_weight_flat,
            size=(BS * N, BS * N),
        )  # [BS*N, D]
        received = received_flat.reshape(BS, N, D)

        # ----------------------------------------------------------
        # Step 2: MODULATE (sees received + inject + h + identity + hebbian)
        # Outputs w_conn for NEXT step's gather, decay, and identity delta.
        # ----------------------------------------------------------
        mod_input = torch.cat([
            hebbian,        # [BS, N, K]
            h,              # [BS, N, D]
            id_expanded,    # [BS, N, D_id]
            received,       # [BS, N, D]
            inject,         # [BS, N, D]
        ], dim=-1)          # [BS, N, K + 3D + D_id]

        mod_out = self.mod_mlp(mod_input)  # [BS, N, K + 1 + D_id]

        w_conn_new = mod_out[..., :K]                    # [BS, N, K]
        decay = torch.sigmoid(mod_out[..., K:K+1])       # [BS, N, 1]
        id_delta = mod_out[..., K+1:]                     # [BS, N, D_id]

        # Update identity (average delta across batch)
        identity_new = identity + id_delta.mean(dim=0)

        # ----------------------------------------------------------
        # Step 3: UPDATE STATE (shared state MLP + structural decay)
        # ----------------------------------------------------------
        id_new_expanded = identity_new.unsqueeze(0).expand(BS, -1, -1)
        state_input = torch.cat([
            received,         # [BS, N, D]
            inject,           # [BS, N, D]
            h,                # [BS, N, D]
            id_new_expanded,  # [BS, N, D_id]
        ], dim=-1)            # [BS, N, 3*D + D_id]

        update = self.state_mlp(state_input)  # [BS, N, D] (tanh-bounded)
        h_new = decay * h + (1.0 - decay) * update

        # ----------------------------------------------------------
        # Step 4: PRODUCE MESSAGE (shared message MLP)
        # ----------------------------------------------------------
        msg_input = torch.cat([
            h_new,            # [BS, N, D]
            id_new_expanded,  # [BS, N, D_id]
        ], dim=-1)            # [BS, N, D + D_id]

        msgs_new = self.msg_mlp(msg_input)  # [BS, N, D] (tanh-bounded)

        return h_new, msgs_new, w_conn_new, decay, identity_new


# ------------------------------------------------------------------
# Memory Graph wrapper
# ------------------------------------------------------------------

class MemoryGraph(nn.Module):
    """GNN memory graph with shared-weight neurons.

    Manages:
    - Sparse connectivity (conn_indices -> edge_index)
    - Neuron identity embeddings
    - Runtime state (h, messages, w_conn, hebbian)
    - Sequential simulation (T steps per segment)
    - Word state collection for decoder cross-attention
    - Structural plasticity (phi-based rewiring)
    """

    def __init__(self, config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K = config.K_connections
        D = config.D_neuron
        D_id = config.D_id

        # Shared neuron step (PyG MessagePassing)
        self.neuron_step = NeuronStep(
            D=D, D_id=D_id, K=K,
            H_state=config.H_state,
            H_msg=config.H_msg,
            H_mod=config.H_mod,
        )

        # Neuron identity: f32, shared across batch
        self.identity = nn.Parameter(
            torch.randn(N, D_id) * (1.0 / math.sqrt(D_id)))

        # Random sparse connectivity: [N, K] indices
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')  # no self-connections
        K_actual = min(K, N - 1)
        conn_indices = torch.zeros(N, K, dtype=torch.long, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        sorted_idx, _ = conn_indices.sort(dim=-1)
        self.register_buffer('conn_indices', sorted_idx)

        # Build PyG edge_index from conn_indices
        self.register_buffer('edge_index', self._build_edge_index())

        # Inject/readout layout
        self._neurons_per_word = config.neurons_per_word
        self._num_words = config.num_words
        self._num_slices = config.D_scan // config.D_neuron

        self._initialized = False

    def _build_edge_index(self) -> Tensor:
        """Convert conn_indices [N, K] to PyG COO edge_index [2, N*K]."""
        N, K = self.conn_indices.shape
        device = self.conn_indices.device
        src = self.conn_indices.reshape(-1)  # source neurons
        tgt = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
        return torch.stack([src, tgt])  # [2, N*K]

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        """Initialize runtime state for batch size BS."""
        device = self.identity.device
        N = self.config.N_neurons
        K = self.config.K_connections
        D = self.config.D_neuron
        dt = self.dtype

        self.h = torch.randn(BS, N, D, device=device, dtype=dt) * 0.1
        self.messages = torch.zeros(BS, N, D, device=device, dtype=dt)
        self.w_conn = torch.zeros(BS, N, K, device=device, dtype=dt)
        self.hebbian_traces = torch.zeros(BS, N, K, device=device, dtype=dt)

        # Structural plasticity: phi correlation matrix
        if self.config.structural_plasticity:
            self.co_activation_ema = torch.zeros(
                N, N, device=device, dtype=torch.float32)
            self._co_activation_ready = False

        self._initialized = True

    def detach_states(self):
        """Detach all runtime state from compute graph."""
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.messages = self.messages.detach()

    # ------------------------------------------------------------------
    # Inject (LM -> memory)
    # ------------------------------------------------------------------

    def _inject_single(self, cc_t: Tensor) -> Tensor:
        """Single-token inject: [BS, D_scan] -> [BS, N, D_neuron].

        Slices D_scan into chunks of D_neuron, replicates each slice
        to its group of neurons. Avoids materializing [BS, T, N, D].
        """
        BS = cc_t.shape[0]
        D = self.config.D_neuron
        n_slices = self._num_slices
        neurons_per_slice = self.config.N_neurons // n_slices

        # [BS, D_scan] -> [BS, n_slices, D_neuron]
        slices = cc_t.view(BS, n_slices, D)
        # [BS, n_slices, D] -> [BS, n_slices, neurons_per_slice, D]
        replicated = slices.unsqueeze(2).expand(-1, -1, neurons_per_slice, -1)
        # [BS, N, D]
        return replicated.reshape(BS, self.config.N_neurons, D)

    # ------------------------------------------------------------------
    # Forward segment (differentiable)
    # ------------------------------------------------------------------

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment of T tokens through the memory graph.

        Args:
            cc_signals: [BS, T, D_scan] -- combined H_mid signals for each token

        Returns:
            word_states: [BS, T, num_words, D_scan] -- per-step neuron states
                grouped into words for decoder cross-attention
        """
        BS, T_seg = cc_signals.shape[0], cc_signals.shape[1]
        N = self.config.N_neurons
        K = self.config.K_connections
        D = self.config.D_neuron

        # TBPTT: detach state at segment boundary
        h = self.h.detach()
        msgs = self.messages.detach()
        w_conn = self.w_conn.detach() if self.w_conn.requires_grad else self.w_conn
        hebbian = self.hebbian_traces.detach()

        # Identity: f32 parameter
        identity = self.identity

        # Edge index for PyG
        edge_index = self.edge_index

        # Collect word states
        word_states_list = []
        total_hebbian = torch.zeros(
            BS, N, K, device=h.device, dtype=h.dtype)
        act_norms = []

        for t in range(T_seg):
            # Per-step inject
            inject_t = self._inject_single(
                cc_signals[:, t].to(self.dtype))

            # Run one neuron step with gradient checkpointing
            # Recomputes forward during backward instead of saving
            # all MLP activations (saves ~18 GB at BS=2, N=4096, T=128)
            def _step_fn(h_, msgs_, inject_, identity_, w_conn_, hebbian_):
                return self.neuron_step(
                    h=h_, msgs=msgs_, inject=inject_,
                    identity=identity_, edge_index=edge_index,
                    w_conn=w_conn_, hebbian=hebbian_,
                )

            h, msgs, w_conn, decay, identity = torch.utils.checkpoint.checkpoint(
                _step_fn, h, msgs, inject_t, identity, w_conn, hebbian,
                use_reentrant=False,
            )

            # Update hebbian traces (detached from autograd)
            with torch.no_grad():
                msg_mag = msgs.detach().norm(dim=-1, keepdim=True)  # [BS, N, 1]
                w_sig = torch.sigmoid(w_conn.detach())  # [BS, N, K]
                total_hebbian = total_hebbian + msg_mag * w_sig
                act_norms.append(msg_mag.squeeze(-1).float())

            # Collect word state: group neurons into words
            # h: [BS, N, D] -> [BS, num_words, neurons_per_word * D]
            word_state_t = h.reshape(
                BS, self._num_words,
                self._neurons_per_word * D,
            ).clone()  # must clone -- h gets overwritten next step
            word_states_list.append(word_state_t)

        # Stack: [BS, T, num_words, D_scan]
        word_states = torch.stack(word_states_list, dim=1)

        # Update persistent state (detached for next segment)
        with torch.no_grad():
            self.h = h.detach()
            self.messages = msgs.detach()
            self.w_conn = w_conn.detach()
            self.hebbian_traces = (
                total_hebbian / max(T_seg, 1)).to(self.dtype)

            # Structural plasticity: phi correlation
            if (self.config.structural_plasticity
                    and hasattr(self, 'co_activation_ema')
                    and act_norms):
                act_trace = torch.stack(act_norms, dim=1)  # [BS, T_seg, N]
                self._update_phi(act_trace)

        return word_states

    # ------------------------------------------------------------------
    # Structural plasticity
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_phi(self, act_trace: Tensor):
        """Compute Pearson phi from per-step activity traces, EMA-smooth.

        Loops over batch elements to avoid [BS, N, N] intermediate
        (would be 3 GB at BS=48, N=4096). Each iteration is [N, N] = 64 MB.

        Args:
            act_trace: [BS, T_seg, N] -- per-step message norms
        """
        BS, T_seg, N = act_trace.shape
        device = act_trace.device

        phi_accum = torch.zeros(N, N, device=device, dtype=torch.float32)

        for b in range(BS):
            trace_b = act_trace[b]  # [T_seg, N]

            # Binary firing: 1 if above 75th percentile
            threshold = torch.quantile(trace_b, 0.75, dim=0, keepdim=True)
            fired = (trace_b > threshold).float()  # [T_seg, N]

            # Pearson correlation
            p_i = fired.mean(dim=0, keepdim=True)       # [1, N]
            fired_centered = fired - p_i                  # [T_seg, N]
            var_i = (p_i * (1 - p_i)).squeeze(0).clamp(min=1e-8)  # [N]
            cov = (fired_centered.T @ fired_centered) / T_seg  # [N, N]
            std_i = var_i.sqrt().unsqueeze(1)            # [N, 1]
            std_j = var_i.sqrt().unsqueeze(0)            # [1, N]
            phi_accum += cov / (std_i * std_j).clamp(min=1e-8)

        phi_mean = phi_accum / BS

        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = (
            ca_decay * self.co_activation_ema
            + (1 - ca_decay) * phi_mean)
        self._co_activation_ready = True

    def rewire_connections(self):
        """Structural plasticity: globally prune/grow connections by phi rank.

        Prunes bottom plasticity_pct% of existing connections (lowest phi)
        and creates new connections for the top plasticity_pct% of unconnected
        pairs (highest phi). 20% of new connections are random (exploration).

        Called at chunk boundaries. Non-differentiable.
        Rebuilds edge_index after rewiring.
        """
        if not self.config.structural_plasticity:
            return
        if not hasattr(self, 'co_activation_ema'):
            return
        if not self._co_activation_ready:
            return

        N = self.config.N_neurons
        K = self.config.K_connections
        explore_frac = self.config.plasticity_exploration_frac
        phi = self.co_activation_ema
        phi.fill_diagonal_(0.0)
        device = phi.device
        conn = self.conn_indices  # [N, K]

        total_conns = N * K
        n_prune = max(1, int(total_conns * self.config.plasticity_pct))

        with torch.no_grad():
            # Gather phi for all existing connections: [N, K]
            conn_phi = phi[
                torch.arange(N, device=device).unsqueeze(1), conn]

            # Flatten and find globally weakest connections
            flat_phi = conn_phi.reshape(-1)  # [N*K]
            _, prune_flat_idx = flat_phi.topk(n_prune, largest=False)

            # Convert flat indices to (neuron, slot) pairs
            prune_n = prune_flat_idx // K
            prune_k = prune_flat_idx % K

            # Build mask of all current connections: [N, N] bool
            conn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            conn_mask.scatter_(1, conn, True)
            conn_mask.fill_diagonal_(True)

            # Find globally strongest UNCONNECTED pairs
            phi_candidates = phi.clone()
            phi_candidates[conn_mask] = -float('inf')
            flat_candidates = phi_candidates.reshape(-1)
            _, grow_flat_idx = flat_candidates.topk(n_prune, largest=True)

            grow_target = grow_flat_idx % N

            # Exploration: random targets for some new connections
            rand_targets = torch.randint(0, N, (n_prune,), device=device)
            use_random = torch.rand(n_prune, device=device) < explore_frac
            grow_target = torch.where(use_random, rand_targets, grow_target)

            # Prevent self-connections and duplicate edges
            for i in range(n_prune):
                n_idx = prune_n[i].item()
                target = grow_target[i].item()
                # No self-connections
                if target == n_idx:
                    target = (target + 1) % N
                # No duplicates: check if target already exists in this neuron's connections
                existing = conn[n_idx].tolist()
                while target in existing or target == n_idx:
                    target = (target + 1) % N
                grow_target[i] = target

            # Apply: replace pruned connections with new targets
            conn[prune_n, prune_k] = grow_target

            # Re-sort
            sorted_idx, _ = conn.sort(dim=-1)
            self.conn_indices.copy_(sorted_idx)

            # Rebuild edge_index
            self.edge_index = self._build_edge_index()

        self._last_rewire_swaps = n_prune

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def runtime_state_dict(self) -> dict:
        state = {
            'h': self.h,
            'messages': self.messages,
            'w_conn': self.w_conn,
            'hebbian_traces': self.hebbian_traces,
        }
        if (self.config.structural_plasticity
                and hasattr(self, 'co_activation_ema')):
            state['co_activation_ema'] = self.co_activation_ema
            state['_co_activation_ready'] = self._co_activation_ready
        return state

    def load_runtime_state(self, state: dict):
        for key, val in state.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if isinstance(current, Tensor) and isinstance(val, Tensor):
                if current.shape != val.shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': "
                        f"expected {current.shape}, got {val.shape}.")
            setattr(self, key, val)
        self._initialized = True
