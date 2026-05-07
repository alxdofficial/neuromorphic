"""ColumnGraphMemory — the memory substrate module.

See docs/column_graph.md for the full design.

Public interface:
    mem = ColumnGraphMemory(cfg)
    mem.begin_segment(B, device)     # reset persistent state
    for token_t in sequence:
        logits = mem.step(token_t)   # [B, K_horizons, V]
        loss = ... (from logits)
    # TBPTT: detach state at block boundary
    mem.detach_state()

The module owns: column state s, E_bias, pred_buf, surprise_ema,
input_ctx_ema, mag_ema, var_ema, traffic_ema, counters. These are
*buffers*, not parameters.

Trainable parameters: update_MLP, content_MLP, delta_MLP, score_MLP,
column identity table, decay_proj, cross-attn input/output projections,
tied token_emb (owned by the StandaloneLM wrapper, passed in at step()),
MultiHorizonReadout (horizon_emb + pred_head), neuromod trunk + heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.kernels import TRITON_AVAILABLE, weighted_gather
from src.column_graph.topology import Topology, build_topology
from src.column_graph.neuromod import Neuromod
from src.column_graph.plasticity import (
    compute_activities,
    hebbian_update_incoming,
)
from src.column_graph.readout import (
    MultiHorizonReadout,
    init_prediction_buffer,
    multi_horizon_surprise,
    write_prediction_buffer,
)


def _rmsnorm(dim: int) -> nn.Module:
    from src.column_graph.readout import _FallbackRMSNorm
    return _FallbackRMSNorm(dim)


# =====================================================================
# Column-local MLPs
# =====================================================================


class ColumnMLPs(nn.Module):
    """Per-column / per-edge shared-weight compute blocks.

    Hot path per token (unchanged in structure):
      content_mlp(s, id) → m_out       # per-column, fires every token
      update_mlp(s, id, incoming) → s_new  # per-column, fires every token

    Per-edge scoring (NEW: MLP-projected multi-head bilinear):
      q[c] = q_proj(s, id)              # [N, H, D_q] — state-dependent
      k[c] = k_proj(id)                 # [N, H, D_q] — static within a segment, cache
      score[A, k] = sum_h q[A, h] · k[out_nbrs[A,k], h] + E_bias[A, k]
      w_out[A, k] = σ(score)

    Dropped: delta_mlp (same content to every out-neighbor; per-edge tailoring
    was 36.5 GFLOPs/token — 72% of per-token compute — and thesis-wise is
    expressable through per-edge w_out scaling of a shared m_out).

    Dropped: score_mlp on concat(s, id_src, id_dst) → now replaced with a
    bilinear form whose q/k projections are small nonlinear MLPs running
    per-column (N work) rather than per-edge (N·K work). Preserves thesis
    requirement that routing depend on (state, src-id, dst-id) while running
    ~27× cheaper.
    """

    def __init__(self, cfg: ColumnGraphConfig) -> None:
        super().__init__()
        D_s = cfg.D_s
        D_id = cfg.D_id

        # content_MLP: (D_s_norm + D_id) → 2·D_s → D_s
        # This is the ONLY nonlinear-transform MLP per tick. Cross-channel
        # mixing happens here at emission time. Receiver-side state update
        # is pure LIF (elementwise) — see _propagate_pure.
        self.content_norm = _rmsnorm(D_s)
        self.content_mlp = nn.Sequential(
            nn.Linear(D_s + D_id, cfg.ffn_mult_content * D_s),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult_content * D_s, D_s),
        )

        # Multi-head bilinear scoring.
        #
        # q_proj maps (s, id_src) → [n_heads, D_q]
        # k_proj maps id_dst → [n_heads, D_q]  — static within a segment
        H = cfg.n_score_heads
        D_q = cfg.D_q_per_head
        self.n_heads = H
        self.D_q = D_q
        self.q_proj = nn.Sequential(
            nn.Linear(D_s + D_id, 2 * H * D_q),
            nn.GELU(),
            nn.Linear(2 * H * D_q, H * D_q),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(D_id, 2 * H * D_q),
            nn.GELU(),
            nn.Linear(2 * H * D_q, H * D_q),
        )
        # Zero-init final q/k projections so initial score = 0 → w_out = 0.5
        # uniform at init (gentle routing, same as old score_MLP zero-init).
        nn.init.zeros_(self.q_proj[-1].weight)
        nn.init.zeros_(self.q_proj[-1].bias)
        nn.init.zeros_(self.k_proj[-1].weight)
        nn.init.zeros_(self.k_proj[-1].bias)

        # Depth-scaled init on first layer of each MLP: std = 1/sqrt(depth=2) ≈ 0.014.
        for module in [self.content_mlp[0], self.q_proj[0], self.k_proj[0]]:
            nn.init.normal_(module.weight, mean=0.0, std=0.014)


# =====================================================================
# ColumnGraphMemory
# =====================================================================


@dataclass
class MemoryReadout:
    """Return bundle from step()."""
    logits: torch.Tensor                  # [B, K_horizons, V]
    surprise_ema: torch.Tensor            # [B, K_horizons]
    eta_global: torch.Tensor | None        # [B] scalar, only on mod_period steps


class ColumnGraphMemory(nn.Module):
    def __init__(self, cfg: ColumnGraphConfig, tied_token_emb: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.tied_token_emb = tied_token_emb   # stored as a reference, not copied

        # Topology (CPU buffers built once at init, moved to device on first step).
        topo = build_topology(
            plane_rows=cfg.plane_rows,
            plane_cols=cfg.plane_cols,
            L=cfg.L,
            K=cfg.K,
            p_rewire=cfg.p_rewire,
            K_intra_fraction=cfg.K_intra_fraction,
            seed=cfg.topology_seed,
        )
        self.register_buffer("out_nbrs", topo.out_nbrs, persistent=False)
        self.register_buffer("edge_src", topo.edge_src, persistent=False)
        self.register_buffer("edge_dst", topo.edge_dst, persistent=False)
        self.register_buffer("plane_ids", topo.plane_ids, persistent=False)
        self.register_buffer("input_positions", topo.input_positions, persistent=False)
        self.register_buffer("output_positions", topo.output_positions, persistent=False)
        # Inverse adjacency for Triton scatter-gather kernel (atomic-free).
        self.register_buffer("in_src", topo.in_src, persistent=False)
        self.register_buffer("in_edge_flat", topo.in_edge_flat, persistent=False)
        self.register_buffer("in_mask", topo.in_mask, persistent=False)
        self.K_in_max = topo.K_in_max

        # Column identity table (learned)
        torch.manual_seed(cfg.init_seed)
        self.col_id = nn.Parameter(torch.randn(cfg.N, cfg.D_id) * 0.02)

        # Per-column learned decay α = sigmoid(decay_proj(id))
        # decay_proj: D_id → 1, bias initialized to logit(0.5) = 0
        self.decay_proj = nn.Linear(cfg.D_id, 1)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.decay_proj.bias)

        # Column compute MLPs
        self.mlps = ColumnMLPs(cfg)

        # Cross-attention: input injection (single multi-head pass)
        # We project the token embedding (= h_input of size D_s) to per-head
        # K and V, then each input-plane column computes a scalar gate via
        # dot(q, k) and adds gate·v to its state. Simple, fast.
        self.in_norm = _rmsnorm(cfg.D_s)
        self.in_q_proj = nn.Linear(cfg.D_s + cfg.D_id, cfg.D_s, bias=False)
        self.in_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.in_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        # Zero-init v_proj so inject is a no-op at day 0 (state stays bounded).
        nn.init.zeros_(self.in_v_proj.weight)

        # Cross-attention: output readout (single motor query over N_out columns)
        self.out_norm = _rmsnorm(cfg.D_s)
        self.motor_query = nn.Parameter(torch.randn(cfg.D_s) * 0.02)
        self.out_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.out_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)

        # Multi-horizon readout
        self.readout = MultiHorizonReadout(cfg)

        # Neuromod
        self.neuromod = Neuromod(cfg)

        # --- Persistent state (buffers, not parameters) ---
        # Allocated lazily in begin_segment() because they depend on B and device.
        self._state_initialized = False
        # Cache for k_proj(id) — static within a segment since id doesn't
        # change between optimizer steps. Invalidated in begin_segment()
        # (pessimistic; works if caller calls begin_segment between opt steps).
        self._k_cache: torch.Tensor | None = None
        self.s: torch.Tensor                # [B, N, D_s]
        self.E_bias_flat: torch.Tensor       # [N*K] fp32 — plastic long-term memory
        self.pred_buf: torch.Tensor          # [B, K_buf, K_horizons, V]
        self.surprise_ema: torch.Tensor      # [B, K_horizons]
        self.surprise_prev: torch.Tensor     # [B, K_horizons]  for Δsurprise
        self.pred_cursor: int = 0
        self.pred_filled: int = 0
        self.tick_counter: int = 0

    # -----------------------------------------------------------------
    # State management — three separate concerns
    # -----------------------------------------------------------------
    #
    # 1. Working memory (h, prediction buffer, surprise EMAs, counters) —
    #    per-document state. `begin_segment()` resets this. Analogous to
    #    "clearing short-term memory at the start of a new conversation."
    #
    # 2. Plastic long-term memory (E_bias) — accumulates across
    #    training steps and across documents at inference. Only reset
    #    by `reset_plastic_memory()` (called once at init / explicit reset).
    #
    # 3. Autograd graph — severed at TBPTT boundaries by `detach_state()`.
    #    Preserves all numerical values; only cuts the gradient trail so
    #    backward is bounded.

    def begin_segment(self, B: int, device: torch.device) -> None:
        """Reset **working memory** for a new document. Preserves plastic E_bias.

        Zeros: column states `s`, prediction ring buffer, surprise EMAs,
        and ring/tick counters. Invalidates `_k_cache`.

        Does NOT touch: `E_bias_flat` (long-term plastic memory). If you
        want a full cold start (e.g., at training launch), call
        `reset_plastic_memory()` separately or together.
        """
        cfg = self.cfg
        dtype_fast = self._fast_dtype(device)
        self._k_cache = None

        self.s = torch.zeros(B, cfg.N, cfg.D_s, device=device, dtype=dtype_fast)
        self.pred_buf = init_prediction_buffer(
            B, cfg.K_buf, cfg.K_horizons, cfg.vocab_size, device, dtype_fast
        )
        self.surprise_ema = torch.zeros(
            B, cfg.K_horizons, device=device, dtype=torch.float32
        )
        self.surprise_prev = torch.zeros_like(self.surprise_ema)
        self.pred_cursor = 0
        self.pred_filled = 0
        self.tick_counter = 0

        # Lazy-initialize E_bias_flat on the first call (so we don't need
        # an explicit reset_plastic_memory for the common training path
        # to just work). Callers who want an explicit hard reset can still
        # call reset_plastic_memory after begin_segment.
        if not hasattr(self, "E_bias_flat") or self.E_bias_flat is None:
            self.reset_plastic_memory(device)

        self._state_initialized = True

    def reset_plastic_memory(self, device: torch.device) -> None:
        """Hard reset of long-term plastic memory (`E_bias_flat`).

        Typically called once at training start or on explicit "forget
        everything" command. Does not affect working memory — call
        `begin_segment()` for that separately (or together for a full
        cold start).
        """
        self.E_bias_flat = torch.zeros(
            self.cfg.num_edges, device=device, dtype=torch.float32
        )

    def detach_state(self) -> None:
        """TBPTT boundary: sever autograd graph, preserve all values.

        Every state tensor is `.detach()`-ed so backward through the next
        block doesn't traverse this block's forward graph. Numerical
        values are unchanged — E_bias keeps accumulating, s keeps
        evolving from here. Invalidates `_k_cache` because k was
        computed under the old graph.
        """
        if not self._state_initialized:
            return
        self.s = self.s.detach()
        self.E_bias_flat = self.E_bias_flat.detach()
        self.pred_buf = self.pred_buf.detach()
        self.surprise_ema = self.surprise_ema.detach()
        self.surprise_prev = self.surprise_prev.detach()
        self._k_cache = None

    def _fast_dtype(self, device: torch.device) -> torch.dtype:
        choice = self.cfg.state_dtype
        if choice == "bf16":
            return torch.bfloat16
        if choice == "fp32":
            return torch.float32
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    # -----------------------------------------------------------------
    # One token step
    # -----------------------------------------------------------------

    def step(self, token_id: torch.Tensor) -> MemoryReadout:
        """Single-token step. Legacy / inference API; training should prefer
        block-level calls via `run_block(...)`.
        """
        assert self._state_initialized, "call begin_segment() first"
        cfg = self.cfg
        logits, m_out, w_out_flat, incoming = self._hot_forward(token_id)
        self._surprise_and_buffer_bookkeeping(token_id, logits)
        eta_global = None
        self.tick_counter += 1
        if self.tick_counter % cfg.mod_period == 0:
            eta_global = self._plasticity_step(m_out, incoming, w_out_flat)
        return MemoryReadout(
            logits=logits, surprise_ema=self.surprise_ema, eta_global=eta_global
        )

    def _hot_forward(self, token_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy per-tick hot path (kept for backward compat + single-token API).

        Returns (logits, m_out, w_out_flat, incoming). Thin wrapper around the
        primitives that now live in _run_prop_block_pure.
        """
        h_input = self.tied_token_emb(token_id)
        self._inject(h_input)
        m_out, w_out_flat, incoming = self._propagate()
        motor = self._readout_motor()
        logits = self.readout(motor, self.tied_token_emb.weight)
        return logits, m_out, w_out_flat, incoming

    # -----------------------------------------------------------------
    # Pure-functional block — for block-level torch.compile
    # -----------------------------------------------------------------
    #
    # Design: given a block of tokens, run block_len ticks of
    # inject+propagate+readout. Return:
    #   - new_s (final column state)
    #   - motor_stack [B, block_len, D_s]  (for out-of-compile readout+CE)
    #   - last_m_out, last_incoming, last_w_out_flat
    #       (activities from the FINAL tick, used by plasticity step
    #        which runs OUTSIDE the compiled region)
    #
    # Nothing inside this method mutates self's Python attrs (pred_cursor,
    # tick_counter, EMAs, ring buffer). Those are bookkeeping and live
    # in walk_segment() outside the compile region. This keeps Inductor's
    # graph stable and doesn't trigger recompiles on Python-int variation.
    #
    # The state `s` flows in as an argument and out as a return. The caller
    # assigns the returned s back to self.s *after* the compiled call.
    def _run_prop_block_pure(
        self,
        tokens_block: torch.Tensor,       # [B, block_len] Long
        s: torch.Tensor,                   # [B, N, D_s]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (s_final, motor_stack, last_m_out, last_incoming, last_w_out_flat)."""
        B, block_len = tokens_block.shape
        N, K, D_s = self.cfg.N, self.cfg.K, self.cfg.D_s
        state_dtype = s.dtype

        motors_list: list[torch.Tensor] = []
        last_m_out = s.new_zeros((B, N, D_s))
        last_incoming = s.new_zeros((B, N, D_s))
        last_w_out = s.new_zeros((B, N * K))

        for t in range(block_len):
            token_id = tokens_block[:, t]
            h_input = self.tied_token_emb(token_id)

            # --- inject (pure on s: returns new s)
            s = self._inject_pure(s, h_input)

            # --- propagate (pure: returns new_s, m_out, w_out_flat, incoming)
            s, m_out, w_out_flat, incoming = self._propagate_pure(s)

            # --- readout motor (pure on s)
            motor = self._readout_motor_pure(s)
            motors_list.append(motor)

            last_m_out = m_out
            last_incoming = incoming
            last_w_out = w_out_flat

        motor_stack = torch.stack(motors_list, dim=1)  # [B, block_len, D_s]
        return s, motor_stack, last_m_out, last_incoming, last_w_out

    # -----------------------------------------------------------------
    # Block-level driver: compilable forward of `block_len` tokens.
    # -----------------------------------------------------------------

    def compile_block(self, mode: str = "default") -> None:
        """Apply torch.compile to the pure propagation block.

        After this, `run_block(...)` goes through the compiled path. Must
        be called after model is on GPU.
        """
        self._run_prop_block_pure = torch.compile(
            self._run_prop_block_pure,
            mode=mode,
            fullgraph=False,
            dynamic=False,
        )

    def run_block(
        self, tokens_block: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drive one block of block_len tokens. Updates self.s, returns:
          - motor_stack [B, block_len, D_s]  — for downstream CE
          - last_m_out [B, N, D_s]
          - last_incoming [B, N, D_s]
          - last_w_out_flat [B, N·K]

        Ring-buffer bookkeeping + surprise EMA updates are driven by the
        caller, outside the compiled region. Plasticity is also called
        outside via `self._plasticity_step(last_m_out, last_incoming,
        last_w_out_flat)` if the tick_counter hits mod_period.
        """
        assert self._state_initialized, "call begin_segment() first"
        s_new, motor_stack, last_m_out, last_incoming, last_w_out = (
            self._run_prop_block_pure(tokens_block, self.s)
        )
        self.s = s_new
        return motor_stack, last_m_out, last_incoming, last_w_out

    # -----------------------------------------------------------------
    # Inject / propagate / readout
    # -----------------------------------------------------------------

    def _inject_pure(self, s: torch.Tensor, h_input: torch.Tensor) -> torch.Tensor:
        """Pure variant of _inject: returns a new s. No self-mutation."""
        B = h_input.shape[0]
        in_cols = self.input_positions
        s_in = s[:, in_cols]
        id_in = self.col_id[in_cols]
        id_in_exp = id_in.unsqueeze(0).expand(B, -1, -1).to(s_in.dtype)

        norm_s = self.in_norm(s_in).to(s_in.dtype)
        q = self.in_q_proj(torch.cat([norm_s, id_in_exp], dim=-1))
        k = self.in_k_proj(h_input)
        v = self.in_v_proj(h_input)

        scale = 1.0 / (self.cfg.D_s ** 0.5)
        gate = torch.sigmoid(torch.sum(q * k.unsqueeze(1), dim=-1) * scale)
        delta = (gate.unsqueeze(-1) * v.unsqueeze(1)).to(s_in.dtype)

        return s.index_copy(
            1, in_cols, (s.index_select(1, in_cols) + delta).to(s.dtype)
        )

    # Legacy in-place wrapper used by _hot_forward / tests.
    def _inject(self, h_input: torch.Tensor) -> None:
        self.s = self._inject_pure(self.s, h_input)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy in-place wrapper — delegates to _propagate_pure and assigns self.s."""
        self.s, m_out, w_out_flat, incoming = self._propagate_pure(self.s)
        return m_out, w_out_flat, incoming

    def _propagate_pure(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure propagation round.

        Core ops (all tensor-core shaped, no per-edge MLPs):
          m_out[c]   = content_mlp(norm(s[c]) || id[c])            # [B, N, D_s]
          q[c, h, d] = q_proj(norm(s[c]) || id[c])                  # [B, N, H, D_q]
          k[c, h, d] = k_proj_cached(id[c])                         # [N, H, D_q]
                       (cached at segment start; id is static)
          score[A, k] = Σ_h q[A, h] · k[out_nbrs[A,k], h]           # [B, N, K]
                         + E_bias[A, k]
          w_out[A, k] = σ(score)                                    # [B, N, K]
          msg_edge[A, k] = w_out[A, k] · m_out[A]   (same content to every neighbour)
          incoming[c] = Σ_{A: (A→c)∈edges} msg_edge[A, edge_idx(A→c)]
          s_new      = α·s + (1-α)·tanh(update_mlp(norm(s), id, norm(incoming)))

        All per-edge work is now O(H·D_q) (one multi-head dot product) rather
        than O(D_s·D_id + D_s²) (an MLP). No K-chunking needed.

        Returns (s_new, m_out, w_out_flat, incoming).
        """
        cfg = self.cfg
        B, N, K, D_s, D_id = s.shape[0], cfg.N, cfg.K, cfg.D_s, cfg.D_id
        H, D_q = self.mlps.n_heads, self.mlps.D_q
        state_dtype = s.dtype

        # --- Content + Q (per-column, state-dependent; run once per tick)
        s_norm = self.mlps.content_norm(s).to(state_dtype)
        id_exp = self.col_id.unsqueeze(0).expand(B, -1, -1).to(state_dtype)
        content_in = torch.cat([s_norm, id_exp], dim=-1)              # [B, N, D_s + D_id]
        m_out = self.mlps.content_mlp(content_in)                      # [B, N, D_s]
        q = self.mlps.q_proj(content_in).view(B, N, H, D_q)            # [B, N, H, D_q]

        # --- K (static within segment; pull from cache if available)
        k = self._k_cache
        if k is None:
            k = self.mlps.k_proj(self.col_id.to(state_dtype)).view(N, H, D_q)
            self._k_cache = k
        k_at_dst = k[self.out_nbrs]                                    # [N, K, H, D_q]
        k_at_dst = k_at_dst.to(state_dtype)

        # --- Bilinear score per edge: Σ_h q[A, h, :] · k[dst, h, :]
        scores = torch.einsum("bnhd,nkhd->bnk", q, k_at_dst)           # [B, N, K]
        E_bias_2d = self.E_bias_flat.view(N, K).to(state_dtype)
        w_out = torch.sigmoid(scores + E_bias_2d.unsqueeze(0))         # [B, N, K]

        # --- Weighted in-gather (Triton kernel on CUDA, reference on CPU).
        # Atomic-free: for each destination column, gather contributions from
        # its in-edges. No [B, N, K, D_s] materialization, no atomic scatter.
        w_out_flat = w_out.reshape(B, N * K)
        if TRITON_AVAILABLE and s.is_cuda:
            incoming = weighted_gather(
                m_out.to(state_dtype).contiguous(),
                w_out_flat.to(state_dtype).contiguous(),
                self.edge_src, self.edge_dst, self.out_nbrs,
                self.in_src, self.in_edge_flat, self.in_mask,
                self.K_in_max, K,
            )
        else:
            msg_edge = w_out.unsqueeze(-1) * m_out.unsqueeze(2)        # [B, N, K, D_s]
            msg_flat = msg_edge.reshape(B, N * K, D_s)
            incoming = torch.zeros_like(s).to(state_dtype)
            incoming = incoming.index_add(1, self.edge_dst, msg_flat)

        # --- LIF state update: pure elementwise leaky-integrator.
        # Cross-channel mixing already happened at emission (content_mlp on
        # the sender side). Receiver just integrates what arrived.
        # α is per-column learned (conditions integration speed on id).
        alpha = torch.sigmoid(self.decay_proj(self.col_id)).squeeze(-1)  # [N]
        alpha = alpha.unsqueeze(0).unsqueeze(-1).to(state_dtype)         # [1, N, 1]
        s_new = (alpha * s + (1.0 - alpha) * torch.tanh(incoming)).to(state_dtype)

        return s_new, m_out, w_out_flat, incoming

    def _readout_motor(self) -> torch.Tensor:
        """Legacy wrapper — uses self.s."""
        return self._readout_motor_pure(self.s)

    def _readout_motor_pure(self, s: torch.Tensor) -> torch.Tensor:
        """Cross-attn motor vector from output-plane column states."""
        cfg = self.cfg
        B = s.shape[0]
        state_dtype = s.dtype
        out_cols = self.output_positions                     # [N_out]
        s_out = self.out_norm(s[:, out_cols]).to(state_dtype)

        k = self.out_k_proj(s_out)                           # [B, N_out, D_s]
        v = self.out_v_proj(s_out)                           # [B, N_out, D_s]
        q = self.motor_query.to(state_dtype).unsqueeze(0).expand(B, -1)  # [B, D_s]

        # Dot-product attention
        scale = 1.0 / (cfg.D_s ** 0.5)
        scores = torch.sum(k * q.unsqueeze(1), dim=-1) * scale   # [B, N_out]
        attn = F.softmax(scores, dim=-1)                          # [B, N_out]
        motor = torch.sum(attn.unsqueeze(-1) * v, dim=1)          # [B, D_s]
        return motor

    # -----------------------------------------------------------------
    # Plasticity
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    def _surprise_and_buffer_bookkeeping(
        self, token_id: torch.Tensor, logits: torch.Tensor
    ) -> None:
        """Per-tick ring-buffer + surprise EMA update (single-token API)."""
        cfg = self.cfg
        self.surprise_prev = self.surprise_ema.detach().clone()
        self.surprise_ema = multi_horizon_surprise(
            self.pred_buf,
            self.pred_cursor,
            cfg.K_buf,
            self.pred_filled,
            token_id,
            self.surprise_ema,
            cfg.alpha_gamma_s,
        )
        write_prediction_buffer(self.pred_buf, self.pred_cursor, logits)
        self.pred_cursor = (self.pred_cursor + 1) % cfg.K_buf
        self.pred_filled = min(self.pred_filled + 1, cfg.K_buf)

    @torch._dynamo.disable
    def _ringbuf_block_bookkeeping(
        self,
        motor_stack: torch.Tensor,       # [B, block_len, D_s]
        tokens_block: torch.Tensor,       # [B, block_len] Long — actual tokens seen
    ) -> None:
        """Block-level ring-buffer + surprise EMA update (training API).

        For each tick in the block, compute logits via factored readout,
        update surprise EMA by comparing past predictions to the actual
        token emitted at that tick, write the new logits into the ring
        buffer. Runs outside the compile region.

        This is cheap: block_len × K_h × B×V elementwise compute.
        """
        cfg = self.cfg
        B, block_len, D_s = motor_stack.shape
        # Apply pred_head once to whole block (cheap residual mlp)
        motor_flat = self.readout.pred_head(
            motor_stack.reshape(B * block_len, D_s)
        ).reshape(B, block_len, D_s)

        with torch.no_grad():
            for t in range(block_len):
                # Compute logits for this tick (for ring buffer only).
                motor_t = motor_flat[:, t]
                # Factored: logits = motor @ W.T + horizon_emb @ W.T
                logits_motor = torch.matmul(
                    motor_t.detach(), self.tied_token_emb.weight.detach().t()
                )
                logits_horizon = torch.matmul(
                    self.readout.horizon_emb.detach(),
                    self.tied_token_emb.weight.detach().t(),
                )
                logits = logits_motor.unsqueeze(1) + logits_horizon  # [B, K_h, V]

                # Surprise vs actual token at t.
                self.surprise_prev = self.surprise_ema.detach().clone()
                self.surprise_ema = multi_horizon_surprise(
                    self.pred_buf,
                    self.pred_cursor,
                    cfg.K_buf,
                    self.pred_filled,
                    tokens_block[:, t],
                    self.surprise_ema,
                    cfg.alpha_gamma_s,
                )
                write_prediction_buffer(self.pred_buf, self.pred_cursor, logits)
                self.pred_cursor = (self.pred_cursor + 1) % cfg.K_buf
                self.pred_filled = min(self.pred_filled + 1, cfg.K_buf)

        self.tick_counter += block_len

    def _plasticity_step(
        self,
        m_out: torch.Tensor,
        incoming: torch.Tensor,
        w_out_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Neuromod forward + Hebbian update on E_bias. Returns η_global [B]."""
        cfg = self.cfg
        with torch.autocast(device_type=self.s.device.type, enabled=False):
            pre, post = compute_activities(m_out, incoming)    # [B, N] each

            B, N, K = self.s.shape[0], cfg.N, cfg.K
            # For the neuromod head features, use outgoing-edge weights and
            # outgoing-neighbour activities as a proxy for "incoming edges at
            # this column." Slightly off-thesis (post-synaptic-local should
            # use in-edges) but cheap — building an in-edge-per-column table
            # would cost extra VRAM with no major quality signal.
            w_out_proxy = w_out_flat.view(B, N, K).float()     # [B, N, K]
            pre_at_nbrs = pre[:, self.out_nbrs].float()        # [B, N, K]

            surprise_delta = (self.surprise_ema - self.surprise_prev).float()

            eta_global, eta, beta = self.neuromod(
                self.surprise_ema.float(),
                surprise_delta,
                self.col_id.float(),
                post.float(),
                w_out_proxy,
                pre_at_nbrs,
            )
            # Average per-batch eta_global into a scalar for the shared E_bias.
            eta_global_scalar = eta_global.mean()
            # Post-synaptic eta and beta: average over batch to get per-column
            # scalars (plastic state is shared across the cohort).
            eta_post = eta.mean(dim=0)                         # [N]
            beta_post = beta.mean(dim=0)                       # [N]

            # --- Hebbian update
            self.E_bias_flat = hebbian_update_incoming(
                E_bias_flat=self.E_bias_flat,
                w_out_flat=w_out_flat.float(),
                edge_src=self.edge_src,
                edge_dst=self.edge_dst,
                pre=pre.float(),
                post=post.float(),
                eta_global=eta_global_scalar,
                eta=eta_post,
                beta=beta_post,
                E_max=cfg.E_bias_max,
            )
        return eta_global
