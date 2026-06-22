"""biomem — a gated fast-Hebbian cortical-column grid memory encoder.

THE BET: memory lives in synaptic STATE (scalar fast edge weights W in [-1,1]
updated by a *gated DELTA rule* per input token), NOT in learned weights. Read and
write are both signal propagation through a small grid. The only LEARNED objects are
small: a shared plasticity-regulator MLP, per-(column,layer-pair) conditioning
vectors, the leak + base-write-rate scalars, the readout MLP, and the read-side
query/fuse projections. theta (per-neuron thresholds) are RANDOM-FIXED.

Spirit of fast-weights (Ba et al. 2016), DeltaNet / delta-rule fast weights, and
neuromodulated / differentiable plasticity / Backpropamine (Miconi et al. 2018).

Substrate
---------
  * neuron state s in [-1,1] (hardtanh); fast edge weight w in [-1,1] (signed).
  * a column = K-wide x H-deep; consecutive layers fully connected (K^2 edges per
    layer-pair). #cols C * K = d_llama (=576) so the token embedding reshapes into
    the input layer for free (C=9, K=64). Scale via #cols; keep K small.

Dynamics (one token = one feed-forward sweep; memory lives in the edges). At
layer-pair l, state s in (B,C,K), fast edges W in (B,C,K,K):
    inp   = einsum('bcij,bci->bcj', W, s) / sqrt(K)          # fan-in normalized
    s_out = hardtanh(inp - theta_l)                          # target post-activity
    dW    = einsum('bci,bcj->bcij', s, s_out - inp)/sqrt(K) - leak*W   # DELTA + leak
    g     = tanh(Regulator([dW, s_pre, s_post, cond_l]))     # plasticity gate [-1,1]
    W     = clamp(W + (g + eta0) * dW, -1, 1)                # eta0 = base write rate
    s     = s_out

The `/sqrt(K)` (fan-in normalization) keeps `inp` O(1) so hardtanh stays in its
gradient-carrying band; the delta error `(s_out - inp)` (vs raw co-activation) writes
only the residual → far less interference (the DeltaNet lesson, and the rule the
STAGE-0 binding probe validated); `eta0` is a learnable base write-rate so the
write-side (in_proj/cond) receives gradient even when the gate g≈0 (the cold-start
fix). The sweep runs in fp32 (autocast disabled) — edge precision is load-bearing.

Write/Read (both = propagation)
-------------------------------
  * WRITE: stream the passage; each token sweeps forward, edges accumulate via the
    gated delta rule. Padded tokens leave their row's edges bit-identical.
  * READ (query-conditioned): the frozen decoder's hidden state at a tap layer is the
    QUERY — `conditioned_read` propagates it through the WRITTEN edges and returns a
    recall delta fused (zero-init) back into the decoder. NO fixed seeds, NO prepend.
  * edges RESET to zero each example (per-passage fast memory).

Engineering: the per-token write sweep is gradient-CHECKPOINTED (recompute in
backward); leak/eta are passed positionally so their gradient survives checkpointing.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from ...config import ReprConfig


def _hardtanh(x: Tensor) -> Tensor:
    return torch.clamp(x, -1.0, 1.0)


class _Regulator(nn.Module):
    """Shared per-edge plasticity regulator: g_ij = tanh(MLP([dW, s_pre, s_post, cond])).

    Vectorized over the full (B,C,K,K) edge tensor of one layer-pair. Output in [-1,1]:
    apply (g~1) / freeze (g~0) / reverse (g~-1). Soft tanh gate → fully differentiable.
    """

    def __init__(self, d_cond: int, hidden: int):
        super().__init__()
        d_in = 3 + d_cond                              # [dW_ij, s_pre_i, s_post_j, cond]
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.act = nn.GELU()
        # 1/sqrt(fan_in) output init (NOT the old 0.1× near-zero): the gate starts at a
        # usable scale so it learns the gating policy quickly. The cold-start (write
        # happening at all so the write-side gets gradient) is now handled by the
        # learnable base write-rate eta0, not by a tiny gate init.
        nn.init.normal_(self.fc3.weight, std=1.0 / math.sqrt(hidden))
        nn.init.zeros_(self.fc3.bias)

    def forward(self, dW: Tensor, s_pre: Tensor, s_post: Tensor, cond: Tensor) -> Tensor:
        B, C, K, _ = dW.shape
        feat = torch.cat([
            dW.unsqueeze(-1),                                   # [B,C,K,K,1]
            s_pre[:, :, :, None, None].expand(B, C, K, K, 1),   # pre on dim i
            s_post[:, :, None, :, None].expand(B, C, K, K, 1),  # post on dim j
            cond[None, :, None, None, :].expand(B, C, K, K, cond.shape[-1]),
        ], dim=-1)
        h = self.act(self.fc1(feat))
        h = self.act(self.fc2(h))
        return torch.tanh(self.fc3(h).squeeze(-1))             # [B,C,K,K] in [-1,1]


class BioMemEncoder(nn.Module):
    """Gated fast-Hebbian (delta-rule) grid memory encoder with a query-conditioned read.

    Interface: init_streaming_state / streaming_write (accumulate the passage into the
    fast edges — NO backbone forward) / finalize_memory (stash the written edges; return
    an EMPTY prepend). The READ is query-conditioned: the harness calls `conditioned_read`
    from a decoder-layer hook with the live hidden states. See docs / model.py.
    """

    is_conditioned_read = True            # uses the tap-layer read hook, not a prepend
    ingest_lm_final_hidden = True         # WRITE ingests the frozen LM's final hidden, not raw embeds

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        C, K, H = cfg.biomem_n_cols, cfg.biomem_k, cfg.biomem_depth_h
        if C * K != cfg.d_llama:
            raise ValueError(
                f"biomem: n_cols*k = {C}*{K} = {C * K} must equal d_llama={cfg.d_llama} "
                f"(token embedding reshapes into the C x K input layer).")
        self.C, self.K, self.H = C, K, H
        self.n_pairs = H - 1                                    # fast-edge layer-pairs
        self.d_llama = cfg.d_llama
        self.grad_checkpoint = bool(cfg.biomem_grad_checkpoint)
        self.read_tap_layer = int(cfg.biomem_read_tap_layer)
        self._read_W = None                                    # stashed each finalize
        self.register_buffer("_sat_last", torch.zeros(()), persistent=False)

        # ── learned objects (all small) ──────────────────────────────────────
        self.in_proj = nn.Linear(cfg.d_llama, cfg.d_llama)     # token embed -> grid input
        d_cond = cfg.biomem_d_cond
        self.cond = nn.Parameter(torch.randn(self.n_pairs, C, d_cond) / math.sqrt(d_cond))
        self.regulator = _Regulator(d_cond, cfg.biomem_reg_hidden)
        # leak lambda in (0,1) via sigmoid; base write-rate eta0 > 0 via softplus.
        leak0 = float(cfg.biomem_leak_init)
        self.leak_raw = nn.Parameter(torch.tensor(math.log(leak0 / (1 - leak0))))
        eta0 = float(cfg.biomem_base_write_rate_init)
        self.eta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(eta0))))  # softplus⁻¹(eta0)
        # readout: grid activity (d_llama) -> recall vector (d_llama).
        ro_h = cfg.biomem_readout_hidden
        self.readout = nn.Sequential(
            nn.Linear(cfg.d_llama, ro_h), nn.GELU(), nn.Linear(ro_h, cfg.d_llama))
        # query-conditioned read projections: query_proj maps the decoder hidden into the
        # grid input; fuse_proj maps the recall back into the residual stream and is
        # ZERO-INIT (ReZero) so the read is a no-op at step 0 (clean gradient + no decoder
        # destabilization). query_proj has normal init so the read path gets gradient as
        # soon as fuse_proj lifts off zero.
        self.read_query_proj = nn.Linear(cfg.d_llama, cfg.d_llama)
        self.read_fuse_proj = nn.Linear(cfg.d_llama, cfg.d_llama)
        # ReZero-style contribution scale, init SMALL but NONZERO (not zero): the read
        # perturbs the residual gently at step 0 (decoder stays usable) AND — critically —
        # gradient reaches the write-side (in_proj/regulator/eta/leak) from the first step.
        # A zero-init fuse would make the read a no-op and re-create the write-side
        # gradient starvation we fought all along (dead until fuse drifts off zero).
        self.read_gate = nn.Parameter(torch.tensor(0.1))
        # random-FIXED per-neuron thresholds theta [n_pairs, C, K] (fp32 buffer).
        self.register_buffer(
            "theta", cfg.biomem_theta_scale * torch.randn(self.n_pairs, C, K))

        print(f"[biomem] gated fast-Hebbian (delta-rule) grid: C={C} x K={K} x H={H} "
              f"({self.n_pairs} edge layer-pairs, {C * self.n_pairs * K * K:,} fast "
              f"edges/example), query-conditioned read @ tap L{self.read_tap_layer}, "
              f"d_cond={d_cond}, reg_h={cfg.biomem_reg_hidden}, readout_h={ro_h}, "
              f"checkpoint={self.grad_checkpoint}")

    @property
    def leak(self) -> Tensor:
        return torch.sigmoid(self.leak_raw)

    @property
    def eta(self) -> Tensor:
        return F.softplus(self.eta_raw)                        # base write-rate > 0

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    # ── write sweep: one feed-forward pass, updating the edges (gated delta rule) ──
    def _sweep(self, x0: Tensor, W: Tensor, leak: Tensor, eta: Tensor):
        """x0: [B,C,K] input activity; W: [B,n_pairs,C,K,K]. Returns (s_final, W_out)."""
        invK = self.K ** -0.5
        s = _hardtanh(x0)
        new_W = []
        for l in range(self.n_pairs):
            Wl = W[:, l]                                        # [B,C,K,K]
            inp = torch.einsum("bcij,bci->bcj", Wl, s) * invK  # fan-in normalized
            s_out = _hardtanh(inp - self.theta[l])             # target post-activity
            # DELTA rule: write the residual error (target − current), fan-in scaled.
            dW = torch.einsum("bci,bcj->bcij", s, s_out - inp) * invK - leak * Wl
            g = self.regulator(dW, s, s_out, self.cond[l])     # [B,C,K,K] in [-1,1]
            Wl = torch.clamp(Wl + (g + eta) * dW, -1.0, 1.0)   # gate + base write-rate
            new_W.append(Wl)
            s = s_out
        self._sat_last = (s.detach().abs() > 0.99).float().mean()   # state-sat canary
        return s, torch.stack(new_W, dim=1)

    def init_streaming_state(self, batch_size: int, device, dtype):
        del dtype
        W = torch.zeros(batch_size, self.n_pairs, self.C, self.K, self.K,
                        device=device, dtype=torch.float32)
        return {"W": W, "device": device, "n_written": 0}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0,
                        **extra):
        """Stream the passage into the fast edges (one gated-delta sweep per token)."""
        del chunk_offset, extra
        B, Wlen = token_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(B, Wlen, device=token_embeds.device, dtype=torch.bool)
        mask = attention_mask.float()                          # [B,W]
        # edge precision is load-bearing → run the grid in fp32 (autocast disabled).
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.in_proj(token_embeds.float())             # [B,W,d_llama]
            x = x.view(B, Wlen, self.C, self.K)
            x = x * mask.view(B, Wlen, 1, 1)                   # mask padded tokens to 0 (no nan)
            leak, eta = self.leak.float(), self.eta.float()
            W = state["W"].float()
            for t in range(Wlen):
                x_t, m_t = x[:, t], mask[:, t]
                if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                    def _step(_xt, _W, _leak, _eta):
                        return self._sweep(_xt, _W, _leak, _eta)[1]
                    W_new = torch.utils.checkpoint.checkpoint(
                        _step, x_t, W, leak, eta, use_reentrant=False)
                else:
                    _, W_new = self._sweep(x_t, W, leak, eta)
                # per-row delta: a padded row (m_t=0) is left bit-identical.
                W = W + m_t.view(B, 1, 1, 1, 1) * (W_new - W)
        state["W"] = W
        state["n_written"] = state.get("n_written", 0) + Wlen
        return state, {}

    def finalize_memory(self, state):
        """Stash the written edges for the query-conditioned read; return an EMPTY
        prepend (the read happens in the decoder hook, not as prepended tokens)."""
        W = state["W"]                                         # [B,n_pairs,C,K,K]
        self._read_W = W
        memory = W.new_zeros(W.shape[0], 0, self.d_llama)      # no prepend
        with torch.no_grad():
            aux = {
                "biomem_edge_absmean": W.abs().mean(),
                "biomem_edge_satfrac": (W.abs() > 0.99).float().mean(),
                "biomem_state_satfrac": self._sat_last,
                "biomem_leak": self.leak,
                "biomem_eta": self.eta,
            }
        return memory, aux

    def conditioned_read(self, h: Tensor) -> Tensor:
        """Query-conditioned READ. h: [B,T,d_llama] decoder hidden at the tap layer.
        Propagate each position's hidden through the FROZEN written edges (W shared per
        example across its T positions) and return a recall delta [B,T,d_llama] to fuse
        back into the residual stream (zero-init → 0 at step 0)."""
        W = self._read_W
        assert W is not None, "conditioned_read called before finalize_memory stashed W"
        B, T, _ = h.shape
        invK = self.K ** -0.5
        with torch.autocast(device_type="cuda", enabled=False):
            x0 = self.read_query_proj(h.float())               # [B,T,d_llama]
            s = _hardtanh(x0.view(B, T, self.C, self.K))       # [B,T,C,K]
            for l in range(self.n_pairs):
                inp = torch.einsum("bcij,btci->btcj", W[:, l], s) * invK
                s = _hardtanh(inp - self.theta[l])             # theta[l] [C,K] broadcasts
            r = self.readout(s.reshape(B, T, self.d_llama))    # [B,T,d_llama]
            delta = self.read_gate * self.read_fuse_proj(r)    # small-init gate → gentle, nonzero
        return delta.to(h.dtype)

    # ── conditioned-read interface (shared hook in model.py) ──
    def has_read_state(self) -> bool:
        return self._read_W is not None

    def roll_read_state(self):                                 # SHUF control: wrong-example edges
        self._read_W = torch.roll(self._read_W, shifts=1, dims=0)

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        """Standalone path (write + finalize). The real read is query-conditioned via the
        decoder hook; this returns the empty prepend + aux (and stashes W)."""
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device,
                                       token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
