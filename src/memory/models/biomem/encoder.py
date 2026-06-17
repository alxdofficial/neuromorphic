"""biomem — a gated fast-Hebbian cortical-column grid memory encoder.

THE BET: memory lives in synaptic STATE (scalar fast edge weights W in [-1,1]
updated by a *gated* Hebbian rule per input token), NOT in learned weights. Read
and write are both signal propagation — one token = one feed-forward sweep through
the H grid layers. The only LEARNED objects are small:
  * a SHARED plasticity-regulator MLP (applied per-edge),
  * per-(column, layer-pair) conditioning vectors cond_l (diversity without a
    net-per-location),
  * the readout MLP (grid activity -> a memory token),
  * the M query seeds,
  * the leak lambda.
theta (per-neuron thresholds) are RANDOM-FIXED. The shared decoder LoRA (the read
protocol) is owned by the harness, not here.

Spirit of fast-weights (Ba et al. 2016) and neuromodulated / differentiable
plasticity / Backpropamine (Miconi et al. 2018), in a cortical-column grid.

Substrate
---------
  * neuron state v in [-1,1] (scalar); fast edge weight w in [-1,1] (signed).
  * a column = K-wide x H-deep; consecutive layers fully connected (K^2 edges per
    layer-pair). #cols C * K = d_llama (=576) so the token embedding reshapes into
    the input layer for free (C=9, K=64). Scale via #cols; keep K small.
  * nonlinearity: per-neuron RANDOM-FIXED threshold theta — s = hardtanh(v - theta).

Dynamics (one token = one feed-forward sweep; neurons transient per token, memory
lives in the edges — NO recurrent ticks in v1). At layer-pair l, state s in (B,C,K),
fast edges W in (B,C,K,K):
    inp   = einsum('bcij,bci->bcj', W, s)               # aggregate pre->post
    s_out = hardtanh(inp - theta_l)
    dW    = einsum('bci,bcj->bcij', s, s_out) - leak * W  # Hebbian proposal + leak
    g     = tanh(Regulator([dW, s_pre, s_post, cond_l]))  # plasticity gate in [-1,1]
    W     = clamp(W + g * dW, -1, 1)
    s     = s_out

Write/Read (both = propagation)
-------------------------------
  * WRITE: reshape each token embedding -> (C,K) input layer; sweep forward through
    H layers; edges accumulate via the gated-Hebbian rule across the whole passage.
  * READ: feed M learned query-seed vectors through the WRITTEN edges; read the
    final-layer activity (C,K)=d_llama per seed; project -> M prepend tokens.
  * edges RESET to zero each example (per-passage fast memory).

Engineering: the per-token sweep is gradient-CHECKPOINTED (recompute in backward)
so the long token loop x H sweep doesn't store every token's W in (B,C,H-1,K,K).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from ...config import ReprConfig


def _hardtanh(x: Tensor) -> Tensor:
    return torch.clamp(x, -1.0, 1.0)


class _Regulator(nn.Module):
    """Shared per-edge plasticity regulator: g_ij = tanh(MLP([dW, s_pre, s_post, cond])).

    Applied VECTORIZED over the full (B, C, K, K) edge tensor of one layer-pair (the
    cond vector for that (col,layer-pair) is broadcast over the K x K edges). Output
    in [-1,1]: apply (g~1) / freeze (g~0) / reverse (g~-1). The tanh gate is SOFT so
    the whole rollout is differentiable (no straight-through needed).
    """

    def __init__(self, d_cond: int, hidden: int):
        super().__init__()
        d_in = 3 + d_cond                              # [dW_ij, s_pre_i, s_post_j, cond]
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.act = nn.GELU()
        # Principled init: 1/sqrt(fan_in). The output layer gets a SMALL-scale
        # (not zero) weight + zero bias so the gate g starts NEAR zero but nonzero —
        # the grid begins as an almost-clean slate yet writes a little from step 0,
        # so gradient reaches everything upstream of the edges immediately (a fully
        # zero-init gate is the LoRA-B=0 cold-start: W stays 0 and cond / query_seeds
        # / in_proj get no gradient until fc3 randomly drifts off zero). Small scale
        # keeps the initial write gentle so the regulator LEARNS the gating policy.
        nn.init.normal_(self.fc3.weight, std=0.1 / math.sqrt(hidden))
        nn.init.zeros_(self.fc3.bias)

    def forward(self, dW: Tensor, s_pre: Tensor, s_post: Tensor, cond: Tensor) -> Tensor:
        # dW: [B,C,K,K]  s_pre: [B,C,K]  s_post: [B,C,K]  cond: [C,d_cond]
        B, C, K, _ = dW.shape
        feat = torch.cat([
            dW.unsqueeze(-1),                                   # [B,C,K,K,1]
            s_pre[:, :, :, None, None].expand(B, C, K, K, 1),   # pre on dim i
            s_post[:, :, None, :, None].expand(B, C, K, K, 1),  # post on dim j
            cond[None, :, None, None, :].expand(B, C, K, K, cond.shape[-1]),
        ], dim=-1)                                              # [B,C,K,K,3+d_cond]
        h = self.act(self.fc1(feat))
        h = self.act(self.fc2(h))
        g = torch.tanh(self.fc3(h).squeeze(-1))                 # [B,C,K,K] in [-1,1]
        return g


class BioMemEncoder(nn.Module):
    """Gated fast-Hebbian grid memory encoder (the biomem arm).

    Interface mirrors the slotmem/graph compressors: init_streaming_state /
    streaming_write (accumulates token embeddings + mask — NO backbone forward, the
    grid IS the encoder) / finalize_memory -> (memory [B,M,d_llama], aux). train()
    keeps nothing of the base (it loads the frozen base only to inherit d_llama; the
    grid never forwards it).
    """

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
        self.M = cfg.biomem_n_slots
        self.d_llama = cfg.d_llama
        self.grad_checkpoint = bool(cfg.biomem_grad_checkpoint)

        # ── learned objects (all small) ──────────────────────────────────────
        # input projection: a learned mix of the token embedding into the grid input
        # layer (keeps the substrate from being shackled to the raw embedding basis).
        self.in_proj = nn.Linear(cfg.d_llama, cfg.d_llama)
        # per-(column, layer-pair) conditioning vectors -> regulator diversity.
        d_cond = cfg.biomem_d_cond
        self.cond = nn.Parameter(torch.randn(self.n_pairs, C, d_cond) / math.sqrt(d_cond))
        self.regulator = _Regulator(d_cond, cfg.biomem_reg_hidden)
        # leak lambda in (0,1) via sigmoid of a raw param (learned, principled range).
        leak0 = float(cfg.biomem_leak_init)
        self.leak_raw = nn.Parameter(torch.tensor(math.log(leak0 / (1 - leak0))))
        # M learned query seeds (each a d_llama input vector) + a shared seed encoder.
        self.query_seeds = nn.Parameter(torch.randn(self.M, cfg.d_llama) / math.sqrt(cfg.d_llama))
        se_h = cfg.biomem_seed_hidden
        self.seed_enc = nn.Sequential(
            nn.Linear(cfg.d_llama, se_h), nn.GELU(), nn.Linear(se_h, cfg.d_llama))
        # readout: grid final-layer activity (d_llama) -> a memory token (d_llama).
        ro_h = cfg.biomem_readout_hidden
        self.readout = nn.Sequential(
            nn.Linear(cfg.d_llama, ro_h), nn.GELU(), nn.Linear(ro_h, cfg.d_llama))
        # random-FIXED per-neuron thresholds theta: one per (layer-pair, col, K). Drawn
        # once, registered as a buffer (NOT a parameter). Output-layer threshold too.
        self.register_buffer(
            "theta", cfg.biomem_theta_scale * torch.randn(self.n_pairs, C, K))

        print(f"[biomem] gated fast-Hebbian grid: C={C} cols x K={K} x H={H} "
              f"({self.n_pairs} edge layer-pairs, {C * self.n_pairs * K * K:,} fast "
              f"edges/example), M={self.M} seeds, d_cond={d_cond}, "
              f"reg_h={cfg.biomem_reg_hidden}, readout_h={ro_h}, "
              f"checkpoint={self.grad_checkpoint}")

    @property
    def leak(self) -> Tensor:
        return torch.sigmoid(self.leak_raw)

    def train(self, mode: bool = True):                        # base is never held; no-op guard
        super().train(mode)
        return self

    # ── sweep: one feed-forward pass through the H layers, updating the edges ──
    def _sweep(self, x0: Tensor, W: Tensor, leak: Tensor) -> Tensor:
        """One token (or seed) sweep. x0: [B,C,K] input-layer activity.
        W: [B, n_pairs, C, K, K] fast edges (UPDATED in place via the gated-Hebbian
        rule across the H-1 layer-pairs). Returns the final-layer activity [B,C,K].
        """
        s = _hardtanh(x0)
        new_W = []
        for l in range(self.n_pairs):
            Wl = W[:, l]                                        # [B,C,K,K]
            inp = torch.einsum("bcij,bci->bcj", Wl, s)         # aggregate pre->post
            s_out = _hardtanh(inp - self.theta[l])             # [B,C,K]
            dW = torch.einsum("bci,bcj->bcij", s, s_out) - leak * Wl
            g = self.regulator(dW, s, s_out, self.cond[l])     # [B,C,K,K] in [-1,1]
            Wl = torch.clamp(Wl + g * dW, -1.0, 1.0)
            new_W.append(Wl)
            s = s_out
        W_out = torch.stack(new_W, dim=1)                      # [B,n_pairs,C,K,K]
        return s, W_out

    def init_streaming_state(self, batch_size: int, device, dtype):
        del dtype
        B = batch_size
        W = torch.zeros(B, self.n_pairs, self.C, self.K, self.K, device=device)
        return {"W": W, "device": device, "n_written": 0}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0,
                        **extra):
        """Accumulate the passage into the fast edges. token_embeds: [B,W,d_llama]
        (the frozen Llama input embeddings — the harness supplies these). For each
        token we reshape its (projected) embedding into the C x K input layer and run
        ONE feed-forward sweep, letting the gated-Hebbian rule update the edges. NO
        backbone forward — the grid IS the encoder.
        """
        del chunk_offset, extra
        B, Wlen = token_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(B, Wlen, device=token_embeds.device, dtype=torch.bool)
        x = self.in_proj(token_embeds.float())                 # [B,W,d_llama]
        x = x.view(B, Wlen, self.C, self.K)                    # reshape into the grid input
        mask = attention_mask.float()                          # [B,W]
        leak = self.leak
        W = state["W"]
        for t in range(Wlen):
            x_t = x[:, t]                                       # [B,C,K]
            m_t = mask[:, t]                                    # [B]
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                def _step(_xt, _W):
                    _, W_new = self._sweep(_xt, _W, leak)
                    return W_new
                W_new = torch.utils.checkpoint.checkpoint(
                    _step, x_t, W, use_reentrant=False)
            else:
                _, W_new = self._sweep(x_t, W, leak)
            # per-row gating: a padded token must NOT update that row's edges.
            mt = m_t.view(B, 1, 1, 1, 1)
            W = mt * W_new + (1.0 - mt) * W
        state["W"] = W
        state["n_written"] = state.get("n_written", 0) + Wlen
        return state, {}

    def finalize_memory(self, state):
        """READ: feed the M query seeds through the WRITTEN edges, read the final-layer
        activity per seed, project -> M memory tokens [B, M, d_llama]. Edges are NOT
        updated during the read (frozen synaptic snapshot)."""
        W = state["W"]                                         # [B,n_pairs,C,K,K]
        B = W.shape[0]
        leak = self.leak
        seeds = self.query_seeds + self.seed_enc(self.query_seeds)   # [M,d_llama]
        seeds = seeds.view(self.M, self.C, self.K)
        mem_tokens = []
        for m in range(self.M):
            x0 = seeds[m][None].expand(B, self.C, self.K).contiguous()
            # READ-ONLY sweep: run the forward propagation but DISCARD the edge update
            # (the read does not write — the synaptic snapshot is the memory).
            s_final = self._read_sweep(x0, W)                  # [B,C,K]
            tok = self.readout(s_final.reshape(B, self.d_llama))   # [B,d_llama]
            mem_tokens.append(tok)
        memory = torch.stack(mem_tokens, dim=1)                # [B,M,d_llama]
        with torch.no_grad():
            aux = {
                "biomem_edge_absmean": W.abs().mean().detach(),
                "biomem_edge_satfrac": (W.abs() > 0.99).float().mean().detach(),
                "biomem_leak": leak.detach(),
                "biomem_mem_absmean": memory.abs().mean().detach(),
            }
        return memory, aux

    def _read_sweep(self, x0: Tensor, W: Tensor) -> Tensor:
        """Forward propagation through the FROZEN edges (no plasticity). Returns the
        final-layer activity [B,C,K]."""
        s = _hardtanh(x0)
        for l in range(self.n_pairs):
            inp = torch.einsum("bcij,bci->bcj", W[:, l], s)
            s = _hardtanh(inp - self.theta[l])
        return s

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device,
                                       token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
