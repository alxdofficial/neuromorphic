"""biomem — a chunk-parallel gated-delta cortical-column grid memory encoder.

THE BET: memory lives in synaptic STATE (fast edge weights W per column), updated by a *gated
DELTA rule*, NOT in learned weights. Read and write are signal propagation through a small grid.

v3 (chunkwise): the write is a STACK of linear Gated-DeltaNet column-layers, each a CHUNK-PARALLEL
scan over the token sequence (fla.chunk_gated_delta_rule), with a pointwise nonlinearity BETWEEN
layers. This deletes the O(T) per-token Python sweep (≈98% of the old wall-clock) — depth (n_pairs)
stays sequential (2 short steps), the long token axis is parallel. Per column-layer l:
    u^0   = in_proj(LM_final_hidden).view(C,K)            # input to layer 0
    k = q = u^l                                            # key/query = layer input (L2-normed in kernel)
    v     = u^l · V_w[l]                                   # per-column value projection (input-derived)
    β     = sigmoid(u^l·βw[l] + βb[l] + βs[l]·surprise)    # input-derived write rate (the gate)
    α     = sigmoid(decay_proj([h_t; surprise]))           # input-derived per-(layer,column) decay ∈(0,1)
    o, W_l = chunk_gated_delta_rule(q,k,v, g=log α, β)      # W_t = α_t W_{t-1}(I−β_t k k^T)+β_t v k^T
    u^{l+1} = hardtanh(o − θ_l)                            # NONLINEARITY between layers + the compounding
The COMPOUNDING (variant A): layer l's input is layer l−1's memory readout, so deeper layers reason
over what the shallower memory holds — and since layer l's key derives from W_{l-1} (precomputed,
not its own W_l), each layer's scan stays linear/chunkable while the address stays memory-aware
across depth. Keys/values are input-derived (NOT the W-propagated state), which is what makes the
recurrence linear; the nonlinearity that gives expressivity lives on the short depth axis.

Lineage: fast weights (Schmidhuber 1992; Ba 2016), DeltaNet / fast-weight programmers (Schlag 2021),
Gated DeltaNet (Yang & Hatamizadeh 2025), neuromodulated plasticity / Backpropamine (Miconi 2018/19).

LEARNED objects (all small): in_proj, per-(layer,column) value projection V_w, the input-derived
write-rate gate (βw/βb/βs), the input-dependent decay projection, the per-neuron thresholds theta
(random INIT, then learned), the readout MLP + boundary norm, the read seeds + per-layer-refresh
projection. The frozen LM's next-token SURPRISE feeds the gate + decay.

READ (prepend): M learned seeds propagate (read-only, q^T W per layer, NO edge update) through the
written W → per-slot readout → _NormMatch → PREPEND M tokens; the LM's own attention does the
addressing. Refreshed at every decoder layer with the attention-mixed slot hiddens (zero-init gate).
W resets to 0 each example. The grid runs in fp32 (edge precision is load-bearing).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from ...common import _NormMatch
from ...config import ReprConfig


def _hardtanh(x: Tensor) -> Tensor:
    return torch.clamp(x, -1.0, 1.0)


class BioMemEncoder(nn.Module):
    """Chunk-parallel gated-delta grid memory encoder with a prepend read.

    Interface: init_streaming_state / streaming_write (accumulate the passage into the per-layer fast
    edges W via a chunk-parallel scan) / finalize_memory (propagate seeds → M prepend tokens). The read
    is a prepend (is_conditioned_read=False); the per-layer refresh re-reads W in the decoder hook.
    """

    ingest_lm_final_hidden = True         # WRITE ingests the frozen LM's final hidden, not raw embeds
    is_conditioned_read = False           # prepend read (the LM's attention addresses the M slots)

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        C, K, H = cfg.biomem_n_cols, cfg.biomem_k, cfg.biomem_depth_h
        self.C, self.K, self.H = C, K, H
        self.n_pairs = H - 1                                    # fast-edge column-layers (DeltaNet stack)
        self.d_grid = C * K                                     # grid activity dim
        self.d_llama = cfg.d_llama
        self.M = int(getattr(cfg, "biomem_n_slots", 32))       # # read seeds = # prepend tokens
        self.scale = K ** -0.5                                  # DeltaNet readout scale (o = scale·qᵀW)
        self.read_tap_layer = int(cfg.biomem_read_tap_layer)
        self.wants_surprise = bool(getattr(cfg, "biomem_use_surprise", True))
        self.per_layer_refresh = bool(getattr(cfg, "biomem_per_layer_refresh", True))
        self.wants_prepend_refresh = self.per_layer_refresh
        self._read_W = None                                    # [B,n_pairs,C,K,K] stashed each finalize
        for c in ("_decay_last", "_surprise_last", "_beta_last", "_edge_last"):
            self.register_buffer(c, torch.zeros(()), persistent=False)

        # ── write projections (per-(layer,column); input-derived → linear/chunkable scan) ──
        self.in_proj = nn.Linear(cfg.d_llama, self.d_grid)            # LM hidden → layer-0 grid input
        self.V_w = nn.Parameter(torch.randn(self.n_pairs, C, K, K) / math.sqrt(K))   # per-col value proj
        self.beta_w = nn.Parameter(torch.zeros(self.n_pairs, C, K))   # write-rate gate (zero → β starts 0.5)
        self.beta_b = nn.Parameter(torch.zeros(self.n_pairs, C))
        self.beta_s = nn.Parameter(torch.zeros(self.n_pairs, C))      # surprise → write-rate weight
        # input-dependent per-(layer,column) decay α = sigmoid(decay_proj([h_t; surprise])); zero weight +
        # logit(decay_init) bias ⇒ α starts uniform ≈ decay_init, then LEARNS input-dependence.
        self.decay_proj = nn.Linear(cfg.d_llama + 1, self.n_pairs * C)
        nn.init.zeros_(self.decay_proj.weight)
        d0 = float(cfg.biomem_decay_init)
        nn.init.constant_(self.decay_proj.bias, math.log(d0 / (1 - d0)))
        # LEARNED per-neuron thresholds (random init); the between-layer nonlinearity is hardtanh(o − θ).
        self.theta = nn.Parameter(cfg.biomem_theta_scale * torch.randn(self.n_pairs, C, K))

        # ── read (prepend) ──
        self.read_seeds = nn.Parameter(torch.randn(self.M, C, K))     # cold read probes
        ro_h = cfg.biomem_readout_hidden
        self.readout = nn.Sequential(
            nn.Linear(self.d_grid, ro_h), nn.GELU(), nn.Linear(ro_h, cfg.d_llama))
        self.out_norm = _NormMatch(cfg.d_llama)                       # prepend tokens → embedding-norm region
        if self.per_layer_refresh:
            self.read_in = nn.Linear(cfg.d_llama, self.d_grid)        # slot hidden → grid (refresh query)
            self.refresh_gate = nn.Parameter(torch.zeros(()))         # ReZero: refresh is a no-op at step 0

        print(f"[biomem] chunk-parallel gated-delta grid: C={C}×K={K}×H={H} "
              f"({self.n_pairs} column-layers, {C*self.n_pairs*K*K:,} fast edges/example), d_grid={self.d_grid}, "
              f"PREPEND {self.M} seeds"
              + (", per-layer refresh" if self.per_layer_refresh else "")
              + f", surprise={self.wants_surprise}, readout_h={ro_h} (fla chunk scan)")

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def init_streaming_state(self, batch_size: int, device, dtype):
        del dtype
        W = torch.zeros(batch_size, self.n_pairs, self.C, self.K, self.K,
                        device=device, dtype=torch.float32)
        return {"W": W, "n_written": 0}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0,
                        surprise=None, **extra):
        """Stream the passage into the per-layer fast edges via a CHUNK-PARALLEL gated-delta scan
        (one scan per column-layer, nonlinearity between). `surprise` [B,W] is the frozen LM's
        per-token prediction error; feeds the write-rate gate + decay. Carries W across windows."""
        del chunk_offset, extra
        B, Wlen = token_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(B, Wlen, device=token_embeds.device, dtype=torch.bool)
        mask = attention_mask.float()                          # [B,W]
        if surprise is None:
            surprise = token_embeds.new_zeros(B, Wlen)
        with torch.autocast(device_type="cuda", enabled=False):
            tok = token_embeds.float()
            sur = surprise.float()                             # [B,W]
            u = (self.in_proj(tok).view(B, Wlen, self.C, self.K)
                 * mask.view(B, Wlen, 1, 1))                   # [B,W,C,K] layer-0 input (padded→0)
            alpha = torch.sigmoid(self.decay_proj(            # [B,W,n_pairs,C] per-(layer,col) retention
                torch.cat([tok, sur.unsqueeze(-1)], dim=-1))).view(B, Wlen, self.n_pairs, self.C)
            W_in = state["W"].float()
            new_W, beta_means = [], []
            for l in range(self.n_pairs):
                v = torch.einsum("btck,ckv->btcv", u, self.V_w[l])               # per-col value
                beta = torch.sigmoid(
                    torch.einsum("btck,ck->btc", u, self.beta_w[l]) + self.beta_b[l]
                    + self.beta_s[l] * sur.unsqueeze(-1)) * mask.unsqueeze(-1)    # [B,W,C] write rate (pad→0)
                g = torch.log(alpha[:, :, l].clamp_min(1e-6)) * mask.unsqueeze(-1)  # log-decay (pad→0=no decay)
                o, W_l = chunk_gated_delta_rule(
                    q=u.contiguous(), k=u.contiguous(), v=v.contiguous(),
                    g=g.contiguous(), beta=beta.contiguous(),
                    initial_state=W_in[:, l].contiguous(), output_final_state=True,
                    use_qk_l2norm_in_kernel=True, scale=self.scale)
                new_W.append(W_l)
                beta_means.append((beta.sum() / mask.sum().clamp_min(1.0) / self.C))
                u = _hardtanh(o - self.theta[l])              # nonlinearity + compounding → next layer input
            W = torch.stack(new_W, dim=1)                     # [B,n_pairs,C,K,K]
            with torch.no_grad():
                self._decay_last = (alpha.mean(2) * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0) / self.C
                self._surprise_last = (sur * mask).sum() / mask.sum().clamp_min(1.0)
                self._beta_last = torch.stack(beta_means).mean()
                self._edge_last = W.abs().mean()
        state["W"] = W
        state["n_written"] = state.get("n_written", 0) + Wlen
        return state, {}

    def _propagate_read(self, x_grid: Tensor, W: Tensor) -> Tensor:
        """Read core (read-only): propagate a grid-space query [B,M,C,K] through the written edges via
        the SAME readout convention as the write (o = scale·qᵀW per column, L2-normed query), with
        hardtanh(o−θ) between layers → M tokens [B,M,d_llama] at Llama's embedding norm. NO W update."""
        Bn, Mn = x_grid.shape[:2]
        s = x_grid
        for l in range(self.n_pairs):
            q = F.normalize(s, dim=-1, eps=1e-6)              # match the write key's in-kernel L2-norm
            r = self.scale * torch.einsum("bmck,bckv->bmcv", q, W[:, l].float())   # qᵀ W_l per column
            # RESIDUAL read (GCNII-style anti-over-smoothing): re-add the running query so the M seeds stay
            # DISTINCT through the deep (n_pairs) propagation. Without it the readout r (~0.04) sits below the
            # threshold θ (~0.1) → hardtanh(r−θ)≈hardtanh(−θ) for every seed → rank collapse (mem_effrank 2.5
            # → 28 with the residual). The write needs no such fix — its per-token inputs are already diverse.
            s = s + _hardtanh(r - self.theta[l])
        return self.out_norm(self.readout(s.reshape(Bn, Mn, self.d_grid)))

    def _read_prepend(self, W: Tensor) -> Tensor:
        """Initial PREPEND read: the COLD learned seeds (pre-decoder-mixing) propagate through W."""
        B = W.shape[0]
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.read_seeds.unsqueeze(0).expand(B, self.M, self.C, self.K)
            return self._propagate_read(x, W.float())

    def refresh_prepend(self, slot_h: Tensor) -> Tensor:
        """Per-layer REFRESH (read-only): re-read W with the CURRENT slot hiddens (attention-mixed →
        query-aware + cross-slot dedup). slot_h [B,M,d_llama] → read_in → grid → propagate → out_norm,
        gated by the zero-init refresh_gate (no-op at step 0). Returns a delta to ADD to the prepend."""
        W = self._read_W
        assert W is not None, "refresh_prepend called before finalize_memory stashed W"
        B, M, _ = slot_h.shape
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.read_in(slot_h.float()).view(B, M, self.C, self.K)
            delta = self.refresh_gate * self._propagate_read(x, W.float())
        return delta.to(slot_h.dtype)

    def finalize_memory(self, state):
        """Propagate the seeds through the written per-layer edges → M prepend tokens [B,M,d_llama]."""
        W = state["W"]                                         # [B,n_pairs,C,K,K]
        self._read_W = W
        memory = self._read_prepend(W).to(W.dtype)            # [B,M,d_llama]
        with torch.no_grad():
            aux = {
                "biomem_edge_absmean": self._edge_last,
                "biomem_decay": self._decay_last,             # mean per-(layer,col) retention α
                "biomem_beta": self._beta_last,               # mean write rate (the gate)
                "biomem_surprise": self._surprise_last,       # mean LM next-token surprise
                "biomem_mem_effrank": self._participation_ratio(memory.reshape(-1, memory.shape[-1])),
            }
        return memory, aux

    @staticmethod
    def _participation_ratio(x: Tensor) -> Tensor:
        """Effective rank (tr C)²/‖C‖_F² of the emitted memory (PR≈1 ⇒ rank-1 collapse)."""
        x = x.detach().float()
        if x.shape[0] < 2:
            return torch.zeros((), device=x.device)
        xc = x - x.mean(0, keepdim=True); C = xc.t() @ xc
        return torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-12)

    # ── prepend SHUF/OFF interface (shared hooks in model.py) ──
    def has_read_state(self) -> bool:
        return self._read_W is not None

    def roll_read_state(self):                                 # SHUF: refresh reads wrong-example edges
        self._read_W = torch.roll(self._read_W, shifts=1, dims=0)

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
