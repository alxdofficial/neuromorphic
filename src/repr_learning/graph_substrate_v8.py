"""graph_substrate_v8.py — columnar co-activation hierarchy — graph_v8 CORRECTED (not a new version).

The 2026-06-09 design session correcting graph_v8's two SHUF=REAL root causes
(pooled convex-average proposal + key-less read). Read this header as the contract.

THE OBJECT
  A trainable memory side-car to a FROZEN LLM. It streams contextualized token
  hidden states through a COLUMN hierarchy of vocabularies and is read back by a
  dedicated K/V cross-attention reader (keys = node keys, values = node values).

THE COLUMNS
  Every layer has the SAME N nodes. Node j at layer L+1 is node j's "higher self":
  upward writes are POSITIONAL (no write-time matching of any kind — the address
  is the column index). Ascending the hierarchy = progressively re-expressing
  concept j more abstractly for THIS passage.

  L0           slow atom vocabulary (atom_keys, atom_values). Never written.
  L1..L_D      per-sequence (key, value) STATE — BOTH delta-written — initialized
               from each layer's own slow learnable base (its learned prior
               vocabulary at that abstraction level).

PER TOKEN, EVERY LAYER (no slow clock; timescales come from learnable decays):
  1. ROUTE the token against the layer's CURRENT keys (refined state for L>=1):
         r_L[t] = softmax( cos(proj_in(h_t), keys_L) / temp_L )
  2. CO-ACTIVATION (cross-token; reads the previous recency trace, then updates):
         coact_L    <- gamma_L * coact_L + r_L[t] (x) windowed_old
         windowed_L <- gamma_L * windowed_old + r_L[t]
  3. FUSION PROPOSAL per node j = SELF + PARTNERS (column j of coact = j's view):
         c_hat[:,j]   = coact_L[:,j] / colsum_j            (column-normalized)
         prop_key_j   = alpha * key_j   + (1-alpha) * sum_i c_hat[i,j] * key_i
         prop_value_j = alpha * bind_j  + (1-alpha) * sum_i c_hat[i,j] * bind_i
     where bind_i = unit(key_i (*) value_i) (HRR self-binds) and alpha =
     sigmoid(self_mix_logit[tgt]) is a LEARNABLE bounded mix (init 0.5). The
     explicit SELF term (user decision 2026-06-10) anchors node j's higher self
     to its own identity + key-tagged content; partners' SELF-binds (not
     owner-binds) keep each constituent individually unbindable — the fused
     value is new vocabulary whose parts remain addressable (Plate). NO
     per-token normalize on proposals: a convex mix of unit-norm items is
     <= 1 by construction — and dropping the normalize keeps the write LINEAR
     in scan quantities (the chunkwise parallel form depends on this; the
     self term is chunk-constant so it adds one broadcast, not a scan).
  4. DELTA WRITE to position j one layer up (keys AND values), magnitude =
     routing x surprise x novelty:
         gate_j = sigmoid(write_strength) * surprise_t * novelty_j * r_L[t,j]
         K_{L+1}[j] <- delta * K_{L+1}[j] + gate_j * (prop_key_j   - K_{L+1}[j])
         V_{L+1}[j] <- delta * V_{L+1}[j] + gate_j * (prop_value_j - V_{L+1}[j])
     novelty is DIMENSION-FREE: sqrt((||eK||^2+||eV||^2)/2) over unit-scale vectors.

THE CHUNKWISE PARALLEL FORM (the training path; tokens batched per chunk):
  All recurrences above are linear-with-decay, so a chunk of C tokens collapses to
  a few batched matmuls via an intra-chunk decay matrix D[t,s] = gamma^(p_t - p_s)
  (p = per-row cumulative REAL-token count, so right-pad rows are EXACT no-ops —
  pads neither write nor decay). Chunk-frozen approximations (documented, the
  standard chunked-linear-attention semantics):
    - routing/fusion at L>=1 read the layer's (keys, values) as of chunk start;
    - novelty and the delta-error reference use chunk-start target state.
  The per-token REFERENCE path implements the SAME chunk-frozen semantics, so
  reference == chunkwise EXACTLY (up to fp tolerance) — tested in the smoke.

THE READ   (NOT in this module — see graph_read.GraphV8SymReader)
  K = final layer's keys (passage-refined!), V = final layer's values, cross-
  attended by Llama hiddens at a few decoder layers, gated-added to the residual.

CONVENTIONS (hard rules)
  - SOFT everywhere. No argmax, no top-k.
  - All decays / temperatures / strengths LEARNABLE but BOUNDED (sigmoid / clamped
    exp). Decay inits form a TIMESCALE LADDER across layers (phrase/sentence/
    passage e-folding windows) — principled init points, training corrects them.
  - All dims multiples of 32 (GPU-friendly; enforced in __init__).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── HRR bind: circular convolution (Plate 1995), fp32 FFT ─────────────────────
def hrr_bind(key: Tensor, value: Tensor) -> Tensor:
    """Bind key ⊛ value by circular convolution, with the KEY projected to UNIT
    SPECTRAL MODULUS (Plate's unitary vectors; Danihelka et al. 2016 bound();
    Ganesan/Alam NeurIPS'21 projection — found NECESSARY for learned vectors).
    A non-unitary learned key acts as a key-shaped bandpass on the value (audit:
    per-key spectral condition ~135 median, chained binds collapse bandwidth
    0.66→0.19 across the hierarchy); unitary keys make the bind an exact isometry
    so ALL value information survives every level. Values stay unprojected (they
    carry content). fp32 (rfft is lossy in bf16); caller casts back."""
    dim = key.shape[-1]
    key_spectrum = torch.fft.rfft(key.float(), n=dim, dim=-1)
    key_spectrum = key_spectrum / key_spectrum.abs().clamp_min(1e-6)   # phase-only
    value_spectrum = torch.fft.rfft(value.float(), n=dim, dim=-1)
    return torch.fft.irfft(key_spectrum * value_spectrum, n=dim, dim=-1)


def _unit(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, dim=dim, eps=1e-6)


def _inverse_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1.0 - p))


@dataclass
class GraphV8Config:
    # dimensions — ALL multiples of 32 (enforced)
    d_model: int = 2048          # frozen-LLM hidden size (input dim)
    d_mem: int = 3072            # key == value dim (HRR bind requires equality)
    n_nodes: int = 1024          # nodes per layer — SAME at every layer (columns)
    n_layers: int = 3            # persistent layers above L0 (L1..L3 → 4 incl. L0)

    # the parallel batch: tokens per chunkwise step (also the checkpoint unit).
    # Within a chunk, L>=1 routing/fusion read chunk-START state (staleness <= chunk).
    chunk: int = 256

    # learnable-but-bounded init points
    # Routing logits are cos·√d_mem / temp (standard attention scaling): cosine of
    # unit vectors has std ~1/√d, so WITHOUT the √d factor the softmax over N nodes
    # is ~uniform at ANY clampable temp (the 2026-06-10 audit measured 99.8%-of-
    # uniform entropy at the old 0.1 "sharp init" — sharpness was unreachable).
    # With √d the logits start O(1) and temp=1.0 is the neutral operating point.
    route_temp_init: float = 1.0
    # co-activation decay TIMESCALE LADDER per source layer (L0,L1,L2). These
    # logits are gradient-starved (audit: ~1e-9), so the INITS are de facto final
    # values and must be defensible against the DATA's length scales: an emat_bio
    # "key = value" line is ~40 tokens with the key at the START, so the L0 window
    # must span key→facts (~half-line e-fold balances own-line coverage against
    # neighbor-line bleed). e-folds ≈ 20 / 50 / 200 tokens (half-line / line-group
    # / passage-section). The old (0.90,...) gave a 10-token e-fold — the key's
    # trace decayed to 0.03 before its own line's last fact arrived.
    coact_decay_inits: tuple = (0.95, 0.98, 0.995)
    # written-state forgetting: decay is TOWARD THE BASE (reversion-to-prior, not
    # erasure — audit: the old decay-to-zero drained 47% of the prior per window).
    # Init from window retention: delta = exp(ln(0.9)/640) ≈ 0.99984 → ~0.9 of a
    # write surviving a full 640-token window. Recalibrate if windows change.
    value_decay_init: float = 0.99984
    write_strength_init: float = 0.10   # max per-write rate, in (0,1)

    def __post_init__(self):
        for name in ("d_model", "d_mem", "n_nodes"):
            v = getattr(self, name)
            if v % 32 != 0:
                raise ValueError(f"GraphV8Config.{name}={v} must be a multiple of 32")
        if len(self.coact_decay_inits) < self.n_layers:
            raise ValueError("need one coact_decay init per source layer (L0..L_{n_layers-1})")


class GraphV8Substrate(nn.Module):
    """Columnar co-activation hierarchy (see module header)."""

    def __init__(self, config: GraphV8Config):
        super().__init__()
        self.config = config
        d, N, L = config.d_mem, config.n_nodes, config.n_layers
        self.depth = L

        # ── L0: slow atom vocabulary (never written) ──────────────────────────
        self.atom_keys = nn.Parameter(torch.randn(N, d) / math.sqrt(d))
        self.atom_values = nn.Parameter(torch.randn(N, d) / math.sqrt(d))
        # ── L1..L_D: slow learnable BASE for the written (key,value) state — each
        # layer's learned prior vocabulary at its abstraction level. Per-sequence
        # state is initialized from these in init_state.
        self.base_keys = nn.Parameter(torch.randn(L, N, d) / math.sqrt(d))
        self.base_values = nn.Parameter(torch.randn(L, N, d) / math.sqrt(d))

        # PER-LAYER input projections (layer-matched design, 2026-06-10). One
        # router per ROUTABLE key bank: atoms (0), K1 (1), K2 (2), K3 (3).
        # Routers 0..L-1 are the WRITE's source routers (gate writes into the
        # layer above) and are SHARED with the read; router L (over K_L) is
        # read-only — same-layer K/V pairing (user decision: read layer ℓ
        # addresses with K_ℓ and fetches V_ℓ; L never writes upward).
        self.proj_in = nn.ModuleList([nn.Linear(config.d_model, d) for _ in range(L + 1)])

        # ── learnable-but-bounded dynamics ────────────────────────────────────
        self.log_route_temp = nn.Parameter(
            torch.full((L + 1,), math.log(config.route_temp_init)))        # routers 0..L
        self.coact_decay_logit = nn.Parameter(torch.tensor(
            [_inverse_sigmoid(g) for g in config.coact_decay_inits[:L]]))  # source layers
        self.value_decay_logit = nn.Parameter(
            torch.full((L,), _inverse_sigmoid(config.value_decay_init)))   # target layers 1..L
        self.write_strength_logit = nn.Parameter(
            torch.full((L,), _inverse_sigmoid(config.write_strength_init)))
        # self-vs-partners proposal mix per target layer: alpha = sigmoid(logit),
        # init 0.5 (logit 0) — the node's OWN key / self-bind is half the proposal
        # at init, learnable thereafter (user decision: self must be part of it).
        self.self_mix_logit = nn.Parameter(torch.zeros(L))

    # ── bounded accessors ──────────────────────────────────────────────────────
    def _route_temp(self, src: int) -> Tensor:
        return self.log_route_temp[src].clamp(math.log(0.05), math.log(20.0)).exp()

    def _coact_decay(self, src: int) -> Tensor:
        return torch.sigmoid(self.coact_decay_logit[src])

    def _value_decay(self, tgt: int) -> Tensor:           # tgt in 1..L
        return torch.sigmoid(self.value_decay_logit[tgt - 1])

    def _write_strength(self, tgt: int) -> Tensor:
        return torch.sigmoid(self.write_strength_logit[tgt - 1])

    def _self_mix(self, tgt: int) -> Tensor:
        return torch.sigmoid(self.self_mix_logit[tgt - 1])

    # ── per-sequence state ─────────────────────────────────────────────────────
    def init_state(self, batch_size: int, device, dtype=torch.float32) -> dict:
        cfg = self.config
        N, d, L = cfg.n_nodes, cfg.d_mem, cfg.n_layers
        z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)
        # keys/values indexed 1..L (index 0 placeholder — L0 is the slow atoms)
        keys = [None] + [self.base_keys[l].to(device, torch.float32)
                         .unsqueeze(0).expand(batch_size, -1, -1).contiguous() for l in range(L)]
        values = [None] + [self.base_values[l].to(device, torch.float32)
                           .unsqueeze(0).expand(batch_size, -1, -1).contiguous() for l in range(L)]
        return {
            "keys": keys, "values": values,
            "coact": [z(batch_size, N, N) for _ in range(L)],     # source layers 0..L-1
            "windowed": [z(batch_size, N) for _ in range(L)],
            "colsum": [z(batch_size, N) for _ in range(L)],       # running column sums of coact
            "step": 0, "telemetry": {},
        }

    # ── checkpoint-safe flat <-> dict state ────────────────────────────────────
    def _flatten_state(self, state: dict) -> tuple:
        L = self.depth
        return (*[state["keys"][l] for l in range(1, L + 1)],
                *[state["values"][l] for l in range(1, L + 1)],
                *state["coact"], *state["windowed"], *state["colsum"])

    def _unflatten_state(self, flat: tuple) -> dict:
        L, i = self.depth, 0
        keys = [None] + list(flat[i:i + L]); i += L
        values = [None] + list(flat[i:i + L]); i += L
        coact = list(flat[i:i + L]); i += L
        windowed = list(flat[i:i + L]); i += L
        colsum = list(flat[i:i + L]); i += L
        return {"keys": keys, "values": values, "coact": coact, "windowed": windowed,
                "colsum": colsum, "step": 0, "telemetry": {}}

    # ── shared per-chunk precompute for one source layer ───────────────────────
    def route(self, src: int, hiddens: Tensor, src_keys: Tensor) -> Tensor:
        """THE addressing function for source layer `src` — used by the WRITE
        (gates writes into src+1) AND, identically, by the READ of layer src+1
        (symmetric addressing). hiddens [B,T,d_model] at the matched Llama depth;
        src_keys [N,d] (atoms) or [B,N,d] (refined state). Returns [B,T,N]."""
        # autocast-OFF: the write calls this under the trainer's bf16 autocast, the
        # read calls it inside an fp32 block — without the guard the SAME shared
        # addressing function ran at two precisions (measured 5e-3 logit asymmetry).
        with torch.autocast(device_type=hiddens.device.type, enabled=False):
            kq = _unit(self.proj_in[src](hiddens.float()))                      # [B,T,d]
            ku = _unit(src_keys.float())
            if src_keys.dim() == 2:
                logits = torch.einsum("bcd,nd->bcn", kq, ku)
            else:
                logits = torch.einsum("bcd,bnd->bcn", kq, ku)
            # √d scaling (standard attention): cos std ~1/√d → logits start O(1),
            # so the clamped learnable temp has real authority in BOTH directions.
            logits = logits * math.sqrt(self.config.d_mem)
            return torch.softmax(logits / self._route_temp(src), dim=-1)

    def source_keys(self, src: int, state: dict) -> Tensor:
        return self.atom_keys.float() if src == 0 else state["keys"][src]

    def _layer_inputs(self, src: int, state: dict, h_src: Tensor, mask: Tensor):
        """Chunk-frozen source quantities for source layer `src` (writes into src+1).

        h_src [B,C,d_model] fp32 — the matched Llama layer's hiddens for THIS source.
        Returns (r, src_keys, bound) — r is mask-zeroed routing [B,C,N] fp32;
        src_keys/bound are the chunk-frozen key bank and unit-norm self-binds.
        """
        src_keys = self.source_keys(src, state)
        if src == 0:
            bound = _unit(hrr_bind(self.atom_keys, self.atom_values))           # [N,d]
        else:
            bound = _unit(hrr_bind(src_keys, state["values"][src]))             # [B,N,d]
        r = self.route(src, h_src, src_keys) * mask.unsqueeze(-1)               # pads: r = 0
        return r, src_keys, bound

    # ── the chunkwise parallel step (the training path) ────────────────────────
    def _chunk_forward(self, state: dict, h_stack: Tensor, surprise: Tensor,
                       mask: Tensor) -> dict:
        """One chunk of C tokens, all layers, batched matmul form.

        h_stack [B,C,S,d_model] fp32 — raw matched-Llama-layer hiddens, one slice
        per SOURCE layer (layer-matched design). surprise [B,C] fp32 in (0,1),
        mask [B,C] fp32 {0,1}. Mutates and returns `state`. EXACTLY equivalent to
        _chunk_forward_reference (same chunk-frozen semantics) — smoke-tested.
        """
        B, C = h_stack.shape[:2]
        eps = 1e-6
        # per-row real-token cumulative position; pads advance nothing → pads
        # neither decay nor contribute anywhere (EXACT per-row no-op).
        p = mask.cumsum(dim=1)                                                  # [B,C]
        p_last = p[:, -1]                                                       # [B]

        new_keys, new_values = list(state["keys"]), list(state["values"])
        new_coact, new_windowed, new_colsum = (list(state["coact"]),
                                               list(state["windowed"]), list(state["colsum"]))

        for src in range(self.depth):                       # source layer src → target src+1
            tgt = src + 1
            gamma = self._coact_decay(src)
            delta = self._value_decay(tgt)
            ws = self._write_strength(tgt)

            r, src_keys, bound = self._layer_inputs(src, state, h_stack[:, :, src], mask)

            # intra-chunk decay matrix D[t,s] = γ^(p_t-p_s) for s<=t, else 0.
            # clamp_min BEFORE the pow: γ^(negative) on the masked s>t side can
            # overflow to inf, and inf in the unselected torch.where branch NaNs
            # the pow backward (0·∂inf). Clamped entries are masked out anyway.
            dp = (p.unsqueeze(-1) - p.unsqueeze(-2)).clamp_min(0.0)             # [B,C,C] p_t - p_s
            tril_incl = torch.tril(torch.ones(C, C, device=r.device, dtype=torch.bool))
            D_incl = torch.where(tril_incl, gamma ** dp, torch.zeros((), device=r.device))  # s<=t
            e = gamma ** p                                                      # [B,C] decay from chunk start

            # windowed trace at each token (post-update), then shift → old_windowed
            w0, c0, s0 = state["windowed"][src], state["coact"][src], state["colsum"][src]
            w_all = e.unsqueeze(-1) * w0.unsqueeze(1) + torch.einsum("bts,bsn->btn", D_incl, r)
            old_w = torch.cat([w0.unsqueeze(1), w_all[:, :-1]], dim=1)           # [B,C,N] = w_{t-1}
            # colsum at each token (post-update): colsum_t = e_t*s0 + Σ_{s<=t} γ^{p_t-p_s} σ_s old_w_s
            colsum_all = (e.unsqueeze(-1) * s0.unsqueeze(1)
                          + torch.einsum("bts,bsn->btn", D_incl, mask.unsqueeze(-1) * old_w))

            # fusion-proposal building blocks (chunk-frozen source bank)
            if src == 0:
                m_k = torch.einsum("bcn,nd->bcd", r, src_keys)                  # r_s @ K0
                m_v = torch.einsum("bcn,nd->bcd", r, bound)
                A0_k = torch.einsum("bnm,nd->bmd", c0, src_keys)                # coact0ᵀ @ K0
                A0_v = torch.einsum("bnm,nd->bmd", c0, bound)
            else:
                m_k = torch.einsum("bcn,bnd->bcd", r, src_keys)
                m_v = torch.einsum("bcn,bnd->bcd", r, bound)
                A0_k = torch.einsum("bnm,bnd->bmd", c0, src_keys)
                A0_v = torch.einsum("bnm,bnd->bmd", c0, bound)

            # chunk-frozen novelty: distance of the chunk-START proposal from the
            # chunk-START target state, in the natural unit-vector scale (dim-free).
            # Proposal = alpha*SELF + (1-alpha)*partners (self term chunk-constant).
            alpha = self._self_mix(tgt)
            self_k = src_keys                                                    # [N,d] or [B,N,d]
            self_v = bound
            tK0, tV0 = state["keys"][tgt], state["values"][tgt]                 # [B,N,d]
            prop0_k = alpha * self_k + (1 - alpha) * (A0_k / (s0.unsqueeze(-1) + eps))
            prop0_v = alpha * self_v + (1 - alpha) * (A0_v / (s0.unsqueeze(-1) + eps))
            novelty = torch.sqrt(((prop0_k - tK0).pow(2).sum(-1)
                                  + (prop0_v - tV0).pow(2).sum(-1)) / 2.0)      # [B,N]
            novelty = novelty / (1.0 + novelty)            # BOUNDED (0,1): the only unbounded
            #                                                gate factor fed divergence (audit).

            # ── ONLINE-MIXING delta write toward BASE (audit fixes, 2026-06-10) ──
            # Per-token semantics: T ← (1-g)·[base + δ(T-base)] + g·prop — i.e. decay
            # REVERTS TO THE PRIOR (not zero) and the mix is the true online delta, so
            # the coefficient on old state is ∏(1-g_s)·δ̂ > 0: overshoot is impossible
            # by construction (vs the old frozen-reference sum, stable only while
            # Σg < δ^P). Proposals stay chunk-frozen; the recurrence stays LINEAR in
            # the state deviation D = T - base, so the chunkwise collapse survives:
            #   D_end = exp(Σ_s L_s)·D0 + Σ_s exp(Σ_{u>s} L_u)·g_s·(prop_s - base),
            #   L_s = log(1-g_s) + mask_s·log δ   (pads: g=0, no δ → exact no-op).
            g = (ws * surprise.unsqueeze(-1) * r * novelty.unsqueeze(1))        # [B,C,N] in [0,1)
            logdec = torch.log1p(-g) + (mask * torch.log(delta)).unsqueeze(-1)  # [B,C,N]
            suffix = logdec.flip(1).cumsum(1).flip(1) - logdec                  # Σ_{u>s} L_u
            w = suffix.exp() * g                                                # online weights [B,C,N]
            survive = logdec.sum(1).exp()                                       # ∏ δ̂(1-g)  [B,N]
            w_sum = w.sum(dim=1)                                                # [B,N]
            g_hat = w / (colsum_all + eps)                                      # [B,C,N]

            # Σ_s w_s ⊙ prop_s — same two-matmul collapse, with w replacing f·g:
            coeff_A0 = torch.einsum("bc,bcn->bn", e, g_hat)                     # [B,N]
            q = torch.einsum("bst,bsn->btn", D_incl, g_hat)                     # (Dᵀ @ ŵ)  [B,C,N]
            qw = q * old_w                                                       # [B,C,N]
            write_k = ((1 - alpha) * (coeff_A0.unsqueeze(-1) * A0_k
                                      + torch.einsum("bcn,bcd->bnd", qw, m_k))
                       + alpha * w_sum.unsqueeze(-1) * self_k)
            write_v = ((1 - alpha) * (coeff_A0.unsqueeze(-1) * A0_v
                                      + torch.einsum("bcn,bcd->bnd", qw, m_v))
                       + alpha * w_sum.unsqueeze(-1) * self_v)

            base_k = self.base_keys[tgt - 1].float()                            # [N,d] the prior
            base_v = self.base_values[tgt - 1].float()
            new_keys[tgt] = (base_k + survive.unsqueeze(-1) * (tK0 - base_k)
                             + write_k - w_sum.unsqueeze(-1) * base_k)
            new_values[tgt] = (base_v + survive.unsqueeze(-1) * (tV0 - base_v)
                               + write_v - w_sum.unsqueeze(-1) * base_v)

            # chunk-end carries for coact / windowed / colsum
            e_end = (gamma ** p_last).view(B, 1)
            fg = (gamma ** (p_last.view(B, 1) - p))                             # [B,C]
            new_coact[src] = (e_end.unsqueeze(-1) * c0
                              + torch.einsum("bcn,bcm->bnm", fg.unsqueeze(-1) * r, old_w))
            new_windowed[src] = e_end * w0 + (fg.unsqueeze(-1) * r).sum(dim=1)
            new_colsum[src] = e_end * s0 + (fg * mask).unsqueeze(-1).mul(old_w).sum(dim=1)

        state["keys"], state["values"] = new_keys, new_values
        state["coact"], state["windowed"], state["colsum"] = new_coact, new_windowed, new_colsum
        return state

    # ── per-token REFERENCE path (same chunk-frozen semantics; smoke only) ─────
    def _chunk_forward_reference(self, state: dict, h_stack: Tensor, surprise: Tensor,
                                 mask: Tensor) -> dict:
        B, C = h_stack.shape[:2]
        eps = 1e-6
        # PARALLEL-across-layers semantics (matches _chunk_forward): every source
        # layer reads the CHUNK-START keys/values, not the ones its lower layer
        # just wrote this chunk. Snapshot before any writes.
        frozen = {"keys": list(state["keys"]), "values": list(state["values"]),
                  "coact": state["coact"], "windowed": state["windowed"],
                  "colsum": state["colsum"]}
        for src in range(self.depth):
            tgt = src + 1
            gamma = self._coact_decay(src)
            delta = self._value_decay(tgt)
            ws = self._write_strength(tgt)
            r_all, src_keys, bound = self._layer_inputs(src, frozen, h_stack[:, :, src], mask)
            # chunk-frozen: novelty + error reference from chunk-start target state
            tK0, tV0 = frozen["keys"][tgt], frozen["values"][tgt]
            c0, s0 = state["coact"][src], state["colsum"][src]
            if src == 0:
                A0_k = torch.einsum("bnm,nd->bmd", c0, src_keys.float())
                A0_v = torch.einsum("bnm,nd->bmd", c0, bound)
            else:
                A0_k = torch.einsum("bnm,bnd->bmd", c0, src_keys)
                A0_v = torch.einsum("bnm,bnd->bmd", c0, bound)
            alpha = self._self_mix(tgt)
            self_k = src_keys.float() if src == 0 else src_keys
            self_v = bound
            prop0_k = alpha * self_k + (1 - alpha) * (A0_k / (s0.unsqueeze(-1) + eps))
            prop0_v = alpha * self_v + (1 - alpha) * (A0_v / (s0.unsqueeze(-1) + eps))
            novelty = torch.sqrt(((prop0_k - tK0).pow(2).sum(-1)
                                  + (prop0_v - tV0).pow(2).sum(-1)) / 2.0)
            novelty = novelty / (1.0 + novelty)                                  # bounded (0,1)

            base_k = self.base_keys[tgt - 1].float()
            base_v = self.base_values[tgt - 1].float()
            coact, windowed, colsum = state["coact"][src], state["windowed"][src], state["colsum"][src]
            tK, tV = state["keys"][tgt], state["values"][tgt]
            for i in range(C):
                r = r_all[:, i]                                                 # [B,N]
                m_i, srp_i = mask[:, i], surprise[:, i]
                keep2 = (m_i > 0).view(B, 1)
                keep3 = (m_i > 0).view(B, 1, 1)
                old_w = windowed
                coact_new = gamma * coact + torch.einsum("bn,bm->bnm", r, old_w)
                windowed_new = gamma * old_w + r
                colsum_new = gamma * colsum + old_w                             # σ=1 for real rows
                coact = torch.where(keep3, coact_new, coact)
                windowed = torch.where(keep2, windowed_new, windowed)
                colsum = torch.where(keep2, colsum_new, colsum)
                # fusion proposal from CURRENT coact, chunk-frozen source bank
                if src == 0:
                    num_k = torch.einsum("bnm,nd->bmd", coact, src_keys.float())
                    num_v = torch.einsum("bnm,nd->bmd", coact, bound)
                else:
                    num_k = torch.einsum("bnm,bnd->bmd", coact, src_keys)
                    num_v = torch.einsum("bnm,bnd->bmd", coact, bound)
                prop_k = alpha * self_k + (1 - alpha) * (num_k / (colsum.unsqueeze(-1) + eps))
                prop_v = alpha * self_v + (1 - alpha) * (num_v / (colsum.unsqueeze(-1) + eps))
                gate = (ws * srp_i.view(B, 1) * novelty * r).unsqueeze(-1)      # [B,N,1] in [0,1)
                # ONLINE delta toward BASE: T ← (1-g)·[base + δ(T-base)] + g·prop
                tK_new = (1 - gate) * (base_k + delta * (tK - base_k)) + gate * prop_k
                tV_new = (1 - gate) * (base_v + delta * (tV - base_v)) + gate * prop_v
                tK = torch.where(keep3, tK_new, tK)
                tV = torch.where(keep3, tV_new, tV)
            state["coact"][src], state["windowed"][src], state["colsum"][src] = coact, windowed, colsum
            state["keys"][tgt], state["values"][tgt] = tK, tV
        return state

    # ── full streaming pass (gradient-checkpointed chunks; full BPTT) ──────────
    def forward(self, token_hiddens: Tensor, token_surprises: Tensor,
                token_mask: Optional[Tensor] = None, state: Optional[dict] = None,
                reference: bool = False) -> dict:
        """token_hiddens [B,T,S,d_model] (S = n_layers, one matched Llama layer's
        hiddens per SOURCE layer) or [B,T,d_model] (broadcast to all sources);
        token_surprises [B,T] in (0,1), token_mask [B,T].
        Full BPTT — no detach between chunks; the tape is bounded by recompute."""
        if token_hiddens.dim() == 3:
            token_hiddens = token_hiddens.unsqueeze(2).expand(-1, -1, self.depth, -1)
        B, T = token_hiddens.shape[:2]
        if state is None:
            state = self.init_state(B, token_hiddens.device)
        if token_mask is None:
            token_mask = torch.ones(B, T, device=token_hiddens.device, dtype=torch.float32)
        token_mask = token_mask.float()
        # pad-skip: stream only to the last position real in ANY row
        real_any = token_mask.bool().any(dim=0)
        nz = torch.nonzero(real_any, as_tuple=False)
        seq_eff = int(nz[-1].item()) + 1 if nz.numel() > 0 else 0

        chunk = max(1, int(self.config.chunk))
        use_ckpt = self.training and torch.is_grad_enabled()
        step_fn = self._chunk_forward_reference if reference else self._chunk_forward

        start = 0
        while start < seq_eff:
            stop = min(start + chunk, seq_eff)
            h = token_hiddens[:, start:stop].float()
            s = token_surprises[:, start:stop].float()
            m = token_mask[:, start:stop]

            def run_chunk(*flat, _h=h, _s=s, _m=m):
                st = self._unflatten_state(flat)
                st = step_fn(st, _h, _s, _m)
                return self._flatten_state(st)

            flat_in = self._flatten_state(state)
            flat_out = (torch.utils.checkpoint.checkpoint(run_chunk, *flat_in, use_reentrant=False)
                        if use_ckpt else run_chunk(*flat_in))
            state = self._unflatten_state(flat_out)
            start = stop

        state["step"] = seq_eff
        with torch.no_grad():
            tele = {}
            for l in range(1, self.depth + 1):
                tele[f"key_rms_L{l}"] = state["keys"][l].float().pow(2).mean().sqrt()
                tele[f"value_rms_L{l}"] = state["values"][l].float().pow(2).mean().sqrt()
            state["telemetry"] = tele
        return state

    # ── the read interface: final layer's (keys, values) ──────────────────────
    def final_kv(self, state: dict) -> tuple[Tensor, Tensor]:
        return state["keys"][self.depth], state["values"][self.depth]
