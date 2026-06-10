"""Graph substrate v7: stable vocabulary-atom bank + co-activation-masked mobile edges.

Design: docs/graph_v7_doctrine.md. Supersedes the graph_v6 WRITE path; reuses the v6
⊙ bind readout (GraphV6FactBuilder) and the prepend read.

Spine:
  - STABLE atom bank (learned, NOT data-modified; the v6 node_gate is gone).
  - per scope: route tokens->atoms, accumulate co-activation C (WITHIN-scope outer
    products, decayed) + per-atom content (pooled token content).
  - per window: a TokenGT over [atom tokens (carry content; un-lit masked) + edge
    tokens] residually updates the FREE edge queries q_src/q_dst + edge_state.
  - finalize: materialize endpoints by masked softmax (src over active atoms; dst over
    active + co-active-with-src via log(p_src @ C)), ⊙ bind -> prepend memory.

Gradient discipline (doc §10): eps-floored log masks, soft softmaxes (no top-k/Gumbel),
residual (differentiable-carry) edge updates, C used as a DETACHED mask (v1).
Competition over edge slots is deferred to phase 2b (edge_id breaks symmetry for now).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_substrate import AttnBlock
from .graph_substrate_v5 import _rmsnorm
from .graph_substrate_v6 import GraphV6FactBuilder

T_NODE, T_SRC, T_DST, T_STATE = 0, 1, 2, 3
N_TYPES = 4


# ── HRR (Holographic Reduced Representation) ops — Plate 1995 ───────────────
# Circular convolution binds a (key, value) pair into one vector of the same
# dim; circular correlation (convolution with the involution of the key)
# unbinds it, recovering an approximation of value. A SUM of bound pairs is a
# recoverable superposition: unbinding by key_i recovers value_i plus crosstalk
# noise that is near-orthogonal to every value when the keys are near-random.
# The FFT path is exact circular conv/corr and is done in fp32 (rfft is not
# supported / is lossy in bf16); callers cast back to the working dtype.
def hrr_bind(k: Tensor, v: Tensor) -> Tensor:
    """Circular convolution  irfft(rfft(k) * rfft(v))  — bind key⊗value. Last-dim batched."""
    d = k.shape[-1]
    kf = torch.fft.rfft(k.float(), n=d, dim=-1)
    vf = torch.fft.rfft(v.float(), n=d, dim=-1)
    return torch.fft.irfft(kf * vf, n=d, dim=-1)


def hrr_unbind(m: Tensor, k: Tensor) -> Tensor:
    """Circular correlation  irfft(rfft(m) * conj(rfft(k)))  — recover value bound under k."""
    d = m.shape[-1]
    mf = torch.fft.rfft(m.float(), n=d, dim=-1)
    kf = torch.fft.rfft(k.float(), n=d, dim=-1)
    return torch.fft.irfft(mf * kf.conj(), n=d, dim=-1)


class _NormMatch(nn.Module):
    """Rescale projected memory tokens to ~Llama token magnitude (local copy of the
    encoder helper, kept here to avoid a circular import)."""

    def __init__(self, d: int, target: float = 0.9):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(float(target)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.ln(x), dim=-1) * self.scale


def init_graph_v7_state(B, K_node, K_edge, d_node, d_state, d_val,
                        mu_q, log_sigma_q, mu_state, log_sigma_state,
                        device, dtype, generator: Optional[torch.Generator] = None) -> dict:
    """Edges drawn fresh per pass from learned (mu, log_sigma); accumulators start at 0.
    Atoms are NOT here — they are a stable module Parameter."""
    def _sample(mu, log_sigma, K, d):
        eps = torch.randn(B, K, d, device=device, dtype=dtype, generator=generator)
        return mu.view(1, 1, -1) + log_sigma.exp().view(1, 1, -1) * eps
    z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
    zf = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)
    return {
        "C": zf(B, K_node, K_node),         # co-activation (fp32 — drives the sharp dst mask)
        "content": z(B, K_node, d_val),     # per-atom pooled token content (the value bundle)
        "a_accum": zf(B, K_node),           # per-atom total activation (fp32 — the active threshold)
        "q_src": _sample(mu_q, log_sigma_q, K_edge, d_node),
        "q_dst": _sample(mu_q, log_sigma_q, K_edge, d_node),
        "state": _sample(mu_state, log_sigma_state, K_edge, d_state),
        "n_windows": 0,
        # D1 routing-collapse telemetry: running sum of per-token routing entropy
        # H(softmax(route_logits)) and the real-token count, averaged at finalize.
        "route_ent_sum": zf(B),
        "route_tok_count": zf(B),
        # D2 relation telemetry: edge-state direction at the previous window
        # (cos drift) — None until the first edge update writes a state.
        "prev_state": None,
    }


class GraphV7Substrate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K_node = cfg.graph_v6_K_node
        self.K_edge = cfg.graph_v6_K_edge
        self.d_node = cfg.graph_v6_d_node
        self.d_state = cfg.graph_v6_d_state
        self.d_val = self.d_state
        d = int(cfg.graph_v7_d_updater)   # ~48M trainable (matches the hand-built cluster)
        d_read = cfg.graph_v6_d_read
        d_llama = cfg.d_llama
        self.scope_size = int(cfg.graph_v7_scope_size)
        self.decay = float(cfg.graph_v7_decay)

        # STABLE atom bank (the vocabulary). Random at construction; lazily
        # re-seeded by a first-batch k-means over the contextualized route_enc
        # token reps (see streaming_write). `_atoms_inited` gates that one-shot
        # init so it fires exactly once, in training, with no grad.
        self.atoms = nn.Parameter(torch.randn(self.K_node, self.d_node) / math.sqrt(self.d_node))
        self.register_buffer("_atoms_inited", torch.tensor(False), persistent=True)
        # SPLIT learnable temperatures (was a SINGLE shared log_tau — sharing
        # forced routing and endpoint pooling to the same sharpness, blocking
        # routing from peaking). Routing over normalized-cosine logits needs a
        # SMALL tau to peak (a sharp per-token atom assignment, the SHUF=REAL
        # root cause); endpoint pooling tolerates a larger tau.
        self.log_route_tau = nn.Parameter(torch.tensor(math.log(float(cfg.graph_v7_route_tau_init))))
        self.log_endpoint_tau = nn.Parameter(torch.tensor(math.log(float(cfg.graph_v7_endpoint_tau_init))))

        # token -> atom-query (routing) and token -> content value
        self.route_enc = nn.Sequential(nn.Linear(d_llama, d), nn.GELU(), nn.Linear(d, self.d_node))
        self.content_enc = nn.Sequential(nn.Linear(d_llama, d), nn.GELU(), nn.Linear(d, self.d_val))

        # learned edge init
        self.mu_q = nn.Parameter(torch.zeros(self.d_node))
        self.log_sigma_q = nn.Parameter(torch.zeros(self.d_node))
        self.mu_state = nn.Parameter(torch.zeros(self.d_state))
        self.log_sigma_state = nn.Parameter(torch.zeros(self.d_state))

        # ── per-window TokenGT edge updater (self-attn over [atoms(+content), edges]) ──
        self.atom_in = nn.Linear(self.d_node, d)
        self.content_in = nn.Linear(self.d_val, d)
        self.src_in = nn.Linear(self.d_node, d)
        self.dst_in = nn.Linear(self.d_node, d)
        self.state_in = nn.Linear(self.d_state, d)
        self.type_emb = nn.Parameter(torch.randn(N_TYPES, d) * 0.02)
        self.atom_id = nn.Parameter(torch.randn(self.K_node, d) * 1.0)  # symmetry-break (write only)
        self.edge_id = nn.Parameter(torch.randn(self.K_edge, d) * 1.0)
        nL, nH = cfg.graph_v6_updater_layers, cfg.graph_v6_updater_heads
        self.self_blocks = nn.ModuleList(AttnBlock(d, nH) for _ in range(nL))
        self.ffns = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(nL))
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(nL))

        def _head(out_dim):
            h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))
            nn.init.normal_(h[-1].weight, std=0.02)   # small (near-carry) but NONZERO -> grad flows
            nn.init.zeros_(h[-1].bias)
            return h
        self.src_head = _head(self.d_node)
        self.dst_head = _head(self.d_node)
        self.state_head = _head(self.d_state)

        # ── readout: reuse the v6 ⊙ bind ──
        self.fact_builder = GraphV6FactBuilder(
            d_node=self.d_node, d_state=self.d_state, d_read=d_read,
            film_hidden=cfg.graph_v6_film_hidden, mlp_hidden=cfg.graph_v6_builder_mlp_hidden)
        self.W_out = nn.Linear(d_read, d_llama)
        self.prepend_norm = _NormMatch(d_llama)
        # competition (edges claim distinct pairs) + atom decorrelation (keep vocab spread)
        self.comp_coef = float(cfg.graph_v7_competition_coef)
        self.decorr_coef = float(cfg.graph_v7_decorr_coef)
        # FIXED top-k active set (replaces the budget-unstable relative-fraction
        # mask). The same active set is reused in materialize and update_edges'
        # KV mask so the two never diverge.
        self.active_topk = int(cfg.graph_v7_active_topk)

        # ── bind-early / unbind-late associative memory (HRR) ──────────────
        # When cfg.graph_v7_bind, each token's VALUE is HRR-bound to its entity
        # KEY the instant before it route-pools into content[atom], so each atom
        # holds a recoverable superposition instead of a collapsed mean. The
        # decoder UNBINDS by the question-derived key at read (see GraphV7UnbindReader
        # + the unbind READ in compute_qa_loss). key_proj/value_proj live HERE
        # (shared by write and the read — the write & read keys MUST match);
        # W_recover lifts the unbound d_val value back to d_llama. All bind at d_val.
        self.bind = bool(getattr(cfg, "graph_v7_bind", False))
        if self.bind:
            # principled 1/√fan_in init (no bare magic constants).
            self.key_proj = nn.Linear(d_llama, self.d_val)
            self.value_proj = nn.Linear(d_llama, self.d_val)
            self.W_recover = nn.Linear(self.d_val, d_llama)
            for lin in (self.key_proj, self.value_proj, self.W_recover):
                nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
                nn.init.zeros_(lin.bias)

    # ── state ────────────────────────────────────────────────────────────────
    def init_state(self, B, device, dtype, generator=None):
        return init_graph_v7_state(
            B, self.K_node, self.K_edge, self.d_node, self.d_state, self.d_val,
            self.mu_q, self.log_sigma_q, self.mu_state, self.log_sigma_state,
            device, dtype, generator)

    def _route_tau(self):
        # Routing needs to peak: allow a much smaller floor (log(0.02)) so the
        # learnable tau CAN sharpen the per-token assignment. Ceiling log(20).
        return self.log_route_tau.clamp(math.log(0.02), math.log(20.0)).exp()

    def _endpoint_tau(self):
        return self.log_endpoint_tau.clamp(-3.0, 3.0).exp()

    def _norm_input(self, token_embeds):
        """Per-token RMS normalization of the encoder INPUT.

        Contextualized hidden states (and partial-layer 'lower attune' states)
        carry ~100× the L2 norm of raw embeds (Llama 'massive activations') and
        have depth-varying scale, while route_enc/content_enc were tuned on
        unit-norm-calibrated inputs. A data-driven RMS rescale (NOT a fixed
        1/√d constant — the partial-layer path has no fixed norm) brings raw,
        contextualized, and lower-attune regimes to the SAME scale identically."""
        rms = token_embeds.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return (token_embeds.float() * rms).to(token_embeds.dtype)

    # ── per-scope cheap accumulation ───────────────────────────────────────────
    def route(self, token_embeds, mask, return_logits: bool = False):
        # token_embeds is ALREADY input-normalized by the caller (accumulate).
        tq = F.normalize(self.route_enc(token_embeds), dim=-1)         # [B,T,d_node]
        ak = F.normalize(self.atoms, dim=-1)                          # [Kn,d_node]
        logits = (torch.einsum('btd,kd->btk', tq, ak) / self._route_tau()).float()
        a = torch.softmax(logits, dim=-1).to(token_embeds.dtype)      # [B,T,Kn] (sharp op in fp32)
        if mask is not None:
            a = a * mask.unsqueeze(-1).to(a.dtype)
        if return_logits:
            return a, logits
        return a

    def accumulate(self, state, token_embeds, mask):
        token_embeds = self._norm_input(token_embeds)                 # C1: scale-match the input
        a, route_logits = self.route(token_embeds, mask, return_logits=True)  # [B,T,Kn]
        B, T, Kn = a.shape
        ss = self.scope_size
        pad = (ss - T % ss) % ss
        a_pad = F.pad(a, (0, 0, 0, pad)) if pad else a
        a_sc = a_pad.view(B, -1, ss, Kn).sum(dim=2)                   # [B,n_scope,Kn] within-scope sum
        C_delta = torch.einsum('bsi,bsj->bij', a_sc, a_sc)           # Σ_scope a⊗a (NOT (Σa)⊗(Σa))
        if self.bind:
            # BIND-EARLY: tag each token's VALUE with its entity-KEY via HRR
            # circular convolution BEFORE it route-pools, so content[atom]
            # becomes a recoverable superposition Σ_t a[t,atom]·(key_t ⊛ value_t)
            # instead of a collapsed mean of raw values (the SHUF=REAL cause).
            # token_embeds here is the CONTEXTUALIZED, RMS-normed input.
            # HRR's recoverability requires UNIT-NORM operands — un-normalized
            # projections (L2≈√d_val≈22) make the circular conv blow up ~1000×,
            # which drowns the read in a passage-agnostic magnitude bias (the
            # SHUF=REAL-after-training cause). L2-normalize key & value to unit
            # norm (principled HRR requirement, not a magic scale constant).
            key_t = F.normalize(self.key_proj(token_embeds), dim=-1)  # [B,T,d_val] unit
            value_t = F.normalize(self.value_proj(token_embeds), dim=-1)  # [B,T,d_val] unit
            content_tok = hrr_bind(key_t, value_t).to(token_embeds.dtype)  # [B,T,d_val]
        else:
            content_tok = self.content_enc(token_embeds)             # [B,T,d_val]
        content_delta = torch.einsum('btk,btd->bkd', a, content_tok)
        a_delta = a.sum(dim=1)                                        # [B,Kn]
        # decay applied once per WINDOW (not per scope) — a longer-memory accumulator that
        # suits encode-once EMAT; the within-scope outer-product STRUCTURE above (no spurious
        # cross-scope edges) is the load-bearing part and holds regardless of the decay cadence.
        d = self.decay
        state = dict(state)
        state["C"] = d * state["C"] + C_delta.float()                # fp32 accumulator
        state["content"] = d * state["content"] + content_delta
        state["a_accum"] = d * state["a_accum"] + a_delta.float()
        # D1: per-token routing entropy (the routing-COLLAPSE signal — peakedness
        # of the softmax, NOT bank coverage). Reads ~log(Kn) at init; a learning
        # router drives it down. Accumulate over real tokens, average at finalize.
        # NOTE: telemetry uses .detach() (NOT a `with torch.no_grad()` block) —
        # this method runs INSIDE activation checkpointing (use_reentrant=False),
        # where a no_grad block forward-vs-recompute mismatch corrupts the saved
        # tensor set. detach() is grad-mode-invariant, so forward and recompute
        # build the identical autograd graph.
        p = torch.softmax(route_logits.detach(), dim=-1).clamp_min(1e-9)  # [B,T,Kn] fp32
        ent_tok = -(p * p.log()).sum(-1)                              # [B,T]
        if mask is not None:
            m = mask.to(ent_tok.dtype)
            state["route_ent_sum"] = state["route_ent_sum"] + (ent_tok * m).sum(1)
            state["route_tok_count"] = state["route_tok_count"] + m.sum(1)
        else:
            state["route_ent_sum"] = state["route_ent_sum"] + ent_tok.sum(1)
            state["route_tok_count"] = state["route_tok_count"] + float(T)
        return state

    # ── active set (B3): FIXED top-k atoms by accumulated activation ───────────
    def _active_mask(self, a_accum):
        """Return a [B,Kn] bool mask selecting the top-`active_topk` atoms by
        accumulated activation. The SAME mask is used in update_edges' KV mask
        and in materialize's endpoint softmax (D5: keep the two consistent).
        Atoms with zero activation (never lit) are excluded — a top-k over an
        all-zero row would otherwise pick arbitrary cold atoms."""
        a = a_accum.float()                                          # [B,Kn]
        B, Kn = a.shape
        k = min(self.active_topk, Kn)
        # top-k indices by a_accum
        topk_idx = a.topk(k, dim=-1).indices                        # [B,k]
        mask = torch.zeros(B, Kn, dtype=torch.bool, device=a.device)
        mask.scatter_(1, topk_idx, True)
        mask = mask & (a > 1e-6)                                     # drop never-lit atoms
        return mask                                                  # [B,Kn]

    # ── per-window edge update ─────────────────────────────────────────────────
    def update_edges(self, state):
        B = state["q_src"].shape[0]
        atoms = self.atoms.unsqueeze(0).expand(B, -1, -1)
        te = self.type_emb
        atom_tok = (self.atom_in(atoms) + self.content_in(state["content"])
                    + te[T_NODE] + self.atom_id.unsqueeze(0))
        src_tok = self.src_in(state["q_src"]) + te[T_SRC] + self.edge_id.unsqueeze(0)
        dst_tok = self.dst_in(state["q_dst"]) + te[T_DST] + self.edge_id.unsqueeze(0)
        st_tok = self.state_in(state["state"]) + te[T_STATE] + self.edge_id.unsqueeze(0)
        tokens = torch.cat([atom_tok, src_tok, dst_tok, st_tok], dim=1)
        Kn, Ke = self.K_node, self.K_edge
        active = self._active_mask(state["a_accum"])                 # [B,Kn] top-k (D5: same as materialize)
        kv_mask = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        kv_mask[:, :Kn] = ~active                                    # un-lit atoms not attendable
        for L in range(len(self.self_blocks)):
            tokens = self.self_blocks[L](tokens, tokens, kv_pad_mask=kv_mask)
            tokens = tokens + self.ffns[L](self.ffn_norms[L](tokens))
        src_o = tokens[:, Kn:Kn + Ke]
        dst_o = tokens[:, Kn + Ke:Kn + 2 * Ke]
        st_o = tokens[:, Kn + 2 * Ke:Kn + 3 * Ke]
        state = dict(state)
        state["q_src"] = state["q_src"] + self.src_head(src_o)        # residual = carry + migrate
        state["q_dst"] = state["q_dst"] + self.dst_head(dst_o)
        prev_field = state["state"]
        state_update = self.state_head(st_o)                         # pre-rmsnorm update
        state["state"] = _rmsnorm(prev_field + state_update)
        # D2 relation telemetry (edge_state_norm was constant — the rmsnorm pins
        # it). Magnitude of the pre-rmsnorm update + direction drift vs last window.
        # detach() (NOT a no_grad block) for checkpoint forward/recompute parity.
        new_s = state["state"].detach()
        state["state_update_mag"] = state_update.detach().float().norm(dim=-1).mean().to(torch.float32)
        prev = state.get("prev_state")
        if prev is not None:
            cos = F.cosine_similarity(new_s.float(), prev.float(), dim=-1)  # [B,Ke]
            state["state_dir_drift"] = (1.0 - cos).mean().to(torch.float32)
        else:
            state["state_dir_drift"] = torch.zeros((), dtype=torch.float32, device=new_s.device)
        state["prev_state"] = new_s
        return state

    @torch.no_grad()
    def _maybe_kmeans_init(self, token_embeds, mask):
        """LAZY first-batch k-means init of the atom bank (phase-5 hook).

        Random atoms make routing start from an arbitrary, uninformative
        partition. Seeding from the data's own route_enc reps gives every atom a
        live receptive field from step 0. Runs ONCE, in training only, no grad,
        deterministic (no rand/seed — the data and a fixed stride pick the seeds
        and assignments). Uses the SAME _norm_input + route_enc the router uses,
        on the (contextualized, in ctx mode) token reps actually being written."""
        if bool(self._atoms_inited) or not self.training:
            return
        x = self._norm_input(token_embeds)                          # [B,T,d_llama]
        reps = F.normalize(self.route_enc(x), dim=-1).reshape(-1, self.d_node)  # [B*T,d_node]
        if mask is not None:
            flat_mask = mask.reshape(-1).bool()
            reps = reps[flat_mask]
        if reps.shape[0] < self.K_node:
            return                                                  # too few real tokens; try next batch
        reps = reps.float()
        # cap the sample deterministically (even stride, not random) to ~4096
        cap = 4096
        if reps.shape[0] > cap:
            stride = reps.shape[0] // cap
            reps = reps[::stride][:cap]
        # deterministic seeds: evenly-strided rows of the sample (no RNG)
        seed_idx = torch.linspace(0, reps.shape[0] - 1, self.K_node, device=reps.device).long()
        centroids = F.normalize(reps[seed_idx].clone(), dim=-1)     # [Kn,d_node]
        for _ in range(5):                                          # a few Lloyd iterations
            sim = reps @ centroids.t()                             # cosine (both unit-norm) [N,Kn]
            assign = sim.argmax(dim=-1)                            # [N]
            for c in range(self.K_node):
                sel = reps[assign == c]
                if sel.shape[0] > 0:
                    centroids[c] = F.normalize(sel.mean(dim=0), dim=-1)
                # empty cluster: keep its previous (seeded) centroid
        self.atoms.data.copy_(F.normalize(centroids, dim=-1).to(self.atoms.dtype))
        self._atoms_inited.fill_(True)

    def streaming_write(self, state, token_embeds, mask):
        B = token_embeds.shape[0]
        # NOTE: k-means atom init is NOT called here — it is a stateful no_grad
        # side-effect (re-seeds self.atoms once) and this method runs INSIDE
        # activation checkpointing (model.py), where a control-flow change
        # between forward and recompute corrupts the saved-tensor metadata. The
        # encoder fires _maybe_kmeans_init EAGERLY before the checkpointed loop.
        has_real = (mask.any(dim=1) if mask is not None
                    else torch.ones(B, dtype=torch.bool, device=token_embeds.device))   # [B]
        new = self.accumulate(state, token_embeds, mask)
        new = self.update_edges(new)
        # all-pad rows: preserve ALL per-row state (no real tokens this window -> no mutation)
        hr3 = has_real.view(-1, 1, 1)
        out = dict(new)
        for k in ("q_src", "q_dst", "state", "content", "C"):
            out[k] = torch.where(hr3, new[k], state[k])
        out["a_accum"] = torch.where(has_real.view(-1, 1), new["a_accum"], state["a_accum"])
        # route-entropy accumulators are [B]; accumulate() already only added real
        # tokens, but for an all-pad row keep the previous running sums untouched.
        out["route_ent_sum"] = torch.where(has_real, new["route_ent_sum"], state["route_ent_sum"])
        out["route_tok_count"] = torch.where(has_real, new["route_tok_count"], state["route_tok_count"])
        # prev_state ([B,Ke,d_state]): all-pad rows keep their prior prev_state
        # so cos-drift is measured between genuine updates (None passes through).
        if new.get("prev_state") is not None and state.get("prev_state") is not None:
            out["prev_state"] = torch.where(hr3, new["prev_state"], state["prev_state"])
        out["n_windows"] = state["n_windows"] + 1
        return out

    # ── endpoint materialization + readout ─────────────────────────────────────
    def materialize(self, state):
        atoms = self.atoms
        ak = F.normalize(atoms, dim=-1)                              # KEYS (addressing) — static bank
        content = state["content"]                                  # VALUES (per-example) [B,Kn,d_val]
        qs = F.normalize(state["q_src"], dim=-1)
        qd = F.normalize(state["q_dst"], dim=-1)
        # FIXED top-k active set (B3): the |active| atoms by accumulated activation.
        # (a_accum > 1e-6 was a NO-OP — softmax routing lights every atom a little, so the old
        #  mask never masked and endpoints pooled over the whole bank = the PMA failure.
        #  The relative-fraction mask was budget-unstable; a fixed top-k is invariant
        #  to the activation scale and reused as the edge-update KV mask.)
        a = state["a_accum"].float()                                # [B,Kn] (fp32)
        active_mask = self._active_mask(a)                          # [B,Kn] top-k (same as update_edges)
        active = active_mask.unsqueeze(1)                           # [B,1,Kn]
        has_active = active_mask.any(dim=-1).view(-1, 1, 1)         # all-pad rows -> zero memory
        active_count = active_mask.float().sum(-1).mean()          # realized |active| telemetry
        neg = torch.finfo(torch.float32).min
        et = self._endpoint_tau()
        s_logits = (torch.einsum('bed,kd->bek', qs, ak) / et).float().masked_fill(~active, neg)
        p_src = torch.softmax(s_logits, dim=-1).to(content.dtype)  # [B,Ke,Kn]
        # dst restricted to src's co-activation partners (p_src @ C), diagonal (self) excluded
        C = state["C"].detach().float()                            # C is a MASK (no grad through accum)
        Cz = C - torch.diag_embed(torch.diagonal(C, dim1=1, dim2=2))
        co = torch.einsum('bek,bkj->bej', p_src.float(), Cz)       # [B,Ke,Kn] genuine-pair mask
        d_logits = (torch.einsum('bed,kd->bek', qd, ak) / et).float()
        d_logits = (d_logits + torch.log(co.clamp_min(1e-6))).masked_fill(~active, neg)
        p_dst = torch.softmax(d_logits, dim=-1).to(content.dtype)
        # endpoints BUNDLE the per-example CONTENT (frozen-LLM value), addressed by the static
        # KEYS — the DKVB split the doctrine specifies (§4/§9): atoms=keys, content=values.
        src_ep = torch.einsum('bek,bkd->bed', p_src, content)      # [B,Ke,d_val]
        dst_ep = torch.einsum('bek,bkd->bed', p_dst, content)
        return src_ep, dst_ep, p_src, p_dst, has_active, active_count

    def finalize(self, state):
        src_ep, dst_ep, p_src, p_dst, has_active, active_count = self.materialize(state)
        fact = self.fact_builder(src_ep, dst_ep, state["state"])     # [B,Ke,d_read]
        memory = self.prepend_norm(self.W_out(fact).float())         # [B,Ke,d_llama]
        memory = memory * has_active.to(memory.dtype)                # all-pad rows -> zero memory
        # ── competition + atom decorrelation (single load_balance_loss the trainer weights;
        # aux-loss-as-fallback for collapse, consistent with the routing load-balance/z-loss) ──
        eye_e = torch.eye(self.K_edge, device=memory.device, dtype=torch.bool)
        joint = (torch.einsum('bek,bfk->bef', p_src, p_src)
                 * torch.einsum('bek,bfk->bef', p_dst, p_dst)).float()   # both-endpoints-shared
        comp_loss = joint.masked_fill(eye_e, 0.0).mean()             # penalize duplicate (src,dst)
        af = F.normalize(self.atoms, dim=-1)
        G = (af @ af.t()).float()
        eye_n = torch.eye(self.K_node, device=G.device, dtype=torch.bool)
        decorr_loss = (G.masked_fill(eye_n, 0.0) ** 2).mean()        # keep the vocabulary spread
        # pre-weighted aux added DIRECTLY by compute_qa_loss via the graph_aux path (NOT
        # re-multiplied by load_balance_coef, which double-scaled it to a ~1e-4 no-op).
        aux = {"graph_aux": self.comp_coef * comp_loss + self.decorr_coef * decorr_loss}
        if self.bind:
            # Export the per-atom HRR-BOUND content for the unbind READ. Mask cold
            # atoms (not in the top-k active set) and all-pad rows to zero so the
            # decoder unbinds only the genuinely-written superposition (matches the
            # materialize active set). This is the RECALL path — the structural
            # ⊙ fact_builder memory above is left in place but BYPASSED for recall.
            a_acc = state["a_accum"].float()
            active_mask = self._active_mask(a_acc).unsqueeze(-1).to(state["content"].dtype)  # [B,Kn,1]
            bound_content = state["content"] * active_mask                 # [B,Kn,d_val]
            bound_content = bound_content * has_active.to(bound_content.dtype)  # all-pad → 0
            aux["graph_v7_bound_content"] = bound_content
        with torch.no_grad():
            def _ent(p):
                p = p.float().clamp_min(1e-9)
                return (-(p * p.log()).sum(-1)).mean().to(torch.float32)
            aux["graph_v7_src_entropy"] = _ent(p_src)
            aux["graph_v7_dst_entropy"] = _ent(p_dst)
            _a = state["a_accum"].float()
            # B3: realized active-set size (≈ active_topk once enough atoms light).
            aux["graph_v7_active_count"] = active_count.to(torch.float32)
            aux["graph_v7_competition_loss"] = comp_loss.to(torch.float32)
            aux["graph_v7_decorr_loss"] = decorr_loss.to(torch.float32)
            aux["graph_v7_atom_collapse_cos"] = G.masked_fill(eye_n, 0.0).abs().mean().to(torch.float32)
            # health: does the relation matter? (||fact − fact(zero edge_state)|| / ||fact||)
            fact0 = self.fact_builder(src_ep, dst_ep, torch.zeros_like(state["state"]))
            num = (fact - fact0).float().norm(dim=-1).mean()
            aux["graph_v7_state_effect"] = (num / fact.float().norm(dim=-1).mean().clamp_min(1e-6)
                                            ).to(torch.float32)
            # D1: bank COVERAGE entropy (spread of usage over the bank in this
            # window) — renamed from atom_usage_entropy to NOT be confused with
            # per-token routing peakedness. Reads ~log(active) regardless of the
            # router; it measures whether atoms are evenly used, not sharpness.
            _u = _a / _a.sum(-1, keepdim=True).clamp_min(1e-9)
            aux["graph_v7_atom_coverage_entropy"] = (-(_u.clamp_min(1e-9) * _u.clamp_min(1e-9).log()
                                                       ).sum(-1)).mean().to(torch.float32)
            # D1: the REAL routing-collapse signal — mean per-token H(softmax(route_logits)),
            # accumulated over the encode. log(Kn) at init; a learning router drives it DOWN.
            tok_count = state["route_tok_count"].clamp_min(1e-6)
            aux["graph_v7_route_entropy_per_token"] = (
                (state["route_ent_sum"] / tok_count).mean().to(torch.float32))
            # D2: edge-state dynamics (edge_state_norm was rmsnorm-pinned constant — dropped).
            aux["graph_v7_state_update_mag"] = state.get(
                "state_update_mag", torch.zeros((), dtype=torch.float32, device=memory.device))
            aux["graph_v7_state_dir_drift"] = state.get(
                "state_dir_drift", torch.zeros((), dtype=torch.float32, device=memory.device))
        return memory, aux
