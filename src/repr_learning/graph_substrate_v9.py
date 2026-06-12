"""graph_substrate_v9.py — operator-node pyramid memory (graph_v9, 2026-06-12 design).

Full design doc: docs/graph_v9_ideas.md. This header is the implementation contract.

THE OBJECT
  A trainable memory side-car to a FROZEN LLM. A pyramid of node layers; each node =
  a SLOW learnable key (never written) + a FAST value: an ordered list of factor
  slots, each slot one generalized Householder factor (direction, strength in
  [0,2]).  apply: code <- code - strength_eff*(dir . code)*dir  per factor, chained
  in fixed (node, slot) index order with strength_eff = routing_score * slot
  strength.

ARM C IS THE PRIMARY DESIGN (vocabulary-by-absorption; pivot 2026-06-12)
  arm "C": NO deposits anywhere. Layer 0 = a STATIC trained atom alphabet; every
    writable layer initializes its per-input state FROM a TRAINED random base
    vocabulary (slow params — random init then gradient-trained, NOT per-input
    noise). The ONLY write is within-layer conserving absorption: binding IS the
    relocation pattern (Marcus's node absorbs the co-firing occupation node's
    factors — that displacement is the stored fact). Total strength per layer is
    CONSTANT for the whole document — the strict conservation invariant. Words
    store PROGRAMS (factor lists move as factor lists), never blended points.
    PROJECTED SEED (decision 2026-06-12): layer 0's apply input = unit-RMS
    projection of the Llama hiddens, per token — mirroring the read, where each
    query token must be projected into code space and travel up the pyramid as
    itself. The projected content RIDES THE STREAM but never lands in a node:
    absorption relocates trained vocabulary only, so the MEMORY stays
    selection-pure even though the stream is content-bearing.
    Writable layers are NOT identity-at-init (there must be content to relocate
    — the same trade the atoms make); Llama-identity at step 0 still holds via
    the reader's zero-init out-projection.
  arm "B": static atoms + projected seed; layers >= 1 take DEPOSITS of the
    operated codes (composed-code deposits — words store points).
  arm "A": projected seed + deposits at every layer (content channel at the
    leaves). The control arm.
  All arms: layer-0 routing reads raw Llama hiddens; layers >= 1 route from the
  OPERATED codes below.

PER TOKEN, PER LAYER (semantics; executed chunkwise, see below)
  1. SCORE   routing_scores = softmax( cos(unit(route_proj(routing_input)),
             unit(node_keys)) * sqrt(d_key) / temp )
  2. DEPOSIT (arms A/B only; arm C has no deposits)
             deposit_rate = write_strength * score * surprise * gate_mlp(...)
     routed into slots by MATCH+ROOM, displacing in proportion (displacement IS
     the eviction policy; there is NO decay anywhere).
  3. APPLY   chain the activated nodes' factors over the layer's apply input.
  4. BOOKKEEPING  lagged coactivation:
             coact <- decay*coact + outer(scores_now, trace_previous);
             trace <- decay*trace + scores_now
     (row = fires-now/absorber, col = fired-earlier/donor — the STDP asymmetry).
  5. PASS UP the operated code (unit-RMS between layers).

CHUNK BOUNDARY (execution; default chunk=128)
  (a) land the chunk's merged deposits (EXACT sequential-lerp closed form — the
      v8 suffix-logsum machinery, never a gate sum); (b) coact/trace closed form
      (EXACT — the table never feeds back within a chunk); (c) ONE ABSORPTION
      pass; (d) next chunk reads updated state. The only approximation anywhere
      is the standard one-chunk self-read staleness. Cadence is part of the model.

ABSORPTION (within-layer Hebbian consolidation; conserving TRANSFER, never copy)
  absorb_gate_ij = sigmoid(absorb_strength) * rowNorm(coact)_ij * surprise_chunk
  * plasticity_mlp(keys_i, keys_j, coact both directions, occupancies) / (1+eta).
  Donor (node j, slot s) content lands in absorber i's slots by MATCH+ROOM with
  room-fractional landing (continuous form: landed = room*(1 - exp(-mass/2)) —
  full slots REFUSE, refused content stays with the donor); the donor is debited
  EXACTLY what landed. Per-layer invariant: total strength changes ONLY via token
  injection + overwrite displacement; absorption relocates.

THE READ  (NOT here — graph_read.GraphV9FlowReader)
  The question flows through the SAME circuit (shared params, write gates closed);
  read result per layer = the OPERATED query code (apply-to-query; nothing is
  retrieved — the memory answers by steering the question). Injected at matched
  decoder depths, zero-init out-projection (Llama bit-identical at step 0).

CONVENTIONS (hard rules)
  - SOFT everywhere: no argmax/argsort/top-k/thresholds. Weak ops fade to identity.
  - Fast state on bounded manifolds: directions normalize-on-use; strengths in
    [0,2] by construction (convex lerps + room-fractional addition). No clipping.
  - Every scale: (a) derived from dims, (b) learnable w/ principled init, or
    (c) an algebra bound. Bare config floats outside these = bug.
  - NAMING: full words. node_keys / factor_dirs / factor_strengths /
    routing_scores / deposit_rate / slot_weights / coact / trace. Shape names:
    batch_size, chunk_len, n_nodes, n_slots, d_code, d_key, d_model, n_layers.
  - fp32 substrate (routing is autocast-guarded).
  - The per-token REFERENCE path implements the SAME chunk-frozen semantics, so
    reference == chunkwise EXACTLY (fp tolerance) — tested in the smoke.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _unit(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, dim=dim, eps=1e-6)


def _unit_rms(x: Tensor) -> Tensor:
    """Unit-RMS norm (no learnable gain — gain is absorbed by the next projection).
    Zero rows stay zero (0 * rsqrt(eps) == 0), so pad rows remain exact no-ops."""
    return x * x.pow(2).mean(-1, keepdim=True).add(1e-12).rsqrt()


def _inverse_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1.0 - p))


@dataclass
class GraphV9Config:
    # dimensions (d_code/d_key multiples of 32; enforced)
    d_model: int = 2048                    # frozen-LLM hidden size
    d_code: int = 64                       # the ONE shared code space (operators act here)
    d_key: int = 64                        # addressing space (separate knob from d_code)
    nodes: tuple = (256, 128)              # nodes per layer (pyramid: decreasing)
    slots: tuple = (1, 4)                  # factor slots per node per layer
    chunk: int = 128                       # chunkwise execution unit (cadence is part of the model)
    arm: str = "C"                         # "C" vocabulary-by-absorption (PRIMARY):
    #   NO deposits anywhere; every writable layer init'd from a TRAINED random
    #   base vocabulary (slow params) and the ONLY write is within-layer
    #   conserving absorption — binding IS the relocation pattern. Total strength
    #   per layer is constant for the whole document (the strict invariant).
    # "B" static atoms + constant seed, deposits at layers >= 1 (composed-code
    #   deposits — words store points). "A" content-channel leaves (control).

    # learnable-but-bounded init points
    effective_k: float = 8.0               # target #active nodes at init -> route temp DERIVED
    write_strength_init: float = 0.50      # max per-token deposit rate (v8 lesson: 0.10 was inert)
    absorb_strength_init: float = 0.10     # max per-boundary transfer rate
    # coact e-fold horizons in TOKENS, geometric ladder tied to chunk (None -> derived:
    # chunk/4, chunk/2, chunk, 2*chunk ... one per layer). Learnable thereafter.
    coact_horizon_inits: Optional[tuple] = None
    plasticity_eta: float = 0.50           # absorption-gate modulation band (identity at init)
    mlp_hidden: int = 16                   # update-gate + plasticity MLP hidden
    absorb_enabled: bool = True            # arm C: REQUIRED (absorption IS the write);
    #                                      # arms A/B probe: OFF (binding only)
    # absorption gate form (overnight H-PMI/H-SHARP, run-1 diagnosis):
    #  "rowfrac"    — row-normalized coactivation (run 1: too diffuse + marginal-
    #                 driven; the template-shared component dominates -> states
    #                 identical across docs -> SHUF==REAL structurally)
    #  "npmi_sharp" — NPMI vs the independence baseline (kills the hot-node/
    #                 template component; only ABOVE-CHANCE co-firing absorbs,
    #                 which is exactly the doc-specific signal) + donor-softmax
    #                 with a calibrated learnable temperature (concentrates each
    #                 absorber's intake on its top partners so real mass moves
    #                 and directions actually rotate).
    absorb_gate: str = "rowfrac"
    effective_k_donors: float = 4.0        # npmi_sharp: target #donors at init (derived temp)
    # HUB-CONVERGENCE fix (overnight, bridge-probe finding): per-node routing-
    # logit CENTERING with running statistics (BN-style, center only, standard
    # BN momentum). Removes each node's always-hot logit component so routing
    # encodes token-RELATIVE structure; running stats are used identically at
    # write and read (symmetry preserved — a per-window mean would break it).
    route_centering: bool = False
    # SURPRISE-WEIGHTED COACTIVATION (overnight run-4 finding): weight each
    # token's contribution to trace/coact by its own surprise, so the table
    # records "which nodes co-fired on NEWS" instead of all co-firing — the
    # template's role-pair traffic (shared across docs) drops out and the
    # doc-specific entity pairs dominate. Predictive-coding-faithful Hebbian
    # (local plasticity x prediction error). Apply/routing still use raw scores.
    surprise_coact: bool = False
    # GATED DIRECTION BLENDING (overnight run-6 verdict): mass-ratio blending
    # froze directions (~1e-5/boundary) in every run — the last untested doc-
    # identity channel. "gated": absorber slots rotate toward their incoming
    # donor mix at a learnable scale-free rate (relative landed share x
    # sigmoid(rate), init 0.5); STRENGTHS STAY CONSERVED. "mass": original.
    dirs_blend: str = "mass"
    wy_block: int = 64                     # factors per WY block in the fast apply

    def __post_init__(self):
        for name in ("d_code", "d_key"):
            value = getattr(self, name)
            if value % 32 != 0:
                raise ValueError(f"GraphV9Config.{name}={value} must be a multiple of 32")
        if len(self.nodes) != len(self.slots):
            raise ValueError("nodes and slots must have one entry per layer")
        if self.arm not in ("A", "B", "C"):
            raise ValueError(f"arm must be 'A', 'B' or 'C', got {self.arm!r}")
        if self.arm == "C" and not self.absorb_enabled:
            raise ValueError("arm C has no deposits — absorption IS the write; "
                             "absorb_enabled must be True")
        if self.coact_horizon_inits is not None and len(self.coact_horizon_inits) < len(self.nodes):
            raise ValueError("need one coact horizon init per layer")


class _GateMLP(nn.Module):
    """Per-(token,node) update-gate multiplier in (0,1). Features: [routing_score,
    surprise, occupancy] — all already in [0,1]. Final layer zero-init => constant
    0.5 at init (gradient wakes the final layer first); no dead gate, no wild gate."""
    N_FEAT = 3

    def __init__(self, hidden: int):
        super().__init__()
        self.fc_in = nn.Linear(self.N_FEAT, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, routing_score: Tensor, surprise: Tensor, occupancy: Tensor) -> Tensor:
        features = torch.stack(
            torch.broadcast_tensors(routing_score, surprise, occupancy), dim=-1)
        return torch.sigmoid(self.fc_out(torch.relu(self.fc_in(features))).squeeze(-1))


class _PlasticityMLP(nn.Module):
    """The learned GRAMMAR: per-(absorber, donor) multiplier on the absorption
    gate. 1 + eta*tanh(MLP) — identity at init (final layer zero), so absorption
    starts as pure row-normalized coactivation x surprise; the MLP only learns
    WHICH KINDS of pairings deserve merging (it sees both keys via cos, both lag
    directions, both occupancies). Frequency proposes, grammar disposes."""
    N_FEAT = 6   # key_cos, coact_ij, coact_ji, occupancy_i, occupancy_j, surprise_chunk

    def __init__(self, eta: float, hidden: int):
        super().__init__()
        self.eta = float(eta)
        self.fc_in = nn.Linear(self.N_FEAT, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, key_cos, coact_ij, coact_ji, occupancy_i, occupancy_j,
                surprise_chunk) -> Tensor:
        features = torch.stack(torch.broadcast_tensors(
            key_cos, coact_ij, coact_ji, occupancy_i, occupancy_j, surprise_chunk), dim=-1)
        modulation = torch.tanh(self.fc_out(torch.relu(self.fc_in(features))).squeeze(-1))
        return 1.0 + self.eta * modulation


class GraphV9Substrate(nn.Module):
    """Operator-node pyramid (see module header)."""

    def __init__(self, config: GraphV9Config):
        super().__init__()
        self.config = config
        n_layers = len(config.nodes)
        self.depth = n_layers

        # ── slow weights ───────────────────────────────────────────────────────
        # node keys: the addressing vocabulary — NEVER written at inference
        self.node_keys = nn.ParameterList(
            nn.Parameter(torch.randn(n_nodes, config.d_key)) for n_nodes in config.nodes)
        # routing projections: layer 0 reads RAW LLAMA HIDDENS (addressing needs
        # the token; content does not — both arms); layers >= 1 read the operated
        # codes of the layer below.
        self.route_projs = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_key, bias=False)]
            + [nn.Linear(config.d_code, config.d_key, bias=False)
               for _ in range(n_layers - 1)])
        for linear in self.route_projs:
            nn.init.normal_(linear.weight, std=1.0 / math.sqrt(linear.in_features))
        if config.arm != "C":
            # deposit content heads (factor-direction + strength proposals), one
            # per layer; layer 0's are USED ONLY IN ARM A. Arm C has NO deposits
            # anywhere — these modules don't exist there.
            self.dir_projs = nn.ModuleList(
                nn.Linear(config.d_code, config.d_code, bias=False) for _ in range(n_layers))
            self.strength_heads = nn.ModuleList(
                nn.Linear(config.d_code, 1) for _ in range(n_layers))
            for linear in self.dir_projs:
                nn.init.normal_(linear.weight, std=1.0 / math.sqrt(config.d_code))
            for linear in self.strength_heads:
                nn.init.normal_(linear.weight, std=1.0 / math.sqrt(config.d_code))
                nn.init.zeros_(linear.bias)   # strength proposal ~ 1 (mid) at init
            self.gate_mlps = nn.ModuleList(_GateMLP(config.mlp_hidden) for _ in range(n_layers))
        # plasticity MLPs only for layers that CAN absorb (no dead params: every
        # trainable tensor must receive gradient — the probe enforces it strictly)
        absorbing_layers = [l for l in range(n_layers)
                            if config.arm == "A" or l > 0]
        self.plasticity_mlps = nn.ModuleDict(
            {str(l): _PlasticityMLP(config.plasticity_eta, config.mlp_hidden)
             for l in absorbing_layers})

        if config.arm in ("B", "C"):
            # the static atom alphabet (layer 0; never written). Atoms start
            # mid-strength (2*sigmoid(0)=1): the alphabet must act from step 0
            # (in arm C it is also the relocatable layer-0 content).
            n_atoms, atom_slots = config.nodes[0], config.slots[0]
            self.atom_dirs = nn.Parameter(torch.randn(n_atoms, atom_slots, config.d_code))
            # init strength 1.5, NOT 1.0: beta=1 is the family's singular point
            # (pure projection — erases its direction and kills that direction's
            # gradient). 1.5 = midpoint of the preserving-write regime [1,2].
            self.atom_strength_logit = nn.Parameter(torch.full(
                (n_atoms, atom_slots), _inverse_sigmoid(0.75)))
        if config.arm == "C":
            # TRAINED random base vocabulary per writable layer: per-input state
            # initializes FROM these (random init, slow-trained — NOT per-input
            # noise, or the read could never learn to interpret relocated
            # content). Absorption then relocates trained content; the writable
            # layers are NOT identity-at-init (there must be content to relocate
            # — the same trade the atoms made). Llama-identity at step 0 still
            # holds via the reader's zero-init out-projection.
            self.base_dirs = nn.ParameterList(
                nn.Parameter(torch.randn(n_nodes, n_slots, config.d_code))
                for n_nodes, n_slots in zip(config.nodes[1:], config.slots[1:]))
            # same off-singular init as the atoms (strength 1.5, not 1.0)
            self.base_strength_logit = nn.ParameterList(
                nn.Parameter(torch.full((n_nodes, n_slots), _inverse_sigmoid(0.75)))
                for n_nodes, n_slots in zip(config.nodes[1:], config.slots[1:]))
        # the PROJECTED SEED (all arms; decision 2026-06-12): layer 0's apply
        # input = unit-RMS projection of the Llama hiddens — per-token, mirroring
        # the read (queries must be projected into code space and travel up the
        # pyramid as themselves; a constant seed would make the read's operand
        # query-independent). In arm C the projected content RIDES THE STREAM but
        # never lands in a node — absorption relocates trained vocabulary only,
        # so the memory stays selection-pure.
        self.seed_proj = nn.Linear(config.d_model, config.d_code, bias=False)
        nn.init.normal_(self.seed_proj.weight, std=1.0 / math.sqrt(config.d_model))

        # ── learnable-but-bounded dynamics ─────────────────────────────────────
        # route temperature: DERIVED at init from (keys, d_key, effective_k) by a
        # one-time data-free calibration (binary search on random unit queries) —
        # changing node counts re-derives, never re-tunes. Learnable thereafter.
        self.log_route_temp = nn.Parameter(torch.zeros(n_layers))
        self._calibrate_route_temps()
        # ACTIVE layers = where any write dynamics run (deposits and/or absorption).
        # Arms B/C: layer 0 is static — its coact table is never consumed, so its
        # dynamics params would be dead ELEMENTS (zero grad forever, review finding)
        # and its coact bookkeeping dead compute. Scope both to active layers only.
        self.active_layers = (tuple(range(n_layers)) if config.arm == "A"
                              else tuple(range(1, n_layers)))
        n_active = len(self.active_layers)
        # coact decay per ACTIVE layer from an e-fold-horizon ladder tied to chunk
        horizons = (config.coact_horizon_inits if config.coact_horizon_inits is not None
                    else tuple(config.chunk * (2.0 ** (i - 2)) for i in range(n_layers)))
        self.coact_decay_logit = nn.Parameter(torch.tensor(
            [_inverse_sigmoid(math.exp(-1.0 / max(horizons[l], 1.0)))
             for l in self.active_layers]))
        if config.arm != "C":      # arm C has no deposits — no dead params
            self.write_strength_logit = nn.Parameter(
                torch.full((n_layers,), _inverse_sigmoid(config.write_strength_init)))
        self.absorb_strength_logit = nn.Parameter(
            torch.full((n_active,), _inverse_sigmoid(config.absorb_strength_init)))
        # match+room mixing (one soft rule, two regimes): slot logits =
        # a*(cos*sqrt(d_code)*(strength/2)) + b*(room/2); a,b learnable, init 1.
        self.match_weight = nn.Parameter(torch.ones(n_active))
        self.room_weight = nn.Parameter(torch.ones(n_active))
        if config.dirs_blend == "gated":
            self.dirs_blend_logit = nn.Parameter(torch.zeros(n_active))  # sigmoid -> 0.5
        if config.absorb_gate in ("npmi_sharp",):
            # donor-softmax temperature, DERIVED at init (data-free, same pattern
            # as route temps): perplexity of softmax(U[0,1]/temp) over the donor
            # count ~= effective_k_donors.
            self.log_donor_temp = nn.Parameter(torch.zeros(n_active))
            with torch.no_grad():
                for slot_i, layer_idx in enumerate(self.active_layers):
                    n_donors = config.nodes[layer_idx] - 1
                    aff = torch.rand(4096, n_donors)
                    low, high = math.log(1e-3), math.log(10.0)
                    target = math.log(min(max(config.effective_k_donors, 1.0), n_donors))
                    for _ in range(30):
                        mid = 0.5 * (low + high)
                        probs = torch.softmax(aff / math.exp(mid), dim=-1)
                        ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean().item()
                        if ent < target:
                            low = mid
                        else:
                            high = mid
                    self.log_donor_temp.data[slot_i] = 0.5 * (low + high)

        if config.route_centering:
            for layer_idx, n_nodes in enumerate(config.nodes):
                self.register_buffer(f"route_logit_mean_L{layer_idx}",
                                     torch.zeros(n_nodes))

        # ── telemetry (docs/graph_v9_ideas.md §8) ──────────────────────────────
        # Per-chunk dynamics metrics, written into a keyed buffer under no_grad.
        # Keys embed the chunk offset so gradient-checkpoint RECOMPUTE overwrites
        # idempotently instead of double-counting. Aggregated (mean over chunks)
        # into state["telemetry"] at forward end. Cheap: small reductions + one
        # 32-token sample for pairwise stats; disable via telemetry_enabled.
        self.telemetry_enabled = True
        self._tele: dict = {}

    # ── bounded accessors ──────────────────────────────────────────────────────
    def _route_temp(self, layer_idx: int) -> Tensor:
        return self.log_route_temp[layer_idx].clamp(math.log(0.05), math.log(20.0)).exp()

    def _active_idx(self, layer_idx: int) -> int:
        return self.active_layers.index(layer_idx)

    def _coact_decay(self, layer_idx: int) -> Tensor:
        return torch.sigmoid(self.coact_decay_logit[self._active_idx(layer_idx)])

    def _write_strength(self, layer_idx: int) -> Tensor:
        return torch.sigmoid(self.write_strength_logit[layer_idx])

    def _absorb_strength(self, layer_idx: int) -> Tensor:
        return torch.sigmoid(self.absorb_strength_logit[self._active_idx(layer_idx)])

    @torch.no_grad()
    def _calibrate_route_temps(self, n_query: int = 4096, iters: int = 30):
        """Set log_route_temp so a random unit query's softmax over THIS layer's
        keys has perplexity ~ effective_k at init. Derivation, not tuning."""
        for layer_idx, keys in enumerate(self.node_keys):
            query = _unit(torch.randn(n_query, self.config.d_key))
            logits = (query @ _unit(keys.float()).t()) * math.sqrt(self.config.d_key)
            low, high = math.log(0.05), math.log(20.0)
            target_entropy = math.log(min(max(self.config.effective_k, 1.0), keys.shape[0]))
            for _ in range(iters):
                mid = 0.5 * (low + high)
                probs = torch.softmax(logits / math.exp(mid), dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean().item()
                if entropy < target_entropy:   # hotter temp -> higher entropy
                    low = mid
                else:
                    high = mid
            self.log_route_temp.data[layer_idx] = 0.5 * (low + high)

    # ── per-sequence state ─────────────────────────────────────────────────────
    def atom_state(self) -> tuple[Tensor, Tensor]:
        """Arms B/C layer-0 factor state, materialized from the slow params."""
        return self.atom_dirs, 2.0 * torch.sigmoid(self.atom_strength_logit)

    def init_state(self, batch_size: int, device, dtype=torch.float32) -> dict:
        del dtype                                            # substrate is fp32
        config = self.config
        factor_dirs, factor_strengths, coact, trace = [], [], [], []
        for layer_idx, (n_nodes, n_slots) in enumerate(zip(config.nodes, config.slots)):
            if layer_idx == 0 and config.arm in ("B", "C"):
                # static alphabet: state tensors are VIEWS of the params (expand
                # keeps autograd; never written at layer 0 in arms B/C).
                dirs_param, strengths_param = self.atom_state()
                factor_dirs.append(dirs_param.to(device, torch.float32)
                                   .unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous())
                factor_strengths.append(strengths_param.to(device, torch.float32)
                                        .unsqueeze(0).expand(batch_size, -1, -1).contiguous())
            elif config.arm == "C":
                # writable layers start AT the trained base vocabulary; the only
                # write is conserving relocation, so total strength per layer is
                # constant for the whole document (the strict invariant).
                base_dirs = self.base_dirs[layer_idx - 1]
                base_strengths = 2.0 * torch.sigmoid(self.base_strength_logit[layer_idx - 1])
                factor_dirs.append(base_dirs.to(device, torch.float32)
                                   .unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous())
                factor_strengths.append(base_strengths.to(device, torch.float32)
                                        .unsqueeze(0).expand(batch_size, -1, -1).contiguous())
            else:
                # empty memory: zero dirs (unit-on-use of 0 == 0 -> exact identity
                # regardless of strength; first deposit SETS the direction), zero
                # strengths (identity operators).
                factor_dirs.append(torch.zeros(
                    batch_size, n_nodes, n_slots, config.d_code, device=device))
                factor_strengths.append(torch.zeros(batch_size, n_nodes, n_slots, device=device))
            coact.append(torch.zeros(batch_size, n_nodes, n_nodes, device=device))
            trace.append(torch.zeros(batch_size, n_nodes, device=device))
        return {"factor_dirs": factor_dirs, "factor_strengths": factor_strengths,
                "coact": coact, "trace": trace, "step": 0, "telemetry": {}}

    # ── checkpoint-safe flat <-> dict state ────────────────────────────────────
    def _flatten_state(self, state: dict) -> tuple:
        return (*state["factor_dirs"], *state["factor_strengths"],
                *state["coact"], *state["trace"])

    def _unflatten_state(self, flat: tuple) -> dict:
        n_layers, i = self.depth, 0
        factor_dirs = list(flat[i:i + n_layers]); i += n_layers
        factor_strengths = list(flat[i:i + n_layers]); i += n_layers
        coact = list(flat[i:i + n_layers]); i += n_layers
        trace = list(flat[i:i + n_layers]); i += n_layers
        return {"factor_dirs": factor_dirs, "factor_strengths": factor_strengths,
                "coact": coact, "trace": trace, "step": 0, "telemetry": {}}

    # ── routing (shared by write AND read — parameter identity is the addressing
    # guarantee: the same content routes the same way at write and read time) ────
    def route(self, layer_idx: int, routing_input: Tensor) -> Tensor:
        """routing_input [..., d_model] for layer 0 (raw Llama hiddens), else
        [..., d_code] (the operated codes of the layer below). Returns
        routing_scores [..., n_nodes]."""
        with torch.autocast(device_type=routing_input.device.type, enabled=False):
            query = _unit(self.route_projs[layer_idx](routing_input.float()))
            keys = _unit(self.node_keys[layer_idx].float())
            logits = torch.einsum("...d,nd->...n", query, keys) * math.sqrt(self.config.d_key)
            if self.config.route_centering:
                running = getattr(self, f"route_logit_mean_L{layer_idx}")
                if self.training:
                    with torch.no_grad():
                        batch_mean = logits.reshape(-1, logits.shape[-1]).mean(0)
                        running.mul_(0.9).add_(batch_mean, alpha=0.1)  # BN convention
                logits = logits - running
            return torch.softmax(logits / self._route_temp(layer_idx), dim=-1)

    # ── the APPLY: chain of generalized Householder factors ────────────────────
    # code_j = code_{j-1} - strength_j (dir_j . code_{j-1}) dir_j over flat factor
    # index (node*n_slots + slot) ascending. The composed map's coefficients c
    # solve the unit-lower-triangular system
    #   (I + diag(strength) tril(DirDir^T, -1)) c = strength * (Dir code)
    # block-solved (wy_block at a time) — exact; strength=0 rows are exact no-ops.
    def apply_chain(self, codes: Tensor, factor_dirs: Tensor,
                    factor_strengths: Tensor, routing_scores: Tensor) -> Tensor:
        """codes [batch, chunk_len, d_code]; factor_dirs [batch, n_nodes, n_slots,
        d_code] (raw; unit-on-use); factor_strengths [batch, n_nodes, n_slots];
        routing_scores [batch, chunk_len, n_nodes]. Returns operated codes."""
        batch_size, chunk_len, d_code = codes.shape
        n_nodes, n_slots = factor_strengths.shape[1], factor_strengths.shape[2]
        n_factors = n_nodes * n_slots
        unit_dirs = _unit(factor_dirs.reshape(batch_size, n_factors, d_code))
        strength_eff = (routing_scores.unsqueeze(-1)
                        * factor_strengths.unsqueeze(1)).reshape(batch_size, chunk_len, n_factors)
        block = max(1, int(self.config.wy_block))
        out = codes
        for block_start in range(0, n_factors, block):
            block_end = min(block_start + block, n_factors)
            dirs_block = unit_dirs[:, block_start:block_end]                  # [B,K,d]
            strength_block = strength_eff[:, :, block_start:block_end]        # [B,C,K]
            overlap = torch.einsum("bkd,bcd->bck", dirs_block, out)
            gram = torch.einsum("bkd,bjd->bkj", dirs_block, dirs_block)
            block_len = block_end - block_start
            gram_strict = gram.tril(-1).unsqueeze(1)
            system = torch.eye(block_len, device=codes.device, dtype=out.dtype) \
                + strength_block.unsqueeze(-1) * gram_strict                  # [B,C,K,K]
            rhs = (strength_block * overlap).unsqueeze(-1)
            coeffs = torch.linalg.solve_triangular(
                system, rhs, upper=False, unitriangular=True).squeeze(-1)
            out = out - torch.einsum("bck,bkd->bcd", coeffs, dirs_block)
        return out

    def apply_chain_reference(self, codes: Tensor, factor_dirs: Tensor,
                              factor_strengths: Tensor, routing_scores: Tensor) -> Tensor:
        """Sequential factor loop — the semantics; smoke-tested == apply_chain."""
        batch_size, chunk_len, d_code = codes.shape
        n_nodes, n_slots = factor_strengths.shape[1], factor_strengths.shape[2]
        n_factors = n_nodes * n_slots
        unit_dirs = _unit(factor_dirs.reshape(batch_size, n_factors, d_code))
        strength_eff = (routing_scores.unsqueeze(-1)
                        * factor_strengths.unsqueeze(1)).reshape(batch_size, chunk_len, n_factors)
        out = codes
        for factor_idx in range(n_factors):
            direction = unit_dirs[:, factor_idx].unsqueeze(1)                 # [B,1,d]
            overlap = (out * direction).sum(-1, keepdim=True)                 # [B,C,1]
            out = out - strength_eff[:, :, factor_idx].unsqueeze(-1) * overlap * direction
        return out

    # ── match+room slot routing (the WHERE of every deposit) ───────────────────
    def _slot_weights(self, layer_idx: int, incoming_dir_unit: Tensor,
                      unit_dirs: Tensor, factor_strengths: Tensor) -> Tensor:
        """incoming_dir_unit [batch, chunk_len, d_code] vs every node's slots:
        unit_dirs [batch, n_nodes, n_slots, d_code], factor_strengths [batch,
        n_nodes, n_slots]. Returns softmax-over-slots weights [batch, chunk_len,
        n_nodes, n_slots]. Match = cos scaled by sqrt(d_code) (random cos ~
        1/sqrt(d) -> dimension-free) and by slot strength (empty slots can't
        match); room = remaining capacity. One soft rule, two regimes: strong
        match -> refine in place; no match -> allocate to free space."""
        cos = torch.einsum("bcd,bnsd->bcns", incoming_dir_unit, unit_dirs)
        match = cos * math.sqrt(self.config.d_code) * (factor_strengths.unsqueeze(1) / 2.0)
        room = ((2.0 - factor_strengths) / 2.0).unsqueeze(1)
        logits = self.match_weight[self._active_idx(layer_idx)] * match + self.room_weight[self._active_idx(layer_idx)] * room
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def _routing_apply_telemetry(self, prefix: str, routing_scores: Tensor,
                                 mask: Tensor, apply_input: Tensor, operated: Tensor,
                                 factor_strengths0: Tensor) -> None:
        """§8 panel, per chunk per layer. Collapse axes: per-token sharpness
        (uniform = the v8 disease), node-usage concentration (few hot nodes =
        node collapse), token routing overlap (1.0 = every token routes the same
        = addressing collapse, the SHUF=REAL precursor). Apply axes: what the
        factors DO (effective beta: 0 skip / 1 erase / 2 reflect), how much the
        chain transforms (norm ratio, rotation), homogenization (code overlap
        in vs out)."""
        t, eps_t = self._tele, 1e-12
        real = mask > 0
        n_real = real.sum().clamp_min(1).float()
        sc = routing_scores
        ent_tok = -(sc.clamp_min(eps_t) * sc.clamp_min(eps_t).log()).sum(-1)
        t[prefix + "route_tok_eff_k"] = float(((ent_tok * mask).sum() / n_real).exp())
        usage = (sc * mask.unsqueeze(-1)).sum(dim=(0, 1))
        usage = usage / usage.sum().clamp_min(eps_t)
        usage_ent = -(usage.clamp_min(eps_t) * usage.clamp_min(eps_t).log()).sum()
        t[prefix + "route_usage_eff_frac"] = float(usage_ent.exp() / sc.shape[-1])
        sample_idx = real[0].nonzero(as_tuple=False).squeeze(-1)[:32]
        if sample_idx.numel() >= 2:
            s_sample = F.normalize(sc[0, sample_idx], dim=-1, eps=1e-9)
            cos_sc = s_sample @ s_sample.t()
            off = ~torch.eye(s_sample.shape[0], dtype=torch.bool, device=cos_sc.device)
            t[prefix + "route_token_overlap"] = float(cos_sc[off].mean())
            beta = (sc[0, sample_idx].unsqueeze(-1)
                    * factor_strengths0[0].unsqueeze(0)).reshape(-1)
            t[prefix + "beta_eff_mean"] = float(beta.mean())
            t[prefix + "beta_frac_skip"] = float((beta < 0.5).float().mean())
            t[prefix + "beta_frac_erase"] = float(((beta >= 0.5) & (beta < 1.5)).float().mean())
            t[prefix + "beta_frac_reflect"] = float((beta >= 1.5).float().mean())
            # effective beta of the TOP-ROUTED factor per token — the work the
            # chain actually does (the all-factor mean is skip-dominated by the
            # softmax tail and says little)
            t[prefix + "beta_eff_top"] = float(
                (sc[0, sample_idx].max(-1).values
                 * factor_strengths0[0].max(dim=-1).values.max()).mean())
            for tag, vec in (("in", apply_input), ("out", operated)):
                v_s = F.normalize(vec[0, sample_idx], dim=-1, eps=1e-9)
                cos_v = v_s @ v_s.t()
                t[prefix + f"code_overlap_{tag}"] = float(cos_v[off].mean())
        rms_in = apply_input.pow(2).mean(-1).sqrt()
        rms_out = operated.pow(2).mean(-1).sqrt()
        t[prefix + "apply_norm_ratio"] = \
            float(((rms_out / rms_in.clamp_min(1e-9)) * mask).sum() / n_real)
        rot = F.cosine_similarity(apply_input, operated, dim=-1)
        t[prefix + "apply_rotation_cos"] = float((rot * mask).sum() / n_real)

    # ── ABSORPTION (chunk-end; shared verbatim by chunkwise and reference) ─────
    def _absorb(self, layer_idx: int, factor_dirs: Tensor, factor_strengths: Tensor,
                coact: Tensor, surprise_chunk: Tensor,
                tele_prefix: Optional[str] = None) -> tuple[Tensor, Tensor, Tensor]:
        """One conserving consolidation pass. factor_dirs [batch, n_nodes, n_slots,
        d_code] raw, factor_strengths [batch, n_nodes, n_slots], coact [batch,
        n_nodes, n_nodes] (row i = absorber/fires-now, col j = donor/fired-
        earlier), surprise_chunk [batch]. Returns (dirs', strengths', flux)."""
        batch_size, n_nodes, n_slots, d_code = factor_dirs.shape
        eps = 1e-6
        unit_dirs = _unit(factor_dirs)
        occupancy = factor_strengths.sum(-1) / (2.0 * n_slots)               # [B,N]
        # row-normalized lagged coactivation: "of everything that co-fired with me
        # lately, what fraction was you?" — scale-free (this IS the squash).
        # Diagonal zeroed BEFORE normalizing: self-coactivation in the denominator
        # would suppress every transfer gate, and MORE as routing sharpens —
        # backwards pressure on the write (review finding, 2026-06-12).
        diagonal = torch.eye(n_nodes, device=factor_dirs.device, dtype=torch.bool)
        coact_offdiag = coact.masked_fill(diagonal, 0.0)
        if self.config.absorb_gate in ("npmi_sharp", "npmi_raw"):
            # NPMI vs independence: positive part = co-firing ABOVE what the two
            # nodes' marginal traffic predicts — the doc-specific signal; the
            # hot-node/template component cancels by construction.
            total = coact_offdiag.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
            p_joint = coact_offdiag / total
            p_row = p_joint.sum(-1, keepdim=True).clamp_min(1e-12)
            p_col = p_joint.sum(-2, keepdim=True).clamp_min(1e-12)
            pmi = p_joint.clamp_min(1e-12).log() - (p_row * p_col).log()
            npmi = (pmi / p_joint.clamp_min(1e-12).log().neg().clamp_min(1e-12))
            affinity = npmi.clamp(0.0, 1.0).masked_fill(diagonal, 0.0)
            if self.config.absorb_gate == "npmi_raw":
                # UNNORMALIZED (run-5 finding): row-normalized gates fix each
                # absorber's total intake to a constant — an identity that makes
                # net relocation doc-INVARIANT under conservation. Raw affinity
                # lets intake totals vary with the doc's above-chance structure:
                # the doc-specific channel the normalized forms destroyed.
                coact_rownorm = affinity
            else:
                temp = self.log_donor_temp[self._active_idx(layer_idx)] \
                    .clamp(math.log(1e-3), math.log(10.0)).exp()
                # donor-softmax: concentrate each absorber's intake on its top
                # above-chance partners (real mass moves; directions rotate)
                coact_rownorm = torch.softmax(
                    affinity.masked_fill(diagonal, -1e4) / temp, dim=-1)
                coact_rownorm = coact_rownorm.masked_fill(diagonal, 0.0)
        else:
            coact_rownorm = coact_offdiag / coact_offdiag.sum(-1, keepdim=True).clamp_min(eps)
        unit_keys = _unit(self.node_keys[layer_idx].float())
        key_cos = (unit_keys @ unit_keys.t()).unsqueeze(0).expand(batch_size, -1, -1)
        grammar = self.plasticity_mlps[str(layer_idx)](
            key_cos, coact_rownorm, coact_rownorm.transpose(1, 2),
            occupancy.unsqueeze(-1).expand(-1, -1, n_nodes),
            occupancy.unsqueeze(1).expand(-1, n_nodes, -1),
            surprise_chunk.view(batch_size, 1, 1).expand(-1, n_nodes, n_nodes))
        absorb_gate = (self._absorb_strength(layer_idx) * coact_rownorm
                       * surprise_chunk.view(batch_size, 1, 1) * grammar
                       / (1.0 + self.config.plasticity_eta))                 # in [0,1)
        absorb_gate = absorb_gate.masked_fill(diagonal, 0.0)
        # per-donor normalization so a donor's total outgoing rate <= 1 (soft —
        # only rescales when already over): rate_ij = gate_ij / max(1, sum_i gate)
        transfer_rate = absorb_gate / absorb_gate.sum(1, keepdim=True).clamp_min(1.0)
        # landing weights: donor (j,s) factor into absorber i's slots (match+room
        # against the absorber's CURRENT slots, frozen for this simultaneous pass)
        cos = torch.einsum("bjsd,bind->bjsin", unit_dirs, unit_dirs)
        match = cos * math.sqrt(d_code) * (factor_strengths / 2.0).unsqueeze(1).unsqueeze(1)
        room_term = ((2.0 - factor_strengths) / 2.0).unsqueeze(1).unsqueeze(1)
        landing_weights = torch.softmax(
            self.match_weight[self._active_idx(layer_idx)] * match
            + self.room_weight[self._active_idx(layer_idx)] * room_term, dim=-1)               # over absorber slots
        # pre-room incoming mass per absorber slot:
        #   outgoing[b,j,s,i] = strength_js * rate_ij
        outgoing = factor_strengths.unsqueeze(-1) * transfer_rate.transpose(1, 2).unsqueeze(2)
        incoming_mass = torch.einsum("bjsi,bjsin->bin", outgoing, landing_weights)
        # continuous room-fractional landing: dS = (2-S)/2 dm  =>  landed =
        # (2-S0)(1-exp(-m/2)). Bounded by room; full slots REFUSE; smooth.
        slot_room = 2.0 - factor_strengths
        landed = slot_room * (1.0 - torch.exp(-incoming_mass / 2.0))
        landed_fraction = landed / incoming_mass.clamp_min(eps)
        # exact attribution back to donors: debit == what actually landed
        debit = torch.einsum("bjsi,bjsin,bin->bjs", outgoing, landing_weights, landed_fraction)
        strengths_new = factor_strengths - debit + landed                    # conservation exact
        incoming_dirs = torch.einsum("bjsi,bjsin,bin,bjsd->bind",
                                     outgoing, landing_weights, landed_fraction, unit_dirs)
        # mass-weighted blend of existing content (RAW — so landed=0 is an EXACT
        # no-op, e.g. all-pad chunks; directions are unit-on-use so raw scale is
        # semantically free) and incoming unit directions. Pure donors keep their
        # direction untouched; only their strength was debited.
        dirs_new = ((factor_strengths.unsqueeze(-1) * factor_dirs + incoming_dirs)
                    / (factor_strengths + landed).clamp_min(eps).unsqueeze(-1))
        if self.config.dirs_blend == "gated":
            # scale-free per-slot blend rate: the top-landing slot rotates at
            # sigma(rate) toward its incoming mix, others proportionally less.
            landed_rel = landed / landed.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
            blend = (torch.sigmoid(self.dirs_blend_logit[self._active_idx(layer_idx)])
                     * landed_rel).unsqueeze(-1)
            incoming_unit = _unit(incoming_dirs)
            has_incoming = (landed > eps).unsqueeze(-1).float()
            dirs_new = ((1.0 - blend * has_incoming) * factor_dirs
                        + blend * has_incoming * incoming_unit)
        if tele_prefix is not None:
            with torch.no_grad():
                t = self._tele
                off = ~diagonal
                t[tele_prefix + "absorb_gate_mean"] = float(absorb_gate[:, off].mean())
                t[tele_prefix + "absorb_gate_max"] = float(absorb_gate.max())
                t[tele_prefix + "grammar_dev"] = float((grammar - 1.0).abs()[:, off].mean())
                inc_total = incoming_mass.sum()
                t[tele_prefix + "absorb_flux"] = float(landed.sum() / batch_size)
                t[tele_prefix + "absorb_refusal"] = \
                    float(1.0 - landed.sum() / inc_total.clamp_min(eps))
        return dirs_new, strengths_new, landed.sum(dim=(1, 2))

    # ── one chunk, one layer: frozen-state semantics ───────────────────────────
    def _layer_chunk(self, layer_idx: int, routing_input: Tensor, apply_input: Tensor,
                     surprise: Tensor, mask: Tensor, factor_dirs0: Tensor,
                     factor_strengths0: Tensor, coact0: Tensor, trace0: Tensor,
                     reference: bool, chunk_start: int = 0):
        """routing_input: [B,C,d_model] (layer 0) or the operated codes (else).
        apply_input [B,C,d_code] unit-scale masked codes. Returns (operated codes,
        dirs', strengths', coact', trace')."""
        config = self.config
        batch_size, chunk_len = apply_input.shape[:2]
        n_slots = config.slots[layer_idx]
        routing_scores = self.route(layer_idx, routing_input) * mask.unsqueeze(-1)

        # APPLY against chunk-start state (the one approximation: in-chunk staleness)
        applier = self.apply_chain_reference if reference else self.apply_chain
        operated = applier(apply_input, factor_dirs0, factor_strengths0, routing_scores)

        # DEPOSIT (token writes). Arm A: every layer. Arm B: layers >= 1 (the
        # alphabet is never written). Arm C: NOWHERE — absorption is the write.
        deposits_here = (config.arm == "A") or (config.arm == "B" and layer_idx > 0)
        if deposits_here:
            proposal_dir = self.dir_projs[layer_idx](apply_input)             # [B,C,d] raw
            proposal_dir_unit = _unit(proposal_dir)
            proposal_strength = 2.0 * torch.sigmoid(
                self.strength_heads[layer_idx](apply_input).squeeze(-1))      # [B,C] in (0,2)
            occupancy0 = (factor_strengths0.sum(-1) / (2.0 * n_slots)).unsqueeze(1)
            gate_mult = self.gate_mlps[layer_idx](
                routing_scores, surprise.unsqueeze(-1), occupancy0)
            deposit_rate_node = (self._write_strength(layer_idx) * routing_scores
                                 * surprise.unsqueeze(-1) * gate_mult)        # [B,C,N] in [0,1)
            slot_weights = self._slot_weights(
                layer_idx, proposal_dir_unit, _unit(factor_dirs0), factor_strengths0)
            deposit_rate = deposit_rate_node.unsqueeze(-1) * slot_weights     # [B,C,N,S]
            if reference:
                dirs_step, strengths_step = factor_dirs0, factor_strengths0
                for t in range(chunk_len):
                    rate_t = deposit_rate[:, t].unsqueeze(-1)                 # [B,N,S,1]
                    dirs_step = (1 - rate_t) * dirs_step \
                        + rate_t * proposal_dir_unit[:, t].view(batch_size, 1, 1, -1)
                    strengths_step = (1 - rate_t.squeeze(-1)) * strengths_step \
                        + rate_t.squeeze(-1) * proposal_strength[:, t].view(batch_size, 1, 1)
                factor_dirs1, factor_strengths1 = dirs_step, strengths_step
            else:
                # EXACT sequential-lerp closed form (the v8 suffix-logsum machinery;
                # NOT a gate sum — a sum silently scales with chunk size):
                #   final = prod_t(1-u_t) * init + sum_t [prod_{t'>t}(1-u_t')] u_t target_t
                log_decay = torch.log1p(-deposit_rate.clamp_max(1 - 1e-6))    # [B,C,N,S]
                suffix = log_decay.flip(1).cumsum(1).flip(1) - log_decay
                online_weights = suffix.exp() * deposit_rate
                survive = log_decay.sum(1).exp()                              # [B,N,S]
                factor_strengths1 = (survive * factor_strengths0
                                     + torch.einsum("bcns,bc->bns",
                                                    online_weights, proposal_strength))
                factor_dirs1 = (survive.unsqueeze(-1) * factor_dirs0
                                + torch.einsum("bcns,bcd->bnsd",
                                               online_weights, proposal_dir_unit))
        else:
            factor_dirs1, factor_strengths1 = factor_dirs0, factor_strengths0

        # ── TELEMETRY (§8) part 1: routing / apply — BEFORE the inactive-layer
        # early return so layer 0 is covered too ──────────────────────────────
        tele_prefix = (f"c{chunk_start}_L{layer_idx}_"
                       if self.telemetry_enabled else None)
        if tele_prefix is not None:
            self._routing_apply_telemetry(tele_prefix, routing_scores, mask,
                                          apply_input, operated, factor_strengths0)

        # COACT / TRACE (exact closed form — never feeds back within the chunk).
        # Skipped on inactive layers (arms B/C layer 0): the table's only consumer
        # is absorption, which never runs there — pure dead compute otherwise.
        if layer_idx not in self.active_layers:
            return _unit_rms(operated) * mask.unsqueeze(-1), \
                factor_dirs0, factor_strengths0, coact0, trace0
        decay = self._coact_decay(layer_idx)
        if config.surprise_coact:
            # the table sees surprise-weighted firing (news only); apply and
            # deposits saw the raw scores above — selection vs plasticity split
            routing_scores = routing_scores * surprise.unsqueeze(-1)
        position = mask.cumsum(dim=1)                                         # [B,C]
        position_last = position[:, -1]
        if reference:
            coact_step, trace_step = coact0, trace0
            for t in range(chunk_len):
                keep_vec = (mask[:, t] > 0).view(batch_size, 1)
                keep_mat = (mask[:, t] > 0).view(batch_size, 1, 1)
                scores_t = routing_scores[:, t]
                coact_new = decay * coact_step + torch.einsum("bn,bm->bnm", scores_t, trace_step)
                trace_new = decay * trace_step + scores_t
                coact_step = torch.where(keep_mat, coact_new, coact_step)
                trace_step = torch.where(keep_vec, trace_new, trace_step)
            coact1, trace1 = coact_step, trace_step
        else:
            position_gap = (position.unsqueeze(-1) - position.unsqueeze(-2)).clamp_min(0.0)
            lower_tri = torch.tril(torch.ones(chunk_len, chunk_len,
                                              device=apply_input.device, dtype=torch.bool))
            decay_matrix = torch.where(lower_tri, decay ** position_gap,
                                       torch.zeros((), device=apply_input.device))
            decay_from_start = decay ** position
            trace_running = (decay_from_start.unsqueeze(-1) * trace0.unsqueeze(1)
                             + torch.einsum("bts,bsn->btn", decay_matrix, routing_scores))
            trace_previous = torch.cat([trace0.unsqueeze(1), trace_running[:, :-1]], dim=1)
            decay_to_end = (decay ** position_last).view(batch_size, 1)
            decay_tail = decay ** (position_last.view(batch_size, 1) - position)
            fresh = torch.einsum("bcn,bcm->bnm",
                                 decay_tail.unsqueeze(-1) * routing_scores, trace_previous)
            coact1 = decay_to_end.unsqueeze(-1) * coact0 + fresh
            trace1 = (decay_to_end * trace0
                      + (decay_tail.unsqueeze(-1) * routing_scores).sum(dim=1))

        # ABSORB (chunk-end; sees the just-landed writes; conserving transfer).
        # Arm C: this IS the entire write path (binding = the relocation pattern).
        absorb_here = config.absorb_enabled and (config.arm == "A" or layer_idx > 0)
        if absorb_here:
            surprise_chunk = (surprise * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            factor_dirs1, factor_strengths1, _flux = self._absorb(
                layer_idx, factor_dirs1, factor_strengths1, coact1, surprise_chunk,
                tele_prefix=tele_prefix)

        # ── TELEMETRY (§8) part 2: coact asymmetry (active layers only) ───────
        if tele_prefix is not None:
            with torch.no_grad():
                asym = (coact1 - coact1.transpose(1, 2))
                sym = (coact1 + coact1.transpose(1, 2))
                self._tele[tele_prefix + "coact_asymmetry"] = \
                    float(asym.norm() / sym.norm().clamp_min(1e-9))

        # PASS UP: inter-layer unit-RMS (mirrors only shrink -> prevents cascade
        # quieting); pads stay exactly zero.
        operated_up = _unit_rms(operated) * mask.unsqueeze(-1)
        return operated_up, factor_dirs1, factor_strengths1, coact1, trace1

    def _chunk_forward(self, state: dict, llama_hiddens: Tensor, surprise: Tensor,
                       mask: Tensor, reference: bool, chunk_start: int = 0) -> dict:
        """One chunk of tokens, all layers (flow-through: layer l+1's input is
        layer l's OPERATED codes; each layer's own state is chunk-frozen)."""
        batch_size, chunk_len = llama_hiddens.shape[:2]
        # fp32 substrate, FOR REAL: the trainer wraps everything in bf16 autocast;
        # without this guard the inner einsums silently run bf16 (measured as
        # conservation drift) while the READER runs fp32 — the v8 audit's
        # write/read precision asymmetry. Guard the whole chunk computation.
        with torch.autocast(device_type=llama_hiddens.device.type, enabled=False):
            return self._chunk_forward_fp32(state, llama_hiddens, surprise, mask,
                                            reference, chunk_start)

    def _chunk_forward_fp32(self, state: dict, llama_hiddens: Tensor, surprise: Tensor,
                            mask: Tensor, reference: bool, chunk_start: int) -> dict:
        batch_size, chunk_len = llama_hiddens.shape[:2]
        codes = _unit_rms(self.seed_proj(llama_hiddens.float())) * mask.unsqueeze(-1)
        new_dirs = list(state["factor_dirs"])
        new_strengths = list(state["factor_strengths"])
        new_coact, new_trace = list(state["coact"]), list(state["trace"])
        for layer_idx in range(self.depth):
            routing_input = llama_hiddens.float() if layer_idx == 0 else codes
            codes, dirs1, strengths1, coact1, trace1 = self._layer_chunk(
                layer_idx, routing_input, codes, surprise, mask,
                state["factor_dirs"][layer_idx], state["factor_strengths"][layer_idx],
                state["coact"][layer_idx], state["trace"][layer_idx], reference,
                chunk_start)
            new_dirs[layer_idx], new_strengths[layer_idx] = dirs1, strengths1
            new_coact[layer_idx], new_trace[layer_idx] = coact1, trace1
        state["factor_dirs"], state["factor_strengths"] = new_dirs, new_strengths
        state["coact"], state["trace"] = new_coact, new_trace
        return state

    # ── full streaming pass (gradient-checkpointed chunks; full BPTT) ──────────
    def forward(self, token_hiddens: Tensor, token_surprises: Tensor,
                token_mask: Optional[Tensor] = None, state: Optional[dict] = None,
                reference: bool = False) -> dict:
        """token_hiddens [batch, seq, d_model] — ONE mid-stack Llama tap
        (flow-through: higher layers see re-described codes, never Llama).
        token_surprises [batch, seq] in (0,1); token_mask [batch, seq].
        Full BPTT across chunks (checkpointed)."""
        batch_size, seq_len = token_hiddens.shape[:2]
        if state is None:
            state = self.init_state(batch_size, token_hiddens.device)
        if token_mask is None:
            token_mask = torch.ones(batch_size, seq_len, device=token_hiddens.device)
        token_mask = token_mask.float()
        real_any = token_mask.bool().any(dim=0)
        nonzero = torch.nonzero(real_any, as_tuple=False)
        seq_eff = int(nonzero[-1].item()) + 1 if nonzero.numel() > 0 else 0

        if self.telemetry_enabled:
            self._tele = {}
        chunk = max(1, int(self.config.chunk))
        use_ckpt = self.training and torch.is_grad_enabled()

        start = 0
        while start < seq_eff:
            stop = min(start + chunk, seq_eff)
            hiddens_chunk = token_hiddens[:, start:stop].float()
            surprise_chunk = token_surprises[:, start:stop].float()
            mask_chunk = token_mask[:, start:stop]

            def run_chunk(*flat, _h=hiddens_chunk, _s=surprise_chunk, _m=mask_chunk,
                          _cs=start):
                chunk_state = self._unflatten_state(flat)
                chunk_state = self._chunk_forward(chunk_state, _h, _s, _m, reference, _cs)
                return self._flatten_state(chunk_state)

            flat_in = self._flatten_state(state)
            flat_out = (torch.utils.checkpoint.checkpoint(run_chunk, *flat_in,
                                                          use_reentrant=False)
                        if use_ckpt else run_chunk(*flat_in))
            state = self._unflatten_state(flat_out)
            state["step"] = stop          # keep step truthful mid-loop (review finding)
            start = stop

        state["step"] = seq_eff
        with torch.no_grad():
            telemetry = {}
            for layer_idx in range(self.depth):
                strengths = state["factor_strengths"][layer_idx]
                telemetry[f"strength_budget_L{layer_idx}"] = strengths.sum() / batch_size
                telemetry[f"slot_occupancy_L{layer_idx}"] = (strengths > 0.05).float().mean()
                telemetry[f"coact_mass_L{layer_idx}"] = state["coact"][layer_idx].sum() / batch_size
            if self.telemetry_enabled and self._tele:
                # mean over chunks: strip the idempotency key "c<start>_"
                grouped: dict = {}
                for key, value in self._tele.items():
                    grouped.setdefault(key.split("_", 1)[1], []).append(value)
                for name, values in grouped.items():
                    telemetry[name] = sum(values) / len(values)
            state["telemetry"] = telemetry
        return state

    # ── the read-side state pack (REAL/SHUF roll this on dim 0) ────────────────
    def pack_state(self, state: dict) -> Tensor:
        """[batch, n_layers, n_nodes_max, n_slots_max*(d_code+1)] — per layer
        cat(dirs.flatten, strengths), zero-padded (padding = strength-0 =
        identity = exactly harmless)."""
        config = self.config
        batch_size = state["factor_strengths"][0].shape[0]
        n_nodes_max, n_slots_max = max(config.nodes), max(config.slots)
        width = n_slots_max * (config.d_code + 1)
        packed = state["factor_strengths"][0].new_zeros(
            batch_size, self.depth, n_nodes_max, width)
        for layer_idx, (n_nodes, n_slots) in enumerate(zip(config.nodes, config.slots)):
            flat = torch.cat(
                [state["factor_dirs"][layer_idx].reshape(batch_size, n_nodes,
                                                         n_slots * config.d_code),
                 state["factor_strengths"][layer_idx]], dim=-1)
            packed[:, layer_idx, :n_nodes, :n_slots * (config.d_code + 1)] = flat
        return packed

    def unpack_layer(self, packed: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        """packed [batch, n_layers, n_nodes_max, width] -> (factor_dirs
        [batch, n_nodes, n_slots, d_code], factor_strengths [batch, n_nodes, n_slots])."""
        config = self.config
        n_nodes, n_slots = config.nodes[layer_idx], config.slots[layer_idx]
        batch_size = packed.shape[0]
        flat = packed[:, layer_idx, :n_nodes, :n_slots * (config.d_code + 1)]
        factor_dirs = flat[..., :n_slots * config.d_code].reshape(
            batch_size, n_nodes, n_slots, config.d_code)
        factor_strengths = flat[..., n_slots * config.d_code:]
        return factor_dirs, factor_strengths
