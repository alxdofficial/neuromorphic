# slotgraph4 — sparse edge-state slot graph (design memo)

*The free-invent graph memory done right: fixed node slots + a fixed-topology edge-state tensor, updated
by a transformer with NO routing head and NO materialization. Differentiable end-to-end; buildable +
trainable now. Read alongside `docs/graph_thesis.md` (why topology collapses) and `docs/OBJECTIVES.md`.*

## 0. TL;DR

slotgraph3's worst pathology was the **routing head** — a hard selection (sparsemax/top-k) that
*materializes* edges into slots: a documented dead-gradient ratchet that also gets absorbed into the
attention's own co-adaptation (Routing-Absorption arXiv:2603.02227 — the mechanistic `routing_diversity≈0.02`).
slotgraph4 **deletes it**. The graph is **N node slots + a FIXED-TOPOLOGY edge-state tensor** (a k-regular
small-world scaffold): every edge slot's *endpoints are fixed by position*, so there is nothing to *choose*
— the model learns only the edge **states** (and the node **assignment**). Topology becomes differentiable
(every existing edge gets gradient every step), and "which entities relate" is realized by the competitive
write **placing related entities onto connected slots**. Edge *states* are vectors (not scalars) so they
carry relation semantics — Relational Attention's "edge vectors read+written each layer" (Diao & Loynd,
ICLR 2023), not a DistMult-symmetric scalar.

**Fixes vs slotgraph3:** routing collapse (no routing head), routing-absorption (graph is a separate
channel), dead-gradient materialization (none), N² blowup (k-regular sparse). **Does NOT fix:**
loss-neutrality — edge states still decay to zero unless the *objective* charges for them (§5).

## 1. State — fixed k-regular small-world topology (coalesced)
- **Nodes** `X : [N, d]` at the LM dimension (d=576). Invented free latents (per-forward init noise for
  symmetry-break). N small (16–32).
- **Edges** `E : [N, k, d_e]` + a fixed neighbor-index table `nbr : [N, k]` — a **degree-preserved
  (k-regular) small-world** graph (ring lattice, rewired while holding degree = k), so every node has
  *exactly* k edges and all shapes are rectangular/coalesced. `d_e` small (~32–64); `E[i,m]` is the state
  of edge i→`nbr[i,m]`, directed/asymmetric — a plain **vector** by default (the r×r keyed-map escalation
  exists only behind the within-edge crowding canary, §3/§6). **The topology is FIXED — never learn which edges exist**
  (that is the routing/selection trap we are deleting). Learn only the edge *states* + the node assignment.
  - *Rationale:* fixed sparse connectivity with good mixing is BigBird (random+window+**global**, provably
    universal) / Exphormer (expander graph transformers); small-world gives short paths (~log N) so a
    relation between non-adjacent nodes is a few hops away, and a couple of **global hub nodes** keep
    long-range cheap. **For small N, keep k high** (≈ N/2 or more — nearly dense); the k-regular scaffold
    is mainly there so the design *scales*, and it keeps compute coalesced.
- **Compute:** at small N, attend **densely over the `N + N·k` write-stack tokens** (cheap, no
  gather/scatter). **The WS topology is NOT an attention mask** — sparsity lives only in *storage*
  (`[N,k]`, not `[N,N]`). Topology influences attention only via **id-incidence**: edge tokens carry
  `[id_src ‖ id_dst]` from the fixed neighbor table, and the shared `tok_proj` puts src/dst ids in the
  same output directions as the corresponding node tokens — so attention naturally learns to compose
  edge tokens with their endpoints by inner product. No adjacency masking in the write stack.

## 2. Tokenization — fully static (no learned selection)
Endpoints are fixed by position, so id/role structure is **known and constant during the forward**:
- node token = `node_proj([ X_i ‖ id_i ‖ role=node ])`
- edge token = `edge_proj([ E[i,m] ‖ id_i ‖ id_{nbr[i,m]} ‖ role=edge ])`

`id_*` = **frozen orthonormal** buffers (never train — the anti-collapse basis + EntNet-style reusable
entity keys); `role` = fixed embedding. There is **no routing weight** — the endpoint ids come from the
fixed neighbor table. Per-block-normalize the concat blocks before projection (so the id isn't drowned by
content). **Do the tokenization/write inner products in fp32** (autocast-off) — the id-incidence match is
numerically delicate. **NO positional encoding on graph tokens** — they're a SET; identity is the id/role
embeddings, and adding RoPE/sinusoidal would impose a spurious slot order. (Word order already lives inside
the frozen-LM text hiddens; leave those alone.)

## 3. Write — frozen-LM hiddens in, a small transformer stack, one boundary write
**Input = the last-layer hiddens of a frozen LM encoder over the 256-token window** (not raw embeds — the
LM has already computed the in-context graph; the memory's job is to *persist + compress* it). Then a stack
of a few (2–6), ideally **shared-weight/recurrent**, custom write layers. One layer =
```
[ fused( graph-self-attention + graph↔text cross-attention ), FFN ]
```
i.e. **graph tokens are the queries; keys/values = [frozen text hiddens ; graph tokens]** — self+cross in
ONE SDPA (slotgraph3's `_SG3Block`). **No token-self-attention** (the frozen LM already contextualized the
text; the text is static — keys/values only, never re-written). Within the stack the state is just the
**residual stream** — no per-layer delta.

**Tricks inherited from slotgraph3 (keep verbatim):**
- **ReZero/Fixup zero-init** the residual output projections (`o` and the FFN's last layer) → each write
  layer is **exact identity at init** and only earns influence as it trains. The load-bearing stability trick.
- **Competitive assignment** — when the graph reads the text, normalize the attention **over the SLOT axis**
  (softmax over nodes/edges) so slots *compete* to claim each token (Slot Attention anti-duplication). This
  is the anti-collapse mechanism; without it slots pool to the mean.
**Propose → commit (the ONE persistent write, window boundary only).** The stack is pure scratch —
activation space; persistent state is an *input*, never mutated in place (functional dataflow, so full
BPTT across windows works exactly like the current streaming path, per-window ckpt included). Persistence
changes at exactly one point:
```
H       = frozen_LM(window)                 # static keys/values
G       = write_stack(tokenize(X, E), H)    # scratch: 2–6 shared-weight layers
ΔX, ΔE  = oX(G_nodes), oE(G_edges)          # ZERO-INIT output projections → proposal is a CHANGE
αX, βX  = gates(G_nodes)                    # gates computed from the write-stack hidden (not [X‖ΔX] — see NOTE)
X       = norm( αX ⊙ X + βX ⊙ ΔX )          # same rule for E with its own gates
```
- **Delta-parameterized proposal:** commit the stack's output as a *change* `Δ` through a zero-init
  projection, so step-0 is a no-op REGARDLESS of the gate. This composes the write-open gate, ReZero,
  and the zero-init write projection into ONE story instead of three tricks: the gate is open from step 1
  (gradient flows immediately — the write-grad-collapse lesson) and the stack only *earns* influence as
  `o*` trains. (Committing an absolute candidate + open gate would interpolate toward garbage at init.)
- **Decoupled erase/write gates** `α, β` (independent content-dependent sigmoids per slot) instead of the
  convex `g / 1−g` tie — writing strongly must not FORCE forgetting strongly (the Gated-DeltaNet-2 lesson,
  minus the delta). **Post-write normalization** (EntNet) restores the magnitude bound that convexity gave.
  This keeps the erase/overwrite semantics streaming needs (a new "Mary" mention *updates* the persistent
  Mary node, never accumulates unboundedly). **Consequence: `‖state‖≡1` after every window** — the
  α-retention is purely *directional* (the direction is retained, magnitude re-pinned), which is the
  correct EntNet behavior for a content-addressed slot.
- **NOTE — gates from `gh`, not `[X‖ΔX]`:** implementing `gates([X‖ΔX])` literally would cause a
  double-zero-init dead-gradient: `ΔX=oX(gh)=0` at init AND `oX.weight=0`, so `∂α/∂gh=∂ΔX/∂gh=0` →
  the write stack receives NO gradient at step 0. Computing gates directly from `gh` (the competitive-read
  slot hidden) with standard (non-zero) weight init keeps `gh` gradient-connected through the retention
  path `α(gh)⊙X` while `ΔX=0` preserves content identity-at-init. Gate weights are **NOT zeroed**; only
  `oX/oE` (the Δ projections) are. The gate still conditions on old-vs-new through the learned
  representation, which is richer than a raw `[X‖ΔX]` concat anyway.
- **Why propose→commit, not per-layer persistent writes:** recurrence depth stays = #windows (not
  windows×layers); scratch can swing large without contaminating persistence; stability–plasticity is
  enforced at ONE bottleneck — which is also the telemetry hook (log `‖Δ‖/‖state‖` + α/β gate means per
  window). GRU / EntNet / RMT-AutoCompressor all commit this way; per-layer persistent writes are the
  NTM/DNC blind-write regime.

**Why NO delta rule (deliberate — settled 2026-07-10).** The delta rule (`S ← S + β(v − S k̂)k̂ᵀ`) earns
its keep only when (a) many items superpose in a SHARED substrate and (b) writes are BLIND — a (k,v)
stream injected without reading the state first (the linear-attention regime; Gated DeltaNet's whole
setting). slotgraph4 meets neither. **Addressing is positional:** each edge (i,m) is its own tensor slot —
the hard-orthogonal-keys limit of the delta rule — so cross-edge interference is structurally zero; and on
a bare vector slot the delta rule *degenerates to gated interp exactly* (no key axis to isolate a
subspace). **Writes are read-modify-write:** the candidate is computed by a transformer attending over the
old state AND the new text, so old/new *merging* happens in activation space, not in the storage rule —
the "smearing" critique applies to blind interp, and ours isn't blind. (slotgraph3's T3 keyed delta DID
have a job: its per-slot matrix aggregated many neighbors into one shared substrate — a little
linear-attention memory inside each slot. Per-pair edges deleted the shared substrate, so the delta lost
its job.) Facts-per-pair ≈ 1 in our data, and a changed relation SHOULD overwrite (coreference), not
superpose. **Escalation path, not commitment:** if the within-edge crowding canary fires (§6 — readback of
fact A degrades after writing fact B to the SAME pair), promote the edge slot to a small r×r keyed
associative map (key = learned relation-aspect projection of the write content, gated-delta write — the
sg3-T3 machinery already exists; r≈8 keeps N·k·r² inside the 55K-float budget). Until then the capacity
knob is `d_e` + the merger network, not the write rule.
(Optional PairNorm / GCNII initial-residual if the eff-rank canary shows over-smoothing — architectural, not a loss.)

## 4. Read — prepend + bidirectional memory attention (NOT per-layer KV)
The graph can't fit all `N + N·k` tokens into M=96, so compress:
- **Node-centric tokens:** each node → 1 (or few) vectors = attention-pool of `[ X_i ‖ its k edge states ]`.
- **Top-k explicit edge tokens:** a few budget tokens on the strongest edges (by a **learned scalar salience
  head** `edge_sal = Linear(d,1)` applied to the lifted edge state `E↑`), soft-gated by `sigmoid(score)`,
  never sparsemax — carrying `id_i, id_{nbr}` pointers so the sharpest relations survive intact. NOTE: the
  obvious alternative `‖E[i,m]‖` is **degenerate** here because the EntNet post-write norm (`_NormMatch`) 
  L2-normalizes every committed edge state to unit norm after each window — so `‖E[i,m]‖ ≡ 1` for all edges
  and norm-ranking would be all-ties. The learned salience head is thus the only meaningful per-edge
  ranking criterion.

Present via **prepend + bidirectional memory attention**, at a **set-consistent position** (`uniform_mem_pos`
/ Set-LLM block — do NOT let the LM read a fake left-to-right order over slots). **Per-layer KV is wrong for
this arm:** it injects memory as *keys only*, so memory tokens never attend to each other — the edge↔endpoint
id-incidence composition is impossible in a prefix-KV read. The relational read *requires* intra-memory
bidirectional attention, which only the prepend path provides. (Per-layer KV stays right for beacon/memoryllm,
whose memory is independent KV facts with no intra-memory relations.)

## 5. The load-bearing part is the OBJECTIVE (architecture ≠ binding)
Deleting the routing head removes the *gradient* trap; it does not remove *loss-neutrality* — edge states
decay to zero if nothing charges for them (`docs/graph_thesis.md`). So slotgraph4 must ship with:
- **behavioral-KL** (USE) + **MAE-CE** (high-rank anti-collapse) — the shared backbone.
- **The graph as the EXCLUSIVE read channel** (NRI): the decoder reads *only* the memory, no parallel
  content path to absorb the structure; make its loss-effect large (multi-hop / multi-horizon).
- **provenance-InfoNCE** (`docs/OBJECTIVES.md` Rung 2): positives = node/edge tokens written from the target
  span — the direct reward for addressing (free labels in synthetic data).
- **bypass-gap** (Larimar): `λ·relu(CE_memory − sg(CE_no_memory))` so memory must beat the no-memory floor.

## 6. The decisive experiment
With the routing head **gone**, slotgraph4 is the clean test of the thesis. Canaries:
**ID-subtracted content-only edge-state effective-rank** (does structure stay high-rank?); **`SHUF−REAL`**
(does it bind?); the **`gate=0` ablation** (are edges decorative?); **write telemetry** (`‖Δ‖/‖state‖` +
α/β gate means per window — is the commit alive, and does α actually erase when content changes?);
**within-edge crowding** (readback of fact A after writing fact B to the same pair — the r×r escalation
trigger, §3). If binding still fails with the gradient
trap removed, the verdict is **objective-bound, not architectural** — the single most valuable thing we can
learn. If it binds, we have the free-invent graph memory slotgraph3 was trying to be.

## 7. Anti-pattern checklist
| pathology | slotgraph4 |
|---|---|
| routing collapse | **GONE** — no routing head; topology is fixed, states are dense |
| routing absorption (2603.02227) | **AVOIDED** — graph is a separate channel, not a content-softmax bias |
| dead-gradient (sparsemax/hard-topk) | **GONE** in write; read top-k is soft-gated only |
| N² blowup | **GONE** — k-regular sparse `[N,k]`, coalesced |
| additive-write saturation | bounded commit: decoupled α/β gates + post-write norm (erase ≠ write) |
| blind-write smearing (what the delta rule fixes) | **N/A structurally** — positional slots = orthogonal-key limit; read-modify-write candidates (merging in activation space) |
| over-smoothing / rank collapse | full-transformer layers + ReZero + optional PairNorm; edge-effrank canary |
| loss-neutral collapse | **NOT architectural** — needs §5 objectives (the real work) |
| per-edge relation semantics | **carried** by edge vectors (Relational Attention) |
| membership / mean-pool | competitive assignment (softmax over slots) |
| symmetry-breaking | per-forward node init noise; frozen orthonormal ids |
| spurious slot order | NO positional encoding on graph tokens; set-position at read |

## 8. Prior art
Relational Attention (Diao & Loynd, ICLR 2023, 2210.05062 — edge vectors r/w each layer); BigBird (Zaheer
2020, 2007.14062 — random+window+global sparse, universal) / Exphormer (Shirzad 2023 — expander GTs);
Watts-Strogatz small-world (1998); Perceiver IO (Jaegle 2021) / Set Transformer (Lee 2019); Graphormer w/
edge features (Ying 2021, 2106.05234); EntNet (Henaff 2017, 1612.03969 — keyed competitive slots, solved
bAbI); Slot Attention (Locatello 2020, 2006.15055); ReZero (Bachlechner 2020) / Fixup (Zhang 2019);
DeltaNet / Gated DeltaNet (Yang 2024, 2412.06464 — "gating enables rapid memory erasure, the delta rule
enables targeted updates"; also our rationale for when delta is NOT needed: blind writes + shared substrate)
/ Gated DeltaNet-2 (Hatamizadeh 2026 — decouple the erase gate from the write gate); GRU (Cho 2014 —
candidate-then-gate commit); PairNorm/GCNII (over-smoothing).

## 9. Status
Buildable + trainable now (fully differentiable, no RL). **slotgraph4 = the free-invent (fixed-slot,
invented-node) arm done right** — the differentiable substrate. **FurlGraph** = the input-grounded
(chain-merge) arm; the graph-generative memory (`docs/graph_generative_memory.md`) is the score-function/RL
upgrade that sits *on top of* slotgraph4 once the memory is competent. Locked decisions (this session):
k-regular small-world topology (storage-sparse [N,k], compute-dense — no WS attention mask; topology via id-incidence); frozen-LM-hiddens input; write layer = fused[graph-self+cross]+FFN with ReZero + competitive assignment, no token-self-attn; propose→commit — one boundary write per window, delta-parameterized proposal (zero-init `oX/oE`, gates from `gh` with non-zero weight init to avoid double-zero-init wall) + decoupled α/β gates + EntNet post-norm (‖state‖≡1 post-window, retention is directional); NO delta rule (positional addressing + read-modify-write); r×r keyed edges only if the crowding canary fires; no PE on graph tokens + set-position at read; prepend+bidir read; learned salience head for read top-k edge selection (‖E‖ norm-ranking is degenerate under the unit-norm post-write).
