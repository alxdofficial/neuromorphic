# Vocabulary-Trajectory Memory — alternative architecture

**Status:** Idea / design doc (2026-05-15). NOT planned for implementation
yet; captured here so the design survives context resets and so
implementation can begin from a complete spec when triggered.

**Sibling docs:** [`design_free_graph.md`](design_free_graph.md) (the
dynamic-N-allocation alternative) and
[`plan_trajectory_memory.md`](plan_trajectory_memory.md) (the current
fixed-N architecture).

**Trigger to revisit:** if the current architecture with entry+per-hop
contrastive plateaus on real (non-synthetic) data due to capacity
overflow OR the dilution failure mode (writes blurring node states beyond
useful recovery). Currently the contrastive run is at val=1.50 (vs
full-context 1.40), so neither trigger has fired yet.

---

## Motivation — what the current architecture got wrong

The current fixed-N=4096 architecture fuses two responsibilities into
node state:

1. **Vocabulary identity** (`concept_ids`): what abstract concept this
   node represents. Static after pretraining.
2. **Per-instance content** (`current_states`): mutable state accumulated
   from writes that passed through.

This fusion creates the **dilution problem**: writing about Carl deposits
Carl-content into specific cells, then writing about Alice partially
overwrites those cells, etc. Cells become incoherent mixtures over time.
The current SimVQ + load-balance + revival machinery is largely a set of
patches around this fundamental tension.

The original intent of the architecture (per project memory) was that
nodes serve as an **abstract vocabulary** — discrete concepts that can
be re-used compositionally, like words in a language. Using a word
doesn't dilute its meaning. The current architecture violates this
because writes mutate node state, so re-use does dilute.

**The right fix is to remove per-instance content from nodes entirely.**
Nodes become pure vocabulary (frozen embeddings); content lives on
edges, which form a sparse graph encoding "what kinds of trajectories
traverse this transition." Recall becomes generative — start with a cue,
autocomplete a vocabulary-trajectory by walking the graph.

### The second motivation, surfaced by per-task R↔W diagnostics (2026-05-18)

V1.5's per-hop contrastive loss tried to fix read↔write hop alignment by
pulling the read trajectory's per-hop state vector toward the write
trajectory's per-hop state vector for the target fact (InfoNCE,
mean-pooled across J trajectories). The loss value dropped from ~2.0 to
~1.43 during training — but per-task RW overlap on val showed
`rw_target_hop ≈ 0.005`, at or below the random floor of 0.008. **The
contrastive loss optimized, but the routing decisions didn't change.**
Three compounding causes (full detail in design_decisions_narrative.md):

1. **State-vector loss vs cell-decision routing.** The loss compared
   continuous embeddings. The optimizer can satisfy it by moving cell
   embeddings together, not by changing which cell the routing picks.
2. **Mean-pool over J trajectories** before the InfoNCE wipes out the
   per-trajectory signal — 4 wandering paths whose mean happens to align
   can satisfy the loss.
3. **Compounding neighbor constraint.** Hop routing is restricted to the
   current cell's K_max=32 outgoing edges. If hop k-1 picked the wrong
   cell, the target's hop k cell isn't even *reachable* — the loss can
   pull the step query in the right direction but argmax over a wrong
   neighborhood picks whatever IS available.

V2 addresses this **structurally**, not through more elaborate loss:
- Routing signal lives in `edge_state` (mutable, EMA-updated by writes),
  so the per-hop routing decision is supervised by the topology that
  writes built rather than a separate contrastive objective.
- W-TinyLFU eviction + plasticity refresh let the graph topology *adapt*
  during training. An edge that's needed for the target trajectory but
  doesn't exist yet can come into being.
- Hopfield-tied EntryProjector makes entry alignment automatic by
  construction (read and write share entry weights), eliminating the
  need for an entry-contrastive loss for that supervision.

The empirical validation came after V2 landed: V2.13's `rw_target_hop`
on boxes = **0.92**, on revisions = 0.55; overall target_lift +0.061
(reads task-specific). V2's routing genuinely walks the target's
trajectory **without any per-hop contrastive loss at all** — the
structural signal worked where V1.5's loss-based supervision failed.

### The compromise

V2 trades one prior for another. V1.5 committed to a **fixed graph
topology** (small-world ring, never reshapes) as part of the grammar
thesis — "the graph IS the learned grammar." Routing was supervised by
loss because the topology was a fixed scaffold. V2 lets the topology
adapt through edge eviction and NPMI plasticity — easier to train, but a
less strong commitment to "there is a fixed grammar." The grammar in V2
is *learned topology* + *learned vocabulary*, while V1.5 was *learned
vocabulary* over a *fixed grammar*. Both are graph-manifold designs;
V2's degrees of freedom are higher, and that's why the routing learned.

There is a thesis cost. The original grammar argument said the COMPOSITION
of fixed cells (vocab) via fixed edges (grammar) yields combinatorial
capacity. V2 keeps the cells fixed-ish (vocabulary anchors) but lets the
edges restructure. That makes V2 closer to "structured attention with
eviction" than to "graph grammar," and the architecture loses some of its
distinction from learned attention banks. Whether that distinction is
worth defending (against flat-bank baselines that already work via
continuous attention) is the open question — and per-task RW overlap on
V2 says the trajectory routing IS doing something different (boxes 0.92
hop overlap is not what attention does), so the bet is still live.

---

## Core thesis

**Memory is a graph of trajectories, not a bank of mutable cells.**

- **Nodes** = abstract vocabulary embeddings (frozen). Used over and
  over without dilution, the way "word" embeddings work in any LM.
- **Trajectories** = sentences in vocabulary space. A specific memory
  is a sequence of vocabulary nodes connected by edges.
- **Edges** = the storage substrate. Each edge carries a D-dim state
  vector that accumulates "what trajectories have used this transition"
  via EMA over writes.
- **Reads** = generative autocomplete. Start with a cue (the question);
  walk the graph following edge resonance and vocabulary affinity.

The vocabulary is fixed-capacity (N≈4096-8192). The edge buffer is
bounded by per-node fan-out (K_max≈16-32 outgoing edges). Memory
capacity scales with the combinatorial structure of edge configurations,
not with raw slot count.

---

## Locked-in design decisions

| Decision | Choice | Why |
|---|---|---|
| Node role | Vocabulary embeddings | Reusable, never diluted |
| Node state | None — only the embeddings exist (no `current_states`). Embeddings remain LEARNABLE throughout training | Removes the dilution problem. Learnability lets vocab adapt to task; SimVQ on `id_basis` prevents collapse |
| Content storage | Per-edge D-dim state vectors, EMA-updated | Many trajectories can share an edge via superposition |
| Edge storage | Sparse buffer, bounded fan-out K_max per node | Memory-affordable at meaningful N and D |
| Edge lifecycle | Allocate on traversal (writes only); evict when fan-out exceeded | Demand-driven, simple |
| Eviction policy | Rule-based 4-feature score, normalized, pure delete | No learned controllers, debuggable |
| Specificity protection | Surprise-weighted accumulator + protection floors | Rare-but-important edges survive |
| Source of writes | Constrained to current node (sequential walk) | Preserves sentence framing |
| Source of reads | Same constraint; no allocation; no fallback | Reads always traverse to argmax |
| Trajectory generation | Joint plan + sequential resolve | Plan-aware queries, graph-respecting walk |
| Per-step query | Cross-attn over window + history-attn over walk + step_mlp | Same machinery as current architecture |
| Edge score formulation | Zero-centered cosine (RMS-norm both sides) | Existing edges must earn their bias; non-existent edges aren't penalized just for being absent |
| Llama injection | Trajectory of vocab embeddings → mem_inject | Same machinery as today, different input |
| Read-write alignment | InfoNCE contrastive on pooled inputs (entry-level) | Carried over from current architecture |
| Per-step alignment | InfoNCE on step_queries between matched read/write | Replaces per-hop trajectory-state contrastive |
| Hebbian | `visit_count` per edge — the only active Hebbian signal | NPMI cell-pair tracker removed; visit_count is the per-edge version |
| Surprise | Feature input to `signature_fn` + telemetry | Not a behavior gate in v1 |

---

## What this architecture explicitly avoids

- **Mutable per-cell content** — the dilution problem is gone by construction
- **Routing collapse** — no learned routing function in the failure-mode sense (only edge availability + vocab affinity)
- **Learnable plasticity controllers** — no MLPs in the eviction/allocation control loop
- **Free-graph dynamic-N machinery** — N is fixed; edges are dynamic instead. Simpler.
- **TPR/HRR binding ops** — state superposition via simple EMA is enough
- **Separate NPMI co-activation tracker** — visit_count is the Hebbian signal
- **Surprise-gated writes** — write at every window; surprise just colors the signature

---

## Walker algorithm (the heart of the architecture)

At each 256-token window (passage during writes, question during reads),
Llama runs a forward pass and produces hidden states. We mean-pool these
to a context vector, and the shared `EntryProjector` produces J query
vectors, one per parallel trajectory. Each query scores against all N
vocabulary embeddings, and the top-1 vocabulary node becomes the
trajectory's starting position. This is the only N-way global lookup
the walker ever does.

From the entry, the walker takes K−1 hops in vocabulary space. At each
hop, the step query is constructed by combining three signals:

1. The **current node's vocabulary embedding** (where we are — frozen).
2. A **history-attention** over the embeddings of all previously
   visited nodes in this trajectory (where we've been).
3. A **cross-attention** over the full window's Llama hiddens (what's
   in front of us in the text).

These three pass through a `step_mlp` that produces a single D-dim
step query for this hop. The query is then scored against:

- **Edge states** of edges OUT of the current node — using zero-centered
  cosine (RMS-norm both step_query and edge_state, take dot product, no
  scale or bias).
- **All vocabulary embeddings** — global affinity fallback.

Score composition:

```
edge_score[next]  = cosine(step_query, edge_state[(current, next)]) for connected next
                 = 0 for unconnected next
vocab_score[next] = step_query @ concept_ids[next].T
combined[next]    = vocab_score[next] + λ · edge_score[next]
```

Note: `edge_score` is **zero-centered cosine**, so connected-but-bad
edges contribute a *penalty* and connected-but-good edges contribute a
*boost*. Unconnected nodes have edge_score=0 (neutral). This is what
makes exploration natural: an unconnected node with high vocab affinity
can beat a connected node with low edge resonance.

We always traverse to `argmax(combined)`. No "stay-at-current" fallback;
the trajectory always has exactly K steps. For writes, we additionally
update the edge state (and possibly allocate it if no edge exists yet).
For reads, we just record the visit.

After K hops, the trajectory has K vocabulary node IDs and K
corresponding embeddings. For reads, the embeddings get aggregated and
injected into Llama via `mem_inject`. For writes, the trajectory's main
output is the *side effect* on edge states.

---

## Edge memory mechanics

### Allocation

When a write walker steps from current=A to next=B and no edge (A, B)
exists in A's outgoing-slot list, the system allocates a new edge slot
initialized with the signature for this write.

If A already has K_max outgoing edges when this happens, the system
first evicts one (described below) to make room.

If, after eviction, no slot can be safely evicted (e.g., all are
protected), the walker still traverses to B (the routing decision
stands) but no edge is allocated this round. The next time the walker
traverses (A, B), it'll try again.

### Edge state update (writes only)

When the walker steps from A to B and edge (A, B) exists:

```python
delta = (signature - edge_state[A, B]).norm()        # how much info the write adds
α     = max(α_base / (1 + log(1 + visit_count)),     # visit-count-stabilized
            α_min)                                    # floored to prevent silent freeze
edge_state[A, B] ← (1 - α) · edge_state[A, B] + α · signature
visit_count[A, B] += 1
last_visit_step[A, B] = step_count
specificity[A, B] = (0.99 · specificity[A, B] + delta).clip(0, SPEC_REF)
```

The **α_min floor** is critical: prevents heavily-used edges from
becoming fully frozen. Without it, an edge with visit_count=10000 has
α ≈ α_base / 9.2, which for small α_base can be vanishingly small.
Setting α_min = 0.01 guarantees every edge keeps learning at least a
little per traversal. Add telemetry on `min(α)` to catch the
"everything's hit the floor" case.

### The `signature_fn`

The edge signature is NOT the routing query directly — that would
conflate "what we were looking for" with "what we found." Instead, a
small MLP takes the routing query, the window context, and the surprise
scalar:

```python
signature = signature_fn(concat([step_query, window_pooled, surprise]))
# signature_fn: nn.Linear(2D + 1, D) → GELU → nn.Linear(D, D)
```

This captures both:
- the *query intent* (what we were looking for at this hop)
- the *context* (what was in the surrounding window)

Future read-time queries that resemble step_query in similar contexts
will resonate with the stored signature.

### Eviction

Triggered when allocating a new edge would exceed K_max for the source.
The eviction policy is **rule-based, normalized**:

```python
visit_term  = clip(1 / (visit_count + 1), 0, 1)              # rare = evictable
stale_term  = clip((step_count - last_visit) / HORIZON, 0, 1) # old = evictable
norm_term   = clip(1 - edge_state_norm, 0, 1)                # weak signal = evictable
spec_term   = clip(specificity / SPEC_REF, 0, 1)             # specific = PROTECT

eviction_score = α·visit_term + β·stale_term + γ·norm_term - δ·spec_term
```

All four components are normalized to [0, 1] so the weights α, β, γ, δ
are interpretable and no metric secretly dominates by magnitude.
`SPEC_REF` is a one-time tunable constant (95th percentile of
observed specificity after some training).

**Protection floors** — an edge is unconditionally protected from
eviction if ALL of:
- `(step_count - alloc_step) >= MIN_AGE` (settled into the system)
- `specificity >= MIN_SPEC` (carries non-trivial signal)
- `edge_state_norm >= MIN_NORM` (state isn't degenerate)

The protected set is capped at 30% of K_max per source to prevent
system lockup ("all slots are protected, can't evict anything").

**Pure delete on eviction** — no dilution to neighbors. The deleted
edge's state is zeroed, its slot is returned to the free list, its
metadata fields reset, and it's removed from the source's outgoing-slot
list. The eviction criterion's specificity protection ensures that
deleted edges had weak signal already; their content loss is minimal.

### Edge reinitialization

None needed. When a slot is freed by eviction, its values are
overwritten when reallocated. There's no random reinit phase. The
`alloc_step` is set to the current step on reallocation, giving the new
edge `MIN_AGE` steps of grace.

### Hebbian = visit_count

The current architecture's NPMI co-activation tracker (pairwise
cell-pair statistics) is gone. `visit_count` per-edge is the only
active Hebbian signal: it increments on traversal, drives the EMA
stability (high visits → small α → resists change), and contributes to
eviction protection (high visits → low `visit_term` → harder to evict).

NPMI could be kept as **pure telemetry** for interpretability, but
it doesn't drive any model behavior in v1.

### Surprise = feature, not gate

Per-token surprise from Llama's prediction loss is computed as today
(existing infrastructure). In v1, it's:
- An **input feature** to `signature_fn` (so high-surprise writes look
  different from low-surprise writes in edge state)
- A **telemetry channel** for monitoring

It does NOT gate whether to write. We write at every window
unconditionally. Surprise-gated writes (a v2 refinement) would introduce
a control loop interaction with eviction that's hard to debug; skip in
v1.

---

## Cross-attention conditioning by training mode

The walker's cross-attention target depends on which window of text we
care about, and this varies by mode.

### Wave 1 (retrieval pretraining, current setup)

| Mode | Cross-attn over | Notes |
|---|---|---|
| Write | Current passage's Llama hiddens | One write window per fact; 8 facts per chunk |
| Read | Question's Llama hiddens (via zero-memory question forward) | Same `_compute_question_hiddens` we already have |

Reads need question-conditioning here because the read window IS the
question; there's no other relevant content to cross-attend to.

### Wave 2+/Streaming (general LM training, GRPO, inference)

| Mode | Cross-attn over | Notes |
|---|---|---|
| Write | Current window's Llama hiddens | Writes happen as text streams in |
| Read | Previous window's Llama hiddens | Reads happen at window d+1 conditioned on window d |

In streaming mode, the original Hopfield-tied insight applies: read at
window d+1 conditions on window d's hiddens, write at window d also
conditions on window d's hiddens. They look at the SAME window content,
just at different times — so the contrastive alignment problem is
trivially solved (same input → same edges, by construction).

### Implementation switch

The trainer chooses between modes via a `read_conditioning_hiddens`
arg (already exists). Setting it to `q_hiddens` invokes Wave 1
question-conditioning; setting it to `None` falls back to
`prev_window_hiddens` (streaming).

### Contrastive loss applicability

Wave 1: the contrastive loss is essential because passage and question
hiddens are different distributions. The entry-pool contrastive (read
question pool ↔ write passage pool) bridges them.

Streaming: the contrastive loss is less critical because the input
distributions match by construction (both reads and writes consume the
same passage windows). Could keep it as a cheap regularizer but it
won't be load-bearing.

---

## Data structures (concrete)

```python
class VocabularyManifold(nn.Module):
    # Vocabulary — frozen embedding bank
    concept_ids:        Tensor [N, D]                          # frozen post-pretraining
    
    # Edge buffer — flat layout with active mask
    edge_state:         Tensor [N_slots, D]                    # EMA-accumulated content
    edge_src:           Tensor [N_slots] long                  # source node (or -1)
    edge_dst:           Tensor [N_slots] long                  # destination node
    edge_active:        Tensor [N_slots] bool                  # slot occupancy
    
    # Per-edge metadata
    visit_count:        Tensor [N_slots] int
    last_visit_step:    Tensor [N_slots] int
    alloc_step:         Tensor [N_slots] int
    specificity:        Tensor [N_slots] float
    
    # CSR-style per-source slot indexing
    out_slot_offsets:   Tensor [N+1] int                       # CSR offsets
    out_slot_indices:   Tensor [N_slots_active] int            # slot ids per row
    out_slot_dst:       Tensor [N_slots_active] int            # destination per slot (denormalized)
    
    free_slot_list:     list[int]                              # available slots
```

**Sizing for N=4096, K_max=32, D=1024, bf16:**
- `concept_ids`: 8 MB
- `edge_state`: 256 MB
- `out_slot_*` bookkeeping: ~2 MB
- Per-edge counters: ~2 MB
- **Total: ~270 MB**

This is ~30× the current cell-state buffer but well within GPU budget.

---

## Carryover from current architecture

### What KEEPS

| Component | Why | Files |
|---|---|---|
| Llama backbone, tokenizer | Pretrained LM core, unchanged | `src/pretrained/hosts/` |
| `mem_inject_layer` | Memory → Llama injection mechanism | `src/pretrained/mem_inject_layer.py` |
| `EntryProjector` | Shared entry routing (selects entry vocab node) | `read_module.py:339+` |
| `CrossAttention` / `precompute_kv` | Per-step cross-attn over window | `read_module.py:267+` |
| `per_j_attn` helper | Per-trajectory attention machinery | `read_module.py:202+` |
| `softmax_top1_ste` | Routing STE for discrete selection | `read_module.py:134+` |
| `step_mlp` pattern | Combines current + history + cross-attn → query | both modules |
| `history_attn` | Within-trajectory attention | `read_module.py:445+` |
| `pos_enc` for hop positions | Per-step positional encoding | `read_module.py:444+` |
| `concept_ids` + SimVQ on `id_basis` | Vocabulary embedding bank | `manifold.py:211+` |
| `_compute_question_hiddens` | Zero-memory question forward (Wave 1 only) | `phase1_retrieval.py:390+` |
| Entry contrastive loss | Pool-level alignment between read and write | `phase1_retrieval.py` |
| Load-balance + z-loss | Routing aux losses (apply to edge selection now) | `read_module.py:54+` |
| `Phase1RetrievalTrainer` outer loop | 8-window writes + 1-window QA | `phase1_retrieval.py` |
| Composite data pipeline | Train/val data | `data/wave1/`, `scripts/data/wave1/` |
| Plotting + telemetry infrastructure | Diagnostic tooling | `training/plotting.py` |
| All RW overlap metrics | Edge-level analogues — verify alignment | `_all_rw_overlaps` |

### What MODIFIES

| Component | Change |
|---|---|
| `Manifold` | Remove `current_states`. Add `edge_state` + bookkeeping. Freeze `concept_ids` post-pretraining. Remove `state_basis`/`state_proj` (SimVQ on state_init). |
| `ReadTrajectoryGenerator` | Replace cell-state gathering with edge-traversal walk. Output trajectory of frozen vocab embeddings (not mutated states). |
| `WriteTrajectoryGenerator` | Replace `scatter_mean` + GRU mutation with edge-state EMA updates + allocation. Same walk structure as read. |
| `forward_window` in `integrated_lm` | Plumb edge memory updates; remove `new_states` return (manifold state is now the edge buffer). |
| `_compute_routing_diagnostics` | Replace cell-utilization metrics with edge-level diagnostics (active edge count, eviction rate, age dist, specificity dist). |
| Per-hop contrastive loss | Operate on `step_queries` rather than trajectory states. Same InfoNCE shape, different inputs. |
| `entry_proj` instantiation | Still shared; now selects entry vocab node rather than initial cell. |

### What REMOVES

| Component | Why |
|---|---|
| `current_states` tensor | Nodes have no per-instance state. Frozen `concept_ids` is the only per-node tensor. |
| `state_basis` + `state_proj` (SimVQ on state_init) | No `state_init` — nodes are pure embeddings. |
| GRU mutation in `WriteTrajectoryGenerator` | No state mutation; just edge EMA. |
| `scatter_mean` write logic | Replaced by edge-level EMA. |
| `revive_dead_concepts` | Vocabulary is frozen; concepts don't die. Edges get evicted instead. |
| `usage_ema` per-cell tracking | Replaced by per-edge `visit_count` + `last_visit_step`. |
| NPMI plasticity / co-activation tracker (cell-level) | Edges form via traversal; no separate tracker needed. |
| `record_visits`, `record_coactivation` (cell-level) | Replaced by per-edge updates during traversal. |
| Decay gate in `WriteTrajectoryGenerator` | No state to decay; edge EMA handles its own decay. |
| Most magic-number init values for `current_states` | Buffer no longer exists. |
| Surprise as control signal into write_module | Surprise is now just a feature input to `signature_fn` + telemetry. |
| Routing-collapse fixes specific to cell collapse | Edges have their own (simpler) plasticity. Some pieces (load_balance, z_loss) carry over. |

### Cleanup tasks already on the list

Per tasks #463-465 (in the existing cleanup list), the following dead
code is already flagged and would be removed as part of this rewrite:

- `gumbel_top1_ste` function body + `tau` parameter plumbing (replaced
  by `softmax_top1_ste`)
- `z_loss_logits` legacy arg in `routing_aux_losses`
- `cosine_logit_scale`, `logit_scale_init`, `mutation_init_scale`,
  `gumbel_tau` in `config.py`

---

## Implementation order (when triggered)

The architecture rewrite is substantial. Recommended order, with rough
effort estimates:

1. **Buffer-mask refactor of manifold** (~2 days). Replace
   `current_states` with sparse `edge_state` buffer + CSR-style
   per-source indexing. Add allocation/eviction primitives.
2. **`EdgeMemory` module** (~3-4 days). `lookup_edges_from(src_nodes)`,
   `update_edges_batched(src, dst, signatures)`, eviction routine,
   `signature_fn` MLP. Standalone unit tests.
3. **`VocabularyManifold` with frozen `concept_ids`** (~1 day). Strip
   `state_init`, `state_basis`, `state_proj`. SimVQ remains on
   `id_basis` only.
4. **`ReadTrajectoryGenerator` rewrite** (~3 days). Joint plan +
   sequential resolve. No state mutation. Cross-attn + history-attn
   preserved.
5. **`WriteTrajectoryGenerator` rewrite** (~3 days). Same walker
   structure as read but with edge updates + allocation.
6. **`forward_window` integration** (~1 day). Plumb new return values.
7. **Trainer updates** (~2 days). Loss composition (entry contrastive,
   per-step contrastive on step_queries, load_balance, z_loss),
   metric replacements.
8. **Telemetry** (~1 day). Edge-level metrics: active count, eviction
   rate, age dist, specificity dist, α distribution.
9. **Tripwire monitors** (~0.5 days). Active edge count, allocation
   rate, eviction rate, `min(α)`, average specificity.
10. **Initial training run + comparison** to current architecture
    (~2 days).

**Total estimated effort: ~2-3 weeks for v1 functional**, plus tuning.

---

## Open questions

These are genuinely unresolved and need design decisions before
implementation.

### `concept_ids` learnability — RESOLVED (2026-05-16)

**Decision: random orthogonal initialization, learnable throughout
training, never frozen.** SimVQ reparameterization on `id_basis`
continues to prevent codebook collapse, same as the current
architecture. The vocabulary keeps adapting through Wave 1, Wave 2,
and Wave 3 — no freeze phase. Gradient flows to `concept_ids` through
the EntryProjector (entry routing) and through the zero-centered
edge-score cosine (which compares step queries against
`concept_ids_normed` indirectly).

Rationale: a frozen vocab risks lock-in to a Wave 1 distribution that
doesn't match later waves' data. Keeping it learnable is the safer
default; SimVQ provides the regularization that prevents
collapse-style failure modes.

### Cue evolution during the walk

The current per-step query construction is:
`step_query = step_mlp(current_state, history_attn, cross_attn)`.

In the new architecture, "current_state" is just the frozen
vocabulary embedding of the current node. Does that contain enough
information about the *running cue* (what we're recalling)?

Maybe we need an additional **running cue vector** carried through
the trajectory:

```python
cue[0] = entry_pool
for t in range(1, K):
    step_query = step_mlp(cue[t-1], current_embed, history_attn, cross_attn)
    next_node = argmax(...)
    cue[t] = cue[t-1] + projection(concept_ids[next_node])
```

Adds one extra state vector per trajectory. Probably worth it for the
"running mental model" aspect of autocomplete.

### Llama injection format

The read trajectory produces K vocab embeddings per trajectory × J
trajectories. How do these get injected to Llama?

- **Concatenate as additional context tokens** (K extra tokens
  prepended via mem_inject). Most natural; treats vocab trajectory as
  a "memory prefix."
- **Attend over via separate memory attention layer.** More flexible
  but requires modifying mem_inject.
- **Pool to a single fixed-size vector.** Loses trajectory structure;
  defeats the purpose.

Probably (1). Same machinery as today's mem_inject, just consuming
vocab embeddings instead of mutated cell states.

### Aggregation across J=4 trajectories

The walker produces J=4 parallel trajectories. For Llama injection, do
we:

- **Concatenate all J trajectories** (4K vocab tokens injected) — most
  expressive, more compute.
- **Mean over J** to produce K vocab tokens — loses J-head diversity.
- **Concatenate + learned aggregation** — flexible but adds params.

Probably (1) for v1.

### Cold-start behavior at Wave 1 start

At training step 0, there are no edges. Every walker step is
vocab-score only. The walker creates ~K-1 edges per write window per
trajectory, so ~ (K-1) × J × 8_windows × M_chunks edges per step. With
K=8, J=4, 8 windows, M=8 chunks: 1792 edges per step. The edge buffer
fills very quickly.

Need to verify the eviction policy + protection floors handle this
gracefully (i.e., don't lock up by protecting too many young edges).

Possibly: very early in training (first ~500 steps), relax MIN_AGE to 0
so young edges can be replaced cheaply. Anneal MIN_AGE up as the
manifold matures.

### Edge-score vs vocab-score weight (λ)

The combined score is `vocab_score + λ · edge_score`. Choice of λ
affects whether the model prefers established edges or vocab-affinity:

- **λ = 0**: pure vocab routing. Edges accumulate state but don't
  affect routing. Useless except for eventually reading from them.
- **λ = 1**: balanced.
- **λ → ∞**: edges dominate; new connections suppressed.

Start with `λ = 0.5` constant. Could make learnable scalar. Could
schedule (low at start, higher as edges mature). Pick one before
implementing.

### What's the K_max sweet spot?

K_max controls the graph's branching factor. Trade-offs:

- **K_max = 8** (low): tight memory budget (~64 MB edge state), but
  trajectories have few options at each hop. May force allocation
  pressure too high.
- **K_max = 32** (default): 256 MB edge state. Comfortable.
- **K_max = 128** (high): 1 GB edge state. Lots of options per hop
  but most edges may be redundant.

The right K_max depends on how branchy the data is. Start with 32,
ablate if eviction rate is too high (suggests K_max too small) or
edge_state_norm is uniformly low (suggests K_max too large for the
data).

### Wave 2+/GRPO migration

Wave 1 freezes `concept_ids` after pretraining. Should Wave 2/Wave 3
unfreeze them for further specialization? Or keep them fully frozen
to preserve vocabulary stability?

Keeping them frozen forces Wave 2+ to adapt via edge state only. This
may be too restrictive (e.g., chat data introduces vocabulary that
Wave 1's vocab can't represent well).

Probably want a low-LR fine-tuning of `concept_ids` during Wave 2+,
but it's a tuning question.

---

## Critical assumptions to validate

Before committing to implementation, these need either small
experiments or strong theoretical justification:

1. **K=8 trajectory length is enough.** May need K=16-32 since each
   hop is a "word" in a sentence, not a relational step. Run a small
   ablation: train current architecture with K_read=K_write=16, see
   if final val improves.

2. **N=4096 vocab is enough abstract vocabulary.** Modern LMs use
   32K-128K subword tokens. N=4096 may be too coarse for the
   "vocabulary" to be expressive. Ablate by running current
   architecture with N=8192 or N=16384.

3. **Edge updates have meaningful gradient flow.** EMA with high
   decay (small α) might give very weak gradient. Verify with a
   gradient-norm check during training: `grad_norm[edge_state] /
   grad_norm[other_params]` should be non-trivial.

4. **The autocomplete read can actually retrieve.** Define a small
   synthetic task: write 100 simple sentences ("A is B"), then read
   with "A is" as cue. Verify the read trajectory contains B more
   often than chance. If not, the architecture is fundamentally
   broken; iterate before committing to full Wave 1.

5. **Eviction policy is well-conditioned.** Run a few thousand steps
   with full eviction enabled, monitor: does the system reach a
   stable active_edge_count? Does eviction_rate equilibrate with
   allocation_rate? Do specifically-protected edges actually survive?

6. **Cross-task transferability.** Some tasks may not fit the vocab-
   trajectory framing well (e.g., counting, arithmetic). Check
   per-task val loss; the architecture should be at least as good as
   the current one across all task families, not just on retrieval.

---

## Relationship to other designs

**vs. current fixed-N architecture:** strictly more elegant (no
mutable cells, no dilution) but a significant rewrite. Carries over
~60% of the implementation surface (Llama, mem_inject, EntryProjector,
cross-attn, history-attn, step_mlp, contrastive loss, trainer outer
loop, data pipeline, telemetry).

**vs. free-graph (dynamic N):** simpler. Free-graph allocates and
prunes NODES dynamically; vocabulary-trajectory allocates and prunes
EDGES. Edges are cheaper to manage (no concept_id init, no usage
tracking per node) and the bookkeeping is more local. Vocabulary-
trajectory keeps N fixed and known; free-graph adds an entire
allocation controller.

**vs. RAG / Memorizing Transformers:** RAG uses an external,
non-differentiable retrieval over a database of chunks. Vocabulary-
trajectory uses an internal, fully-differentiable retrieval over a
learned graph. Same "retrieve and condition" pattern, different
substrate. The advantage is that the graph topology is itself learned
from training data; the disadvantage is the higher implementation
complexity.

**vs. continuous-state memory (DeltaNet, Titans):** continuous-state
systems eliminate discrete cells entirely. Vocabulary-trajectory keeps
discrete cells because of (a) interpretability — you can inspect the
visited trajectory, (b) the "vocabulary as reusable concept" framing
matches the original architectural intent, and (c) edges over discrete
nodes give a clear plasticity mechanism. If discrete turns out to be a
mistake, continuous is the obvious pivot.

---

## See also

- [`plan_trajectory_memory.md`](plan_trajectory_memory.md) — current fixed-N architecture
- [`design_free_graph.md`](design_free_graph.md) — the dynamic-N-allocation alternative
- [`architectural_fixes_catalog.md`](architectural_fixes_catalog.md) — fixes / hacks accumulated in the current architecture (many of which become unnecessary in vocabulary-trajectory)
- [[project_capacity_concern]] — earlier note flagging the relational-binding gap that edge-state addresses
- [[research_memory_throughput]] — DeltaNet / Titans / Infini-attention as the continuous-memory alternative
