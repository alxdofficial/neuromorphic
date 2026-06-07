# graph_v7 — design doctrine (vocabulary-atoms + co-activation edges)

*Branch: `emat-baselines`. Supersedes the graph_v6 write path. Read interface (prepend) and the ⊙ bind are already landed (commits a0124bc, 89ba4f2). This doc is the spec for the write redesign.*

## 0. Thesis (why this shape)

Memory as a **graph of vocabulary atoms + relational edges**, not a flat bank of "one cell = one memory." Two payoffs:

1. **Localized writes → no catastrophic forgetting.** A flat bank stores facts *in* the cells, so encoding a new fact perturbs shared cell content and disturbs every old fact that leaned on it. Here a new fact = pick two existing atoms + write one edge; the atoms are untouched, the write is localized to one edge slot, every other fact is undisturbed. Interference goes from bank-wide to per-edge.
2. **Compositional capacity.** N flat cells = N rememberable events. N atoms + edges = O(N²) rememberable *relations*, because information lives in the connections, not the cell contents — which is also why the write can be small and local.

Framing (complementary learning systems): **atoms = neocortex** (slow, stable vocabulary), **edges = hippocampus** (fast, episodic, forgettable). The atoms are an addressing/vocabulary basis; specific entities are *sparse codes* over them, and their content rides in from the frozen LLM.

This directly fixes the diagnosed wall: the v6 graph hit `REAL ≈ SHUF` ("pool destroys binding" — averaging is membership-only, PMA theorem). v7 replaces every averaging step with a sharp, masked, content-addressed mechanism + a multiplicative bind.

## 1. Persistent stores

- **Atom bank** — `K_node` *stable, learned* vectors (a codebook). **Not modified by data per-example** (the v6 `node_gate` is removed). Because the atoms can no longer adapt to the current passage, bank *quality is load-bearing*: a collapsed bank → distinct entities map to overlapping atoms → indistinguishable addresses → REAL≈SHUF. Health recipe:
  - **k-means init** over a sample of (projected) frozen-LLM token hiddens — the frozen LLM makes this a *stationary* target, so the codebook aims at a fixed distribution.
  - **decorrelation/orthogonality** pressure + **dead-code revival** (the repo's existing codebook-collapse recipe, `project_repr_codebook_collapse_fix`, SimVQ).
  - **telemetry:** `node_collapse_cos` (↓), `node_active_frac` (↑, spread), per-atom usage histogram.
- **Edge bank** — `K_edge` *mobile* edges, each `(q_src, q_dst, edge_state)`. Persistent across scopes/windows. **No allocation/eviction** — edges *migrate* to new pairs and *update* states; forgetting happens by relocation (an edge leaving an old pair overwrites its state). Fully differentiable (endpoint softmax updates), no discrete ops.

## 2. Three timescales (the scalability lever)

Decouple the *cheap* co-occurrence measurement (fine-grained) from the *expensive* edge update (coarse). Per-token/per-scope TokenGT would be ~256 transform steps over a 4096 input — wasteful; the cheap accumulator buffers between scope-granularity and window-granularity.

| level | size | operation | cost |
|---|---|---|---|
| **scope** | ~16 tok | accumulate `C` + `content` (routing + outer-product + pool) | cheap, no transformer |
| **window** | ~256–512 tok | TokenGT edge update | the expensive step (~8–16×/input) |
| **input end** | ~4096 tok | ⊙ bind → prepend | cheap (bind only) |

(The TokenGT is small — a few hundred tokens, ~5 layers, d≈640 — tiny next to the frozen-Llama forward every arm runs, so per-window is comfortably in baseline compute territory; materialization is cheap so it can also be done per-window for streaming reads.)

## 3. Per scope (cheap, no transformer)

Tokens are **LLM-contextualized** (coreference already resolved by the host — "She" lights up Curie's atoms for free).

1. **Route:** `a_s = sparse_softmax(token · atom_keys)` — which atoms each token lights up (sharp/sparse; mask requires the allowed set to stay small).
2. **Co-activation:** `C ← λ·C + a_s ⊗ a_s`, `C: [B, N, N]`. *Within-scope* outer product → only same-clause atoms get linked. Crucial identity: `Σ_s (a_s⊗a_s) ≠ (Σ_s a_s)⊗(Σ_s a_s)` — the per-scope sum keeps only within-scope co-occurrence; the cross-scope terms (the spurious cross-clause edges) are exactly what's excluded. `λ` = decay (forgetting / bounded memory).
3. **Content:** `content[n] += Σ_t a_t[n]·MLP(token_t)`, `content: [B, N, d_val]`, static shape (un-lit atoms = 0, masked later). The MLP (= the existing `pin_encoder` role) projects token-space → value-space AND extracts the relation-relevant content. Raw token hiddens never reach an edge.

## 4. Per window (the TokenGT edge update)

4. **Tokens in:** atom tokens (all N, static; value = `content[n]`; un-lit atoms masked out of attention via the activation mask) + edge tokens `(q_src, q_dst, state)`. The per-node activation enters as the **attention mask** (which atoms are attendable); an explicit activation *feature* concat is optional and skipped for v1 (content magnitude already carries salience). `C` (the matrix) does **not** enter the transformer.
5. **TokenGT** self-attends over atoms+edges → emits **free** updated `q_src, q_dst, state` per edge. The queries are unconstrained in value (the model learns where to point); all structural safety is in the materialization masks.
6. **Materialize endpoints** (soft blends over *tight* allowed sets = sparse codes):
   - **src:** `p_src = softmax(q_src · atom_keys)` masked to active atoms → `src_ep = p_src @ atom_values`.
   - **dst:** `p_dst = softmax(q_dst · atom_keys + log(p_src @ C + ε))` masked to active atoms, **diagonal excluded** (no self-loops by default) → `dst_ep = p_dst @ atom_values`.
   - `p_src @ C` (`[B,E,N]@[B,N,N]→[B,E,N]`) is the only use of the full matrix — it turns "where is my src" into "which atoms co-fired with my src," enforcing a **genuine pair** (not two unrelated-but-both-active atoms). `p_src` is soft → dst conditions on the whole src distribution → differentiable, no argmax. Sequential (src then dst), two softmaxes, cheap.
7. **Competition:** softmax-over-edge-slots so the K edges claim *distinct* pairs (prevents all edges collapsing onto the hottest pair). Compatible with mobile edges — it's a spread constraint on where they migrate, not eviction. Only active edges move; dormant edges hold (localized write).
8. **Write `edge_state`** from the cross-attended content (the relation). Re-mention policy default = reinforce-with-decay; recency/versioning deferred.

## 5. Per input end (read)

9. Per edge: `fact = LN( (W_src·src_ep) ⊙ (W_dst·dst_ep) ⊙ (1 + W_rel·edge_state) )` — multiplicative **bind** (already implemented in `GraphV6FactBuilder`). `⊙` = Hadamard; role-specific W encode direction; `(1+W_rel·state)` keeps the relation un-ignorable, transparent at zero-init.
10. Project → d_llama + `_NormMatch` → `[B, K_edge, d_llama]` **prepend** memory. Frozen Llama attends; REAL/SHUF/OFF gate applies (already wired).

## 6. Invariants, knobs, risks

- **Free queries, masked materialization** — the queries are free; safety is entirely in the masks (active for src, `p_src @ C` for dst). The soft blend behaves as a sharp sparse code **iff the allowed set is tight**. If activation/scope sprawls (40 atoms through the mask), it reverts to a pool.
- **Load-bearing knobs:** scope size, routing sharpness (peakedness of `token·atom_keys`), node-bank health.
- **Tunables:** decay `λ`, `K_node`, `K_edge`, window size, `d_val`.
- **Known-open policies (deferred):** relation re-mention (reinforce vs recency/versioning); whether to add a question-conditioned read on the prepend path (reserve lever, the dead `GraphV6FactReader.q_proj/attn` can be revived) if the bind alone leaves SHUF flat.

## 7. Relationship to the EMAT roadmap

- **L1 (this doc):** single-layer atoms+edges. Gate = `REAL ≫ SHUF ≫ OFF` on coined-entity closed-book key→value (composite_v1).
- **L2 (later, "earns the hierarchy"):** the hierarchy is *implicit* in a flat atom+edge graph (entities = dense clusters, relations = bridges) — no explicit coarsening/super-node machinery needed; tested on multi-hop.
- **Co-activation prior** is in the core (it provides the dst mask = the sparsity), not a reserve. Hebbian/NPMI edge-topology is the on-thesis differentiator (machinery already exists in the trajectory lineage).

## 8. Build phases

1. **Substrate skeleton** `graph_substrate_v7.py`: config, `AtomBank` (stable, k-means-init hook), routing, the per-scope `C` + `content` accumulator. Smoke: shapes + grad.
2. **Per-window TokenGT update:** tokens-in, free `q` out, masked src/dst materialization (active + `p_src @ C`), competition. Smoke: shapes + grad + endpoint sharpness telemetry.
3. **Readout:** reuse the ⊙ `FactBuilder` → prepend `[B,K_edge,d_llama]`. Smoke vs the existing prepend path.
4. **Encoder variant** `GraphV7BaselineEncoder` (streaming interface init/streaming_write/finalize) + wire into config / `ReprLearningModel.VARIANTS` / `__init__`. Smoke end-to-end (forward+backward, REAL/SHUF computable).
5. **Node-bank health:** k-means init from LLM hiddens + decorrelation + revival + telemetry.
6. **Validate:** short EMAT run, REAL/SHUF/OFF.

## 9. Bundle vs bind — where each belongs (don't "fix" the endpoints into binds)

VSA distinction, and it tells us exactly where the multiplicative ⊙ goes and where it must NOT:

- **Bundling = sum/superposition = representing a SET of the *same kind*.** An entity is a *set of its atoms* (Zorblax = atoms {7,23,91}); `src_ep = p_src @ atom_values` is a *bundle*, and that's the **correct** op — the entity literally *is* the superposition of its atoms, and bundles of sparse high-dim sets stay quasi-orthogonal so distinct entities stay distinct (sharing an atom is fine; they differ on the others). Same for `content[n]` (a bundle of one node's token mentions) and `edge_state` (a bundle of the relational tokens). These are **bundles, not the PMA-pool problem** — the PMA failure is averaging things you must keep *distinct as roles*; bundling a set you *want* superposed is correct.
- **Binding = ⊙ = combining DIFFERENT roles into a structured association.** A fact binds *src ⊗ rel ⊗ dst* — three different roles that must stay recoverable-by-role. That's the readout `fact = LN((W_src·src_ep) ⊙ (W_dst·dst_ep) ⊙ (1+W_rel·state))`, and it's the **only** place ⊙ belongs.

So the bind is already correctly placed (the fact-builder); the endpoint/content "blends" are bundles and must stay sums. **Do not convert the endpoint materialization into a ⊙** — binding an entity's own atoms would destroy the set semantics (and recoverability) and is the wrong operation. Rule of thumb: **bundle within a role (entity, relation), bind across roles (the fact).**

## 10. Gradient-flow discipline (catch dead modules from smoke #1)

The v6 graph partly died from dead gradient paths (inert read, floored ReZero, query-leak). v7 has new masking/product/recurrence paths that can silently kill training. Disciplines:

1. **ε-floor every log-mask:** `log(p_src @ C + ε)`, never `log(0)` → no `-inf`/NaN gradient. Same for any masked softmax (add a large-negative bias, not `-inf` on the only surviving option).
2. **Keep all softmaxes soft** (routing, src/dst endpoints, competition) — *not* hard top-k. Near-miss atoms/edges must get gradient or the selection can never correct itself. (No Gumbel/STE — that was the user's explicit constraint.)
3. **"Hold" dormant edges via a differentiable identity carry, NOT detach.** A window where an edge is inactive must pass its state through unchanged *with grad attached*, so BPTT reaches back to when it was last written. Detaching here = the edge never learns across windows.
4. **`C` is a *detached* mask for v1.** Use `C` only as the dst mask, no gradient through the cross-scope outer-product accumulation (that path is a deep BPTT through `Σ a⊗a` — expensive + unstable). The routing still trains via the **content path** (`a_t[n]` weights the content pool → values → fact → loss) and via the endpoint softmax (`q·atom_keys`, `p@atom_values`). Revisit (make `C` differentiable) only if routing under-trains. Atoms get gradient through endpoint materialization; routing through content; nothing is orphaned.
5. **Watch the ⊙ bind:** a product attenuates gradient to a factor when another factor is near-zero. The `(1+W_rel·state)` form (≈1 at zero-init) + post-`LN` + residual MLP keep it alive (the bind smoke already showed `W_src`/`W_rel` grads flow); keep monitoring under training.
6. **Revival is also gradient hygiene:** an atom/edge never selected gets *zero* gradient and freezes at init forever — dead-code revival (reinit from heavy users) is what unsticks it.
7. **Instrument per-module grad norms in EVERY smoke** (routing, atom bank, TokenGT, fact-builder, edge states) — a near-zero norm on any module = a dead path, caught immediately instead of after a wasted training run. This is the project's grad-norm-probe / debug-in-the-real-bf16-path discipline; it is non-optional for v7.
