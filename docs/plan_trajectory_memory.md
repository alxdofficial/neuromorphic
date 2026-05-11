# Trajectory-Memory LM: design, implementation plan, training notes

**Status:** v1 implementation landed on `main`. Tests + smoke runs pending in a
torch-enabled environment; design notes here remain authoritative for the
architecture (deviations should be documented as "implementation note" or
fixed in code).
**Lineage:** refactor of the per-token `graph_walker` architecture, which
is now archived on the `abandoned/graph-walker` branch. The persistent
walker, neuromodulator, per-token routing, edge weights, and Hebbian
plasticity all go away. Plasticity surface reduces to one channel:
concept-state mutation during writes.

This doc is self-contained — someone picking it up cold should be able to
implement from it without re-reading the design conversation that produced it.

---

## Motivation and goals

**Problem.** Modern LLMs handle within-context information well but
struggle with content beyond their context window. Pushing context
arbitrarily large is expensive (quadratic attention, KV-cache memory
growth, scarce long-context training data) and gives the model no
inductive bias for what to forget. Existing alternatives have known
limits:

- **Vanilla long-context training** — quadratic cost; long-doc data is
  sparse; no compression prior.
- **RAG / external retrieval** — retrieval is decoupled from the LM's
  reasoning; chunks are noisy; not trained end-to-end.
- **Recurrent state models** (RWKV, Mamba, S4) — bounded inference
  memory but the state is dense and opaque; selective retention is
  implicit.
- **`graph_walker`** (this project's prior approach) — per-token walker
  over a concept graph. Suffers from launch overhead, neuromod gradient
  noise, and a per-token plasticity timescale that doesn't match
  meaningful "memory events."

**Goal.** A memory module that:

1. **Learns end-to-end** from NTP loss — no separate retrieval training,
   no privileged supervision, no hand-crafted indexing.
2. **Compresses lossy** — long content survives only as a sparse
   trajectory through a fixed-size manifold. Mirrors human memory:
   gist, not transcript.
3. **Stays constant-footprint at inference** — the manifold doesn't
   grow with session length; long sessions overwrite older state via
   plasticity.
4. **Plugs onto a frozen pretrained LM** — additive to a strong base
   model, not a from-scratch training problem. Llama-3.2-1B as starting
   point; same architecture should generalize to larger.
5. **Has a learnable curriculum** — windows + cross-window TBPTT give
   clear gradient signal to memory even when the LM's effective context
   is intentionally capped.

**Why a manifold of concepts?** Conceptually: thoughts are trajectories
through a space of abstract primitives; concepts evolve as they're
touched; recall is graph-autocompletion from a partial trajectory.
Stronger inductive bias than dense recurrent memory, naturally extends
to multi-modal substrates, and produces interpretable retrieval traces.

**Out of scope for v1:** verbatim recall (would need an external
non-overwriting store on top); cross-modal manifolds (extension, see
§8); a concept-space planning model (NCP, see §8).

---

## 1. Overview

A **manifold of N abstract concepts**, each with a stable identity vector
(`concept_ids`, used as a routing key) and a volatile state vector
(`concept_states`, content payload that mutates on writes). Concepts are
connected by sparse fixed edges. Thoughts and sentences are represented as
**ordered trajectories** through this manifold — paths whose nodes are the
concepts touched and whose order encodes structure (verbs, time, causation,
grammar).

The model wraps a frozen Llama backbone. Every 256-token window:

1. **READ** J parallel trajectories from the manifold, conditioning only on
   the *previous* window's tokens. Each trajectory is K_read concept states.
   The J·K_read states are injected into Llama via cross-attention so the LM
   can use retrieved memory while predicting this window's tokens.
2. **PREDICT** the window's 256 tokens with Llama, conditioned on the read
   trajectories.
3. **WRITE** J parallel trajectories at end of window, conditioned on the
   current window's tokens + a window-level surprise signal. The writes
   persistently mutate the visited concepts' states (with `scatter_mean`
   across overlapping visits).

Reading is non-destructive — visits don't change `concept_states`. Writing
is destructive — visits persist back into the manifold. The two are
decoupled: separate modules, separate parameters, different conditioning
sources.

The architecture has **constant inference footprint**. Long-range memory
is achieved via lossy compression — newer writes overwrite older state via
repeated visits.

---

## 2. Architecture

### 2.1 Manifold

Three tensors, fixed size:

```
Manifold:
    concept_ids    : [N, D_concept]    stable routing keys, only updated via backprop
    concept_states : [N, D_concept]    volatile content, mutates on write visits
    edge_indices   : [N, K_max]        sparse adjacency, fixed at init
```

The id/state split has clean roles:

- **`concept_ids`** — routing **key**. Stable. Only updated via backprop
  through the neighbor-scoring paths. Never mutated by visits. Keeps
  "concept i" recognizable across training even when its state has been
  pushed around by aggressive plasticity.
- **`concept_states`** — content **payload**. Mutated on every write visit
  (persistent), and also receives backprop gradient through whatever paths
  consume it. Carries the etched trace from past writes.
- **`edge_indices`** — sparse adjacency, fixed at init via small-world
  ring rewire (Watts-Strogatz style): concepts laid out on a ring of size
  N; each concept's default neighbors are picked from within ±radius
  positions (wraparound); each default edge is rewired to a uniformly
  random target with probability `p_rewire`. Result: ~half local + ~half
  random connections, giving short paths between distant regions while
  preserving semantic locality. Constraint: `K_max <= 2*radius`, since
  the local zone has 2*radius candidates. Defaults (medium tier, post
  2026-05-09 bump): `K_max_neighbors=64`, `radius=32`, `p_rewire=0.5`. (Note: graph_walker
  uses `radius=4` on a 2D Moore neighborhood with ~80 candidates; for
  our 1D ring the radius needs to be K_max/2 to give the same K_max.)
  Used to mask the neighbor softmax during trajectory hops (avoids N-way
  softmax per step). Optional v2: periodic K-NN refresh on `concept_ids`
  cosine geometry every N training steps. No scalar weights — edges are
  pure connectivity.

`D_concept = 256`. Same dim for id and state for symmetry — both occupy
the same vector space, just different update dynamics.

There is no `edge_weights`, no walker state, no neuromodulator, no global
state, no role embeddings.

**Initialization:**
- `concept_ids`: standard `nn.Embedding`-style init, `N(0, 1/√D_concept)`.
- `concept_states`: starts at `state_init`, a learnable `[N, D_concept]`
  parameter that gets reset to at every training-sequence start. Allows
  the model to learn a useful "default" manifold state.

### 2.2 Read module (start of window)

Produces J parallel trajectories, each K_read concept states long. Output
is flattened to `[J·K_read, D_concept]` for Llama cross-attention.

Read conditions only on `prev_window_hiddens: [256, D_lm]` — Llama's
hidden states from the previous window. No separate endpoint or boundary
state.

**Entry-point selection — Hopfield-tied projection (shared with write):**

The `EntryProjector` module (`src/trajectory_memory/read_module.py`) is
**shared** between the read and write modules. Same `head_query: [J, D_concept]`
+ same `entry_mlp` for both heads:

```
pooled = mean(prev_window_hiddens)                    # [D_lm]
for j in 1..J:
    Q_entry[j] = entry_proj(pooled, head_query[j])    # [D_concept]
    score[j]   = Q_entry[j] @ concept_ids.T           # [N]
    entry[j]   = gumbel_top1(score[j])                # discrete selection
```

**Why shared:** read at window d+1 and write at window d both pool the
same hiddens (window d's). Shared projection ⇒ same address logits ⇒
write deposits where the next read will retrieve from, *by construction*.
This is the modern-Hopfield-network identity: attention-on-energy-surface
is one step of gradient descent on the surface the write just deposited.
Without this tie, read and write collapse to disjoint subsets under
Gumbel-top-1 routing and write_module's gradient drops to zero. See
[feedback_write_grad_collapse memory] for the empirical history.

`head_query: [J, D_concept]` is the learnable per-trajectory bias. Each j
gets a different bias toward different regions of the manifold — same
multi-head trick attention uses. Without per-j bias the J trajectories
would collapse to near-identical paths.

**K_read autoregressive hops, all J trajectories batched:**

```
for t in 1..K_read:
    for j in 1..J (batched on GPU):
        history_attn = attn(
            query=current_state[j],
            keys/values=visited[j][:t] + pos_enc[:t],
        )
        cross_attn = attn(
            query=current_state[j],
            keys/values=prev_window_hiddens,
        )
        Q_t = MLP_step_read(current_state[j], history_attn, cross_attn)

        # pure QK matmul: Q_t against neighbor id_vecs, no edge bias
        nbr_ids = concept_ids[edge_indices[current[j]]]   # [K_max, D_concept]
        next_logits = Q_t @ nbr_ids.T                     # [K_max]
        next[j] = gumbel_top1(next_logits)                # discrete pick

        # append raw state to visited list — no mutation
        visited[j].append(concept_states[current[j]])
        current[j] = next[j]
        current_state[j] = concept_states[current[j]]
```

- **No `mutate_read`.** Visit order is encoded by the `pos_enc` added at
  attention time (see §3.1).
- **Q construction takes only `current_state[j]`, not `id_vec[current[j]]`.**
  The id_vec is what Q is matched against on the K side; passing it on
  the Q side is redundant.
- **Discrete selection via Gumbel-top-1 STE during training, argmax at
  inference.** Phase 1 needs Gumbel-STE so subsequent hops have a hard
  `current[j]` index for `concept_states[current[j]]` lookup. (Same
  trick graph_walker's Phase 1 already uses for walker routing.)
- Per-hop MLPs are shared across j; the j-divergence comes from per-j
  entry + per-j accumulated history, not per-j params.

Output: stack visited lists → `[J, K_read, D_concept]` → flatten to
`[J·K_read, D_concept]` for Llama.

### 2.3 Write module (end of window)

Same shape as read. Crucial differences:

- Conditions on `current_window_hiddens` (now available) plus a
  window-level surprise signal (consumed by hop / mutate MLPs, NOT by
  the entry projection — see §2.2 on Hopfield tie).
- **Shares entry projection** with read (§2.2). All hop-level MLPs
  (`step_mlp`, `history_attn`, `cross_attn`) and the `mutate_mlp` are
  independent.
- **Persistent mutation:** visited concept states get written back into
  the manifold via `scatter_mean` (handles collisions across J
  trajectories). `mutate_write` is the only mutation function (read
  module has none).
- No Hebbian update — there is no `edge_weights` to update.

**Entry-point selection (uses shared EntryProjector, identical pool of
current_window_hiddens that next window's read will see as
prev_window_hiddens):**

```
pooled = mean(current_window_hiddens)
for j in 1..J:
    Q_entry[j] = entry_proj(pooled, head_query[j])   # same as read!
    entry[j]   = gumbel_top1(Q_entry[j] @ concept_ids.T)
```

Note: surprise is no longer an input to entry routing — only to per-hop
step_mlp and mutate_mlp. Surprise controls *write strength*, not *write
location*. Putting surprise back in the address would re-introduce the
read/write divergence pathology.

**K_write autoregressive hops, all J batched:**

```
for t in 1..K_write:
    for j in 1..J (batched):
        history_attn = attn(
            query=current_state[j],
            keys/values=proposed[j][:t] + pos_enc[:t],
        )
        cross_attn = attn(
            query=current_state[j],
            keys/values=current_window_hiddens,
        )
        Q_t = MLP_step_write(
            current_state[j], history_attn, cross_attn, surprise,
        )

        next_logits = Q_t @ concept_ids[edge_indices[current[j]]].T
        next[j] = gumbel_top1(next_logits)

        # produce proposed new state — do NOT write to manifold yet
        new_state = mutate_write(
            current_state[j], cross_attn, history_attn, surprise,
        )
        proposed[j].append((current[j], new_state))
        current[j] = next[j]
        current_state[j] = concept_states[current[j]]   # raw lookup; pre-mutation values
```

**Conflict resolution at end of write — functional scatter_mean:**

```
all_ids    = [cid for j in 1..J for (cid, _) in proposed[j]]      # [J·K_write]
all_states = [s   for j in 1..J for (_, s)   in proposed[j]]      # [J·K_write, D_concept]

# functional update: returns new tensor, doesn't mutate concept_states in place
# (avoids autograd version-counter issues across TBPTT windows)
concept_states_new = scatter_mean(concept_states, all_ids, all_states)
```

The functional form (returns `concept_states_new`) is required so PyTorch
autograd can backprop through the chain of D windows without hitting
"modified in place" errors. The next window's read uses
`concept_states_new`; the original tensor stays valid for the previous
window's autograd graph.

`surprise` is the per-token NTP cross-entropy of the just-predicted
window (mean-pooled to a scalar, computed only on tokens that have a
prediction target — see §3.3). It enters both the step MLP and the
mutation MLP — high surprise scales the magnitude of state mutation,
recovering prediction-error-gated plasticity without a separate
neuromodulator module.

### 2.4 Llama integration

**Reuses the existing `MemInjectLayer`** from
`src/pretrained/mem_inject_layer.py` — same shape as graph_walker, just
with the read trajectory as KV source instead of per-token walker
readout. A single cross-attention layer at a chosen mid-stack location
of an otherwise-frozen Llama:

```
input_ids
  ▼
embed_tokens                           (frozen Llama embedding)
  ▼
layers 0..L-1                          (frozen)
  ▼
MemInjectLayer at layer L:
    h_proj = W_in(h)                  # [BS, T, D_concept]
    attn_out = cross_attn(
        query  = h_proj,
        keys   = read_trajectory,     # [BS, J·K_read, D_concept]
        values = read_trajectory,
    )
    h' = h + scale * W_out(attn_out)
    orig_layer_L(h')                  (frozen)
  ▼
layers L+1..N-1                        (frozen)
  ▼
norm → lm_head → logits
```

We inject concept *states* (the read trajectory's output), not id_vecs.
State carries content; id_vec is just the routing key.

Trainable surface: `W_in`, `W_out`, `scale`, cross-attn weights,
manifold (`concept_ids`, `concept_states`, `state_init`), read module,
write module. Llama backbone fully frozen.

**No role embeddings.** Llama's chat-template tokens (`<|im_start|>user`,
etc.) already carry role information through the token sequence; adding
a separate role embedding would shift Llama's input distribution off
its pretraining manifold. Role-aware behavior (e.g., surprise on
assistant tokens only) is handled at the data-loader level, not as a
model-side embedding.

### 2.5 Per-window cycle

```
                window N-1                   window N                 window N+1
            ┌───────────────┐           ┌───────────────┐         ┌───────────────┐
tokens  ──> │ 256 tokens    │           │ 256 tokens    │         │ 256 tokens    │
            └───────────────┘           └───────┬───────┘         └───────────────┘
                    │                           │
                    │ prev_window_hiddens       │
                    │ ──────────────────────────│
                    │                           ▼
                    │                  1. READ at start of N
                    │                  pool prev hiddens, J head queries
                    │                  J × K_read autoregressive hops
                    │                  output: [J·K_read, D_concept]
                    │                           │
                    │                           ▼ inject as cross-attn KV
                    │                  2. PREDICT
                    │                  Llama generates N's 256 tokens
                    │                           │
                    │                           ▼ surprise = mean per-tok NTP CE
                    │                  3. WRITE at end of N
                    │                  cross-attn to N's tokens + surprise
                    │                  J × K_write trajectories
                    │                  scatter_mean → concept_states_new
                    │                           │
                                  manifold updated, ready for N+1's read
```

Cross-window linkage flows through *only* two channels:
1. The previous window's token hiddens (used as the read query source).
2. The manifold's etched `concept_states` (used implicitly, since per-hop
   Q construction reads `current_state[j]`).

No bespoke continuity vector, no walker state.

---

## 3. Mechanics in detail

### 3.1 Order encoding via positional encoding

Visit order in the trajectory is encoded by **adding sinusoidal positional
encoding to the visited list** at attention time. At hop t, the
history_attn keys/values are `visited[j][:t] + pos_enc[:t]`, so subsequent
hops see "this state was visited at position i."

This is cheap (a fixed `[K, D_concept]` sin/cos table), removable later
if proven unnecessary, and provides the order signal upfront — no need
to rely on a learned mutation function to encode order implicitly.

`pos_enc` applies during trajectory generation (history_attn for both
read and write) but NOT to the manifold's persisted `concept_states` —
write proposals use raw `new_state` without pos_enc when scattering back.

### 3.2 Plasticity at chunk boundary

The plasticity surface is reduced to one channel: **concept state
mutation during writes, applied via `scatter_mean` at the end of the
write trajectory**.

- Read: no mutation, no persistence. Read just builds a trajectory
  representation by hopping through the manifold and recording raw
  visited states.
- Write mutations: persist via `concept_states_new = scatter_mean(...)`
  (functional, returns a new tensor — the in-place version is unsafe
  across TBPTT windows).
- `concept_ids`: only changes via backprop (no in-place update).
- `edge_indices`: fixed at init (or refreshed periodically as v2).
- No `edge_weights` to update. No Hebbian update.
- No structural rewiring of edges based on co-activation in v1.

This means within a single trajectory, all states the trajectory
generator sees are *frozen at chunk-entry values* — proposed mutations
only land in the manifold once at end of write, simultaneously across
all J trajectories via scatter_mean.

The neuromodulator is gone; surprise enters as a direct input to the
write module's step MLP and mutation MLP.

### 3.3 Surprise integration

Per-token NTP cross-entropy is computed per window, then mean-pooled
over **target-eligible positions only** to produce a window-level
scalar. Target-eligible means tokens for which we have a meaningful
prediction target — typically:

- **TF training (long doc)**: all tokens are eligible.
- **Multi-turn TF**: only assistant tokens (we don't NTP-train on user
  input).
- **AR inference**: no targets exist (we generated), so surprise is set
  to 0 (or a constant). See §5.4.

The pooled scalar feeds into both `MLP_step_write` and `mutate_write`.
High surprise scales mutation magnitude — prediction-error-gated
plasticity without a dedicated neuromodulator.

The role-distinction logic (which tokens are assistant tokens) lives in
the data loader, not the model. The model itself is role-agnostic.

---

## 4. Training

### 4.1 Objective: TF NTP on long documents

Standard teacher-forced next-token prediction. The training signal that
forces memory to be useful is **document length > Llama's effective
context window**. If the data fits in Llama's context, the LM solves
NTP without memory; memory is only load-bearing when relevant
information is too old for the context window.

Concretely: Llama-3.2-1B has 128K native context, but we **deliberately
cap effective LM context at 2K tokens** by hard-truncating the LM input
at every prediction step. The LM only ever sees its 2K sliding window
plus the read trajectories. Cross-window dependencies must route
through the manifold.

**Implementation: sliding KV cache (LANDED 2026-05-10).** Each window's
forward only encodes the new T_window tokens against an HF DynamicCache
that carries prior windows' KVs. Sliding-window-trimmed at
`effective_lm_context = 2048` tokens. Cache carries across windows of
a chunk; detached (but kept) at chunk boundaries.

CRITICAL implementation detail — `cache_position` must be CACHE-INTERNAL
indices (0..target_length-1) for HF's causal mask to be correct, BUT
`position_ids` must be ABSOLUTE positions for RoPE to be consistent
with cached KVs (which baked in their original-position rotations).
Both are passed to `llama.model()` separately.

ALSO CRITICAL — gradient checkpointing is INCOMPATIBLE with use_cache=True
in HF Llama (silently sets use_cache=False). We pick KV cache (1.79× /
~5 GB win) over gradient checkpointing for our BS=1/2 use case. Don't
re-enable gradient_checkpointing_enable() unless you switch to
`--no-kv-cache` for a particular run.

The earlier hard-truncation rolling-buffer mode is preserved as the
`--no-kv-cache` fallback.

The 2K cap is the load-bearing knob: smaller cap → stronger memory
pressure but harder convergence; larger cap → direct attention solves
more, memory atrophies. 2K is ~8 windows of 256 — a reasonable middle.

### 4.2 Cross-window TBPTT

TBPTT depth = D windows is **load-bearing** for memory training.
Without cross-window gradient flow, the write module receives no
signal: window N's NTP loss flows back through window N's *read* into
the manifold etched by window N-1's *write*; if gradient stops at the
manifold, the write trajectory generator is trained by nothing.

- Minimum useful D = 2 (so window N's loss reaches window N-1's write).
- Recommended starting D = 4 to 8.
- Higher D lets read+write co-adapt over longer horizons.

Functional `scatter_mean` (§2.3) is what lets gradient flow cleanly
across D windows — each window produces a new `concept_states` tensor
linked into the autograd graph.

### 4.3 Gradient checkpointing strategy

What we ship in production (verified on RTX 4090, BS=8 D=4, 11.6 GB peak):

1. **Llama gradient checkpointing — OFF.** Tried, regressed VRAM by
   ~+6 GB and slowed steps by ~25% in our setup. Likely a bad
   interaction between HF's `gradient_checkpointing_enable` wrapper and
   our custom `MemInjectLayer` replacing transformer layer 8 (the
   wrapper expects a stock transformer block at every position). Not
   debugged further — the win below made it unnecessary.
2. **Trajectory-generator activation checkpointing — ON.** `read_module`
   and `write_module` forward calls are wrapped with
   `torch.utils.checkpoint(use_reentrant=False)` inside `forward_window`.
   Per-hop cross-attn + history-attn + step_mlp activations get
   recomputed in backward. Cheap because trajectory generators are small.
3. **Cross-attn K/V sharing — ON.** `_CrossAttn.precompute_kv()` computes
   K, V from `prev_window_hiddens` once per window and reuses across
   J trajectories and K_read/K_write hops. Saves the per-hop
   `[BS, J, T, d_lm]` materialization that `.expand().reshape()` was
   forcing. ~3–4 GB at BS=4 D=4 freed.
4. **bf16 autocast on cross-attn body — ON.** Manifold stays fp32 (Adam
   stability), but the cross-attn matmuls + softmax run under
   `torch.autocast(bf16)`. Halves activation memory on attention without
   touching the param storage.

The combination above puts BS=8 D=4 at 11.6 GB / 27.2k tok/s with
`torch.compile` on (the production config). BS=12 fits but compile
regresses on that shape. BS=16 OOMs.

**Linear-in-D vs constant-in-D:**

- Linear-in-D: every window's write graph alive simultaneously.
  Standard PyTorch checkpointing path. Easy to write. Memory grows with D.
- Constant-in-D: custom autograd function streams windows, checkpointing
  at window boundaries. Memory bounded regardless of D. Harder to write.

We use linear-in-D at D=4 in production. The architectural fixes above
gave enough headroom that constant-in-D isn't needed at current scale.

### 4.4 Hyperparameters (current `medium` config)

Updated 2026-05-09 to match `TrajMemConfig.medium()` in code (the
2026-05-09 N=2048→4096 / K=32→64 bump landed for richer manifold
capacity at trivial perf cost):

| Knob                  | Value           | Notes                            |
|-----------------------|-----------------|----------------------------------|
| `T_window`            | 256             | Window size in tokens            |
| `N`                   | **4096**        | Manifold size (post-bump)        |
| `D_concept`           | 256             | Concept id + state dim (single)  |
| `K_max_neighbors`     | **64**          | Sparse degree (post-bump)        |
| `p_rewire`            | 0.5             | Watts-Strogatz rewire probability |
| `radius`              | **32**          | 1D-ring local zone half-width (≥ K_max/2) |
| `J`                   | 4               | Parallel trajectories per window |
| `K_read`              | 8               | Length of each read trajectory   |
| `K_write`             | 8               | Length of each write trajectory  |
| `D` (TBPTT depth)     | 4               | Windows of cross-window grad     |
| Llama base            | Llama-3.2-1B    | Frozen backbone                  |
| Inject layer L        | 8 (mid-stack)   | Where MemInjectLayer goes        |
| `bridge_hidden`       | 2048            | MemInjectLayer 2-layer MLP hidden dim |
| `D_lm`                | 2048 (Llama)    | Llama hidden dim                 |
| Effective LM context  | 2048            | Sliding KV cache cap (was hard-trunc; now KV-cached) |
| LR (memory params)    | **1.5e-4**      | Read+write modules + manifold (Tier 2 #8 — halved from 3e-4 for stability) |
| LR (Llama-side adapters) | **5e-5**     | W_in, W_out, scale, cross-attn (halved from 1e-4) |
| Mutation init scale   | 0.1             | `new = state + 0.1 · MLP(...)`   |
| Surprise pool         | weighted mean   | Per-token weighted CE → window scalar (writer input) |
| `state_init`          | learnable `[N, D_concept]` | Reset target each sequence |
| Trainable params      | **16.46M**      | bridge 9.44M + write 2.49M + read 2.43M + manifold 2.10M |
| `prior_loss_weight` (W2) | **0.1**      | NTP CE weight on prior tokens (B12). Was 0; matches §4.8 surprise table. |

K scales with N if you sweep manifold size: rough rule `K ≈ √N / 4`. J
is multi-head — independent of N.

Total per-window memory footprint (read trajectories injected into
Llama): J·K_read concept states. At J=4, K_read=8 → 32 keys/values per
cross-attn call. Cheap.

### 4.5 Data and training waves

Four waves build the model end-to-end. Wave 1 and 2 are Phase 1 (TF). Wave
3 and 4 are Phase 2 (GRPO). Each wave directly exercises memory.

#### Wave 1 — Phase 1 (TF) — Long-doc memory pretraining

Goal: teach the memory module to encode and retrieve. The architecture's
most critical training stage; everything downstream assumes a working
manifold.

**Original target mix (aspirational):**

| Source                                | Weight | Why                                          |
|---------------------------------------|--------|----------------------------------------------|
| Books (Gutenberg, Books3, BookCorpus) | 40%    | Strongest natural long-range structure (characters, plots, callbacks) |
| Code (TheStack subset)                | 25%    | Explicit cross-references (defs → calls, imports → usages) |
| ArXiv papers                          | 15%    | Definition-use, terminology consistency, citations |
| Web (FineWeb / RedPajama, length > 8K filter) | 10% | Diversity, but de-prioritized                |
| Synthetic needle-in-haystack          | 10%    | Plant fact at position X, query at Y > X+2K. Forces measurable memory contribution to NTP loss. |

**Actual implemented mix (2026-05-09):**

| Source                                | Status | Notes |
|---------------------------------------|--------|-------|
| `HuggingFaceFW/fineweb-edu` (sample-10BT) | wired ✓ | `preprocess_longdoc.py fineweb-edu`. Edu-classifier filtered, len ≥ 4K tokens. |
| `wikimedia/wikipedia` (20231101.en)   | wired ✓ | Encyclopedia long-form; substitute for ArXiv definition-use structure. |
| `DKYoon/SlimPajama-6B`                | wired ✓ | RedPajama-derived mix incl. books — partial replacement for Books slot. |
| Needle-haystack synthetic             | wired ✓ | `synthesize_needle.py` with curriculum distances 3K/8K/16K/32K, fillers from FineWeb-Edu. |
| Books (pure)                          | gap — `deepmind/pg19` script-based, broke when `datasets ≥3.0` removed script support. SlimPajama provides partial coverage. |
| Code                                  | gap — `bigcode/the-stack-dedup` is HF-gated. Open alternative `codeparrot/github-code` not yet wired into `preprocess_longdoc.py`. |
| ArXiv                                 | gap — `EleutherAI/proof-pile-2` arXiv subset not yet wired. Partially substituted by Wikipedia structure. |

**First preprocessing run (2026-05-09, ~600-800M tokens preprocessed):**

| Source | Stream cap | Kept (≥4K tok) | Tokens (M) |
|--------|-----------|----------------|-----------|
| `wikipedia-en` | 100K | 3975 docs | 30.8 |
| `slimpajama-6b` | 100K | 3134 docs | 35.5 |
| `fineweb-edu` | 500K | TBD (expect ~15K) | ~75-100 |
| `needle` | 5000 docs × 4 distance buckets | 20000 | ~200-240 |

Long-doc keep rate at min_tokens=4096 is ~3-4% across all three real-text sources. To scale up to converged-Wave-1 size (~5-10B tokens), bump streams to 5M+ examples or relax min_tokens — the 4K threshold is the dominant bottleneck.

For converged training, adding `codeparrot/github-code` is the highest-priority gap to close — its def→call structure is a distinct memory-pressure signal that no other source provides. arXiv (proof-pile-2) is second priority. AgentInstruct is a Wave 2/4 gap, not Wave 1.

Format: streaming tokens, 256-token windows. No turn structure. NTP loss
on every token. Surprise = mean per-token NTP CE per window.

Curriculum: start at moderate doc length (4–8K, fits 16-32 windows),
extend to 16K then 32K+ (fits 64-128 windows). Shorter docs first help
convergence; longer docs late stress the architecture.

Replaces graph_walker's W1 (was generic FineWeb).

#### Wave 2 — Phase 1 (TF) — Long-chat warmup

Goal: instruction-following on a memory-trained base, while exercising
memory across multi-turn structure.

The challenge: standard chat datasets (UltraChat, ShareGPT) are mostly
short — turns of a few hundred tokens, all fitting in the 2K cap. Memory
wouldn't get exercised.

**Original target sources (aspirational):**
- **WildChat filtered for session length** > 4K total tokens or > 10 turns.
- **AgentInstruct** (long tool-use sessions).
- **LongAlpaca / LongInstruct** (long-context instruction tuning).
- **Synthetic conversion**: take a long doc from Wave 1, generate multi-turn Q&A about content scattered through it.

**Actual implemented sources (2026-05-09):**

| Source                                | Status | Notes |
|---------------------------------------|--------|-------|
| `allenai/WildChat-1M`                 | wired ✓ | `preprocess_chat.py wildchat-1m`. English-only + non-toxic filters; min_prior=4K. |
| `HuggingFaceH4/ultrachat_200k`        | wired ✓ | `preprocess_chat.py ultrachat-200k`. Pre-filtered English. UltraChat sessions are short (~6 turns × ~200 tok); at min_prior=4K filter yields ~0 pairs. Used at **min_prior=1024** as instruction-format warmup; WildChat carries the long-prior weight. |
| `HuggingFaceTB/smoltalk`              | wired (registry only) | Mostly 2-turn, less useful for memory; not used for first run. |
| AgentInstruct                         | gap — `microsoft/orca-agentinstruct-1M-v1` not yet wired. |
| LongAlpaca / LongInstruct             | gap — not wired. |
| Synthetic chat conversion             | gap — converter not yet built. |

For converged training, AgentInstruct would be the highest-priority gap to close — its long tool-use sessions are the closest non-synthetic match to deployment-shape Wave 4 use. WildChat-long covers a similar shape but with shorter, less structured tool-use signal.

Format: TurnPair extraction via `session_to_turn_pairs` (graph_walker's
existing util). For a session of N assistant turns, generate N training
examples, each `(prior, response)`. Filter for `len(prior) > 4K` so
memory has work to do.

Surprise: per-token NTP CE only on response tokens (data loader masks
prior to 0). Prior tokens get TF-forwarded but contribute zero NTP loss.

Replaces graph_walker's W2 (was UltraChat).

#### Wave 3 — Phase 2 (GRPO) — Verifiable-reward reasoning

Goal: AR rollouts + verifiable rewards to refine reasoning. Same
structure as graph_walker's W3 but with memory in the loop and a
different dataset shape.

**Implemented sources (2026-05-09, all wired ✓):**

| Source                            | HF id                  | Use |
|-----------------------------------|------------------------|-----|
| GSM8K (full train, 7473 examples) | `openai/gsm8k`         | Math word problems; gold = number after `####`, regex tolerates decimals/commas/negatives. |
| NuminaMath-TIR (50K examples)     | `AI-MO/NuminaMath-TIR` | Math with code execution; gold = final `\boxed{}` answer. |
| HumanEval (full test, 164)        | `openai_humaneval`     | Code with verifiable test pass/fail; rule_based_exec reward. |
| NarrativeQA (5K examples)         | `deepmind/narrativeqa` | Long-context QA. **Uses 32K-char slice of full document** (~8K tokens) so prompt extends past 2K LM cap (memory-stress). `use_summary=True` available as fallback for faster but lower-stress runs. |

Format: `(prompt, response)`. For math/code, prompt is the problem.
For NarrativeQA, prompt is `passage + question`. 1–5k prompts × 4–8
responses each.

Reward: rule-based accuracy for math/code (exact match on final answer);
exact match + BERT cosine for NarrativeQA (matches project's stored
reward preference — no LLM-as-judge).

Routes through `grpo_session_step` like the existing graph_walker plan.
Surprise during the response (sampled) windows: 0 (default) or entropy
of the sampled distribution (proxy for "model uncertain"). GRPO advantage
is the dominant write-strength signal.

#### Wave 4 — Phase 2 (GRPO) — Long-session / agentic

Goal: align memory behavior in deployment-shaped scenarios. Stresses
lifelong-style behavior across many turns.

**Original target sources:**
- **WildChat-long → TurnPair extraction**, filtered for prior length > 4K
- **AgentInstruct → TurnPair extraction** (long tool-use sessions)
- Optional: curated Claude Code session traces or other agentic datasets

**Implemented (2026-05-09):**

| Source                     | Status | Notes |
|----------------------------|--------|-------|
| WildChat-long (reuse W2)   | wired ✓ | Same parquet as W2 (`data/wave2/wildchat_long.parquet`); English+non-toxic, min_prior=4K. |
| AgentInstruct              | gap    | `microsoft/orca-agentinstruct-1M-v1` not yet wired. |
| Claude Code session traces | gap    | No corpus exists. |

For converged training, AgentInstruct is the highest-priority W4 add — it's the only non-synthetic source designed specifically for long tool-use session structure. Without it, W4 is essentially "WildChat reused with GRPO instead of TF" rather than the deployment-shape coverage the plan calls for.

Format: TurnPair, length-bucketed (graph_walker's existing W4 approach:
sort by prior length, sample windows of B near-uniform-length neighbors,
truncate to min length within batch).

Reward: exact match + BERT cosine to ground-truth assistant turn.

Reuses `grpo_session_step` and Verlog-style turn-batching machinery
directly. Drop-in substitution of the trajectory module for graph_walker.

#### Compute split (rough)

| Wave | Phase    | Compute share | Notes                                           |
|------|----------|---------------|-------------------------------------------------|
| 1    | Phase 1  | ~60–70%       | Where memory actually learns; needs the most data |
| 2    | Phase 1  | ~10%          | Short context, fast iterations                   |
| 3    | Phase 2  | ~10%          | Rollouts expensive but dataset small             |
| 4    | Phase 2  | ~10–15%       | Long sessions, expensive per-example             |

#### Departures from graph_walker's wave plan

- W1 changes from generic FineWeb to a targeted long-doc mix. Biggest
  change, driven by our memory-specific training pressure.
- W2 changes from UltraChat to filtered-long-chat + AgentInstruct.
- W3 keeps the verifiable-reward GRPO structure but uses real reasoning
  datasets (GSM8K, NuminaMath, HumanEval) instead of synthetic 2-turn
  passphrase. Adds NarrativeQA for memory stress.
- W4 retains the WildChat-TurnPair structure unchanged.

#### Cross-cutting

- **Memory probe eval (§6.5) at every checkpoint**, not just end of wave.
- Manifold resets per training sequence (§5.4); document boundaries
  align to sequence boundaries.
- **Dev sanity run before Wave 1** — small synthetic long-context smoke
  test (~1k examples, 8-window docs with planted facts) — verify the
  architecture is plumbed correctly before throwing real training
  compute at it.

### 4.6 Two-pass GRPO (default — was originally documented here as fallback)

The two-pass structure is now the default for Wave 3 / Wave 4 (changed
2026-05-09 from single-pass; see §4.7 for the bug rationale):

```
Per prompt × K samples:
    Pass 1 (sampling, no_grad, KV-cached):
      - Walk forward_window over prompt → KV cache + manifold state
      - AR-generate K samples token-by-token with KV cache
      - Per-window memory ops: read at window start, write at window end
      - Record logp_old per sampled token (cheap, near-free)
    Pass 2 (TF replay, with grad):
      - Shared no_grad prefill: encode prompt ONCE
      - Per sample: TF-replay sample suffix only with grad
      - Recompute logp_new per sample-token under current policy
      - Compute IS ratio, PPO clip, advantage-weighted policy loss
    Optional Pass 2.5 (KL term, no_grad, ref-policy):
      - Param-swap to ref weights, shared ref prefill, K ref TF replays
      - K3 KL estimator added to total loss
    Single backward at end of step.
```

Originally listed as "fallback for exposure bias" with single-pass as
default. Single-pass turned out to have a real bug: detaching state
per generated token to bound activation memory cut the gradient chain
to write_module. Two-pass replaces this; see §4.7 for full mechanics.

The "rewind-and-re-forward over the SAME ground-truth tokens" idea
(replace generated tokens with ground truth in pass 2) remains a
fallback for the edge case where post-Wave-4 evals show memory
degradation specifically traceable to exposure bias. Not currently
implemented — pass 2 always uses the sampled tokens (matching pass 1).

### 4.7 Training-phase mapping vs graph_walker

The existing graph_walker has Phase 1 (TF Gumbel-STE bootstrap) and
Phase 2 (AR GRPO). Both carry over with minor adaptations:

**Phase 1 (TF NTP) — mostly unchanged.**
- Trajectory module slots in where graph_walker's walker is.
- NTP loss flows through the same `MemInjectLayer` cross-attn path.
- Gumbel-top-1 STE on routing is still needed (per-hop discrete
  next-concept selection — same trick graph_walker already uses).
- Cross-window TBPTT (D windows) replaces within-T TBPTT (`tbptt_block`
  tokens). Same idea, different boundary.

**Phase 2 (AR GRPO) — implemented as two-pass with batched rollouts.**

Single-pass GRPO (§4.8 default) had a real bug: detaching state per AR
token to bound activation memory cut the gradient chain to write_module
entirely. Replaced with **two-pass GRPO** (originally listed as fallback
§4.6, made default 2026-05-09):

  Pass 1 — sample (`_ar_sample_batch`, no_grad, KV-cached, with prefill).
    Walk `forward_window` over the prompt at BS=1 to populate KV cache +
    manifold state. Then expand the prefill state to BS=K and run all K
    rollouts **in parallel as a single BS=K AR loop** (R1 optimization,
    2026-05-11). Per-sample EOS tracking via a `finished` mask; finished
    samples get force-padded for remaining steps. Per-window memory ops
    (read at window start, write at window end) fire at BS=K. Each
    sampled token's `logp_old` recorded for pass 2's IS clip. Replaces
    K × T_gen sequential single-token forwards with T_gen BS=K forwards
    — kills the launch-bound bottleneck. Bench at K=8, prompt=1K, gen=256:
    rollout time dropped 11.3s → 1.6s (7.2× speedup) vs the prior
    K-serial implementation.

  Pass 2 — TF replay (`_tf_compute_sample_logp`, with grad).
    TF-forward through (prompt + sample) in T_window windows. Memory
    state carries through `write_module` without detach. **Shared
    prefill** (no_grad) optimization: prompt is encoded ONCE, K samples
    each start from the cloned prefill state. **Selective log-softmax**
    (R2): logp = gather(scaled, target) - logsumexp(scaled) instead of
    `F.log_softmax().gather()` — avoids the [n_pred, V=128k] transient.
    **Per-sample backward** (R3): each sample's policy + KL loss
    `.backward()` fires inside the loop, freeing that sample's TF
    activation graph before the next sample's graph builds. PyTorch
    accumulates gradients into `.grad` buffers across the K backwards;
    mathematically identical to a joint backward (`∇Σᵢ lᵢ = Σᵢ ∇lᵢ`)
    but ~K× less peak VRAM. Bench at K=8 / prompt=512 / gen=256: total
    step (pass 1 + pass 2 + backward) = 1.78s at 3.48 GB peak.

GRPO regularization (matches DeepSeek-R1, TRL, verl, OpenRLHF):

- **Group-relative advantage**: `(reward - mean(rewards)) / std(rewards)`
- **Dr.GRPO loss normalization**: divide by `K * max_new_tokens`
  (constant), not `K * len(sample)`. Removes documented length-bias
  pathologies (arxiv 2503.20783, Sea AI Lab COLM 2025).
- **PPO importance-sampling clip**: `clip(ratio, 1-ε, 1+ε)` with ε=0.2
  default. Optional asymmetric upper clip via `--clip-eps-higher`
  (DeepSeek-R1's `clip_higher` mode, e.g. ε=10).
- **KL regularization to reference policy**: `β * D_KL(π_θ ‖ π_ref)`
  with β=0.001 (verl default). Reference is `--checkpoint-in` weights
  snapshotted by `Phase2Trainer.set_reference_state()` at trainer init.
  K3 estimator: `KL ≈ exp(log r) - log r - 1`. Param-swap pattern
  shares the ref forward across K samples (one swap-cycle per step).

**Phase 2 routing is currently FROZEN — known limitation.** The IS-ratio
correctness fix (N3) switched both pass-1 and pass-2 to `hard_routing=False`
(deterministic argmax, no Gumbel noise) so the same routing path fires
on both sides → IS ratio is meaningful. But `argmax + one_hot` has no
gradient path back to logits → `entry_mlp`, `step_mlp`, `head_query`
in the read/write modules receive ZERO gradient in Phase 2.

Effect: Phase 2 GRPO refines `mutate_mlp` (writer's content output),
`read_attn` (Llama bridge cross-attn), W_in/W_out/scale (MemInjectLayer
adapter), and concept_states/concept_ids (via gradient through the
attention chain) — but routing decisions are LOCKED at Phase-1-end
values. Acceptable for first GRPO runs since Phase 1 trains routing
via Gumbel-STE; bigger Phase 2 refinements would benefit from also
training routing.

Proper fix (deferred): record routing IDs/seeds in pass 1, force them
in pass 2 (`hard=True` with shared Gumbel state) so both passes follow
the same path AND gradient flows via STE. Real refactor — affects
read/write module forward signatures and per-window seed plumbing
through generation windows.

Diagnostics surfaced in `Phase2Metrics`:
- `clip_fraction`: % of tokens where ratio was clipped this step
- `mean_ratio`: mean(ratio); should stay near 1.0 if policy isn't drifting
- `kl_to_ref`: mean per-token K3 KL estimate

Policy granularity remains much smaller than walker's per-token policy:
J·K decisions per window instead of T per window. Lower variance,
easier training.

KV cache (sliding-window-trimmed to `effective_lm_context`) was added
2026-05-10 to all four waves. Phase 1 went from 9.9k → 17.7k tok/s
(1.79× speedup, 5GB less memory) — the rolling LM buffer re-encode was
~70% of the T1-vs-vanilla gap. Phase 2 went 32.4 → 132.8 tok/s with
shared prefill + bigger K (4 → 16). See `docs/bench_results.md`.

Trainer scaffolding lives in `src/trajectory_memory/training/`:
`Phase1Trainer.step_wave1` / `step_wave2`, `Phase2Trainer.step`. Entry
points `scripts/train_wave{1,2,3,4}.py`. Defaults: `--use-kv-cache`
ON, `--compile` ON, `--temperature 0.7`, `--clip-eps 0.2`,
`--kl-coef 0.001`.

### 4.8 Training mechanics (operational details)

How the per-window loop actually runs in each wave.

**Window structure:**
- 256-token windows, strict (no variable size).
- Windows are agnostic to turn boundaries — they may span user →
  assistant transitions, tool_call → tool_result transitions, etc. The
  chat template tokens within the window's 256 carry role info; cross-attn
  over the full window picks up role transitions naturally.
- No special "turn end" plumbing required.

**Sampling location (Wave 3 / Wave 4 GRPO):**
- Sampling happens *only at assistant turns*. User input, tool output,
  system content are ground truth and TF-forwarded.
- Within an assistant turn, the model autoregressively samples until
  it emits an end-of-turn token (or hits a max-tokens cap, typically
  1K–2K tokens).

**Two-pass GRPO (current default for Wave 3 / Wave 4 — implementation
in `Phase2Trainer.step`):**

The original §4.8 single-pass design (sample with logp recorded, single
backward) had a real bug: detaching state per generated token to bound
memory cut the gradient chain to write_module. Replaced with the
§4.6 fallback as default (2026-05-09):

1. **Pass 1 — sample (no_grad, KV-cached, with prefill).** Walk
   forward_window over the prompt to populate KV cache + manifold state.
   AR-generate the assistant response token-by-token with KV cache
   (1-token forward against cached prefix per token). Record `logp_old`
   per sampled token (cheap — softmax is computed for sampling anyway).
   Per-window memory ops: read once at start of each generation window,
   write once at end. NOT per-token (that was a separate fixed bug).

2. **Pass 2 — TF replay (with grad).** Shared no_grad prefill runs the
   prompt through forward_window ONCE. K samples each TF-replay only
   the SAMPLE portion from the cloned prefill state, with grad through
   write_module. Per-sample-token logp recomputed under current policy
   (= `logp_new`).

3. **Score the rollout** (verifiable for Wave 3, exact-match + BERT
   cosine for Wave 4).

4. **PPO-clipped GRPO loss + KL regularization:**
   - Group-relative advantage: `A = (r - mean(r)) / std(r)`
   - IS ratio: `r = exp(logp_new - logp_old)`, clipped to `[1-ε, 1+ε]`
   - Surrogate: `loss = -min(A·r, A·clip(r))`
   - KL term: `β · D_KL(π_θ ‖ π_ref)` via K3 estimator
   - Total: `(loss + β·kl) / (K · max_new_tokens)` — Dr.GRPO norm
   - Single backward at end of step.

**Surprise per wave:**

| Wave | Prior windows                  | Response windows                      |
|------|--------------------------------|---------------------------------------|
| 1    | NTP CE on all tokens (no prior/response distinction) | —                                |
| 2    | NTP CE on prior tokens (TF)    | NTP CE only on response tokens (TF)   |
| 3    | NTP CE on prior tokens (TF)    | 0 (or entropy of sampled distribution) |
| 4    | NTP CE on prior tokens (TF)    | 0 (or entropy)                        |

In Wave 3/4 response windows, GRPO advantage is the dominant
write-strength signal — surprise's role is partly subsumed.

**Length-bucket batching (Waves 2, 3, 4):**

For TurnPair (Wave 2, 4) and (prompt, response) pairs (Wave 3):
- Sort examples by prior length.
- Sample windows of B near-uniform-length neighbors.
- Truncate to min length within batch.
- Same pattern as graph_walker's existing W4. Reuse the machinery
  directly.

For Wave 1 (no turn structure), windows are streamed from documents
using standard token packing — no length-bucket gymnastics needed.

**TurnPair flattening — rationale and tradeoff:**

Multi-turn chat data is naturally
`[user_1, assistant_1, user_2, assistant_2, ...]`. We flatten to
`(prior, response)` TurnPairs via `session_to_turn_pairs`. For a
session of N assistant turns, this generates N training examples — the
k-th has `prior = [user_1, asst_1, ..., user_k]`, `response = asst_k`.

Why flatten:
1. **Batching**: TurnPairs of similar prior length batch cleanly.
   Multi-turn-as-is would have wildly varying shapes (different
   numbers of turns, different total lengths).
2. **No turn-to-turn dependency**: in true multi-turn rollout, the
   sample at turn 2 depends on what turn 1 sampled. Reward attribution
   and gradient flow get messy and variance compounds. With TurnPair,
   every example samples one response from a clean ground-truth prior.

Tradeoff: memory does *not* accumulate across TurnPair examples within
a session. Every TurnPair starts with a freshly reset manifold. So we
never explicitly train the case "the model's own previous assistant
response is in memory, and it has to use that." This is a small
distribution shift between training (always-ground-truth prior) and
inference (own-generations-in-memory).

In practice, rarely a problem — graph_walker's W3/W4 use TurnPair and
the resulting models behave fine in multi-turn deployment. Worth
measuring post-Wave-4 but not worth pre-engineering around.

**Manifold reset semantics:**

| Scenario                                | Action                                  |
|-----------------------------------------|-----------------------------------------|
| Wave 1 training sequence (D windows)    | Reset to `state_init` at sequence start |
| Wave 2 / 3 / 4 TurnPair example         | Reset to `state_init` at example start  |
| Within a TurnPair (prior → response)    | Persist (this is the example's session) |
| TBPTT depth boundary                    | Persist (gradient cut, state continues) |

---

## 5. Inference

### 5.1 Constant footprint

At any moment during inference:

- Manifold: `2 N D_concept + N K_max` (fixed)
- Llama context: 2K-token sliding window (fixed)
- Read trajectories: `J · K_read · D_concept` (fixed)
- Write trajectories: `J · K_write · D_concept` (fixed)
- Previous-window buffer: `256 · D_lm` (fixed)

No structure grows over time. The manifold is a fixed-capacity
associative memory — older content is overwritten as repeated visits
scatter_mean newer states over the same concept slots. Long-range
recall is *gist*, not *transcript*. For verbatim recall of specific
items (names, IDs, etc.) a separate non-overwriting store would need
to be added on top — not in scope for v1.

### 5.2 Ground-truth vs predicted writes

| Source                          | Write conditions on |
|---------------------------------|---------------------|
| User message                    | ground truth        |
| Tool / system output            | ground truth        |
| Assistant turn (just generated) | predicted hiddens   |

Anything *given* to the model is ground truth — write directly. Only
the model's own generations have the predicted-hidden distribution
shift.

### 5.3 Multi-turn agent loops

Each user/tool input is one or more windows — written as ground truth.
Each assistant generation is one or more windows — written from
predicted hiddens. Manifold persists across turns; long sessions
accumulate session memory.

Long horizons stress the lossy-compression property: early facts can
be overwritten by repeated visits later. Treat as the architecture's
known limit.

### 5.4 Deployment-mode behavior — state lifecycle and surprise

Operational behavior differs across deployment modes in two axes:
(a) what gets reset when, and (b) whether surprise is computable.

**`concept_states` lifecycle:**

| Boundary                          | `concept_states` action |
|-----------------------------------|-------------------------|
| Window → next window              | persist                 |
| TBPTT depth boundary (training)   | persist (gradient cut)  |
| Training sequence end             | reset to `state_init`   |
| Document end (TF training)        | reset (align with seq)  |
| Inference single request end      | reset                   |
| Conversation turn end             | persist                 |
| Conversation session end          | reset                   |
| Per-user persistence (v2)         | save and reload         |

**Surprise availability:**

| Window type                                   | Surprise              |
|-----------------------------------------------|-----------------------|
| TF NTP target window (any role)               | Mean per-token CE     |
| User / tool / system input window             | Mean per-token CE     |
| AR-generated assistant window                 | 0 (or entropy proxy)  |
| AR fine-tune Pass 2 (rewind-and-re-forward)   | Mean per-token CE     |

**Per-mode gotchas:**

- **TF pretraining**: align training sequences with documents to avoid
  cross-document leakage in the manifold.
- **AR inference (single-shot)**: surprise is 0; plasticity still fires
  via mutations at "default strength."
- **Chat / multi-turn**: surprise drives plasticity strongly on
  user/tool inputs (rich signal); weakly on assistant generations
  (defaults).
- **Agentic**: same as chat, longer horizons expose lossy-compression
  limits earlier.

`concept_ids` and `edge_indices` are parameters — they live with the
model and don't have a "session lifecycle." Only `concept_states`
resets across deployment-mode boundaries.

---

## 6. Implementation plan

### 6.1 File layout

New package `src/trajectory_memory/`:

```
src/trajectory_memory/
├── __init__.py
├── manifold.py             # Manifold class — concept_ids, concept_states, edge_indices
├── read_module.py          # ReadTrajectoryGenerator — J parallel autoregressive
├── write_module.py         # WriteTrajectoryGenerator — J parallel + scatter_mean persist + mutate_write
├── integrated_lm.py        # IntegratedLM with reused MemInjectLayer + surprise pooling
├── tbptt.py                # multi-window TBPTT scaffolding + checkpointing
└── config.py               # TrajMemConfig + factories (small / medium / large)

tests/                      # at repo root, per project convention
├── test_trajectory_memory_manifold.py
├── test_trajectory_memory_read.py
├── test_trajectory_memory_write.py
├── test_trajectory_memory_smoke.py        # IntegratedLM + TBPTT (no Llama)
└── test_trajectory_memory_surprise.py     # _compute_surprise math
```

Reuse from existing code:

- `src/pretrained/hosts/llama.py` — Llama wrapper (already vocab-agnostic).
- `src/pretrained/mem_inject_layer.py:MemInjectLayer` — drop-in reuse.
  KV source becomes our flattened read trajectory.

Landed in this repo (originally specced as future-ports from
`abandoned/graph-walker`, all now reimplemented or no longer needed):

- `src/trajectory_memory/training/plotting.py` + per-step history dict
  in `train_wave1.py` / `train_wave2.py` — telemetry + 9-panel PNG.
- `src/trajectory_memory/training/phase1.py` — Phase 1 trainer scaffold.
- `src/trajectory_memory/training/phase2.py` — Phase 2 (GRPO) trainer
  scaffold (replaces the old `grpo_session_step`).
- `src/trajectory_memory/training/loaders.py:TurnPairDataset` — long-chat
  TurnPair extraction + length-bucket batching.

### 6.2 Build order

1. **Manifold** (`manifold.py`): just data + neighbor lookup.
   - `init_small_world_ring(N, K_max, p_rewire, radius, D_concept)` —
     initialize edges via Watts-Strogatz on a ring.
   - `concept_states`, `concept_ids`, `state_init` as parameters.
   - `get_neighbor_ids(concept_id) -> [K_max, D_concept]`.
   - `scatter_mean_states(prev_states, visited_ids, visited_states) ->
     new_states` (functional, returns fresh tensor for autograd safety).
   - **Test:** state in/out, neighbor-mask correctness, scatter_mean
     idempotence on empty input, ring-rewire structural properties.

2. **Read module** (`read_module.py`): forward only, no plasticity.
   - J parallel head queries, J entry concepts.
   - K_read autoregressive hops with Gumbel-top-1 STE.
   - Positional encoding added to history-attn keys/values.
   - Returns `[J, K_read, D_concept]`.
   - **Test:** gradient flows from output back to manifold tensors used;
     `concept_states` is unmodified after a read pass.

3. **Write module** (`write_module.py`): forward + functional scatter_mean.
   - Includes `mutate_write` MLP (the only mutation function in the system).
   - Same hop machinery as read but with persistent state update via
     functional scatter_mean.
   - **Test:** after write, `concept_states_new` reflects scatter_mean
     of all J·K_write proposals; collisions average correctly;
     `concept_ids` is untouched; original `concept_states` tensor is
     not mutated in place.

4. **Integrated LM** (`integrated_lm.py`): wraps Llama + reused
   MemInjectLayer + read + write + manifold. Defines `forward_window`
   and includes surprise pooling logic.
   - **Test:** single-window forward end-to-end, output shape sane,
     2K hard-truncation behaves correctly when input < 2K.

5. **TBPTT** (`tbptt.py`): multi-window forward with checkpointing.
   - `forward_chunk(windows: List[TokenIds]) -> losses, final_concept_states`
   - Linear-in-D first; constant-in-D as fallback.
   - **Test:** D=2, gradient on window 0's write module params is
     non-zero when only window 1's loss is backpropagated.

6. **Training loop** (LANDED): see
   `src/trajectory_memory/training/phase1.py` (Wave 1 + Wave 2) and
   `src/trajectory_memory/training/phase2.py` (Wave 3 + Wave 4 GRPO).
   Entry scripts at `scripts/train_wave{1,2,3,4}.py`.

7. **Telemetry** (LANDED): trajectory diversity, surprise distribution,
   inject SNR, per-component grad norms, throughput/VRAM, read/write
   `unique_frac` + `self_overlap`. See `src/trajectory_memory/training/
   plotting.py` + per-step history dict in `train_wave1.py`. Dashboard
   is a 9-panel PNG refreshed every `--plot-every-seconds`.

8. **Long-context evals**: synthetic recall task — embed a fact early
   in the doc, query at a position far enough that Llama's 2K sliding
   context can't see it. Memory should bridge the gap.

### 6.3 Tests

- **Unit (CPU):** manifold ops, single hop, single window forward,
  single window write persistence.
- **Forward (CPU, BS=1, T=512=2 windows):** full per-window cycle,
  output sanity.
- **Backward (CPU, D=2):** loss in window 1 produces non-zero gradient
  on write module params used in window 0.
- **scatter_mean correctness:** synthetic test where 3 of J trajectories
  visit the same concept — verify the resulting state is the mean.
- **Functional update:** verify that after a write, the original
  `concept_states` tensor is unchanged (functional contract).
- **VRAM (GPU, target config):** confirm OOM headroom at D=4, BS≥2.
- **Integration (GPU, small training run):** does NTP loss decrease on
  a long-context synthetic task that requires memory.

### 6.4 Migration from current graph_walker

The existing graph_walker on `main` shares plumbing the new design can
reuse:

- `IntegratedLM` and `LlamaHost` — keep; replace memory module.
- `MemInjectLayer` — **reused directly** (same shape, KV is now our
  flattened read trajectory rather than walker readout).
- `update_plasticity` external trigger — gone; plasticity is internal
  to the write module's scatter_mean.
- Telemetry — reimplemented in `src/trajectory_memory/training/plotting.py`.
- Phase 1 trainer — reimplemented in
  `src/trajectory_memory/training/phase1.py`.
- `session_to_turn_pairs` + Verlog-style length-bucket batching —
  reimplemented in `src/trajectory_memory/training/loaders.py:TurnPairDataset`.
- `grpo_session_step` — superseded by `Phase2Trainer.step` in
  `src/trajectory_memory/training/phase2.py` (two-pass GRPO).

Trajectory-memory has been promoted to `main`. The earlier graph_walker
production state is preserved on `abandoned/graph-walker` (commit 3b69366);
all graph_walker / column_graph / model / pretrained-extras code was
removed from `main` in the post-promotion purge. To inspect or revive
graph_walker's plumbing (e.g., to port `session_to_turn_pairs`,
length-bucket batching, or `grpo_session_step` for the trainer scaffold),
check out `abandoned/graph-walker`.

### 6.5 Evaluation suite

Memory has to actually work. The eval suite directly measures the
architecture's intended capability — long-range retrieval and use.

**Memory probes (architecture-specific, run every checkpoint):**

- **MemoryAgentBench**: gold-standard for memory-augmented agents —
  retrieval, test-time learning, long-range understanding, selective
  forgetting via multi-turn interactions on long-context data.
- **LongMemEval**: long-context memory; fact retention and multi-hop
  recall.
- **LoCoMo**: long-conversation memory.
- **NarrativeQA**: long-passage QA, naturally tests bridging across
  the 2K cap.
- **Synthetic needle-in-haystack** at varying distances (2K, 8K, 32K,
  128K). Most direct measurement of "did the manifold encode this and
  can the model retrieve it." Build a small, self-contained version
  for fast iteration during training.

**General capability (sanity check, end of each wave):**

- MMLU subsets — confirm memory isn't degrading base reasoning.
- HumanEval, GSM8K — gauge reasoning capability after each wave.

**Ablations (publishable framing, end of training):**

- Vanilla Llama-3.2-1B (no memory module) — baseline.
- Trajectory-memory after Wave 1 only (memory-pretrained, no SFT).
- After Wave 1 + 2 (no GRPO).
- After Wave 1 + 2 + 3 (no agentic GRPO).
- Full pipeline (all 4 waves).
- Optional: graph_walker baseline (current production lineage) for
  direct architectural comparison.

**Memory utilization metrics (training-side telemetry, continuous):**

WIRED as of 2026-05-10 (see `train_wave1.py` history dict + plotting.py
9-panel PNG). Per-step:

- **`loss`** — train NTP CE (token-weighted average across chunk; N6 fix).
- **`grad_norm` (overall and by component)** — `bridge_in_llama`,
  `read`, `write`, `manifold`, `llama_other`. The pre-Phase-D
  writer-zero-grad bug would have shown as flat `grad_norm_write`.
- **`inject_snr` = ‖scale·W_out(readout)‖ / ‖hidden‖** — memory-module
  contribution diagnostic. Read from MemInjectLayer's `_last_inj_norm`
  / `_last_hidden_norm` buffers. Plot panel 3. Flat at zero = memory
  silently collapsed (scale → 0).
- **`read_unique_frac`, `write_unique_frac`, `read_self_overlap`,
  `write_self_overlap`** — per-step trajectory diversity (computed
  from `read_visited`/`write_visited` IDs across windows). Collapse
  to identical paths is a failure mode.
- **`surprise_mean / surprise_std`** — per-window NTP CE distribution
  (writer's input). Flat or rising = memory not learning to predict.
- **`tok_per_sec`** — throughput (regression detector).
- **`vram_peak_gb`** — memory creep detector.
- **LR per param group** — schedule sanity check.

Per-save-step (validation):
- **Per-source val loss** — needle val plotted distinctly as the
  memory-bridging probe. Other sources (FineWeb, Wikipedia, SlimPajama)
  side-by-side. State THREADS across val chunks of the same doc (N9
  fix) so the needle val actually measures cross-chunk memory ability.

Per-step alerts:
- **NaN-loss kill switch** — `assert math.isfinite(loss)` →
  `sys.exit(1)`. Same for `grad_norm`.
- **Grad-spike alert** — warn if grad_norm > 5× rolling-100 median.

Phase 2 specific (`Phase2Metrics`):
- **`clip_fraction`** — % of tokens where IS ratio was clipped (PPO).
- **`mean_ratio`** — mean(ratio); should stay ~1 if policy isn't drifting.
- **`kl_to_ref`** — K3 estimator of KL(π_θ ‖ π_ref).
(currently logged to stdout only — full plot wiring is a known gap.)

Not yet wired (potential future work):
- Cumulative dead-concept tracking (concepts never visited).
- Routing entropy (per-hop softmax distribution).
- scatter_mean cross-trajectory collision rate.
- Per-window surprise within chunk (D-axis structure currently aggregated).

**Cadence summary:**

| Eval                          | When                                      |
|-------------------------------|-------------------------------------------|
| Memory probes                 | Every N training steps (every checkpoint) |
| General capability sanity     | End of each wave                          |
| Ablation snapshots            | End of training                           |
| Telemetry metrics             | Continuous during training (logs)         |

If memory probe scores stagnate while NTP loss drops, the model has
learned to ignore memory — surface this as a hard failure signal,
revisit data mix or architecture before continuing.

---

## 7. Open questions and risks

**Pure-graph autocomplete stability.** With edge weights gone,
autocomplete relies entirely on `concept_states` (mutated by past
writes) feeding into the read module's Q construction. A single bad
write to a concept that's later visited can poison subsequent reads of
that concept until it's re-mutated. Mitigation: `prev_window_hiddens`
cross-attention is the primary read signal — the LM-side query
dominates over potentially-corrupted concept state. If retrieval still
proves unstable, edge weights are the easiest thing to add back.

**Cross-window TBPTT memory cost.** D=4 with checkpointing should fit
a 24GB GPU at BS=2. D=8 may need constant-in-D autograd. Bench early.

**J trajectory collapse.** Without strong per-j differentiation, the J
trajectories may collapse to the same path. Mitigation: head queries
(specced); also monitor trajectory diversity in telemetry. If collapse
persists, add an explicit diversity loss (penalty on inter-trajectory
cosine similarity) or initialize `head_query` with opposite-sign biases.

**scatter_mean dilution.** When J trajectories disagree on how to
mutate a concept, simple averaging dilutes high-confidence proposals
when paired with low-confidence ones. Surprise enters `mutate_write`,
so proposals are *implicitly* surprise-weighted via the new_state's
magnitude. If dilution is empirically a problem, switch to:
- scatter with per-trajectory confidence weighting (a learned scalar),
  OR
- pick-largest-norm (effectively "winner takes the write").

**Functional scatter_mean implementation.** PyTorch's autograd with
in-place tensor updates is version-counter sensitive — naive
`concept_states[i] = new_state` inside a TBPTT loop can fail at backward.
The functional `scatter_mean(prev, ids, vals) → new` form sidesteps
this. Implementation pitfall to test for explicitly.

**Read–write co-adaptation collapse.** With separate parameters and
joint training via NTP, there's a risk that read and write co-adapt to
a degenerate equilibrium (both modules ignore the manifold and route
information through Llama's residual stream). Mitigation: if NTP
improvement vanishes on long-context evals, add an auxiliary loss on
memory utilization (concept state variance, read trajectory diversity)
to prevent collapse.

**Exposure bias on writes (AR fine-tuning).** During AR fine-tuning,
write conditions on Llama's predicted hidden states, off-distribution
from training. Mitigation: rewind-and-re-forward (§4.6).

**J·K cross-attn cost during training.** With J=4, K=8, D=4 windows,
BS=2 → 4·8·4·2 = 256 cross-attn calls per backward. Each is small
(query=1, KV=256), but adds up. Bench this early; if expensive, batch
the J·K hops more aggressively or reduce J.

---

## 8. Future direction: concept-space NCP

Once the manifold is trained and stable, the trajectory generator's
machinery can drive a separate model that operates **purely in concept
space**: next-concept prediction (NCP) instead of next-token prediction.

```
text corpus
   ▼
trained trajectory-memory model (inference mode)
   ▼
extract (text_window → write_trajectory) pairs
   ▼
train NCP model: predict next concept given history of concepts
```

Hierarchical decoder: concept-level planning + token-level rendering.
Same shape as recent latent-CoT / Coconut work, but with discrete graph
structure instead of continuous latents — stronger inductive bias, more
interpretable.

The manifold is also a candidate cross-modal substrate: if image or
audio encoders later map into the same manifold, those modalities share
a "concept language" with text.

Out of scope for v1. Worth keeping in mind so the manifold and
trajectory APIs stay clean enough to support this later (manifold should
be serializable independently of Llama; trajectory generator should be
runnable without Llama if conditioned on a non-text source).

**Lifelong / continual fine-tune (also v2):** take the trained model
and fine-tune on long continuous streams *without* `concept_states`
reset — accumulating session-style memory across the entire fine-tune.
A continual-learning study, not from-scratch training.

---

## 9. Delta from current `graph_walker`

| Concept                | graph_walker (main)               | trajectory-memory (this design) |
|------------------------|-----------------------------------|---------------------------------|
| Routing granularity    | Per token                         | Per window (256 tokens)         |
| Walker state vector    | Yes — recurrent, per-token        | Gone; trajectory IS the state   |
| Neuromodulator         | Yes — separate module             | Gone; surprise is direct input  |
| Read/write modules     | Shared (one walker)               | Separate, no shared params      |
| Read conditioning      | Per-token h_t                     | Pooled prev_window_hiddens      |
| Write conditioning     | Per-token h_t                     | Current_window_hiddens + surprise |
| Trajectories per win.  | 1 (the walker)                    | J parallel (e.g. J=4)           |
| Trajectory length      | T (one hop per token)             | Fixed K_read / K_write (~8)     |
| Plasticity firing      | Per-token (with phase gating)     | Per-window (chunk boundary)     |
| Identity vector        | Separate from state               | Separate (concept_ids vs concept_states), same dim |
| Edge weights           | Yes — Hebbian, edge-bias routing  | Gone — pure adjacency only      |
| Routing scoring        | Q vs walker readout + Hebbian     | Pure QK: Q vs neighbor concept_ids |
| Mutation function      | mutate-on-visit                   | `mutate_write` only (no read mutation) |
| Mutation conflicts     | N/A (per-token, no parallel)      | Functional scatter_mean         |
| Order signal           | Implicit in walker recurrence     | Sinusoidal positional encoding  |
| Role embeddings        | None                              | None (Llama's chat tokens suffice) |
| TBPTT scope            | Within T (one sequence)           | Across D windows (cross-window) |
| Inference write trigger| Always (per token)                | Per window, ground-truth-aware  |

The core idea — concepts as nodes, plasticity-driven memory, manifold
for retrieval, Llama as backbone — is preserved. What changes is
granularity, which auxiliary state vectors exist, and how plasticity
is realized.

---

## 10. Efficiency analysis: speed and VRAM

> **2026-05-10 update:** all sub-sections below were written before the
> KV-cache and shared-prefill landings. Headline-number sections (§10.1
> VRAM, §10.2 per-window time, §10.4 priorities) have been retro-edited
> to match the bench reality. **For current production numbers always
> consult `docs/bench_results.md` — that doc is the source of truth;
> this section is design rationale.**

Current `medium` config (§4.4 table): N=4096, D_concept=256, K_max=64,
J=4, K_read=K_write=8, D=4, T_window=256, Llama-3.2-1B bf16,
FlashAttention enabled, RTX 4090. Phase 1 BS=4 max, Phase 2 K=16 max.

### 10.1 VRAM breakdown

**Parameter memory (persistent):**

| Component                      | Size                       | Notes                                |
|--------------------------------|----------------------------|--------------------------------------|
| Llama-3.2-1B (bf16, frozen)    | ~2.5 GB                    | No optimizer state                   |
| concept_ids (fp32)             | N·D_concept·4 ≈ 2 MB       | Trainable                            |
| concept_states (fp32 buffer)   | ~2 MB                      | Mutable activation-like state        |
| state_init (fp32)              | ~2 MB                      | Trainable parameter                  |
| edge_indices (int32)           | ~256 KB                    | Fixed at init                        |
| Read + write modules           | ~10 MB                     | MLPs + cross-attn weights            |
| MemInjectLayer adapter         | ~1 MB                      | W_in, W_out, scale, cross-attn       |
| **Total params**               | **~2.5 GB**                | Llama dominates                      |

**Optimizer state:** trainable params ≈ 20 MB; Adam (m + v) ≈ 40 MB.
Trivial vs Llama's frozen 2.5 GB.

**Activations during training (BS=2, D=4 windows, per-block Llama checkpointing):**

| Component                                | Size estimate    | Notes                              |
|------------------------------------------|------------------|------------------------------------|
| Llama activations (1 window, FlashAttn)  | ~30–60 MB        | Per-block ckpt → recomputed in backward |
| Read trajectory activations (D windows)  | ~20–40 MB        | J·K_read hops × small MLPs         |
| Write trajectory activations (D windows) | ~20–40 MB        | Same scale as read                 |
| Cross-window write graph skeleton        | ~10 MB           | Visited-concept refs, scatter_mean autograd nodes |
| KV cache (2K context, bf16)              | ~250–500 MB      | Llama-1B specific                  |
| **Working set with checkpointing**       | **~400–700 MB**  |                                    |
| Without Llama checkpointing              | **~1.5–2.5 GB**  | D × per-window activations         |

**Measured VRAM in production (2026-05-10 bench, KV cache + no grad
checkpointing):**
- Phase 1 BS=4: **~15 GB peak** (T1 trajmem step). BS=5 would fit; BS=8 OOMs.
- Phase 2 K=16: **~13 GB peak** (T2 GRPO with shared prefill).
- KV cache is the dominant live tensor (~2-4 GB for 2K context).

**KV cache vs gradient checkpointing — mutually exclusive.** HF's
Llama silently sets `use_cache=False` when gradient checkpointing is
on, returning `past_key_values=None`. We picked the cache (~1.79×
speedup, 5 GB less peak in practice) over the activation savings; see
the NOTE in `scripts/train_wave1.py`. If a config-tier-large run
needs gradient checkpointing, it must also pass `--no-kv-cache`.

The earlier "3.5–5 GB" estimate was the pre-KV-cache projection with
gradient checkpointing on. Reality with KV cache + no checkpointing
runs 11-15 GB on `medium`.

**Inference (single user, no autograd):** Llama params + KV cache +
manifold + transient trajectory states ≈ 3 GB total. Easily fits on
consumer GPUs.

### 10.2 Per-window time breakdown

> **Superseded:** the rough estimates below were pre-KV-cache. See
> `docs/bench_results.md` for measured numbers. Current Phase 1 throughput
> is ~17.7K tok/s (5× the projection here), and trajmem is now FASTER
> than vanilla per-token because the per-window Llama forward only
> encodes the new T_window=256 tokens against the cached prefix
> instead of re-encoding the rolling buffer.

Architectural rationale (still valid): per-window cost is dominated by
Llama's transformer forward; trajectory ops are small but launch-bound.
The KV cache eliminated the rolling-buffer re-encode that was ~70% of
the T1-vs-V1.B gap (per `docs/profile_analysis.md`).

### 10.3 Bottlenecks

**Primary: Llama forward.** ~80% of step time. Frozen, but compute is
unavoidable. Mitigations LANDED:
- FlashAttention (default).
- bf16 backbone.
- **Sliding KV cache** (was specced as "v2 optimization" here; landed
  2026-05-10). Eliminated rolling-buffer re-encode; closed the entire
  T1-vs-V1.B gap and made trajmem faster than vanilla per-token.

**Secondary: trajectory hop dispatch overhead.** Each hop has a few
small kernel launches (history-attn, cross-attn, MLP, neighbor gather,
softmax). At J=4, K=8, D=4 → ~256 small launches per training step.
Mitigations LANDED:
- Batched J cross-attn per hop.
- `torch.compile(model.forward_window, dynamic=True)` (default ON in
  Wave 1/2 entry scripts).

**Tertiary: cross-window TBPTT memory.** D windows of write graph alive
during backward. Currently linear-in-D — fits with KV cache + no
gradient checkpointing at BS=4. Constant-in-D autograd remains the
fallback for D≥8 / large-tier configs.

**Phase 2 specific:** ~30% slowdown vs vanilla GRPO at K=16 (132.8 vs
173.6 tok/s in current bench). The remaining cost is read+write per
generation window (architecturally unavoidable — that's the side-car).
Bigger win available from batched K rollouts (currently serial), see
§10.4.

### 10.4 Optimization plan, priority-ranked

**LANDED (2026-05-10):**
1. ✅ bf16 + FlashAttention on Llama backbone.
2. ✅ Sliding KV cache for Llama context (was #4 here, promoted to #1
   in practice; 1.79× Phase 1 speedup, eliminated T1-vs-V1.B gap).
3. ✅ Regional + dynamic `torch.compile(forward_window)` (~28% at low BS).
4. ✅ Phase 2 shared-prefill (K-1 prompt encodings saved per step).
5. ✅ Phase 2 two-pass GRPO (single backward at end of TF replay).

**STILL OPEN (priority-ranked):**
1. **Phase 2 batched-K rollouts** — current AR sampling loop generates
   K samples serially (Python loop over K). Highest-value Phase 2
   speedup remaining. Estimated: 2-3× pass-1 wall time reduction.
   Implementation: medium (need to expand memory states + KV cache to
   K batch slots, maintain active/eos mask).
2. **Phase 2 routing-frozen fix** — N3's `hard_routing=False` in pass 2
   means routing modules (entry_mlp, step_mlp, head_query) get no
   gradient. Proper fix: record routing decisions in pass 1, force
   them in pass 2. ~150-200 LOC. Phase 2 currently only refines
   bridge + cross-attn + writer mutate_mlp.
3. **Constant-in-D autograd** — fallback for D≥8 if linear-in-D OOMs;
   sizable engineering cost. Not blocking; D=4 at BS=4 fits with KV
   cache + no checkpointing.
4. **Reference-policy frozen side-car copy** — `_compute_ref_logps`
   currently param-swap-loads ref weights every Phase 2 step. Keep a
   permanent frozen-copy of trainable side-car params on GPU instead.
5. **Triton-fused hop kernel** — only if profiling shows dispatch is
   the bottleneck. Almost certainly premature (see note below).

**Note on custom Triton kernels — not needed for v1.** graph_walker
required custom Triton (dendritic gather, fused per-token plasticity)
for two architecture-specific reasons that do not apply here:

- Per-token sequential dispatch produced ~20K-40K kernel launches per
  step; torch.compile couldn't fuse across iterations because of the
  per-token dependency chain.
- The dendritic gather kernel avoided materializing a `[BS, N, K, D]`
  intermediate — a memory-bandwidth problem, not just launch overhead.

Our trajectory architecture has neither: per-step dispatch is ~50× less
(per-window hop sequence instead of per-token), and the biggest tensor
we touch is `concept_states[N, D_concept]` ≈ 2048×256, trivial. Regional
`torch.compile` on the hop module should suffice — Triton stays as an
"if profiling tells you to" v2 option, not a v1 requirement. This is a
real architectural simplification worth not paying for prematurely.

### 10.5 Compared to graph_walker

graph_walker is launch-bound: per-token sequential dispatch through the
walker. Our trajectory architecture moves dispatch from per-token to
per-window — ~256× fewer hop dispatches per token, since each window's
J·K hops produce content for all 256 tokens that follow.

In return, each "memory event" is a J·K hop sequence rather than a
single hop, but that's still much smaller than 256 per-token hops.
Llama's forward is also more expensive per-window because we cap at 2K
context (vs graph_walker's smaller windows), but that's the load-bearing
training pressure — sacrificing it would defeat the architecture.

Expected outcome: comparable raw throughput vs graph_walker, with
**much cleaner gradient flow** (cross-window TBPTT on the actual write
module, vs graph_walker's per-token surprise gating). The architectural
win is gradient quality and training simplicity, not speed.

---

## Appendix A: Glossary

- **Manifold:** the graph of N concepts (with id + state vectors) +
  sparse edges. The model's long-term memory substrate.
- **Concept:** one node in the manifold. Has a stable id_vec (routing
  key) and a volatile state (content payload).
- **`concept_ids`:** `[N, D_concept]` tensor. Stable. Updated only via
  backprop. Used as routing keys in neighbor scoring.
- **`concept_states`:** `[N, D_concept]` tensor. Mutable. Updated via
  functional scatter_mean from write trajectories, and via backprop
  through paths that consume them.
- **`state_init`:** `[N, D_concept]` learnable parameter. Reset target
  for `concept_states` at training-sequence start / inference session
  start.
- **`edge_indices`:** `[N, K_max]` tensor. Sparse adjacency. Fixed at
  init via small-world ring rewire.
- **Trajectory:** an ordered sequence of K visited concepts produced
  by a read or write module. J parallel trajectories per window.
- **Window:** a 256-token chunk. The fundamental unit of time.
- **Read:** at start of window, autocomplete J trajectories using
  previous window's tokens. Output flattened and injected into Llama
  via the reused MemInjectLayer.
- **Write:** at end of window, generate J trajectories conditioned on
  current window's tokens + surprise. Functional-scatter_mean the
  proposed state mutations into the manifold.
- **`mutate_write`:** the per-visit mutation MLP used during writes.
  Only mutation function in the system — read has none.
- **Surprise:** mean per-token NTP cross-entropy of the just-predicted
  window over target-eligible tokens (e.g., assistant tokens only in
  multi-turn). Gates write mutation strength.
- **TBPTT depth D:** number of windows over which gradient flows. D ≥ 2
  required for write module to receive any signal.
- **J:** number of parallel trajectories per read/write. Multi-head
  analog. v1 fixed; adaptive J is v2.
- **Gumbel-top-1 STE:** discrete next-concept selection during
  training. Hard forward (one concept), soft backward (gradient through
  the softmax). Argmax at inference.

---

## Appendix B: Scaling knobs (capacity dials)

Hyperparameters that tune model capacity. Listed roughly in order of
impact on capability (vs operational knobs like LR, batch size, warmup
which live in §4.4):

| Knob                   | Controls                                       | Range          | Cost shape                              |
|------------------------|------------------------------------------------|----------------|-----------------------------------------|
| `N` (manifold size)    | Number of distinct concepts                    | 1024–16384     | Linear memory (params), trivial compute |
| `D_concept`            | Per-concept rep dim (id + state share dim)     | 128–512        | Quadratic on cross-attn cost; expressivity |
| `J`                    | Parallel trajectories per window               | 1–16           | Linear in compute; multi-head analog    |
| `K_read` / `K_write`   | Trajectory length                              | 4–32           | Linear in compute; deeper manifold reach |
| `K_max_neighbors`      | Sparse degree per concept                      | 8–64           | Linear in scoring cost; routing flexibility |
| `D` (TBPTT depth)      | Cross-window gradient horizon                  | 2–16           | Linear in train-time VRAM (no inference cost) |
| Llama base size        | LM backbone capacity                           | 1B / 3B / 8B   | Most expensive; affects everything      |
| Inject layer L         | MemInjectLayer position in Llama stack         | 4–24           | Earlier = more LM-side processing after; later = quicker memory access |
| Effective LM context cap | Sliding window for Llama                     | 1K–8K          | Smaller = stronger memory pressure, harder to converge |
| `T_window`             | Window size in tokens                          | 128–512        | Smaller = finer plasticity; larger = more context per memory event |

**Scaling rules of thumb:**

- `K ≈ √N / 4` — trajectory length scales with manifold size so each
  trajectory covers a meaningful fraction of the graph.
- `K_max_neighbors ≈ √N / 2` — neighborhood grows with N to preserve
  small-world graph diameter.
- `D_concept ≈ D_lm / 8` — concept dim tracks LM hidden dim.
- `J` is multi-head — independent of N. 4–8 sweet spot.
- `D` (TBPTT depth) is bounded by VRAM, not by capacity needs.

**Concrete tier presets:**

| Tier     | N     | D_concept | K_read=K_write | J   | Memory params | Llama base | Use case                |
|----------|-------|-----------|----------------|-----|---------------|------------|-------------------------|
| smoke    | 1024  | 128       | 4              | 2   | ~2 MB         | 1B         | CPU smoke tests, debugging |
| **medium** | **4096** | 256   | 8              | 4   | **~16.5M (post-bump)** | 1B | **v1 default (§4.4 — N=4096 / K=64)** |
| large    | 8192  | 256       | 16             | 8   | ~33 MB        | 3B         | Post-v1 scale-up        |
| xl       | 16384 | 512       | 32             | 8   | ~100 MB       | 8B         | Hypothetical ceiling     |

The memory-side parameter count is **trivial** relative to the Llama
backbone in all tiers — capacity scaling is essentially free in
parameters; the real cost is in compute (J·K trajectory hops per
window) and TBPTT memory (D windows alive during backward).

**What to scale first when you have more compute, in order:**

1. `D_concept` (more expressive concepts) — small change, big effect.
2. `J` (more parallel retrieval / encoding) — multi-head capacity.
3. `K_read = K_write` (deeper trajectories) — longer reach per window.
4. `N` (more concepts) — diversifies the substrate; cheap.
5. Llama base size — most expensive, but biggest direct capability boost.
6. `D` (TBPTT depth) — only worth scaling if you can afford the VRAM
   AND see continued improvement on long-horizon evals.

What NOT to scale up casually: `K_max_neighbors` (more is rarely
helpful past a small value), `T_window` (288, 320 don't matter much vs
256; deviating breaks alignment with chat-template structure), effective
LM context cap (raising it weakens memory training pressure).
