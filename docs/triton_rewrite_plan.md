# GraphWalker Triton Rewrite Plan

**Branch:** `graph-walker` @ `00d4084` (or later)
**Status:** Phases 0+1+5 landed (target speedup hit); Phases 2-3 deferred — see "Outcome" below
**Author:** session 2026-04-25
**Goal:** Push walker hot path from current 3.5× over eager (whole-block compile) to **6-8×** via custom Triton kernels for the dispatch-bound small-op chains, keeping cuBLAS for matmuls.

---

## Outcome (2026-04-25 — same session)

**Target hit on Phase 5 alone:** manual CUDA-graph capture wrapped around an
inductor-compiled ``block_forward`` reaches **13.98× over eager** at B=4,
T=128, and **13.20× at B=16** — well past the 6-8× target.

Reproduce with ``PYTHONPATH=. python scripts/bench_walker_full.py [B]``
or ``scripts/bench_walker_quick.py cudagraph [B]`` for a single config.

```
Bench (RTX 4090, segment_T=128, mod_period=64, use_neuromod=False)

B=4:    eager                       1260 tok/s    1.00×    warmup  2.0s
        whole-block compile         5049 tok/s    4.01×    warmup 14.3s
        cudagraph + compile inner  17615 tok/s   13.98×    warmup  101s

B=16:   eager                       5028 tok/s    1.00×
        cudagraph + compile inner  66358 tok/s   13.20×    warmup  251s
```

Compile + cudagraph capture is a one-time ~100-250s warmup (longer for
larger B due to Triton autotune). After that each replay is pure CUDA —
Python dispatch overhead drops to noise.

Why Phase 2 (`step_postlif`) and Phase 3 (`anchor_pick`) Triton kernels
were **not** implemented:

- Their value proposition was reducing kernel-launch overhead in the
  small-op chain. CUDA-graph replay collapses the entire forward+backward
  into a single replayed graph — launch overhead drops by ~100×, so the
  remaining headroom from Triton fusion is in the noise (~1-2%).
- Implementing them as designed (custom Triton forward + analytical
  backward) would be ~700-1000 lines + parity tests, with negligible
  speedup contribution. The 8.7× regression in commit ``a2f3f50`` from
  putting matmul-shaped backward in Triton is the cautionary precedent.
- Phase 4 (integration) and the autograd.Function wrappers (originally
  meant to make capture clean) became unnecessary once we discovered
  manual capture works without them.

**Recommendation:** keep this section as the canonical record. Re-open
Phases 2-3 only if profiling at production scale shows a remaining
launch-bound or memory-bound bottleneck inside the captured graph.

### What did land

- `src/graph_walker/triton/__init__.py` — package marker, lazy
  `HAS_TRITON` flag.
- `src/graph_walker/triton/lif.py` — Triton sparse LIF deposit with
  static-shape preprocessing (no `torch.unique`), pre-allocated
  save-for-backward buffers, sentinel-padded unique_dests grid. Pure-
  torch fallback retained as the cudagraph-safe default backend.
- `src/graph_walker/triton/cudagraph_trainer.py` — manual CUDA-graph
  capture wrapper. Owns all input/output staging buffers; warmup +
  capture share a side stream so AccumulateGrad nodes match. Captured
  body covers caches refresh + block_forward + CE + backward + state
  writeback + surprise EMA streaming + Hebbian plasticity update —
  all in-place against stable buffers.
- `phase1_step_cudagraph` in `train_phase1.py` — lazy build of the
  trainer on first call; replays per block; cross-block stat
  accumulation stays GPU-resident.
- Two cudagraph-capture-blocking PyTorch ops fixed in
  `_step_core_pure`: `torch.bincount` → `torch.zeros + index_add_`,
  bool-masked indexing → multiplied-mask sum.

### Constraints accepted

- ``use_neuromod=False`` is required for the cudagraph path. The
  neuromod's ``_active_delta_nm`` rebuilds per window with a fresh
  memory address; the captured graph would read from a stale pointer
  otherwise. Fixing this requires either pre-allocating a stable
  ``_active_delta_nm_buf`` that the neuromod fire writes into, OR
  routing the neuromod gradient via a checkpoint-at-boundary autograd
  pattern (capture treats e_bias as a leaf, the neuromod chain runs
  outside and consumes ``e_bias.grad``). Neither is hard, but they're
  out of scope for this rewrite.
- ``tbptt_block == mod_period`` (already required by the eager path —
  one full plasticity window per block).
- ``segment_T % mod_period == 0`` (already required).

---

## TL;DR

Don't write one giant Triton kernel. Don't put matmuls in Triton. Do this:

1. **Triton kernels** for the small dispatch-bound ops chains in each step:
   gather → softmax → gumbel → STE → scatter → EMA blend → mask. These are
   what's currently 30+ tiny PyTorch op launches per step.
2. **PyTorch nn.Linear (cuBLAS)** for the big matmuls: `content_mlp` (deep
   residual FFN), `q_proj`, `k_proj`, `out_k/v_proj`, `state_to_model`. These
   are already near-peak at our shapes.
3. **One `autograd.Function` per step** wraps the matmul + Triton chain.
   Custom `forward` and `backward` so we control save-for-backward layout
   and dtype boundaries explicitly.
4. **One `autograd.Function` for the block** sequences T_block steps with
   pre-allocated scratch buffers. This is what cudagraph-captures cleanly —
   no more autograd-vs-pool conflicts.
5. **Keep plasticity + neuromod + multi-horizon CE readout in PyTorch**,
   they're cold-path (run once per window or block) and already efficient.

Estimated 700-1000 lines of Triton + ~250 lines of autograd glue. 1-2 weeks
focused work. Target: 6-8× walker speedup, durable cudagraph compatibility.

---

## Lessons from prior Triton work in this codebase

These are non-negotiable. Future me: read these before writing any kernel.

### 1. `a2f3f50`: "Triton backward was 8.7× SLOWER than PyTorch"

Verbatim from the commit message:

> Result: 1286 μs/token Triton vs 147 μs/token PyTorch — 8.7× regression.
> End-to-end BS=64 dropped from 67K to 27K tok/s.
>
> Reason: at our shape (Nc=32, D_n=256, BS=64, NC=8), the Triton backward's
> grid (BS, NC) = 512 programs each do three small 32×32 / 32×256 tl.dots.
> Tile sizes too small for tensor cores to amortize launch + memory-load
> cost. PyTorch's analytical backward instead does 2 well-batched bmms at
> [BS*NC, Nc, Nc] / [BS*NC, Nc, D_n] which saturate tensor cores.
>
> **Forward fusion won because the per-token hot path was dispatch-bound
> (5+ small PyTorch ops per token with lots of Python overhead). Backward
> is matmul-bound and PyTorch's bmm at [BS*NC, ...] is already near-peak —
> there's no dispatch headroom to unlock.**

**Implication for our rewrite:** every matmul-shaped op stays in PyTorch
(or uses `tl.dot` only when the tile is large enough for tensor cores —
roughly `M, N, K ≥ 128` each). For `content_mlp` (BH × D_steer × D_hid,
where BH ~64 and D_hid ~1024-2048 in production), we're already in
"M too small, leave it to bmm" territory.

### 2. v9-backprop's `fused_dendritic_gather` — what worked

Found at `abandoned/v9-backprop-correctness-fixes:src/v8/triton_kernels.py`.
Pattern that worked there:
- Per-program work: ONE `(batch, neuron)` index → load K connection
  weights, gather K source vectors, per-element-multiply-and-sum, apply tanh
- PyTorch handled state MLP + message MLP as bmm
- Backward kernel: similar grid, computed analytically per program (no
  small `tl.dot`)

**Pattern:** the kernel handles **gather + weighted-sum-along-K + nonlinearity**.
This is gather-bound, not matmul-bound. Triton wins here.

### 3. v8-broadcast-io: "bf16 dtype corruption (f32 store into bf16 buffer)"

Mentioned in `MEMORY.md`. Stores must match buffer dtype. Use
`store_dtype = ptr.dtype.element_ty; tl.store(ptr + offs, x.to(store_dtype))`.
Already enforced in `triton_sparse_update.py` — keep that pattern.

### 4. The current `triton_sparse_update.py` (still in tree)

Reference implementation for: forward + backward + autograd.Function +
PyTorch fallback for CPU/no-Triton + parity tests. Uses pre-allocated
save-for-backward buffers. The thing that broke for cudagraphs was the
PyTorch-side pre-processing (`torch.unique` returns dynamic shape). We'll
avoid that by passing pre-padded buffers to the new kernel.

---

## Model invariants — what MUST be preserved

Source of truth: `docs/graph_walker.md`. Reproducing the per-step flow here
because the kernel ABI must match exactly.

### State

Per-segment (allocated by `begin_segment`, threaded through forward):
- `s [B, N, D_s]` — column state, LIF-integrated
- `walker_pos [B, H]` int64 — persistent walker positions
- `walker_state [B, H, D_s]` — walker private running state (EMA)
- `prev_motor [B, D_s]` — last step's motor output, fed into anchor query

Persistent (registered buffers, survive `detach_state`):
- `E_bias_flat [N·K]` — plastic edge biases
- `co_visit_flat [N·K]` — accumulated edge visits this window
- `visit_count [N]` — column visit tally this window
- `_active_delta_nm [N·K]` or None — neuromod delta for current window
- `_prev_snapshot_*` — neuromod's previous-window snapshot

### Per-step flow (must match `_step_core_pure` in graph_walker.py:894+)

For each token (B walkers in parallel, H walkers per batch element):

**Common prefix:**
1. `h_input = token_to_state(embed(token))` ∈ [B, D_s]

**Window-boundary only (first token of plasticity window):**
2. Anchor pick: `q_in = h_input + prev_motor_proj(prev_motor)`;
   scores against input-plane keys; Gumbel top-1 STE → `anchor_cols [B, H]`
3. Teleport: `walker_pos ← anchor_cols`
4. Anchor injection: `v_inject = input_v_proj(token_emb)` gated by anchor
   STE one-hot — added to LIF deposit at the anchor cols

**Common suffix:**
5. **Walker message.** Read `s_cur_old = s[walker_pos]` via gather. Build
   steering input `cat(s_cur_old, col_id[walker_pos], walker_state, h_input)`
   of dim `3·D_s + D_id`. Feed through `content_mlp` → `m_out [B*H, D_s]`.
6. **Sparse LIF deposit.** Walker writes `m_out` at its CURRENT column.
   On window-start steps the anchor injection stacks into the same
   call. LIF blend: `s_new[c] = α(c)·s_old[c] + (1-α(c))·tanh(Σ msgs)`.
7. **Re-read post-update:** `s_cur_new = s_new[walker_pos]` via gather.
8. **Routing query.** Build `cat(s_cur_new, col_id[c], walker_state, h_input)`
   → `q_proj` → `[B*H, n_score_heads, D_q]`.
9. **Hop scores.** Score against `k_proj(col_id[nbrs[walker_pos]])`. Add
   `active_E_bias[edge_flat]`. Gumbel top-1 STE → `next_col`.
10. **Endpoint readout.** `end_state = s_cur_new + Σ_k ste[k] · nbr_id_to_s(col_id[nbrs[k]])`.
11. **Motor.** Cross-attention over H end-states (learned `motor_query`)
    → `motor_state ∈ [B, D_s]`.
12. **Walker state EMA.** `walker_state_new = σ(α_h)·walker_state + (1-σ(α_h))·m_out`.
13. **Walker moves.** `walker_pos ← next_col`.

### Per-block bookkeeping (cold path — stays in PyTorch)

- Multi-horizon CE readout via `MultiHorizonReadout.cross_entropy_factorized`
- Plasticity update at window close (Hebbian + neuromod commit)
- Neuromod transformer (graph attention over touched columns) at window start
- Surprise EMA streaming (`accumulate_block_ce`)

---

## Architecture: kernel boundaries

### Hot path (per-step) decomposition

```
INPUT TO STEP: state, params, token_id

(A) PyTorch — token embed + first-half cat
    h_input = token_to_state(embed(token_id))           [B, D_s]
    s_cur_old = gather(s, walker_pos)                  [B*H, D_s]   <-- Triton fused later
    cat_pre = cat(s_cur_old, col_id[walker_pos],       [B*H, D_steer]
                  walker_state, h_input_per_walker)

(B) PyTorch nn.Linear — content_mlp deep stack         <-- BIG MATMUL, leave in cuBLAS
    m_out = content_mlp(cat_pre)                       [B*H, D_s]

(C) Triton kernel #1 — LIF_DEPOSIT (rewrite of existing)
    inputs:  s [B, N, D_s], all_msgs, all_dests, alpha
    outputs: s_new [B, N, D_s], saved-for-bwd buffers
    Pre-allocated buffers, no torch.unique.

(D) Triton kernel #2 — STEP_POSTLIF (gather post-update + routing scoring)
    inputs:  s_new, walker_pos, q (from q_proj), k_all (cached),
             nbrs_of_cur, e_bias_at_edges, tau, eps, rng_seed
    outputs: s_cur_new [B*H, D_s], scores [B*H, K], soft_probs [B*H, K],
             ste_weights [B*H, K], selected_idx [B*H], end_states [B, H, D_s]
    Internal: gather s_new[walker_pos], compute K scores via dot product,
              add E_bias, Gumbel sample, argmax, build STE, gather neighbor
              col_ids and nbr_id_to_s embeddings, sum-with-STE for end_states.

(E) PyTorch — motor cross-attn (small but matmul-shaped)
    Cross-attention over H end-states with learned motor_query.
    Could be Triton (see "Optional Target C" below) but cuBLAS handles small
    matmul OK via attention.

(F) Triton kernel #3 — WALKER_EMA (small fused EMA + state writeback)
    walker_state_new = σ(α_h)·walker_state + (1-σ(α_h))·m_out
    Just an elementwise op — could stay in PyTorch but keeping in Triton
    avoids one launch and one HBM round-trip.

OUTPUT FROM STEP: state_new, motor_state, m_out, deltas
```

**Why this split:**
- (B) `content_mlp` is the only matmul we explicitly leave in PyTorch — it's
  4 layers deep (residual FFN), each has GELU + RMSNorm. Inductor can fuse
  the small ops between matmuls; cuBLAS handles the matmul.
- (C) `LIF_DEPOSIT` is gather-bound. Triton wins here (proven by current
  kernel; just needs cudagraph-friendly buffer pre-allocation).
- (D) `STEP_POSTLIF` collapses ~7 small PyTorch ops (gather, einsum,
  add, softmax, argmax, one_hot, where, gather, einsum, sum) into one
  kernel. This is the biggest dispatch-overhead win.
- (F) Saves one launch; minor.
- (A) Token embed stays in PyTorch — it's an embedding lookup which is
  already efficient.

### Cold path (per-block / per-window)

```
PyTorch only:
- MultiHorizonReadout.cross_entropy_factorized   [B, T_block, D_model] -> [B, T, K_h]
- accumulate_block_ce → surprise_ema EMA
- _maybe_finalize_surprise_and_plasticity:
    - Surprise-gated Hebbian update of E_bias_flat
    - Neuromod commit (detach _active_delta_nm, fold into E_bias_flat)
    - Snapshot touched columns
- _begin_plastic_window (next window's neuromod fire)
- detach_state (TBPTT)
```

These are correct, well-tested, and not on the per-token hot path. Touch
only if profiling shows them as a bottleneck (currently they aren't).

---

## Data layout & static shapes

For cudagraph compatibility, every tensor that crosses an autograd.Function
boundary must have static shape. Pre-allocate all scratch buffers at
`begin_segment` with shapes derived from `(B, N, K, D_s, D_id, H, T_block)`.

### Static buffers allocated in `begin_segment`

```
state_buffers = StateBuffers(
    s              [B, N, D_s]               bf16
    walker_pos     [B, H]                    int64
    walker_state   [B, H, D_s]               bf16
    prev_motor     [B, D_s]                  bf16
)

scratch_per_block = ScratchBuffers(
    motor_states_bt [B, T_block, D_s]        bf16   # for readout
    co_visit_total  [N*K]                    fp32   # window accumulator
    visit_count_total [N]                    fp32   # window accumulator
    lb_loss_total   ()                       fp32   # scalar
)

scratch_per_step_for_bwd = StepScratchBuffers(
    # Pre-allocated [T_block, ...] buffers, indexed by t in the kernel.
    # Saves one alloc per step.
    s_cur_old      [T_block, B*H, D_s]       bf16
    m_out          [T_block, B*H, D_s]       bf16
    cat_pre        [T_block, B*H, D_steer]   bf16   # input to content_mlp
    s_cur_new      [T_block, B*H, D_s]       bf16
    cat_post       [T_block, B*H, D_steer]   bf16
    soft_probs     [T_block, B*H, K]         fp32
    ste_weights    [T_block, B*H, K]         bf16
    selected_idx   [T_block, B*H]            int64
    explore_mask   [T_block, B*H]            bool
    s_old_lif      [T_block, U_max, D_s]     bf16   # for LIF backward
    incoming_lif   [T_block, U_max, D_s]     bf16
    alpha_at_dest  [T_block, U_max]          fp32
    unique_dests_lif [T_block, U_max]        int64
    sort_idx_lif   [T_block, M_max]          int64
    segment_offs   [T_block, U_max + 1]      int64
)
```

`U_max` = max possible unique destinations per step = `2 * B * H` (anchor
step) or `B * H` (interior). We allocate for `2 * B * H` and pass actual U
as a tensor. `M_max` = `2 * B * H` similarly.

### Output of `block_forward`

```
BlockOutput {
    s_new           [B, N, D_s]      bf16    # written into state_buffers.s by caller
    walker_pos_new  [B, H]           int64
    walker_state_new [B, H, D_s]     bf16
    prev_motor_new  [B, D_s]         bf16
    motor_states_bt [B, T_block, D_s] bf16
    co_visit_total  [N*K]            fp32
    visit_count_total [N]            fp32
    load_balance_loss ()             fp32
}
```

Output addresses MUST be stable across replays — caller passes pre-allocated
output buffers and the kernel writes into them. (This is what fixes our
current cudagraph headache.)

### Numerical precision strategy

- **Forward state storage:** bf16 (matches current `state_dtype="bf16"`)
- **Internal Triton compute:** fp32 (load bf16, accumulate fp32, store bf16)
- **Saved-for-backward:** bf16 (storage size matters)
- **Backward grad accumulators:** fp32 (esp. atomic adds for grad_alpha,
  grad_E_bias)
- **Gumbel noise:** fp32 (numerics are sensitive at low τ)
- **Softmax:** fp32 internally (subtract max in fp32, exp, sum, divide)

Match the existing `triton_sparse_update.py` precision conventions. Already
proven in 9 unit tests.

---

## Forward kernel design

### Kernel #1: `lif_deposit_fwd`

**Signature:**
```python
@triton.jit
def lif_deposit_fwd(
    s_in_ptr,             # [B*N, D_s] bf16
    s_out_ptr,            # [B*N, D_s] bf16   <-- pre-allocated by caller
    msgs_ptr,             # [M_max, D_s] bf16
    dests_ptr,            # [M_max] int64
    M,                    # actual number of msgs (passed as scalar arg)
    alpha_ptr,            # [N] fp32
    # Outputs for backward:
    incoming_save_ptr,    # [U_max, D_s] bf16 (sum of msgs per unique dest)
    s_old_save_ptr,       # [U_max, D_s] bf16 (gathered s_in[unique_dests])
    alpha_save_ptr,       # [U_max] fp32
    unique_dests_ptr,     # [U_max] int64 (output of unique-dedup step)
    U_out_ptr,            # () int32 (actual U written here)
    N, D_s, M_MAX, U_MAX,
    BLOCK_D: tl.constexpr,
):
```

**Approach:**
- Pre-process (DONE OUTSIDE the kernel, in PyTorch but with static shapes):
  Sort `dests` (use `torch.argsort` — has static-shape output if input is
  static), find unique boundaries via `diff` + cumsum. Output:
  `unique_dests` (padded to `U_max`), `sort_idx`, `segment_offs`. Actual U
  passed as tensor.
- Forward kernel: one program per `(u, d_block)`. Loads alpha, segment-sums
  messages from `sort_idx[seg_start:seg_end]`, applies tanh, blends with
  `s_in[unique_dests[u]]`, stores to `s_out[unique_dests[u]]`.
- For `u >= U`, the program early-exits (load `U` once at top, compare).
- Non-touched rows of `s_out` must equal `s_in` — pre-copy `s_out := s_in`
  before kernel (one cudaMemcpyAsync, cheap), then kernel only modifies
  touched rows.

**Note on torch.unique:** the current implementation uses `torch.unique`
which has dynamic output shape. Replace with a fixed-shape pipeline:

```python
# All static-shape under cudagraph:
sorted_dests, sort_idx = dests.sort(stable=True)              # [M_max]
diff = torch.cat([                                             # [M_max]
    torch.tensor([1], device=device, dtype=torch.bool),
    sorted_dests[1:] != sorted_dests[:-1],
])
unique_idx_in_sorted = diff.cumsum(0) - 1                      # [M_max]
U = diff.sum()                                                 # scalar tensor
# Use scatter-via-unique_idx_in_sorted to build unique_dests, segment_offs
unique_dests = torch.empty(U_MAX, dtype=torch.int64, device=device)
unique_dests.scatter_(0, unique_idx_in_sorted, sorted_dests)   # static shape
# segment_offs[u] = first index in sort_idx belonging to u
segment_offs = torch.searchsorted(unique_idx_in_sorted, torch.arange(U_MAX + 1, device=device))
```

`U` is a tensor scalar, passed to kernel as scalar arg. Kernel programs
`u >= U` no-op.

### Kernel #2: `step_postlif_fwd`

This is the new big one. Replaces ~7 PyTorch ops.

**Signature:**
```python
@triton.jit
def step_postlif_fwd(
    s_new_ptr,            # [B*N, D_s] bf16
    walker_pos_ptr,       # [B*H] int64 (current cols)
    q_ptr,                # [B*H, H_score, D_q] bf16 (output of q_proj)
    k_all_ptr,            # [N, H_score*D_q] bf16 (cached k_proj output)
    nbrs_of_cur_ptr,      # [B*H, K] int64
    e_bias_at_edges_ptr,  # [B*H, K] fp32 (gathered E_bias[edge_flat])
    col_id_ptr,           # [N, D_id] bf16
    nbr_id_proj_ptr,      # [D_id, D_s] bf16 (nbr_id_to_s.weight, transposed)
    tau_ptr, eps_ptr,     # [] fp32
    rng_seed,             # int (fixed per step)
    # Outputs:
    s_cur_new_ptr,        # [B*H, D_s] bf16
    end_states_ptr,       # [B, H, D_s] bf16
    soft_probs_ptr,       # [B*H, K] fp32 (saved for backward + load balance)
    ste_weights_ptr,      # [B*H, K] bf16 (saved for backward)
    selected_idx_ptr,     # [B*H] int64
    explore_mask_ptr,     # [B*H] bool
    B, H, N, K, D_s, D_id, D_q, H_SCORE,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
```

**Per-program work (one program per (B*H, D tile)):**
1. Load `walker_pos[bh]` (one int64), gather `s_new[walker_pos[bh], :]`
   into `s_cur_new` (D_s floats in registers, store to s_cur_new_ptr).
2. For k=0..K: load `nbrs_of_cur[bh, k]`, gather `k_all[nbr, :]` (D_q for
   each of H_score heads), compute `score[k] = q[bh] · k_nbr[k]` (per-head
   sum). Add `e_bias_at_edges[bh, k]`. Store scores in registers.
3. Generate K Gumbel samples via `tl.rand(rng_seed, offset=bh*K + k)` and
   `-log(-log(u))`. Add to scores, divide by tau. Compute softmax →
   `soft_probs`. Argmax → `selected_idx`.
4. ε-exploration: with prob ε replace with uniform random + detached STE.
   Generate explore decision via `tl.rand(rng_seed + N, offset=bh)`.
   Generate uniform K-pick via `tl.rand(rng_seed + 2*N, offset=bh)`.
5. Build `hard = one_hot(selected_idx)` and `ste = hard - soft.detach() + soft`.
6. For endpoint: gather `col_id[nbrs[bh, k]]`, project via `nbr_id_proj`
   (small matmul D_id × D_s — INSIDE the kernel since it's tiny). Multiply
   by `ste[k]` and sum over K. Add to `s_cur_new` → `end_states[b, h]`.
7. Store all outputs to their pre-allocated buffers.

**Tile sizes:** `BLOCK_D = 64` (matches existing kernel). `BLOCK_K = K`
(K=16 typical, fits in registers). `H_SCORE = 4` (hardcoded constexpr,
re-codegen if changes).

**Why nbr_id_to_s INSIDE the kernel:** D_id × D_s = 32 × 256 = small
matmul over (BH, K) = ~64 × 16 = 1024 rows. Way below cuBLAS efficiency
threshold. Triton fuses cleanly.

**Why content_mlp / q_proj / k_proj NOT inside the kernel:**
- content_mlp: 4 layers × (D_steer × D_hid + D_hid × D_s) — too deep,
  needs RMSNorm + GELU + residual between layers. Cleaner as PyTorch.
- q_proj: 2-layer Sequential(Linear, GELU, Linear) — 2 cuBLAS calls.
  Could fuse but the GELU+Linear pattern is well-handled by inductor codegen.
- k_proj: similar. Already cached at block level (`_k_all_cache`).

### Kernel #3: `walker_ema_fwd`

Simplest of the three. Could just be PyTorch `α·w + (1-α)·m`. Triton version:

```python
@triton.jit
def walker_ema_fwd(
    walker_state_ptr,     # [B*H, D_s] bf16
    m_out_ptr,            # [B*H, D_s] bf16
    alpha_ptr,            # [H] fp32 (per-walker EMA rate, σ(walker_state_alpha))
    walker_state_new_ptr, # [B*H, D_s] bf16
    H, D_s,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    d_block = tl.program_id(1)
    h = bh % H
    a = tl.load(alpha_ptr + h)
    offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D_s
    w_old = tl.load(walker_state_ptr + bh * D_s + offs, mask=mask).to(tl.float32)
    m = tl.load(m_out_ptr + bh * D_s + offs, mask=mask).to(tl.float32)
    w_new = a * w_old + (1.0 - a) * m
    tl.store(walker_state_new_ptr + bh * D_s + offs, w_new.to(tl.bfloat16), mask=mask)
```

Tiny but eliminates one launch + saves one HBM round-trip. Optional —
inductor might already fuse this.

### Kernel #4: `anchor_pick_fwd` (only at t=0 of each block)

Builds `q_in = h_input + prev_motor_proj(prev_motor)`, scores against
input-plane keys (cached), Gumbel top-1 STE → anchor_cols + STE weights.

Same pattern as routing kernel but:
- Inputs from `[B, D_s]` (h_input) + `[B, D_s]` (prev_motor)
- Score against `[N_per_plane, D_q_in]` (cached `_input_keys_cache`)
- Output: `anchor_cols [B, H]`, `anchor_ste_weights [B, H, N_per_plane]`,
  `anchor_soft_probs [B, H, N_per_plane]`

Fires only at t=0 of a block. Cleaner as separate kernel than branching
inside step_postlif.

---

## Backward kernel design

### Lesson: avoid Triton backward for matmul-shaped ops (a2f3f50)

We have several gradient paths to compute:
- `grad_q [B*H, H_score, D_q]` (back into q_proj)
- `grad_k_all [N, H_score, D_q]` (back into k_proj cache)
- `grad_e_bias_at_edges [B*H, K]` (back into E_bias_flat[edge_flat])
- `grad_s_new [B*N, D_s]` (back into LIF kernel input)
- `grad_walker_state_in [B*H, D_s]`
- `grad_m_out [B*H, D_s]` (back into content_mlp)
- `grad_col_id [N, D_id]` (back into col_id parameter)
- `grad_nbr_id_proj [D_id, D_s]` (back into nbr_id_to_s parameter)
- `grad_alpha [N]` (back into decay_proj param)
- `grad_walker_state_alpha [H]`

**Strategy:** Hand-write Triton kernels for backward of the three forward
kernels, but **keep all the matmul backwards in PyTorch**. The matmul backwards
are:
- `q.backward` — backprop through q_proj is PyTorch's `nn.Linear.backward`,
  which is two well-batched bmms. Stays in autograd's graph.
- `m_out.backward` — through content_mlp.backward, also pure PyTorch.

The Triton backward kernels compute:
- `step_postlif_bwd`: from `(grad_s_cur_new, grad_end_states, grad_soft_probs)`
  → `grad_s_new` (sparse), `grad_q`, `grad_k_all`, `grad_e_bias_at_edges`,
  `grad_col_id` (for nbr_id reads).
- `lif_deposit_bwd`: from `grad_s_new` (sparse) → `grad_msgs`, `grad_alpha`,
  `grad_s_in`. Same pattern as current `_sparse_lif_bwd_kernel`.
- `walker_ema_bwd`: from `grad_walker_state_new` → `grad_walker_state_in`,
  `grad_m_out`, `grad_walker_state_alpha`.

### Backward sizes — verify Triton wins

For the v9-backprop's pattern (gather + weighted-sum-along-K + tanh), Triton
backward worked because each program does O(K) work, no `tl.dot`. For us:

- `step_postlif_bwd`: per program does sum-over-K with K=16. No `tl.dot`
  unless we add nbr_id_to_s backward. The nbr_id_to_s backward has shape
  `(D_s × D_id)` outer-product per (bh, k) — small; per-program tile is
  D_s × D_id = 256 × 32 = 8K elements. Fine for register accumulation.
- `lif_deposit_bwd`: same as current kernel. Already proven.
- `walker_ema_bwd`: elementwise, trivial.

**Risk check:** if any of these end up needing `tl.dot([M_small, K, N])`,
profile against PyTorch's bmm equivalent BEFORE committing. The 8.7× lesson
applies to those.

### autograd.Function for the WHOLE step

```python
class WalkerStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *,
        # State inputs
        s_in, walker_pos, walker_state, prev_motor,
        # Parameters
        params: WalkerStepParams,   # bundled: q_proj/k_proj/content_mlp/...
        # Per-step inputs
        token_id, e_bias_flat,
        # Schedule + RNG
        tau, eps, rng_seed,
        # Pre-allocated output buffers
        out_buffers: StepOutBuffers,
        # Pre-allocated scratch (saved for backward)
        scratch: StepScratchBuffers,
        # Static config
        is_anchor: bool,
    ):
        # 1. PyTorch matmul: content_mlp(cat_pre)
        # 2. Triton: lif_deposit
        # 3. PyTorch matmul: q_proj(cat_post)
        # 4. Triton: step_postlif
        # 5. PyTorch matmul: motor cross-attn (small)
        # 6. Triton: walker_ema
        # ...

        # Save tensors needed for backward.
        ctx.save_for_backward(
            scratch.cat_pre, scratch.cat_post,
            scratch.s_cur_old, scratch.s_cur_new,
            scratch.m_out,
            scratch.soft_probs, scratch.ste_weights, scratch.explore_mask,
            scratch.s_old_lif, scratch.incoming_lif, scratch.alpha_at_dest,
            scratch.unique_dests_lif, scratch.sort_idx_lif, scratch.segment_offs,
            ...
        )

    @staticmethod
    def backward(ctx, grad_motor_state, grad_m_out_external):
        # 1. Triton: walker_ema_bwd
        # 2. PyTorch matmul: motor cross-attn backward (small)
        # 3. Triton: step_postlif_bwd
        # 4. PyTorch matmul: q_proj backward
        # 5. Triton: lif_deposit_bwd
        # 6. PyTorch matmul: content_mlp backward
        # ...
```

### autograd.Function for the BLOCK

```python
class WalkerBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        # State inputs (will be threaded through T_block step calls)
        s_in, walker_pos, walker_state, prev_motor,
        # Bundled params
        params: WalkerStepParams,
        # Per-block inputs
        tokens_block,           # [B, T_block]
        e_bias_flat,
        # Schedule
        tau, eps,
        # Pre-allocated buffers
        out_buffers: BlockOutBuffers,           # motor_states_bt, co_visit_total, ...
        scratch: BlockScratchBuffers,           # T_block x StepScratchBuffers
        # Static config
        anchor_at_t0: bool,
    ):
        # Per-step loop, calling WalkerStepFunction... but we want to capture
        # this whole block as ONE autograd.Function so save-for-backward
        # is per-block (not per-step). Implementation details TBD: either
        # call WalkerStepFunction T times (autograd builds nested graph)
        # OR manually orchestrate forward + write all saves into the
        # block scratch buffers.
        #
        # Recommended: do the full unroll manually so backward is also
        # one Function. This is what enables clean cudagraph capture.
```

The block-level Function owns ALL the saved tensors as a single set. The
backward unrolls T_block step backwards in reverse order. cudagraph captures
this single Function cleanly because there are no nested autograd contexts
that fight buffer reuse.

---

## CUDA graph integration

Once we have `WalkerBlockFunction` with stable input/output buffers and no
hidden state mutations, we can capture it as a CUDA graph trivially:

```python
# In phase1_step:
e_bias = mem._active_e_bias()
torch.compiler.cudagraph_mark_step_begin()
out_buffers = mem.block_forward(  # this calls WalkerBlockFunction.apply
    state=mem.state_buffers,
    params=mem.step_params,
    tokens_block=tokens_block,
    e_bias=e_bias,
    tau=tau, eps=eps,
    anchor_at_t0=True,
)
loss = compute_loss(out_buffers.motor_states_bt, ...)
loss.backward()  # WalkerBlockFunction.backward — one cudagraph
mem.copy_state_back(out_buffers)  # writes into stable state buffers
mem.detach_state()  # cuts grad chain for next block
```

We can wrap the WHOLE iteration body (forward + loss + backward + state
copy + detach) in `torch.cuda.graph(g, ...)` and replay. No reduce-overhead
needed — we control everything.

**Why this works where reduce-overhead fails:**
- We allocate ALL output buffers ourselves (not in inductor's pool)
- Save-for-backward lives in scratch buffers we allocate (not the pool)
- Replays read/write the same addresses every time
- Backward graph captures cleanly because forward graph is one Function

---

## Anchor-window special case

`is_anchor` is a Python bool that's True for the first step of each
plasticity window (t=0 of each TBPTT block, given `tbptt_block ==
mod_period` invariant). It controls:

1. Whether to run `anchor_pick_fwd` (extra Triton kernel)
2. Whether walker_pos gets teleported to anchor cols
3. Whether the LIF deposit includes the anchor injection

**Implementation:** dynamo specializes on the bool (already proven in
existing code). At capture time, we generate two compiled variants (or
just one for t=0 case and reuse for t≥1). For Triton, both paths use
identical kernels — just different inputs (anchor cols vs walker_pos for
the LIF deposit's `dests`, anchor injection added to msgs vs not).

The cleaner refactor: ALWAYS run the anchor pick, but on t≥1 zero-out the
anchor STE weights. This unifies the path at the cost of running the
anchor matmul on every step (BS·H·D_q_in × N_per_plane = small, fine).
Trade-off TBD; both work.

---

## Plasticity / neuromod (cold path — leave in PyTorch)

These run at most once per `mod_period` tokens, and the work is shape-
varying (touched_ids has dynamic length). Triton is the wrong tool.

Confirmed correct in current code:
- `_begin_plastic_window` (graph_walker.py:660) — neuromod fire from
  previous snapshot
- `_active_e_bias` (graph_walker.py:705) — adds detached base + grad
  delta
- `_snapshot_touched_columns` (graph_walker.py:723)
- `_plasticity_step` (graph_walker.py:1303) — Hebbian + neuromod commit
- `MultiHorizonReadout.cross_entropy_factorized` (readout.py:133)
- `accumulate_block_ce` (graph_walker.py:1199) — vectorize this Python
  for-loop later if it shows up in profile (it shouldn't — runs T_block
  iterations once per block, on detached fp32)

Things to keep an eye on:
- The `enumerate_touched_edges` and `build_adjacency_bias` helpers in
  neuromod.py use `torch.isin` and `torch.searchsorted` which have
  dynamic-shape outputs. Fine because they run outside any cudagraph.

---

## Migration plan — incremental phases

Phase 0: scaffolding. ~1 day.
- Create `src/graph_walker/triton/` package (rename existing
  `triton_sparse_update.py` into it as `lif.py`)
- Add `step_params.py` — bundles all parameters as a dataclass for
  passing to the autograd.Function

Phase 1: rewrite `lif_deposit_fwd/bwd` with pre-alloc buffers + no
torch.unique. ~2 days.
- Replace `SparseLIFUpdate.apply` with new `LIFDepositFunction.apply` that
  takes pre-allocated output buffer + saved-for-backward buffers
- Validation: existing 14 tests in
  `tests/test_sparse_lif_puretorch_parity.py` and
  `tests/test_triton_sparse_update.py` all pass with same atol/rtol
- Bench: same speed or better at production scale

Phase 2: write `step_postlif_fwd` + backward. ~3 days.
- Forward kernel: gather + scoring + Gumbel + STE + endpoint readout fused
- Backward kernel: analytical, compute grads of each output w.r.t. each input
- Tests: new `tests/test_triton_step_postlif.py` with parity vs current
  PyTorch path; gradcheck on small B/H/K
- Bench: per-step speedup vs current pure-torch

Phase 3: write `anchor_pick_fwd` + backward. ~1 day.
- Cousin of `step_postlif`. Mostly the same kernel pattern.
- Tests: parity vs current PyTorch anchor branch
- Integration: dispatch `is_anchor=True` path through new kernel

Phase 4: assemble `WalkerStepFunction` (autograd.Function for one step). ~2 days.
- Orchestrates: PyTorch matmul (content_mlp.fwd) → lif_deposit_fwd →
  PyTorch matmul (q_proj.fwd) → step_postlif_fwd → PyTorch matmul
  (motor cross-attn) → walker_ema_fwd
- Backward orchestrates the reverse with PyTorch matmul backward + Triton
  backward kernels
- Tests: integration test running 5 step calls, comparing to current
  `_step_core_pure` output (atol=5e-3 in bf16)
- Optional: add walker_ema_fwd Triton kernel

Phase 5: assemble `WalkerBlockFunction`. ~2 days.
- T_block sequential calls with pre-allocated scratch
- Save all intermediates for backward
- Tests: parity with current `block_forward` output
- Bench: should give same 3.5× as current default+fg=T (we haven't won
  speed yet, just changed the abstraction)

Phase 6: cudagraph capture wrapper. ~2 days.
- Pre-allocate state buffers in `__init__` (not just begin_segment)
- Wrap one-iteration-body in `torch.cuda.CUDAGraph`
- Replay per training step
- Bench: target 6-8× over eager
- Tests: full 98-test suite passes; new gradient-equality test confirms
  outputs match non-cudagraph path

Phase 7: cleanup. ~1 day.
- Delete old paths once new path is canonical:
  - `triton_sparse_update.py:SparseLIFUpdate` (replaced by lif.py)
  - `_step_core_pure` (replaced by WalkerStepFunction)
  - `compile_step` / `_compiled_step` machinery (no longer needed)
  - `block_forward` Python wrapper (replaced by WalkerBlockFunction)
- Update `docs/graph_walker.md` to reflect new architecture
- Run all 98 tests + benchmark final number

Total: ~14 days (1.5-2 weeks). Track via TaskList #260+ when implementing.

---

## Testing strategy

### Per-kernel parity tests

For each Triton kernel, write `tests/test_triton_<kernel>.py` with:
- Random small inputs
- Random realistic-scale inputs (B=4, H=4, N=256, D_s=128 — matches bench)
- Production-scale inputs (B=16, H=4, N=1024, D_s=512 — ensures no overflow)
- Edge cases: all walkers at same column, lone walker, K=2 (smallest valid)

Each test asserts:
- Forward output matches PyTorch reference (atol=1e-2, rtol=1e-2 in bf16;
  atol=1e-5, rtol=1e-5 in fp32)
- Backward gradients match (atol=1e-2 to 5e-2 for grad_alpha which uses
  atomic adds — same tolerance as current `test_sparse_lif_puretorch_parity`)

### Block-level parity test

`tests/test_walker_block_function.py`:
- Build tiny LM, run 1 segment via current `block_forward` and via new
  `WalkerBlockFunction.apply`. Compare:
  - All output tensors match
  - All param gradients match after backward
  - State buffers match after writeback

### Integration / regression tests

All 98 existing tests must pass without modification. The migration is a
behind-the-scenes change to the `block_forward` implementation. Public API
(memory.step, memory.step_core, phase1_step) unchanged.

### Benchmark

`scripts/bench_triton_walker.py`:
- Eager (forward+backward, no compile)
- Current ship: whole-block compile (default mode)
- New: WalkerBlockFunction without cudagraph
- New: WalkerBlockFunction with cudagraph capture

Expected progression: 1× / 3.5× / 3.5× / 6-8×.

---

## Risks & open questions

### Risks

1. **Backward kernel is slower than expected (the a2f3f50 problem).**
   Mitigation: profile each backward kernel against PyTorch reference
   BEFORE relying on it. If a kernel loses, fall back to PyTorch for that
   piece. The whole block can still be one autograd.Function with mixed
   Triton+PyTorch backward.

2. **Numerical drift accumulating across 64-step block.** Forward error
   compounds; with bf16 internals at 64 sequential ops, cumulative error
   could exceed atol=1e-2.
   Mitigation: do the LIF blend's `tanh` in fp32 (already does in current
   kernel). For walker_ema and EMA-style ops, accumulate in fp32 too.
   Add a "tight tolerance" parity test after every 16 steps to catch drift.

3. **`tl.rand` reproducibility under cudagraph.** Captured graphs may
   freeze the random sequence — every replay produces the same Gumbel
   samples within a step. We need different randomness across steps within
   a block.
   Mitigation: pass `rng_seed_offset = step_idx * B * H * K` as scalar arg.
   Each program computes its own offset from `rng_seed + offset + bh*K+k`.
   Static across replays for the same step but varies across steps.

4. **Scratch buffer memory blow-up.** `T_block × U_max × D_s` for the LIF
   intermediates: at B=16, H=4, T=64, D_s=512 → 64 × 128 × 512 × 2 bytes
   = ~8 MB per block. Plus `T_block × B*H × D_steer` for cat_pre/post:
   64 × 64 × 1568 × 2 = ~12 MB. Total ~20-30 MB scratch. Acceptable.

5. **CUDA graph capture interacts badly with optimizer.step().** If we
   capture the whole training iteration including opt.step(), the optimizer
   state must also have stable storage. Adam keeps mom1/mom2 buffers per
   param — these are nn.Parameter so they have stable addresses. But fused
   AdamW kernel might do funny things. Mitigation: capture only the
   forward+backward, keep opt.step() outside the graph.

6. **Production-scale verification.** All the specific numbers in this
   plan assume B=4, H=4 (bench scale). At production B=16, H=4, the
   per-program work changes. Some kernels might shift from launch-bound
   to compute-bound. Profile at production scale BEFORE committing to
   final design.

### Open questions

- **Should `nbr_id_to_s` matmul (D_id × D_s) live in step_postlif kernel
  or as a separate PyTorch matmul?** Tiny matmul, register-fits at our
  sizes. Probably in-kernel. Verify by benching both ways.

- **Should anchor pick always run (with masked output on t≥1) or only at
  t=0?** Trade-off: simpler graph vs ~5% extra compute per step.

- **Optional Target C (motor cross-attn fusion)** — can stay in PyTorch
  unless profile shows it as a hot spot. End_states is small ([B, H, D_s]
  with H=4); cross-attn is 4 small matmuls. Probably fine in PyTorch.

- **Should we make the anchor STE weights a SEPARATE save-for-backward
  buffer or pack into the same buffer as routing STE?** Decide during
  Phase 3 implementation.

---

## File-level changes summary

### New files

```
src/graph_walker/triton/
    __init__.py
    lif.py                    # lif_deposit_fwd/bwd (rewrite of triton_sparse_update.py)
    step_postlif.py           # step_postlif_fwd/bwd (gather + score + STE + endpoint)
    anchor_pick.py            # anchor_pick_fwd/bwd
    walker_ema.py             # walker_ema_fwd/bwd (optional)
    walker_step.py            # WalkerStepFunction (autograd.Function for one step)
    walker_block.py           # WalkerBlockFunction (autograd.Function for T_block steps)
    step_params.py            # WalkerStepParams dataclass
    buffers.py                # StateBuffers, ScratchBuffers, OutputBuffers dataclasses
```

### Modified files

```
src/graph_walker/graph_walker.py:
    - Add allocate_persistent_buffers() method (called by __init__)
    - Replace block_forward() Python loop with WalkerBlockFunction.apply()
    - Keep step_core, step as backward-compat (each calls block_forward
      with T_block=1)
    - Remove _step_core_pure (or keep as gold reference for tests)
    - Remove _compiled_step / _compiled_block (cudagraph wrapper handles this)

src/graph_walker/train_phase1.py:
    - Replace torch.compile(block_forward) with cudagraph capture + replay
    - State writeback via copy_ from BlockOutputBuffers (already done)

src/graph_walker/triton_sparse_update.py:
    - Delete after Phase 1 lands lif.py and tests pass (kept until then
      for fallback safety)
```

### Tests to add

```
tests/test_triton_lif.py                  # parity with sparse_lif_update_puretorch
tests/test_triton_step_postlif.py         # parity with current routing+endpoint code
tests/test_triton_anchor_pick.py          # parity with current anchor branch
tests/test_walker_step_function.py        # autograd.Function-level parity
tests/test_walker_block_function.py       # block-level parity
tests/test_cudagraph_capture.py           # capture + replay + gradient equality
```

### Tests to keep (validate migration is behavior-preserving)

```
tests/test_graph_walker.py                # 16 tests
tests/test_neuromod.py                    # 8 tests
tests/test_gradient_flow.py               # 5 tests
tests/test_smoke.py                       # 5 tests
tests/test_guards.py                      # 5 tests
tests/test_pretrained_smoke.py            # 26 tests
... (all 98 tests must pass after Phase 7)
```

---

## Why this plan is bounded (vs spiraling)

**What this plan does NOT do:**
- Rewrite content_mlp / q_proj / k_proj as Triton matmuls (would lose to
  cuBLAS — see a2f3f50 lesson)
- Rewrite the multi-horizon CE readout (cold path, already efficient)
- Rewrite plasticity / neuromod (cold path, dynamic shapes don't suit
  Triton)
- Change the model design (write-first-then-route, anchor-once-per-window,
  STE bridge — all preserved)

**What this plan DOES:**
- Replace the dispatch-bound chain of small ops in each step with 3
  Triton kernels
- Wrap one full step as one autograd.Function for clean cudagraph capture
- Wrap T_block steps as one block-level autograd.Function
- Pre-allocate ALL buffers up front (kills cudagraph-pool conflicts)
- Keep matmuls in PyTorch / cuBLAS where they belong

This is bounded because: each Triton kernel has a clear ABI (input/output
tensors with static shapes), each phase has parity tests against the
existing PyTorch path, and we can fall back at any phase if a kernel
turns out slower than expected.

---

## Where to pick up next session

If returning here cold:

1. Read `docs/graph_walker.md` to refresh the model design.
2. Read this plan's "Lessons from prior Triton work" section (don't skip).
3. Start with Phase 0 (scaffolding). Use existing `triton_sparse_update.py`
   as the structural template for new Triton modules.
4. Each phase has a parity test target. Don't move to the next phase until
   the current phase's tests pass. The migration is a refactor, not a
   feature — outputs MUST match the current PyTorch path within tolerance.

Branch state at planning time: `graph-walker @ 00d4084`. 98 tests passing.
Current bench: 3.49× over eager (whole-block compile, default mode).

---

*Plan author note: I considered going straight to "one giant Triton kernel
for the entire block" and rejected it. The 8.7× backward regression in
a2f3f50 came from putting too many small matmuls in Triton. The right move
is "fuse the small ops chain in Triton, leave matmuls in cuBLAS, wrap as
one autograd.Function." That's what this plan implements.*
