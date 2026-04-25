# Walker Speedup Plan — CUDA Graphs + Triton Fusion

**Status:** SUPERSEDED — see [`docs/triton_rewrite_plan.md`](triton_rewrite_plan.md)
**Original author:** session 2026-04-25 (early planning notes)
**Outcome:** the cudagraph-capture half of this plan landed (Phase 5
of the rewrite plan) and delivered ~13.5–14× over eager, exceeding the
10–15× target. The Triton-fusion half was deferred — under cudagraph
replay the launch-overhead reduction it would buy drops to noise.
This file is preserved for context; refer to `triton_rewrite_plan.md`
for the current design + outcome record.

---

## Thesis

The persistent-walker design is intentionally sequential across time (no
parallelisation across T). A profile of `_step_core_pure` (graph_walker.py:886)
showed:

- ~9% matmul (cuBLAS — already efficient)
- ~48% small elementwise / gather / scatter / softmax ops
- ~30+ kernel launches per token-step
- Walker is **launch-bound**, not compute-bound

So the optimisation frontier is *kernel launches per step*, not flops per step.
Two stacking levers attack this:

1. **CUDA graphs** — capture the per-step kernel sequence, replay with one
   dispatch (5–8× walker speedup, no architecture change)
2. **Triton fusion** — collapse small-op clusters (routing softmax/Gumbel,
   endpoint readout, motor cross-attn) into single kernels (further 2–4×)

Stacked target: **~10–15× walker speedup**, dropping the Llama+walker overhead
from ~13× to ~1.5×.

---

## Current state of the walker hot path

Main entry: `GraphWalkerMemory.step_core` (graph_walker.py:803) →
`_step_core_pure` (graph_walker.py:886).

Existing compile harness:
- `compile_step()` (graph_walker.py:529) calls
  `torch.compile(self._step_core_pure, mode="default", fullgraph=False)`
- `fullgraph=False` because the body has Python-side branches dynamo can't
  stay inside (`is_new_window`, lazy block-cache init, `is_training` in
  routing.py:81).
- `mode="default"` does not enable CUDA graphs.

Existing Triton work:
- `triton_sparse_update.py` covers only the LIF deposit (1 kernel out of ~30
  per step). Has custom forward + backward, opaque to dynamo.

Caller loop: `train_phase1.phase1_step` (train_phase1.py:159) — Python
`for t in range(T_seq)` invoking `step_core(tokens[:, t])`. Appends
`r.motor_state` to a growing Python list per step (line 167) — dynamo-hostile.

---

## Part A: CUDA graphs via `torch.compile(mode="reduce-overhead", fullgraph=True)`

Recommended path. `reduce-overhead` mode internally uses CUDA graphs on the
compiled region. To switch we have to make `_step_core_pure` `fullgraph=True`-
clean (no graph breaks).

### Refactor list

**A1. Eliminate `is_new_window` Python branch.**
- Split `_step_core_pure` into two static-shape variants:
  - `_step_anchor_pure` — does input-plane Gumbel pick + STE-gated injection
    + walker hop (graph_walker.py:945–980)
  - `_step_interior_pure` — walker hop only, no anchor work
- Compile each variant separately; `step_core` dispatches before the call
- Costs ~2× compile cache memory; avoids per-step branch divergence

**A2. Move block-cache init out of hot path.**
- Currently `step_core` calls `_ensure_block_caches` (graph_walker.py:810),
  which has `if x is None` branches that force `@torch._dynamo.disable`
  (graph_walker.py:553).
- Populate caches in `begin_segment` (line 578) and `detach_state` (line 745)
  unconditionally.
- Drop the `_ensure_block_caches` lazy helper. Caches are read-only inside
  step_core.

**A3. Persistent in-place buffers for scalars.**
- `_schedule_tensors` (line 785) builds new `torch.tensor(...)` each call —
  these become graph inputs that need stable addresses across replays.
- Allocate at `begin_segment`:
  - `self._tau_buf = torch.zeros(1, device=device, dtype=torch.float32)`
  - `self._eps_buf = torch.zeros(1, device=device, dtype=torch.float32)`
  - `self._tok_buf = torch.zeros(B, device=device, dtype=torch.long)`
- `step_core` updates them via `.copy_(...)` before invoking the compiled
  variant.

**A4. Remove `is_training` Python flag from `gumbel_top1_softmax`.**
- routing.py:81 branches on Python bool. Either compile two variants per anchor
  / interior (training × eval = 4 total) or always run the training-path
  with `epsilon=0` at eval.
- Lean toward the eval-via-zero-epsilon path: simpler, only 2 compiled
  variants total.

**A5. Backward-compat sanity checks.**
- `torch.bincount(edge_taken_flat, minlength=cfg.num_edges)` (line 1123) —
  output shape static via `minlength`; should compile.
- `torch.unique` + `searchsorted` inside `SparseLIFUpdate.forward` (lines
  236, 244) — opaque to dynamo (custom autograd.Function). No refactor.
- All `torch.where` / `F.softmax` / `F.gelu` / `F.one_hot` (constant
  num_classes) are fullgraph-friendly.

**A6. Pre-allocate `motor_state_buf` in train loop.**
- train_phase1.py:81 grows a Python list (`motor_state_block.append`).
- Replace with `torch.empty(B, tbptt, cfg.D_s, device=device, dtype=...)`
  pre-allocated at flush boundary. Loop body writes
  `motor_state_buf[:, i_in_block].copy_(r.motor_state)`.
- Flush slice: `motor_state_buf[:, :T_block]`.

**A7. Wire `compile_step` to use new mode.**
- Update `compile_step()` (line 529) to compile both anchor + interior
  variants with `mode="reduce-overhead", fullgraph=True`.
- Add `cfg.compile_mode: Literal["none", "default", "reduce-overhead"]` for
  test/dev (default `"default"`, train sets `"reduce-overhead"`).
- Pre-warm both variants on dummy inputs immediately after compile so
  the first real step doesn't pay the latency.

### Validation tests

- `tests/test_compile_anchor_interior.py`: anchor and interior pure variants
  produce numerically equal output to current `_step_core_pure` with the
  same `is_new_window` arg (atol=1e-4 fp32, 5e-3 bf16).
- `tests/test_cuda_graph_replay.py`: 5-step capture+replay matches eager.
- `scripts/bench_walker_only.py`: 512-token segment under
  {eager, mode=default, mode=reduce-overhead}. Confirm 5–8× speedup on the
  walker-only path.

### Risks

- **Backward through CUDA-graphed forward.** `reduce-overhead` mode handles
  autograd via `make_graphed_callables` semantics — needs warmup iterations
  (typically 3) before steady-state replay. Train loop must include warmup.
- **Recompile triggers.** Pin BS, dtype, contiguity at `begin_segment`. Each
  recompile is ~3 min cold cache.
- **bf16 autocast + reduce-overhead.** Known sharp edges in this combo on
  some PyTorch builds. Validate with a 10-line micro-test before doing the
  big refactor. Fallback: `fullgraph=True, mode="default"` first
  (no CUDA graphs but cleaner trace), then add reduce-overhead.
- **PRNG reproducibility under graph capture.** Captured kernels reuse the
  RNG state from capture time, so replays produce identical "random" Gumbel
  samples each step. Either accept (deterministic walks within segment) or
  thread an explicit per-step seed buffer through the captured graph. Need
  to test if the determinism degrades training.

### Compile time budget

- Cold cache, first run: ~3–5 min total (60–120s anchor variant + 60–90s
  interior + 30–60s backward graph).
- Warm cache (`TORCHINDUCTOR_CACHE_DIR` set): ~5–10s.
- Recompile triggers: BS change, dtype change, requires_grad flip, stride
  change. Lock all of these at segment boundary.

---

## Part B: Triton kernel fusion (after Part A lands)

Targets ranked by ROI:

### Target A: Routing kernel (highest priority)

Replaces ~7 small kernels with 1. Lives in graph_walker.py:1045–1062.

New file: `src/graph_walker/triton_routing.py`

```python
class FusedGumbelTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_all, nbrs_of_cur, e_bias_at_edges,
                tau, epsilon, training_mode, rng_seed):
        # q:                [BH, H_score, D_q]
        # k_all:            [N, H_score * D_q]   (cached)
        # nbrs_of_cur:      [BH, K] int64
        # e_bias_at_edges:  [BH, K] fp32
        # Outputs:
        #   selected_idx:   [BH] int64
        #   ste_weights:    [BH, K]
        #   soft_probs:     [BH, K]   (saved for backward + load balance)
```

One Triton program per BH:
- Load q[bh] into registers (H_score × D_q = 4 × 64 = 256 floats)
- For k=0..K: load nbrs_of_cur[bh,k], gather k_all[neighbor_id], compute
  K dot products (per-head sum), → score[k]
- Add e_bias_at_edges[bh, k]
- Generate K Gumbel samples via `tl.rand(seed, offset)`
- Add Gumbel, divide by tau, softmax → soft_probs
- Argmax → selected_idx
- Build hard one-hot, compute `ste = hard - soft.detach() + soft`
- ε-exploration: with prob ε replace with uniform random + detached STE
- Store outputs

Backward kernel: standard softmax + bilinear backward, STE bridge for
explore vs non-explore rows (zero grad for explore, soft_probs.grad for
non-explore).

Expected gain: ~2× on routing block (~20% of step) → ~0.5× overall.

### Target B: Endpoint readout fusion

Lines 1092–1100. New file: `src/graph_walker/triton_endpoint.py`.

One program per BH:
- Gather K col_id rows from nbrs_of_cur
- Apply `nbr_id_to_s.weight` (D_id × D_s, in registers)
- Multiply by ste_weights[k], sum over K
- Add s_cur_new

Expected gain: ~5% of step → ~1.5× on that block.

### Target C: Motor cross-attn fusion

Lines 1102–1110. New file: `src/graph_walker/triton_motor.py`.

One program per B (or per (B, D_s tile)):
- Norm end_states[b] (B × H × D_s where H=4, very small)
- Compute K[h] = norm @ out_k_proj.weight, V[h] = norm @ out_v_proj.weight
- Scores: sum(K[h] * motor_query) * scale → [H]
- Softmax over H
- motor_state = sum(attn[h] * V[h]) → [D_s]

Expected gain: ~5–10% of step.

### Target D: skip

LIF + walker EMA fusion — diminishing returns after CUDA graphs absorb the
launch overhead.

---

## Suggested execution order

| Phase | Work | Effort | Walker speedup | Cumulative ratio vs Llama-1B |
|---|---|---|---|---|
| Baseline | — | — | 1× | 13× slowdown |
| Phase A (CUDA graphs) | A1–A7 | 2–3 days | 5–8× | 2–3× slowdown |
| Phase B-A (routing) | Target A | 3–4 days | +2× | 1.5–2× slowdown |
| Phase B-B (end+motor) | Targets B, C | 2–3 days | +1.3× | 1.2–1.5× slowdown |

Net: ~10–15× cumulative on walker portion.

---

## Open questions to validate before bulk refactor

1. Does `torch.compile(mode="reduce-overhead", fullgraph=True)` accept the
   existing Triton `sparse_lif_update` autograd.Function? Quick check via
   a 10-line micro-test wrapping it in a fullgraph compiled function.
2. Does bf16 autocast play with reduce-overhead capture for our shapes?
3. Is per-step BS truly fixed within a TBPTT segment? (Confirmed via
   reading train_phase1.py — yes, set at `begin_segment`.)
4. PRNG reproducibility across replays — does training degrade if every
   token's Gumbel sample is identical within a segment?

Validate (1) and (2) FIRST via micro-tests before doing A1–A7 refactors.
Failures should drop us to fallback (`fullgraph=True, mode="default"`)
without CUDA graphs but still bigger fused regions.

---

## Where to pick up next session

Tasks live in TaskList (#243–#252). #243 (validation micro-test) is in
progress. Suggested first move: write
`scripts/validate_compile_modes.py` that:

1. Builds tiny `GraphWalkerMemory` (plane_rows=4, plane_cols=4, L=2,
   K=4, D_s=32, K_horizons=2, mod_period=4)
2. Compiles `_step_core_pure` with `mode="reduce-overhead", fullgraph=True`
3. Runs 5 forward steps under bf16 autocast
4. Runs backward on a synthetic loss
5. Reports any errors, suggested fixes

If that script runs end-to-end, Path 1 is viable and we can proceed with
A1–A7 refactors. If it errors, the error message tells us which constraint
to relax (drop `fullgraph=True`, drop `reduce-overhead`, or refactor
specific ops).

Branch: `graph-walker` @ `024a6cb`. Working tree should stay clean —
no commits planned until validation passes.

---

## Session results (2026-04-25)

### What worked

**Replaced SparseLIFUpdate's Triton+autograd.Function with pure-torch ops.**
The old custom autograd.Function did `s_flat.clone()` and `torch.unique(...)`
inside `forward`, allocating tensors via the standard allocator. Those tensors
escaped inductor's cudagraph pool and tripped `check_memory_pool` under
`mode="reduce-overhead"`. The new `sparse_lif_update_puretorch`
(triton_sparse_update.py) uses only standard PyTorch ops (`index_add`,
`tanh`, masked blend) — no custom Function, no `torch.unique`, all static
shapes. Inductor can fuse it.

Side effect: **`default+fg=T` jumped from 1.49× → 1.92× over eager** because
the LIF body is no longer opaque to inductor. Free win.

All 98 tests pass with the new path as default. The Triton path is still
available as `backend="triton"` in case we want to switch back at production
scale where memory matters (the pure-torch path's saved-for-backward
intermediates are O(BN·D_s) vs the Triton path's O(U·D_s); at B=16, N=1024,
D_s=512 that's ~4 GB of extra activation memory across a TBPTT block).

### What didn't work (yet)

**`mode="reduce-overhead"` compiles successfully now** — all four sub-configs
(fg×{T,F} × triton.cudagraph_*={on,off}) pass the validation script's
end-to-end forward+backward. **But running it inside a TBPTT loop fails**
with `RuntimeError: Error: accessing tensor output of CUDAGraphs that has
been overwritten`.

Root cause: TBPTT runs T forwards then T backwards (via `loss.backward()`).
Each compiled forward step saves tensors for its own backward. Under
cudagraphs, the saved-tensor pool is shared across replays — replay t+1
overwrites replay t's saves before backward can consume them.
Cloning `r.motor_state` doesn't help because autograd's internal saves
(intermediates inside `_step_core_pure`) aren't reachable from user code.

PyTorch's standard answer: `cudagraph_mark_step_begin()` between iterations.
That works for inference loops or single-step train (forward + backward
within one iteration). It does NOT work for our pattern (T forwards, then
backward across all T).

### Bench results (B=4, T=64, mod_period=64)

```
config                    tok/s      rel
------------------------------------------
eager                    1342.2    1.00x
default+fg=F             2507.5    1.87x
default+fg=T             2572.5    1.92x   <-- ship this as Phase A.1
reduce-overhead          FAILED (TBPTT/cudagraph buffer reuse)
ro+fg=T                  FAILED (same)
```

### Phase A.1 (ship now): default mode + pure-torch sparse_lif

The pure-torch sparse_lif refactor + flipping `compile_step()` to
`mode="default", fullgraph=True` gives **~1.92×** with zero risk. Already
ships because the sparse_lif backend changed to `puretorch` and inductor
auto-fuses the body.

Remaining one-line change: update `compile_step()` default to
`fullgraph=True`. The `is_new_window` Python branch turns out NOT to need
the A1 refactor — dynamo specializes on the bool automatically and produces
two compiled variants under the hood.

### Phase A.2 (next): unlock reduce-overhead via TBPTT-aware capture

Two paths to investigate:

1. **Per-step save extraction** — wrap the compiled step in a custom
   autograd.Function that copies saved-for-backward tensors out of the
   cudagraph pool into per-step storage. Heavy lifting; PyTorch internals.
2. **Selective checkpointing** — wrap the compiled step in
   `torch.utils.checkpoint` so backward recomputes forward (no saved tensors
   across steps). Each (forward, backward) pair fits within one
   `cudagraph_mark_step_begin()` cycle. Pays 2× compute on backward but might
   still net positive if the cudagraph win is large.
3. **Inference-only cudagraph** — for eval / GRPO rollout (no autograd),
   reduce-overhead works directly. Easy win for inference-heavy code paths.

Tasks #244–#250 (A1–A7 refactors) are mostly redundant now that fullgraph=T
works without them. Keep them only if Phase A.2 needs them. Task #252
tracks the Phase A.2 design.
