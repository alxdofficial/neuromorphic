# biomem chunkwise-parallel write — design plan (for review)

Goal: replace the **sequential per-token Python sweep** in `streaming_write` with a **chunk-parallel
scan** (DeltaNet-style), without losing the cross-token dependency or the nonlinear depth. This is a
*design* doc — annotate it.

## DECISION (locked, user)

**Variant A — compositional, single LLaMA tap.** Keep using ONLY the final LLaMA hidden as input (no
per-LLaMA-layer tap), but stack **multiple neural-memory column-layers**, with the nonlinearity *between*
them. We want the **compounding**: layer *l*'s input is layer *l−1*'s memory readout, so deeper layers
reason over what the shallower memory holds. Per column-layer we still **parallelize the scan over all
tokens**; depth (`n_pairs`) stays sequential (2 short steps).

Why this is also correct-by-construction for chunkwise: layer *l*'s key/value derive from `u^l_t =
hardtanh(r^{l-1}_t − θ)`, i.e. from the **previous** layer's memory `W_{l-1}` (fully computed before
layer *l* runs) — NOT from its own `W_l` (the state being scanned). So each layer's scan stays linear in
`W_l` → chunkable, while the address is still *memory-aware* across depth. Only layer 0's key is purely
input-derived (from the final LLaMA hidden). Rejected: per-LLaMA-tap (variant B) — it would make the
column-layers independent and outsource the nonlinearity to LLaMA, losing the compounding we want.

---

## 0. Why it's sequential today

`streaming_write` is a Python `for t in range(T)` loop. Each token does one `_sweep` through the
`n_pairs` depth layers, and each `_sweep` reads+writes the **current** `W`:

```
for t in 1..T:                         # SEQUENTIAL (T≈1024 kernel launches)
  s^0 = hardtanh(in_proj(h_t))
  for l in 0..n_pairs-1:               # depth (only 2 steps)
    inp   = W_l(t-1) · s^l / √K        # read current memory
    s^{l+1} = hardtanh(inp - θ_l)      # NONLINEAR, inside the recurrence
    dW    = outer(s^l, s^{l+1} - inp)  # delta
    W_l(t) = clamp(α_t·W_l(t-1) + (g+η)·dW)
```

Two couplings block parallelization:
- **(A) cross-token**: `W_l(t)` depends on `W_l(t-1)`. This is a *scan* — and a scan is exactly what
  DeltaNet parallelizes. **Not the real blocker.**
- **(B) in-recurrence nonlinearity**: the key/value `s^l, s^{l+1}` depend on `W(t-1)` through
  `hardtanh`. Because the scan's per-step operator is a *nonlinear* function of the running state,
  the products can't be folded into the WY/matmul form. **This is the blocker.**

Your insight is exactly right: keep (A), kill (B) by moving the nonlinearity **between layers** instead
of **inside the token recurrence**.

---

## 1. The redesign — a stack of *linear* delta layers with nonlinearity between them

This is how real DeltaNet/GLA LMs are built: each layer is a linear recurrence (chunk-parallel over
tokens), and pointwise nonlinearities sit *between* layers (parallel over tokens). Depth stays
sequential, but depth is tiny (`n_pairs = 2`); the expensive axis (T≈1024) becomes parallel.

Per layer `l` (run for `l = 0 … n_pairs-1`, sequential in depth):

```
# u^l : [B, T, C, K]  — this layer's per-token input (u^0 = in_proj(h).view(C,K); precomputed for ALL t)
k_t = normalize(K_proj_l(u^l_t))        # KEY  — INPUT-derived (function of u^l_t, NOT of W_l)
v_t =            V_proj_l(u^l_t)         # VALUE — INPUT-derived
β_t = gate_l(u^l_t, surprise_t)          # write rate ∈[0,1] (or [0,2]) — INPUT-derived
α_t = sigmoid(decay_l(h_t, surprise_t))  # decay  — INPUT-derived  (already have this, #2)

# ── CHUNK-PARALLEL SCAN over t (per column c = a K×K head), DeltaNet/Gated-DeltaNet WY form ──
W_l(t) = α_t · W_l(t-1)·(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ      # LINEAR in W_l(t-1) → chunkable
r_t    = W_l(t) · q_t                                          # readout (q_t = k_t or a query proj)

# ── nonlinearity BETWEEN layers (pointwise per-token → parallel) ──
u^{l+1}_t = hardtanh(r_t − θ_l)          # the "subtract θ + saturate", applied ONCE per layer
```

The **memory** is the set of final states `{W_l(T)}` — unchanged in meaning. The **read** (M seeds →
propagate through `{W_l(T)}` → prepend, + per-layer refresh) is untouched, except its per-layer query
should use the same `K_proj_l` for retrieval consistency.

**Wall-clock**: today `O(T)` sequential launches; new = `n_pairs` (=2) sequential depth steps, each a
chunk-parallel scan over T (matmul/tensor-core bound). DeltaNet reports 4–10× even before counting the
Python-loop overhead we'd delete.

---

## 2. What changes mathematically

| | today | chunkwise |
|---|---|---|
| **key/value** | `s^l` = the *propagated state* (depends on `W_{l-1}`, nonlinear) | `k,v` = **input-derived** projections of the layer input `u^l` |
| **per-step operator** | nonlinear (`hardtanh` inside the read→write) | **linear** `α(I − βkkᵀ)` (chunkable WY) |
| **nonlinearity** | inside the token recurrence | **between layers**, on the readout `r_t` |
| **write rate β / gate** | `g = regulator(dW, s, …)` — depends on the in-loop `dW` (∴ on `W`) | `β = gate(u^l, surprise)` — **input-derived** (loses dW-conditioning) |
| **state bound** | `clamp(W, −1, 1)` each step | clamp doesn't fit the WY form → rely on DeltaNet stability (L2 key + the `(I−βkkᵀ)` contraction + decay α) |

The **essential recurrence is preserved**: `W(t)` still depends on `W(t−1)` (the scan), still
error-corrects (`v − Wk`), still decays per-token (`α_t`). What's *given up* is using the
**memory-derived state as the address**: today the token is filtered through current `W` *before* it
addresses the write; in the linear form the address is a fixed projection of the input. (DeltaNet still
*reads* memory via `r_t = W q_t` — we lose only the memory-dependent *key*, not memory access itself.)

---

## 3. What stays mostly the same

- **Memory = per-example fast weights `W_l`**, gated **delta** rule, reset per example. Unchanged.
- **#2 input-dependent decay `α_t`** — already input-derived → folds straight into the Gated-DeltaNet scan. **Free.**
- **Surprise signal** — input-derived → feeds `β_t`/`α_t` exactly as now. **Free.**
- **C columns** → C parallel heads (each a K×K state). **n_pairs layers** → the depth stack. Same grid shape.
- **Prepend read + per-layer refresh** — separate from the write; unchanged (reads the final `W_l(T)`).
- **Frozen LM, capacity match, per-example reset, episodic semantics** — all unchanged.

## 4. How the nonlinearity comes back

It moves from the **token axis** to the **depth axis**. Within a layer the token scan is purely linear
(so it's chunkable); *between* layers we apply `hardtanh(r_t − θ_l)` pointwise — which is parallel over
all tokens. Stacking `n_pairs` such layers gives genuine nonlinear depth (a deep network in `l`), while
every token scan stays linear. "Do the right mechanism at every layer, subtract once at the end" =
linear delta scan per layer; threshold/saturate once at each layer boundary.

---

## 5. Forks to decide (where you can improve it)

1. **β / gate**: making it input-derived loses the dW (local-surprise) conditioning we have now. Options:
   (a) accept input-derived `β = gate(u^l, surprise)` (chunkable, simplest); (b) a hybrid where a coarse
   chunk-level `dW` proxy re-enters the gate. Lean (a).
2. **Clamp removal**: dropping `clamp(W,−1,1)` for DeltaNet-style stability (L2 key + contraction + decay).
   **Possible silver lining**: the L2-key-norm that *collapsed* the clamped grid (signal fell below θ)
   may be *fine* here — with no clamp, `W` (hence the readout `r_t = Wq`) isn't pinned to ±1, so `r_t`
   can stay `O(1)` above θ. The chunkwise rewrite might be what *unlocks* key-norm (#1), not just speed.
3. **q vs k**: separate query projection for the readout, or reuse `k`? Separate `q` ≈ decoupled read/write
   keys (a capacity win from the research) — cheap to add here.
4. **Chunk size & kernel**: use the FLA `chunk` kernel (head_dim=K=16, heads=C=36) vs a hand-rolled WY
   scan. K=16 is small per head but there are many heads — check tensor-core utilization.
5. **Read consistency**: should the seed read also use `K_proj_l` (address the final `W` the same way the
   write did)? Probably yes.

---

## 6. Code touch-list (when we build it)

- `encoder.py`: replace `_sweep` + the `for t` loop in `streaming_write` with a per-layer chunk scan;
  add `K_proj_l / V_proj_l / q` (or reuse); move `hardtanh(·−θ)` to the layer boundary; drop the clamp.
- New dependency: FLA chunk kernel (or a small WY scan impl).
- `model.py`: unchanged (surprise/decay already input-derived and passed in).
- Read path: optionally route seeds through `K_proj_l`.
- Smoke: assert chunk-write == sequential-write within tolerance on a short sequence (correctness gate).
