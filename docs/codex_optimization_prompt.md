# Codex / GPT-5 optimization briefing — graph_walker × Llama-3.2-1B

## TL;DR for the model

We have a recurrent memory module ("graph_walker") grafted onto a frozen
Llama-3.2-1B at one mid-stack decoder layer. Training throughput is 13.5×
slower than vanilla Llama. We want to get it to ≤5× slowdown without
fundamentally changing the thesis (online recurrent memory at every
token). Identify the highest-impact algorithmic and engineering levers,
ranked by effort vs payoff, and call out anything we may have overlooked.

## Architecture in one paragraph

A graph of N=1024 columns (16×16 plane × L=4 planes) with K=16 out-edges
per column. H=4 persistent "walker heads" traverse this graph: each token,
every walker writes a content message to its current column (sparse LIF
update), re-reads its column post-write, then routes to one of the K
neighbors via Gumbel-soft top-1 with a straight-through estimator. The
selected neighbor's embedding feeds an endpoint readout; cross-attention
over the H walker endpoints produces a single motor_state per token.
Plasticity (Hebbian on co-visit counts + a graph-transformer "neuromod"
predicting per-edge bias deltas) fires every mod_period=64 tokens.
Surprise EMA is folded from the walker's own multi-horizon CE.

Pretrained integration: at one Llama decoder layer L (mid-stack), a
`MemInjectLayer` projects `h ∈ [BS, T, d_lm=2048]` down to
`h_mem ∈ [BS, T, d_mem=512]` (W_in, fp32 trainable), feeds h_mem to the
walker which runs T=256 sequential per-token steps and produces readouts
back in d_mem space, then projects back up via W_out (fp32 trainable) and
adds `scale * W_out(readout)` to the residual before delegating to the
original frozen Llama layer body. All 16 Llama decoder layers (except the
trainable W_in/W_out/scale at L) are frozen bf16. The walker has 92.9M
trainable parameters (fp32 for Adam stability); the inject layer adds
2.1M.

## Where the time goes (RTX 4090, BS=4, T=256, d_mem=512, compile_on_train=True, autocast bf16)

Measured under torch.profiler (5 iters per config):

| Configuration | fwd ms/iter | fwd VRAM | step ms/iter | step VRAM | step tok/s |
|---|---|---|---|---|---|
| A. Vanilla Llama-3.2-1B | 19.3 | 2.78 GB | 55.6 | 5.84 GB | 18.4k |
| B. Walker standalone (same config) | 284.7 | 1.06 GB | **1059.4** | 5.17 GB | 1.0k |
| C. Llama + walker | 369.5 | 5.46 GB | **1090.7** | 19.69 GB | 0.9k |

**Key insight from profile (B vs A, both training step):**

- A. Vanilla Llama: 55.6 ms/iter, **1135 `aten::mm` calls @ ~150 µs each** —
  large matmuls, GPU compute-bound, happy.
- B. Walker only: 1059.4 ms/iter, **51,645 `aten::mm` calls @ ~6 µs each** —
  50× more matmul calls but each is 25× smaller. **Launch-bound, GPU starving.**
- `cuLaunchKernel`: 212,275 calls in walker-only, 461 ms total — ~2 µs per
  launch. Just kernel-launch overhead is ~9% of walker step time.
- `vectorized_elementwise_kernel`: 1.374 s CUDA across 57K calls in walker
  — many small elementwise ops (tanh, sigmoid, add, mul) for LIF, walker
  state EMA, normalizations.
- `triton_poi_fused_embedding_dense_backward`: 315 ms CUDA across 1260
  calls — backward through `tied_token_emb` per readout call; 11% of
  walker CUDA time.
- C - (A + B) = -24.4 ms — **integration overhead is essentially zero**.
  The 13.5× slowdown is entirely the walker; Llama's frozen layers add
  almost nothing.

**Conclusion**: the walker is bandwidth/launch-bound, not compute-bound.
50K small matmul calls per step is the disease. The cure is "fewer, larger
kernels": CUDA graphs, custom Triton fused steps, or batching across more
dimensions (BS, H, T).

So:
- Per training step: vanilla 55 ms, walker-only 1059 ms, combined 1091 ms.
- Llama backbone is fully parallel via attention; walker is fully sequential
  with thousands of small kernels per step.
- We cannot increase BS beyond 4 on a 4090 (24 GB) at d_mem=512 because
  walker activations across 256 steps + Llama backward dominate VRAM.

## What we've already tried

- ✓ **bf16 autocast**: walker state (`s`, `walker_state`) is bf16. All matmuls run
  in bf16 under autocast. fp32 retained only for trainable Linear weights and
  Adam state (stability — bf16 Adam loses small grad updates).
- ✓ **torch.compile (compile_step)** on the per-token graph core. ~1.5×
  speedup. Uses `fullgraph=False` because the routing exploration
  (`gumbel_top1_softmax` with Gumbel noise + ε-exploration) and the
  `sparse_lif_update` Triton kernel have data-dependent control flow.
- ✓ **Triton fused sparse LIF update** (custom kernel) with fused forward +
  backward — eliminated the dense [B, N, D_s] scatter that the PyTorch
  reference would do.
- ✓ **Factorized per-horizon CE** (avoids the [B, T, K_h, V] dense logit
  broadcast).
- ✓ **TBPTT detach** every mod_period=64 tokens (limits backward graph depth
  per block, but not VRAM during forward).
- ✓ **Surprise fold + plasticity firing** wired to streamed block CE (no
  redundant full-vocab readout pass per fire).

## Constraints (do NOT change)

1. **Walker is recurrent and online**: each token's state depends on previous
   tokens' state through column updates and walker_state EMA. We cannot
   rewrite to linear-recurrent (Mamba-style parallel scan) — the recurrence
   is nonlinear (LIF + tanh + content_mlp + Gumbel sampling).
2. **Llama is frozen**: only the inject layer (W_in/W_out/scale) and walker
   parameters train. We're not modifying Llama's body.
3. **Per-token routing decisions are discrete**: Gumbel-STE in phase 1, hard
   Categorical in phase 2 (REINFORCE / GRPO). Routing must produce real
   discrete picks, not soft distributions.

## Constraints (CAN flex)

- Walker's clock: stride=2 or stride=4 (fire walker every K tokens) is on
  the table; thesis still "online recurrent" just at slower clock.
- Walker dimensions: d_mem, content_mlp_depth, n_heads (H walkers),
  n_score_heads — all tunable.
- Plasticity cadence (mod_period): tunable.
- The inject_layer position L within Llama: tunable (currently L=8 of 16).
- gradient_checkpointing: currently OFF, available in PyTorch.

## Goal

Reduce training-step slowdown from **13.5× → ≤5×** vs vanilla Llama-1B at
the same resource budget (single 4090). Equivalently: get from 1.4k tok/s
training to ≥3.7k tok/s.

## Files to look at

- `src/graph_walker/graph_walker.py` (1700+ lines) — main module.
  - `forward_segment` (line ~1010) — top-level pretrained-LM entry point.
  - `step_core_from_h` (~957) — single-step pretrained walker call.
  - `_step_core_pure` (~1175) — pure-functional one-token compute.
  - `_apply_step_state` (~870) — applies pure-step output to self.
  - `class ColumnCompute` (~84) — content_mlp.
  - `_run_block_pure` is NOT yet implemented (would be the natural seam
    for block-level gradient checkpointing).
- `src/graph_walker/triton_sparse_update.py` — the fused LIF kernel.
- `src/graph_walker/routing.py` — Gumbel-STE and phase-2 hard Categorical.
- `src/graph_walker/pretrained/llm_wrapper.py` — `GraphWalkerPretrainedLM`.
- `src/graph_walker/pretrained/train_phase1.py` — training step.
- `src/pretrained/mem_inject_layer.py` — the W_in/W_out/scale wrapper.
- `scripts/profile_pretrained_gw.py` — the profiler that produced the
  numbers above. Per-op tables in /tmp/gw_profile/*.txt.
- `scripts/bench_pretrained_gw.py` — the benchmark.

## Specific questions for the model

1. **Per-token Python loop overhead**: T=256 sequential Python iterations,
   each invoking a torch.compile-fused step. Is there a way to run the
   loop ITSELF inside the compiled region (e.g., a `torch._higher_order_ops.scan`-style
   primitive that compiles a loop body)? What's the realistic speedup from
   pulling the loop inside `torch.compile(fullgraph=True)` vs the current
   `fullgraph=False` per-step compile?

2. **Custom Triton kernel for the full walker step**: the per-step compute
   does ~5 small matmuls (content_mlp blocks, q_proj, k_lookup, cross-attn,
   readout). Each is bandwidth-bound at our shapes. Would it be worth
   fusing them into a single Triton kernel? Estimated effort vs payoff?
   The shapes are:
   - content_mlp: `[B*H, 4*D_s + D_id] → 4*D_s → D_s` × content_mlp_depth=4 blocks
   - q_proj: `[B*H, 4*D_s + D_id] → n_score_heads * D_q_per_head` (D_q_per_head=64)
   - k_proj on neighbors: shapes `[B*H, K, n_score_heads, D_q]`
   - sparse_lif_update: already Triton-fused, ~30% of step time
   - cross-attn over H endpoints: `[B, H, D_s] → [B, D_s]`

3. **Gradient checkpointing**: block-level checkpointing across each
   tbptt_block=64 of walker steps. Trades 1.5× compute for ~halved
   activation memory. This would let us move from BS=4 to BS=8 or BS=12,
   amortizing per-step cost. Worth doing? Recommended granularity (per-step
   vs per-block vs per-segment)?

4. **Reduce walker state dim d_mem**: 512 → 256 is a quadratic VRAM win on
   walker activations (~4× headroom for BS). Likely allows BS=12-16. The
   walker's expressiveness drops but content_mlp_depth could compensate.
   What's the rule of thumb here?

5. **Stride the walker** (fire every K=2 or K=4 tokens, not every token):
   K-fold speedup on the walker's contribution. Llama still gets per-token
   logits. The "thesis cost" is losing per-token routing resolution. We're
   willing to take stride=2; stride=4 is a stretch. Any cleaner way to
   preserve per-token decisions while parallelizing the column updates?

6. **CUDA graphs**: would CUDA-graph capture of the per-step compute help?
   With dynamic ops (multinomial sampling, unique() in sparse_lif), maybe
   not — but is there a hybrid approach (capture the deterministic prefix
   of the step, run dynamic ops eager)?

7. **Anything we've missed**: are there standard tricks for recurrent-
   memory transformers (NTM, DNC, Memorizing Transformers, LongMem) that
   we could borrow for throughput?

## Output format we want

A ranked list of recommendations:
1. <Lever name> — <expected slowdown reduction> — <effort>
   <One paragraph: what to do, why it works, what it costs.>
   <Concrete first-step pseudocode if applicable.>

Up to 8 recommendations. Be specific about expected slowdown improvements
(don't say "significant" — say "1.5×" or "~30%"). Distinguish between
"easy win" (1 day) and "research project" (weeks).

## Numbers we'd accept

| Slowdown ratio | Verdict |
|---|---|
| 1–2× | "Comparable training cost" — would shock us if achievable. |
| 2–5× | **Target zone.** Defensible for any paper, deployable at scale. |
| 5–10× | Acceptable for research, footnoted in paper. |
| 10–20× | **Where we are now.** OK for small benchmark runs. |
| >20× | NTM-class — research only, no scale. |

We want to land in the 2–5× zone if possible, otherwise 5–10×.
