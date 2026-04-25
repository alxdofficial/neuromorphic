# Pretrained-LM + GraphWalker integration

**Status:** smoke suite (42 tests) passes on CPU. No GPU smoke wired yet.
**Branch:** `main` (graph-walker promoted on 2026-04-25).

This doc describes how the `GraphWalkerMemory` is grafted onto a frozen
HuggingFace causal LM (Llama / TinyLlama / SmolLM2) at one mid-stack
decoder layer, plus the two-phase training protocol (Gumbel-STE bootstrap
→ autoregressive GRPO).

The integration plumbing (`MemInjectLayer`, `HostAdapter`, `MemAdapter`)
is **shared with v2** in `src/pretrained/`. Only the memory module and a
small adapter on `GraphWalkerMemory` are new.

---

## 1. High-level picture

```
input_ids
   ▼
embed_tokens
   ▼
layers 0..L-1               (frozen)
   ▼
MemInjectLayer:             (W_in / W_out / scale trainable)
    h_mem = W_in(h)         [BS, T, d_mem]
    readout = memory_fn(h_mem, input_ids)
    h' = h + scale * W_out(readout)
    orig_layer(h')          (frozen Llama layer body)
   ▼
layers L+1..N-1             (frozen)
   ▼
norm → lm_head → logits
```

- `d_mem = D_s` — MemInjectLayer's memory-side dim equals the walker's
  internal column-state dim. `forward_segment` passes h_mem directly
  into the walker as `h_input`; no extra projection.
- The walker runs T sequential steps inside `forward_segment`, with TBPTT
  detach every `cfg.tbptt_block` tokens.
- Trainable surface: walker (~4-8M) + W_in/W_out (~12.6M on Llama-3.2-3B)
  + scale (~3K). Llama itself fully frozen.

---

## 2. File map

```
src/graph_walker/
├── graph_walker.py              # GraphWalkerMemory + step_core_from_h + forward_segment
├── routing.py                   # gumbel_top1_softmax (now phase-aware)
├── telemetry.py                 # StatsCollector + plot_dashboard
└── pretrained/
    ├── __init__.py              # public API
    ├── config.py                # PretrainedGWConfig + factories (llama_1b/3b/etc)
    ├── llm_wrapper.py           # GraphWalkerPretrainedLM
    ├── train_phase1.py          # phase1_pretrained_step (parallel teacher-forced)
    ├── train_phase1_ar.py       # phase1_ar_pretrained_step (AR unroll)
    ├── train_phase2.py          # grpo_step
    ├── rollout.py               # autoregressive_rollout primitive
    └── train_loop.py            # run_cycle_loop (bootstrap → cycle)

src/pretrained/                  # SHARED with v2 — re-used by graph_walker
├── mem_inject_layer.py          # MemInjectLayer
├── mem_adapter.py               # MemAdapter (provides mem_head_logits)
└── hosts/                       # LlamaHost + future families
    ├── base.py
    └── llama.py
```

---

## 3. The new GraphWalkerMemory adapter API

Two methods added to `GraphWalkerMemory`:

```python
def step_core_from_h(self, h_input):
    """One walker step driven by [B, D_s] vector instead of token_id.
    Used internally by forward_segment."""

def forward_segment(self, h_mem, input_ids, adapter, *,
                    compute_aux_loss=True, preserve_graph=False,
                    walker_aux_weight=1.0):
    """[BS, T, D_s] → [BS, T, D_s] readouts + aux CE.

    Aux CE combines:
    - Walker-side multi-horizon CE (gradient flows to state_to_model + walker
      hot path). Uses walker's own tied_token_emb readout.
    - Llama-side horizon-1 CE via adapter.mem_head_logits (gradient flows to
      W_out and walker through the inject path).

    Plasticity + surprise fold fire at cfg.mod_period cadence using the
    walker's own multi-horizon CE — independent of the Llama side.
    """
```

One new parameter on the walker: `mem_input_v_proj` (D_s → D_s) — a
parallel anchor v_inject projection used when h_input arrives in D_s
already (no D_model embedding to project from).

Two walker params are token-path-only and frozen by the wrapper at init:
`token_to_state` and `input_v_proj`.

---

## 4. Phase 1 — Gumbel-STE bootstrap (`phase1_pretrained_step`)

Per step:
1. `wrapper.reset_memory(bs)` — zero walker working state.
2. Llama forward with memory closure at layer L. Walker steps T tokens
   sequentially inside `forward_segment` with TBPTT-block detach.
3. Primary loss: Llama CE on next-token prediction.
4. Aux loss: walker-side multi-horizon CE + Llama-side horizon-1 CE
   (combined inside `forward_segment`, returned via
   `wrapper._last_mem_loss`).
5. `loss = ce_weight * CE + lm_aux_weight * aux`.
6. Backward, grad-clip, opt.step, `wrapper.detach_memory()`.

Routing during phase 1: Gumbel-soft + STE (existing `gumbel_top1_softmax`
with `phase="phase1"`). τ schedule comes from `GraphWalkerConfig`.

### Phase-1 AR unroll (`phase1_ar_pretrained_step`)

Same per-step structure but with `wrapper.preserve_memory_graph()` keeping
walker state graph-connected across the prefix pass + per-token
continuation forwards. Continuation predictions' gradient reaches the
prefix-pass writes through the walker's recurrent state. Trains the
walker to actually USE its prefix writes.

### Default knobs (`PretrainedGWConfig`)

| Knob | Default | Notes |
|---|---|---|
| `T` | 256 | segment length per forward |
| `bs` | 8 | phase-1 batch |
| `walker_aux_weight` | 1.0 | scale on walker's own multi-horizon CE |
| `lm_aux_weight` | 0.1 | scale on adapter-side horizon-1 CE |
| `ce_weight` | 1.0 | scale on primary Llama CE |
| `grad_clip` | 1.0 | |
| `inject_layer` | 8 (1B) / 14 (3B) | mid-stack, doc-recommended |
| `d_mem` | 512 | MemInjectLayer dim; equals walker's `D_s` |

---

## 5. Phase 2 — Autoregressive GRPO (`grpo_step`)

Why AR + GRPO instead of teacher-forced GRPO: v1 measured SNR=1e-4 on
naive teacher-forced CE rewards. K rollouts saw the same teacher tokens,
so reward hardly varied across rollouts even with different memory states.
AR unroll fixes this — K rollouts diverge in token space, so reward
genuinely varies.

Per step (`grpo_step`):
1. Replicate prefix K times → [K, T_pre].
2. `wrapper.current_phase = "phase2"`. Hard Categorical routing (no
   Gumbel noise, no τ); routes return `log_pi` per decision.
3. Prefix pass with `grad_during_prefix=True`. Walker accumulates
   `log_pi_sum [K]` over every routing decision (anchor + per-token).
   Plasticity fires inside the prefix per `mod_period`. KV cache captured.
4. Generation pass with `grad_during_gen=False`. Plasticity FROZEN
   (`_freeze_plasticity_ctx` sets `mod_period → ∞` for the duration).
5. Reward: `reward_fn(generated [K, T_pre+L], reference [L]) → rewards [K]`.
6. Advantage: `A = (r - r.mean()) / max(r.std(), adv_std_floor)`.
7. REINFORCE loss: `L = -(log_pi_sum * A.detach()).mean()`.
8. Backward, grad-clip, opt.step, detach.

### Phase-2 minimum policy surface

`wrapper.freeze_all_but_E_bias_and_neuromod()` pins everything except
`memory.neuromod.*`. Reasoning: REINFORCE has high variance; minimizing
the trainable surface stabilizes the gradient. The neuromod is the
explicit policy head — it produces target deltas on E_bias that shape
routing scores. E_bias itself is a buffer (not a parameter) and evolves
via the plasticity pathway in-place.

---

## 6. Cycle orchestrator (`run_cycle_loop`)

```python
cfg = CycleConfig(
    work_dir="outputs/run1",
    bootstrap_steps=5000, cycles=5,
    cycle_phase1_steps=1000, cycle_phase2_steps=2000,
    grpo_K=8, grpo_rollout_len=128,
)
run_cycle_loop(
    wrapper, bootstrap_iter, cycle_p1_iter, cycle_p2_iter,
    reward_fn=my_reward, cfg=cfg,
)
```

| Stage | Trainable | Loop |
|---|---|---|
| Bootstrap | full surface (walker + W_in/W_out/scale) | `phase1_pretrained_step` |
| Cycle phase-1 AR | full surface | `phase1_ar_pretrained_step` |
| Cycle phase-2 GRPO | only `memory.neuromod.*` | `grpo_step` |

A fresh optimizer is constructed per stage so Adam momentum from now-
frozen params doesn't leak across stage boundaries. Caller owns the
data iterators.

---

## 7. Telemetry

`src/graph_walker/telemetry.py` — `StatsCollector` writes one JSONL row
per training step. The cycle loop wires this automatically; for ad-hoc
scripts:

```python
with StatsCollector(work_dir="outputs/run1") as collector:
    for step in range(N):
        stats = phase1_pretrained_step(wrapper, opt, batch)
        collector.snapshot(wrapper, step=step, phase="phase1", stats=stats)
```

Captured per step (~40 metrics):

| Category | Metrics |
|---|---|
| Training | loss, ce_loss, aux_loss, grad_norm, tok_per_sec, inject_residual_norm |
| Column state | s_norm percentiles (P50/P90/P99), touched_frac, visit_entropy_norm |
| Walker heads | per-head walker_state norm, per-head α (sigmoid'd), co-location rate |
| Routing | τ, ε, plast_eta |
| Neuromod | delta_nm_norm, γ, E_bias_norm/max/active_frac |
| Surprise | per-horizon EMA, plast_eta |
| Gradient flow | per-component grad norms (token_emb, content_mlp, q_proj, k_all, nbr_id_to_s, mem_input_v_proj, state_to_model, walker_state_alpha, neuromod.*) |
| Llama | W_in/W_out/scale value+grad norms, inject_residual_norm |

Offline plot generation: `plot_dashboard("outputs/run1/stats.jsonl",
out_dir="outputs/run1/plots")` writes 8 PNGs (training, column state,
walker heads, routing, neuromod, surprise, gradient flow, Llama side).

---

## 8. What's built vs what's not

| Component | Status |
|---|---|
| `step_core_from_h`, `forward_segment` | ✓ smoke-covered |
| `GraphWalkerPretrainedLM` wrapper | ✓ smoke-covered (random Llama) |
| Phase-1 parallel `phase1_pretrained_step` | ✓ smoke-covered |
| Phase-1 AR `phase1_ar_pretrained_step` | ✓ smoke-covered |
| Phase-2 GRPO `grpo_step` | ✓ smoke-covered (random reward) |
| AR rollout primitive | ✓ smoke-covered (K-rollout divergence) |
| Cycle loop `run_cycle_loop` | ✓ smoke-covered (e2e) |
| Telemetry `StatsCollector` + plots | ✓ smoke-covered |
| Phase-2 freeze (E_bias + neuromod only) | ✓ available (helper on wrapper) |
| GPU smoke | ✗ not wired |
| Gradient checkpointing on AR unroll | ✗ T_cont bounded by VRAM |
| KL penalty / entropy bonus on phase-2 | ✗ future work |
| Real phase-1 data shards | ✗ caller-owned |
| Production phase-2 reward (PrefBERT, entity-F1) | ✗ caller-owned |
| Checkpoint save/load | ✗ not built |

---

## 9. Known invariants / footguns

- `tbptt_block == mod_period`. Enforced in `GraphWalkerConfig.__post_init__`.
- `T % mod_period == 0`. Enforced both in config and in `forward_segment`'s
  per-block CE math.
- `d_mem == memory.D_s`. Enforced in `PretrainedGWConfig.validate()`.
- `wrapper.reset_memory(bs)` MUST be called before every segment forward.
- Phase-2 generation pass automatically pins `mod_period = 10**9` so
  per-token gen doesn't trigger plasticity fires that would re-randomize
  the policy mid-rollout.
- `consume_log_pi_sum()` clears the buffer on read — call once per phase-2
  prefix pass, before the generation phase.
- `wrapper._last_mem_loss` is only populated when `compute_aux_loss=True`
  (i.e. training mode + no override). Read it immediately after the
  forward; don't carry it across calls.
- `MemInjectLayer.W_in/W_out/scale` stay fp32 for stable Adam updates;
  Llama backbone is bf16 in production. Cross-dtype handling is inside
  `MemInjectLayer.forward`.

---

## 10. Quick start

```python
import torch
from src.graph_walker.pretrained import (
    PretrainedGWConfig, GraphWalkerPretrainedLM,
    Phase1Batch, phase1_pretrained_step,
)

cfg = PretrainedGWConfig.llama_1b()
wrapper = GraphWalkerPretrainedLM(cfg).cuda()
opt = torch.optim.AdamW([p for _, p in wrapper.trainable_parameters()], lr=1e-4)

for step in range(N):
    input_ids = next_batch()              # [BS, T]
    target_ids = input_ids                # teacher forced
    stats = phase1_pretrained_step(
        wrapper, opt,
        Phase1Batch(input_ids=input_ids, target_ids=target_ids),
    )
    if step % 100 == 0:
        print(step, stats.loss, stats.ce_loss, stats.aux_loss)
```
