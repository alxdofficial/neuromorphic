# Pretrained-LM + GraphWalker integration

**Status:** 118-test suite passes on CPU. No GPU smoke wired yet.
**Branch:** `main` (graph-walker promoted on 2026-04-25; vocab-agnostic
walker landed 2026-04-30).

This doc describes how the `GraphWalkerMemory` is grafted onto a frozen
HuggingFace causal LM (Llama / TinyLlama / SmolLM2) at one mid-stack
decoder layer, plus the two-phase training protocol (Gumbel-STE bootstrap
→ autoregressive GRPO).

The walker is **vocab-agnostic** in the integration: it has no LM head
of its own, no aux loss, no walker-side multi-horizon CE. It learns end
to end via gradient flowing back from Llama's primary CE through `W_out`.
Plasticity is driven externally — the trainer computes Llama's per-token
CE post-backward and feeds it to `walker.update_plasticity()`.

The integration plumbing (`MemInjectLayer`, `HostAdapter`) is **shared
with v2** in `src/pretrained/`. The old `MemAdapter` (provided
`mem_head_logits` for an aux-CE side-head) is no longer wired into the
graph_walker integration but is still imported by the v2 attention-
neuromod path.

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
  internal column-state dim. `walk_segment` passes h_mem directly
  into the walker as `h_input`; no extra projection.
- The walker runs T sequential steps inside `walk_segment`, with TBPTT
  detach every `cfg.tbptt_block` tokens.
- Trainable surface (production walker, post-scale-up, Llama-3.2-1B):
  walker ~24.2M + W_in/W_out/scale ~1.1M ≈ **25.3M total**. (Earlier
  versions of this doc claimed "~4-8M" — that was the pre-scale-up
  walker config. Verified 2026-05-04 from
  `wrapper.trainable_parameters()`.) Llama backbone fully frozen.

---

## 2. File map

```
src/graph_walker/
├── graph_walker.py              # GraphWalkerMemory + step_core_from_h + walk_segment
├── routing.py                   # gumbel_top1_softmax (phase-aware) +
│                                # route_or_replay / routing_log_pi_for_action
│                                # for DeepSeek-style replay (Phase A)
├── telemetry.py                 # StatsCollector + plot_dashboard
└── pretrained/
    ├── __init__.py              # public API
    ├── config.py                # PretrainedGWConfig + factories (llama_1b/3b/etc)
    ├── llm_wrapper.py           # IntegratedLM
    ├── train_phase1.py          # phase1_pretrained_step (parallel teacher-forced)
    ├── train_phase1_ar.py       # phase1_ar_pretrained_step (AR unroll)
    ├── train_phase2.py          # grpo_step
    ├── rollout.py               # sample_grpo_rollout + replay_grpo_rollout
    │                            # (DeepSeek-style two-phase) +
    │                            # autoregressive_rollout (inference only)
    └── train_loop.py            # run_cycle_loop (bootstrap → cycle)

src/pretrained/                  # SHARED with v2 — re-used by graph_walker
├── mem_inject_layer.py          # MemInjectLayer
├── mem_adapter.py               # MemAdapter (v2-only now; not used by graph_walker)
└── hosts/                       # LlamaHost + future families
    ├── base.py
    └── llama.py
```

---

## 3. The GraphWalkerMemory integration API

```python
def step_core_from_h(self, h_input):
    """One walker step driven by [B, D_s] vector instead of token_id."""

def walk_segment(self, h_mem, *, preserve_graph=False):
    """[BS, T, D_s] → [BS, T, D_s] readouts. Pure forward.

    The walker is vocab-agnostic: no LM head, no aux loss, no surprise
    is computed inside the segment. TBPTT detach happens every
    `cfg.tbptt_block` tokens unless `preserve_graph=True`.
    """

def update_plasticity(self, per_token_surprise: Tensor | None):
    """Externally-driven plasticity. Call AFTER `loss.backward()`.

    - per_token_surprise=None: AR free-gen / inference. No plastic update.
    - per_token_surprise: [B, T_supervised]: training. Fold per-token CE
      into surprise_ema, fire one plasticity step (commit the current
      `_active_neuromod_delta` into `E_bias_flat`, snapshot, build the next
      segment's neuromod delta).
    """
```

One new parameter on the walker: `mem_input_v_proj` (D_s → D_s) — a
parallel anchor v_inject projection used when h_input arrives in D_s
already (no D_model embedding to project from).

Walker params frozen by the wrapper at init (dead in the integration's
vocab-agnostic forward path):
- `token_to_state`, `input_v_proj`, `tied_token_emb` — token-id-driven
  hot path only used by the standalone walker.
- `state_to_model` and the entire `readout` submodule (multi-horizon
  walker LM head) — exercised only by the dropped aux-CE path.

---

## 4. Phase 1 — Gumbel-STE bootstrap (`phase1_pretrained_step`)

Per step:
1. `wrapper.begin_segment(bs)` — zero walker working state. Preserves
   `_neuromod_input_*` so the next segment's `_active_neuromod_delta` is
   non-None and routing carries gradient to neuromod.
2. Llama forward with memory closure at layer L. Walker steps T tokens
   sequentially inside `walk_segment` with TBPTT-block detach.
   `walk_segment` is now a pure forward — no aux loss, no plasticity.
3. Loss: Llama CE on next-token prediction (`ce_weight * CE`). The walker
   has no aux loss of its own; its parameters learn purely via gradient
   flowing back through `W_out` from Llama's CE.
4. `loss.backward()`, grad-clip, `opt.step()`.
5. **Surprise / plasticity**: with `reduction='none'`, recompute Llama's
   per-token CE on the shifted logits and feed `[BS, T-1]` into
   `wrapper.memory.update_plasticity(per_token_ce)`. This folds
   surprise into `surprise_ema`, fires one plasticity step, and builds
   the next segment's neuromod delta.
6. `wrapper.detach_memory()`.

Routing during phase 1: Gumbel-soft + STE (existing `gumbel_top1_softmax`
with `phase="phase1"`). τ schedule comes from `GraphWalkerConfig`.

### Phase-1 AR unroll (`phase1_ar_pretrained_step`)

Same per-step structure but with `wrapper.preserve_autograd_graph()` keeping
walker state graph-connected across the prefix pass + per-token
continuation forwards. Continuation predictions' gradient reaches the
prefix-pass writes through the walker's recurrent state. Trains the
walker to actually USE its prefix writes. Surprise for plasticity is
the per-token CE over the continuation tokens, computed post-backward.

### Default knobs (`PretrainedGWConfig`)

| Knob | Default | Notes |
|---|---|---|
| `T` | 256 | segment length per forward |
| `bs` | 8 | phase-1 batch |
| `ce_weight` | 1.0 | scale on primary Llama CE |
| `grad_clip` | 1.0 | |
| `inject_layer` | 8 (1B) / 14 (3B) | mid-stack, doc-recommended |
| `d_mem` | 512 | MemInjectLayer dim; equals walker's `D_s` |

The old `walker_aux_weight` and `lm_aux_weight` knobs were removed when
the walker became vocab-agnostic — there is no aux loss to weight.

---

## 5. Phase 2 — DeepSeek-style GRPO (`grpo_step`)

Why AR + GRPO instead of teacher-forced GRPO: v1 measured SNR=1e-4 on
naive teacher-forced CE rewards. K rollouts saw the same teacher tokens,
so reward hardly varied across rollouts even with different memory states.
AR unroll fixes this — K rollouts diverge in token space, so reward
genuinely varies.

Phase 2 follows DeepSeek's GRPO structure: **sample under no_grad,
score, then teacher-forced replay with grad** for the policy gradient.
This gives per-routing-decision credit assignment (analogous to
DeepSeek's per-generated-token credit, but for the walker's routing
action space).

Per step (`grpo_step`):

1. **Sample phase** (`sample_grpo_rollout`, no_grad):
   - Replicate prefix K times → [K, T_pre].
   - `wrapper.memory.train(True)` (NOT `wrapper.train(True)` — host LM
     stays in caller-set mode so dropout doesn't desync sample-vs-replay).
   - Arm `wrapper.memory.start_capturing_routes()`.
   - Run prefix forward + (gen_length - 1) AR-gen forwards. The L-th
     sampled token is added to `generated` for the reward function but
     never forwarded through the walker — its routing decision could
     only affect logits at position L+1, which the trajectory does not
     include and `reward_fn` does not see, so crediting it would be
     non-causal noise.
   - Drain captured trace → `routing_trace`, length T_pre + L_gen - 1.
   - `_freeze_plasticity_ctx` is intentionally NOT used: under the
     external-surprise design, plasticity only fires via
     `update_plasticity` (called by the trainer post-backward), so
     forward-only paths can't fire plasticity regardless. The old
     freeze-by-mutating-mod_period trick silently corrupted the
     `_active_neuromod_delta` rebuild that runs at TBPTT boundaries.

2. **Score phase**:
   - `rewards = reward_fn(generated [K, T_pre+L], reference [L])`
   - `advantages = (rewards - rewards.mean()) / max(rewards.std(), adv_std_floor)`

3. **Replay phase** (`replay_grpo_rollout`, with grad):
   - `wrapper.begin_segment(K)` + `wrapper.memory.arm_replay_trace(...)`.
   - `wrapper.memory.train(True)` (walker only).
   - One parallel forward through `wrapper(replay_seq)` where
     `replay_seq = generated[:, :-1]` (drop the L-th token; matches
     trace length T_pre + L_gen - 1).
   - `wrapper.preserve_autograd_graph()` is held — `walk_segment`
     would otherwise call `detach_state()` at every `tbptt_block`-token
     boundary inside the replay, and that detach also rebuilds
     `_active_neuromod_delta` via `_begin_plastic_window`. With preserve_graph,
     the autograd graph stays alive end-to-end and the active delta
     stays consistent across blocks.
   - The walker uses the saved trace as `replay_choices` per step;
     `is_new_window` is reconstructed from the saved `anchor_idx`
     presence so sample-vs-replay routing patterns match exactly.
   - Drain `consume_log_pi_mean()` → `log_pi [K]` with grad attached.
   - The replay forward also produces logits at every position; per-
     token CE against `generated[:, 1:]` is computed under no_grad
     during the same scope (logits are freed immediately after the
     chunked CE compute, so peak memory matches the original "discard
     logits" code path). Returned as `ReplayResult.per_token_ce` for
     the unified plasticity update in step 5.

4. **REINFORCE backward**:
   - `loss = -(log_pi * advantages.detach()).mean()`
   - `loss.backward()`, grad-clip, `opt.step()`.

5. **Unified plasticity** (mirrors Phase 1):
   - `wrapper.memory.update_plasticity(per_token_ce)` — folds the
     replay's per-token CE into `surprise_ema`, fires
     `_plasticity_step` (commits the active neuromod delta into
     `E_bias_flat`), and rebuilds the next window's `_active_neuromod_delta`
     with fresh grad_fn for the next training step.
   - Walker behavior is phase-agnostic by design — Phase 2 calls this
     with the same shape contract as Phase 1, just with the surprise
     target sourced from the replay sequence (prefix tokens =
     ground-truth cumulative prior; gen tokens = the model's own
     samples). Without this call, `E_bias_flat` would freeze during
     all of Phase 2, silently disabling the long-term plastic
     encoding pathway. See `docs/training_strategy.md` § "Unified
     plasticity" for the full design rationale.
   - `wrapper.detach_memory()`.

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
| Training | loss, ce_loss, grad_norm, tok_per_sec, inject_residual_norm |
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
| `step_core_from_h`, `walk_segment` | ✓ smoke-covered |
| `IntegratedLM` wrapper | ✓ smoke-covered (random Llama) |
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
- `T % mod_period == 0`. Enforced in config.
- `d_mem == memory.D_s`. Enforced in `PretrainedGWConfig.validate()`.
- `wrapper.begin_segment(bs)` MUST be called before every segment forward.
  Default `clear_neuromod_carryover=False` preserves `_neuromod_input_*`
  across calls — required so neuromod params receive gradient via
  `_active_neuromod_delta`. Setting it to `True` starves neuromod of gradient.
- `wrapper.memory.update_plasticity(per_token_ce)` MUST be called after
  `loss.backward()` (Phase 1) or `opt.step()` (Phase 2) for plastic
  state and neuromod params to advance. Walker behavior is phase-
  agnostic — `update_plasticity` fires in BOTH phases with a per-token
  CE tensor of shape `[B, T]`. The surprise SOURCE varies: Phase 1
  uses CE against ground-truth tokens; Phase 2 uses CE from the replay
  sequence (prefix = ground-truth cumulative prior; gen = the model's
  own samples). Pass `None` (or skip the call) only for AR free-
  generation / inference, where no next-token target exists.
- `consume_log_pi_sum()` clears the buffer on read — call once per phase-2
  prefix pass, before the generation phase.
- `MemInjectLayer.W_in/W_out/scale` stay fp32 for stable Adam updates;
  Llama backbone is bf16 in production. Cross-dtype handling is inside
  `MemInjectLayer.forward`.

---

## 10. Quick start

```python
import torch
from src.graph_walker.pretrained import (
    PretrainedGWConfig, IntegratedLM,
    Phase1Batch, phase1_pretrained_step,
)

cfg = PretrainedGWConfig.llama_1b()
wrapper = IntegratedLM(cfg).cuda()
opt = torch.optim.AdamW([p for _, p in wrapper.trainable_parameters()], lr=1e-4)

for step in range(N):
    input_ids = next_batch()              # [BS, T]
    target_ids = input_ids                # teacher forced
    stats = phase1_pretrained_step(
        wrapper, opt,
        Phase1Batch(input_ids=input_ids, target_ids=target_ids),
    )
    if step % 100 == 0:
        print(step, stats.loss, stats.ce_loss, stats.grad_norm)
```
