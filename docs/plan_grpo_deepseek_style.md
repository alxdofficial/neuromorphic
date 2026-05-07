# Plan: DeepSeek-style GRPO for graph-walker

**Status:** Phase A complete (2026-05-05). Phases B + C (PPO clipping, KL penalty, entropy bonus) deferred — this doc kept for B/C reference.

## Why

Current `grpo_step` only credits routing decisions made **during the prefix pass** — gen-time routing decisions get no gradient signal. This is coarse credit assignment; the walker's gen-time read decisions matter for reward but aren't optimized.

DeepSeek's GRPO does per-token credit: each generated token's log-π contributes a gradient term, all weighted by the trajectory's advantage. We want the analogous per-action thing for walker routing — every routing decision, prefix and gen, should have a log-π × advantage term.

## How (DeepSeek-style replay)

1. **Sample phase** (no grad): K rollouts of (prefix + AR gen). Capture per-step routing choices into a trace; capture sampled tokens.
2. **Score phase**: reward = BERT-cosine(generated, reference) → group-relative advantage.
3. **Replay phase** (with grad): one parallel forward [K, T_pre + L_gen] through Llama teacher-forced (sampled tokens fed in directly, no AR loop), walker uses captured routing trace as `replay_choices` per step → per-action log-π accumulates with grad.
4. **Loss**: `L = -(log_pi_sum_over_actions * advantage.detach()).mean()`.

This matches DeepSeek's structure: sample → score → teacher-forced re-forward with grad → per-action gradient.

## What's done — foundation (Phase A.1) ✓

- **`src/graph_walker/routing.py`**:
  - `routing_log_pi_for_action(scores, selected_idx) → log_pi [B]` — re-evaluates a saved action's log-prob under current scores (with grad).
  - `route_or_replay(scores, *, saved_idx=None, ...)` — single dispatch point: sample if `saved_idx=None`, otherwise replay using `routing_log_pi_for_action`.
  - `StepRoutingChoices` dataclass — captures `(anchor_idx, edge_idx)` per walker step.
- **`src/graph_walker/graph_walker.py`**:
  - `WalkerCorePureOutput.routing_choices: StepRoutingChoices | None` — populated when sampling, None when replaying.
  - `_step_core_pure(..., replay_choices=None)` — uses `route_or_replay` at both routing call sites; captures choices when sampling.
  - `step_core_from_h(..., replay_choices=None)` — bypasses compiled-step path when replaying (compiled-step doesn't take the kwarg; phase-2 already disables compile so no perf loss).
  - `_apply_step_state` appends captured choices to `_captured_routes` buffer when armed.
  - `start_capturing_routes()` / `consume_routing_trace()` API — arm / drain the per-step capture buffer.
- **`tests/test_routing_replay.py`** (5 tests):
  - Math parity: replay log-π matches sample log-π exactly.
  - Replay log-π carries grad → backward fires through routing scores.
  - `saved_idx=None` falls back to sampling cleanly.
  - End-to-end walker capture+replay produces grad-carrying log-π.
  - Replay doesn't re-capture (no double-buffering).

All 127 existing tests pass — backward compat preserved.

## What's done — Phase A.2 (rollout + train refactor) ✓

- **`src/graph_walker/graph_walker.py`** — `walk_segment` consumes `_next_replay_trace` and threads per-step `replay_choices` to `step_core_from_h`. Block-compiled path is force-disabled when replaying. `is_new_window` is reconstructed from the saved trace's `anchor_idx` presence to keep the routing pattern identical between sample and replay.
- **`src/graph_walker/pretrained/rollout.py`** — two new functions:
  - `sample_grpo_rollout` — sample phase under `no_grad` with capture armed, drains the routing trace from prefix + AR gen.
  - `replay_grpo_rollout` — teacher-forced replay via `wrapper.memory.arm_replay_trace(...)` → one parallel forward through the wrapper with grad enabled → `consume_log_pi_mean()` returns grad-carrying per-rollout log-π.
  - New `GRPOSampledRollout` dataclass.
- **`src/graph_walker/pretrained/train_phase2.py`** — `grpo_step` rewritten:
  1. Sample under `no_grad` (capture trace).
  2. Score → group-relative advantage.
  3. Replay teacher-forced with grad → per-action log-π over both prefix and gen routing decisions.
  4. REINFORCE: `loss = -(log_pi_mean * advantages.detach()).mean()`.
- **`tests/test_routing_replay.py`** — added `test_grpo_deepseek_style_full_flow`: end-to-end sample → replay → REINFORCE backward, asserts gradient reaches `memory.neuromod.*` params.

All 130 tests pass (127 pre-existing + 3 new — DeepSeek end-to-end + autocast-recursion regression + multi-block parity).

### Post-implementation audit (2026-05-05)

Self-audit + Codex audit caught 6 bugs, all fixed:

1. **CRIT (self-found)** — `walk_segment` consumed `_next_replay_trace` BEFORE the autocast-recursion guard. On CUDA-without-external-autocast the outer call cleared the stash and the recursive inner call found None, silently falling through to no-replay. Fixed by reordering; pinned with regression test.
2. **HIGH (Codex)** — `_freeze_plasticity_ctx` mutated `cfg.mod_period = 10**9` to "freeze plasticity". After TBPTT detach inside replay (multi-block trajectories), `_begin_plastic_window` rebuilt `_active_neuromod_delta` reading the fake mod_period, so replay log-π after the first block were gradients for a different policy than the sampler used. Fixed by dropping `_freeze_plasticity_ctx` (vestigial under external-surprise design — plasticity only fires from external `update_plasticity`) and holding `wrapper.preserve_autograd_graph()` during replay so `detach_state` never fires.
3. **HIGH (Codex)** — Final gen-token routing was credited under REINFORCE but its readout would only affect logits at position L+1 (not in trajectory, not scored). Fixed by skipping the L-th gen-token forward through the walker; the L-th token still appears in `generated` for the reward function. Replay input is now `[K, T_pre + L_gen - 1]`. Trace length is now `T_pre + L_gen - 1`.
4. **MED (Codex)** — `gen_sample_routing=False` was silently overridden by `wrapper.train(True)` always making routing categorical. Removed the flag (vestigial).
5. **MED (Codex)** — `wrapper.train(True)` engaged Llama train mode → host dropout/randomness desynced sample-vs-replay hidden states. Switched to `wrapper.memory.train(True)`; host LM stays in caller-set mode.
6. **LOW (Codex)** — Existing tests didn't exercise multi-block replay, so the HIGH #2 bug was missed. Added `test_grpo_replay_spans_multiple_tbptt_blocks` (tbptt=4, T_pre=8, L_gen=4 → 11 steps × 3 blocks).

### Bench

Re-run `scripts/bench_grpo.py` after this lands. Replay does an extra parallel forward [K, T_pre + L_gen] through Llama with grad — compute increases vs the old "prefix-with-grad + gen-no-grad" path, but gradient signal per step is much richer (per-action vs only-prefix-actions). Expected steps/sec to drop ~20-40%; will measure post-merge and update `docs/bench_results.md`.

## What's NOT in Phase A — deferred

- **PPO clipping** (`min(ρA, clip(ρ, 1±ε)A)`): would need a `π_old` snapshot kept across one update step. Phase B.
- **KL penalty** against `π_ref`: need a reference policy snapshot (post-Wave-2 checkpoint). Phase C.
- **Entropy bonus**: `+ β · H(π_θ)` over the routing distribution. Phase C.

## Order of operations for completing Phase A.2

1. Add `replay_trace` kwarg to `walk_segment` (LLM wrapper) — small, mechanical
2. Implement `sample_rollout_for_grpo` + `replay_rollout_with_grad` in rollout.py — medium
3. Rewrite `grpo_step` to use both — small
4. Smoke test → verify grad reaches gen-time routing decisions
5. CPU smoke run on `tiny_test` config end-to-end
6. GPU bench re-run
7. Update `docs/bench_results.md` Phase-2 section + Notion training doc

Estimated: 1-2 working sessions.

## References

- DeepSeekMath GRPO formulation: `J = E[Σ_t (ρ_t · A_i - β·KL)]` — per-token policy ratio with shared per-trajectory advantage.
- Our adaptation: action space = walker routing decisions (anchor + edge per step), not LM tokens. LM stays frozen; walker is the policy.
- Foundation tests: `tests/test_routing_replay.py`.
