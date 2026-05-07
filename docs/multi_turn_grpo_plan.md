# Multi-turn GRPO protocol — Wave 4 design

**Status:** implementing 2026-05-06.

The single-turn `grpo_step` we built handles the simple "one prefix → one
generation → one reward" case. Wave 4 (WildChat overflow) needs a
fundamentally different protocol because real chat sessions are
multi-turn, and the walker's value-prop *is* maintaining state across
those turn boundaries.

## The aligned-trajectory protocol

For each WildChat session in the dataset:

```
[walker.begin_segment(bs=K)]                                    ← session start

walker observes user_1 (sampled routing, log_pi captured)
For each assistant turn t = 1 .. N:
    snapshot canonical walker state S_t                        ← K rollouts share this
    For k = 1 .. K:
        restore S_t
        sample assistant_t_k tokens until EOS or max_response_len
                                                               ← log_pi[t,k] accumulated
                                                               ← reward[t,k] = BERT-cosine
    advantage[t,:] = group_normalize(reward[t,:])
    loss_t = -mean_k(log_pi[t,k] · adv[t,k]) / max_response_len
    loss_t.backward()
    opt.step(); opt.zero_grad()                                ← per-turn update

    restore S_t
    walker.forward(assistant_t_ground_truth, no_grad=True)
                                                               ← canonical state advances
                                                               ← surprise signal → walker.update_plasticity
    if t < N:
        walker.forward(user_{t+1}, no_grad=False, sampled routing)
                                                               ← log_pi for turn t+1 starts here
    walker.detach_memory()
```

**The aligned-trajectory invariant.** At the start of every assistant
turn `t`, all K rollouts share an identical walker state — the canonical
state that's a function of *ground-truth* turns 1..t-1. The K rollouts
diverge only during turn t itself, then we re-converge by:
1. Discarding the K diverged states
2. Restoring the snapshot
3. Forwarding ground-truth assistant_t (everyone gets the same input → state stays identical)
4. Forwarding user_{t+1} (everyone gets the same input → state stays identical, with diverged routing samples)

Snapshot at turn t+1 is taken just before the next turn's K rollouts begin.

This is the standard "offline RL on conversations" pattern (TRL agent
training, swift `rlhf` mask, VAGEN, MTIR-SQL, VerlTool) — the K rollouts
each turn are independent samples from the same starting state, scored
independently against ground truth, with the dataset's user turns
verbatim.

## Why this is the right design

1. **Per-turn losses are independent.** Because we re-converge to ground
   truth between turns, there's no autograd graph crossing turn
   boundaries. Per-turn `.backward()` gives the same gradient as
   end-of-session backward — just with N× more SGD updates per session.
2. **Walker exercise scales with session length.** A 10-turn WildChat
   session gives 10 reward signals + 10 SGD steps + 10 turns of walker
   memory carry. Currently (single-turn) it gives 1 of each.
3. **Observation routing is credited correctly.** Walker routing during
   user-turn observation is sampled (matches deployment), log_pi is
   captured, and credited to the *next* assistant turn's reward. Within
   a turn, the loss is `log_pi[t] · advantage[t]` where log_pi[t]
   includes both user-obs routing and assistant-gen routing.
4. **Plasticity gets a clean surprise signal.** During the ground-truth
   replay step, the LM produces logits, we compute per-token CE against
   the ground-truth tokens, and feed that to `walker.update_plasticity()`
   — same mechanism as Phase-1 SFT, but interleaved.

## What gets sampled vs teacher-forced

| Phase | Sampled? | log_pi captured? | LM loss applied? | walker.update_plasticity? |
|---|---|---|---|---|
| User turn obs (cumulative prior re-forward) | walker routing: yes; tokens: N/A (input given) | yes (credited via REINFORCE on the next assistant turn's advantage) | no | yes (each Phase-2 step's replay drives one `update_plasticity` covering prefix + gen — see "Unified plasticity" below) |
| Assistant turn gen | walker routing AND tokens | yes (REINFORCE) | no (REINFORCE only — no LM CE loss) | yes (gen tokens contribute to the same per-step `update_plasticity` call) |

**Unified plasticity (this is the original design intent).** The walker
is phase-agnostic: ``update_plasticity`` fires once per training step in
BOTH Phase 1 (parallel teacher-forced SFT) and Phase 2 (GRPO sample/
replay). Walker behavior — anchor selection, surprise EMA decay,
plasticity window length — does NOT change between phases. What varies
is the *source* of the per-token surprise:

- **Phase 1**: CE against ground-truth tokens (teacher-forced).
- **Phase 2 prefix tokens**: CE against ground-truth cumulative prior
  (= same as Phase 1 — the prefix IS teacher-forced during replay).
- **Phase 2 gen tokens**: CE against the model's own sampled tokens
  (= self-entropy at sample time; high under uncertainty, low under
  confident generation).
- **Future agentic / tool-use scenarios**: CE against environment-
  emitted tokens (tool outputs, retrieved documents). Same mechanism.

The contract is: whatever next-token target the trainer can construct,
feed CE to ``update_plasticity``. The walker integrates surprise the
same way regardless of whether the target came from data, the model
itself, or a tool. Without this, ``E_bias_flat`` would freeze during
Phase 2, silently disabling the long-term plastic encoding pathway —
the very thing the memory graph exists to provide.

## EOS early-stop

Each assistant rollout ends as soon as the LM emits EOS, OR when the
rollout reaches `max_response_len`. Currently the sample loop runs the
full `gen_length=128` regardless — that needs fixing as part of this
change. EOS-stopped rollouts have shorter trajectories; the trace-length
assertion in `replay_grpo_rollout` needs to be per-rollout (not a single
trace_length) to handle this.

## Loss normalization

We use **Dr. GRPO** style: divide by `max_response_len` (constant).
Standard practice from
[Understanding R1-Zero-Like Training](https://huggingface.co/papers/2503.20783).
Removes length bias completely (longer rollouts don't dilute gradient
per token; shorter rollouts don't get inflated gradient).

This is a small change from the current `grpo_step`'s per-rollout-mean
normalization, which is fine when `gen_length` is fixed but breaks under
EOS early-stop (variable-length rollouts).

## Walker state snapshot/restore

New API on `IntegratedLM`:

```python
state = wrapper.snapshot_memory_state()    # dict of cloned tensors
wrapper.restore_memory_state(state)        # broadcasts/copies back
```

Snapshots ONLY the per-batch working state (s, walker_pos, walker_state,
surprise EMAs, log_pi accumulators, segment counters). Does NOT snapshot
the long-term plastic state (E_bias_flat, _neuromod_input_*) — those keep
evolving across the whole session via the canonical replay path.

Implementation: `state` is a dict of cloned tensors. `restore` copies
back into the walker's named attributes.

## K rollouts: shared state going in, divergent during turn

To run K rollouts sharing a snapshot:
1. `wrapper.begin_segment(bs=K)` once at session start
2. Walker state is shape `[K, ...]` — all K elements identical at turn
   boundaries (snapshot-restore copies the same tensor across all K)
3. During turn t: K rollouts diverge naturally (Categorical routing
   sampling varies per K element, LM token sampling varies per K element)
4. After turn t: restore drops the K=K divergent state back to the
   identical canonical state

## Single-turn `grpo_step` — internal helper only

The single-turn `grpo_step` stays in the codebase as an internal
implementation detail used by `_grpo_session_step_uniform_batched`'s
fast path (it batches B sessions × K rollouts in one B*K-rollout
GRPO call). It's no longer the public Wave 3 entry point — `train_grpo.py`
routes both Wave 3 and Wave 4 through `grpo_session_step`. See the
"Wave 4 turn-batching refinement" section below for how Wave 4 hits
the same fast path.

## Implementation stages

1. **Foundations**: EOS early-stop + walker snapshot/restore + Dr. GRPO loss
2. **Re-preprocess WildChat**: full conversation tokens + per-session turn-boundary index
3. **Multi-turn loader**: yields `MultiTurnSession(turns: list[(role, ids)])`
4. **`grpo_session_step`** in `train_phase2.py`
5. **Tests**: snapshot round-trip, EOS-stop trace consistency, end-to-end session step
6. **Wire into `train_grpo.py`** for `--data wildchat-grpo`

---

## Post-implementation refinement (2026-05-06, "ABC")

After the initial multi-turn protocol landed, three structural refinements
were added:

### A — `lm_context_window`: decouple walker context from LM context

The first multi-turn implementation had walker context = LM context (both
saw the full cumulative prior). For Wave 3 with T_pre=256, this meant
filler ended up at ~100-180 tokens with the fact directly attendable —
walker was barely tested. The fix:

- New parameter `lm_context_window` on `sample_grpo_rollout`,
  `replay_grpo_rollout`, `grpo_step`, `grpo_session_step`.
- When `lm_context_window < T_pre`: two-phase forward.
  - **Phase 1**: `wrapper(prefix_ids[:, :T_pre - lm_context_window], use_cache=False)`.
    Walker advances state through the early portion. LM forwards too
    (necessary because walker's `walk_segment` requires LM hidden
    states), but its KV cache is discarded.
  - **Phase 2**: `wrapper(prefix_ids[:, T_pre - lm_context_window:], use_cache=True)`.
    Walker continues from its already-advanced state (memory now reflects
    the FULL prefix). LM caches only these recent tokens.
  - **AR gen**: continues from phase-2's KV cache. From the LM's
    perspective, the prefix was just `lm_context_window` tokens long.
- **Replay**: same two-phase split. Trace is sliced into early + recent
  portions (re-armed per phase). Walker advances no_grad through phase 1
  (matching sample's no_grad full-rollout) and with_grad through phase 2.

This forces the walker to actually carry information that the LM can no
longer attend to directly. Wave 3 with `T_pre=2048` and
`lm_context_window=256`: walker absorbs the fact at position 100 + 1700
tokens of filler + question at position 1900, but the LM only sees the
last 256 tokens (= the question + a tiny tail of filler). The walker's
memory is the only path from fact to answer.

### B — Wave 3 + Wave 4 unified under `grpo_session_step`

Originally Wave 3 used single-turn `grpo_step` and Wave 4 used multi-turn
`grpo_session_step` — two separate code paths. But Wave 3 is structurally
just Wave 4 with N_assistant_turns=1, so the split was wasted complexity.

- New `passphrase_chat_grpo_session_iter` in `passphrase_chat_loader.py`
  yields `MultiTurnSession` with 2 turns (user prefix + assistant ref)
  instead of `ChatGRPOBatch`.
- `train_grpo.py` routes both `--data passphrase-chat-grpo` and
  `--data wildchat-grpo` through `grpo_session_step`.
- The single-turn `grpo_step` is now an internal implementation detail
  (called by the uniform-batched session path — see C).

### C — Multi-session batching in `grpo_session_step`

Previously `grpo_session_step` handled one session at a time. The
unification under multi-turn lost Wave 3's BS_outer ~5× speedup. Fix:
the function now accepts `sessions: list[MultiTurnSession]` and:

- **Uniform-shape fast path** (`_can_batch_as_single_turn`): if all
  sessions are 2-turn (user, assistant) with matching prefix lengths
  — the Wave 3 case — they're stacked to `[B, T_pre]` and processed
  via one internal `grpo_step` call with B*K parallel rollouts. Recovers
  Wave 3's ~5× session-throughput speedup under the unified API.
- **Sequential fallback**: variable-shape sessions (Wave 4 with
  different N_assistant_turns or different per-turn lengths) are
  processed one at a time, each with K-batched rollouts. Loses
  cross-session parallelism but keeps the unified API.
- True multi-session parallelism for Wave 4 (with padding/masking or
  N-bucketing) is **future work** — bench results will show whether it
  pays off enough to justify the implementation cost.

### Final API surface

```python
grpo_session_step(
    wrapper, opt,
    *,
    session=None,                    # OR
    sessions=None,                   # list[MultiTurnSession]; B sessions
    reward_fn=None,                  # BERT-cosine
    num_rollouts=8,                  # K
    max_response_len=128,            # AR gen length cap
    eos_id=None,                     # early-stop
    lm_context_window=None,          # walker/LM context decoupling
    max_prior_tokens=None,           # cumulative-prior cap (Wave 4 memory)
    ...                              # standard knobs (lr, grad_clip, etc.)
)
```

Wave 3 invocation: `sessions=[s1, ..., s_B]` where each `s_i` has 2 turns.
Wave 4 invocation: `sessions=[s1, ..., s_B]` where each `s_i` has N_i turns.
Both go through the same function; the function routes internally based on
session shape.

---

## Wave 4 turn-batching refinement (2026-05-06, post-ABC)

Initial Wave 4 routed through the sequential fallback path of
`grpo_session_step` — variable-shape sessions processed one at a time.
This works correctly but doesn't parallelize across sessions, so Wave 4
was effectively B=1.

**Insight that unlocks batching:** our current implementation already
doesn't carry walker memory state across turns within a session. Look
at `grpo_session_step`:

```python
for t, turn in enumerate(session.turns):
    if turn.role != "assistant":
        cumulative_prior.extend(turn.ids.cpu().tolist())
        continue
    # ... below, sample_grpo_rollout calls wrapper.begin_segment(bs=K)
    # which RESETS walker state. State is rebuilt from scratch by
    # forwarding cumulative_prior tokens.
```

What's carried across turns within a session is the **token history**
(`cumulative_prior`), not the walker memory state. Walker state at
turn t is a function of cumulative_prior_through_t-1, regardless of
whether we walked the session sequentially or pulled this `(prior, ref)`
pair from a flat pool.

**This means turn-pairs from any session at any round can be batched
together** — there's no implicit "carry state from session A's turn 3 to
session A's turn 4 within the same training step" we'd be breaking.

### The Verlog-style approach

Reformulate the dataset as a flat pool of `TurnPair(cumulative_prior,
response, session_idx, turn_idx)` units. Each WildChat session expands
into N_assistant_turns of these. Trainer sees a flat pool of independent
training units.

For B-way parallelism: pull B turn-pairs with similar prior lengths,
stack as `[B, T_pre]`, route through the existing
`_grpo_session_step_uniform_batched` fast path (same code as Wave 3).
True B*K parallel rollouts per outer step.

### Sort-and-sample (not fixed bucketing)

To keep prior lengths uniform within a batch:
1. Maintain a pool of M=2048 turn-pairs.
2. Sort by `prior_len` after each refill.
3. Per batch: pick a uniform-random window of B contiguous neighbors
   from the sorted pool.
4. Truncate all priors in the batch to the SHORTEST prior length within
   the batch (left-truncate, keep most-recent context). Stackable as
   `[B, T_pre]` with no padding/masking.

Benefits over fixed bucketing:
- Adapts to actual length distribution; no manual bucket tuning
- Same algorithmic cost
- Padding waste is minimal because contiguous-neighbors-in-sorted-array
  have near-identical lengths

### Implementation surface

- `TurnPair` dataclass + `session_to_turn_pairs(session)` flattener in
  `src/data/wildchat_loader.py`
- `wildchat_turn_pair_grpo_batch_iter(batch_size, pool_size)` yielding
  `list[MultiTurnSession]` of size B per call (each pair wrapped via
  `TurnPair.to_two_turn_session()` so the existing uniform-batched fast
  path consumes it transparently)
- `train_grpo.py`: dispatch logic adapted to handle both single-yielding
  iters (Wave 3) and list-yielding iters (Wave 4 turn-pair)
- New args: `--bs-outer` (B sessions/turn-pairs per outer step),
  `--turn-pair-pool-size` (M, default 2048)

No walker changes needed. ~200 LOC total.

### What it gives us

Wave 4 now routes through the same uniform-batched fast path Wave 3
uses, so:
- True B*K parallel rollouts per outer step (was B=1 effective)
- Expected ~5× session-throughput speedup at B=8 (matching Wave 3's
  BS_outer numbers)
- Per-prompt advantage normalization (each turn-pair's K rollouts
  compared against each other only — no cross-prompt mixing)
- Honest per-prompt reward stats in logging (the per-group stats fix
  that landed alongside this — `GRPOStats` now exposes
  `per_group_reward_mean: list[B]` and `per_group_reward_std: list[B]`
  populated from the actual per-row mean/std used in advantage
  normalization, not the global B*K mean replicated)

### What it doesn't give us

- "Walker carries state across turns within a session within a single
  training step" — we never had this; not a regression. (Class B state
  is per-turn-pair; Class A E_bias_flat is shared across all
  turn-pairs.)
- Cross-session walker state via the long-term plastic E_bias_flat
  buffer DOES persist (it's a registered buffer, mutated by
  `update_plasticity` calls). ``update_plasticity`` fires per Phase-2
  GRPO step (mirroring Phase 1), so the plastic encoding pathway is
  active throughout Wave 3 + Wave 4. See "Unified plasticity" above.

### Sequential fallback retained

The `_grpo_session_step_sequential` path is still in the codebase (gets
hit when sessions in a list don't have uniform 2-turn shape). Useful
for: (a) debugging, (b) future "true multi-turn" experiments where
walker state DOES carry across turns within a step. Not on the
production path for Wave 4 anymore.
