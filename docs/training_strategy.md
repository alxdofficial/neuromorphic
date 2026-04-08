# Training Strategy — Two-Phase Plan

## Premise

The neuromodulator's job is to learn **what is worth remembering for the long term**.
Doing this well requires credit assignment over long horizons (hundreds to thousands
of tokens), but the rest of the network (scan layers, dynamics MLPs, lm_head) is
trained perfectly well by short-horizon TBPTT.

This creates a tension: a single training regime that works for everything has to
choose between (a) short TBPTT — fast and stable, but the modulator only learns
"helps the next ~8 tokens," and (b) long TBPTT — gives the modulator long-horizon
credit but is slow, memory-intensive, and overkill for the dynamics MLPs.

We resolve it with a **bootstrap + iterative cycle** structure:

- **Bootstrap (one-time, ~200M tokens)**: standard phase 1 TBPTT with all params
  trainable, including the modulator. This gives the modulator a long stable
  warmup under the natural `ce_loss + mem_pred_loss` objective before any
  GRPO machinery comes online. The dynamics co-adapt to the modulator and the
  modulator settles into a baseline competent policy. This is *only* run once,
  at the very start, before any codebook fit.

- **Iterative cycles (repeat indefinitely)**:
  - **Phase 1 (~50M tokens, modulator frozen)**: TBPTT trains everything *except*
    the modulator. The modulator's weights are frozen (`requires_grad=False`)
    so its long-horizon policy from the previous cycle's phase 2 is preserved
    exactly. The dynamics MLPs, scan, lm_head co-adapt to whatever the modulator
    is currently doing.
  - **Codebook refresh**: collect modulator action snapshots during last steps
    of phase 1, fit a fresh Residual VQ-VAE (RVQ) on them.
  - **Phase 2 (~50M tokens, everything frozen except modulator)**: GRPO over
    discretized action space, with a curriculum-stepped reward window
    (512 → 2048 → 4096). Discrete categorical log-pi sidesteps the variance hell
    of high-dim continuous PG; long-horizon reward gives the modulator the
    credit signal TBPTT structurally cannot provide.

The iterative loop is the answer to "the codebook is a snapshot that bounds
phase 2's expressivity." By rebuilding the codebook every cycle, the modulator's
action space is never frozen — each cycle captures the current distribution and
gives GRPO a fresh, distribution-tracking action vocabulary.

The freeze of the modulator during phase 1 (after bootstrap) is the answer to
"phase 1's short-horizon TBPTT gradient would undo phase 2's long-horizon GRPO
work." See the *Critical: modulator is frozen in phase 1 after bootstrap*
section below for details.

---

## Phase 1: TBPTT (current state)

This is what already exists in the codebase. No new training algorithm needed; just
new telemetry so we can detect plateau and decide when to switch to phase 2.

### What's already in place

- Loss: `total_loss = ce_loss + mem_pred_weight * mem_pred_loss`
- TBPTT: detach all memory state every `tbptt_block=8` tokens
- All params trained: LM scan, embedding, lm_head, mem_scale, neuron_id,
  state/msg/inject/modulator MLPs
- Per-cell modulator with live in-block surprise signal (`s_mem_live`,
  `s_mem_ema_fast`, `readout_drift`) feeding the modulator at every
  modulation step (`modulation_interval=4`)
- Memory is lifelong (no EOT reset on memory state, only on LM scan carry)
- Cross-segment carry of `prev_readout`, `readout_drift`, etc.

### Telemetry to add (before any phase-1 training run)

The plateau decision in phase 1 is the single most important judgment call in
the whole strategy. We need data, not vibes. Logged at every `log_interval` step:

| Metric | Why |
|---|---|
| `ce_loss`, `mem_pred_loss`, `total_loss` | Standard training curves |
| `mem_pred_loss / ce_loss` | Ratio — is memory contributing meaningfully? |
| `lm_grad_norm`, `mem_grad_norm` | Already logged |
| `mod_grad_norm` (per-cell mean) | When this drops < 0.5 of its peak, modulator is no longer learning much from TBPTT |
| `mod_action_norm` (per-cell mean of \|delta_W\| and \|delta_decay\|) | Does the modulator's output magnitude stabilize? |
| `mod_action_var` (per-cell variance of action across batch) | Is the modulator producing diverse actions or collapsing? |
| `W_norm`, `W_sparsity`, `W_max` (already in smoke_test) | W health |
| `s_mem_live` mean & ema | Modulator's input signal strength |
| `decay_logit` mean | Persistence regime the modulator has settled on |

Optional but worth having:
- A periodic dump of the action distribution (histogram of `delta_W` magnitudes
  per cell) — needed anyway for the codebook fit, so we may as well log it.

### Phase 1 duration

- **Bootstrap phase 1**: ~200M tokens (one-time). All params trainable
  including modulator. This is the warmup before iterative cycles begin.
- **Cycle phase 1**: ~50M tokens per cycle. Modulator frozen. Dynamics
  re-adapt to whatever the previous cycle's GRPO modulator looks like.

Both use a fixed token budget; the telemetry is for monitoring health, not for
plateau-triggered switching. When the budget is exhausted, phase 1 ends.

### What phase 1 does NOT do

- It does not train the modulator on long-horizon credit. The modulator gets
  ~8-token TBPTT credit only.
- It does not validate the long-horizon hypothesis. That's phase 2's job.

---

## Codebook refresh (between phases, every cycle)

At the end of every phase 1, fit a fresh Residual VQ-VAE on the modulator's
current action distribution. The codebook is **not** persistent across cycles —
each cycle gets its own codebook trained on the current modulator's actions.

### Action database collection

During the **last N steps of phase 1** (e.g. N=2000), record every modulator
output to a database. Each record:

```
action_t = concat([delta_W.flatten(), delta_decay.flatten()])  # [BS, NC, ~8000]
```

We flatten across cells and across the W and decay slots, treating each per-cell
modulator call as one "action vector." Store as a flat tensor of shape
`[N_actions, NC, action_dim]`. Total size: `N_steps * BS * (T / mod_interval)`
actions. With N=2000, BS=96, T=128, mod_interval=4 → ~6M actions × 8 cells × ~1056 dims
≈ 200GB if stored full precision. We need to **subsample** — keep ~100K random
actions, which is plenty for fitting a 4096-effective codebook.

Actions are recorded **per-cell** because we have a per-cell modulator with cell-
local state. The codebook is trained per-cell or shared across cells — TBD when
we look at the actual data, but **default to shared** (one codebook for all 8 cells)
unless cell-specific patterns are obvious.

### RVQ-VAE training

**Architecture**:
- Encoder: 2-layer MLP, 512 hidden, action_dim → 32-dim latent
- Quantizer: Residual VQ with **4 levels of 16 codes** (effective vocab = 16⁴ = 65,536)
- Decoder: 2-layer MLP, 512 hidden, 32-dim latent → action_dim
- Code dim: 32 (Yu et al. ViT-VQGAN principle — low code dim avoids degenerate
  nearest-neighbor in high dim)

**Loss**: standard VQ-VAE
```
L = ||decode(quantize(encode(a))) - a||² + β * ||sg(z) - e||² + ||z - sg(e)||²
```
With β = 0.25 (commitment loss weight, Oord 2017 default). EMA codebook updates
with decay 0.99. Dead-code resampling: any code used in < 1% of a 100-step batch
window is replaced with a random encoder output from the current batch.

**Training**: 10-20 epochs on the subsampled action database, AdamW lr=3e-4.
This is a standalone training step, not part of the LM training loop. ~minutes
on the same GPU.

### Codebook validation

Before committing to phase 2, sanity-check the codebook:

- **Code usage histogram**: are all 65K effective combinations used? If only
  100 are, the codebook collapsed and we need to retrain (or reduce the level
  count).
- **Reconstruction error**: how well does `decode(quantize(encode(a)))` reconstruct
  the original action? Per-element MSE should be small relative to action norm.
- **Reconstructed action sanity**: feed reconstructed actions through the actual
  memory loop and check that mem_pred_loss is comparable to using raw actions.
  This is the *real* test that the codebook captures the meaningful action space.

If validation fails (codebook collapse, large reconstruction error, large
mem_pred_loss gap), retrain with different hyperparameters: smaller codebook,
more levels, larger encoder. Don't proceed to phase 2 with a bad codebook.

Save the trained codebook as `codebook_v1.pt`.

---

## Phase 2: GRPO over discrete codes

### What changes from phase 1

- **Frozen**: embedding, scan layers, lm_head, mem_scale, neuron_id, all dynamics
  MLPs (state, msg, inject), memory_head projection (proj_down, ln_final). Set
  `requires_grad = False`.
- **Trainable**: only `mod_w1, mod_b1, mod_w2, mod_b2` (the per-cell modulator MLP).
- **Loss**: only the GRPO policy gradient. No `ce_loss`, no `mem_pred_loss`
  in the gradient path. (`mem_pred_loss` is still computed during eval as a
  monitoring metric and during rollouts as the reward source.)
- **Optimizer**: fresh `AdamW` over modulator params only, no weight-decay
  groups, no LM/mem split. Lower LR than phase 1 (e.g. 1e-4) since we're
  fine-tuning a converged module.

### Action representation in phase 2

The modulator's continuous head still emits the raw `delta_W + delta_decay`
output (we don't change the head architecture). At each modulator call, the
continuous output is **encoded → quantized → decoded** through the frozen
RVQ-VAE before being applied to memory state:

```python
raw_action = modulator(state)              # continuous, as in phase 1
z = vq.encoder(raw_action)                 # [latent_dim=32]
codes = vq.quantize(z)                     # tuple of 4 ints in [0, 16)
z_q = vq.codes_to_latent(codes)            # quantized latent
quantized_action = vq.decoder(z_q)         # back to action_dim
delta_W, delta_decay = unpack(quantized_action)
W += delta_W; decay_logit += delta_decay
```

**The policy** is `π(codes | state) = ∏_l π_l(c_l | state, c_<l)` where each
`π_l` is a 16-way categorical induced by:

```python
logits_l = -‖z - codebook_l[i]‖² / τ  for i in 0..15
```

(Distance-based logits with temperature τ. Sampling from this distribution is
equivalent to sampling a code; argmax recovers the deterministic phase-1 behavior.)

This formulation has two important properties:

1. **At τ → 0 and σ_explore = 0, phase 2 reduces to phase 1.** The deterministic
   modulator output passes through the codebook and produces exactly the action
   the deterministic head would have produced (modulo VQ reconstruction error).
   So phase 2 starts from the phase-1 policy by construction.
2. **Gradients flow through the encoder** to update modulator params. We compute
   `log π_l(c_l | state)` from the distance-based logits, and `c_l` is sampled
   in the rollout. The gradient `∂ log π / ∂ θ_mod` updates the modulator's
   continuous head to bias `z` toward sampled-code regions.

### GRPO rollouts

For each training batch:

1. **Sample K trajectories** with the same input tokens (same `H_mid` from
   phase-1-frozen scan layers, but different sampled codes per modulator call).
   K = 32 (group size). Run the trajectories in `no_grad`.
2. **Per modulator call site**, record `(state_t, codes_t)`. Across K trajectories
   × T/M modulator calls, this is ~8K records per batch.
3. **Compute rewards per action**. For an action at token t, reward is:
   ```
   r_t = -mean(mem_pred_loss[t : t + W])
   ```
   where W is the curriculum-controlled reward window (256 → 512 → 1024 → 2048).
   Note: `mem_pred_loss` is computed in `no_grad` with the frozen `lm_head` and
   `proj_down`, so it's a stable yardstick.
4. **Group baseline**: for each action position t, compute the mean reward over
   the K trajectories: `b_t = mean(r_t over K)`. Advantage: `A_t = r_t - b_t`.
5. **Advantage normalization**: per-batch `A_t = (A_t - mean(A_t)) / (std(A_t) + 1e-8)`.

### GRPO gradient pass

Single batched modulator forward over the stacked `(state_t, codes_t)` records:

```python
mu = modulator(state_batch)                # batched modulator forward
z = vq.encoder(mu)                          # [B*N_actions, 32]
log_pi = 0
for level in range(4):
    logits_l = -‖z - codebook_l[:]‖² / τ   # [B*N_actions, 16]
    log_pi_l = log_softmax(logits_l).gather(codes_batch[:, level])
    log_pi = log_pi + log_pi_l
    z = z - codebook_l[codes_batch[:, level]]  # next residual

loss = -(advantage_batch * log_pi).mean()
loss.backward()
optimizer.step()
```

The **autograd graph is just the modulator forward + the VQ encoder forward +
the categorical log-probs**. No memory dynamics, no scan layers, no decoder.
VRAM footprint is small and bounded by `B * N_actions * action_dim`.

### Curriculum schedule (per cycle)

Within a single phase 2 run, the reward window steps through a fixed schedule
of (W, token_budget) stages. Each stage trains for its budget then advances:

| Stage | W (tokens) | Token budget per cycle | Notes |
|---|---|---|---|
| 1 | 512 | 25M | First real long-horizon test |
| 2 | 2048 | 15M | Mid-range, where most useful retrieval lives |
| 3 | 4096 | 10M | Stress test for the longest credit horizon |

Total phase 2 per cycle: **50M tokens**. Total cycle: **100M tokens** (50M phase
1 + 50M phase 2). The curriculum is automatic, not operator-advanced — each
stage runs to its budget then the next stage starts.

The user can monitor and abort if a stage diverges (eval `mem_pred_loss` rises
significantly above start-of-stage baseline), but the default is to let the
curriculum run.

### Phase-2 segment length

Phase 2 rollouts use **segment length T = current curriculum window W**. Phase 1's
constraint of T=128 came from TBPTT activation memory, which doesn't apply here
because rollouts are `no_grad`. At W=4096, each rollout processes 4096 tokens of
memory dynamics with no gradient activations stored.

### Eval cadence

Every 100 GRPO updates:

- Run a full eval pass with deterministic modulator (no sampling, argmax codes).
- Compute `ce_loss`, `mem_pred_loss` over a fixed eval set.
- Compare to phase-1 baseline. If `mem_pred_loss` is meaningfully better → phase 2
  is helping. If it's worse → something is wrong, hard stop.

### Phase-2 stopping criteria (per cycle)

Phase 2 within a cycle ends when:
- All 3 curriculum stages have completed their token budgets, **OR**
- Eval `mem_pred_loss` has risen >10% above start-of-cycle baseline (early abort).

When phase 2 ends, control returns to the outer loop, which starts the next cycle's
phase 1 with the post-GRPO modulator weights as the starting point.

## Bootstrap + iterative cycle loop

The full training pipeline is:

```
state: model weights (LM + memory + modulator)

# === BOOTSTRAP (one-time, ~200M tokens) ===
# All params trainable, including modulator. No GRPO. No codebook.
# Just normal phase 1 TBPTT until the modulator and dynamics are well-warmed.
run_phase_1(model, tokens=200M, freeze_modulator=False)
save_checkpoint("bootstrap.pt")

# === ITERATIVE CYCLES ===
for cycle in 0..N_CYCLES:
    # Phase 1 — train dynamics + LM via TBPTT, modulator FROZEN
    run_phase_1(model, tokens=50M, freeze_modulator=True)

    # Action collection — capture modulator outputs at end of phase 1
    actions = collect_actions(model, tokens=2M)

    # Codebook refresh — fit fresh RVQ-VAE on current actions
    codebook = train_codebook(actions, levels=4, codes_per_level=16)

    # Phase 2 — freeze everything but modulator, GRPO over codes
    for (W, budget) in [(512, 25M), (2048, 15M), (4096, 10M)]:
        run_phase_2(model, codebook, reward_window=W, tokens=budget)

    save_checkpoint(f"cycle_{cycle}.pt")
```

A single cycle is ~100M tokens. At phase-1 throughput of ~50K tok/s and phase-2
throughput of ~10-20K tok/s effective (depending on K and W), one cycle takes
roughly 2-4 hours of wall-clock on a 4090. 10 cycles = 1-2 days.

Bootstrap is a separate ~1 hour (~200M tokens at 50K tok/s).

### What persists across cycles, what doesn't

| Item | Persists? | Trainable in bootstrap? | Trainable in cycle phase 1? | Trainable in phase 2? |
|---|---|---|---|---|
| LM scan, lm_head, embedding | Yes | Yes | Yes | No (frozen) |
| State/msg/inject MLPs, neuron_id | Yes | Yes | Yes | No (frozen) |
| **Modulator weights** | **Yes** | **Yes** | **No (frozen)** | **Yes (GRPO)** |
| Memory runtime state (h, msg, W, decay_logit) | Yes (lifelong) | Updated | Updated | Updated |
| Codebook | **No** — refit per cycle | n/a | n/a | Used (frozen) |
| Phase-2 categorical head state | **No** — distance-based logits stateless | n/a | n/a | n/a |
| GRPO optimizer state | **No** — fresh per cycle's phase 2 | n/a | n/a | Yes |
| Phase-1 optimizer state | Yes — reused across cycles | Yes | Yes | n/a |

### Critical: modulator is frozen in phase 1 after bootstrap

There's a structural conflict between the two phases' modulator training signals:

- Phase 1's modulator gradient (via TBPTT, `tbptt_block=8`) optimizes 8-token
  short-horizon prediction.
- Phase 2's modulator gradient (via GRPO, windowed reward up to W=4096)
  optimizes long-horizon prediction.

These objectives are **not the same function**. A modulator policy that's
optimal for "what helps in 2048 tokens" might actively hurt "what helps in 8
tokens." Without intervention, each cycle's phase 1 would partially undo the
previous cycle's phase 2 work — pulling the modulator back toward short-term
optima that GRPO had moved it away from.

**Mitigation**:
- **During bootstrap** (~200M tokens, before any cycle): modulator trains
  normally. This warms it up to baseline competence so that the first codebook
  fit has a non-degenerate action distribution to learn from.
- **During every iterative cycle's phase 1**: modulator is **frozen**
  (`requires_grad=False` on `mod_w1/b1/w2/b2`). Only dynamics MLPs, scan
  layers, lm_head, and embedding train. The modulator weights from the
  previous cycle's phase 2 are preserved exactly through phase 1.

After bootstrap, the modulator is **only ever updated by GRPO**. TBPTT cannot
teach the long-horizon credit assignment the modulator needs, so it shouldn't
be the modulator's training signal at all once we have a working GRPO loop.

This is philosophically clean: phase 1 in cycles 1+ becomes "dynamics
fine-tuning under the current modulator policy" — analogous to how RLHF freezes
the LM during reward model training and only updates during the policy step.

**Soft failure mode that remains**: even with the modulator frozen, the
dynamics MLPs in cycles 1+ phase 1 are trained against `ce_loss + mem_pred_loss`,
which depend on modulator outputs. The dynamics can in principle "compensate"
for modulator behavior changes — e.g. learning to treat the new write pattern
similarly to the old one, washing out phase 2's long-horizon improvements.
This is softer than direct modulator regression but real. **Monitoring**:
track eval `mem_pred_loss` at the 2048-token window at end of each cycle's
phase 1. If it has degraded significantly from end-of-phase-2, the dynamics
are washing out the GRPO signal — respond by reducing phase 1's token budget
in subsequent cycles or by adding the dynamics MLPs to the freeze set in
cycles 2+.

The "categorical head" question — what happens when the codebook re-indexes
between cycles? **Answer: nothing.** Because the policy is implemented as
distance-based logits over the current codebook (`logits[i] = -‖encode(mu) - codebook[i]‖²`),
there's no learned per-code parameter to re-index. The modulator's continuous
head doesn't know or care which code is which; it just produces a `mu`, and the
nearest-code lookup is done against whichever codebook is currently active. When
the codebook changes between cycles, the policy automatically tracks the new
codes without any parameter reset.

This is the entire reason we chose distance-based logits over a separate
classifier head. With a classifier head we'd need Hungarian matching across
codebook refreshes; with distance-based logits we get free re-indexing.

---

## Hyperparameter summary

### Phase 1 (current; minor additions)
| Knob | Value | Notes |
|---|---|---|
| `tbptt_block` | 8 | unchanged |
| `T` (segment length) | 128 | unchanged |
| `BS` | 96 | unchanged |
| `lr` | 3e-4 | unchanged |
| `mem_lr_scale` | 0.3 | unchanged |
| `mem_pred_weight` | 0.1 | unchanged |
| logging | + 4 new metrics (mod_grad_norm, mod_action_norm, mod_action_var, action histogram dump) | new |

### Codebook fit
| Knob | Value | Notes |
|---|---|---|
| Action database size | ~100K subsampled actions | from last 2000 phase-1 steps |
| RVQ levels | 4 | residual quantization |
| Codes per level | 16 | small per-level codebook |
| Effective vocabulary | 65,536 | 16⁴ |
| Latent dim | 32 | low to avoid degenerate NN |
| Encoder/decoder hidden | 512 | 2-layer MLP each |
| Commitment β | 0.25 | Oord 2017 default |
| EMA decay | 0.99 | standard |
| Dead-code resample | < 1% usage over 100 steps | VQ-BeT recipe |
| Training epochs | 10-20 | until reconstruction loss plateaus |

### Phase 2 (per cycle)
| Knob | Value | Notes |
|---|---|---|
| Frozen | everything except mod_w1/b1/w2/b2 | |
| `lr` | 1e-4 | lower than phase 1 — fine-tune |
| GRPO group size K | 8 | start small; scale up if variance too high |
| Curriculum (W, budget) | (512, 25M), (2048, 15M), (4096, 10M) | automatic, not operator-advanced |
| Segment length T | = W | rollouts in no_grad |
| Logits temperature τ | 1.0 | tunable, may need lower if exploration too noisy |
| Advantage normalization | yes | per-batch mean/std |
| Eval cadence | every 100 GRPO updates | |

### Bootstrap + iterative loop
| Knob | Value | Notes |
|---|---|---|
| Bootstrap tokens (one-time) | 200M | normal phase 1, modulator trains |
| Phase 1 tokens / cycle | 50M | modulator FROZEN |
| Phase 2 tokens / cycle | 50M | sum of curriculum stage budgets |
| Action collection tokens | ~2M | last steps of phase 1 |
| Total tokens / cycle | ~100M | excluding bootstrap |
| Modulator trains during | Bootstrap (TBPTT) + every phase 2 (GRPO) | never via TBPTT after bootstrap |
| N cycles | open | run until plateau or budget exhausted |

---

## Build order

1. **Phase 1 telemetry** ✅. Done in commit 13b17fb. `mod_grad_norm`,
   `mod_action_norm`, `mod_action_var` printed at every log_interval.

2. **Action collection.** Add `MemoryGraph.collect_modulator_action()` and a
   trainer collection mode. Snapshot one action per training step at end of
   segment. Flush to a database tensor.

3. **RVQ-VAE module + standalone trainer.** `src/codebook/rvq.py` with
   `ResidualVQ` and `ActionVQVAE`. `scripts/train_codebook.py` loads action
   database, fits codebook, validates, saves.

4. **Phase 2 trainer.** `src/phase2/trainer.py` with `Phase2Trainer`.
   - `MemoryGraph.forward_segment_phase2()` runs the memory loop with
     VQ sampling and records `(state, codes)` per modulator call.
   - `rollout()` runs K trajectories batched as one BS=K*BS forward.
   - `compute_rewards()` computes per-token mem_pred_loss and windows it.
   - `grpo_step()` does the modulator-only gradient pass over stacked records.
   - Curriculum loop over (W, budget) stages.

5. **Phase 2 entry point.** `src/train_phase2.py` — CLI that loads a phase 1
   checkpoint + a codebook, runs `Phase2Trainer` for one cycle's phase 2.

6. **Outer loop driver.** `src/train_loop.py` — orchestrates the full
   iterative pipeline: phase 1 → action collection → codebook fit → phase 2 →
   repeat. Single-process, single-GPU. Each step calls into the existing
   sub-trainers; no duplication of training logic.

7. **Tests.** `tests/test_rvq.py` for RVQ correctness, `tests/test_phase2.py`
   for the GRPO gradient flow on a tiny config.

Each step is independently testable. Don't start step N+1 until step N is
working.

---

## Open questions / future work

### Things we don't know yet
- Whether the long-horizon hypothesis is correct. Maybe phase 1 alone is
  good enough and phase 2 doesn't help. Phase 2 is the experiment that tests this.
- Whether 65K effective codes is the right size, or whether the action manifold
  collapses to ~100 distinct meaningful clusters (in which case smaller RVQ).
- Whether the deterministic phase-1 → distance-based-logit phase-2 handoff is
  smooth or whether the modulator's policy abruptly shifts at the phase boundary.
- Whether GRPO with K=32 has enough variance reduction without a critic.

### Future extensions (not in scope for the initial build)
- **Iterative phase A/B**: alternate phase 1 (full TBPTT, dynamics co-adapt)
  and phase 2 (GRPO on codes). Refit the codebook at each cycle. Promising
  but no published precedent and several real failure modes (codebook
  re-indexing, distribution shift). Build only after one-shot phase 2 is
  validated.
- **State-conditional codebook**: encode action and state jointly so the K
  codes available at any state are state-relevant. Tried by some skill-discovery
  papers, mixed results, adds chicken-and-egg coupling. Skip for now.
- **Critic for variance reduction**: standard PPO move. We're holding off
  to keep the system simple, but if GRPO variance turns out to be the
  bottleneck, this is the cheapest mitigation.
- **Reward beyond windowed mem_pred_loss**: maybe long-horizon retrieval
  benchmarks (associative recall, needle-in-haystack), maybe a learned reward
  model. Out of scope until we know the windowed-loss reward isn't enough.

### Things that would invalidate the plan
- If phase 1 doesn't plateau on `ce_loss` within a reasonable budget, we're
  not training the dynamics well enough and phase 2 doesn't make sense yet.
- If the action distribution at end of phase 1 is degenerate (modulator
  outputs near-zero for all calls), the codebook fit is meaningless and we
  need to fix the modulator architecture / phase-1 training first.
- If phase 2 actively hurts even at W=256, the discrete-action + GRPO
  approach probably isn't right and we should reconsider option α (long-TBPTT)
  from the design discussion.
