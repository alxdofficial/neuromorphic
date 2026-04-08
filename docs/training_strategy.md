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

We resolve it by training in two phases:

- **Phase 1 (current)**: TBPTT with `tbptt_block=8`. Trains everything. The
  modulator gets short-horizon credit, which is enough to bring it to a baseline
  competence. The dynamics MLPs, lm_head, scan layers all learn their actual jobs.
- **Phase 2**: Freeze everything except the modulator. Discretize the modulator's
  action space via Residual VQ-VAE (RVQ) fit on phase-1 actions. Train the
  modulator with GRPO over the discrete codebook, against a long-horizon windowed
  reward. The discretization makes per-step policy gradients well-conditioned;
  the long-horizon reward gives the modulator the credit signal it actually needs.

Phase 2 is **not** a replacement for phase 1 — it is a fine-tune of the modulator
on top of a converged phase-1 model. The phase-1 checkpoint is the anchor and is
never overwritten. If phase 2 fails or makes things worse, we can revert losslessly.

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

### Phase-1 plateau criteria (operator judgment)

The user will look at the curves and decide. Rough heuristics:

- `ce_loss` flat (within 2% of min) for ≥ 1000 steps
- `mem_pred_loss` flat for ≥ 1000 steps
- `mod_grad_norm` has dropped to < 50% of its phase-1 peak
- `mod_action_norm` and `mod_action_var` are stable

When all four are satisfied, phase 1 is done. Save a checkpoint as `phase1_anchor.pt`.

### What phase 1 does NOT do

- It does not train the modulator on long-horizon credit. The modulator gets
  ~8-token TBPTT credit only.
- It does not validate the long-horizon hypothesis. We don't know yet if a
  better long-horizon modulator policy actually helps `mem_pred_loss` over
  long windows. Phase 2 is the test of that hypothesis.

---

## Codebook fit (between phases)

Between phase 1 and phase 2, fit a Residual VQ-VAE on the modulator's recent
action distribution.

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

### Curriculum schedule (operator-controlled)

Reward window W steps through a fixed schedule, advanced manually by the user
based on training curves:

| Stage | W (tokens) | Notes |
|---|---|---|
| 1 | 256 | Easy credit assignment, modulator finds local patterns |
| 2 | 512 | First real long-horizon test |
| 3 | 1024 | Approaching the limit of useful prediction horizon |
| 4 | 2048 | Match the longest segment we'd realistically train against |

The user advances stages manually. Suggested signal: when eval `mem_pred_loss`
at the current window has been within 2% of its minimum for ≥500 GRPO steps
**and** the GRPO policy entropy is no longer dropping, advance to the next stage.

If a stage diverges (eval `mem_pred_loss` rises >5% above the phase-1 baseline),
the user reverts to the previous stage's checkpoint and considers adjusting
exploration noise, group size, or KL anchor weight.

### Phase-2 segment length

Phase 2 rollouts use **segment length T = current curriculum window W**. Phase 1's
constraint of T=128 came from TBPTT activation memory, which doesn't apply here
because rollouts are `no_grad`. At W=2048, each rollout processes 2048 tokens of
memory dynamics with no gradient activations stored — cheap.

### Eval cadence

Every 100 GRPO updates:

- Run a full eval pass with deterministic modulator (no sampling, argmax codes).
- Compute `ce_loss`, `mem_pred_loss` over a fixed eval set.
- Compare to phase-1 baseline. If `mem_pred_loss` is meaningfully better → phase 2
  is helping. If it's worse → something is wrong, hard stop.

### Phase-2 stopping criteria

Stop phase 2 when any of:
- All 4 curriculum stages have plateaued.
- Eval `mem_pred_loss` has been worse than phase-1 baseline for ≥1000 steps
  (phase 2 is hurting, not helping).
- Wall-clock budget exceeded.

The output is `phase2_modulator.pt` — only the modulator params are saved; everything
else is the unchanged phase-1 checkpoint.

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

### Phase 2
| Knob | Value | Notes |
|---|---|---|
| Frozen | everything except mod_w1/b1/w2/b2 | |
| `lr` | 1e-4 | lower than phase 1 — fine-tune |
| GRPO group size K | 32 | minimum for 65K action space without critic |
| Curriculum W | 256 → 512 → 1024 → 2048 | operator-advanced |
| Segment length T | = W | rollouts in no_grad |
| Logits temperature τ | 1.0 | tunable, may need lower if exploration too noisy |
| Advantage normalization | yes | per-batch mean/std |
| KL anchor to phase-1 modulator | 0.0 (default off) | enable if codebook drift hurts |
| Eval cadence | every 100 GRPO updates | |

---

## Build order

1. **Phase 1 telemetry** (next concrete task). Add the four new logged metrics
   plus action histogram dump. No training algorithm changes. Smoke-test on a
   short run.

2. **Phase 1 training run.** Train to plateau using the new telemetry. User
   monitors and decides when to stop. Save `phase1_anchor.pt`.

3. **Action database collection.** Add a hook in the modulator forward path
   that, when enabled, accumulates actions to a buffer. Run the last 2000
   steps of phase 1 with collection enabled. Save `action_database.pt`.

4. **RVQ-VAE module + training script.** Standalone trainer that loads
   `action_database.pt`, fits the RVQ codebook, validates it (usage histogram,
   reconstruction error, frozen-loop sanity check). Save `codebook_v1.pt`.

5. **Phase 2 trainer.** New training loop:
   - Loads `phase1_anchor.pt` and `codebook_v1.pt`.
   - Freezes all params except modulator.
   - Implements the GRPO rollout + categorical log-pi gradient pass over codes.
   - Implements per-action windowed reward with operator-controlled W.
   - Eval pass every 100 updates.
   - Save `phase2_modulator.pt`.

6. **Phase 2 training run.** User monitors curves and advances curriculum
   manually.

Each step is independently testable. Don't start step N+1 until step N is
working. If any step turns out infeasible, we revert to the prior checkpoint
and the project is no worse off than where we started.

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
