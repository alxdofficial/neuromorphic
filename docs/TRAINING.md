# V8 Training Health Monitoring Guide

A comprehensive guide for interpreting training metrics, diagnosing issues,
and monitoring the health of all model components during training.

## Quick Reference: What to Check

When checking on a training run, look at these in order:

1. **Loss** — is it decreasing?
2. **Throughput** — stable? (drop = something changed)
3. **Memory gate** — moving from 0.5? (stuck = LM ignoring memory)
4. **h_norm / msg_norm** — stable? (exploding = dynamics broken)
5. **tanh saturation** — low? (high = messages are binary noise)
6. **Firing rate** — port and internal both active?
7. **Decay** — reasonable range? (all at 0 or 1 = neuromod pushing to extremes)

---

## Metrics Reference

### Training Metrics (every step)

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `loss` | Decreasing over time | CE loss on next-token prediction | Plateau early, NaN, increasing |
| `ppl` | Decreasing (starts ~50K, should reach ~100-500) | exp(loss) | Stuck above 1000 after 10K steps |
| `aux_loss` | Small, decreasing | PCM prediction loss | Increasing = PCM diverging |
| `lr` | Follows schedule (warmup → peak → cosine decay) | LM learning rate | Unexpected shape |
| `nm_lr` | Same shape as `lr` | Neuromod learning rate | Missing = neuromod not training |
| `tok_s` | Stable (±10%) | Throughput in tokens/sec | Sudden drop = OOM or bug |
| `grad_norm` | Stable, typically 0.5-5.0 | LM gradient norm after clipping | Spikes >100 = instability |

### Memory Graph Metrics (every step)

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `mem_h_norm` | Stable, proportional to sqrt(BS×N×D) | Total h state norm | Exploding = dynamics unstable |
| `mem_h_mean_abs` | Small positive (0.001-0.1) | Average absolute h per element | Growing unbounded |
| `mem_msg_norm` | Similar to h_norm | Total message norm | Diverging from h_norm |
| `mem_msg_rms_mean` | 0.01-0.5 (in tanh linear regime) | Average per-neuron message RMS | Near 1.0 = saturated, near 0 = dead |
| `mem_msg_rms_std` | Nonzero (variation is good) | Variation of message RMS across neurons | Zero = all neurons identical |
| `mem_tanh_saturated` | < 5% | Fraction of message dims near ±1 | > 20% = h too large or primitives wrong |
| `mem_prim_std` | Stable (from init noise) | Primitive diversity across neurons | Exploding = no RMS normalization |
| `mem_key_diversity` | Nonzero | Key vector diversity across neurons | Zero = all keys identical (bad routing) |

### Memory Gate (critical coupling signal)

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `mem_gate_mean` | Should MOVE from 0.5 during training | How much LM uses memory (sigmoid) | Stuck at 0.5 = LM ignoring memory |
| `mem_gate_min` | > 0 | Minimum gate across CCs | 0 = some columns completely ignore memory |
| `mem_gate_max` | < 1 | Maximum gate across CCs | 1 = some columns completely rely on memory |

**Interpretation:**
- Gate at 0.5 (init) after thousands of steps → LM hasn't learned to use or reject memory
- Gate moving toward 1.0 → LM is learning to use memory (good!)
- Gate moving toward 0.0 → LM is learning to reject memory (memory is noise)
- Different gates per CC → LM selectively uses memory for some columns

### Decay

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `mem_decay_mean` | 0.1-0.9 (diversity is good) | Average sigmoid(decay_logit) | All at same extreme (0 or 1) |
| `mem_decay_std` | > 0 | Variation across neurons | Zero = neuromod made all neurons identical |

**Interpretation:**
- Decay near 0 → neurons are memoryless (pure relay, respond to current input only)
- Decay near 1 → neurons are sticky (hold state, slow to change)
- Mixed decay values → different neurons serve different temporal roles (ideal)
- In Phase 1 (no neuromod): decay stays at init (0.5). This is fine.

### Firing Rate and Co-activation

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `mem_firing_rate` | 0.2-0.3 (by construction, ~25% from percentile) | Overall firing rate | 0 = dead, 1 = always firing |
| `mem_firing_rate_port` | Similar to nonport | Port neuron firing rate | Much higher/lower than internal = scale mismatch |
| `mem_firing_rate_nonport` | Similar to port | Internal neuron firing rate | Zero = internal neurons dead |
| `mem_usage_frac` | > 0.9 (most neurons active) | Fraction of neurons with firing_rate > 0.01 | < 0.5 = many dead neurons |

### Co-activation (Plasticity Signal)

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `mem_phi_mean` | Small positive or near 0 | Average co-activation across all pairs | Very high (>0.5) = everything correlates |
| `mem_phi_std` | > 0 | Variation in co-activation | Zero = no structure |
| `mem_phi_pos_frac` | 0.3-0.7 | Fraction of positively correlated pairs | 1.0 = all positive (no pruning will happen) |
| `mem_phi_neg_frac` | > 0 | Fraction of anti-correlated pairs | 0.0 = plasticity cannot prune |
| `mem_phi_abs_max` | > 0.1 | Strongest co-activation signal | Near 0 = no meaningful correlations |
| `mem_plasticity_rewires` | Growing, then tapering | Cumulative topology changes | Constant = plasticity stopped. Linear = never stabilizes |

**Interpretation:**
- phi_pos_frac = 1.0, phi_neg_frac = 0.0 → all neurons positively correlated → plasticity frozen (no anti-correlated connections to prune). This happened in early runs due to shared CC driving all neurons similarly.
- Rewires tapering → topology stabilizing (good)
- Rewires linear forever → topology never settles (too much exploration or noisy phi)
- Rewires stopped early → co-activation not measured (check phi_abs_max)

### PCM

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `pcm_gain_scale` | Should evolve from init (2.0) | Learnable surprise modulation scale | Stuck at init = PCM not learning |

### RL / Neuromod Metrics (Phase 2 only, every rl_collect_chunks steps)

| Metric | Healthy Range | What It Means | Red Flags |
|--------|---------------|---------------|-----------|
| `rl_policy_loss` | Decreasing (slowly) | REINFORCE policy gradient loss | Increasing = policy diverging |
| `rl_adv_mean` | Near 0 (baseline is good) | Mean advantage | Large magnitude = bad baseline |
| `rl_adv_std` | > 0 (signal exists) | Advantage signal strength | Decreasing = RL losing signal |
| `rl_cf_adv` | Informative (nonzero) | Counterfactual advantage (actual - counterfactual loss) | Always zero = counterfactual not measuring effect |
| `rl_nm_grad_norm` | Stable, < 10000 | Neuromod gradient norm | Spikes > 100K = instability |
| `rl_entropy` | Should not collapse to 0 | Policy entropy (exploration) | Zero = deterministic policy (no exploration) |
| `nm_logstd_prim` | Should evolve from init (-2.0) | Primitive action exploration rate | Frozen at init = not learning |
| `nm_logstd_key` | Should evolve from init (-2.0) | Key action exploration rate | Frozen at init = not learning |
| `nm_logstd_decay` | Should evolve from init (-2.0) | Decay action exploration rate | Frozen at init = not learning |

---

## Common Failure Patterns

### Pattern 1: Memory Ignored (Gate Stuck)

**Symptoms:** `mem_gate_mean` stays at ~0.5, loss matches no-memory baseline.

**Cause:** Memory signals are uninformative noise. The LM can't distinguish
"with memory" from "without memory" so the gate has no gradient signal.

**Fix:** This is expected in Phase 1 with random init memory. If it persists
after Phase 2 neuromod training, the memory graph isn't producing useful signals.

### Pattern 2: Primitive/Key Explosion

**Symptoms:** `mem_prim_std` or `mem_key_diversity` grows unbounded, `mem_tanh_saturated`
increases toward 1.0.

**Cause:** Neuromod pushes primitive/key magnitudes up, normalization not applied.

**Fix:** Check that `apply_actions()` applies RMS normalization on primitives
and L2 normalization on key after every neuromod action.

### Pattern 3: All Neurons Positively Correlated

**Symptoms:** `mem_phi_pos_frac` = 1.0, `mem_phi_neg_frac` = 0.0,
`mem_plasticity_rewires` stops growing.

**Cause:** All neurons are driven by the same CC input through the graph. With
random init primitives/keys, neurons don't differentiate enough for distinct
firing patterns.

**Expected in:** Phase 1 (frozen random memory params). Should improve in
Phase 2 as neuromod differentiates neurons.

### Pattern 4: RL Not Learning

**Symptoms:** `rl_cf_adv` always near 0, `nm_logstd_*` frozen at -2.0, entropy constant.

**Cause:** Neuromod actions have no measurable effect on loss. Could be:
- Normalizations erasing action effects (was the issue in the first 1.5B run)
- LM not using memory (gate stuck → actions don't reach the loss)
- Action space too large for the reward signal

**Fix:** Ensure Phase 1 trains the LM to use memory first, then Phase 2
neuromod has a loss-sensitive system to optimize.

### Pattern 5: h Explosion

**Symptoms:** `mem_h_norm` growing without bound, `mem_tanh_saturated` → 1.0.

**Cause:** Should NOT happen with the convex combination `h = d*h + (1-d)*received`.
If it does, check that decay_logit isn't somehow producing decay > 1.0
(sigmoid should prevent this).

### Pattern 6: Port/Internal Gap

**Symptoms:** `mem_firing_rate_port` >> `mem_firing_rate_nonport`, or h_norm
dominated by port neurons in connectivity snapshot.

**Cause:** CC signals enter at a much larger scale than internal graph messages.
The percentile-based firing should handle moderate gaps. Extreme gaps
(100×+) indicate CC normalization issues.

**Current state:** With tanh + RMS primitives, a ~16× gap is expected
and acceptable. The adaptive firing threshold handles it.

---

## Phase-Specific Monitoring

### Phase 1 (LM + frozen memory, no neuromod)

**What to watch:**
- `loss` — is the LM learning? Compare with no-memory baseline.
- `mem_gate_mean` — is the LM learning to use memory?
- `mem_h_norm`, `mem_tanh_saturated` — are dynamics stable?
- `mem_plasticity_rewires` — is topology evolving?
- `mem_firing_rate_port` vs `_nonport` — balanced activity?

**What's expected to be bad:**
- `phi_pos_frac` = 1.0 (all positive correlation — no neuromod to differentiate)
- `mem_prim_std` constant (primitives frozen)
- `mem_decay_mean` = 0.5 (decay frozen at init)
- No RL metrics (neuromod disabled)

### Phase 2 (Frozen LM, neuromod active)

**What to watch:**
- `rl_cf_adv` — is the counterfactual measuring a causal effect?
- `nm_logstd_*` — are exploration rates adapting?
- `mem_gate_mean` — should already be at a non-0.5 value from Phase 1
- `loss` — should improve from Phase 1 endpoint
- `mem_decay_mean/std` — neuromod should differentiate neurons
- `mem_phi_neg_frac` — should become nonzero as neurons differentiate

**What indicates success:**
- `rl_cf_adv` nonzero and informative (counterfactual detects action effects)
- Loss improves beyond Phase 1 endpoint
- `nm_logstd_*` moves from init (exploration is adapting)
- Neurons develop diverse decay values and primitive directions

---

## Generating Plots

```bash
# Generate all plots from a training run
python -m scripts.plot_training outputs/v8/<run_id>/

# With a specific snapshot for connectivity visualization
python -m scripts.plot_training outputs/v8/<run_id>/ --snapshot 5000
```

Plots are auto-generated during training every `--plot-interval` steps (default 500).

### Plot Panels Guide

**training_curves.png:**
- Training Loss — main signal, should decrease
- Perplexity — log scale, tracks loss
- Learning Rate — verify schedule shape
- Throughput — should be stable

**rl_curves.png** (Phase 2 only):
- Policy Loss — RL loss, should decrease slowly
- Counterfactual Advantage — causal effect of neuromod actions
- Advantage Std — signal strength, should not collapse to 0
- Policy Log-Std — exploration rates per action group
- Neuromod Gradient Norm — stability indicator

**memory_health.png:**
- State Norms — h and msg norms, should be stable
- Memory Gate — the critical LM-memory coupling signal
- Primitive Diversity — should be stable (Phase 1) or evolving (Phase 2)
- Decay — should be 0.5 (Phase 1) or diverse (Phase 2)
- Firing Rate — port vs internal, both should be active
- tanh Saturation — should be low (< 5%)
- Key & Primitive Diversity — routing and broadcast differentiation
- Neuron Usage & Co-activation — alive neurons and phi statistics
- Cumulative Plasticity Rewires — topology evolution

**connectivity_*.png** (from snapshots):
- Per-Neuron State Norms — port neurons vs internal
- Decay per Neuron — should show variation in Phase 2
- Key Vectors — routing selectivity patterns
- Fan-In per Neuron — how many neurons send to each

---

## CLI Quick Reference

```bash
# Phase 1: LM + frozen memory
python -m src.v8.train --bs 8 --steps 30517 --no-neuromod

# Phase 2: Frozen LM + neuromod (resume from Phase 1 checkpoint)
# Throughput: ~27K tok/s (counterfactual baseline costs ~2x memory graph)
python -m src.v8.train --bs 8 --steps 61035 --resume outputs/v8/<run>/v8_step30517.pt --freeze-lm

# No-memory baseline
python -m src.v8.train --bs 12 --steps 30517 --no-memory

# Check training health
python -c "import json; r=[json.loads(l) for l in open('outputs/v8/<run>/metrics.jsonl')]; print(f'step {r[-1][\"step\"]}: loss={r[-1][\"loss\"]:.3f} gate={r[-1].get(\"mem_gate_mean\",\"?\")} h_norm={r[-1].get(\"mem_h_norm\",\"?\")}')"
```

### All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bs` | 8 | Batch size |
| `--steps` | 30517 | Total training steps |
| `--no-neuromod` | off | Phase 1: disable neuromodulator, train LM + frozen memory graph |
| `--no-memory` | off | No-memory baseline (pure LM) |
| `--resume` | none | Resume from checkpoint path |
| `--freeze-lower` | off | Freeze lower scan layers (0-3) |
| `--freeze-lm` | off | Phase 2: freeze entire LM (lower + upper scan + embed + output head), only neuromod trains |
| `--action-every` | 256 | Segment length override (tokens per neuromod action) |
| `--plot-interval` | 500 | Steps between auto-generated plots |
| `--log-interval` | 10 | Steps between metric logging |
| `--no-compile` | off | Disable torch.compile |
