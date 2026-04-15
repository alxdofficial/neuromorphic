# Training Strategy — Two-Phase Plan

## Premise

The neuromodulator's job is to learn **what is worth remembering for the long term**.
Doing this well requires credit assignment over long horizons (hundreds to thousands
of tokens), but the rest of the network (scan layers, dynamics MLPs, lm_head) is
trained perfectly well by short-horizon TBPTT.

We resolve that tension with a **bootstrap + iterative cycle** structure, plus an
integrated quantization bottleneck inside the modulator itself so that the phase-2
RL problem is over a small discrete code vocabulary rather than a high-dimensional
continuous action.

- **Bootstrap (one-time, ~500M tokens default)**: standard phase 1 TBPTT with
  everything trainable — LM, memory dynamics, and the full `DiscreteActionPolicy`
  (logit head + codebook + decoder). This gives the modulator a long stable
  warmup under the natural `ce_loss + mem_pred_loss` objective before any GRPO
  machinery comes online. Gumbel-softmax sampling with an annealed temperature
  (1.0 → 0.3) keeps the discrete bottleneck differentiable end-to-end. Dead-code
  reset fires periodically during bootstrap to keep the codebook healthy.

- **Iterative cycles (repeat indefinitely)**:
  - **Phase 1 (~10M tokens, codebook + decoder FROZEN; logit head keeps training)**:
    TBPTT trains the LM, memory dynamics, and the logit head. Freezing the
    codebook + decoder keeps the code *semantics* stable, so phase 2 GRPO has a
    consistent target to optimize. The logit head stays trainable so the
    modulator can adapt to LM improvements.
  - **Phase 2 (~40M tokens, everything frozen except the logit head)**: GRPO
    over the factored categorical policy (8 cells × 256 codes), with a
    curriculum-stepped reward window (512 → 1024 → 2048 → 4096). Rollouts use
    hard Categorical sampling; the gradient pass recomputes logits on the saved
    codes and scales by GRPO-normalized advantage.

  **Per-cycle total: ~50M tokens** (10M phase 1 + 40M phase 2 curriculum).

There is **no separate action-collection sub-phase** and **no per-cycle codebook
refit** in the current architecture — both are subsumed into the integrated
`DiscreteActionPolicy` module. Historically (pre-refactor) we ran a post-hoc
RVQ-VAE over collected actions every cycle, but that created cluster-identity
drift between cycles that actively hurt GRPO learning. Integrating quantization
into the modulator and freezing codebook+decoder after bootstrap fixes this.

---

## Phase 1: TBPTT

### What gets trained in bootstrap

All params in the model: LM scan, embedding, lm_head, mem_scale, memory
dynamics MLPs (state/msg/inject), `neuron_id`, the Hebbian/decay/W learnable
plasticity rates, and the **entire** `DiscreteActionPolicy` (logit head +
codebook + decoder). The Gumbel-softmax in phase 1 lets gradients flow through
the discrete code choice back into the logit head.

### What gets trained in cycle phase 1

Everything from bootstrap, **except**: codebook and decoder params are frozen
(`--freeze-codebook-decoder` flag on `src/train.py`). The logit head, LM,
memory dynamics, and learnable plasticity rates remain trainable. This keeps
the phase-2 GRPO target stable across cycles while letting the rest of the
system co-adapt.

### Key phase-1 settings

- `T = 128` tokens per segment, `tbptt_block = 8` (detach at 8-token boundaries)
- `modulation_interval = 4` tokens between modulator calls
- `BS = 80` (RTX 4090 max at ~113M params)
- `lr = 3e-4` → `3e-5` cosine, `lr_target_step` covers bootstrap + all cycles
- `mem_pred_weight = 0.1` on the memory-head auxiliary CE loss
- Gumbel τ annealing: linear 1.0 → 0.3 across `lr_target_step` (auto-wired in
  `src/train.py`'s step_callback)
- Dead-code reset: `threshold = 0.001`, `noise_std = 0.01`, every 500 steps
  during bootstrap only (disabled when `codebook.requires_grad = False`)

### Key telemetry

- `loss` / `eval_ce_loss` — LM CE (main quality metric)
- `aux_loss` / `eval_aux_loss` — memory-head CE, `aux_ce_ratio` = aux/main
- `mem_leverage_ce` — `eval_ce_loss_no_mem − eval_ce_loss` (memory contribution)
- `applied_dW_norm`, `applied_dDecay_norm` — actual plasticity magnitude
- `mod_action_norm` — raw modulator output magnitude
- `W_offdiag_norm`, `W_hebbian_offdiag_cos` — structure of W relative to hebbian
- `h_norm`, `msg_norm`, `mem_scale_abs_mean` — state magnitudes
- `τ` — current Gumbel temperature
- `lane_W_divergence` — how much lanes diverge (expected; diagnostic)

---

## Phase 2: GRPO over factored categorical codes

### What changes from phase 1

- **Frozen**: LM, memory dynamics MLPs, codebook + decoder, `neuron_id`, all
  learnable plasticity rates, `mem_scale`. Only `logit_w1/b1/w2/b2` are
  trainable (the per-cell policy head).
- **No CE/aux loss** in the gradient path. The gradient comes only from GRPO
  policy-gradient on the logit head. (`eval_ce_loss` is still tracked as a
  monitoring metric during eval.)
- **Optimizer**: fresh `AdamW` over the 4 logit-head tensors only.
- **Sampling**: hard Categorical (`torch.multinomial`) during rollouts,
  log-prob via standard `log_softmax`.

### Factored policy structure

Each modulation call produces `[BS, NC, K]` logits. Codes are sampled
independently per cell (no conditioning between cells), giving a factored
distribution `π(codes | state) = ∏_c π_c(code_c | state)`. Total combinations
per call: `K^NC = 256^8 ≈ 1.8e19`, though cells are structurally independent
so the effective policy complexity is `NC × K = 2048` categorical slots.

### Rollouts

For each batch:

1. Expand memory state to `K × BS` lanes, run `forward_segment_phase2` with
   `sample=True`. Hard Categorical sampling at each modulation event, codes
   stored along with `mod_input` for the gradient pass.
2. Reward per modulation event: windowed negative next-token CE over the next
   `W` tokens (curriculum stage's reward window). Incomplete windows at the
   end of the rollout are marked via a `complete` mask; the GRPO advantage
   normalization uses only complete slots.
3. Group baseline: for each `(call, sample)` slot, `baseline = mean_K(rewards)`.
4. Advantage: `A = (r - baseline) / (std_K + 1e-8)`, centered and scaled
   within each K-group.
5. Carry forward one trajectory end-state (uniform random pick, not best-of-K)
   to keep the memory state distribution unbiased relative to the deployed
   policy.

### GRPO gradient pass

```python
# mod_inputs: [K, n_calls, BS, NC, mod_in]   (detached, saved during rollout)
# codes:      [K, n_calls, BS, NC]            (long, no grad)
# advantage:  [K, n_calls, BS]                (broadcast to NC inside)

logits = policy.compute_logits(mod_inputs_flat)        # [*, NC, K]
log_probs = F.log_softmax(logits / τ, dim=-1)
log_pi = log_probs.gather(-1, codes_flat.unsqueeze(-1)).squeeze(-1)  # [*, NC]
entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
loss = -(advantage_broadcast * log_pi).mean() - entropy_coeff * entropy.mean()
loss.backward()
```

The autograd graph is **only the logit head** — codebook, decoder, memory
dynamics, and LM are all frozen. VRAM footprint is bounded by `K × BS × T × D`
(the readouts tensor during rollout) and the tiny logit-head gradient pass.

### Curriculum

| Stage | W (tokens) | Budget per cycle | BS ladder |
|---|---|---|---|
| 1 | 512  | 10M | 24 |
| 2 | 1024 | 10M | 16 |
| 3 | 2048 | 10M | 12 |
| 4 | 4096 | 10M | 8  |

Rollout segment length T = 2W per stage, so roughly half the modulation events
get complete (non-truncated) reward windows. BS scales down as W grows to keep
peak VRAM roughly constant across stages.

### Eval

Every 50 GRPO steps (`--eval-interval 50`):

- `evaluate()` runs a held-out eval pass via `model.forward_chunk`. In eval
  mode the `DiscreteActionPolicy` takes the **argmax** path (deterministic —
  no Gumbel noise, no Categorical sample). Measures what the deployed policy
  would do.
- `evaluate_quantized()` runs `forward_segment_phase2` with `sample=False`,
  measuring the same argmax policy via a slightly different code path.
- `evaluate_no_mem()` runs with memory off for the `mem_leverage_ce` diagnostic.

All eval passes use `--eval-warmup-batches` forward-only batches first to warm
the memory state before scoring.

---

## Bootstrap + iterative cycle loop

```
# === BOOTSTRAP (one-time, ~500M tokens) ===
# All params trainable: LM, memory, DiscreteActionPolicy (encoder + codebook + decoder).
# Gumbel-softmax with τ annealing; dead-code reset every 500 steps.
run_phase_1(model, tokens=500M, freeze_codebook_decoder=False)
save_checkpoint("bootstrap.pt")

# === ITERATIVE CYCLES ===
for cycle in 0..N_CYCLES:
    # Phase 1 — codebook + decoder frozen, everything else trainable.
    # Keeps code semantics stable for phase 2 GRPO to train against.
    run_phase_1(model, tokens=10M, freeze_codebook_decoder=True)

    # Phase 2 — only logit head trainable. GRPO over factored categorical.
    for (W, budget) in [(512, 10M), (1024, 10M), (2048, 10M), (4096, 10M)]:
        run_phase_2(model, reward_window=W, tokens=budget, group_size=8)

    save_checkpoint(f"cycle_{cycle}.pt")
```

A single cycle is ~50M tokens. 20 cycles × 50M + 500M bootstrap = 1.5B total
(matches baseline budget).

### What persists across cycles

| Item | Persists across cycles? | Trainable in bootstrap? | Cycle phase 1? | Phase 2? |
|---|---|---|---|---|
| LM scan, lm_head, embedding, mem_scale | Yes | Yes | Yes | No |
| State/msg/inject MLPs, neuron_id | Yes | Yes | Yes | No |
| Learnable plasticity rates (W/decay/hebbian logits) | Yes | Yes | Yes | No |
| Logit head (`logit_w1/b1/w2/b2`) | Yes | Yes | Yes | **Yes (GRPO)** |
| Codebook + Decoder | Yes | **Yes** | **No (frozen)** | No (frozen) |
| Memory runtime state (h, msg, W, decay, hebbian) | Yes (lifelong) | Updated | Updated | Updated |
| Phase-1 optimizer + LR scheduler + dataloader offset | Yes (passed through phase 2 checkpoint) | — | Yes | — (phase 2 has its own AdamW) |
| Phase-2 AdamW state | **No** — fresh per cycle's phase 2 | — | — | Yes |

### Per-lane memory state (no merging)

The memory runtime state (`W`, `decay`, `hebbian`) has a leading batch
dimension. Each lane's state reflects the shared modulator policy applied to
that lane's content stream — they diverge naturally, which is expected. The
modulator weights are `nn.Parameter`s shared across lanes and updated by every
backward pass, so the policy stays synchronized by construction.

`resize_to_bs(new_bs)` handles BS changes (phase 1 BS=80 → phase 2 BS=8/12/16/24
→ next cycle BS=80). Shrink samples lanes at random; grow tiles cyclically.
Transient state (`h`, `msg`, LM scan carries) is reset on BS change and
repopulated via `--warmup-batches` forward-only passes.

### Why freeze codebook + decoder in cycles?

If the codebook shifts between cycles, the "meaning" of code `k` in phase 2
cycle `N` differs from code `k` in cycle `N+1`. GRPO learns an association
between codes and outcomes; if that association is reset every cycle, the
phase-2 signal is destroyed cycle-to-cycle.

Freezing codebook + decoder after bootstrap means each code has a stable
"template" for memory update. Phase 2 GRPO then trains the logit head (the
policy) to select good templates given the observation. Across cycles, the
policy keeps improving; the templates stay fixed.

The logit head keeps training in cycle phase 1 because the LM improves every
cycle — the same memory-observation input no longer needs to produce the same
code. Letting the logit head absorb that shift keeps the system coherent.

---

## Hyperparameter summary

### Phase 1 (bootstrap + cycle)
| Knob | Value | Notes |
|---|---|---|
| `tbptt_block` | 8 | |
| `T` | 128 | |
| `BS` | 80 | |
| `lr` | 3e-4 → 3e-5 | cosine, over `lr_target_step` |
| `mem_pred_weight` | 0.1 | |
| Gumbel τ schedule | 1.0 → 0.3 linear | across `lr_target_step` |
| Dead-code reset | every 500 steps, threshold 0.001 | bootstrap only |
| `--freeze-codebook-decoder` | off for bootstrap, on for cycles | |

### Phase 2 (per cycle)
| Knob | Value | Notes |
|---|---|---|
| Frozen | everything except `logit_w1/b1/w2/b2` | |
| `lr` | 1e-4 | fresh AdamW |
| Group size K | 8 | `--group-size` |
| Entropy coeff | 0.01 | |
| Temperature τ | 1.0 | |
| Curriculum (W, budget) | (512, 10M), (1024, 10M), (2048, 10M), (4096, 10M) | auto-advance |
| Segment T per stage | 2 × W | half the mod events get complete windows |
| BS ladder | 24 / 16 / 12 / 8 | scales with W to hold VRAM |
| Reward | −mean(next-token CE over W) | complete-window mask |
| Advantage | `(r − mean_K) / (std_K + 1e-8)` | per-(call, sample) |
| Eval cadence | every 50 steps | argmax deterministic policy |
| Warmup batches | 8 | forward-only, warms memory at phase-2 BS |

### Bootstrap + iterative loop
| Knob | Value | Notes |
|---|---|---|
| Bootstrap tokens | 500M | `--bootstrap-tokens` |
| Phase 1 tokens / cycle | 10M | `--phase1-tokens-per-cycle` |
| Phase 2 tokens / cycle | 40M | sum of 4 stage budgets |
| Total / cycle | 50M | |
| N cycles | 20 | 20 × 50M + 500M = 1.5B total |

---

## Open questions / future work

- Whether the long-horizon hypothesis is correct. Phase 2 is the experiment that tests it.
- Whether K=256 codes × 8 cells is the right effective vocabulary size, or whether
  the action manifold collapses to fewer meaningful clusters.
- Whether the `code_dim=32` bottleneck limits decoder expressiveness — the rank
  of the decoded action set is bounded by `min(code_dim, decoder_hidden)` = 32.
- Whether GRPO with K=8 trajectories has enough variance reduction without a critic.
- Whether the shared advantage broadcast across the 8 cells (every cell sees the
  same per-slot advantage) causes policy collapse toward cell-uniform behavior —
  diagnostic: watch `per_cell_logpi_std` and `frac_all_k_same_code`.

### Things that would invalidate the plan

- If phase 1 doesn't plateau on `ce_loss` in the bootstrap budget, the dynamics
  aren't well-enough trained and phase 2 doesn't make sense yet.
- If the modulator collapses to near-zero outputs during bootstrap, the initial
  codebook is meaningless and phase 2 can't recover. Monitor `mod_action_norm`.
- If phase 2 actively hurts `eval_ce_loss` even at the shortest window (W=512),
  the discrete-action + GRPO approach probably isn't right for this problem.
