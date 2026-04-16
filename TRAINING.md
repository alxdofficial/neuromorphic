# Training

Quick-start commands for the two-phase training pipeline. For architectural
background see `docs/design.md`; for the training protocol rationale see
`docs/training_strategy.md`.

## Prerequisites

1. Tokenized data shards live under `data/`. Regenerate with:
   ```bash
   python scripts/prepare_data.py --tokens 1.5B --seed 42
   ```
   This produces `pile_train.bin` + `pile_val.bin` with document-level disjoint
   splits (SHA-256 hash exclusion — see `scripts/prepare_data.py`).

2. Always run Python with `-u` so stdout isn't buffered during long runs.

3. `outputs/` is in `.gitignore`. Commit code before starting long runs.

## Full pipeline: bootstrap + iterative cycles

The end-to-end driver is `src/train_loop.py`. It orchestrates:

```
bootstrap  →  (phase 1 [codebook+decoder frozen]  →  phase 2 GRPO)  ×  N cycles
```

The codebook and decoder are trained once in bootstrap and then frozen across
all cycles, so the code semantics phase 2 GRPO trains against are stable
across cycles. The logit head keeps training in phase 1 to adapt to LM
improvements, then gets updated by GRPO in phase 2.

**Typical 1.5B-token config** (BS=72, RTX 4090, ~10h @ 42K tok/s):

```bash
python -u -m src.train_loop \
    --work-dir outputs/tiny_01 \
    --bs 72 \
    --phase2-bs 24 \
    --phase2-group-size 8 \
    --bootstrap-tokens 500_000_000 \
    --phase1-tokens-per-cycle 10_000_000 \
    --cycles 20
```

Token budget breakdown:
- Bootstrap: 500M tokens (~3.3h at 42K tok/s)
- Per cycle: ~50M tokens (10M phase 1 + 40M phase 2 curriculum)
- 20 cycles: 1,000M tokens
- **Total: 1,500M tokens** (matches baseline training budget)

Use `--skip-bootstrap` if `bootstrap.pt` already exists in `--work-dir`, and
`--start-cycle N` to resume mid-run (picks up from the last completed cycle's
`phase2.pt`).

## Sub-commands (one-off invocation)

### Phase 1 / bootstrap (`src.train`)

```bash
python -u -m src.train \
    --bs 72 \
    --steps 54253 \
    --lr-target-step 48828 \
    --save-dir outputs/run1
```

Important flags:
- `--resume PATH` — resume from a checkpoint (restores optimizer, scheduler,
  runtime state, dataloader offset).
- `--freeze-codebook-decoder` — freezes codebook + decoder params in the
  `DiscreteActionPolicy` (used in cycle phase 1 to keep code semantics stable
  for phase 2 GRPO to train against). Does **not** freeze the logit head.
- `--freeze-modulator` — freezes the logit head only (rarely useful; keeps
  the policy stationary while dynamics co-adapt).
- `--no-memory` — LM-only baseline (disables the memory graph read).
- `--eval-interval N` — held-out eval every N steps (`phase="A-val"` shard).
  `--eval-batches`, `--eval-warmup-batches`, `--eval-bs` tune the eval pass.

The step callback automatically anneals the Gumbel-softmax temperature from
1.0 → 0.3 linearly across `--lr-target-step`, and triggers periodic
dead-code reset during bootstrap (gated on `codebook.requires_grad`, so
cycle phase 1 runs skip the reset).

### Phase 2 GRPO (`src.train_phase2`)

```bash
python -u -m src.train_phase2 \
    --checkpoint outputs/run1/phase1_end.pt \
    --out outputs/run1/phase2.pt \
    --bs 8 \
    --group-size 8
```

Phase 2 uses LM CE reward: each modulation event's reward is the windowed
negative cross-entropy of the next-token predictions under the frozen LM,
with memory on vs off in adjacent rollouts. Only the logit head in
`DiscreteActionPolicy` trains — codebook, decoder, memory dynamics, and LM
are all frozen.

Important flags:
- `--stage1-tokens` ... `--stage4-tokens` — per-stage token budgets for the
  curriculum (reward windows 512 / 1024 / 2048 / 4096).
- `--warmup-batches N` — forward-only batches to warm the memory state at
  phase-2 BS before GRPO starts (phase-1 BS=72 memory state is resized to
  phase-2 BS via lane tile/trim).
- `--entropy-coeff` — coefficient on categorical entropy bonus (default 0.01).
- `--group-size` — K rollouts per batch item (default 8). GRPO normalizes
  advantages within each K-group.
- `--eval-interval`, `--eval-batches`, `--eval-warmup-batches`, `--eval-bs`.
- `--eval-seed` (default 0) — RNG seed for the held-out eval loader. Kept
  decoupled from `--seed` so different replication runs score on the same
  held-out batches, making comparisons meaningful.

## Outputs

Per run directory:
- `bootstrap.pt`, `cycle_NN/phase1_end.pt`, `cycle_NN/phase2.pt` — checkpoints
- `metrics.jsonl`, `cycle_NN/phase2_metrics.jsonl` — per-step metrics
- `plots/` — auto-generated training plots (re-rendered each cycle)

## Monitoring

During training, watch:
- `loss` / `eval_ce_loss` — LM CE, primary quality metric
- `aux_loss` / `aux_ce_ratio` / `eval_aux_loss` — memory-head CE and ratio to main CE
- `mem_leverage_ce` — eval_ce_loss_no_mem − eval_ce_loss (memory's contribution)
- `lm_grad_norm`, `dyn_grad_norm`, `mod_clip_norm` — per-pool grad norms
- `mod_grad_norm`, `mod_action_norm` — modulator health
- `applied_dW_norm`, `applied_dDecay_norm` — actual magnitude of plasticity
- `W_offdiag_norm`, `W_offdiag_max`, `W_hebbian_offdiag_cos` — memory graph structure
- `h_norm`, `h_max`, `msg_norm`, `msg_max` — state magnitudes (watch for tanh saturation)
- `mem_scale_abs_mean`, `mem_scale_abs_max` — drift of the memory readout scale
- `τ` (tau) — Gumbel-softmax temperature (anneals 1.0 → 0.3 across LR schedule)
- `lane_W_divergence`, `lane_W_relative_div` — how much BS lanes diverge
  (expected — each lane reflects its own content stream)
- `stream_restarts_total` — data pipeline health (should stay 0)

Phase 2 specific:
- `reward_mean` / `reward_mean_complete` — windowed CE reward (complete-window
  mean excludes incomplete end-of-sequence slots)
- `eval_ce` trend — should monotonically decrease across GRPO steps if the
  policy is genuinely learning
- `per_cell_logpi_std` / `frac_all_k_same_code` — policy diversity across cells

## Baselines

Three baselines trained from scratch on the same 1.5B tokens (The Pile) with
the same tokenizer (TinyLlama 32K):

```bash
python -u auxiliary_repos/baselines/eval_scripts/train_baseline.py --model gpt2-small
python -u auxiliary_repos/baselines/eval_scripts/train_baseline.py --model pythia-160m
python -u auxiliary_repos/baselines/eval_scripts/train_baseline.py --model mamba-130m
```

Results (1.5B tokens, RTX 4090):
| Model | Params | Val Loss | Val PPL |
|-------|--------|----------|---------|
| GPT-2 small | 124M | 2.955 | 19.2 |
| Pythia-160m | 160M | 2.736 | 15.4 |
| Mamba-130m | 130M | 3.294 | 27.0 |
| **Neuromorphic LM** | **109M** | **TBD** | **TBD** |
