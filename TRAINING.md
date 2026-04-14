# Training

Quick-start commands for the multi-phase training pipeline. For architectural
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

The end-to-end driver is `src/train_loop.py`. It orchestrates bootstrap →
(phase 1 main → action collection → codebook fit → phase 2 curriculum) × N.

**Current validated config** (1.5B total tokens, BS=80, RTX 4090, ~25h):

```bash
python -u -m src.train_loop \
    --work-dir outputs/v12 \
    --bs 80 \
    --phase2-bs 8 \
    --phase2-group-size 8 \
    --bootstrap-tokens 500_000_000 \
    --phase1-tokens-per-cycle 10_000_000 \
    --action-collection-tokens 2_000_000 \
    --cycles 20
```

Token budget breakdown:
- Bootstrap: 500M tokens (~3.2h at 43K tok/s)
- Per cycle: 50M tokens (8M phase 1 + 2M action collection + 40M phase 2)
- 20 cycles: 1,000M tokens
- **Total: 1,500M tokens** (matches baseline training budget)

Use `--skip-bootstrap` if `bootstrap.pt` already exists in `--work-dir`, and
`--start-cycle N` to resume mid-run (picks up from the last completed cycle's
`phase2.pt`).

## Sub-commands (one-off invocation)

### Phase 1 / bootstrap (`src.train`)

```bash
python -u -m src.train \
    --bs 80 \
    --steps 48828 \
    --lr-target-step 48828 \
    --save-dir outputs/run1
```

Important flags:
- `--resume PATH` — resume from a checkpoint (restores optimizer, scheduler,
  runtime state, dataloader offset).
- `--freeze-modulator` — freezes `mod_w1/b1/w2/b2` (cycle phase 1 setting).
- `--collect-actions` + `--action-db-out PATH` — record modulator outputs at
  every modulation event; paired with `--no-train` for action collection.
- `--no-train` — pure inference (no backward, no optimizer step, no LR
  scheduler advance); the LM stays stationary while actions are collected.
- `--no-memory` — LM-only baseline (disables the memory graph read).
- `--eval-interval N` — held-out eval every N steps (`phase="A-val"` shard).
  `--eval-batches`, `--eval-warmup-batches`, `--eval-bs` tune the eval pass.

### Phase 2 GRPO (`src.train_phase2`)

```bash
python -u -m src.train_phase2 \
    --checkpoint outputs/run1/phase1_end.pt \
    --codebook outputs/run1/codebook.pt \
    --out outputs/run1/phase2.pt \
    --bs 8 \
    --group-size 8
```

Phase 2 uses LM CE reward: each action's reward is negative of the windowed
cross-entropy of the next-token predictions under the frozen upper scan +
LM head on H_enriched = H_mid + mem_scale * readout.

Important flags:
- `--traj-noise-sigma FLOAT` — per-trajectory fixed perturbation magnitude
  in VQ latent space (default 3.0). Each K trajectory gets ξ_k ~ N(0, σ²)
  reused at every mod event, giving sustained trajectory identity. Essential
  for GRPO signal to survive window averaging. Set to 0 to disable.
- `--stage1-tokens`, ..., `--stage4-tokens` — per-stage token budgets for the
  curriculum (reward windows 512 / 1024 / 2048 / 4096).
- `--warmup-batches N` — forward-only batches to warm the memory state before
  GRPO starts.
- `--eval-interval`, `--eval-batches`, `--eval-warmup-batches`, `--eval-bs`.

### Codebook fit (`scripts.train_codebook`)

```bash
python -u -m scripts.train_codebook \
    --actions outputs/run1/action_database.pt \
    --out outputs/run1/codebook.pt \
    --epochs 20 \
    --num-levels 1 \
    --codes-per-level 256
```

Defaults to **1 level × 256 codes** (flat 256-way categorical). Residual
quantization is available via `--num-levels > 1` but the current validated
config is flat.

## Outputs

Per run directory:
- `bootstrap.pt`, `cycle_NN/phase1_main.pt`, `cycle_NN/phase1_end.pt`,
  `cycle_NN/phase2.pt` — checkpoints
- `cycle_NN/action_database.pt` — collected modulator actions
- `cycle_NN/codebook.pt` — fitted RVQ codebook
- `metrics.jsonl`, `cycle_NN/phase2_metrics.jsonl` — per-step metrics
- `plots/` — auto-generated training plots (re-rendered end of each phase 2 stage)

## Monitoring

During training, watch:
- `loss` / `eval_ce_loss` — LM CE, primary quality metric
- `aux_loss` / `aux_ce_ratio` / `eval_aux_loss` — memory-head CE and its ratio to main CE
- `quant_eval_ce` (phase 2) — VQ-argmax deterministic-policy eval; divergence
  from `eval_ce_loss` indicates proxy drift through the codebook
- `lm_grad_norm`, `dyn_grad_norm`, `mod_clip_norm` — per-pool grad norms (sqrt-param-scaled budgets)
- `mod_grad_norm`, `mod_action_norm` — modulator health
- `applied_dW_norm`, `applied_dDecay_norm` — actual magnitude of plasticity applied each step
- `W_offdiag_norm`, `W_offdiag_max`, `W_hebbian_offdiag_cos` — memory graph structure
- `h_norm`, `h_max`, `msg_norm`, `msg_max` — state magnitudes (watch for tanh saturation)
- `mem_scale_abs_mean`, `mem_scale_abs_max` — drift of the memory readout scale
- `lane_W_divergence`, `lane_W_relative_div` — how much BS lanes diverge
  in their W/decay/hebbian (expected — each lane reflects the modulator's
  response to its own content stream)
- `stream_restarts_total` — data pipeline health (should stay 0 after the
  exhaustion-raises-loudly fix)

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
| **Neuromorphic LM** | **105M** | **TBD** | **TBD** |
