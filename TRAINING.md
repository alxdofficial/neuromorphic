# Training

Quick-start commands for the multi-phase training pipeline. For architectural
background see `docs/design.md`; for the training protocol rationale see
`docs/training_strategy.md`.

## Prerequisites

1. Tokenized data shards live under `data/`. Regenerate with:
   ```bash
   python scripts/prepare_data.py --tokens 12B --seed 42
   ```
   This produces `pile_train.bin` + `pile_val.bin` with document-level disjoint
   splits (SHA-256 hash exclusion — see `scripts/prepare_data.py`).

2. Always run Python with `-u` so stdout isn't buffered during long runs.

3. `outputs/` is in `.gitignore`. Commit code before starting long runs.

## Full pipeline: bootstrap + iterative cycles

The end-to-end driver is `src/train_loop.py`. It orchestrates bootstrap →
(phase 1 main → action collection → codebook fit → phase 2 curriculum) × N.

```bash
python -u -m src.train_loop \
    --work-dir outputs/v12 \
    --bs 96 \
    --phase2-bs 8 \
    --phase2-group-size 8 \
    --bootstrap-tokens 500_000_000 \
    --phase1-tokens-per-cycle 10_000_000 \
    --action-collection-tokens 2_000_000 \
    --cycles 5
```

Use `--skip-bootstrap` if `bootstrap.pt` already exists in `--work-dir`, and
`--start-cycle N` to resume mid-run (picks up from the last completed cycle's
`phase2.pt`).

## Sub-commands (one-off invocation)

### Phase 1 / bootstrap (`src.train`)

```bash
python -u -m src.train \
    --bs 96 \
    --steps 50000 \
    --lr-target-step 50000 \
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
    --group-size 8 \
    --reward-mode lm_ce
```

Important flags:
- `--reward-mode {lm_ce,mem_pred}` — `lm_ce` (default) runs the frozen upper
  scan + LM head on `H_enriched` for principled per-token CE reward; `mem_pred`
  uses the cheap memory-head proxy.
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
