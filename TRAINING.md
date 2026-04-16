# Training

Branch `conv-grid-modulator`. For the design see `docs/design_conv_modulator.md`;
for the ordered implementation plan see `docs/plan_conv_modulator.md`.

## Prerequisites

1. Tokenized data shards live under `data/`. Regenerate with:
   ```bash
   python scripts/prepare_data.py --tokens 1.5B --seed 42
   ```
   This produces `pile_train.bin` + `pile_val.bin` with document-level disjoint
   splits (SHA-256 hash exclusion — see `scripts/prepare_data.py`).

2. Always run Python with `-u` so stdout isn't buffered during long runs.

3. `outputs/` is in `.gitignore`. Commit code before starting long runs.

## Phase 1 (bootstrap) — the only trainable phase on this branch

Phase 2 (GRPO) is deferred until the pretrained-LM pivot; see
`src/train_phase2.py`. For now, just train the conv-grid memory module
end-to-end with Gumbel-softmax in phase 1.

```bash
python -u -m src.train \
    --bs 72 \
    --steps 54253 \
    --lr-target-step 48828 \
    --save-dir outputs/conv_mod_run1
```

Important flags:
- `--resume PATH` — resume from a checkpoint (restores optimizer, scheduler,
  runtime state, dataloader offset).
- `--freeze-modulator` — freezes the conv encoder + logit head. Used later
  when phase 2 GRPO comes back (phase 1 trains everything else).
- `--freeze-codebook-decoder` — freezes the codebook and the
  conv-transpose decoder. Same purpose as before: stabilize code semantics.
- `--no-memory` — LM-only baseline (disables the memory graph read).
- `--eval-interval N` — held-out eval every N steps.

The step callback automatically anneals the Gumbel-softmax temperature from
1.0 → 0.3 linearly across `--lr-target-step`, and triggers periodic
dead-code reset during bootstrap (gated on `codebook.requires_grad`).

## What changed vs the old multi-cell design

The old 8-cell × 32-neuron architecture has been replaced with a single
connectivity pool of N=256 neurons. The per-cell MLP encoder is replaced by
a conv-over-edge-grid encoder, and the dense MLP decoder is replaced by a
conv-transpose generator that produces full-rank ΔW. See design doc for the
full story. Old checkpoints are not loadable on this branch.

## Outputs

Per run directory:
- `bootstrap.pt`, `phase1_end.pt` — checkpoints
- `metrics.jsonl` — per-step metrics
- `plots/` — auto-generated training plots

## Baselines

Three baselines trained from scratch on the same 1.5B tokens (The Pile) with
the same tokenizer (TinyLlama 32K):

| Model | Params | Val Loss | Val PPL |
|-------|--------|----------|---------|
| GPT-2 small | 124M | 2.955 | 19.2 |
| Pythia-160m | 160M | 2.736 | 15.4 |
| Mamba-130m | 130M | 3.294 | 27.0 |
| **Neuromorphic LM** | **~84M** | **TBD** | **TBD** |

Scripts for the baselines live under `auxiliary_repos/baselines/eval_scripts/`.
