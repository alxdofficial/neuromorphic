# Training

Branch `attention-neuromod`. For the current architecture see
`docs/design.md`; for the throughput / param history see `docs/RESULTS.md`.

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
`src/train_phase2.py`. For now, train the memory graph end-to-end with
Gumbel-softmax in phase 1.

```bash
python -u -m src.train \
    --bs 64 \
    --steps 54253 \
    --lr-target-step 48828 \
    --save-dir outputs/attn_neuromod_run1
```

Important flags:
- `--resume PATH` — resume from a checkpoint (restores optimizer, scheduler,
  runtime state, dataloader offset).
- `--freeze-modulator` — freezes the per-cell attention modulator + logit
  head. Used later when phase 2 GRPO comes back (phase 1 trains everything
  else).
- `--freeze-codebook-decoder` — freezes the codebook and the per-cell
  decoder. Stabilizes code semantics before phase 2.
- `--no-memory` — LM-only baseline (disables the memory graph read).
- `--eval-interval N` — held-out eval every N steps.

The step callback automatically anneals the Gumbel-softmax temperature from
1.0 → 0.3 linearly across `--lr-target-step`, and triggers periodic
dead-code reset during bootstrap (gated on `codebook.requires_grad`).

## Architecture summary (see `docs/design.md` for detail)

- NC=8 cells × Nc=32 neurons × D_n=256 per-neuron state (block-diagonal W)
- Per-token: LIF state update + W@msg + inject + readout, fused in Triton
- Every 4 tokens: msg MLP + hebbian EMA
- Every 16 tokens: per-cell attention neuromodulator → codebook → per-cell
  decoder → ΔW + Δdecay, γ-clamped EMA blend

Old pre-redesign checkpoints (shared-weights modulator + 2-layer state MLP)
are not loadable on the current design.

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
| **Neuromorphic LM** | **~82M** | **TBD** | **TBD** |

Scripts for the baselines live under `auxiliary_repos/baselines/eval_scripts/`.
