# Training Instructions

## v8 (Neural Memory Graph) — Current

```bash
# Phase 1: LM + frozen memory graph (no neuromod)
python -u -m src.v8.train --bs 8 --steps 30517 --no-neuromod

# Phase 2: Frozen LM + neuromod (resume from Phase 1)
python -u -m src.v8.train --bs 8 --steps 61035 --resume outputs/v8/<run>/v8_step30517.pt --freeze-lm

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 12 --steps 30517 --no-memory
```

### Architecture

Split-scan with mid-scan memory injection + three-factor learning:
- **Lower scan**: layers 0-3 + BatchedPCM (bmm across all C=16 columns, RMSNorm, surprise as side input)
- **Memory**: 16 segments of 128 tokens, per-token dendritic tree neuron loop
- **Inject**: H_enriched = H_mid + sigmoid(gate) * mem_signals (mid-scan, not end)
- **Upper scan**: layers 4-6 on memory-enriched representations (surprise as side input to first layer)
- **Output**: proj_down -> LayerNorm -> lm_head
- **RL**: GRPO trajectory scoring (8 trajectories, K=96 neurons, scored across 4 chunks)
- **Plasticity**: Three-factor learning (Hebbian eligibility traces gated by neuromod)

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~103M |
| Lower scan | layers 0-3 (D=2048, d_inner=1024) |
| Upper scan | layers 4-6 |
| PCM | BatchedPCM at split point (bmm, RMSNorm, no gain modulation) |
| Memory neurons | 1024, 96 presynaptic connections, key-based sigmoid routing |
| Dendritic tree | 3 levels: 8 branches of 12, 2 groups of 4, soma avg |
| action_every | 128 tokens (16 segments per chunk) |
| D_mem = D_cc | 128 |
| Neuromodulator | 2-layer MLP hidden=512, obs=516, act=2 (gate + decay_target) |
| hebbian_lr | 0.01, trace_decay=0.95 |
| T (chunk) | 2048 tokens |
| RL | GRPO (8 trajectories, K=96 neurons, 4 chunks, best state persists) |
| Neuromod LR | 3e-4 with warmup + cosine decay |
| Throughput | ~64K Phase 1, ~87K Phase 2, ~161K no memory (RTX 4090) |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--no-neuromod      # Phase 1: disable neuromodulator, train LM + frozen memory
--no-memory        # LM-only baseline (no memory graph, no RL)
--freeze-lm        # Phase 2: freeze LM, only neuromod trains via GRPO
--resume PATH      # Resume from checkpoint
--no-compile       # Disable torch.compile
--lr FLOAT         # Learning rate (default 3e-4)
--save-dir PATH    # Output directory (default outputs/v8)
--save-interval N  # Checkpoint interval (default 5000)
--keep-checkpoints N  # Keep only last N checkpoints (default 3)
--snapshot-interval N # Memory graph snapshot interval (default 1000)
--plot-interval N  # Steps between auto-generated plots (default 500)
--log-interval N   # Steps between metric logging (default 10)
```

### Outputs

All outputs go to `outputs/v8/<run_id>/`:
- `v8_step{N}.pt` — checkpoints (LM + neuromod + memory graph state)
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, RL, memory graph health)
- `config.json` — run configuration
- `plots/` — auto-generated training curves, RL curves, memory health
- `snapshots/` — periodic memory graph state dumps (~1.4MB each)

### Analysis

```bash
# Plot training curves from metrics
python -m scripts.plot_training outputs/v8/<run_id>/

# Deep per-token memory analysis from checkpoint
python -m scripts.analyze_memory outputs/v8/<run_id>/v8_step*.pt
```

---

## Data Pipeline

The Pile via HuggingFace streaming or pre-tokenized .bin shards:

```bash
python scripts/prepare_data.py --tokens 12B --seed 42
```

## General Notes

- Always use `-u` flag with python to disable output buffering
- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
