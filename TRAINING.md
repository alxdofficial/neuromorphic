# Training Instructions

## v9 (Differentiable Memory Graph) — Current

```bash
# Standard training: LM + memory graph end-to-end
python -u -m src.v8.train --bs 8 --steps 30000

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 12 --steps 30000 --no-memory

# Resume from checkpoint
python -u -m src.v8.train --bs 8 --steps 60000 --resume outputs/v8/<run>/v8_step30000.pt
```

### Architecture

Split-scan with differentiable memory graph (end-to-end backprop):
- **Lower scan**: layers 0-2 + BatchedPCM (bmm across all C=16 columns, RMSNorm, surprise)
- **Memory**: 16 segments of 128 tokens, per-token dendritic FC neuron dynamics (differentiable)
- **Inject**: H_enriched = H_mid + sigmoid(gate) * mem_signals (mid-scan)
- **Upper scan**: layers 3-4 on memory-enriched representations (surprise as side input)
- **Output**: proj_down -> LayerNorm -> lm_head
- **Per-neuron modulator**: each neuron has its own MLP (D_mem→64→3) predicting gate/decay
- **Dendritic FC**: per-neuron learned weights at branch and group levels

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~104M |
| Lower scan | layers 0-2 (D=2048, d_inner=1024) |
| Upper scan | layers 3-4 |
| PCM | BatchedPCM at split point (bmm, RMSNorm) |
| Memory neurons | 1024, 96 connections, sigmoid routing, ~22.5M learned params |
| Dendritic FC | per-neuron branch weights [N,8,12,D] + group weights [N,2,4,D] |
| Per-neuron modulator | each neuron's own MLP (128→64→3), 8.65M total |
| Segment length | 128 tokens (16 segments per chunk) |
| D_mem = D_cc | 128 |
| trace_decay | 0.95 |
| T (chunk) | 2048 tokens |
| Training | Single AdamW, memory LR = 0.3x LM LR |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--no-memory        # LM-only baseline (no memory graph)
--resume PATH      # Resume from checkpoint
--no-compile       # Disable torch.compile
--lr FLOAT         # Learning rate (default 3e-4)
--mem-lr-scale F   # Memory graph LR multiplier (default 0.3)
--save-dir PATH    # Output directory (default outputs/v8)
--save-interval N  # Checkpoint interval (default 5000)
--keep-checkpoints N  # Keep only last N checkpoints (default 3)
--snapshot-interval N # Memory graph snapshot interval (default 1000)
--plot-interval N  # Steps between auto-generated plots (default 500)
--log-interval N   # Steps between metric logging (default 50)
```

### Outputs

All outputs go to `outputs/v8/<run_id>/`:
- `v8_step{N}.pt` — checkpoints (full model state_dict + memory runtime state)
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, memory graph health)
- `config.json` — run configuration
- `plots/` — auto-generated training curves, memory health
- `snapshots/` — periodic memory graph state dumps

### Analysis

```bash
# Plot training curves from metrics
python -m scripts.plot_training outputs/v8/<run_id>/
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
- v8 RL-based code is preserved on the `v8-rl-neuromod` branch
