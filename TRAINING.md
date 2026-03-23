# Training Instructions

## v8 (Neural Memory Graph) — Current

```bash
# Full v8: split-scan + per-token memory graph + RL neuromodulator
python -u -m src.v8.train --bs 12 --steps 61035

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 12 --steps 61035 --no-memory
```

### Architecture

Split-scan with mid-scan memory injection:
- **Lower scan**: layers 0-3 + per-CC PCM (surprise + gain modulation)
- **Memory**: 8 segments of 256 tokens, per-token neuron loop (receive -> integrate -> message)
- **Inject**: H_enriched = H_mid + gate * mem_signals (mid-scan, not end)
- **Upper scan**: layers 4-6 on memory-enriched representations
- **Output**: proj_down -> LayerNorm -> lm_head
- **RL**: per-segment REINFORCE with discounted returns, entropy bonus, LR schedule

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~103M |
| Lower scan | layers 0-3 (D=2048, d_inner=1024) |
| Upper scan | layers 4-6 |
| PCM | at split point, learnable gain_scale |
| Memory neurons | 1024, 96 presynaptic connections, L1-normalized weights |
| action_every | 256 tokens (8 segments per chunk) |
| D_mem = D_cc | 128 |
| Neuromodulator | 3-layer MLP hidden=2048, obs=387, act=225 |
| T (chunk) | 2048 tokens |
| RL | Per-segment REINFORCE, gamma=0.99, entropy bonus 0.01 |
| Neuromod LR | 3e-4 with warmup + cosine decay to 10% |
| Throughput | ~44K tok/s with memory, ~85K without (RTX 4090, BS=12) |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--compile          # Enable torch.compile
--no-compile       # Disable torch.compile
--no-memory        # LM-only baseline (no memory graph, no RL)
--grad-ckpt        # Enable gradient checkpointing
--lr FLOAT         # Learning rate (default 3e-4)
--save-dir PATH    # Output directory (default outputs/v8)
--save-interval N  # Checkpoint interval (default 5000)
--keep-checkpoints N  # Keep only last N checkpoints (default 3)
--snapshot-interval N # Memory graph snapshot interval (default 1000)
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
