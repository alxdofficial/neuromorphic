# Training Instructions

## v9-backprop with 2-Pass Simulation — Current (branch: v9-backprop)

```bash
# Standard training
python -u -m src.v8.train --bs 48 --steps 30000

# LM-only baseline (no memory graph)
python -u -m src.v8.train --bs 48 --steps 30000 --no-memory

# Resume from checkpoint
python -u -m src.v8.train --bs 48 --steps 60000 --resume outputs/v9/<run>/v9_step30000.pt
```

### Architecture

Split-scan LM with differentiable memory graph, trained end-to-end by backprop:
- **Lower scan**: 2 layers (D=2048, d_inner=580, GLU)
- **PCM**: Predicts H_{t+1} directly. Surprise mixed via split-point MLP.
- **Memory graph**: 512 neurons, D_neuron=256, K=32 connections
  - 2-pass simulation: freeze inter-neuron messages, run T MLP steps per pass
  - Per-neuron modulator MLP (hidden=80): predicts w_conn, decay, primitives
  - Per-neuron state MLP + message MLP: update hidden state and generate messages
  - Dendritic tree: hierarchical integration of neighbor signals
  - Structural plasticity: rewire weakest connections between chunks
- **Inject**: H_enriched = H_mid + mem_scale * mem_readout  [learnable per-dim scale]
- **Upper scan**: 2 layers on memory-enriched representations

### Config (Tier A)

| | Value |
|---|---|
| Total params | 110M (LM=52M, Memory=58M) |
| Lower scan | 2 layers (D=2048, d_inner=580) |
| Upper scan | 2 layers |
| Memory neurons | 512, D=256, K=32 connections |
| Modulator | Per-neuron MLP, hidden=80 |
| State/Message MLPs | Per-neuron, hidden=24 |
| Dendritic tree | 2 branches × 16 synapses |
| Segment length | T=128 tokens (1 segment per chunk) |
| Simulation | 2-pass (2 gathers + 256 MLP steps) |
| Throughput | 24.8K tok/s at BS=48 on RTX 4090 |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--lr FLOAT         # Learning rate (default 3e-4)
--no-memory        # LM-only baseline (no memory graph)
--resume PATH      # Resume from checkpoint
--save-dir PATH    # Output directory (default outputs/v9)
--save-interval N  # Checkpoint interval (default 5000)
--keep-checkpoints N  # Keep only last N checkpoints (default 3)
--snapshot-interval N # Memory graph snapshot interval (default 1000)
--log-interval N   # Steps between metric logging (default 50)
```

### Outputs

All outputs go to `outputs/v9/<run_id>/`:
- `v9_step{N}.pt` — checkpoints (LM + memory params + runtime state)
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, memory health)
- `config.json` — run configuration
- `snapshots/` — periodic memory graph state dumps

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
