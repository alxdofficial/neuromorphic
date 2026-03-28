# Training Instructions

## v9-backprop — Current (branch: v9-backprop)

```bash
# Standard training: LM backprop + differentiable memory graph
python -u -m src.v8.train --bs 8 --steps 30000

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 12 --steps 30000 --no-memory

# Resume from checkpoint
python -u -m src.v8.train --bs 8 --steps 60000 --resume outputs/v8/<run>/v8_step30000.pt
```

### Architecture

Split-scan LM with fully differentiable memory graph, trained end-to-end by backprop:
- **Lower scan**: 2 layers (D=2048, d_inner=512, GLU)
- **PCM**: Predicts H_{t+1} directly from H_t. Surprise = predicted - actual.
- **Split-point MLP**: Combines H_mid + surprise via residual MLP (zero-init)
- **Memory graph**: 4096 neurons, D_neuron=32, K=128 connections
  - Per-step: gather → weight → dendritic tree → inject → state MLP → msg MLP → +neuron_id
  - Segment boundary: modulator predicts new w_conn, decay, primitives from hebbian traces
  - Chunk boundary: structural plasticity rewires connections via co-activation
- **Inject**: H_enriched = H_combined + sigmoid(gate) * mem_readout (mid-scan)
- **Upper scan**: 2 layers on memory-enriched representations
- **Output**: proj_down → LayerNorm → lm_head

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~114M (53M LM + 61M memory) |
| Lower scan | 2 layers (D=2048, d_inner=512) |
| Upper scan | 2 layers |
| PCM | Predict H_{t+1}, surprise via split-point MLP |
| Memory neurons | 4096, D=32, K=128 connections |
| Dendritic tree | 8 branches × 16 synapses, 2 groups × 4 branches |
| Per-neuron MLPs | state (64→24→32), message (64→24→32), modulator (193→16→161) |
| Neuron ID | Learnable [N, D=32] embedding, added to messages |
| Hebbian traces | Per-segment avg of |msg| × σ(w_conn), shape [N, K] |
| Structural plasticity | Swap 8 connections/neuron at chunk boundary via N² co-activation |
| Inject/Readout | Parameter-free: replicate → reshape / mean → reshape |
| D_neuron | 32 |
| Segment length | 128 tokens (64 steps at stride=2) |
| T (chunk) | 2048 tokens (16 segments) |
| Memory LR | 0.3× base LR, params in f32 |
| Triton | Fused dendritic gather kernel (fwd+bwd) |
| Throughput | ~2.1K tok/s at BS=8 on RTX 4090 (optimization in progress) |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--no-memory        # LM-only baseline (no memory graph)
--resume PATH      # Resume from checkpoint
--no-compile       # Disable torch.compile
--lr FLOAT         # Learning rate (default 3e-4)
--save-dir PATH    # Output directory (default outputs/v8)
--save-interval N  # Checkpoint interval (default 5000)
--keep-checkpoints N  # Keep only last N checkpoints (default 3)
--snapshot-interval N # Memory graph snapshot interval (default 1000)
--plot-interval N  # Steps between auto-generated plots (default 500)
--log-interval N   # Steps between metric logging (default 50)
```

### Outputs

All outputs go to `outputs/v8/<run_id>/`:
- `v8_step{N}.pt` — checkpoints (LM state_dict + memory params + runtime state)
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, memory health, hebbian, plasticity)
- `config.json` — run configuration
- `plots/` — auto-generated training curves
- `snapshots/` — periodic memory graph state dumps (per-neuron stats)

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
- Previous versions preserved on branches: v8-rl-neuromod, v8-broadcast-io-backup, v9-es-backup, v9-es-neuron-mlps, v9.1-triton-attempt, v10-design-backup
