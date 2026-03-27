# Training Instructions

## v9-ES with Broadcast I/O — Current

```bash
# Standard training: LM backprop + ES memory graph
python -u -m src.v8.train --bs 8 --steps 30000

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 12 --steps 30000 --no-memory

# Resume from checkpoint
python -u -m src.v8.train --bs 8 --steps 60000 --resume outputs/v8/<run>/v8_step30000.pt

# No ES warmup (start ES immediately)
python -u -m src.v8.train --bs 8 --steps 30000 --es-warmup 0
```

### Architecture

Split-scan LM with ES-trained memory graph:
- **Lower scan**: layers 0-2 + BatchedPCM (surprise signals)
- **Memory graph**: 1024 neurons, D_mem=128, K=96 connections, Triton-accelerated
  - Per-neuron: primitives, key, decay_logit, dendritic FC, modulator MLP
  - Broadcast inject: all neurons receive weighted CC signals (inject_w [N, C_mem])
  - Broadcast readout: all neurons contribute to output (readout_w [C_mem, N])
  - Per-segment modulator: gate_prim, gate_key, decay_mod from eligibility traces
- **Inject**: H_enriched = H_mid + sigmoid(gate) * mem_signals (mid-scan)
- **Upper scan**: layers 3-4 on memory-enriched representations (surprise as side input)
- **Output**: proj_down -> LayerNorm -> lm_head

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~94M |
| Lower scan | layers 0-2 (D=2048, d_inner=768) |
| Upper scan | layers 3-4 |
| PCM | BatchedPCM at split point |
| Memory neurons | 1024, K=96 connections, ~24.5M ES-trained params |
| Dendritic FC | per-neuron branch [N,8,12,D] + group [N,2,4,D] weights |
| Per-neuron modulator | MLP (D*5 → hidden → 3), zero-init output |
| Broadcast I/O | inject_w [N,16], readout_w [16,N] (all neurons participate) |
| D_mem = D_cc | 128 |
| Segment length | 128 tokens (16 per chunk) |
| T (chunk) | 2048 tokens |
| ES | K=96 neurons, 8 trajectories, σ=0.05, rank-based fitness shaping |
| Throughput | ~40K tok/s at BS=8 on RTX 4090 |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--no-memory        # LM-only baseline (no memory graph)
--resume PATH      # Resume from checkpoint
--no-compile       # Disable torch.compile
--lr FLOAT         # Learning rate (default 3e-4)
--es-warmup N      # Steps before ES starts (default 2000, set 0 to skip)
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
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, memory graph health, ES health)
- `config.json` — run configuration
- `plots/` — auto-generated training curves, memory health
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
- Previous versions preserved on branches: v8-rl-neuromod, v9-es-neuron-mlps, v9.1-triton-attempt, v10-design-backup
