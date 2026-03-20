# Training Instructions

## v8 (Neural Memory Graph) — Current

```bash
# Full v8: scan stack + memory graph + PPO neuromodulator
python -u -m src.v8.train --bs 8 --steps 10000 --compile

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 8 --steps 10000 --compile --no-memory
```

### Architecture

Single-pass scan stack (7 layers, D=2048) with per-token memory access:
- **Scan**: all 7 layers + per-CC PCM → H, surprise (parallel over T=2048, once)
- **Memory loop**: step memory graph per token, neuromodulator acts every 8 tokens
- **Output**: logits = output_head(H + gate * mem_signal) — cheap, position-wise
- **RL update**: sample K=4 trajectories, compare losses, policy gradient

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~113M |
| Scan layers | 7 (shared, full D=2048, single pass) |
| Memory neurons | 8192 (8 blocks × 1024) |
| D_mem = D_cc | 128 |
| Neuromodulator | 3-layer MLP hidden=2048, ~19.5M params (actor only) |
| T (chunk) | 2048 tokens |
| RL | sampling-based, K=4 trajectories, actions every 8 tokens |

### CLI Options

```bash
--bs N             # Batch size (default 8)
--steps N          # Training steps (default 10000)
--compile          # Enable torch.compile
--no-compile       # Disable torch.compile
--no-memory        # LM-only baseline (no memory graph, no PPO)
--grad-ckpt        # Enable gradient checkpointing
--lr FLOAT         # Learning rate (default 3e-4)
--save-dir PATH    # Output directory (default outputs/v8)
--save-interval N  # Checkpoint interval (default 5000)
```

### Outputs

All outputs go to `outputs/v8/<run_id>/`:
- `v8_step{N}.pt` — checkpoints
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, PPO stats)
- `config.json` — run configuration

---

## v7 (Single Scan Stack) — Previous

```bash
python -u -m src.train --tier a --phase A --bs 16 --compile
```

See `docs/archive/` for v7 architecture docs.

---

## Data Pipeline

Download The Pile locally before training:

```bash
python scripts/prepare_data.py --tokens 12B --seed 42
cat data/pile/manifest.json
```

## General Notes

- Always use `-u` flag with python to disable output buffering
- `--compile` is required for reasonable throughput on CUDA
- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
