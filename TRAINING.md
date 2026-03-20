# Training Instructions

## v8 (Neural Memory Graph) — Current

```bash
# Full v8: scan stack + memory graph + PPO neuromodulator
python -u -m src.v8.train --bs 8 --steps 10000 --compile

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 8 --steps 10000 --compile --no-memory
```

### Architecture

Two-pass scan stack (8 layers, D=2048) with per-token memory access:
- **Pass 1**: Pre-memory scan + per-CC PCM → H, surprise (parallel over T=2048)
- **Memory loop**: step memory graph per token, neuromodulator acts every 8 tokens
- **Pass 2**: Post-memory scan with injected memory signals → logits (parallel)
- **PPO update**: train neuromodulator on collected experience at chunk end

### Config (Tier A)

| | Value |
|---|---|
| Total params | 115.4M |
| Scan layers | 8 (shared, full D=2048) |
| Memory neurons | 4096 (16 blocks × 256) |
| D_mem | 256 |
| Neuromodulator | 3-layer MLP, 6.6M params |
| T (chunk) | 2048 tokens |
| PPO | gamma=0.99, actions every 8 tokens |

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
