# Training Instructions

## v8 (Neural Memory Graph) — Current

```bash
# Full v8: scan stack + per-token memory graph + RL neuromodulator
python -u -m src.v8.train --bs 8 --steps 10000

# LM-only baseline: scan stack only, no memory
python -u -m src.v8.train --bs 8 --steps 10000 --no-memory
```

### Architecture

Single-pass scan stack (7 layers, D=2048) with per-token memory graph:
- **Scan**: all 7 layers + per-CC PCM -> H, surprise (parallel over T=2048, once)
- **Memory**: 8 segments of 256 tokens, per-token neuron loop (receive -> integrate -> message)
- **Output**: logits = output_head(H + gate * mem_signal) — cheap, position-wise
- **RL**: per-segment REINFORCE with discounted returns, batch-mean baseline

### Config (Tier A)

| | Value |
|---|---|
| Total params | ~103M |
| Scan layers | 7 (shared, full D=2048, single pass) |
| Memory neurons | 1024, 96 presynaptic connections each |
| action_every | 256 tokens (8 segments per chunk) |
| D_mem = D_cc | 128 |
| Neuromodulator | 3-layer MLP hidden=2048, ~10M params (actor only) |
| T (chunk) | 2048 tokens |
| RL | Per-segment REINFORCE, discounted returns (gamma=0.99), 8 actions/chunk |
| Throughput | ~42K tok/s with memory, ~85K without (RTX 4090, BS=4) |

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
```

### Outputs

All outputs go to `outputs/v8/<run_id>/`:
- `v8_step{N}.pt` — checkpoints
- `metrics.jsonl` — per-step metrics (loss, ppl, tok/s, RL stats)
- `config.json` — run configuration

---

## v7 (Single Scan Stack) — Previous

```bash
python -u -m src.train --tier a --phase A --bs 16 --compile
```

---

## Data Pipeline

Download The Pile locally before training:

```bash
python scripts/prepare_data.py --tokens 12B --seed 42
cat data/pile/manifest.json
```

## General Notes

- Always use `-u` flag with python to disable output buffering
- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
