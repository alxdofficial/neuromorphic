# Training Instructions for Neuromorphic LM (v5)

## Quick Start

```bash
# Tier A on RTX 4090
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --tier a --phase A --bs 20 --compile 2>&1 | tee outputs/train_a_$(date +%Y%m%d_%H%M%S).log"

# Attach to monitor
tmux attach -t neuromorphic-train

# Detach: Ctrl+B, then D
```

## Architecture (v5: Scan-Memory-Scan)

Three-stage cycle per N-token segment:
- **Stage 1**: Fast causal scan (element-wise recurrence, C independent columns)
- **Stage 2**: Memory ops (write-before-read with causal prefix sums)
- **Stage 3**: Integration scan (H + pm_delta + em_read + cum_em)

NTP training only (no FITB). Causal scans are inherently autoregressive.

## Model Tiers

| Tier | Params | D | D_embed | C | B | L_scan | M | Target GPU |
|------|--------|------|---------|---|---|--------|-----|------------|
| **A** | ~92M | 2048 | 384 | 16 | 6 | 12 | 256 | RTX 4090 |
| **B** | ~252M | 3072 | 512 | 16 | 12 | 16 | 512 | RTX 4090 |

Select tier with `--tier a` / `--tier b`.

**Baselines per tier:**

| Tier | Transformer | SSM |
|------|-------------|-----|
| **A** | Pythia-160M | Mamba-130M |
| **B** | Pythia-410M | Mamba-370M |

## Phases

| Phase | Features | Description |
|-------|----------|-------------|
| **A** | PM + EM + PCM | All memory systems active (default) |
| **B** | A + lifelong mode | PM/EM persist across doc boundaries |

## Key Settings

- **Always use `-u` flag** with python to disable output buffering
- **`--compile`**: Enables `torch.compile(mode="default")` -- required for reasonable throughput on CUDA
- **`--no-compile`**: Disable torch.compile (useful for debugging or BS sweeps)
- **`--bs N`**: Batch size (max ~20 for Tier A at seq_len=512 on RTX 4090)

## Tokens Per Step

One step = BS * T tokens where T = K_segments * N.
Default: BS * 2 * 512 = BS * 1024 tokens per step.

## CLI Overrides

```bash
--tier a               # Select model tier (a or b)
--phase A              # Training phase (A or B)
--save-interval 5000   # Less frequent checkpoints
--plot-interval 500    # More frequent plot regeneration
--no-plots             # Disable all plot generation
--steps 5000           # Override step count
--bs 20                # Override batch size
--compile              # Enable torch.compile (recommended for CUDA)
--no-compile           # Disable torch.compile
--pcm / --no-pcm       # Enable/disable Predictive Coding Module
```

## Local Data Pipeline

Download The Pile locally before training:

```bash
python scripts/prepare_data.py --tokens 12B --seed 42

# Verify
cat data/pile/manifest.json
```

Training data: The Pile (deduplicated), downloaded locally to `data/pile/`.

## Monitoring

- **tqdm progress bar**: Shows in tmux with loss, ppl, tok/s, lr, ETA
- **Log lines**: Every 50 steps (LOG_INTERVAL)
- **Validation**: Every 2000 steps (VAL_INTERVAL)
- **Text samples**: Every 2000 steps (TEXT_SAMPLE_INTERVAL)
- **Training curve plots**: Every 2000 steps (PLOT_INTERVAL)
- **Checkpoints**: Every 10000 steps (SAVE_INTERVAL)

## Outputs

All outputs go to `outputs/<run_name>/<run_id>/`:
- `checkpoints/` -- model checkpoints (.pt files)
- `metrics.jsonl` -- all training/val metrics
- `text_samples_step{N}.png` -- text prediction visualizations
- `through_phase_{X}_plot_*.png` -- training curve plots
- `run_meta.json` -- run configuration

## Resuming

```bash
python -u -m src.train --phase B --tier a --bs 20 --compile --resume outputs/.../checkpoints/latest.pt
```

## Important Notes

- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
- Monitor GPU memory in later phases -- PM/EM add VRAM overhead
- If OOM, reduce BS or enable gradient checkpointing (`gradient_checkpointing: true` in config)
