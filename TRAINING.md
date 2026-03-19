# Training Instructions for Neuromorphic LM (v7)

## Quick Start

```bash
# Tier A on RTX 4090
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --tier a --phase A --bs 16 --compile 2>&1 | tee outputs/train_a_$(date +%Y%m%d_%H%M%S).log"

# Attach to monitor
tmux attach -t neuromorphic-train

# Detach: Ctrl+B, then D
```

## Architecture (v7: Single Scan Stack)

Single scan stack per N-token segment with memory injection at layer L_mem:
- **layers[0..L_mem-1]**: Build representation (causal scan)
- **PCM**: Compute surprise; **W_seed_w**: Produce seed + write candidates
- **Memory injection**: PM read + EM trail read → additive: H = H + pm_read + em_read
- **layers[L_mem..L_total-1]**: Integrate with memory context
- **Segment end**: PM Hebbian commit, EM neuromodulated commit

NTP training only. Causal scans are inherently autoregressive.

## Model Tiers

| Tier | Params | D | D_embed | C | B | L_total | L_mem | d_inner | N | K_segments | M | Target GPU |
|------|--------|------|---------|---|---|---------|-------|---------|-----|------------|-----|------------|
| **A** | ~116M | 2048 | 768 | 16 | 4 | 10 | 5 | 1024 | 128 | 16 | 384 | RTX 4090 |
| **B** | ~250M | 3072 | 1024 | 16 | 6 | 20 | 10 | 1024 | 128 | 16 | 512 | RTX 4090 |
| **C** | ~844M | 4096 | 2048 | 16 | 8 | 28 | 14 | 2048 | 128 | 16 | 768 | A100 / Blackwell |

Select tier with `--tier a` / `--tier b` / `--tier c`.

**Baselines per tier:**

| Tier | Transformer | SSM |
|------|-------------|-----|
| **A** | Pythia-160M | Mamba-130M |
| **B** | Pythia-410M | Mamba-370M |
| **C** | Qwen3.5-0.8B | Mamba-1.4B |

## Phases

| Phase | Features | Description |
|-------|----------|-------------|
| **A** | PM + EM + PCM | All memory systems active (default) |
| **B** | A + lifelong mode | PM/EM persist across doc boundaries |

## Key Settings

- **Always use `-u` flag** with python to disable output buffering
- **`--compile`**: Enables `torch.compile(mode="default")` — required for reasonable throughput on CUDA
- **`--no-compile`**: Disable torch.compile (useful for debugging or BS sweeps)
- **`--bs N`**: Batch size (max ~16 for Tier A at K=16×N=128 on RTX 4090)

## Tokens Per Step

One step = BS × T tokens where T = K_segments × N.
Tier A default: BS × 16 × 128 = BS × 2048 tokens per step.

## CLI Overrides

```bash
--tier a               # Select model tier (a, b, or c)
--phase A              # Training phase (A or B)
--save-interval 5000   # Less frequent checkpoints
--plot-interval 500    # More frequent plot regeneration
--no-plots             # Disable all plot generation
--steps 5000           # Override step count
--bs 16                # Override batch size
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
- `checkpoints/` — model checkpoints (.pt files)
- `metrics.jsonl` — all training/val metrics
- `text_samples_step{N}.png` — text prediction visualizations
- `through_phase_{X}_plot_*.png` — training curve plots
- `run_meta.json` — run configuration

## Resuming

```bash
python -u -m src.train --phase B --tier a --bs 16 --compile --resume outputs/.../checkpoints/latest.pt
```

## Important Notes

- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
- Monitor GPU memory in later phases — PM/EM add VRAM overhead
- If OOM, reduce BS or enable gradient checkpointing (`gradient_checkpointing: true` in config)
