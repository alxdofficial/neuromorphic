# Training Instructions for Neuromorphic LM

## Quick Start

```bash
# Tier A on RTX 4090 (all 3 phases, optimized BS)
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --preset phase_a_to_c --tier a --bs 32 --compile 2>&1 | tee outputs/train_full_\$(date +%Y%m%d_%H%M%S).log"

# Tier 1B on A100 80GB (cloud training)
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --preset phase_a_to_c --tier 1b --bs 8 --compile 2>&1 | tee outputs/train_1b_\$(date +%Y%m%d_%H%M%S).log"

# Attach to monitor
tmux attach -t neuromorphic-train

# Detach: Ctrl+B, then D
```

## Model Tiers

| Tier | Params | D | L | B | D_h | Target GPU | Recommended BS |
|------|--------|---|---|---|-----|------------|---------------|
| **A** | ~87M | 768 | 8 | 2 | 384 | RTX 4090 | 32 |
| **B** | ~408M | 2048 | 10 | 4 | 512 | RTX 4090 / A100 | 8-16 |
| **C** | ~980M | 4096 | 16 | 8 | 512 | A100 80GB | 8 |

**Baselines per tier:**

| Tier | Transformer | SSM | Recurrent |
|------|-------------|-----|-----------|
| **A** | gpt2-small, pythia-160m | mamba-130m | rwkv7-168m |
| **B** | gpt2-medium, pythia-410m | mamba-370m | rwkv7-421m |
| **C** | pythia-1b, tinyllama-1.1b | mamba-1.4b | rwkv7-1.5b |

Select tier with `--tier a` / `--tier b` / `--tier c`.

## Key Settings

- **Always use `-u` flag** with python to disable output buffering (otherwise tee/log files stay empty)
- **Tier A** (~85M params): Primary development tier on RTX 4090
- **Tier 1B** (~1.07B params): Target tier for conversational quality (cloud GPU)
- **`--compile`**: Enables `torch.compile(mode="default")` — required for reasonable throughput on CUDA

## Performance by Tier (compiled, Phase B)

| Tier | Params | tok/s | ms/step | Peak VRAM | 1.5B tokens | GPU |
|------|--------|-------|---------|-----------|-------------|-----|
| **A Wide** (BS=32) | 85M | ~24,000 | ~340 ms | ~2.4 GB | ~17h | RTX 4090 |
| **1B** (BS=8) | 1,070M | ~3-5K (est.) | TBD | ~21 GB est. | TBD | A100 80GB |

### Baseline Comparison (1.5B tokens, RTX 4090)

| Model | Params | tok/s | 1.5B train time |
|-------|--------|-------|-----------------|
| Pythia-160M | 134M | ~116K | ~2.7h |
| Mamba-130M | 115M | ~52K | ~3.3h |
| **Neuromorphic A Wide** | **85M** | **~24K** | **~17h** |

The neuromorphic model is slower due to three memory systems (PM/EM/WM), sequential span processing, and span-boundary operations. This is the cost of persistent adaptive memory.

## Phase Plan (Default Steps)

| Phase | Steps | Tokens (BS=32) | Features |
|-------|-------|-----------------|----------|
| A     | 5K    | ~40.9M          | WM + PM (TinyStories) |
| B     | 5K    | ~40.9M          | WM + PM + EM (FineWeb-Edu + DCLM) |
| C     | 2.5K  | ~20.5M          | WM + PM + EM + lifelong (PM/EM persist across docs) |

Total (preset `phase_a_to_c`): ~12.5K steps, ~102M tokens at BS=32

Neuromodulators (PM and EM) are trained by main-loss gradient in all phases — no separate RL optimizer.

## Tokens Per Step

One step = BS * T tokens (e.g., 32 * 256 = 8,192 tokens).
Each step processes T/P = 4 forward_span calls (P=64 tokens each).

## Local Data Pipeline

For reliable training without network dependencies, download data locally first:

```bash
# Download ~2B tokens of FineWeb-Edu + DCLM as local parquet files
python scripts/prepare_data.py --tokens 2B --seed 42

# Verify
cat data/phase_B/manifest.json
```

This creates `data/phase_B/` with parquet files (~4-8 GB). The training pipeline auto-detects local files and uses them instead of streaming.

## Monitoring

- **tqdm progress bar**: Shows in tmux with loss, ppl, tok/s, lr, ETA
- **Log lines**: Every 50 steps (LOG_INTERVAL)
- **Validation**: Every 200 steps (VAL_INTERVAL)
- **Text samples**: Every 200 steps (TEXT_SAMPLE_INTERVAL) — saved as PNG
- **Training curve plots**: Every 500 steps (PLOT_INTERVAL) — regenerates all 6 PNGs
- **Checkpoints**: Every 2500 steps (SAVE_INTERVAL)
- **Ablation eval**: Every 1000 steps (ABLATE_INTERVAL)

## CLI Overrides

```bash
--tier a               # Select model tier (a, b, 1b, c)
--save-interval 5000   # Less frequent checkpoints
--plot-interval 500    # More frequent plot regeneration
--no-plots             # Disable all plot generation
--steps 5000           # Override step count (per phase)
--bs 32                # Override batch size
--compile              # Enable torch.compile (recommended for CUDA)
--no-compile           # Disable torch.compile (default)
```

## Outputs

All outputs go to `outputs/<run_name>/<run_id>/`:
- `checkpoints/` — model checkpoints (.pt files)
- `metrics.jsonl` — all training/val/ablation metrics
- `text_samples_step{N}.png` — text prediction visualizations
- `through_phase_{X}_plot_*.png` — training curve plots
- `run_meta.json` — run configuration
- `run_state.json` — live run state (for resume)

## Cleanup

Delete short/aborted runs from `outputs/` to save disk.
Checkpoints are ~170MB each (tier A) or ~4GB (tier 1B).

## Resuming

```bash
python -u -m src.train --phase B --tier a --bs 32 --compile --resume outputs/.../checkpoints/latest.pt
```

## Important Notes

- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
- Monitor GPU memory in later phases — PM/EM add VRAM overhead
- If OOM in later phases, reduce BS or enable gradient checkpointing
