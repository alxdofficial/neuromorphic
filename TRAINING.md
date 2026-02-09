# Training Instructions for Neuromorphic LM

## Quick Start

```bash
# Full training (all 5 phases, tier B, optimized BS)
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --phases A,B,C,D,E --tier b --bs 48 2>&1 | tee outputs/train_full_\$(date +%Y%m%d_%H%M%S).log"

# Attach to monitor
tmux attach -t neuromorphic-train

# Detach: Ctrl+B, then D
```

## Key Settings

- **Always use `-u` flag** with python to disable output buffering (otherwise tee/log files stay empty)
- **Batch size**: BS=48 is optimal for RTX 4090 (24GB). Uses ~15GB VRAM, leaves headroom for PM/EM in later phases.
  - BS=64 works for Phase A (19.4GB) but may OOM in later phases with PM/EM enabled
  - BS=32 is a safe fallback (~10.7GB)
- **Tier B** (~90.8M params) is the current training tier

## GPU Memory by Batch Size (Tier B, Phase A)

| BS | Peak VRAM | tok/s | Headroom |
|----|-----------|-------|----------|
| 16 | 6.3GB     | 373   | 18GB     |
| 24 | 8.6GB     | 558   | 16GB     |
| 32 | 10.7GB    | 742   | 14GB     |
| 48 | 15.1GB    | 1,107 | 9.5GB    |
| 64 | 19.4GB    | 1,423 | 5GB      |

## Phase Plan (Default Steps)

| Phase | Steps | Tokens (BS=48) | Features |
|-------|-------|-----------------|----------|
| A     | 10K   | ~125M           | WM only (TinyStories) |
| B     | 300K  | ~3.7B           | WM + PM (FineWeb-Edu + DCLM) |
| C     | 300K  | ~3.7B           | WM + PM + EM |
| D     | 300K  | ~3.7B           | WM + PM + EM + RL |
| E     | 120K  | ~1.5B           | Lifelong learning |

Total: ~1.03M steps, ~12.7B tokens at BS=48

## Tokens Per Step

One step = BS * T tokens (e.g., 48 * 256 = 12,288 tokens).
Each step takes ~13s because TBPTT processes tokens sequentially (256 forward passes per step).

## Monitoring

- **tqdm progress bar**: Shows in tmux with loss, ppl, tok/s, lr, ETA
- **Log lines**: Every 50 steps (LOG_INTERVAL)
- **Validation**: Every 200 steps (VAL_INTERVAL)
- **Text samples**: Every 200 steps (TEXT_SAMPLE_INTERVAL) — saved as PNG
- **Training curve plots**: Every 1000 steps (PLOT_INTERVAL) — regenerates all 6 PNGs
- **Checkpoints**: Every 2500 steps (SAVE_INTERVAL)
- **Ablation eval**: Every 1000 steps (ABLATE_INTERVAL)

## Dataset Caching

Datasets are cached in `~/.cache/huggingface/datasets/` (~2.4GB).
Restarts do NOT re-download. Phase A (TinyStories) is fully local.
Later phases use streaming with local caching.

## CLI Overrides

```bash
--save-interval 5000   # Less frequent checkpoints
--plot-interval 500    # More frequent plot regeneration
--no-plots             # Disable all plot generation
--steps 5000           # Override step count (per phase)
--bs 32                # Override batch size
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
Checkpoints are ~170MB each (tier A) or larger for tier B.

## Resuming

```bash
python -u -m src.train --phase B --tier b --bs 48 --resume outputs/.../checkpoints/latest.pt
```

## Important Notes

- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
- Monitor GPU memory in later phases — PM/EM add VRAM overhead
- If OOM in later phases, reduce BS (try 32)
