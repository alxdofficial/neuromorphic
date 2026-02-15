# Training Instructions for Neuromorphic LM

## Quick Start

```bash
# Full training (all 3 phases, tier B, optimized BS)
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --preset phase_a_to_c --bs 48 2>&1 | tee outputs/train_full_\$(date +%Y%m%d_%H%M%S).log"

# Attach to monitor
tmux attach -t neuromorphic-train

# Detach: Ctrl+B, then D
```

## Key Settings

- **Always use `-u` flag** with python to disable output buffering (otherwise tee/log files stay empty)
- **Batch size**: BS=48 is optimal for RTX 4090 (24GB). Uses ~15GB VRAM, leaves headroom for PM/EM in later phases.
  - BS=64 works for Phase A (19.4GB) but may OOM in later phases with PM/EM enabled
  - BS=32 is a safe fallback (~10.7GB)
- **Tier B** (~102.7M params) is the current training tier

## GPU Memory by Batch Size (Tier B, Phase A)

| BS | Peak VRAM | tok/s (eager) | tok/s (compiled) | Headroom |
|----|-----------|---------------|------------------|----------|
| 16 | 6.3GB     | ~1,800        | ~5,000           | 18GB     |
| 24 | 8.6GB     | ~2,700        | ~7,500           | 16GB     |
| 32 | 10.7GB    | ~3,700        | ~10,300          | 14GB     |
| 48 | 15.1GB    | ~5,500        | ~15,000          | 9.5GB    |
| 64 | 19.4GB    | ~7,000        | ~19,000          | 5GB      |

Enable compiled mode with `--compile` for ~2.8× speedup on CUDA (see below).

## Phase Plan (Default Steps)

| Phase | Steps | Tokens (BS=48) | Features |
|-------|-------|-----------------|----------|
| A     | 5K    | ~61.4M          | WM + PM (TinyStories) |
| B     | 5K    | ~61.4M          | WM + PM + EM (FineWeb-Edu + DCLM) |
| C     | 2.5K  | ~30.7M          | WM + PM + EM + lifelong (PM/EM persist across docs) |

Total (preset `phase_a_to_c`): ~12.5K steps, ~153.6M tokens at BS=48

Neuromodulators (PM and EM) are trained by main-loss gradient in all phases — no separate RL optimizer.

## Tokens Per Step

One step = BS * T tokens (e.g., 48 * 256 = 12,288 tokens).
Each step processes T/P = 4 forward_span calls (P=64 tokens each). With `--compile` on a 4090, each step takes ~1-2s at BS=32.

## Monitoring

- **tqdm progress bar**: Shows in tmux with loss, ppl, tok/s, lr, ETA
- **Log lines**: Every 50 steps (LOG_INTERVAL)
- **Validation**: Every 200 steps (VAL_INTERVAL)
- **Text samples**: Every 200 steps (TEXT_SAMPLE_INTERVAL) — saved as PNG
- **Training curve plots**: Every 500 steps (PLOT_INTERVAL) — regenerates all 6 PNGs
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
--compile              # Enable torch.compile (~2.8× speedup on CUDA)
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
