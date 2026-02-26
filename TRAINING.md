# Training Instructions for Neuromorphic LM

## Quick Start

```bash
# Tier A on RTX 4090
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --tier a --phase A --bs 32 --compile 2>&1 | tee outputs/train_a_\$(date +%Y%m%d_%H%M%S).log"

# Tier B on RTX 4090
tmux new-session -d -s neuromorphic-train -c /home/alex/code/neuromorphic \
  "python -u -m src.train --tier b --phase A --bs 24 --compile 2>&1 | tee outputs/train_b_\$(date +%Y%m%d_%H%M%S).log"

# Attach to monitor
tmux attach -t neuromorphic-train

# Detach: Ctrl+B, then D
```

## Model Tiers

| Tier | Params | D | L | B | D_h | Token Budget | Target GPU | Recommended BS |
|------|--------|---|---|---|-----|-------------|------------|---------------|
| **A** | ~87M | 768 | 8 | 2 | 384 | 1.5B | RTX 4090 | 32 |
| **B** | ~335M | 2048 | 10 | 4 | 512 | 10B | RTX 4090 | 24 |
| **C** | ~980M | 4096 | 16 | 8 | 512 | TBD | A100 80GB | TBD |

Token budgets follow Chinchilla scaling (~20 tokens/param for A, ~30x for B).

**Baselines per tier:**

| Tier | Transformer | SSM | Recurrent |
|------|-------------|-----|-----------|
| **A** | gpt2-small, pythia-160m | mamba-130m | rwkv7-168m |
| **B** | gpt2-medium, pythia-410m | mamba-370m | rwkv7-421m |
| **C** | pythia-1b, tinyllama-1.1b | mamba-1.4b | rwkv7-1.5b |

Select tier with `--tier a` / `--tier b` / `--tier c`.

### Baseline Comparison Strategy

**Tier A**: Train all baselines from scratch on The Pile (1.5B tokens). Fair apples-to-apples.

**Tier B**: Two-stage comparison:
1. **Primary**: Compare against published fully-trained checkpoints (300B tokens)
   — Pythia-410M, Mamba-370M, GPT-2 Medium, RWKV-7 421M. If our model at 10B
   tokens outperforms these at 300B, no further work needed.
2. **If needed**: Fair 10B-token comparison. Pythia-410M has an intermediate
   checkpoint at exactly 10.5B tokens (`revision="step5000"` on HuggingFace).
   GPT-2 Medium and Mamba-370M would need to be trained from scratch to 10B.

**Tier C**: Use published fully-trained checkpoints only (Pythia-1B at 300B,
Mamba-1.4B at 300B, RWKV-7 1.5B at 332B, TinyLlama at 3T). Too expensive to
retrain on one GPU.

**Pythia intermediate checkpoints** (useful for learning curves):
- Published every 1000 steps, each step = 2,097,152 tokens
- `EleutherAI/pythia-410m-deduped` with `revision="step{N}"`
- Key checkpoints: step1000=2.1B, step3000=6.3B, **step5000=10.5B**, step10000=21B

## Key Settings

- **Always use `-u` flag** with python to disable output buffering (otherwise tee/log files stay empty)
- **Tier A** (~85M params): Primary development tier on RTX 4090
- **Tier 1B** (~1.07B params): Target tier for conversational quality (cloud GPU)
- **`--compile`**: Enables `torch.compile(mode="default")` — required for reasonable throughput on CUDA

## Performance (compiled, Phase A, RTX 4090)

| Tier | Params | BS | tok/s | ms/step | VRAM | Train time |
|------|--------|----|-------|---------|------|------------|
| **A** | 87M | 32 | ~24K | ~340ms | ~2.4 GB | ~17h (1.5B) |
| **B** | 335M | 24 | ~9.3K | ~662ms | 23.4 GB | ~45h (10B) |

### Tier A Baseline Comparison (1.5B tokens, RTX 4090)

| Model | Params | BS | tok/s | Train time |
|-------|--------|----|-------|------------|
| GPT-2 Small | 111M | 96 | ~135K | ~3.1h |
| Pythia-160M | 134M | 96 | ~116K | ~2.7h |
| RWKV-7 168M | 140M | 16 | ~63K | ~6.6h |
| Mamba-130M | 115M | 64 | ~52K | ~3.3h |
| **Neuromorphic A** | **87M** | **32** | **~24K** | **~17h** |

The neuromorphic model is slower due to three memory systems (PM/EM/WM), sequential span processing, and span-boundary operations. This is the cost of persistent adaptive memory.

## Phases

| Phase | Features | Description |
|-------|----------|-------------|
| **A** | WM + PM + EM + PCM | All memory systems active (default) |
| **B** | A + lifelong mode | PM/EM persist across doc boundaries |

Phase A is the standard training phase. Phase B enables lifelong learning where
procedural and episodic memory survive document resets.

Training data: The Pile (deduplicated), downloaded locally to `data/pile/`.

Neuromodulators (PM and EM) are trained by main-loss gradient — no separate RL optimizer.

## Tokens Per Step

One step = BS * T tokens (e.g., 24 * 256 = 6,144 tokens for Tier B).
Each step processes T/P = 8 forward_span calls (P=32 tokens each).

## Local Data Pipeline

Download The Pile locally before training:

```bash
# Download training + validation data from The Pile (deduplicated)
python scripts/prepare_data.py --tokens 12B --seed 42

# Verify
cat data/pile/manifest.json
```

This creates `data/pile/` with parquet files. The training pipeline loads from
local files — no streaming, no network dependency.

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
--tier a               # Select model tier (a, b, c)
--phase A              # Training phase (A or B)
--save-interval 5000   # Less frequent checkpoints
--plot-interval 500    # More frequent plot regeneration
--no-plots             # Disable all plot generation
--steps 5000           # Override step count
--bs 24                # Override batch size
--compile              # Enable torch.compile (recommended for CUDA)
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
