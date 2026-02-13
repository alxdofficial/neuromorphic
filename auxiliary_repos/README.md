# Auxiliary Repos — Baseline Models for Comparison

Baseline models for benchmarking the Neuromorphic LM against established architectures.

## Models

### Pretrained checkpoints (for published-number reference)

| Model | Params | Type | HuggingFace ID | Training data |
|-------|--------|------|----------------|---------------|
| **Pythia-160M** | 162M | Transformer | `EleutherAI/pythia-160m` | Pile (~300B tokens) |
| **Mamba-130M** | 129M | SSM | `state-spaces/mamba-130m-hf` | Pile (~300B tokens) |

### From-scratch baselines (for apples-to-apples comparison)

Trained on the same data (FineWeb-Edu 60% + DCLM 40%) with the same tokenizer
(TinyLlama, 32K vocab) and same token budget as our neuromorphic model.

| Model | Params (32K vocab) | Architecture | Notes |
|-------|-------------------|-------------|-------|
| **Pythia-style** | 134.2M | GPT-NeoX (D=768, L=12, H=12) | Standard Pythia arch, smaller vocab |
| **Mamba-style** | 115.1M | Mamba (D=768, L=24, S=16) | Standard Mamba arch, smaller vocab |
| **Neuromorphic LM** | 90.8M | Recurrent + PM/EM/WM (Tier B) | Our model |

Both baselines are larger than our model, which makes the comparison generous toward
baselines. If our model matches or beats them, the architecture argument is stronger.

## Setup

```bash
# Dependencies (already installed if you ran the neuromorphic LM setup)
pip install lm-eval>=0.4.0 accelerate

# Download pretrained weights (for reference benchmarks)
python auxiliary_repos/baselines/eval_scripts/download_models.py
```

## Evaluation (pretrained checkpoints)

```bash
# Smoke test — verify models load and generate
python auxiliary_repos/baselines/eval_scripts/smoke_test.py

# Perplexity evaluation (WikiText-2, WikiText-103, PG19 subset)
python auxiliary_repos/baselines/eval_scripts/eval_perplexity.py --model pythia-160m
python auxiliary_repos/baselines/eval_scripts/eval_perplexity.py --model mamba-130m

# Zero-shot benchmarks via lm-eval-harness
python auxiliary_repos/baselines/eval_scripts/run_benchmarks.py --model pythia-160m
python auxiliary_repos/baselines/eval_scripts/run_benchmarks.py --model mamba-130m

# Run all evaluations
python auxiliary_repos/baselines/eval_scripts/run_all.py
```

## Training from scratch (fair comparison)

```bash
# Train Pythia-style baseline (300K steps, same as Phase B)
python auxiliary_repos/baselines/eval_scripts/train_baseline.py \
    --model pythia-160m --steps 300000 --bs 48

# Train Mamba-style baseline (300K steps, same as Phase B)
python auxiliary_repos/baselines/eval_scripts/train_baseline.py \
    --model mamba-130m --steps 300000 --bs 48

# Quick test run (verify everything works)
python auxiliary_repos/baselines/eval_scripts/train_baseline.py \
    --model pythia-160m --steps 100 --bs 8
```

Training uses the same data pipeline, tokenizer, and optimizer schedule as our
neuromorphic model. Outputs go to `outputs/baseline_<model>/`.

## Results

Results are saved to `auxiliary_repos/baselines/results/` as JSON files.
