# Auxiliary Repos — Baseline Models for Comparison

Baseline models for benchmarking the Neuromorphic LM against established architectures.

## From-scratch baselines (apples-to-apples comparison)

Trained on the same data (The Pile, deduplicated) with the same tokenizer
(TinyLlama, 32K vocab) and same token budget (1.5B tokens) as our model.

```bash
# Train all three Tier A baselines
python auxiliary_repos/baselines/eval_scripts/train_baseline.py --model gpt2-small
python auxiliary_repos/baselines/eval_scripts/train_baseline.py --model pythia-160m
python auxiliary_repos/baselines/eval_scripts/train_baseline.py --model mamba-130m
```

| Model | Params | Architecture |
|-------|--------|-------------|
| **GPT-2 small** | ~124M | Transformer (GPT-2, abs pos emb) |
| **Pythia-160m** | ~160M | Transformer (GPT-NeoX, rotary emb) |
| **Mamba-130m** | ~130M | SSM (selective state space) |
| **Neuromorphic LM** | ~105M | Recurrent scan + dense-W memory graph |

All baselines are larger than our model, which makes the comparison generous
toward baselines. If our model matches or beats them, the architecture
argument is stronger.

## Prerequisites

```bash
# Data (run once — downloads 1.5B tokens from The Pile)
python scripts/prepare_data.py --tokens 1.5B

# Dependencies (already installed if you ran the neuromorphic LM setup)
pip install transformers datasets tokenizers numpy pyarrow
```

## Evaluation

```bash
# Perplexity on WikiText-2, WikiText-103, PG19
python auxiliary_repos/baselines/eval_scripts/eval_perplexity.py --model all

# Position-binned perplexity (memory benefit test)
python auxiliary_repos/baselines/eval_scripts/run_position_ppl.py
```

## Outputs

Training outputs go to `outputs/baseline_<model>/` with:
- `metrics.jsonl` — per-step training and validation metrics
- `checkpoints/` — periodic model checkpoints
- `config.json` — training configuration
