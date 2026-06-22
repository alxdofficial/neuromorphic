# docs/ index

The active line is an **always-on implicit memory layer** for a frozen LM: an encoder compresses a
1024-token context into M=32 memory tokens (32:1), trained on a mixed 4-task objective
(mae / babi / continuation / condrecon_bio) on SmolLM2-135M. The source of truth for each model is
its code under `src/memory/models/<name>/` (no per-model design doc).

## Active cohort (see `src/memory/model.py` VARIANTS)
- **Published baselines:** `icae`, `ccm`, `autocompressor`, `beacon` — learnable tokens through the
  frozen LM (± LoRA), read their hiddens; prepend M memory tokens.
- **biomem** — gated fast-Hebbian cortical-column grid (memory lives in fast synaptic state).
- **slotgraph** — ICAE write + a per-LM-layer hard (straight-through) head predicting node/edge role
  and edge endpoints, concretized as TokenGT role/identity embeddings; the de-facto graph arm
  (supersedes the retired relational-parser `graph` model).
- **vqicae** — ICAE with VQ-VAE-discretized slots (a large EMA codebook); the discreteness probe.
- **vanilla_llama / vanilla_full_context** — loss floor (no memory) / ceiling (full context).

## Docs
- **`slotgraph_diagnostics.md`** — the current graph arm: cohort results, structure diagnostics
  (UMAP + histograms), and *why* the emergent topology doesn't form (the membership-only wall, with
  the trained structure heads sitting at init). Supersedes the old graph retrospective.
- **`mamba_two_lenses_memory.md`** — research note: Mamba/linear-attention lenses on compress-and-recall.

## Harness / diagnostics
- `scripts/train/train.py` — trainer (`--task mixed`).
- `scripts/diagnostics/mixed_band_gate_eval.py` — REAL/SHUF/OFF band + binding gate over the cohort.
- `scripts/diagnostics/debug_sweep_new_models.py`, `smoke_slotgraph.py` — slotgraph/vqicae checks.
- `scripts/diagnostics/mixed_dashboard.py` — per-task training/val dashboard from the run JSONLs.
