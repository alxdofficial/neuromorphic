# docs/ index

The active line is an **always-on implicit memory layer** for a frozen LM: an encoder compresses a
1024-token context into M=32 memory tokens (32:1), trained on a mixed 4-task objective
(mae / babi / continuation / condrecon_bio) on SmolLM2-135M. The source of truth for each model is
its code under `src/memory/models/<name>/` (no per-model design doc).

## Active cohort (see `src/memory/model.py` VARIANTS)
- **Published baselines:** `icae`, `ccm`, `autocompressor`, `beacon` — learnable tokens through the
  frozen LM (± LoRA), read their hiddens; prepend M memory tokens.
- **biomem** — gated fast-Hebbian cortical-column grid (memory lives in fast synaptic state).
- **slotgraph** — ICAE write + a FIXED node/edge partition + per-edge hard (straight-through) endpoint
  prediction (masked edges→nodes), read out by a multi-hop residual message-passing GNN over the
  predicted graph. The de-facto graph arm (supersedes the retired relational-parser `graph` model).
  **Current cohort leader** (bAbI EM 37.5% vs icae 26.2%; highest binding/SHUF−REAL on all 4 tasks).
- **vqicae** — ICAE with VQ-VAE-discretized slots (a large EMA codebook); the discreteness probe.
- **vanilla_llama / vanilla_full_context** — loss floor (no memory) / ceiling (full context).

## Docs
- **`slotgraph_diagnostics.md`** — the current graph arm: capacity (params + read-floats), cohort
  results (bAbI EM + binding gate), and structure diagnostics (gradient-flow, UMAP + histograms) showing
  the message-passing read + fixed partition make the graph *bind* (with caveats: hub-collapse, low rank,
  attribution-ablation pending). Includes the before→after vs the old inert prepend-read version.
- **`mamba_two_lenses_memory.md`** — research note: Mamba/linear-attention lenses on compress-and-recall.

## Harness / diagnostics
- `scripts/train/train.py` — trainer (`--task mixed`).
- `scripts/diagnostics/mixed_band_gate_eval.py` — REAL/SHUF/OFF band + binding gate over the cohort.
- `scripts/diagnostics/debug_sweep_new_models.py`, `smoke_slotgraph.py` — slotgraph/vqicae checks.
- `scripts/diagnostics/mixed_dashboard.py` — per-task training/val dashboard from the run JSONLs.
