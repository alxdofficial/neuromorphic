# Memory-Compression LM

A research project on always-on implicit memory for a frozen LM. An encoder compresses a 2048-token context into M=96 memory tokens (21:1 ratio), trained on a mixed 5-task objective (`reconstruct` / `babi` / `doc_qa` / `continuation` / `fact_recall`) with a frozen SmolLM2-135M backbone (d=576). The goal is to evaluate whether structured or biologically-inspired encoders outperform flat published compressors at the same parameter count.

## Layout

```
src/memory/          — core package: model.py (VARIANTS registry), models/<name>/
                       for each encoder, data loaders, trainer utilities
scripts/train/       — train.py: single harness for every variant + task
scripts/diagnostics/ — cohort evaluation, mixed band+gate eval, dashboards,
                       per-arm probes
docs/                — cohort results, model attribution reports, design notes
```

## Train

```bash
.venv/bin/python scripts/train/train.py --task mixed
```

Default objective is **behavioral-KL** (context distillation) on the frozen SmolLM2-135M backbone.
Active cohort: **`icae` · `autocompressor` · `titans` · `gisting` · `memoryllm` · `slotgraph`** (trainable,
param-matched ~7M / M=96) + **`h2o`** (training-free KV-eviction reference) + **`vanilla_llama` /
`vanilla_full_context`** (loss floor / ceiling). Retired 2026-07: beacon, ccm, vqicae, biomem, and the
exploratory slotgraph 1–4 (superseded by the single canonical `slotgraph`).

## Results

Results live in `docs/` (see `docs/README.md` for the index).
