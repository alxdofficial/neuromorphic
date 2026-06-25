# Memory-Compression LM

A research project on always-on implicit memory for a frozen LM. An encoder compresses a 1024-token context into M=32 memory tokens (32:1 ratio), trained on a mixed 4-task objective (mae / babi / continuation / condrecon_bio) with a frozen SmolLM2-135M backbone (d=576). The goal is to evaluate whether structured or biologically-inspired encoders outperform flat published compressors at the same parameter count.

## Layout

```
src/memory/          — core package: model.py (VARIANTS registry), models/<name>/
                       for each encoder, data loaders, trainer utilities
scripts/train/       — train.py: single harness for every variant + task
scripts/diagnostics/ — cohort evaluation, slotgraph attribution/metrics/probes,
                       mixed band+gate eval, dashboards
docs/                — cohort results, model attribution reports, design notes
```

## Train

```bash
.venv/bin/python scripts/train/train.py --task mixed
```

Default variants: `slotgraph_baseline biomem_baseline icae_baseline ccm_baseline autocompressor_baseline beacon_baseline`.

## Results

Results live in `docs/`:
- **`cohort_results.md`** — head-to-head table across all variants (REAL loss + babi EM).
- **`slotgraph_attribution.md`** — 2×2 attribution study isolating message-passing vs id-tags.
- **`slotgraph_metrics.md`** — standing instrument panel for slotgraph structure canaries.
