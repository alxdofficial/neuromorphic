# Memory-Compression LM

A research project on always-on implicit memory for a frozen LM. An encoder compresses a 2048-token context into M=96 memory tokens (21:1 ratio), trained on a mixed 5-task objective (`reconstruct` / `babi` / `doc_qa` / `continuation` / `fact_recall`) with a frozen SmolLM2-135M backbone (d=576). The goal is to evaluate whether structured or biologically-inspired encoders outperform flat published compressors at the same parameter count. (The 1024-ctx/M=32/4-task numbers still quoted in some `docs/` results files, e.g. `docs/history/cohort_results.md`, are that earlier cohort's own fixed config — not the current default; see `docs/README.md`.)

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

Results live in `docs/`. The earlier slotgraph-cohort snapshots are archived under
**`docs/history/`** (see `docs/README.md` for the index):
- **`docs/history/cohort_results.md`** — old slotgraph cohort head-to-head (REAL loss + babi EM).
- **`docs/history/slotgraph_attribution.md`** / **`docs/history/slotgraph_metrics.md`** — frozen slotgraph structure studies.
