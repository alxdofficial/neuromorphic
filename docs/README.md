# docs/ index

The active line is an **always-on implicit memory layer** for a FROZEN LM: a small trainable
encoder compresses a 2048-token context into M=96 memory (read by a frozen SmolLM2-135M, d=576),
trained on a mixed 5-task objective (`reconstruct` / `babi` / `doc_qa` / `continuation` /
`fact_recall`) with **behavioral-KL context distillation** (`src/memory/training/objectives.py`).
The current focus is the **baseline comparison** — how our structured memory compares to the
published compressor / memory techniques at matched capacity. The source of truth for each model
is its code under `src/memory/models/<name>/`; papers & dataset links are in **`REFERENCES.md`**.

## Baseline cohort (see `src/memory/model.py` VARIANTS)

Fairness policy: **match the persistent memory-slot count M and the trainable param count (~7M);
each baseline uses the READ mechanism from its OWN paper** (not a forced-uniform read), and any
departure from a clean fixed-footprint feed-forward memory is disclosed as an asterisk in results.

- **Prepend-read compressors:** `icae`, `autocompressor` (faithful summary-accumulation), `titans`
  (deep-MLP memory + test-time autograd write, MAC prepend), `vqicae` (VQ-discretized ICAE),
  `ccm` (this port reads normalized COMP-token hidden states via prepend).
- **Per-layer-KV compressors** (native read via the shared prefix-cache path, `decoder.build_prefix_cache`
  + `model._prefix_kv_forward`): `beacon`, `gisting`, `memoryllm` (per-layer pool + random-drop).
- **`h2o`** — training-free KV-cache eviction; an eval-only efficiency/KV-ratio reference (0 trainable params).
- **Our arms:** `slotgraph` / `slotgraph2` / `slotgraph3` (graph-over-learned-vocab) and `biomem`
  (fast-Hebbian columns); **FurlGraph** is the next design (`furlgraph_design.md`, deferred).
- **`vanilla_llama` / `vanilla_full_context`** — loss floor (no memory) / ceiling (full context).

The **active locked set** (2026-07-09) is icae · autocompressor · gisting · titans · memoryllm ·
beacon · h2o (+ our arms later); `ccm` / `vqicae` remain in code but are dropped from the active
comparison. Titans requires `--no-grad-ckpt-stream` (its inner `create_graph` conflicts with the
outer streaming checkpoint). All comparisons must be **same-backbone, matched-params**.

## Docs (current)
- **`REFERENCES.md`** — authoritative paper/dataset links for every baseline & data source (never re-search).
- **`DATA_PHASES_PLAN.md`** — the canonical phase plan: Phase-1 full-corpus training + Phase-2 test-eval
  (headline table, the 4 comparison axes, run matrix, invariants). *Merged `PHASE_PLAN.md` into this.*
- **`SCRUTINY_PHASE_DATA.md`** — Phase-0 (architecture-scrutiny) datasets / generation / objectives reference.
- **`DATA_TASK_GUIDE.md`** — what each task does, with an example per task × datasource + config knobs.
- **`data_arch_plan.md`** — the 4-layer data model (Source × Task × EpisodeSpec × Objective); where data code lives.
- **`OBJECTIVES.md`** — the binding objective ladder (MAE-CE → behavioral-KL → provenance-InfoNCE → bypass-gap → GRPO), with math + citations. Why binding is an *objective* problem.
- **`graph_thesis.md`** — why a graph memory (the two lenses), and what the literature says about making latent topology load-bearing instead of collapsing. The standing rationale.
- **Graph-memory arm designs:** `furlgraph_design.md` (input-grounded chain-merge), `slotgraph4_design.md` (free-invent slots + fixed k-regular small-world edge states, propose→commit gated write — differentiable, buildable now), `graph_generative_memory.md` (the "spider web" — score-function/GRPO-era). All deferred behind the baseline sweep.
- **`mamba_two_lenses_memory.md`** — research note: Mamba/linear-attention lenses on compress-and-recall.
- **`history/`** — archived records of the superseded slotgraph/biomem/treemem line and completed reorg
  plans (`cohort_results`, `slotgraph_*`, `biomem_chunkwise_plan`, `{data,harness}_reorg_plan`, …).
  Point-in-time snapshots at their own fixed config; kept for provenance, not current.

## Harness / diagnostics
- `scripts/train/train.py` — trainer (`--task mixed`, `--objective-mode behavioral_kl`); `--variants` selects arms.
- `scripts/diagnostics/mixed/mixed_band_gate_eval.py` — REAL/SHUF/OFF binding gate over the cohort.
- `scripts/diagnostics/mixed/mixed_dashboard.py` — per-task training/val dashboard from the run JSONLs.
- See `scripts/README.md` / `HARNESS.md` for the full diagnostics layout (subject subdirs
  `slotgraph3/`, `slotgraph/`, `biomem/`, `objective/`, `mixed/`, `cohort/`).
