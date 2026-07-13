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

Provenance policy: **every arm is REIMPLEMENTED under this matched harness — none is a drop-in of an
official repo** (each official repo targets a 7B backbone it trains end-to-end under its own objective,
which would confound the very axis we isolate). One arm has its core WRITE mechanism **ported faithfully**
using the official code as a verified reference — `memoryllm` (co-attention compress) — but its pool is
a **trainable learned init that resets per episode** (vs upstream's non-trainable, initially-empty,
persistent pool), so the arm as a whole is **MemoryLLM-*adapted*, not faithful** (~half its memory is
learned init, not injected content; see `REFERENCES.md` + the encoder docstring). `h2o` was
**improved** to per-layer selection over original-context KV but stays **H2O-*inspired*** (query-blind,
offline, pre-RoPE / position-free — not H2O's online post-RoPE eviction). `icae` / `autocompressor` /
`gisting` are built from the paper (repos are 7B training programs / unlicensed / train-time-only), and
`titans` has **no official repo** — its test-time-gradient write is faithful but its read is a static
prepend, so it is **Titans-*inspired***, not MAC. `icae` streams via recurrent RMT recompression across
the 8 windows (single-window operator is faithful ICAE) → an **ICAE/RMT hybrid**. Repo + paper links and
the per-arm provenance tag are in **`REFERENCES.md`**.

- **Prepend-read compressors:** `icae` (single-window ICAE operator; recurrent-RMT across windows →
  ICAE/RMT hybrid), `autocompressor` (faithful summary-accumulation), `titans` (deep-MLP memory +
  test-time autograd write; static-prepend read → **Titans-inspired**, not MAC).
- **Per-layer-KV compressors** (native read via the shared prefix-cache path, `decoder.build_prefix_cache`
  + `model._prefix_kv_forward`): `gisting` (per-layer gist-KV; final-layer `q_proj` LoRA is structurally
  inert under KV-capture, ~120k params → effective ~6.80M trainable), `memoryllm` (per-layer pool +
  random-drop + co-attention compress — the write mechanism is ported faithfully, but the pool is a
  **trainable/per-episode learned init** (vs upstream's non-trainable, initially-empty, persistent pool),
  so the arm is **MemoryLLM-*adapted***, not faithful; MAE is streamed in 8 windows for this arm).
- **Our arm:** `slotgraph` — THE canonical graph memory (96 nodes / value-path plastic edge state /
  prepend+bidir read; see `docs/design/slotgraph_design.md`).
- **`h2o`** — training-free KV-cache eviction (eval-only, 0 trainable params); **H2O-*inspired* static
  per-layer heavy-hitter selection** over original-context (pre-RoPE / position-free) KV — improved from a
  global re-encode, but query-blind + offline, so NOT a faithful online-eviction port.
- **`vanilla_llama` / `vanilla_full_context`** — loss floor (no memory) / ceiling (full context). NB for
  analysis: make causal memory claims from each arm's own **OFF** control (memory-zeroed), not from %band
  vs these fresh controls — the fresh controls use *untrained* decoder-LoRA / mask-embed, so %band folds in
  decoder-adaptation effects absent from a memory-vs-no-memory comparison.

The **active cohort** (2026-07-11) is icae · autocompressor · titans · gisting · memoryllm · slotgraph
(trainable, ~7M) + h2o / vanilla×2 (eval-only). Retired + removed from the code: beacon, ccm, vqicae,
biomem, and the exploratory slotgraph 1–4 (→ the single canonical `slotgraph`). Titans auto-disables the
streaming activation-checkpoint per-arm (its inner `create_graph` conflicts) — one sweep command works for
all arms. All comparisons must be **same-backbone, matched-params**.

## Docs (current)

The docs are grouped into subfolders: **`design/`** (architecture & objectives), **`data/`** (what the
data is + the phase plan), **`baselines/`** (competitor menu), **`ops/`** (remote-training runbook). Two
cross-cutting indexes live at the docs root: this `README.md` and `REFERENCES.md`.

**Root**
- **`REFERENCES.md`** — authoritative paper/dataset links for every baseline & data source (never re-search).
- **`RESULTS.md`** — the **running Phase-0 results log**: band-gate (REAL/SHUF/OFF) scores, %band, per-run
  configs, and the interpreted observations. Append a run section per campaign; the metric caveats live at
  the top. Raw dumps stay in `outputs/memory/<run>_band_gate.json`.

**`data/`**
- **`data/DATA.md`** — THE authoritative "what runs now": sources, the 5 tasks (with worked examples), the shared
  packer, multi-horizon continuation, the per-task objective dispatch (MAE→CE / QA→KL / continuation-fallback),
  and the current sweep config. Consolidates the former SCRUTINY_PHASE_DATA + DATA_TASK_GUIDE + data_arch_plan.
- **`data/DATA_PHASES_PLAN.md`** — the FUTURE phase plan: Phase-1 full-corpus training + Phase-2 test-eval
  (headline table, the 4 comparison axes, run matrix, invariants). *Merged `PHASE_PLAN.md` into this.*

**`baselines/`**
- **`baselines/FROZEN_COMPETITORS.md`** — the Phase-2 competitor menu (2026-07-12 survey): which frozen/off-the-shelf
  SOTA memory systems to compare against (long-context / RAG / agent-memory / compression cousins), which
  actually ship weights, the LongMemEval anchor numbers, and the vendor-number red flags. Frames the
  "Phase-0 reimplement vs Phase-2 frozen-SOTA" split.

**`design/`**
- **`design/OBJECTIVES.md`** — the binding objective ladder (MAE-CE → behavioral-KL → provenance-InfoNCE → bypass-gap → GRPO), with math + citations. Why binding is an *objective* problem.
- **`design/graph_thesis.md`** — why a graph memory (the two lenses), and what the literature says about making latent topology load-bearing instead of collapsing. The standing rationale.
- **Graph-memory arm design:** **`design/slotgraph_design.md` — THE slotgraph** (the canonical arm:
  96 nodes / no edge tokens / unit relation vector + dynamic confidence per pair / value-path feedback
  operator / propose→commit / prepend+bidir read; built + stabilized 2026-07-11). Companion future designs,
  NOT current: `design/furlgraph_design.md` (input-grounded chain-merge), `design/graph_generative_memory.md` (the
  "spider web" — score-function/GRPO-era). The exploratory slotgraph 1–4 design docs were removed with
  their code; their lessons are folded into `design/slotgraph_design.md` §10 and `design/graph_thesis.md`.
- **`design/mamba_two_lenses_memory.md`** — research note: Mamba/linear-attention lenses on compress-and-recall.

## Harness / diagnostics
- `scripts/train/train.py` — trainer (`--task mixed`, `--objective-mode behavioral_kl`); `--variants` selects arms.
- `scripts/diagnostics/mixed/mixed_band_gate_eval.py` — REAL/SHUF/OFF binding gate over the cohort.
- `scripts/diagnostics/mixed/mixed_dashboard.py` — per-task training/val dashboard from the run JSONLs.
- See `scripts/README.md` / `HARNESS.md` for the full diagnostics layout (subject subdirs
  `objective/`, `mixed/`, `cohort/`).
- **`ops/runpod_workflow.md`** — the remote-training runbook (RunPod + R2): the flow, GPU selection, the
  CPU-launch-bound perf finding, and every gotcha already solved. Command list in `scripts/pod/README.md`.
