# Diagnostics

Scripts for inspecting trajectory-memory training runs and the resulting
concept manifold. All scripts in this directory are **standalone** — they
import from `src.trajectory_memory` but don't modify any persistent state.

## During-training scripts

These are safe to run while Wave 1 or Wave 2 is actively training:

- **`diagnose.py`** — single-command training health check. Synthesizes
  liveness, GPU usage, log state, training history, concept usage, and
  checkpoint freshness into a Green/Yellow/Red report. CPU-only; reads
  state_dict on host RAM via `map_location="cpu"`, refuses to read a
  ckpt written less than 30 s ago (mid-save guard).

  ```bash
  python scripts/diagnostics/diagnose.py            # auto-pick latest wave
  python scripts/diagnostics/diagnose.py --wave 1
  ```

- **`viz_manifold.py`** — Tier 1 manifold-structure visualization. Six
  panels in one PNG: UMAP(concept_ids), UMAP(concept_states), ring
  topology with usage overlay, drift histogram, usage Zipf curve, text
  summary. CPU-only with the same safety guards as `diagnose.py`.

  ```bash
  python scripts/diagnostics/viz_manifold.py        # auto-pick latest wave
  python scripts/diagnostics/viz_manifold.py --wave 1
  python scripts/diagnostics/viz_manifold.py --ckpt outputs/wave1/ckpt.best.pt
  ```

## Post-training / paused-training scripts

These require running model forward passes — they need the GPU. **Don't
run them while Wave 1/2 is training** (they will compete for VRAM).

- **`viz_concept_language.py`** *(Tier 2 — stub)* — concept ↔ language
  correspondence. For each of the 4096 concepts, finds the top-K tokens
  that co-occur in routed windows. Reveals whether concepts are semantic
  or just routing slots. Also produces per-source usage overlays and
  per-document trajectory traces.

- **`viz_needle_recall.py`** *(Tier 3 — stub)* — needle write→read
  overlap as a memory probe. The cleanest direct test that the memory
  manifold is doing useful work. Computes Jaccard overlap between
  `write_visited@needle_pos` and `read_visited@answer_pos` for each
  needle doc, stratified by needle→answer distance.

- **`viz_trajectory_decoder.py`** *(Tier 4 — stub)* — inversion probe.
  Trains a small `trajectory → text` decoder, then uses it to *speak*
  the manifold's contents. Produces a "concept dictionary" (the
  interpretability gold standard), interpolation panels, counterfactual
  probes, and needle-recall direct readouts.

## Tier roadmap

| Tier | Status         | Cost            | What it answers                                                                       |
| ---- | -------------- | --------------- | ------------------------------------------------------------------------------------- |
| 1    | implemented    | ~1 min (CPU)    | Do concepts have structure? Are they being used? Where do high-usage concepts sit?    |
| 2    | stub           | ~30 min (GPU)   | Are concepts *semantic*? Do different domains use different concepts?                 |
| 3    | stub           | ~10 min (GPU)   | Is memory actually carrying info past Llama's context window?                         |
| 4    | stub           | ~3-5 hr (GPU)   | What does each concept *mean*? Can we explore the manifold by synthesizing paths?     |

## Safety conventions

Every script in this directory should:
- Default to `map_location="cpu"` for `torch.load` (no CUDA touch).
- Use `matplotlib.use("Agg")` (no display required).
- Guard against reading a checkpoint that was written < 30 s ago.
- Print what file it's reading and what it's writing.
- Be runnable from the repo root (`python scripts/diagnostics/X.py`).

If a script needs GPU (Tier 2+), say so prominently in the docstring and
add a sanity check before allocating CUDA tensors (e.g., warn if Wave 1
training is detected as alive).
