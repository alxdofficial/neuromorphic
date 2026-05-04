# Training Strategy — Graph-Walker + Frozen Llama

**Status:** active. This is the canonical doc for *how* to train the
graph-walker integration. Updated 2026-05-04. Supersedes parts of
`docs/training_plan.md` (which was written for the v2 attention-neuromod
path and predates the graph-walker integration).

## What we're training

Frozen Llama-3.2-1B with the graph_walker memory module injected at
decoder layer 8. Walker has ~25M trainable params (24M walker +
1.1M `MemInjectLayer.W_in`/`W_out`/`scale`); Llama backbone is frozen.

**The memory pressure is created by capping segment length at T=256 tokens
and detaching state between segments.** Llama-3.2-1B's native context is
128k tokens, but we never feed more than 256 to it at once. The walker's
persistent state (`E_bias_flat`, walker_pos, walker_state, neuromod
snapshot) is the *only* thing carrying information across segment
boundaries. So **the small T is the lever that makes the walker matter**.

Verified throughput (`docs/bench_results.md`, 2026-05-03):
**8.8k tok/s @ BS=16 / T=256 / 14.6 GB peak VRAM** on a single RTX 4090.
69% of the hot-Llama bar (12.7k tok/s) — same iteration regime, ~1B-token
training run takes ~32 hours.

## The 4-wave training plan

The architecture is novel enough that doing all of pretraining + SFT
+ memory-task RL in one shot is risky. We sequence in 4 waves so each
stage adds one capability to a model that's known-good at the previous
stage.

### Wave 1 — Phase-1a natural-text bootstrap (FineWeb-edu)

- **Goal:** walker representations grounded in real natural text.
  Walker learns to be a useful side-channel for next-token CE before
  we ask it to do anything memory-specific.
- **Data:** `data/phase_B/fineweb_edu.parquet` (1.9 GB of educational
  web text). Streamed and tokenized on-the-fly with the Llama-3.2
  tokenizer (`src/data/phase1_loaders.py:fineweb_edu_phase1_iter`).
- **Step function:** `phase1_pretrained_step` (parallel teacher-forced).
- **Token budget:** ~100M tokens (~5000 steps at BS=20, T=256).
- **Wall-clock:** ~3-5 hours including compile-block warmup.
- **Why this matters:** Per [ProLong (Gao et al., 2024)](https://arxiv.org/html/2410.02660v3),
  the standard recipe for long-context post-training preserves a
  short-context mix to avoid breaking the model's natural language
  abilities. Wave 1 is our equivalent of that "preserve short-context"
  baseline. Without it, the walker would be fed pure memory tasks and
  never learn good general representations.

### Wave 2 — Phase-1b instruction/chat SFT (UltraChat-200k)

- **Goal:** model can follow chat-format prompts and instructions.
  This is what makes the eventual GRPO rollouts make sense (you can't
  reward "good chat continuation" if the model doesn't know what a
  chat continuation looks like).
- **Data:** `HuggingFaceH4/ultrachat_200k` (~200k chat conversations,
  Llama-3 chat template via Llama-3.2-Instruct's tokenizer).
- **Step function:** `phase1_pretrained_step` (still teacher-forced —
  per `train_phase1_ar.py` docstring, parallel SFT is the standard
  for instruction tuning; AR is reserved for memory-targeted training
  in Wave 3).
- **Token budget:** ~200M tokens (~10000 steps at BS=20, T=256).
- **Wall-clock:** ~7-9 hours including compile warmup.
- **Resume from:** Wave 1 checkpoint (`--resume`).

### Wave 3 — Synthetic passphrase recall (teacher-forced AR)

**REORDERED 2026-05-04** — passphrase comes BEFORE chat overflow because
it's the cheap controlled "does memory work at all" signal. See
`docs/wave3_passphrase_plan.md` for the full design + build spec.

- **Goal:** verify walker can store + retrieve user-specific facts
  buried in long filler text and asked about via flexibly-phrased
  questions. **No exact-match scoring** — BERT-cosine only.
- **Data:** ~150 user-curated facts × FineWeb-edu filler. Facts
  expanded to paraphrases + questions + reference answers via Claude
  API (one-time prep). Per [Zhao et al. 2024
  (Understanding Synthetic Context Extension via Retrieval Heads)](https://arxiv.org/html/2410.21276),
  use REAL text for filler — pure-synthetic filler underperforms.
- **Step function:** `phase1_ar_pretrained_step` (teacher-forced
  autoregressive — walker must carry the fact across filler segments
  because the LM only sees one continuation token at a time).
- **Loss:** CE on answer tokens.
- **Eval metric:** BERT-cosine (`all-mpnet-base-v2`) vs reference
  answers, on held-out facts (20 of 200).
- **Curriculum:** filler_mid length 100 → 1500 tokens.
- **Wall-clock:** ~1 day for first end-to-end smoke.

### Wave 4 — Phase-2 GRPO on real chat overflow (WildChat-1M)

- **Goal:** test naturalistic long-context recall. Real chat sessions
  where total turn count > T=256 segment forces the walker to be the
  only continuity carrier across boundaries.
- **Data:** [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat)
  (allenai, ODC-BY) — real user↔ChatGPT conversations. Filter to
  ≥600 tokens total so we have ≥2 segments of prefix + meaningful
  next-turn target. 650K-1M conversations available.
- **Eval (held out):**
  - [LoCoMo](https://github.com/snap-research/LoCoMo) (10 long-term
    conversations, QA + event-summarization annotations)
  - [REALTALK](https://github.com/danny911kr/REALTALK) (21-day real
    conversations, persona simulation + memory probing tasks)
- **Optional supplement:** 10-20% mix of [T1](https://consensus.app/papers/details/94cdeb424c96541b956bca1120bf77c0/)
  or [CoALM-IT](https://consensus.app/papers/details/65496bb1cfee5cb3b3543639c298e062/)
  for agent / tool-use flavor. Defer to v2.
- **Step function:** `grpo_step` (AR, REINFORCE on routing decisions).
  Walker's `freeze_all_but_E_bias_and_neuromod` freezes everything
  except `memory.neuromod.*`.
- **Reward:** BERT-cosine of generated next-turn vs reference, with
  `all-mpnet-base-v2`. NO exact match (per user preference and
  consistency with Wave 3).
- **Token budget:** ~50M tokens (GRPO is expensive per token —
  K=8 rollouts per step).
- **Wall-clock:** ~1-2 days per cycle.

## Concrete commands

### Wave 1 — production run

```bash
PYTHONPATH=. .venv/bin/python scripts/train_pretrained_gw.py \
    --data fineweb-edu \
    --max-steps 5000 \
    --bs 20 --T 256 \
    --work-dir outputs/wave1_fineweb \
    --warmup 200 \
    --lr 1e-4 \
    --ckpt-every 500 \
    --log-every 25
```

Saves checkpoints to `outputs/wave1_fineweb/ckpt_step{N}.pt` plus
`ckpt_final.pt`. Telemetry to `outputs/wave1_fineweb/stats.jsonl`
(parseable per-step row, ~40 metrics — see `src/graph_walker/telemetry.py`).

### Wave 2 — production run (resumes from Wave 1)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_pretrained_gw.py \
    --data ultrachat \
    --max-steps 10000 \
    --bs 20 --T 256 \
    --work-dir outputs/wave2_ultrachat \
    --warmup 100 \
    --lr 5e-5 \
    --ckpt-every 500 \
    --log-every 25 \
    --resume outputs/wave1_fineweb/ckpt_final.pt
```

LR is bumped down 2x for the SFT phase (standard SFT-after-pretrain
practice). `--warmup 100` is shorter since we're fine-tuning a
warmed-up model.

### Wave 3 — TBD (entry-point not yet wired)

Will need:
1. Pick a chat-overflow dataset
2. Build `(prefix_ids, reference_cont)` iterator where prefix spans ≥2
   segments (a real chat session with N turns where total length > T)
3. Reward function — see open question in `docs/training_strategy.md` § Wave 3
4. Adapt `scripts/train_pretrained_gw.py` to support `--data wildchat`
   or similar, calling `grpo_step` instead of `phase1_pretrained_step`

### Wave 4 — TBD (entry-point not yet wired)

Will need:
1. Synthetic data generator script (`scripts/build_memory_data.py`?)
   that reads FineWeb-edu paragraphs as filler and constructs
   `(prefix, reference_cont)` tuples per the passphrase template
2. Reward function combining exact-match + BERT-cosine
3. Curriculum scheduler (1 needle → 5 needles)

## Eval harness (cross-cutting)

After each wave, run a fixed eval suite to track progress:

- **Held-out FineWeb-edu CE** — language modeling perplexity on
  `data/phase_B/val_fineweb_edu.parquet`.
- **BABILong subset** — qa1 (single-fact retrieval), qa2 (two-fact
  joining), qa5 (counting), at lengths 1K-32K. Per
  [Kuratov et al. 2024](https://arxiv.org/abs/2406.10149) most LLMs
  effectively use only 10-20% of context — this is the gap we hope to
  close.
- **Custom NIAH** — single fact at varying depth + single needle at
  end (sanity baseline).

Eval harness not yet built. Should land alongside Wave 3 setup so we
have an "is the walker actually doing anything" signal after Wave 1+2
finish.

## What's wired / what's not

| Component | Status |
|---|---|
| `phase1_pretrained_step` (parallel teacher-forced) | ✓ wired, tested |
| `phase1_ar_pretrained_step` (AR unroll for memory-targeted SFT) | ✓ wired, tested, NOT used in waves 1+2 (per train_phase1_ar.py docstring, AR is for memory targeting) |
| `grpo_step` (Phase-2 GRPO REINFORCE) | ✓ step function exists; data + reward NOT wired |
| `run_cycle_loop` (orchestrator) | ✓ exists, NOT used by `train_pretrained_gw.py` (we drive `phase1_pretrained_step` directly for waves 1+2; cycle loop is reserved for the AR↔GRPO cycles in waves 3-4) |
| FineWeb-edu data iterator | ✓ `src/data/phase1_loaders.py:fineweb_edu_phase1_iter` |
| UltraChat data iterator | ✓ `src/data/phase1_loaders.py:chat_sft_phase1_iter` |
| Wave 1 entry-point | ✓ `scripts/train_pretrained_gw.py --data fineweb-edu` |
| Wave 2 entry-point | ✓ `scripts/train_pretrained_gw.py --data ultrachat` |
| Wave 3 entry-point | ✗ TBD |
| Wave 4 entry-point | ✗ TBD |
| Eval harness (BABILong / NIAH / held-out CE) | ✗ TBD |
| Checkpoint resume | ✓ `--resume <ckpt.pt>` (wrapper + opt + sched) |
| Telemetry | ✓ `StatsCollector` writes per-step jsonl with ~40 metrics |
| LR schedule | ✓ linear-warmup + cosine-decay to 10% of peak |

## Reference docs

- `docs/bench_results.md` — verified throughput numbers (BS, T, VRAM, tok/s)
- `docs/pretrained_graph_walker.md` — architecture detail for the integration
- `docs/graph_walker.md` — standalone walker design
- `docs/training_plan.md` — older v2-era design notes (partially superseded)
