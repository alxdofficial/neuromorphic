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

## The 5-wave training plan

The architecture is novel enough that doing all of pretraining + SFT
+ memory-task RL in one shot is risky. We sequence in 5 waves so each
stage adds one capability to a model that's known-good at the previous
stage. **Reordered 2026-05-04** to split the passphrase task into a
teacher-forced (long-text only, no turns) and AR-GRPO (chat-injected,
turns present) variant, with chat overflow as the final wave.

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
- **Step function:** `phase1_pretrained_step` (parallel teacher-forced —
  the standard for instruction tuning).
- **Token budget:** ~200M tokens (~10000 steps at BS=20, T=256).
- **Wall-clock:** ~7-9 hours including compile warmup.
- **Resume from:** Wave 1 checkpoint (`--resume`).

### Wave 3 — Passphrase recall, **chat-injected** (GRPO)

**Walker must hold a user-injected personal fact across filler chat
turns and recall it via AR generation when later asked.** This is
where the walker's value prop starts mattering: under GRPO, the LM
samples its own tokens (no teacher forcing), so the walker IS the
continuity carrier.

- **Goal:** when the user injects a personal fact mid-conversation, can
  the walker retain it long enough that a later turn can recall it via
  AR generation?
- **Data:** 500 mock facts × UltraChat-style chat filler. Fact embedded
  as a user turn ("by the way, ..."). Question asked many turns later.
  Built via `scripts/build_user_facts.py` →
  `data/passphrase/expanded.json`.
- **Step function:** `grpo_session_step` with `sessions=[s1..s_B]`,
  each `s_i` a 2-turn `MultiTurnSession` (user prefix + assistant ref).
  Internally routes to the uniform-batched fast path → one
  `grpo_step` call with B*K parallel rollouts. `freeze_all_but_E_bias_and_neuromod`
  for the trainable surface (memory.neuromod.* only).
- **Reward:** BERT-cosine of generated answer vs reference answers
  (`all-mpnet-base-v2`). Pure semantic, no exact match.
- **`lm_context_window`** (recommended for Wave 3): when set < `T_pre`,
  walker absorbs the full prefix but LM only attends to the last
  `lm_context_window` tokens. With long filler (`filler_max=1500-3000`)
  and `lm_context_window=256`, the LM can't see the fact directly —
  the walker's memory is the only path from fact to answer. This is
  the configuration where the walker actually has to earn its keep.
- **Resume from:** Wave 2 checkpoint (REINFORCE has no signal from
  fresh init — needs a chat-aware policy first).
- **Wall-clock:** ~3-4 hours per pass over the 30K-pair corpus at
  B=8, K=8 (see `docs/bench_results.md` 2026-05-06 BS_outer sweep).

**Historical note:** an earlier Wave 3 (AR-unrolled teacher-forced SFT
on filler+fact+filler+question) was retired in scope-B cleanup
(2026-05-06). Under teacher forcing, the LM has full attention to the
prefix and can solve the task without the walker, so the walker only
contributed a "tiny CE delta" (per `phase1_ar_pretrained_step`'s own
docstring at the time). Wave 3 is now the chat-injected GRPO that was
formerly numbered Wave 4.

### Wave 4 — Real long chat / agent overflow (WildChat-1M, multi-turn GRPO)

- **Goal:** test naturalistic long-context recall. Real chat sessions
  where total length > T=256 segment forces the walker to be the only
  continuity carrier across boundaries.
- **Data:** [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat)
  (allenai, ODC-BY). Pretokenized to v2 multi-turn schema via
  `scripts/preprocess_wildchat_llama32.py`: full conversation tokens +
  per-turn boundary index `[session_idx, role_id, turn_start, turn_end]`
  + per-session boundary index. Filters: ≥1500 tokens AND ≥4 assistant
  turns; max 8000 tokens/session (truncated at turn boundary). Target
  30K sessions.
- **Eval (held out):**
  - [LoCoMo](https://github.com/snap-research/LoCoMo) (10 long-term conversations)
  - [REALTALK](https://github.com/danny911kr/REALTALK) (21-day real conversations)
- **Step function:** `grpo_session_step` (multi-turn aligned-trajectory
  protocol — see `docs/multi_turn_grpo_plan.md`). **Turn-batched
  (Verlog-style):** the dataset is reformulated as a flat pool of
  `TurnPair(cumulative_prior, response)` units — each assistant turn
  in each WildChat session becomes one independent training unit. The
  loader maintains a sort-and-sample pool of M=2048 turn-pairs; per
  outer step it picks B contiguous neighbors (near-uniform prior
  length), truncates to the shortest prior in the batch, wraps each as
  a 2-turn `MultiTurnSession`, and yields the list. These slot into
  the same uniform-batched fast path Wave 3 uses → true B*K parallel
  rollouts. Per-prompt advantage normalization within each K-group.
- **Reward:** BERT-cosine vs reference assistant turn, per turn,
  `all-mpnet-base-v2`. No exact match.
- **EOS early-stop:** rollouts terminate at `eos_id` or
  `max_response_len` (whichever first). Post-EOS pad force-fed eos_id
  for trace-length consistency; reward decoder strips.
- **Loss masking:** assistant turns sampled+REINFORCE; user/system
  turns teacher-forced into the cumulative prior with no GRPO loss
  (the standard "rlhf" mask in 2025 multi-turn GRPO practice).
- **`lm_context_window`** (recommended for Wave 4): set `< prior_len`
  to force the walker to bridge across turns. Each turn-pair's prior
  is naturally capped by the preprocessor's `max_tokens_per_session`
  (default 8000) — so walker absorbs up to 8K tokens of conversation
  context. With `lm_context_window=1024`, the LM can only attend to
  the last ~1K tokens; the walker is the only thing carrying earlier
  context into the assistant turn's gen.
- **Wall-clock:** TBD — first cycle estimate after MT-GRPO bench runs.

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

### Wave 3 — production run (resumes from Wave 2)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_grpo.py \
    --data passphrase-chat-grpo \
    --resume outputs/wave2_ultrachat/ckpt_final.pt \
    --max-steps 75000 --grpo-K 8 --bs-outer 8 \
    --T-pre 2048 --gen-length 128 \
    --filler-min 1000 --filler-max 1800 \
    --lm-context-window 256 \
    --work-dir outputs/wave3_passphrase_chat \
    --lr 3e-5 --warmup 200
```

Routes through `grpo_session_step` with the uniform-batched fast path
(8 sessions/step, all 2-turn, matching prefix length → one B*K=64-rollout
GRPO update per step). With `T_pre=2048` and `lm_context_window=256`,
the walker absorbs the fact + 1700 tokens of filler + question, but the
LM only attends to the last 256 tokens — the walker's memory is the
only path from fact to answer.

### Wave 4 — production run (resumes from Wave 3)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_grpo.py \
    --data wildchat-grpo \
    --resume outputs/wave3_passphrase_chat/ckpt_final.pt \
    --max-steps 100000 --grpo-K 8 --bs-outer 8 \
    --turn-pair-pool-size 2048 \
    --gen-length 128 \
    --lm-context-window 1024 \
    --work-dir outputs/wave4_wildchat \
    --lr 1e-5 --warmup 200
```

Wave 4 turn-batching reuses Wave 3's uniform-batched fast path via the
`TurnPair` flattener — each call yields B near-uniform-length turn-pairs
extracted from random sessions/rounds. `lm_context_window=1024` forces
the walker to bridge information across turns that fall outside the LM's
1K-token attention reach.

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
| `grpo_session_step` (unified Phase-2 GRPO; uniform-batched fast path for both Wave 3 and Wave 4 turn-pairs) | ✓ wired, tested — used by both waves |
| `grpo_step` (single-turn BS_outer-aware) | ✓ kept as internal helper called by uniform-batched session path |
| `wildchat_turn_pair_grpo_batch_iter` (Wave 4 turn-pair flattener + sort-and-sample) | ✓ wired, tested |
| Walker `snapshot_memory_state` / `restore_memory_state` | ✓ wired, tested (used by sequential fallback path; not on Wave 4 production path anymore) |
| `lm_context_window` two-phase forward (walker / LM context decouple) | ✓ wired, tested |
| Per-group reward stats in `GRPOStats` (`per_group_reward_mean/std`) | ✓ wired, tested |
| FineWeb-edu data iterator | ✓ `src/data/phase1_loaders.py:fineweb_edu_phase1_iter` |
| UltraChat data iterator | ✓ `src/data/phase1_loaders.py:chat_sft_phase1_iter` |
| Wave 1 entry-point | ✓ `scripts/train_pretrained_gw.py --data fineweb-edu` |
| Wave 2 entry-point | ✓ `scripts/train_pretrained_gw.py --data ultrachat` |
| Wave 3 entry-point | ✓ `scripts/train_grpo.py --data passphrase-chat-grpo` |
| Wave 4 entry-point | ✓ `scripts/train_grpo.py --data wildchat-grpo` (turn-batched, B*K parallel via Wave 3's uniform-batched fast path) |
| Eval harness (BABILong / NIAH / held-out CE) | ✗ TBD |
| Checkpoint resume | ✓ `--resume <ckpt.pt>` (wrapper + opt + sched) |
| Telemetry | ✓ `StatsCollector` writes per-step jsonl with ~40 metrics |
| LR schedule | ✓ linear-warmup + cosine-decay to 10% of peak |

## Reference docs

- `docs/bench_results.md` — verified throughput numbers (BS, T, VRAM, tok/s)
- `docs/pretrained_graph_walker.md` — architecture detail for the integration
- `docs/graph_walker.md` — standalone walker design
- `docs/training_plan.md` — older v2-era design notes (partially superseded)
