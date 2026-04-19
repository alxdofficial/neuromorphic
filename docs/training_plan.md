# Training Plan — Dataset + Reward Strategy for Memory-Augmented Llama

**Status:** design. No training run launched yet. Written 2026-04-19 as a
reference after researching the long-context memory benchmark landscape.

This doc captures:
1. The three candidate data tiers and when to use each.
2. The reward-function options and their paraphrase-tolerance tradeoffs.
3. The recommended sequenced training approach and the reasoning behind it.
4. Concrete files / modules to build for each stage.

## 1. The core design question

Our memory architecture is unusual — it **writes, it doesn't retrieve**.
Every 4 tokens the modulator emits a code → decoder → ΔW, and ΔW is
γ-clamped-EMA-blended into W. No attention-based KV lookup, no stored
(K,V) bank. This is closer to a **plastic recurrent cell** than to the
retrieval banks of LongMem / Memorizing Transformers / RAG.

Two consequences:
- Whether this architecture can learn to **store** anything useful is an
  open question. verify_01 on the from-scratch branch already failed
  once with SNR=1e-4 on naive teacher-forced GRPO CE rewards.
- The first job of training is answering *"does the memory graph learn
  to store and retrieve information at all?"* before we worry about
  *"does memory help on real conversational benchmarks?"*.

## 2. Data tier landscape

### Tier 1 — Synthetic, controlled ("memory-or-fail" signal)

| Dataset          | What it tests                                     | Pros                                    | Cons                                               |
|------------------|---------------------------------------------------|-----------------------------------------|----------------------------------------------------|
| Passkey / NIAH   | Exact recall of planted fact at unknown distance | Binary reward, unambiguous signal, ∞ data on the fly, fastest debug loop | Narrow, can train memory to be a lookup table rather than a compressor |
| K:V recall       | "Key1: Alice, ..., Key37: ?" — associative lookup | Compositional, still binary-scorable, tests write-then-read | Same narrowness                                    |
| Ordered list     | "the sequence is apple pear orange ... what's at position 5?" (user's own framing) | Matches user's mental model; tests sequential indexing | Same                                               |
| BABILong         | bAbI reasoning tasks embedded in distractor text  | Tests reasoning + memory jointly        | Requires bAbI dataset; more complex to score       |
| RULER            | Multi-needle, multi-hop, aggregation over haystack| NVIDIA open-source; publishable baseline| Larger, more to orchestrate                        |

**Key reference:** [RULER](https://arxiv.org/abs/2404.06654) (Hsieh 2024, 509 citations) —
the current yardstick for long-context eval. 17 LLMs tested; half
collapsed at 32K tokens despite claiming 32K+ context. Uses synthetic
needles + multi-hop tracing + aggregation. [Larimar NIAH paper](https://arxiv.org/abs/2406.00009)
(Nelson 2024) is the closest precedent to our architecture: tests a
write/read external memory on passkey + NIAH.

### Tier 2 — Conversational / agentic ("realistic memory" signal)

| Dataset       | Scale                                    | What it tests                                       |
|---------------|------------------------------------------|-----------------------------------------------------|
| [LongMemEval](https://arxiv.org/abs/2410.10813) | 500 questions, 115K–1.5M tokens, chat histories | Info extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention |
| [LoCoMo](https://arxiv.org/abs/2402.17753)      | 300 turns × 9K tokens × 35 sessions per convo    | Very long-term dialogue consistency                  |
| DuLeMon / PLATO-LTM | Persona-memory dialogue                  | Earlier conversational-memory baseline               |
| Custom synthetic dialogues | Controlled by us                   | Plant a "personal fact" early, query late; tests the user's own framing |

**Finding:** LongMemEval measured **30% accuracy drop** on commercial
chat assistants when memory-requiring facts are separated by sustained
interactions. This is the target regime for our model.

### Tier 3 — Natural long-form ("dense training signal")

- **PG-19** (already cached in `data/` via `datasets--deepmind--pg19`) —
  28K Project Gutenberg books, ~69K tokens each.
- **BookCorpus**, **FineWeb-edu** long documents.

Dense LM training. Memory helps via book-level coherence (characters,
plot). No explicit per-example memory test — we'd evaluate via perplexity
improvement over vanilla Llama on the continuation.

## 3. Reward function landscape

### The paraphrase-tolerance problem

Exact token-match reward on a 128K-vocab continuation is near-zero-signal
— the reviewer flagged this as a footgun and we've confirmed it
empirically (the verify_01 SNR failure comes from exactly this).
Paraphrase-tolerant rewards are the research frontier for GRPO on
open-ended generation.

### Candidate rewards, ordered by cost and fidelity

| Reward                          | Cost      | Paraphrase-tolerant | Dense? | Use case                                |
|---------------------------------|-----------|---------------------|--------|-----------------------------------------|
| Exact token match               | ~0        | ✗                   | no     | Synthetic tasks with known exact answer |
| Entity / keyword F1             | trivial   | ±                   | ±      | Fact-recall tasks (entity names, numbers) |
| Sentence-BERT cosine similarity | ~1ms      | ✓                   | yes    | Open generation with paraphrase         |
| [PrefBERT](https://arxiv.org/abs/2506.15068) (encoder-only scoring model) | ~5ms | ✓ | yes | Open-ended long-form GRPO — outperforms ROUGE-L and BERTScore as reward signal |
| BLEURT / BERTScore              | ~10ms     | ✓                   | yes    | Traditional; known to correlate poorly with human prefs on long-form |
| Log-prob of reference under memory-augmented LM | ~20ms | N/A (scores model, not generation) | yes | Direct "does memory help LM predict the truth" signal |
| [LLM-as-judge](https://arxiv.org/abs/2410.10813) (GPT-4o) | ~$$$/step | ✓ | yes | Gold-standard (97% human agreement); prohibitively expensive per training step |
| [J1](https://arxiv.org/abs/2506.06499) self-hosted judge | ~100ms | ✓ | yes | Trained dedicated judge LLM; strong alternative if we can afford to host it |

### Critical vulnerabilities to avoid

[One Token to Fool LLM-as-Judge](https://arxiv.org/abs/2507.08427) (Zhao 2025) — master-key
tokens like `":"` or `"Thought process:"` can elicit false-positive
rewards from generative reward models, including GPT-o1 and Claude-4.
**If we go LLM-as-judge, we need adversarial training (Master-RM
pattern) to defend against reward hacking.** For per-step GRPO reward,
prefer cheaper embedding-based alternatives that are harder to hack.

### Recommended reward for each task type

| Task              | Reward                                                                                                 |
|-------------------|--------------------------------------------------------------------------------------------------------|
| Passkey           | `1 if passkey_tokens ⊂ generated else 0` — exact-match is correct here                                  |
| K:V recall        | `F1(value_tokens_generated, value_tokens_reference)` — partial credit for partial match                  |
| Ordered list      | `1 if correct_item_at_position in generated else 0` — binary match                                      |
| Custom convo fact | `0.5·sentence_bert_cosine + 0.3·neg_CE_of_reference + 0.2·entity_recall` — paraphrase-tolerant composite |
| LongMemEval       | Per their protocol: GPT-4o-as-judge for eval; for TRAINING use composite above                          |
| PG-19             | `neg_CE_of_ground_truth_continuation` over ≥ 512 tokens (compound memory effect over length)             |

## 4. Sequenced training approach — recommendation

**Not either/or. Sequenced.**

### Stage 0 — Architecture sanity (1-2 weeks, Tier 1)

**Data:** Custom synthetic — passkey + K:V recall + ordered-list.
Generated on-the-fly in Llama-3.2 token IDs (skip tokenizer pass).

**Objective:** Answer *"does the memory graph learn to store anything?"*

**Reward (phase 2 GRPO):** Exact-match / F1. Binary signal.

**Decision gate:**
- ✓ Memory solves passkey @ 2K+ distance → proceed to Stage 1.
- ✗ Memory fails passkey after reasonable iteration → **change the
  architecture, not the data**. Don't move to realistic data to paper
  over an architectural failure.

**What to iterate on if passkey fails:**
- Modulation interval (fewer fires = less capacity; more fires = slower
  but more expressive writes)
- γ clamp on W update (currently 0.5; might need higher for active
  rewriting or lower for stability)
- Codebook size (currently 2048; might need 8K for richer writes)
- Decoder capacity
- Scale init (currently √α=2; might need colder init to warm up)
- W init density (currently sparse-8-neighbors; passkey might need
  denser or sparser)

**What to NOT iterate on:** more real data. A broken architecture won't
be fixed by richer text.

### Stage 1 — Conversational memory (2-3 weeks, Tier 2 custom)

**Data:** Custom synthetic personal-fact dialogues. Template:

```
[User]: Hi, my name is <name> and I live in <city>.
[Asst]: Nice to meet you, <name>!
[...5-20 turns of unrelated chat, ~2-5K tokens...]
[User]: By the way, what city am I from?
[Asst reference]: You mentioned you live in <city>.
```

Vary: distance from fact to query, number of distractor turns, type of
fact (name, city, occupation, preference, numeric detail). Controlled by
us — unlimited data, no licensing.

**Objective:** Memory can extract and retrieve facts from early
conversation history under paraphrase-tolerant evaluation.

**Reward (phase 2 GRPO):** Composite
`0.5·sentence_bert_cosine + 0.3·neg_CE_reference + 0.2·entity_recall`.

**Decision gate:**
- ✓ Memory outperforms vanilla Llama on the custom benchmark → proceed.
- ✗ Fails after iteration → go back to Stage 0 with what we learned.

### Stage 2 — Publishable benchmarks (open-ended)

**Data:** LongMemEval proper + RULER + BABILong.

**Objective:** Match / beat published baselines.

**Reward:** Per-benchmark protocol. LongMemEval uses GPT-4o-as-judge for
eval. For continued training (if needed) use the same composite reward
as Stage 1 plus J1-style self-hosted judge if budget allows.

### Stage 3 (stretch) — PG-19 / natural long-form

**Data:** PG-19 streaming with on-the-fly Llama-3.2 tokenization.

**Objective:** Does memory lift perplexity on naturalistic long-form
text? Does the trained memory generalize beyond controlled tasks?

**Reward:** `neg_CE` over long continuations (≥ 512 tokens).

## 5. Concrete build plan (what to implement next)

### `src/pretrained/rewards.py`

Module with:
- `exact_match_reward(generated, reference)` — binary; handles passkey, ordered-list, K:V recall
- `entity_recall_reward(generated, reference_entities)` — token-set F1 on entities
- `sentence_bert_cosine_reward(generated, reference, encoder=None)` — uses
  `sentence-transformers/all-MiniLM-L6-v2` (~90MB, fast) as default
- `neg_ce_reward(wrapper, prefix, reference_continuation)` — log-prob of
  reference under memory-augmented LM
- `composite_reward(...)` — weighted combination of above

### `scripts/build_synthetic_memory_data.py`

Generators for Stage 0 + Stage 1:
- `passkey_iterator(length, bs, distance_range, vocab_size, tokenizer)` —
  yields `Phase1Batch` / `(prefix, reference)` pairs
- `kv_recall_iterator(n_pairs, query_offset, ...)` — associative K:V
- `ordered_list_iterator(list_len, query_position, ...)` — "what's at
  position N?"
- `personal_fact_dialogue_iterator(distance, distractor_types, ...)` —
  Stage 1 conversational format

All yield both (a) phase-1 `Phase1Batch(input_ids, target_ids, prev_token)`
for bootstrap training and (b) phase-2 `(prefix, reference)` pairs for
GRPO rollouts.

### `src/pretrained/train_stage0_driver.py`

Driver that wires synthetic iterators + exact-match reward into
`run_cycle_loop`. This is what we'd run as `python -m src.pretrained.train_stage0_driver`.

### `src/data/long_memory_stream.py`

Stage 2/3 wiring: LongMemEval + PG-19 streamers into the
`Phase1Batch` / `Phase1ARBatch` / `(prefix, reference)` shapes.

## 6. Research references

### Long-context memory benchmarks

- [RULER](https://arxiv.org/abs/2404.06654) (Hsieh et al., 2024) — NVIDIA.
  509 citations. The yardstick. NIAH + multi-hop + aggregation.
- [BABILong](https://arxiv.org/abs/2406.10149) (NeurIPS 2024) — bAbI
  reasoning tasks embedded in distractor text. Scales to 10M tokens.
  Measures LLMs effectively use only 10-20% of claimed context.
- [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., 2024, ICLR 2025) —
  500 chat-history QA questions, 115K–1.5M tokens. GPT-4o judge has 97%
  human agreement.
- [LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., 2024) —
  very long dialogue, 300 turns / 35 sessions / 9K tokens per conversation.
- [Larimar NIAH](https://arxiv.org/abs/2406.00009) (Nelson et al., 2024) —
  external write/read memory tested on passkey. Closest published
  precedent to our architecture.

### Semantic rewards for GRPO

- [PrefBERT](https://arxiv.org/abs/2506.15068) (Li et al., 2025) —
  encoder-only semantic reward model for GRPO on open-ended generation.
  Outperforms ROUGE-L and BERTScore as reward signal.
- [Shaping Explanations](https://arxiv.org/abs/2509.13081) (2025) —
  cosine similarity via encoder-only transformer as GRPO reward.
- [J1](https://arxiv.org/abs/2506.06499) (Whitehouse et al., 2025) —
  RL-trained dedicated LLM judges. J1-Qwen-32B beats o3, o1-mini, and
  671B DeepSeek-R1 on judgment benchmarks. Self-hostable alternative.
- [One Token to Fool LLM-as-Judge](https://arxiv.org/abs/2507.08427) (Zhao et al., 2025) —
  generative reward models systematically hackable via master-key
  tokens. Adversarial training (Master-RM) recommended if LLM-as-judge.

## 7. Open questions / design decisions to revisit

1. **Composite reward weights** — the `0.5 / 0.3 / 0.2` split for
   cosine / neg-CE / entity-F1 is a guess. Should be tuned once Stage 1
   data exists and we can measure reward-SNR.
2. **Embedding encoder choice** — `all-MiniLM-L6-v2` is fast and cheap
   but English-only and English-leaning for semantics. If we want
   better long-context or multilingual, consider `sentence-t5-large` or
   `bge-large-en-v1.5`.
3. **Reference policy for KL regularization** — not yet wired in
   `grpo_step` (flagged in `pretrained_lm_memory.md` as future work).
   Needed before production phase-2 runs to prevent policy collapse.
4. **Train on PG-19 during Stage 2 concurrently** — might help
   memory generalize beyond synthetic. Or might harm Stage 1
   specialization. Empirical.
5. **Single-stage vs cycle** — current plan assumes `run_cycle_loop`'s
   bootstrap → cycle-p1 → cycle-p2 structure works for all three
   stages. Might turn out some stages are better as flat phase-1-only
   or flat phase-2-only. Revisit after Stage 0 results.
