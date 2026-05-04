# Wave 3 — Passphrase Memory Recall: Data Generation + Training Plan

**Status:** plan, not built. **Wave 3 = LONG-TEXT FILLER ONLY** (no
chat turns). The chat-injected variant moved to Wave 4 (AR+GRPO) so
we can validate basic recall under cheap teacher-forcing first. This
doc is the design + build spec for Wave 3.

## Goal

Verify that the walker can store + retrieve user-specific facts that
are buried in long filler text and asked about later via flexibly-phrased
questions. **No exact-match scoring** — BERT-cosine only — so the model
must show real semantic recall, not memorized template matching.

## Single mode for Wave 3 — Long-text passphrase (teacher-forced AR)

- **Filler source:** natural text from `data/phase_B/fineweb_edu.parquet`
- **Single document:** filler_pre + injected_fact + filler_mid + question + answer
- **Training step:** `phase1_ar_pretrained_step` — prefix = everything up
  to and including the question, continuation = answer
- **Loss:** CE on answer tokens only (teacher-forced AR — walker has to
  carry the fact through filler_mid because the LM only sees one token
  at a time during continuation)
- **Why AR not parallel:** parallel teacher-forced lets the LM see all
  prefix tokens directly at every position, diluting the walker's
  contribution. AR forces the walker to be the only continuity carrier
  during continuation. See `train_phase1_ar.py` docstring.

The chat-injected variant (Wave 4) reuses Wave 3's `expanded.json`
fact set but with chat filler and `grpo_step` instead of teacher-forced
AR — only the filler source + step function differ.

## Continual-learning extension (deferred sub-task within Wave 3)

Once basic single-fact recall works, add fact-overwrite eval:
- Construction: `filler_pre + fact_v1 + filler_mid_1 + fact_v2_same_topic + filler_mid_2 + question + answer_based_on_v2`
- Tests whether the walker's plasticity dynamics support **fact
  overwriting** — does the latest neuromod-driven `E_bias_flat` update
  win over an earlier conflicting one?
- Implementation cost: ~1 day of added work — one extra construction
  in `passphrase_loader.py` + one extra eval split. Simple structurally.
- Three possible outcomes, all informative:
  - Walker correctly recalls v2 → continual learning works ✓
  - Walker stuck on v1 → plasticity too sticky (γ too small / decay too slow)
  - Walker confused / hallucinates → not selective enough
- Closely related to LongMemEval's "knowledge update" task and Larimar's
  "contextual editing" — published precedent for the architecture class.
- **Defer to after baseline single-fact recall hits ≥0.7 BERT-cosine
  on held-out**.

## Data generation pipeline

### Step 1 — User-curated fact list

User maintains `data/passphrase/user_facts.yaml`:

```yaml
- id: 1
  fact: "I prefer pepperoni pizza with onions instead of mushrooms."
  topic: "food"
- id: 2
  fact: "I always run AI agents inside tmux when they control a remote host."
  topic: "tooling"
# ... 100-200 facts total
```

**Recommendation:** ~150 facts, diverse topics, NOT guessable from
common sense. User-specific preferences, behaviors, or personal context.

### Step 2 — Auto-expand via Claude API (one-time prep)

`scripts/build_passphrase_data.py` reads `user_facts.yaml` and emits
`data/passphrase/expanded.json` with, per fact:

```json
{
  "id": 1,
  "fact": "I prefer pepperoni pizza with onions instead of mushrooms.",
  "topic": "food",
  "fact_paraphrases": [
    "When I order pizza I get pepperoni with onions, not mushrooms.",
    "My go-to pizza is pepperoni and onions; I avoid mushrooms.",
    "Pepperoni and onions on pizza for me, never mushrooms."
  ],
  "questions": [
    "What kind of pizza does the user prefer?",
    "What pizza toppings does the user like?",
    "Does the user like mushrooms on pizza?",
    "Tell me about the user's pizza preferences.",
    "If we were ordering pizza for the user, what should we get?"
  ],
  "reference_answers": [
    "The user prefers pepperoni pizza with onions.",
    "Pepperoni and onions, not mushrooms.",
    "They like pepperoni with onions; avoid mushrooms."
  ]
}
```

**Why generate via Claude API:** the user explicitly does NOT want
template cookie-cutter Q&A. Variation in question phrasing and answer
style is essential — generating these by hand for 150 facts is too
expensive, and templated generation defeats the purpose. Claude (or
GPT) generates natural paraphrases at low one-time cost.

**Generation prompt sketch** (for the script to use):
> Given the user's fact: "{fact}"
> Generate:
> - 3 paraphrases of the fact (different phrasings, same meaning, all in
>   first-person from the user's perspective)
> - 5 questions that someone might ask to elicit this fact, varying in
>   directness and phrasing (some concrete, some abstract, one possibly
>   negation-style)
> - 3 reference answers a good assistant would give, in different
>   surface forms
> Output as JSON.

**One-time cost:** 150 facts × ~$0.01 per Claude API call ≈ $1.50 total.
Trivial.

### Step 3 — Optional distractor / negation facts

Add 30-50 "distractor" facts that LOOK plausible but are NOT in the
training set. At training time, occasionally construct examples where:
- The question asks about a distractor fact (none of the prefix
  contains it)
- The reference answer is "The user hasn't mentioned that" or similar

This trains the model to be **selective** — to say "I don't know" rather
than hallucinate. Critical for real-world deployment.

## Per-example construction (`src/data/passphrase_loader.py`)

For each training example:

1. **Sample a fact** from the training split (180 of 200 facts)
2. **Sample a paraphrase** of the fact (random from `fact_paraphrases`)
3. **Sample a question** (random from `questions`)
4. **Sample a reference answer** (random from `reference_answers`)
5. **Sample filler from FineWeb-edu** — read 1-3 contiguous paragraphs
6. **Inject the fact** at a random position within the filler
7. **Construct the prompt** and tokenize:
   ```
   <bos>
   [FILLER_PRE: 200-500 tokens]
   [FACT_PARAPHRASE]
   [FILLER_MID: 500-1500 tokens — controlled by curriculum]
   [QUESTION]
   <answer_marker?>
   [REFERENCE_ANSWER]
   <eos>
   ```
8. **Split for AR step:** `prefix_ids = [bos … question]`,
   `continuation_ids = [reference_answer … eos]`
9. **Yield** `Phase1ARBatch(prefix_ids=prefix_ids, continuation_ids=continuation_ids)`

**Curriculum schedule** (linear ramp over training):
- First 20% of steps: `FILLER_MID` length 100-300 tokens (~1 segment)
- Next 40%: 300-800 tokens (~2-3 segments)
- Final 40%: 800-1500 tokens (~3-6 segments)

This is the standard "curriculum on horizon length" pattern from
[Synthetic Curriculum (BABILong, 2024)](https://arxiv.org/html/2406.10149v1):
"raising success rates at lower horizons also raises success rates for
longer horizons."

## Multi-needle variant (later, optional)

Once single-needle works, support N=2-5 needles per example with
question targeting one of them. Tests capacity, not just retention.

## Eval harness (`src/eval/passphrase_eval.py`)

For each held-out fact (20 of 200):
1. Construct example exactly as in training (sample paraphrase,
   question, filler) BUT do NOT include reference answer in the prompt
2. Use `wrapper.memory.update_plasticity(None)` mode (no plasticity
   firing during eval)
3. AR generate the answer (`autoregressive_rollout` with
   `temperature=0.7`, `top_p=0.9`, `gen_length=64`)
4. Encode generated answer + each reference answer with
   `sentence-transformers/all-mpnet-base-v2`
5. Compute cosine similarity for each reference; take MAX
6. Aggregate metrics:
   - **Recall@0.7**: fraction of held-out facts where max BERT-cosine ≥ 0.7
   - **Recall@0.5**: fraction where ≥ 0.5
   - **Mean similarity** across all held-out facts
   - **Per-curriculum-level breakdown**: easy / med / hard filler lengths

Track these across training to see when memory "kicks in." Expected
trajectory: starts ~0.3 (random-baseline level), rises with training.

## Implementation files (to build)

| File | Purpose |
|---|---|
| `data/passphrase/user_facts.yaml` | User maintains. ~150 facts. |
| `scripts/build_passphrase_data.py` | One-time: expand facts → expanded.json via Claude API. |
| `data/passphrase/expanded.json` | Generated. The training-time data source. |
| `src/data/passphrase_loader.py` | `passphrase_phase1ar_iter` — yields `Phase1ARBatch` for training. |
| `src/eval/passphrase_eval.py` | Held-out eval with BERT-cosine scoring. |
| Adapt `scripts/train_pretrained_gw.py` | Add `--data passphrase` option, dispatching to `phase1_ar_pretrained_step`. |
| `tests/test_passphrase_loader.py` | Smoke test: 1 example constructs cleanly, prefix/continuation split is correct. |

## Reward / loss summary

| Mode | Loss/reward at train time | Eval metric |
|---|---|---|
| 3a (long-text, teacher-forced AR) | CE on answer tokens | BERT-cosine vs ref answers (held out) |
| 3b (chat, AR GRPO) — deferred | BERT-cosine reward, REINFORCE | Same as training reward, held-out split |

Per user: BERT-cosine ONLY for evaluation. **No exact match scoring at
any point** — the user explicitly rejects exact-match because they want
flexible recall.

## BERT encoder choice

`sentence-transformers/all-mpnet-base-v2`:
- ~110M params, ~80 MB in fp16 on GPU
- Strong on short-text similarity (STS-b: 86.5)
- Fast: ~5ms per encode on RTX 4090 for short answers
- Loaded once at eval/training start, kept on GPU

Smaller alternative: `all-MiniLM-L6-v2` (~23M, slightly worse quality,
2x faster). Default to mpnet; can swap if encoding becomes a
bottleneck (unlikely at our scale).

## Token budget per example (initial defaults)

- FILLER_PRE: 300 tokens (median; sampled 200-500)
- FACT: 20 tokens
- FILLER_MID: curriculum-controlled, 100-1500 tokens
- QUESTION: 30 tokens
- ANSWER (continuation, teacher-force target): 30-80 tokens
- **Total prefix length: ~500-2000 tokens (2-8 segments at T=256)**
- **Continuation length: ~50-100 tokens**

At BS=20 the prefix dominates VRAM. Smaller BS may be needed (e.g.
BS=8 for the long-filler curriculum stage). Will measure during smoke.

## Open implementation questions for the user

1. **Generate paraphrases / questions / reference answers via Claude
   API, or write by hand?** (Defaulting to Claude — same workflow as
   `scripts/build_passphrase_data.py` already in the v2 path. Cost ~$2.)
2. **Distractor / negation facts?** I'd recommend yes — adds 20% more
   data for "selectivity training" with little extra effort.
3. **Multi-needle variant — start with single, defer multi?** Yes,
   defer multi to after single-needle works.
4. **Multiple reference answers — pick random at training, or always
   first?** Random at training (exposes model to multiple correct
   surface forms); use all at eval (max BERT-cosine).
5. **Curriculum start point — start at FILLER_MID=100 (1 segment) or
   jump straight to overflow at FILLER_MID=500 (2 segments)?** Start
   at 100 to validate the easy case before forcing memory.
