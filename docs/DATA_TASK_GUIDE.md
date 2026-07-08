# Data Task Guide — What Each Task Does, With Examples

A guided, example-driven tour of the five training tasks and the config knobs that dial their
difficulty. Companion to [`SCRUTINY_PHASE_DATA.md`](SCRUTINY_PHASE_DATA.md) (the architectural
reference) — read this one to build intuition for *what the model actually sees*. Every example below
was drawn through the real training loaders (`scripts/diagnostics/mixed/episode_peek.py`).

The setup in one line: a **frozen** decoder reads `M=96` memory tokens produced from a `total_len=2048`
context (streamed in `window_size=256` chunks); it **never sees the raw context**, only the memory. The
five tasks probe different things memory must do.

| mix-task (old alias) | source | what it probes | question → answer |
|---|---|---|---|
| **reconstruct** (`mae`) | fineweb | storage / fidelity | "reconstruct the text" → the passage |
| **continuation** | multicorpus | gist under accumulation | "continue" → the next tokens (at each window boundary) |
| **fact_recall** (`condrecon_bio`) | bio | key→value binding | "`<key>` =" → the key's value-facts |
| **babi** | babi | relational binding | "Where is `<person>`?" → a location |
| **doc_qa** (`qa_rc`) | qa_multi | retrieval / addressing | a real question → its answer in one packed passage |

Round-robin, equal share (20% each). Every context is exactly 2048 tokens (fixed compression
denominator); the decoder is told nothing but the memory + the question.

---

## 1. `reconstruct` — storage / fidelity (MAE)

**What it does.** Draw one fineweb passage, compress the whole 2048-token span into memory, then
reconstruct it — with ~85% of positions masked in the decoder input (`mask_ratio=0.85`), so the model
must recall them *from memory*, not copy visible neighbors. The answer *is* the input span; loss on
every token. This is the pure "can memory store the bits" probe (the adversarial worst case).

```
Q:      "Reconstruct the text above."
CONTEXT: " expounds in his 1942 book; Grundformen und Erkenntnis menschlichen Daseins (Basic Forms
          and the Realization of Human "Being-in-the-World"). In this work he explains existential
          analysis as an empirical science …"                                       (2048 tokens)
GOLDEN:  ← the same 2048 tokens (reconstruct all)
```

---

## 2. `continuation` — gist under accumulation (multi-horizon)

**What it does.** Draw one document (fineweb/pile/redpajama/**code**), and at **each streaming-window
boundary** (256, 512, …, 2048) compress the prefix-so-far and predict the next `predict_len=64` tokens
from memory alone. One episode tests memory at 8 growing horizons; the losses average into one
backward. Input is always ground truth (teacher-forced); memory **accumulates** across windows.

```
Q:      "Continue the passage."
CONTEXT: "…A third theory was advanced by defense counsel George Vanderveer. In his opening statement,
          Vanderveer said 'I exonerate now and forever the American Legion …'"       (2048 tokens)
GOLDEN:  " March 1, 1920\n- 'Scene in I.W.W. Hall Prior to Shooting is Explained in Detail', Chronicle
          (Spokane, WA), February 13, 1920 …"                          (the next 64 tokens, ×8 horizons)
```

---

## 3. `fact_recall` — key→value binding (bio)

**What it does.** Pack ~30 `key = value` biographical facts (entities from a 410-entity procedural
world) to fill 2048, then ask about 1–3 keys. Loss falls **only on the un-guessable value fragments** —
never the entity name or template scaffolding. This is the realistic binding anchor: memory must bind
each key to its specific attributes and read the right one back.

```
Q:      "the Suspension of the Egilstad Assembly ="                            (+ a 2nd key, multi2)
CONTEXT: "Edvard Reinholdt = Note: Edvard, the elder of two daughters of a Lutheran pastor, known for
          the management of high-risk home births …  Halsten Sjoblom, born 1988 = Halsten Sjoblom
          (fond of amateur lithography) …"                             (~30 packed key=value facts)
GOLDEN:  " twenty-first  the end of a long dispute over inland water rights   watercolour painting of
          coastal landscapes  wearing the same gray wool scarf every winter …"   (value-spans only)
```
Note the golden skips the entity name and "Note:/known for/=" scaffolding — only the load-bearing
value fragments are scored (that's what makes the answer un-guessable from the question alone).

---

## 4. `babi` — relational binding (multi-segment, entity-renamed)

**What it does.** bAbI stories are tiny (~86 tok), so we co-pack **~24 stories to fill 2048** and query
1–3 of them. bAbI reuses a handful of names across stories, so each packed segment is **entity-renamed
disjoint** — people (subjects of action verbs) and objects get fresh names from large pools; locations
are left alone (they're answers). One segment's "Ulyssa" can't collide with another's. The task becomes
*retrieve the right segment, then reason about it*.

```
Q:      "What is Ulyssa carrying?"
CONTEXT: "Ulyssa travelled to the bedroom.  Ulyssa grabbed the phial there.  Xavier went to the
          bathroom.  Ulyssa travelled to the office. … Ulyssa discarded the phial there. …
          Tobias grabbed the gimlet there.  Grimwald and Querela travelled to the garden. …"
          (~24 renamed stories: people Ulyssa/Xavier/Waldemar/…, objects phial/urn/gimlet, real locations)
GOLDEN:  " nothing"     ← Ulyssa grabbed the phial then discarded it → carrying nothing
```
Original bAbI names (Mary/John/…) never appear; the answer is derived by reasoning over Ulyssa's
segment alone (verified: no cross-segment collision over 450 episodes).

---

## 5. `doc_qa` — retrieval / addressing (5 real RC sources)

**What it does.** Real reading-comprehension QA. Each episode packs a gold passage + other passages as
distractors (from a random mix of 5 sub-sources), and asks 1–2 questions; the answer lives in exactly
one packed passage (un-guessability enforced — no distractor contains it). Memory must *address* the
right passage among noise. One example per sub-source:

**squad** (single-paragraph factoid):
```
Q: "Who believe that the Ming dynasty did not exercise any direct political control over Tibet?"
   (gold Tibet paragraph packed with a Kondo-orchestral-music paragraph as distractor)
A: "Josef Kolmaš"
```
**hotpot** (multi-hop, comparison):
```
Q: "Was Nam Woo-hyun or Eddie Vedder born first?"     (both bios packed; compare 1991 vs 1964)
A: "Eddie Vedder"
```
**musique** (multi-hop, chained):
```
Q: "The legal system of Au Kam San's country of birth comes from what tradition?"
   (Au Kam San → Macau → Portuguese law — hops across packed paragraphs)
A: "Portuguese-based legal system"
```
**multiwoz** (dialogue slot recall):
```
Q: "What party size did the user request for the hotel?"      (gold booking dialogue + distractors)
A: "2"
```
**triviaqa** (open trivia over evidence):
```
Q: "In which decade did Billboard magazine first publish an American hit chart?"
A: "30s"
```
Multi-query episodes ask about two packed passages (which may be from different sub-sources), and score
both answers.

---

## 6. Config knobs — how difficulty is dialed

Every difficulty lever, where it lives, and what it does. Fill is always **budget-driven** (pack until
`total_len`), so the *segment count* falls out of item size — you set the ratio and the pressure, not
the count.

| knob | layer | what it controls | default | effect of increasing |
|---|---|---|---|---|
| `total_len` | EpisodeSpec / `--mixed-ctx` | context tokens (compression numerator) | **2048** | more to compress (harder); 21:1 vs M=96 |
| `--mixed-M` | CLI | memory tokens M (compression denominator) | **96** | more capacity (easier); keeps capacity off the table |
| `window_size` | EpisodeSpec / `--window-size` | streaming-write chunk (≈ paragraph) | **256** | fewer/larger windows; also #continuation horizons |
| `pack_n_queries` | **Source** attr `(min,max)` | how many facts asked per episode | bio/babi (1,3), qa_multi (1,2) | more reads → more **addressing** pressure |
| `query_lag` | EpisodeSpec (`vary_lag` per task) | WHERE the queried fact sits | `"vary"` (early/recent/any) | "early" = long retention lag (hardest) |
| `n_inputs` | EpisodeSpec | MAX items sampled into the pool | 40 (bio pairs) | bigger pool to fill from (fill is budget-capped) |
| `predict_len` | EpisodeSpec / `--predict-len` | continuation block size per horizon | **64** | longer prediction from memory (harder) |
| `n_horizons` | EpisodeSpec | continuation boundaries scored | None (= all 8) | more horizons scored per episode |
| `mask_ratio` | EpisodeSpec / cfg | MAE fraction recalled from memory | **0.85** | higher = harder (less anchor-copying) |
| `pack_rename` | Source attr (bAbI) | co-packed segments get disjoint entities | True (bAbI) | — (correctness switch, not difficulty) |

**Key design point (why `pack_n_queries` is per-SOURCE, not per-task):** item size varies by source
(squad ~200 tok vs hotpot ~800 vs bio ~65 vs babi ~86), so one per-task constant can't fill 2048
meaningfully everywhere. Each source declares its own `(min,max)`; the task samples in-range per episode
and the packer feasibility-caps it (two big hotpot passages can't both fit if it would overflow). Old
per-task `n_queries` is gone; `EpisodeSpec.n_queries` remains only as a fallback for sources that don't
declare `pack_n_queries`.

**Worked example — "make an episode harder":** raise `total_len` (more to compress) OR lower `--mixed-M`
(less capacity) to tighten the ratio; set a source's `pack_n_queries` max higher (more simultaneous
reads); pin `query_lag="early"` (queried fact buried at the front, maximal retention interference); for
MAE, raise `mask_ratio` toward 0.9. Padding and segment count adjust automatically.

---

## Where to look in the code

- **what a task emits**: `src/memory/data/tasks/{reconstruction,qa,continuation,mae}.py` (all ~15-line
  adapters over `_pack.pack_streaming_episode`)
- **the packer** (causality, un-guessability, fill, multi-query): `src/memory/data/tasks/_pack.py`
- **per-source packing profile**: `Source.pack_n_queries` / `pack_rename` in `sources/*.py`
- **the knobs**: `src/memory/data/schedule.py` (`EpisodeSpec`) + `scripts/train/cli.py`
- **draw + inspect your own**: `python scripts/diagnostics/mixed/episode_peek.py --n-detail 20`
