# Phase-2 baseline harness — audit #2 fixes (2026-07-18) — ✅ ALL DONE

Second external audit of the Tier-1/Tier-2 baseline harness. All 20 findings verified against code
(+ spot-checked vs cached LongMemEval data). Zero false alarms. **All 20 fixed; 65 tests pass
(34 new); real-data smoke + offline runner-integration validated.**

> **Test counts in this file (65 / 83 / 86) are HISTORICAL SNAPSHOTS at each audit** — they are not the
> current suite size. As of 2026-07-21 the suite is **133** (`.venv/bin/python -m pytest tests/ -q`).

Legend: `[x]` done · **(J)** = scoring-policy/fidelity JUDGMENT call (not pure mechanical).

Official prompts pulled VERBATIM at pinned commits (LongMemEval@9e0b455 src/generation/run_generation.py;
MemoryAgentBench@455306d utils/templates.py) so #1/#7 reproduce the benchmarks exactly, not paraphrased.

## Batch A — Tier-1 fidelity (BLOCKS the real run)
- [x] #1  Inject `question_date` into LongMemEval prompts + carry field through `load_longmemeval_text`.
- [x] #2  **(J)** Token-accurate `full_context` budgeting (Llama-3 tokenizer count, stop over-truncating
        histories that fit the served window). `baselines.py` / `run_api_eval.py`.
- [x] #5  **(J)** Preserve preference subtype so it ROUTES to the preference scorer (stop collapsing
        `-preference`→`single-session`). `longmemeval.py` + `longmemeval_score.py`.
- [x] #6  **(J)** Report at the paper's type granularity (don't merge the 3 single-session subtypes);
        document that abstention is reported separately (defensible reporting choice).
- [x] #7  **(J)** Per-competency MAB system prompts faithful to the MAB repo (ICL numeric-only; conflict
        "larger serial = newer"; detective strict letter; eventqa/ruler retrieval). `baselines.py` MAB path.

## Batch B — Tier-1 scoring precision
- [x] #3  **(J)** Whitespace-guard the `/` gold-alternate split (stop corrupting `1/48`, dates, ratios);
        keep paren + " or "; lean on BEM for verbose golds. `longmemeval_score.py`.
- [x] #4  **(J)** Bare-number containment false-positive: strict-numeric handling + documented caveat.
- [x] #10 `finish_reason=="length"` is retained for completion-rate QC. **Superseded 2026-07-20:**
        capped text is a model prediction under the fixed output budget and is now scored/cached; only
        provider/execution failures are excluded and retried. A larger cap has a distinct run signature.

## Batch C — Tier-1 harness bookkeeping
- [x] #9  Surface COVERAGE prominently (n_scored/n_total) + report strict-accuracy alongside; distinguish
        transient (retryable) from permanent (400) errors. `run_api_eval.py` / `report.py`.
- [x] #11 Score ONLY currently-selected item IDs, not the whole store (cache reuse stays; scope is exact).
        `run_api_eval.py` `_score_records`/`run_one`.
- [x] #12 Thread `competency` through store.append + `_score_records` → fix `competency_averaged_accuracy`
        (was all "unknown"). `run_api_eval.py`.
- [x] #13 MAB loader: raise on 0 items; record which splits loaded/failed (surface partial coverage).
        `memoryagentbench.py` + runner guard.
- [x] #14 Drop recsys from DEFAULT MAB load (unscoreable recall@5 → wasted API spend). `memoryagentbench.py`.
- [x] #15 Deterministic STRATIFIED sampling for `--max-examples` (representative smoke runs, not first-N).

## Batch D — Tier-2 hardening (deferred; off critical path, fix while here)
- [x] #8  M+ full per-item state reset (LTM/keys/ages/caches, not just `model.memory`). `run_memoryllm.py`.
- [x] #16 SnapKV/H2O: use instruct chat template + inject question_date. `run_kvcompress.py`.
- [x] #17 KVzip: actually pass `--max-new-tokens` into generation. `run_kvcompress.py`.
- [x] #18 M+: seed py/np/torch/cuda for determinism. `run_memoryllm.py`.
- [x] #19 Tier-2 incremental recovery (write per-item via ResultStore, not batch-at-end). both runners.
- [x] #20 Tier-2 artifact names carry capacity/ratio/gen-cap/variant/n/commit. both runners.

## Re-verification round (5 parallel reviewers over the diffs) — 8 MORE bugs found + fixed
The adversarial pass over the fixes themselves caught real defects the fixes introduced:
- **[HIGH] budget==0 negative-zero slice**: `fit_history`/`_truncate_tail` with a 0 budget returned the WHOLE
  history (`ids[-0:]`==all). Reachable on ≤6000-ctx models. Guarded → keep nothing.
- **[HIGH] unsized char fallback**: live `run_one` passed `token_budget` but not `char_budget` → if the gated
  Llama tokenizer is unavailable, fell back to a flat 440k chars for every model. Now threads
  `char_budget_for(ctx_len[model])`.
- **[MED] MAB stratification grouped by fine-grained `source`** → small `--max-examples` = 100%
  Accurate_Retrieval. Now nested competency→source round-robin + per-competency cap.
- **[MED] #3 `" or "` over-split** verb-phrase disjunctions ("I have worked on OR bought…" → "I have worked
  on") → real false-positive in the data. Now only splits SHORT (≤6-tok) alternations; tail-strip gated to
  acceptability golds + `\b`.
- **[MED] #4 over-rejected enumerated-but-correct** (gold "3" in a list containing "3 projects"). Now strips
  list markers and re-tests: accept if the number survives, reject only if it was purely a list index.
- **[MED] Tier-2**: kvcompress cache tag omitted `--seed` (seed collisions); snapshot counted a `None` LTM
  attr as "captured" (muted the leak warning); kvzip `max_new_tokens` kwarg now guarded (TypeError fallback);
  tier-2 `finish_reason` now detects length-truncation for the retry path.

## Audit #3 (third external review) — Batches A/B/C fixed + tested (83 tests)
A No-Go review found production-path scoring/faithfulness defects the earlier audits missed (they covered
internal-consistency + protocol-fidelity; this one went into containment precision + actual upstream-repo
behavior). All addressed:
- **A — Tier-1 scoring (blocked LongMemEval):** **#1** containment negation-guard (`_negated`) — a wrong
  answer stated as "3, not 2" / "London, not Paris" no longer scores correct (fired before BEM); + drop
  note-style parenthetical candidates ("(including the last day)"). **#2** preference polarity — exclude
  "avoid/not/instead-of" concepts from the positive keys + flag preference `low_confidence`. **#3** explicit
  `scoring_note` disclaiming the non-official metric. *(Real-data: 470/470 oracle golds still self-match.)*
- **B — MAB coverage:** **#4** per-SOURCE cap (was per-competency → only 1 source/competency in bounded
  runs) so smoke samples span variants. **#5** faithful `recall_at_k` (verbatim port of the repo's Recsys
  algorithm — parse→fuzzy-snap→Recall@k, pure-python Levenshtein fallback) as an OPT-IN utility;
  default remains 4/5 competencies with a `coverage_note` disclosure.
- **C — Tier-2/2b upstream faithfulness:** **#6** real M+ mutable field names (`ltm_recall_frequencies`,
  `cached_dropped_*`). **#7** 512-token injection blocks (was whole-session → far harsher compression than
  M+ was evaluated at). **#8** LCLM new-token slicing (`outputs[input_len:]`) instead of the prompt-echo-
  prone `generate_text` string-strip. **#9** A-MEM `OPENAI_BASE_URL` env routing (its openai backend drops
  `api_base`). **#10** KVzip pod-verify note (pass-through is a no-op if upstream accept-and-ignores).

## Audit #4 (fourth external review) — fixed + open items (86 tests)
Code fixes:
- **LCLM empty-answer bug (my audit-3 #8 was wrong):** LCLM generates via `inputs_embeds`, so HF `generate`
  returns ONLY new tokens — the unconditional `[input_len:]` slice removed the whole answer. Now slice ONLY
  if the sequence is longer than the input (`run_lclm._generate_new_tokens`); + length-cutoff detection.
- **Dense retrieval re-embedding:** `DenseRetriever` now caches passage embeddings per context (~36 unique
  MAB contexts were re-encoded per question → ~1.4M encodings; now ~16k).
- **Tier-2 length cutoffs are tracked separately from score coverage.** Superseded 2026-07-20: emitted text
  is scored; `eos_completion_rate` records how often generation terminated naturally.
- **MAB source-filter under-fill:** per-source cap is relaxed to `max_examples` when `--sources` narrows the
  set (was `ceil(max_examples/12)` → a single-source filter returned only 9).
- **2b fidelity:** A-MEM now passes each session's REAL date (not wall-clock) to `add_note`; MemoryOS state
  keyed by question_id + cleared per item (no dup-on-retry). Both retrieval-flow/turn-flattening deviations
  documented in-file.
- **Provenance:** Tier-2 artifacts now record `upstream_commit` (the cloned baseline-repo HEAD) + `scorer`.

Open items — NOT code-fixable here (setup / pod / user):
- ✅ **RESOLVED (2026-07-20) — KVzip generation cap:** upstream exposes generation controls through
  `ModelKVzip.gen_kwargs`; the runner now sets and records `max_new_tokens` there instead of passing an
  unsupported `generate()` keyword.
- ✅ **RESOLVED (2026-07-20) — `meta-llama/Llama-3.1-8B-Instruct` gate** accepted on the HF token (account
  `alxd219p1`, download verified); SnapKV/H2O unblocked. Ungated Mistral remains a fallback.
- **Tier-2 env isolation:** KVzip / KVCache-Factory / MemoryLLM / LCLM need mutually incompatible
  torch/transformers/numpy/python pins → run each in its OWN env (micromamba per method, baked into
  `scripts/pod/tier2_bootstrap.sh`).
- ✅ **RESOLVED (2026-07-20) — Phase-2 pod path** exists: `scripts/pod/tier2_bootstrap.sh` (+ `tier2_pod.py`)
  provision the eval envs directly and sync/pin the repo per run (the old Phase-0 `bootstrap.sh` is separate).
- **Panel-A matched-135M runner + Tier-2-on-MemoryAgentBench matrix** still to build (tasks #212).

## Verify
- [x] Regression tests per fix; full suite green (65 pass; +34 new across test_audit2_fixes,
      test_run_api_eval_integration, and the earlier audit-1 files).
- [x] Offline runner-integration test (stub client) covering the async `run_one` path: coverage/error/
      cutoff accounting, selection-scoped scoring, competency threading, resume.
- [x] LongMemEval LIVE smoke (cached data): visual-inspected date injection, stratified type spread, and
      that a real 506k-char history is NO LONGER truncated at a 131k budget (was, pre-#2); real-gold scoring
      of the `7 days / 8 days` verbose gold + the `gold=2` enumeration false-positive.
- [~] MAB live smoke NOT run (dataset not cached locally → would download); MAB prompts, scoring, competency
      routing, recsys-drop, and stratification are covered by unit + integration tests instead.
- [x] Adversarial re-verification workflow over the diffs (parallel read-only reviewers).
