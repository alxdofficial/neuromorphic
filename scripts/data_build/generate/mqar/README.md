# `generate/mqar/` — MQAR (multi-query associative recall)

**Runtime-procedural — nothing to build here.** The MQAR source synthesizes random key→value
pairs on the fly per draw (Zoology-style), so there is no offline generation step and nothing is
stored under `data/mqar/`.

- **Source (load):** `src/memory/data/sources/mqar.py` (`MqarSource`, `kind="keyed"`).
- **Task:** pairs with the existing `reconstruction` task (multi-query ⇒ addressing).
- Keys/values are short random alnum strings (`rand_alnum`), un-guessable and BPE-round-trippable
  for exact-match scoring; `key_len` / `val_len` set the number of random chunks (≈ tokens).

To use it, just register-and-go — e.g. `SOURCE_REGISTRY["mqar"](tokenizer, key_len=2, val_len=2)`.
See `DATASETS.md` and `docs/data_arch_plan.md` (Layer L1).
