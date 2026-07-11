"""FineWeb corpus source — contiguous natural-text token spans (shared by mae + continuation).

Source half of the old ``data/mae.py`` + ``data/continuation.py``: the DUPLICATED doc-loading that
both readers carried — decode the FineWeb-EDU parquet's SOURCE-tokenizer (Llama-3) ids back to text
(cached once via ``_decode_cache``), re-tokenize with the BACKBONE tokenizer (SmolLM2), and keep
docs long enough for the caller's window. Windowing / span placement is the Task's job; this Source
just yields whole re-tokenized documents as ``CorpusItem`` token arrays. **The dedup win.**

The local FineWeb-EDU parquet stores Llama-3 ``input_ids`` only, so feeding them raw to a different
backbone (e.g. SmolLM2) would be a token-identity mismatch — hence the decode → text (cached) →
re-tokenize firewall. ``min_len`` is the minimum re-tokenized doc length to keep: pass ``ctx_len``
for the mae task, ``compress_len + predict_len`` for continuation.

Data: ``data/fineweb_edu/{train,val}.parquet`` (+ ``cache/``); ingest =
``scripts/data_build/ingest/fineweb.py`` (TODO). See DATASETS.md / docs/history/docs/history/data_arch_plan.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Source, CorpusItem, REPO
from ._corpus import _tokenize_cached

FINEWEB_TRAIN = REPO / "data/fineweb_edu/train.parquet"
FINEWEB_VAL = REPO / "data/fineweb_edu/val.parquet"
TEXT_CACHE = REPO / "data/fineweb_edu/cache"   # decoded-text cache dir


def _decode_cache(parquet_path: Path, split: str, src_tokenizer_name: str) -> Path:
    """Decode the parquet's stored input_ids back to text ONCE, cache to jsonl.

    Keyed by source-tokenizer NAME (in the filename) PLUS a sidecar `.meta`
    fingerprint (name + vocab_size). SAFE FIX: a cache built with a *different*
    tokenizer — same registered name but a changed vocab — is regenerated rather
    than silently reused (which would decode ids against the wrong vocab and yield
    garbage text). Legacy caches that predate the fingerprint are adopted as-is
    (their filename already pins the tokenizer name), so a matching cache is never
    needlessly rebuilt."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = src_tokenizer_name.replace("/", "_")
    out = TEXT_CACHE / f"{split}.{key}.jsonl"
    meta = out.with_name(out.name + ".meta")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    src_tok = AutoTokenizer.from_pretrained(src_tokenizer_name)
    fingerprint = f"{src_tokenizer_name}|vocab={src_tok.vocab_size}"

    if out.exists():
        prior = meta.read_text().strip() if meta.exists() else None
        if prior == fingerprint:
            return out                                   # verified match → reuse
        if prior is None:
            # Legacy cache (no fingerprint). Filename already pins the tokenizer
            # NAME, so adopt this decode for the current fingerprint instead of
            # discarding a valid cache; write the meta going forward.
            meta.write_text(fingerprint)
            return out
        print(f"[data.mae] cache fingerprint changed ({prior!r} → {fingerprint!r}); "
              f"regenerating {out.name}", flush=True)

    tbl = pq.read_table(str(parquet_path), columns=["input_ids"])
    ids = tbl.column("input_ids").to_pylist()
    with open(out, "w") as fp:
        for doc in ids:
            text = src_tok.decode(doc, skip_special_tokens=True)
            fp.write(json.dumps({"text": text}) + "\n")
    meta.write_text(fingerprint)
    print(f"[data.mae] decoded {len(ids)} docs → {out.name}", flush=True)
    return out


class FinewebSource(Source):
    """Yields whole FineWeb-EDU documents (backbone-tokenized) as ``CorpusItem`` token arrays.

    Constructed once per split; the doc-loading (decode-cache → re-tokenize with the backbone tok →
    keep docs ≥ ``min_len``) runs in ``__init__``. ``sample`` draws docs uniformly with the Task's
    worker-seeded stdlib RNG (``rng.randrange``), matching how ``TaskDataset`` calls sources.
    """

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train",
                 src_tokenizer_name: str = "meta-llama/Llama-3.2-1B",
                 min_len: int = 1024, seed: int = 0, **kw):
        self.tok = tokenizer
        self.split = split
        self.min_len = min_len
        self.seed = seed
        path = FINEWEB_TRAIN if split == "train" else FINEWEB_VAL

        # The parquet stores SOURCE-tokenizer (Llama-3) ids. Feeding them raw to a different backbone
        # (e.g. SmolLM2) would be a token-identity mismatch, so decode → text (cached) → re-tokenize
        # with the BACKBONE tokenizer — the firewall the old mae/continuation readers shared.
        cache = _decode_cache(Path(path), split, src_tokenizer_name)
        # BACKBONE-tokenize the decoded text once and disk-cache the ids (npz), so this ~14k-doc
        # re-tokenization doesn't run on every construction (diagnostics / per-variant startup). Whole
        # docs are sliced to a window at emit time, so over-length docs are harmless — silence that warn.
        from transformers.utils import logging as _hf_logging
        _prev_verbosity = _hf_logging.get_verbosity()
        _hf_logging.set_verbosity_error()
        try:
            docs_all = _tokenize_cached(cache, self.tok,
                                        lambda: (json.loads(line)["text"] for line in open(cache)))
        finally:
            _hf_logging.set_verbosity(_prev_verbosity)
        n_total = len(docs_all)
        self.docs = [a for a in docs_all if a.shape[0] >= min_len]
        if not self.docs:
            raise ValueError(
                f"No FineWeb-edu doc has >= {min_len} tokens in {path} after re-tokenization. "
                f"Lower the task's total_len (mae) / compress_len+predict_len (continuation).")
        print(f"[data.fineweb] {split}: {len(self.docs)}/{n_total} docs >= {min_len} tok "
              f"from {Path(path).name}", flush=True)

    def sample(self, rng, n: int) -> list:
        return [CorpusItem(tokens=self.docs[rng.randrange(len(self.docs))]) for _ in range(n)]
