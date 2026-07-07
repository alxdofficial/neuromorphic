"""Sentence-pair compression data (compression-objective direction, 2026-06-12).

Unit = adjacent SENTENCE PAIR from FineWeb-EDU, 24 <= tokens <= 128, code size
k = ceil(len / ratio) (ratio 8 → k in [3,16]), bucketed by k so each batch is
uniform width. The local FineWeb-EDU parquet stores Llama input_ids only, so we
decode → text (Llama tok, cached once) → sentence-segment → re-tokenize with the
BACKBONE tokenizer (SmolLM2). Capacity is per-example (k), NOT fixed — the model
learns a compression POLICY, not one size.

Emits the QABatch contract so the existing reconstruction (AE) path consumes it
directly; adds `k_slots` (the per-example code size) for capacity-relative models
and a future true-MAE forward. AE here = teacher-forced reconstruct-the-pair;
true MAE (mask-infill, non-teacher-forced) is a separate decode path (TODO).

Data: `data/fineweb_edu/{train,val}.parquet` (+ `cache/`); ingest =
`scripts/data_build/ingest/fineweb.py` (TODO). See DATASETS.md.
"""
from __future__ import annotations

import math
import re
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa

REPO = Path(__file__).resolve().parents[3]
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


class LongPassageMAEDataset(IterableDataset):
    """Contiguous-passage MAE: yield one fixed-length contiguous FineWeb-EDU span
    per example (NOT sentence pairs). Same QABatch contract the sentence MAE emits,
    so `compute_masked_reconstruction_loss` consumes it unchanged — the MAE path
    masks positions internally; we add NO question/answer.

    Per example: pick a doc with >= ctx_len tokens, a random start offset, and emit
    the contiguous span as context_ids with a full (all-True) context_mask. k_slots
    is the FIXED memory budget M (the mixed-mode uniform interface), so the MAE path's
    capacity-relative slice `memory[:, :k]` keeps all M emitted tokens.

    Span loading mirrors data_continuation.ContinuationDataset: decode the parquet's
    SOURCE-tokenizer ids → text (cached) → re-tokenize with the BACKBONE tokenizer.
    """

    def __init__(self, parquet_path, tokenizer, *, src_tokenizer_name: str,
                 split: str, ctx_len: int = 1024, m_slots: int = 32,
                 seed: int = 0, n_items: int = 1_000_000, pad_token_id: int = 0):
        super().__init__()
        self.tok = tokenizer
        self.ctx_len = ctx_len
        self.m_slots = m_slots
        self.seed = seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id
        self.trigger_ids = tokenizer("Reconstruct the text above.",
                                     add_special_tokens=False).input_ids
        cache = _decode_cache(Path(parquet_path), split, src_tokenizer_name)
        self.docs = []
        n_total = 0
        # Whole docs are tokenized here (for the cache) then sliced to ctx_len at emit time, so a doc
        # longer than the tokenizer's model_max_length is expected and harmless — silence the noisy
        # "sequence longer than max length" warning so it can't mask real ones. Restored after.
        from transformers.utils import logging as _hf_logging
        _prev_verbosity = _hf_logging.get_verbosity()
        _hf_logging.set_verbosity_error()
        try:
            for line in open(cache):
                n_total += 1
                arr = np.asarray(self.tok(json.loads(line)["text"],
                                          add_special_tokens=False).input_ids, dtype=np.int64)
                if arr.shape[0] >= ctx_len:
                    self.docs.append(arr)
        finally:
            _hf_logging.set_verbosity(_prev_verbosity)
        if not self.docs:
            raise ValueError(
                f"No FineWeb-edu doc has >= {ctx_len} tokens in {parquet_path} after "
                f"re-tokenization. Lower --mixed-ctx.")
        print(f"[data_masked_reconstruction:long_passage] {split}: "
              f"{len(self.docs)}/{n_total} docs >= {ctx_len} tok; M={m_slots} "
              f"(ratio {ctx_len}/{m_slots} = {ctx_len // m_slots}:1)", flush=True)

    def _gen(self, rng: np.random.Generator) -> dict:
        d = self.docs[rng.integers(len(self.docs))]
        s = int(rng.integers(0, len(d) - self.ctx_len + 1))
        span = torch.tensor(d[s: s + self.ctx_len], dtype=torch.long)
        return {
            "context_ids": span,
            "context_mask": torch.ones(self.ctx_len, dtype=torch.bool),   # full span, no pad
            "question_ids": torch.tensor(self.trigger_ids, dtype=torch.long),
            "answer_ids": span.clone(),
            "answer_content_mask_list": [True] * self.ctx_len,
            "task_family": "masked_reconstruction", "question_type": "masked_reconstruction",
            "answer_refs": [], "k_slots": self.m_slots, "n_tokens": self.ctx_len,
        }

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi is not None else 0))
        for _ in range(self.n_items):
            yield self._gen(rng)


def _collate_long_passage(samples, pad_token_id):
    """Fixed-length contiguous spans → collate_qa; stamp the uniform k_slots."""
    batch = collate_qa(samples, pad_token_id)
    batch.k_slots = max(int(s["k_slots"]) for s in samples)   # uniform M
    batch.n_tokens = [int(s["n_tokens"]) for s in samples]
    return batch


def make_long_passage_mae_dataloader(tokenizer, batch_size: int, *, src_tokenizer_name: str,
                                     split: str = "train", ctx_len: int = 1024,
                                     m_slots: int = 32, seed: int = 0,
                                     pad_token_id: int = 0, num_workers: int = 2) -> DataLoader:
    """Contiguous-passage MAE loader (the mixed-mode compression task). Selectable
    alongside the sentence-pair MAE; both feed the same masked_reconstruction loss."""
    path = FINEWEB_TRAIN if split == "train" else FINEWEB_VAL
    ds = LongPassageMAEDataset(path, tokenizer, src_tokenizer_name=src_tokenizer_name,
                               split=split, ctx_len=ctx_len, m_slots=m_slots,
                               seed=seed, pad_token_id=pad_token_id)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: _collate_long_passage(s, pad_token_id))
