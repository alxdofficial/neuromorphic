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
"""
from __future__ import annotations

import math
import re
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .data_qa import collate_qa

REPO = Path(__file__).resolve().parents[2]
FINEWEB_TRAIN = REPO / "data/fineweb_edu/train.parquet"
FINEWEB_VAL = REPO / "data/fineweb_edu/val.parquet"
TEXT_CACHE = REPO / "data/fineweb_edu/text_cache"   # decoded-text cache dir

_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\d])')


def _segment(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _decode_cache(parquet_path: Path, split: str, src_tokenizer_name: str) -> Path:
    """Decode the parquet's stored input_ids back to text ONCE, cache to jsonl.
    Keyed by source tokenizer so a tokenizer change invalidates the cache."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = src_tokenizer_name.replace("/", "_")
    out = TEXT_CACHE / f"{split}.{key}.jsonl"
    if out.exists():
        return out
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    src_tok = AutoTokenizer.from_pretrained(src_tokenizer_name)
    tbl = pq.read_table(str(parquet_path), columns=["input_ids"])
    ids = tbl.column("input_ids").to_pylist()
    with open(out, "w") as fp:
        for doc in ids:
            text = src_tok.decode(doc, skip_special_tokens=True)
            fp.write(json.dumps({"text": text}) + "\n")
    print(f"[data_masked_reconstruction] decoded {len(ids)} docs → {out.name}", flush=True)
    return out


class SentencePairDataset(IterableDataset):
    def __init__(self, parquet_path, tokenizer, *, src_tokenizer_name: str,
                 split: str, batch_size: int, ratio: int = 8, min_len: int = 24,
                 max_len: int = 128, seed: int = 0, n_items: int = 1_000_000,
                 pad_token_id: int = 0, trigger: str = "Reconstruct the text above."):
        self.tok = tokenizer
        self.batch_size = batch_size
        self.ratio = ratio
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id
        cache = _decode_cache(Path(parquet_path), split, src_tokenizer_name)
        self.texts = [json.loads(l)["text"] for l in open(cache)]
        self.trigger_ids = tokenizer(trigger, add_special_tokens=False).input_ids
        print(f"[data_masked_reconstruction] {split}: {len(self.texts)} docs; pair, ratio {ratio}, "
              f"{min_len}-{max_len} tok", flush=True)

    def k_slots(self, length: int) -> int:
        return int(math.ceil(length / self.ratio))

    def _gen(self, rng) -> dict | None:
        text = self.texts[rng.integers(len(self.texts))]
        sents = _segment(text)
        if len(sents) < 2:
            return None
        i = int(rng.integers(len(sents) - 1))
        ids = self.tok(sents[i] + " " + sents[i + 1], add_special_tokens=False).input_ids
        n = len(ids)
        if n < self.min_len or n > self.max_len:
            return None
        k = self.k_slots(n)
        span = torch.tensor(ids, dtype=torch.long)
        cm = torch.ones(n, dtype=torch.bool)
        return {
            "context_ids": span, "context_mask": cm,
            "question_ids": torch.tensor(self.trigger_ids, dtype=torch.long),
            "answer_ids": span.clone(),
            "answer_content_mask_list": [True] * n,
            "task_family": "masked_reconstruction", "question_type": "masked_reconstruction",
            "answer_refs": [], "k_slots": k, "n_tokens": n,
        }

    def __iter__(self):
        """Yields BATCHES (use DataLoader batch_size=None). Length-bucketed: one
        buffer per k; emit a full uniform-k batch when a buffer fills, so every
        batch shares a single code size M=k (capacity-relative invariant)."""
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi is not None else 0))
        buffers: dict[int, list] = {}
        produced = 0
        while produced < self.n_items:
            ex = self._gen(rng)
            if ex is None:
                continue
            buf = buffers.setdefault(ex["k_slots"], [])
            buf.append(ex)
            if len(buf) == self.batch_size:
                produced += 1
                yield _collate_bucketed(buf, self.pad_token_id)
                buffers[ex["k_slots"]] = []


def _collate_bucketed(samples, pad_token_id):
    """Pad context to the batch-max (within a k-bucket, k*ratio); reuse collate_qa
    for the rest. context_ids must be equal length to stack."""
    T = max(int(s["context_ids"].shape[0]) for s in samples)
    for s in samples:
        n = int(s["context_ids"].shape[0])
        if n < T:
            s["context_ids"] = torch.cat(
                [s["context_ids"], torch.full((T - n,), pad_token_id, dtype=torch.long)])
            s["context_mask"] = torch.cat(
                [s["context_mask"], torch.zeros(T - n, dtype=torch.bool)])
    batch = collate_qa(samples, pad_token_id)
    batch.k_slots = max(int(s["k_slots"]) for s in samples)   # bucket code size
    batch.n_tokens = [int(s["n_tokens"]) for s in samples]
    return batch


def make_sentence_dataloader(tokenizer, batch_size: int, *, src_tokenizer_name: str,
                             split: str = "train", ratio: int = 8, min_len: int = 24,
                             max_len: int = 128, seed: int = 0, pad_token_id: int = 0,
                             num_workers: int = 2) -> DataLoader:
    path = FINEWEB_TRAIN if split == "train" else FINEWEB_VAL
    ds = SentencePairDataset(path, tokenizer, src_tokenizer_name=src_tokenizer_name,
                             split=split, batch_size=batch_size, ratio=ratio,
                             min_len=min_len, max_len=max_len, seed=seed,
                             pad_token_id=pad_token_id)
    # the dataset yields fully-collated, length-bucketed batches (batch_size=None)
    return DataLoader(ds, batch_size=None, num_workers=num_workers)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    from transformers import AutoTokenizer
    BACKBONE = "HuggingFaceTB/SmolLM2-135M"
    SRC = "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    dl = make_sentence_dataloader(tok, batch_size=8, src_tokenizer_name=SRC,
                                  split="val", num_workers=0)
    it = iter(dl)
    b = next(it)
    print("context", tuple(b.context_ids.shape), "k_slots", b.k_slots,
          "n_tokens", b.n_tokens)
    print("sample 0:", repr(tok.decode(b.context_ids[0][b.context_mask[0]])))
    # length/k distribution over a few batches
    ks, ns = [], []
    for _ in range(40):
        b = next(it); ks.append(b.k_slots); ns.extend(b.n_tokens)
    import numpy as np
    print(f"over 40 batches: n_tokens p50={int(np.percentile(ns,50))} "
          f"p90={int(np.percentile(ns,90))}; k range {min(ks)}-{max(ks)}")
