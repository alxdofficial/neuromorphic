"""Continuation / streaming-LM objective — the published-baselines' NATIVE task.

This is the AutoCompressors / Activation-Beacon / CCM family objective (gist, not binding):
take a contiguous span of natural text, **compress the first `compress_len` tokens into the
memory, throw those tokens away**, and **predict the next `predict_len` tokens** with the
decoder seeing ONLY the memory (closed-book continuation). CE on the continuation tokens.

Why it complements EMAT: AE/continuation are *unconditioned* (regenerate gist / continue) —
a membership-pool memory passes them, and the REAL≪SHUF signal comes for free because a
passage's continuation is passage-specific. EMAT is *addressed* (retrieve a specific value by
key) — the thing a pool can't do. So running both on the same arms separates "keeps gist"
(this task) from "stores addressable bindings" (EMAT). See [[project_emat_port_scale_fairness]].

Data: pre-tokenized FineWeb-edu docs (`data/fineweb_edu/{train,val}.parquet`, Llama-3 ids).
The predicted window sits *immediately after* the compressed cutoff, so the early continuation
tokens genuinely depend on the compressed context.

Emits the per-sample dict that `data_qa.collate_qa` consumes → reuses the whole
`compute_loss` path + REAL/SHUF/OFF gate unchanged; only the data differs.

  python -m src.memory.data.continuation     # smoke: print a rendered example

Data: `data/fineweb_edu/{train,val}.parquet` (shares `mae`'s `cache/`); ingest =
`scripts/data_build/ingest/fineweb.py` (TODO). See DATASETS.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa

REPO = Path(__file__).resolve().parents[3]
FINEWEB_TRAIN = REPO / "data/fineweb_edu/train.parquet"
FINEWEB_VAL = REPO / "data/fineweb_edu/val.parquet"


class ContinuationDataset(IterableDataset):
    """Infinite stream of (compress-span → predict-span) continuation examples from FineWeb-edu.

    Per example: pick a doc with ≥ compress_len+predict_len tokens, a random start offset, then
    context = the compress span (→ memory, then dropped), answer = the next predict_len tokens.
    """

    def __init__(self, parquet_path, tokenizer, *,
                 src_tokenizer_name: str = "meta-llama/Llama-3.2-1B",
                 split: str = "train", compress_len: int = 4096,
                 predict_len: int = 64, seed: int = 0, n_items: int = 1_000_000,
                 pad_token_id: int = 128_001, trigger: str = None,
                 objective: str = "continuation"):
        import json
        from .mae import _decode_cache
        assert objective == "continuation"
        self.tok = tokenizer
        self.compress_len = compress_len
        # continuation predicts the NEXT span after the compressed prefix.
        self.objective = objective
        self.predict_len = predict_len
        self.seed = seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id
        need = compress_len + predict_len
        if trigger is None:
            trigger = "Continue the passage."

        # The parquet stores SOURCE-tokenizer (Llama-3) ids. Feeding them raw to a
        # different backbone (e.g. SmolLM2) would be a token-identity mismatch, so
        # decode → text (cached) → re-tokenize with the BACKBONE tokenizer — the
        # same firewall data_masked_reconstruction uses.
        cache = _decode_cache(Path(parquet_path), split, src_tokenizer_name)
        self.docs = []
        n_total = 0
        # Silence the benign "sequence longer than model max length" warning: whole docs are
        # tokenized for the cache then sliced at emit time, so over-length is expected/harmless.
        from transformers.utils import logging as _hf_logging
        _prev_verbosity = _hf_logging.get_verbosity()
        _hf_logging.set_verbosity_error()
        try:
            for line in open(cache):
                n_total += 1
                arr = np.asarray(tokenizer(json.loads(line)["text"],
                                           add_special_tokens=False).input_ids, dtype=np.int64)
                if arr.shape[0] >= need:
                    self.docs.append(arr)
        finally:
            _hf_logging.set_verbosity(_prev_verbosity)
        if not self.docs:
            raise ValueError(
                f"No FineWeb-edu doc has ≥ {need} tokens (compress_len+predict_len) in "
                f"{parquet_path} after re-tokenization. Lower --compress-len/--predict-len.")
        # continuation cue (the chat-template user turn); the value to predict is raw document text.
        self.trigger_ids = tokenizer(trigger, add_special_tokens=False).input_ids
        print(f"[{objective}] {len(self.docs)}/{n_total} docs ≥ {need} tok from "
              f"{Path(parquet_path).name}; compress={compress_len} predict={self.predict_len} "
              f"(ratio {compress_len}/M memory tokens)", flush=True)

    def _gen(self, rng: np.random.Generator) -> dict:
        d = self.docs[rng.integers(len(self.docs))]
        span_need = self.compress_len + self.predict_len
        s = int(rng.integers(0, len(d) - span_need + 1))
        compress = d[s: s + self.compress_len]
        predict = d[s + self.compress_len: s + self.compress_len + self.predict_len]
        return {
            "context_ids": torch.tensor(compress, dtype=torch.long),
            "context_mask": torch.ones(self.compress_len, dtype=torch.bool),   # full span, no pad
            "question_ids": torch.tensor(self.trigger_ids, dtype=torch.long),
            "answer_ids": torch.tensor(predict, dtype=torch.long),
            "answer_content_mask_list": [True] * self.predict_len,             # predict every token
            "task_family": self.objective,
            "question_type": self.objective,
            "answer_refs": [],
        }

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi is not None else 0))
        for _ in range(self.n_items):
            yield self._gen(rng)


def make_continuation_dataloader(tokenizer, batch_size: int, compress_len: int = 4096,
                                 predict_len: int = 64, split: str = "train", seed: int = 0,
                                 pad_token_id: int = None, num_workers: int = 2,
                                 objective: str = "continuation",
                                 src_tokenizer_name: str = "meta-llama/Llama-3.2-1B") -> DataLoader:
    if pad_token_id is None:                                  # LLM-agnostic default
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    path = FINEWEB_TRAIN if split == "train" else FINEWEB_VAL
    ds = ContinuationDataset(path, tokenizer, src_tokenizer_name=src_tokenizer_name, split=split,
                             compress_len=compress_len, predict_len=predict_len,
                             seed=seed, pad_token_id=pad_token_id, objective=objective)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id))


if __name__ == "__main__":  # smoke
    import sys
    sys.path.insert(0, str(REPO))
    from transformers import AutoTokenizer
    from src.memory.config import ReprConfig
    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    ds = ContinuationDataset(FINEWEB_TRAIN, tok, compress_len=2048, predict_len=512, seed=1)
    s = next(iter(ds))
    print("context len:", s["context_ids"].numel(), "answer len:", s["answer_ids"].numel())
    print("compress tail:", repr(tok.decode(s["context_ids"][-40:])))
    print("predict head :", repr(tok.decode(s["answer_ids"][:40])))
