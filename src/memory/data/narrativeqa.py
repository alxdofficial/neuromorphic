"""NarrativeQA — long-form QA over books and movie scripts (summaries-only).

EVAL reader. HF-auto-downloaded (`narrativeqa`) on first use; no build/ingest
step and no `data/narrativeqa/` on disk. See DATASETS.md.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import _pack_context, collate_qa


class NarrativeQADataset(IterableDataset):
    """NarrativeQA in the **summaries-only** setting (Kočiský et al., 2018, TACL).

    NarrativeQA ships two official tasks: 'summaries only' (the ~659-token
    human-written plot summary) and 'stories only' (the full book/script,
    ~60k tokens average, up to 400k+). The QA pairs were authored by
    annotators who read the SUMMARY, so the answer-bearing facts live in the
    summary — not at an arbitrary offset in the full story. We therefore use
    the summary as the gold context and pack OTHER stories' summaries as
    distractors to fill the chunk (mirrors HotpotQA/MuSiQue packing), so the
    supporting facts are guaranteed present and the example is a genuine
    retrieve-among-distractors read.

    (Earlier versions took a random `chunk_size` window of the full ~60k-token
    story — that is NEITHER official setting and leaves the answer facts
    absent ≈100% of the time; verified by data-sample audit 2026-06-01.)

    Answers are free-form abstractive (mean ~4.7 tokens, not document spans),
    so EM/containment under-credit them; the headline metric is the LLM judge
    with EM/containment as quick secondary signals (max-over-refs).

    First call downloads ~700MB from HuggingFace.
    """

    def __init__(
        self,
        split: str,                          # "train" or "validation"
        tokenizer,
        chunk_size: int = 4096,
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        from datasets import load_dataset
        print(f"[data v1h] loading NarrativeQA (summaries-only) split={split} "
              f"(downloads ~700MB on first call)...")
        self.data = load_dataset("narrativeqa", split=split, trust_remote_code=True)
        print(f"[data v1h]   {len(self.data):,} NarrativeQA {split} examples")

    def _summary_ids(self, ex) -> list:
        """Tokenize a story's human-written summary. Raises a clear error if
        the expected schema field is missing."""
        try:
            txt = ex["document"]["summary"]["text"]
        except (KeyError, TypeError) as e:  # pragma: no cover — schema guard
            raise KeyError(
                "NarrativeQA example missing document.summary.text — the HF "
                "`narrativeqa` schema changed. Summaries-only setting requires "
                "the summary field."
            ) from e
        return self.tokenizer(txt, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _doc_id(self, ex):
        return ex["document"]["id"]

    def _sample_distractor_summary(self, rng, avoid_doc_id: str) -> list:
        """A random OTHER story's summary, for distractor packing."""
        n = len(self.data)
        for _ in range(8):
            j = rng.randint(0, n - 1)
            ex_j = self.data[j]
            if self._doc_id(ex_j) == avoid_doc_id:
                continue
            toks = self._summary_ids(ex_j)
            if toks:
                return toks
        return []

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 17 + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        FILL_THRESHOLD = 0.95
        MAX_TOPUP_TRIES = 200

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            q_text = ex["question"]["text"]
            # Multiple reference answers; sample one for the training tensor,
            # surface ALL for multi-ref eval (scoring is max over references).
            ans_list = ex["answers"]
            if not ans_list:
                continue
            answer_refs = [a["text"] for a in ans_list if a.get("text")]
            if not answer_refs:
                continue
            a_text = rng.choice(answer_refs)

            gold = self._summary_ids(ex)
            if not gold:
                continue
            if len(gold) > self.chunk_size:        # ~659 tok; guard anyway
                gold = gold[:self.chunk_size]

            # Gold summary guaranteed in; top-up with other-story summaries as
            # distractors until ≥95% fill, inserting at random positions so the
            # gold summary isn't always first (positional fairness).
            doc_id = self._doc_id(ex)
            packed = [gold]
            cur_total = len(gold)
            tries = 0
            while (cur_total < FILL_THRESHOLD * self.chunk_size
                   and tries < MAX_TOPUP_TRIES):
                tries += 1
                d = self._sample_distractor_summary(rng, doc_id)
                if not d:
                    continue
                add = len(d) + 1  # plus separator
                if cur_total + add > self.chunk_size:
                    continue
                insert_at = rng.randint(0, len(packed))
                packed.insert(insert_at, d)
                cur_total += add

            ctx_tokens, valid_len = _pack_context(
                packed, chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id, pad_token_id=self.pad_token_id,
            )

            q_ids = tok(q_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            content_mask = [True] * len(a_ids)

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": "narrative_qa",
                "question_type": "narrative",
                "answer_refs": answer_refs,  # multi-ref for eval; ignored by collate
            }


def make_narrativeqa_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 4096,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone NarrativeQA loader (EVAL). Mirrors the mixed-sampler construction."""
    ds = NarrativeQADataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                            sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
