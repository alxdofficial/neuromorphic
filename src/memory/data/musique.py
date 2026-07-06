"""MuSiQue-Ans — 2-4 hop QA with contamination-controlled chains.

EVAL reader. HF-auto-downloaded (`dgslibisey/MuSiQue`) on first use; no
build/ingest step and no `data/musique/` on disk. See DATASETS.md.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import _pack_context, collate_qa


class MuSiQueDataset(IterableDataset):
    """MuSiQue-Ans (Trivedi et al., 2022). 2-4 hop questions constructed by
    chaining single-hop questions with answerability filters — wider
    human-machine gap than HotpotQA's shortcut-friendly setup. Same
    Wikipedia substrate, so the contrast diagnoses shortcut learning.

    Each example: 20 paragraphs (mix of supporting + distractor) + a
    multi-hop question + answer (+ answer_aliases for liberal scoring).
    Packing mirrors HotpotQA: supporting first (guaranteed inclusion),
    own distractors fill, cross-example top-up to ≥95% fill.
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
        print(f"[data v1h] loading MuSiQue split={split} "
              f"(downloads ~120MB on first call)...")
        # MuSiQue-Ans is hosted on HF as `dgslibisey/MuSiQue` (answerable
        # only) and provides 'train' / 'validation' splits.
        self.data = load_dataset("dgslibisey/MuSiQue", split=split)
        # Filter to answerable rows only (some splits include unanswerable
        # examples — those are useful but a different framing). Many
        # downstream papers report on the answerable subset.
        if "answerable" in self.data.column_names:
            self.data = self.data.filter(lambda ex: ex["answerable"])
        print(f"[data v1h]   {len(self.data):,} MuSiQue {split} examples (answerable)")

    def _tokenize_paragraph(self, title: str, text: str) -> list:
        full = (title.strip() + " " + text.strip()).strip()
        return self.tokenizer(full, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _sample_distractor_paragraph(self, rng, avoid_idx: int) -> list:
        n = len(self.data)
        for _ in range(8):
            j = rng.randint(0, n - 1)
            if j == avoid_idx:
                continue
            ex_j = self.data[j]
            ps = ex_j["paragraphs"]
            if not ps:
                continue
            p = rng.choice(ps)
            return self._tokenize_paragraph(p["title"], p["paragraph_text"])
        return []

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 23 + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        FILL_THRESHOLD = 0.95
        MAX_TOPUP_TRIES = 200

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            paragraphs = ex["paragraphs"]
            # is_supporting flags supporting paragraphs (guaranteed include).
            own_tok = [self._tokenize_paragraph(p["title"], p["paragraph_text"])
                       for p in paragraphs]
            is_support = [bool(p["is_supporting"]) for p in paragraphs]

            # Pack supports first; skip example if any support won't fit
            # (would yield an unsolvable QA pair otherwise).
            packed = []
            cur_total = 0
            sep_cost = 1
            order = sorted(range(len(own_tok)),
                           key=lambda i: not is_support[i])
            example_ok = True
            for i in order:
                tk = own_tok[i]
                add = len(tk) + (sep_cost if packed else 0)
                if cur_total + add > self.chunk_size:
                    if is_support[i]:
                        example_ok = False
                        break
                    continue
                packed.append(tk)
                cur_total += add
            if not example_ok:
                continue  # outer while — resample

            tries = 0
            while (cur_total < FILL_THRESHOLD * self.chunk_size
                   and tries < MAX_TOPUP_TRIES):
                tries += 1
                tk = self._sample_distractor_paragraph(rng, avoid_idx=idx)
                if not tk:
                    continue
                add = len(tk) + sep_cost
                if cur_total + add > self.chunk_size:
                    continue
                insert_at = rng.randint(0, len(packed))
                packed.insert(insert_at, tk)
                cur_total += add

            ctx_tokens, valid_len = _pack_context(
                packed, chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id, pad_token_id=self.pad_token_id,
            )

            q_ids = tok(ex["question"], add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_text = ex["answer"] or ""
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            content_mask = [True] * len(a_ids)
            aliases = ex.get("answer_aliases") or []
            answer_refs = [a_text] + [a for a in aliases if a and a != a_text]

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": "musique",
                "question_type": "multi_hop_ans",
                "answer_refs": answer_refs,
            }


def make_musique_dataloader(
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
    """Standalone MuSiQue loader (EVAL). Mirrors the mixed-sampler construction."""
    ds = MuSiQueDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                        sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
