"""HotpotQA — distractor split, multi-hop questions over Wikipedia paragraphs.

EVAL reader. HF-auto-downloaded (`hotpot_qa`, distractor config) on first use;
no build/ingest step and no `data/hotpot/` on disk. See DATASETS.md.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import _pack_context, collate_qa


class HotpotQADataset(IterableDataset):
    """HotpotQA distractor config. Each example is a question + answer + 10
    Wikipedia paragraphs (mix of gold supporting paragraphs and distractors).
    We concatenate the 10 paragraphs, tokenize, and pack to `chunk_size`.

    Multi-hop questions force memory to combine information from multiple
    paragraphs — a meaningfully harder memory test than single-fact recall.

    First call downloads ~570MB from HuggingFace.
    """

    def __init__(
        self,
        split: str,                          # "train" or "validation"
        tokenizer,
        chunk_size: int = 4096,
        sep_token_id: int = 198,             # newline
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
        print(f"[data v1h] loading HotpotQA distractor split={split} "
              f"(downloads ~570MB on first call)...")
        self.data = load_dataset("hotpot_qa", "distractor", split=split,
                                  trust_remote_code=True)
        print(f"[data v1h]   {len(self.data):,} HotpotQA {split} examples")

    def _tokenize_paragraph(self, title: str, sents: list) -> list:
        text = title.strip() + " " + " ".join(s.strip() for s in sents)
        return self.tokenizer(text, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _sample_distractor_paragraph(self, rng, avoid_idx: int) -> list:
        """Pick a random paragraph from a random example (not avoid_idx)."""
        n = len(self.data)
        for _ in range(8):  # bounded retries to dodge degenerate cases
            j = rng.randint(0, n - 1)
            if j == avoid_idx:
                continue
            ex_j = self.data[j]
            ctx_j = ex_j["context"]
            titles = ctx_j["title"]
            sents_list = ctx_j["sentences"]
            if not titles:
                continue
            k = rng.randint(0, len(titles) - 1)
            return self._tokenize_paragraph(titles[k], sents_list[k])
        return []  # gave up

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        # Top-up loop targets ≥95% fill. Each candidate paragraph is
        # ~150-200 tokens; the loop keeps trying smaller paragraphs once
        # the budget tightens. MAX_TOPUP_TRIES bounds worst case.
        FILL_THRESHOLD = 0.95
        MAX_TOPUP_TRIES = 200

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            ctx = ex["context"]
            titles = ctx["title"]
            sents_list = ctx["sentences"]

            # Identify supporting (gold) paragraphs — these MUST stay in context.
            # supporting_facts has parallel lists `title` and `sent_id`. We mark
            # any paragraph whose title appears in supporting_facts as supporting.
            sf = ex.get("supporting_facts", {})
            support_titles = set(sf.get("title", []))
            is_support = [t in support_titles for t in titles]

            # Tokenize own paragraphs
            own_tok = [self._tokenize_paragraph(t, s)
                       for t, s in zip(titles, sents_list)]

            # Pack supporting first (guaranteed inclusion). If ANY support
            # paragraph won't fit chunk_size, this example is unsolvable —
            # skip it (resample) rather than yield an example whose answer
            # is not derivable from the visible context.
            packed = []
            cur_total = 0
            sep_cost = 1  # one sep_token between paragraphs
            order = sorted(range(len(own_tok)),
                           key=lambda i: not is_support[i])  # support first
            example_ok = True
            for i in order:
                tk = own_tok[i]
                add = len(tk) + (sep_cost if packed else 0)
                if cur_total + add > self.chunk_size:
                    if is_support[i]:
                        example_ok = False
                        break  # bail; this example is unsolvable
                    continue   # non-supporting can be skipped
                packed.append(tk)
                cur_total += add
            if not example_ok:
                continue  # outer while loop — resample a different example

            # Top-up: add random distractors from OTHER examples until full
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
                # Insert at random position to avoid front-loading
                insert_at = rng.randint(0, len(packed))
                packed.insert(insert_at, tk)
                cur_total += add

            ctx_tokens, valid_len = _pack_context(
                packed,
                chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id,
                pad_token_id=self.pad_token_id,
            )

            q_ids = tok(ex["question"], add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_text = ex["answer"]
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
                "task_family": "hotpot_qa",
                "question_type": ex.get("type", "comparison_or_bridge"),
                "answer_refs": [a_text],
            }


def make_hotpot_dataloader(
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
    """Standalone HotpotQA loader (EVAL). Mirrors the mixed-sampler construction."""
    ds = HotpotQADataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                         sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
