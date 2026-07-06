"""RULER — synthetic multi-key needle-in-a-haystack (EVAL-ONLY, OOD).

EVAL reader. Fully procedural at runtime (no download, no `data/ruler/`);
the needles/filler are synthesized here. See DATASETS.md.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa


class RULERNIAHDataset(IterableDataset):
    """RULER-style multi-key needle-in-a-haystack (Hsieh et al., 2024, NVIDIA).

    EVAL-ONLY out-of-distribution probe — never trained on. K distinct "magic
    number" needles are hidden in benign filler; given one key, the model must
    retrieve its value. Fully synthetic → contamination-free (Llama cannot know
    the random values); tests pure associative recall at long context. The
    answer is the exact value, so EM/F1 are clean (no echo/abstraction).
    """

    _FILLER = [
        "The afternoon light moved slowly across the wooden floor.",
        "A train passed somewhere far away, and then the quiet returned.",
        "He counted the steps out of habit, not because it mattered.",
        "Rain had been promised all week, but the clouds never broke.",
        "The market was busy with the ordinary errands of a Tuesday.",
        "She folded the letter twice and left it on the table.",
        "Outside, the wind turned the pages of a forgotten newspaper.",
        "They spoke of small things, the weather and the price of bread.",
        "A door closed somewhere upstairs and the house settled again.",
        "The river was low that summer and the stones showed through.",
        "He sharpened the pencil to a fine point and set it down.",
        "The cat watched the birds without any real intention.",
        "Lamps came on one by one along the length of the street.",
        "Someone was practicing scales on a piano two floors below.",
        "The kettle clicked off and the steam thinned into nothing.",
        "Leaves gathered in the corner of the yard near the fence.",
        "The bus was late, as it usually was on rainy mornings.",
        "She traced the rim of the cup with one absent finger.",
        "The clock in the hall had been five minutes slow for years.",
        "A gull wheeled once over the harbor and was gone.",
        "The bookshelf sagged a little under its uneven weight.",
        "He left the radio on low, more for company than for news.",
        "The path curved away under the bare and patient trees.",
        "Footsteps crossed the ceiling and then went quiet.",
    ]
    _ADJ = ["silver", "quiet", "ancient", "golden", "hollow", "crimson",
            "northern", "restless", "frozen", "gentle", "distant", "amber",
            "velvet", "iron", "scarlet", "winding"]
    _NOUN = ["harbor", "lantern", "meadow", "compass", "willow", "cipher",
             "falcon", "garden", "ferry", "beacon", "orchard", "summit",
             "quartz", "river", "thicket", "anchor"]

    def __init__(
        self,
        split: str,                 # accepted for interface parity; ignored (synthetic)
        tokenizer,
        chunk_size: int = 8192,
        n_needles: int = 1,         # single needle: 4-needle/64:1 is unwinnable for compressive memory
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        del split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.n_needles = n_needles
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

    def _key(self, rng) -> str:
        return f"{rng.choice(self._ADJ)}-{rng.choice(self._NOUN)}"

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + 911 + wid * 100_003)
        tok = self.tokenizer

        def _tok(s: str) -> list:
            return tok(s, add_special_tokens=False,
                       return_attention_mask=False)["input_ids"]

        while True:
            # Ordered list (not a set) so key order is deterministic regardless
            # of PYTHONHASHSEED.
            keys, _seen = [], set()
            while len(keys) < self.n_needles:
                k = self._key(rng)
                if k not in _seen:
                    _seen.add(k)
                    keys.append(k)
            vals = [rng.randint(1_000_000, 9_999_999) for _ in keys]
            needles = [_tok(f"The special magic number for {k} is {v}.")
                       for k, v in zip(keys, vals)]
            needle_total = sum(len(n) for n in needles) + len(needles)

            # Fill benign filler up to ~chunk_size minus the needle budget, so
            # the needles fit without truncation. Guard against OVERSHOOT: a
            # final filler unit (~10-15 tok) could push the total past chunk_size
            # and clip the tail — which, if a needle landed last, would drop the
            # answer from the context. Stop before exceeding budget so the
            # post-insertion total stays ≤ chunk_size and no needle is truncated.
            budget = self.chunk_size - needle_total - 8
            filler_units, acc = [], 0
            while acc < budget:
                u = _tok(rng.choice(self._FILLER))
                if acc + len(u) + 1 > budget:
                    break
                filler_units.append(u)
                acc += len(u) + 1

            # Insert the needles at distinct random positions among the filler.
            n_slots = len(filler_units) + 1
            positions = sorted(rng.sample(range(n_slots),
                                          min(self.n_needles, n_slots)))
            units, ni = [], 0
            for i in range(n_slots):
                while ni < len(positions) and positions[ni] == i:
                    units.append(needles[ni])
                    ni += 1
                if i < len(filler_units):
                    units.append(filler_units[i])

            ctx_ids: list = []
            for u in units:
                ctx_ids.extend(u)
                ctx_ids.append(self.sep_token_id)
            ctx_ids = ctx_ids[:self.chunk_size]
            valid_len = len(ctx_ids)
            if valid_len < self.chunk_size:
                ctx_ids = ctx_ids + [self.pad_token_id] * (self.chunk_size - valid_len)

            # Target only from needles actually inserted (small chunks may fit
            # fewer than n_needles), so the answer is always present in context.
            target = rng.randrange(len(positions))
            # Explicit instruction: without it the Instruct model refuses the
            # unusual "magic number" framing ("I'm not aware of any...") and the
            # whole task scores 0 regardless of memory (verified 2026-05-30).
            q_ids = _tok(f"What is the special magic number for {keys[target]}? "
                         f"Answer with only the number.")
            a_text = str(vals[target])
            a_ids = _tok(a_text)

            yield {
                "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": [True] * len(a_ids),
                "answer_refs": [a_text],
                "task_family": "ruler_niah",
                "question_type": "niah_multikey",
            }


def make_ruler_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 8192,
    n_needles: int = 1,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone RULER NIAH loader (EVAL). Fully synthetic."""
    ds = RULERNIAHDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                          n_needles=n_needles, sep_token_id=sep_token_id,
                          pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
