"""NIAH — classic depth-controlled needle-in-a-haystack (EVAL-ONLY, procedural).

EVAL reader. Fully procedural at runtime (no download, no `data/niah/`); the
haystack + needle(s) are synthesized here. See DATASETS.md.

Complements `ruler.py` (RULER's fixed-mid-context, KEYED multi-key
addressing: "the special magic number for {key} is {v}", asked BY key) with
the classic Kamradt-style probe: a single unkeyed needle
("The magic number is {N}.") whose INSERTION DEPTH is the controlled,
sweepable variable (0.0 = start .. 1.0 = end of context) — depth x length is
the standard NIAH heatmap axis, and it is saturated for frontier long-context
LLMs but NOT for fixed-footprint compressive memory, which can plausibly show
a recency/primacy bias RULER's random-position needle would average away.

Multi-needle mode (`n_needles > 1`) inserts K distinct un-keyed needles at
spread depths and asks for ALL K numbers (in appearance order) — a multi-ITEM
retention probe (capacity), as opposed to RULER's multi-KEY probe (addressing
precision). Single-needle NIAH is essentially saturated for strong LLMs, but
is included for completeness/regression; multi-needle is the discriminating
variant for a bounded-capacity memory.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa

# Neutral, needle-phrase-disjoint filler sentences (no "magic"/"number") so a
# naive substring search can't cheat off the filler. Distinct pool from
# ruler.py's (different eval axis; no reuse needed, no harm either).
_FILLER_SENTENCES = [
    "The old library smelled of dust and warm paper on quiet afternoons.",
    "Somewhere a dog barked twice and then settled back into silence.",
    "The tide came in slowly, erasing the footprints one by one.",
    "A single lamp lit the far end of the corridor all night.",
    "Traffic thinned out after the evening rush, and the street went calm.",
    "The bakery's ovens cooled slowly after the last loaf came out.",
    "Clouds drifted east over the valley without any real hurry.",
    "The old clock tower chimed, a little late, as it always did.",
    "Someone had left a bicycle leaning against the garden wall.",
    "The ferry crossed the strait twice a day, rain or shine.",
    "A handful of leaves spun once in the wind and settled again.",
    "The coffee shop on the corner opened early for the delivery trucks.",
    "Snow gathered quietly on the windowsill through the night.",
    "The orchard rows stretched out in even, sunlit lines.",
    "A kettle whistled somewhere down the hall and then went quiet.",
    "The museum's east wing was closed for repairs that whole season.",
    "Fishing boats bobbed gently against the worn wooden dock.",
    "The printer in the office hummed along, page after page.",
    "A pair of sparrows argued briefly over a crust of bread.",
    "The mountain road switched back on itself a dozen times.",
    "Streetlights flickered on one by one as the sky dimmed.",
    "The tailor measured twice, out of long habit, before cutting.",
    "Warm bread and cold butter were about all the cafe served.",
    "The commuter train was three minutes late, as usual.",
    "Ivy had climbed halfway up the old brick chimney.",
    "The night market sold lanterns, noodles, and secondhand books.",
    "A slow rain settled over the rooftops just after midnight.",
    "The harbor bell rang out once for every departing ship.",
    "Chalk dust drifted in the light near the classroom window.",
    "The garden gate creaked the same way it had for years.",
]

# A short, self-contained neutral multi-paragraph passage, used (tiled) as an
# alternate haystack style ("essay") to the shuffled-sentence pool. Original
# text written for this reader — no external copyright/attribution concerns.
_ESSAY = """
The history of the humble doorstop is, on reflection, a history of small
compromises. Before hinges were common, a door was often just a slab of wood
propped against an opening, held in place by whatever was heavy and nearby: a
stone, a sack of grain, a length of rope tied to a post. Once hinges
appeared, the door swung freely, and freedom, as always, created a new
problem — the door that swings can also swing shut at the worst possible
moment, and so people needed a way to hold it open.

Early solutions were opportunistic rather than designed. A brick, a folded
rag, a boot left just so under the edge. It was only later, as households
accumulated more small objects and more reasons to keep doors open — airing a
room, carrying dishes back and forth, watching a child in the next room —
that the doorstop became a category of object in its own right, sold rather
than improvised, shaped rather than found.

The materials tell their own story. Cast iron doorstops shaped like animals
or ships were common in the nineteenth century, heavy enough to hold a door
against a draft and decorative enough to be left in view rather than hidden
behind a curtain. Later, rubber wedges took over for their portability, small
enough to keep in a pocket or a drawer, silent against a hardwood floor.

None of this is remarkable on its own. But it is a useful reminder that a
great many household objects exist not because someone set out to invent
them, but because a small, recurring inconvenience eventually earned itself a
dedicated tool. The doorstop, the paperweight, the oven mitt — each one
began as an improvisation and ended as a category.
""".strip()


class NIAHDataset(IterableDataset):
    """Depth-controlled needle-in-a-haystack. Single needle
    ("The magic number is {N}.") by default; `n_needles > 1` switches to the
    multi-needle "recall them all" variant.
    """

    def __init__(
        self,
        split: str,                          # accepted for interface parity; ignored (synthetic)
        tokenizer,
        chunk_size: int = 8192,
        n_needles: int = 1,
        depth: Optional[float] = None,       # single-needle: fixed insertion depth in [0,1]; None = random/sample
        depths: Optional[list] = None,       # multi-needle: explicit depths (len == n_needles); None = spread+jitter
        haystack_style: str = "sentences",   # "sentences" (shuffled pool) | "essay" (tiled canned passage)
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        del split
        if n_needles < 1:
            raise ValueError(f"NIAH: n_needles must be >= 1, got {n_needles}")
        if depth is not None and not (0.0 <= depth <= 1.0):
            raise ValueError(f"NIAH: depth must be in [0,1], got {depth}")
        if depths is not None and len(depths) != n_needles:
            raise ValueError(f"NIAH: len(depths)={len(depths)} != n_needles={n_needles}")
        if haystack_style not in ("sentences", "essay"):
            raise ValueError(f"NIAH: haystack_style must be 'sentences'/'essay', got {haystack_style!r}")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.n_needles = n_needles
        self.depth = depth
        self.depths = list(depths) if depths is not None else None
        self.haystack_style = haystack_style
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

    def _filler_source(self, tok) -> list:
        """Pre-tokenized filler units to draw from (cycled/reshuffled by the
        caller — this just returns the fixed pool once)."""
        def _tk(s: str) -> list:
            return tok(s, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if self.haystack_style == "essay":
            # Split into sentences on ". " so units are small enough to place
            # a needle between any two of them (needed for depth control).
            parts = [p.strip() for p in _ESSAY.replace("\n", " ").split(". ") if p.strip()]
            parts = [p if p.endswith(".") else p + "." for p in parts]
            return [_tk(p) for p in parts]
        return [_tk(s) for s in _FILLER_SENTENCES]

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + 5231 + wid * 100_003)
        tok = self.tokenizer
        cs = self.chunk_size
        k = self.n_needles

        def _tok(s: str) -> list:
            return tok(s, add_special_tokens=False, return_attention_mask=False)["input_ids"]

        base_pool = self._filler_source(tok)

        while True:
            # Distinct magic numbers, one per needle.
            vals, seen = [], set()
            while len(vals) < k:
                v = rng.randint(1_000_000, 9_999_999)
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
            needle_ids = [_tok(f"The magic number is {v}.") for v in vals]
            needle_total = sum(len(n) for n in needle_ids) + len(needle_ids)

            # Fill the remaining budget with shuffled/cycled filler units so
            # the needle(s) always fit without truncation (mirrors ruler.py's
            # overshoot guard: stop before the last unit could push us past
            # chunk_size and clip a needle off the tail).
            budget = cs - needle_total - 8
            pool = list(base_pool)
            rng.shuffle(pool)
            filler_units, acc, pi = [], 0, 0
            while acc < budget:
                if pi >= len(pool):
                    rng.shuffle(pool)
                    pi = 0
                u = pool[pi]
                pi += 1
                if acc + len(u) + 1 > budget:
                    break
                filler_units.append(u)
                acc += len(u) + 1

            n_slots = len(filler_units) + 1  # insertion points: before unit 0 .. after the last unit

            # Resolve target depths -> distinct slot indices, in ASCENDING
            # depth order (so "appearance order" == "depth order").
            if k == 1:
                d0 = self.depth if self.depth is not None else rng.random()
                depth_fracs = [d0]
            elif self.depths is not None:
                depth_fracs = list(self.depths)
            else:
                # Evenly spread across the document with a little jitter so a
                # fixed loader doesn't always probe the exact same k offsets.
                jitter = 0.4 / k
                depth_fracs = [min(max((i + 0.5) / k + rng.uniform(-jitter, jitter), 0.0), 1.0)
                               for i in range(k)]

            order_idx = sorted(range(k), key=lambda i: depth_fracs[i])  # ascending-depth needle order
            used_slots = set()
            slot_for_rank: list[int] = []
            for rank, i in enumerate(order_idx):
                p = round(depth_fracs[i] * (n_slots - 1)) if n_slots > 1 else 0
                while p in used_slots and p < n_slots - 1:
                    p += 1
                p = min(max(p, 0), n_slots - 1)
                used_slots.add(p)
                slot_for_rank.append(p)
            slot_to_needle = {slot_for_rank[rank]: order_idx[rank] for rank in range(k)}

            units: list = []
            for i in range(n_slots):
                if i in slot_to_needle:
                    units.append(needle_ids[slot_to_needle[i]])
                if i < len(filler_units):
                    units.append(filler_units[i])

            ctx_ids: list = []
            for u in units:
                ctx_ids.extend(u)
                ctx_ids.append(self.sep_token_id)
            ctx_ids = ctx_ids[:cs]
            valid_len = len(ctx_ids)
            if valid_len < cs:
                ctx_ids = ctx_ids + [self.pad_token_id] * (cs - valid_len)

            appearance_vals = [vals[i] for i in order_idx]
            if k == 1:
                q_text = "What is the magic number? Answer with only the number."
                a_text = str(vals[0])
                d_bucket = int(round(depth_fracs[0] * 10)) * 10  # nearest decile, 0..100
                q_type = f"niah_single_d{d_bucket:03d}"
            else:
                q_text = (f"There are {k} magic numbers hidden in the text above. "
                          f"List all {k} of them, in the order they appear, separated by commas. "
                          f"Answer with only the numbers, comma-separated.")
                a_text = ", ".join(str(v) for v in appearance_vals)
                q_type = f"niah_multi_k{k}"

            q_ids = _tok(q_text)
            a_ids = _tok(a_text)

            yield {
                "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (cs - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": [True] * len(a_ids),
                "answer_refs": [a_text],
                "task_family": "niah",
                "question_type": q_type,
            }


def make_niah_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 8192,
    n_needles: int = 1,
    depth: Optional[float] = None,
    depths: Optional[list] = None,
    haystack_style: str = "sentences",
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone NIAH loader (EVAL-ONLY). Fully synthetic/procedural — no
    download. `n_needles=1` (default) = classic single-needle depth-sweep
    probe; `n_needles>1` = multi-needle "recall them all" capacity probe."""
    ds = NIAHDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                     n_needles=n_needles, depth=depth, depths=depths,
                     haystack_style=haystack_style, sep_token_id=sep_token_id,
                     pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
