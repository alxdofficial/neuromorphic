"""Strict EMAT data — verbatim key→value reconstruction (single + multi).

This is the canonical EMAT objective (NOT QA): the context the encoder ingests is a
list of N ``key = value`` pairs (you input many *values*); at decode time the frozen LM
starts from the memory, is conditioned on a **verbatim key**, and must **autoregressively
reproduce that key's value** — the exact span it was given, absent from the decoder input.
CE is taken on the value tokens only. The value is *one of the inputs*, reproduced; this is
conditional reconstruction, not an answer reasoned out.

  single-EMAT : reconstruct ONE value from one key.
  multi-EMAT  : condition on several keys in sequence and reproduce each of their values.

Shortcut-proofing is the **random pairing** of real, single-token common words (per the
Stage-A dossier: NOT coined gibberish — BPE shreds those into off-distribution subword soup;
the random `barn = ocean` binding has no semantic reason to exist → it can only come from the
memory). Keys and values are drawn from DISJOINT word pools.

Emits the exact per-sample dict that `data_qa.collate_qa` consumes, so the whole
`ReprLearningModel.compute_qa_loss` path (encoder → memory → prepend → frozen-LM CE on
`answer_content_mask`) and the REAL/SHUF/OFF binding gate are reused unchanged — only the
*data* differs from the composite-QA pipeline.

  python -m src.repr_learning.data_emat        # smoke: print a few rendered EMAT examples
"""
from __future__ import annotations

import random
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .data_qa import collate_qa

# ── common, concrete, mostly single-token English words ───────────────────────
# Filtered at init to those that are EXACTLY one token under the active tokenizer
# (with a leading space, the Llama-3 " word" form), then split disjointly into a
# KEY pool and a VALUE pool. The list is long so each pool stays ≳150 after filtering.
COMMON_WORDS: List[str] = [
    "apple", "river", "mountain", "table", "window", "garden", "bottle", "candle",
    "pencil", "ticket", "pocket", "ladder", "basket", "carpet", "engine", "anchor",
    "pillow", "mirror", "wallet", "button", "needle", "saddle", "kettle", "hammer",
    "marble", "pepper", "ribbon", "shadow", "valley", "meadow", "forest", "desert",
    "island", "harbor", "bridge", "tunnel", "castle", "temple", "palace", "cabin",
    "ocean", "planet", "comet", "rocket", "signal", "beacon", "lantern", "compass",
    "copper", "silver", "marble", "granite", "crystal", "diamond", "amber", "ember",
    "thunder", "breeze", "winter", "summer", "autumn", "season", "morning", "evening",
    "tiger", "falcon", "salmon", "rabbit", "donkey", "beaver", "badger", "otter",
    "spider", "beetle", "cricket", "sparrow", "pigeon", "turtle", "dolphin", "walrus",
    "violin", "guitar", "trumpet", "drum", "flute", "piano", "banjo", "cello",
    "doctor", "farmer", "sailor", "baker", "tailor", "miner", "painter", "writer",
    "barrel", "bucket", "shovel", "wrench", "anvil", "pliers", "chisel", "nozzle",
    "cotton", "linen", "velvet", "denim", "leather", "rubber", "plastic", "ceramic",
    "lemon", "cherry", "banana", "melon", "grape", "peach", "pumpkin", "carrot",
    "onion", "potato", "ginger", "garlic", "walnut", "almond", "pepper", "honey",
    "meadow", "canyon", "glacier", "lagoon", "prairie", "swamp", "tundra", "savanna",
    "junction", "station", "market", "factory", "library", "museum", "stadium", "theater",
    "blanket", "curtain", "cushion", "mattress", "drawer", "cabinet", "shelf", "closet",
    "magnet", "battery", "circuit", "switch", "sensor", "motor", "piston", "valve",
    "feather", "antler", "talon", "whisker", "scale", "flipper", "tusk", "hoof",
    "bishop", "knight", "castle", "wizard", "dragon", "goblin", "phantom", "giant",
    "harvest", "festival", "carnival", "parade", "banquet", "voyage", "journey", "summit",
    "glacier", "boulder", "pebble", "gravel", "quartz", "basalt", "marble", "slate",
    "cobra", "viper", "gecko", "iguana", "lizard", "newt", "toad", "frog",
    "eagle", "raven", "heron", "stork", "crane", "swan", "goose", "duck",
    "maple", "willow", "cedar", "birch", "spruce", "poplar", "aspen", "juniper",
    "ribbon", "buckle", "zipper", "collar", "sleeve", "pocket", "cuff", "hem",
    "kettle", "skillet", "spatula", "ladle", "whisk", "grater", "strainer", "tongs",
    "compass", "sextant", "telescope", "barometer", "gauge", "dial", "lever", "crank",
]


def _word_to_id(tokenizer, word: str) -> Optional[int]:
    """Token id of ' word' iff it is a single token under this tokenizer, else None."""
    ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


class EMATDataset(IterableDataset):
    """Infinite stream of strict EMAT examples at a fixed (n_pairs, n_query, value_len).

    Per example: sample ``n_pairs`` distinct keys and (random, with replacement) values
    → render the context ``k = v\\n`` lines → pick ``n_query`` keys to recall.
    """

    def __init__(self, tokenizer, context_len: int, n_pairs: int = 64,
                 n_query: int = 1, value_len: int = 1, seed: int = 0,
                 n_items: int = 1_000_000, pad_token_id: int = 128_001):
        self.tok = tokenizer
        self.context_len = context_len
        self.n_pairs = n_pairs
        self.n_query = n_query
        self.value_len = value_len
        self.seed = seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id

        words, seen = [], set()
        for w in COMMON_WORDS:                       # de-dupe (the list has intentional repeats)
            if w in seen:
                continue
            seen.add(w)
            if _word_to_id(tokenizer, w) is not None:
                words.append(w)
        h = len(words) // 2
        self.key_words = words[:h]                   # disjoint pools: a value can't be read as a key
        self.val_words = words[h:]
        assert n_pairs <= len(self.key_words), (
            f"n_pairs={n_pairs} exceeds key-pool size {len(self.key_words)} "
            f"(distinct keys per example); add words to COMMON_WORDS or lower n_pairs")
        assert value_len <= len(self.val_words)
        assert 1 <= n_query <= n_pairs

    def _ids(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def _gen(self, rng: random.Random) -> dict:
        keys = rng.sample(self.key_words, self.n_pairs)                       # distinct keys
        values = [" ".join(rng.choice(self.val_words) for _ in range(self.value_len))
                  for _ in range(self.n_pairs)]                              # random pairing (with repl.)

        # ── context the encoder ingests: the N (key = value) pairs ──
        context_str = "".join(f"{k} = {v}\n" for k, v in zip(keys, values))
        ctx = self._ids(context_str)
        if len(ctx) > self.context_len:
            raise ValueError(
                f"EMAT context {len(ctx)} tok > context_len {self.context_len} "
                f"at n_pairs={self.n_pairs}, value_len={self.value_len}; raise --chunk-size")
        valid = len(ctx)
        ctx = ctx + [self.pad_token_id] * (self.context_len - valid)

        # ── conditioning + target ──
        # First queried key conditions in the question slot; its value (and, for multi-EMAT,
        # each subsequent `key = value`) is reproduced in the answer slot. Loss only on value
        # tokens — the inline keys are teacher-forced cues (given), not predicted.
        qi = rng.sample(range(self.n_pairs), self.n_query)
        question_ids = self._ids(keys[qi[0]])

        answer_ids: List[int] = []
        content: List[bool] = []
        for n, j in enumerate(qi):
            if n == 0:
                v_ids = self._ids(values[j])                                 # start of assistant turn
            else:
                cue = self._ids(f" {keys[j]} =")                            # inline cue for the next pair
                answer_ids += cue
                content += [False] * len(cue)
                v_ids = self._ids(f" {values[j]}")
            answer_ids += v_ids
            content += [True] * len(v_ids)                                   # the value IS the load-bearing span

        return {
            "context_ids": torch.tensor(ctx, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (self.context_len - valid),
                                         dtype=torch.bool),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_content_mask_list": content,
            "task_family": "emat",
            "question_type": "single" if self.n_query == 1 else f"multi{self.n_query}",
            "answer_refs": [values[qi[0]]],
        }

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        for _ in range(self.n_items):
            yield self._gen(rng)


def make_emat_dataloader(tokenizer, context_len: int, batch_size: int,
                         n_pairs: int = 64, n_query: int = 1, value_len: int = 1,
                         split: str = "train", seed: int = 0,
                         pad_token_id: int = 128_001, num_workers: int = 2) -> DataLoader:
    ds = EMATDataset(tokenizer, context_len=context_len, n_pairs=n_pairs, n_query=n_query,
                     value_len=value_len, seed=seed, pad_token_id=pad_token_id)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id))


if __name__ == "__main__":  # smoke: render a few EMAT examples
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from src.repr_learning.config import ReprConfig

    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    for nq, vl in [(1, 1), (1, 2), (3, 1)]:
        ds = EMATDataset(tok, context_len=1024, n_pairs=8, n_query=nq, value_len=vl, seed=1)
        print(f"\n===== single={nq==1} n_query={nq} value_len={vl} | "
              f"|key_pool|={len(ds.key_words)} |val_pool|={len(ds.val_words)} =====")
        s = next(iter(ds))
        ctx = tok.decode(s["context_ids"][s["context_mask"]])
        print("CONTEXT:\n" + ctx)
        print("QUESTION(key):", repr(tok.decode(s["question_ids"])))
        ans = s["answer_ids"]; cm = s["answer_content_mask_list"]
        print("ANSWER       :", repr(tok.decode(ans)))
        print("VALUE(content):", repr(tok.decode([t for t, c in zip(ans.tolist(), cm) if c])))
