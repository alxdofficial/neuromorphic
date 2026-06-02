"""Stage-A memory-warmup data (see docs/memory_warmup_curriculum.md).

Write a short passage about NOVEL entities, then retrieve one fact's value by a
linguistic key. Values are *coined* (made-up) words, so they are UNGUESSABLE
without the memory — a no-memory readout must score ~0 (the unguessable check).

Each item:
  passage  — a few sentences stating ``n_pairs`` facts (each: <owner>'s <relation> = <value>)
  pairs    — list of (key, value): key = "<owner>'s <relation>", value = a coined span

Facts spread over a few owners so ``n_pairs`` can scale past the relation count
(this is the difficulty axis for the recall-vs-#pairs capacity sweep).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset

# ── coined-word generator (novel → unguessable, multi-token) ──────────────────
_ONSET = ["b", "d", "dr", "f", "g", "gr", "k", "kr", "l", "m", "n", "p", "qu", "r",
          "s", "sh", "t", "th", "tr", "v", "x", "y", "z", "vr", "zh", "bl", "sk"]
_NUC = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "y", "io"]
_CODA = ["", "l", "n", "r", "m", "x", "th", "sh", "ld", "rn", "lm", "nd", "k", "ss"]


def _syllable(rng: random.Random) -> str:
    return rng.choice(_ONSET) + rng.choice(_NUC) + rng.choice(_CODA)


def coined_word(rng: random.Random, n_syl: int | None = None) -> str:
    n = n_syl or rng.choice([1, 1, 2, 2, 3])
    w = "".join(_syllable(rng) for _ in range(n))
    return w[:1].upper() + w[1:]


def _person(rng): return rng.choice(["Ms.", "Mr.", "Dr.", "Mx."]) + " " + coined_word(rng, rng.choice([1, 2]))
def _place(rng):  return coined_word(rng, rng.choice([1, 2]))
def _plain(rng):  return coined_word(rng, rng.choice([1, 2]))


# (relation phrase, value renderer) — values are all coined, hence unguessable
RELATIONS = [
    ("second-grade teacher", _person), ("hometown", _place), ("best friend", _person),
    ("mentor", _person), ("birthplace", _place), ("employer", _plain), ("pet", _person),
    ("favorite dish", _plain), ("neighbor", _person), ("alma mater", _plain),
    ("manager", _person), ("birth city", _place),
]

PASSAGE_TEMPLATES = [
    "{owner}'s {rel} is {val}.",
    "{owner}'s {rel} was {val}.",
    "The {rel} of {owner} is {val}.",
]


@dataclass
class StageAItem:
    passage_ids: torch.Tensor    # [T_p]
    key_ids: List[torch.Tensor]  # P × [T_k]
    val_ids: List[torch.Tensor]  # P × [T_v]  (value tokens + EOS)
    meta: dict


class StageAKVDataset(IterableDataset):
    """Infinite stream of (passage, [(key, value)…]) items at a fixed ``n_pairs``."""

    def __init__(self, tokenizer, n_pairs: int = 4, seed: int = 0,
                 n_items: int = 1_000_000, max_passage_tok: int = 256,
                 rels_per_owner: int = 8):
        self.tok = tokenizer
        self.n_pairs = n_pairs
        self.seed = seed
        self.n_items = n_items
        self.max_passage_tok = max_passage_tok
        self.rels_per_owner = rels_per_owner
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    def _ids(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def _gen(self, rng: random.Random) -> StageAItem:
        n_owners = max(1, -(-self.n_pairs // self.rels_per_owner))  # ceil
        owners = [coined_word(rng, rng.choice([1, 2])) for _ in range(n_owners)]
        facts, used = [], set()
        guard = 0
        while len(facts) < self.n_pairs and guard < 1000:
            guard += 1
            owner = rng.choice(owners)
            rel, vfn = rng.choice(RELATIONS)
            if (owner, rel) in used:
                continue
            used.add((owner, rel))
            facts.append((owner, rel, vfn(rng)))

        order = list(range(len(facts)))
        rng.shuffle(order)
        passage = " ".join(
            rng.choice(PASSAGE_TEMPLATES).format(owner=facts[i][0], rel=facts[i][1], val=facts[i][2])
            for i in order)
        pid = self._ids(passage)[:self.max_passage_tok]

        key_ids, val_ids = [], []
        for owner, rel, val in facts:
            key_ids.append(torch.tensor(self._ids(f"{owner}'s {rel}"), dtype=torch.long))
            val_ids.append(torch.tensor(self._ids(val) + [self.eos], dtype=torch.long))
        return StageAItem(torch.tensor(pid, dtype=torch.long), key_ids, val_ids,
                          {"owners": owners, "facts": facts, "passage": passage})

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.n_items):
            yield self._gen(rng)


def collate_stage_a(batch: List[StageAItem], pad_id: int = 0) -> dict:
    B, P = len(batch), len(batch[0].key_ids)
    Tp = max(x.passage_ids.numel() for x in batch)
    Tk = max(k.numel() for x in batch for k in x.key_ids)
    Tv = max(v.numel() for x in batch for v in x.val_ids)
    passage = torch.full((B, Tp), pad_id, dtype=torch.long); p_mask = torch.zeros(B, Tp, dtype=torch.bool)
    keys = torch.full((B, P, Tk), pad_id, dtype=torch.long); k_mask = torch.zeros(B, P, Tk, dtype=torch.bool)
    vals = torch.full((B, P, Tv), pad_id, dtype=torch.long); v_mask = torch.zeros(B, P, Tv, dtype=torch.bool)
    for i, x in enumerate(batch):
        n = x.passage_ids.numel(); passage[i, :n] = x.passage_ids; p_mask[i, :n] = True
        for j in range(P):
            k, v = x.key_ids[j], x.val_ids[j]
            keys[i, j, :k.numel()] = k; k_mask[i, j, :k.numel()] = True
            vals[i, j, :v.numel()] = v; v_mask[i, j, :v.numel()] = True
    return {"passage": passage, "passage_mask": p_mask, "keys": keys, "keys_mask": k_mask,
            "values": vals, "values_mask": v_mask, "meta": [x.meta for x in batch]}


def make_stage_a_loader(tokenizer, batch_size: int = 8, n_pairs: int = 4, seed: int = 0, **kw):
    ds = StageAKVDataset(tokenizer, n_pairs=n_pairs, seed=seed, **kw)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_stage_a)


if __name__ == "__main__":  # smoke + unguessable sanity
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from src.repr_learning.config import ReprConfig

    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    ds = StageAKVDataset(tok, n_pairs=4, seed=1)
    it = iter(ds)
    x = next(it)
    print("=== sample item ===")
    print("passage:", x.meta["passage"])
    for (o, r, v), k_ids, v_ids in zip(x.meta["facts"], x.key_ids, x.val_ids):
        print(f"  key={o}'s {r!r:30s} -> val={v!r:18s}  "
              f"(key {k_ids.numel()} tok, val {v_ids.numel()} tok incl EOS)")
    # checks
    import statistics as st
    vals = [vi.numel() - 1 for _ in range(200) for vi in next(it).val_ids]  # minus EOS
    print(f"\nvalue length (tokens): mean={st.mean(vals):.2f} min={min(vals)} max={max(vals)} "
          f"-> {100*sum(v>1 for v in vals)/len(vals):.0f}% are multi-token")
    # batch
    from torch.utils.data import DataLoader as DL
    b = next(iter(DL(StageAKVDataset(tok, n_pairs=4, seed=2), batch_size=4, collate_fn=collate_stage_a)))
    print("batch shapes:", {k: tuple(v.shape) for k, v in b.items() if torch.is_tensor(v)})
    print("\nunguessable: values are coined words assigned at random per (owner,relation); "
          "given only the key there is no signal -> a no-memory readout must score ~0.")
