"""Stage-A memory-warmup data (see docs/memory_warmup_curriculum.md).

Write a short passage about NOVEL entities, then retrieve one fact's value by a
**paraphrased natural-language question** (the key). Values are *coined* (made-up)
words, so they are UNGUESSABLE without the memory.

The passage and the question are deliberately *different surface language* (distinct
templates, neither a substring of the other), so the read must do semantic matching,
not string copying — mirroring how the biographical generator varies phrasing across
personas/question forms.

Each item:
  passage  — a few sentences stating ``n_pairs`` facts, each in a varied declarative form
  pairs    — list of (key, value): key = a QUESTION about <owner>'s <relation>,
             value = a coined span

Facts spread over a few owners so ``n_pairs`` can scale past the relation count.
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


# Each relation has a DECLARATIVE form (for the passage) and an INTERROGATIVE form
# (for the key/question). The two share no template text, so the key is never a
# substring of the passage — the read has to match meaning, not surface form.
RELATIONS = [
    dict(rel="primary school", vfn=_place,
         passage=["{o}'s primary school was {v}.",
                  "{o} attended {v} as a young child.",
                  "For primary school, {o} enrolled at {v}."],
         question=["Which primary school did {o} attend?",
                   "Where did {o} go to primary school?",
                   "What was the name of {o}'s primary school?"]),
    dict(rel="hometown", vfn=_place,
         passage=["{o} grew up in {v}.",
                  "{o} hails from the town of {v}.",
                  "{o}'s hometown is {v}."],
         question=["Where did {o} grow up?",
                   "What town is {o} from?",
                   "Which place does {o} call home?"]),
    dict(rel="best friend", vfn=_person,
         passage=["{o}'s closest friend is {v}.",
                  "{o} is best friends with {v}.",
                  "The person {o} trusts most is {v}."],
         question=["Who is {o}'s best friend?",
                   "Whom does {o} consider their closest friend?",
                   "Who is closest to {o}?"]),
    dict(rel="employer", vfn=_plain,
         passage=["{o} works at {v}.",
                  "{o} is employed by {v}.",
                  "{o} earns a living at {v}."],
         question=["Where does {o} work?",
                   "Which company employs {o}?",
                   "Who is {o}'s employer?"]),
    dict(rel="mentor", vfn=_person,
         passage=["{o} was mentored by {v}.",
                  "{v} took {o} under their wing.",
                  "{o}'s mentor is {v}."],
         question=["Who mentored {o}?",
                   "Who is {o}'s mentor?",
                   "Under whom did {o} train?"]),
    dict(rel="favorite dish", vfn=_plain,
         passage=["{o}'s favorite dish is {v}.",
                  "{o} loves to eat {v}.",
                  "Nothing pleases {o} more than {v}."],
         question=["What is {o}'s favorite dish?",
                   "What food does {o} love most?",
                   "Which dish does {o} enjoy the most?"]),
    dict(rel="pet", vfn=_plain,
         passage=["{o}'s pet is named {v}.",
                  "{o} keeps a pet called {v}.",
                  "At home, {o} is greeted by {v}."],
         question=["What is the name of {o}'s pet?",
                   "What does {o} call their pet?",
                   "Which pet belongs to {o}?"]),
    dict(rel="neighbor", vfn=_person,
         passage=["{o}'s next-door neighbor is {v}.",
                  "{o} lives beside {v}.",
                  "Right next to {o} lives {v}."],
         question=["Who is {o}'s neighbor?",
                   "Who lives next to {o}?",
                   "Which neighbor does {o} have?"]),
    dict(rel="birthplace", vfn=_place,
         passage=["{o} was born in {v}.",
                  "{o}'s birthplace is {v}.",
                  "{v} is where {o} was born."],
         question=["Where was {o} born?",
                   "What is {o}'s birthplace?",
                   "In which place was {o} born?"]),
    dict(rel="alma mater", vfn=_plain,
         passage=["{o} studied at {v}.",
                  "{o} is a graduate of {v}.",
                  "{o} earned a degree from {v}."],
         question=["Where did {o} study?",
                   "Which institution did {o} graduate from?",
                   "What is {o}'s alma mater?"]),
]


@dataclass
class StageAItem:
    passage_ids: torch.Tensor    # [T_p]
    key_ids: List[torch.Tensor]  # P × [T_k]  (a QUESTION)
    val_ids: List[torch.Tensor]  # P × [T_v]  (value tokens + EOS)
    meta: dict


class StageAKVDataset(IterableDataset):
    """Infinite stream of (passage, [(question, value)…]) items at a fixed ``n_pairs``."""

    def __init__(self, tokenizer, n_pairs: int = 4, seed: int = 0,
                 n_items: int = 1_000_000, max_passage_tok: int = 256,
                 rels_per_owner: int = 8):
        self.tok = tokenizer
        self.n_pairs = n_pairs
        self.seed = seed
        self.n_items = n_items
        self.max_passage_tok = max_passage_tok
        self.rels_per_owner = min(rels_per_owner, len(RELATIONS))
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        n_owners = max(1, -(-n_pairs // self.rels_per_owner))
        cap = n_owners * len(RELATIONS)
        assert n_pairs <= cap, (f"n_pairs={n_pairs} exceeds capacity {cap} "
                                f"(raise rels_per_owner or add RELATIONS)")

    def _ids(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def _gen(self, rng: random.Random) -> StageAItem:
        n_owners = max(1, -(-self.n_pairs // self.rels_per_owner))  # ceil
        owners = [coined_word(rng, rng.choice([1, 2])) for _ in range(n_owners)]
        facts, used = [], set()
        guard = 0
        while len(facts) < self.n_pairs and guard < 2000:
            guard += 1
            owner = rng.choice(owners)
            ri = rng.randrange(len(RELATIONS))
            if (owner, ri) in used:
                continue
            used.add((owner, ri))
            rel = RELATIONS[ri]
            facts.append((owner, ri, rel["vfn"](rng)))
        assert len(facts) == self.n_pairs, f"guard tripped: {len(facts)}/{self.n_pairs} facts"

        order = list(range(len(facts)))
        rng.shuffle(order)
        passage = " ".join(
            rng.choice(RELATIONS[facts[i][1]]["passage"]).format(o=facts[i][0], v=facts[i][2])
            for i in order)
        pid = self._ids(passage)[:self.max_passage_tok]

        key_ids, val_ids, meta_facts = [], [], []
        for owner, ri, val in facts:
            question = rng.choice(RELATIONS[ri]["question"]).format(o=owner)
            key_ids.append(torch.tensor(self._ids(question), dtype=torch.long))
            val_ids.append(torch.tensor(self._ids(val) + [self.eos], dtype=torch.long))
            meta_facts.append((owner, RELATIONS[ri]["rel"], val, question))
        return StageAItem(torch.tensor(pid, dtype=torch.long), key_ids, val_ids,
                          {"owners": owners, "facts": meta_facts, "passage": passage})

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()           # distinct stream per worker
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        for _ in range(self.n_items):
            yield self._gen(rng)


def collate_stage_a(batch: List[StageAItem], pad_id: int = 128_001) -> dict:
    # pad_id default = Llama-3 pad token (NOT 0, which is a real BPE piece).
    B, P = len(batch), len(batch[0].key_ids)
    assert all(len(x.key_ids) == P for x in batch), "items in a batch have differing #pairs"
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


if __name__ == "__main__":  # smoke + paraphrase sanity
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from src.repr_learning.config import ReprConfig

    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    it = iter(StageAKVDataset(tok, n_pairs=1, seed=1))
    for _ in range(4):
        x = next(it)
        o, r, v, q = x.meta["facts"][0]
        print(f"PASSAGE : {x.meta['passage']}")
        print(f"  KEY(Q): {q}")
        print(f"  VALUE : {v!r}  ({x.val_ids[0].numel()} tok incl EOS)\n")
    # multi-fact
    x = next(iter(StageAKVDataset(tok, n_pairs=3, seed=5)))
    print("=== n_pairs=3 ===\nPASSAGE:", x.meta["passage"])
    for o, r, v, q in x.meta["facts"]:
        print(f"  Q: {q}  ->  {v!r}")
