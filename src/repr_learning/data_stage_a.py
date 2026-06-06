"""Stage-A memory-warmup data (see docs/memory_warmup_curriculum.md).

Write a short passage about NOVEL entities, then retrieve one fact's value by a
**paraphrased natural-language question** (the key). Values are *coined* (made-up)
multi-token words, so they are UNGUESSABLE without the memory.

The passage and the question are deliberately *different surface language* (distinct
templates, neither a substring of the other), so the read must do semantic matching,
not string copying — mirroring how the biographical generator varies phrasing across
personas/question forms.

Each item:
  passage  — several sentences stating ``n_pairs`` facts, each in a varied declarative form
  pairs    — list of (key, value): key = a QUESTION about <owner>'s <relation>,
             value = a coined multi-token span

Facts spread over MULTIPLE owners (``rels_per_owner``) so a passage reads as interleaved
mini-bios, not one attribute list, and so ``n_pairs`` can scale past the relation count.

Variety / quality (2026-06-03 expansion):
  • 22 relations × 5 declarative + 5 interrogative forms  (110 passage / 110 question frames).
  • Coined values are ≥2 syllables AND verified to tokenize to ≥2 tokens — this kills the
    1-token real-word collisions ("To", "Ty", "Pio") that were guessable/unnatural.
  • ~2–3 owners per 8-fact passage (rels_per_owner=3, capped per owner for a balanced split).
"""
from __future__ import annotations

import random
from collections import Counter
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
    # Exactly 2 syllables by default: 1 syllable too often IS a real, single-token English
    # word ("To", "Ty", "Pio") — guessable; 3+ get cartoonishly long. 2 → short, novel,
    # multi-token. (_value() still verifies the realized token count lands in range.)
    n = n_syl or 2
    w = "".join(_syllable(rng) for _ in range(n))
    return w[:1].upper() + w[1:]


_TITLES = ["Ms.", "Mr.", "Dr.", "Mx.", "Prof."]
_ORG_SUFFIX = ["Inc.", "Co.", "Ltd.", "Group", "Labs", "Holdings"]


def _person(rng): return rng.choice(_TITLES) + " " + coined_word(rng, rng.choice([2, 3]))
def _place(rng):  return coined_word(rng, rng.choice([2, 3]))
def _org(rng):    return coined_word(rng, rng.choice([2, 3])) + " " + rng.choice(_ORG_SUFFIX)
def _plain(rng):  return coined_word(rng, rng.choice([2, 3]))


# Each relation has DECLARATIVE forms (passage) and INTERROGATIVE forms (key/question).
# The two share no template text, so the key is never a substring of the passage — the
# read must match meaning, not surface form. 5 phrasings each → low frame-repetition.
RELATIONS = [
    dict(rel="primary school", vfn=_place,
         passage=["{o}'s primary school was {v}.", "{o} attended {v} as a young child.",
                  "For primary school, {o} enrolled at {v}.", "As a kid, {o} went to {v}.",
                  "{o} spent their early school years at {v}."],
         question=["Which primary school did {o} attend?", "Where did {o} go to primary school?",
                   "What was the name of {o}'s primary school?", "Which school did {o} attend as a child?",
                   "Where did {o} study as a young child?"]),
    dict(rel="hometown", vfn=_place,
         passage=["{o} grew up in {v}.", "{o} hails from the town of {v}.", "{o}'s hometown is {v}.",
                  "{o} was raised in {v}.", "Home for {o} growing up was {v}."],
         question=["Where did {o} grow up?", "What town is {o} from?", "Which place does {o} call home?",
                   "Where was {o} raised?", "What is {o}'s hometown?"]),
    dict(rel="best friend", vfn=_person,
         passage=["{o}'s closest friend is {v}.", "{o} is best friends with {v}.",
                  "The person {o} trusts most is {v}.", "{o}'s dearest friend is {v}.",
                  "Nobody is closer to {o} than {v}."],
         question=["Who is {o}'s best friend?", "Whom does {o} consider their closest friend?",
                   "Who is closest to {o}?", "Who is {o}'s dearest friend?",
                   "Which friend does {o} trust most?"]),
    dict(rel="employer", vfn=_org,
         passage=["{o} works at {v}.", "{o} is employed by {v}.", "{o} earns a living at {v}.",
                  "{o} is on the payroll at {v}.", "These days {o} works for {v}."],
         question=["Where does {o} work?", "Which company employs {o}?", "Who is {o}'s employer?",
                   "What firm does {o} work for?", "Where is {o} employed?"]),
    dict(rel="mentor", vfn=_person,
         passage=["{o} was mentored by {v}.", "{v} took {o} under their wing.", "{o}'s mentor is {v}.",
                  "{o} learned the trade from {v}.", "The one who guided {o} was {v}."],
         question=["Who mentored {o}?", "Who is {o}'s mentor?", "Under whom did {o} train?",
                   "Who guided {o}?", "From whom did {o} learn the trade?"]),
    dict(rel="favorite dish", vfn=_plain,
         passage=["{o}'s favorite dish is {v}.", "{o} loves to eat {v}.",
                  "Nothing pleases {o} more than {v}.", "{o}'s go-to meal is {v}.",
                  "When {o} eats out, it's always {v}."],
         question=["What is {o}'s favorite dish?", "What food does {o} love most?",
                   "Which dish does {o} enjoy the most?", "What is {o}'s go-to meal?",
                   "What does {o} most like to eat?"]),
    dict(rel="pet", vfn=_plain,
         passage=["{o}'s pet is named {v}.", "{o} keeps a pet called {v}.",
                  "At home, {o} is greeted by {v}.", "{o}'s beloved pet is {v}.",
                  "{o} dotes on a pet named {v}."],
         question=["What is the name of {o}'s pet?", "What does {o} call their pet?",
                   "Which pet belongs to {o}?", "What is {o}'s pet named?",
                   "What is the name of the pet {o} keeps?"]),
    dict(rel="neighbor", vfn=_person,
         passage=["{o}'s next-door neighbor is {v}.", "{o} lives beside {v}.",
                  "Right next to {o} lives {v}.", "{o}'s neighbor is {v}.",
                  "The house next to {o}'s belongs to {v}."],
         question=["Who is {o}'s neighbor?", "Who lives next to {o}?", "Which neighbor does {o} have?",
                   "Who lives next door to {o}?", "Whose house is right next to {o}'s?"]),
    dict(rel="birthplace", vfn=_place,
         passage=["{o} was born in {v}.", "{o}'s birthplace is {v}.", "{v} is where {o} was born.",
                  "{o} came into the world in {v}.", "{o} first drew breath in {v}."],
         question=["Where was {o} born?", "What is {o}'s birthplace?", "In which place was {o} born?",
                   "Where did {o} come into the world?", "What place did {o} get born in?"]),
    dict(rel="alma mater", vfn=_org,
         passage=["{o} studied at {v}.", "{o} is a graduate of {v}.", "{o} earned a degree from {v}.",
                  "{o} holds a degree from {v}.", "{o} did their university studies at {v}."],
         question=["Where did {o} study?", "Which institution did {o} graduate from?",
                   "What is {o}'s alma mater?", "From where did {o} earn a degree?",
                   "Which university did {o} attend?"]),
    dict(rel="spouse", vfn=_person,
         passage=["{o} is married to {v}.", "{o}'s spouse is {v}.", "{o} tied the knot with {v}.",
                  "{o} and {v} are married.", "{o} exchanged vows with {v}."],
         question=["Who is {o} married to?", "Who is {o}'s spouse?", "Whom did {o} marry?",
                   "Who is {o}'s partner in marriage?", "To whom is {o} wed?"]),
    dict(rel="favorite book", vfn=_plain,
         passage=["{o}'s favorite book is {v}.", "{o} can't put down {v}.",
                  "The book {o} loves most is {v}.", "{o} rereads {v} every year.",
                  "{o}'s most treasured read is {v}."],
         question=["What is {o}'s favorite book?", "Which book does {o} love most?",
                   "What book does {o} treasure most?", "Which book does {o} reread?",
                   "What is the book {o} can't put down?"]),
    dict(rel="physician", vfn=_person,
         passage=["{o}'s physician is {v}.", "{o} sees {v} for checkups.", "{o}'s doctor is {v}.",
                  "When ill, {o} visits {v}.", "{o} is under the care of {v}."],
         question=["Who is {o}'s doctor?", "Which physician does {o} see?", "Who treats {o} when ill?",
                   "Who is {o}'s physician?", "Whom does {o} see for checkups?"]),
    dict(rel="landlord", vfn=_person,
         passage=["{o} rents from {v}.", "{o}'s landlord is {v}.", "{o} pays rent to {v}.",
                  "The owner of {o}'s flat is {v}.", "{o} leases their home from {v}."],
         question=["Who is {o}'s landlord?", "From whom does {o} rent?", "To whom does {o} pay rent?",
                   "Who owns the flat {o} rents?", "Who leases a home to {o}?"]),
    dict(rel="car", vfn=_plain,
         passage=["{o} drives a {v}.", "{o}'s car is a {v}.", "Parked outside {o}'s home is a {v}.",
                  "{o} gets around in a {v}.", "{o}'s vehicle is a {v}."],
         question=["What car does {o} drive?", "What is {o}'s car?", "Which vehicle does {o} drive?",
                   "What model does {o} get around in?", "What is parked outside {o}'s home?"]),
    dict(rel="business partner", vfn=_person,
         passage=["{o} went into business with {v}.", "{o}'s business partner is {v}.",
                  "{o} co-founded the venture with {v}.", "{o} runs the firm alongside {v}.",
                  "{o}'s partner in business is {v}."],
         question=["Who is {o}'s business partner?", "With whom did {o} go into business?",
                   "Who co-founded the venture with {o}?", "Who runs the firm with {o}?",
                   "Whom did {o} partner with in business?"]),
    dict(rel="favorite city", vfn=_place,
         passage=["{o}'s favorite city is {v}.", "{o} loves visiting {v}.",
                  "The city {o} adores is {v}.", "{o} dreams of moving to {v}.",
                  "{o}'s favorite place to travel is {v}."],
         question=["What is {o}'s favorite city?", "Which city does {o} love?",
                   "What city does {o} adore?", "Where does {o} dream of moving?",
                   "What is {o}'s favorite place to travel?"]),
    dict(rel="street", vfn=_place,
         passage=["{o} lives on {v}.", "{o}'s home is on {v}.", "{o}'s address is on {v}.",
                  "You'll find {o}'s house on {v}.", "{o} resides on {v}."],
         question=["What street does {o} live on?", "Where is {o}'s home?",
                   "On which street is {o}'s address?", "Where will you find {o}'s house?",
                   "What street does {o} reside on?"]),
    dict(rel="former teacher", vfn=_person,
         passage=["{o}'s favorite teacher was {v}.", "{o} was taught by {v}.",
                  "In school, {o} looked up to {v}.", "{o}'s most memorable teacher was {v}.",
                  "{v} taught {o} years ago."],
         question=["Who was {o}'s favorite teacher?", "Who taught {o}?",
                   "Whom did {o} look up to in school?", "Who was {o}'s most memorable teacher?",
                   "Which teacher did {o} have years ago?"]),
    dict(rel="dentist", vfn=_person,
         passage=["{o}'s dentist is {v}.", "{o} sees {v} for dental work.",
                  "For toothaches, {o} visits {v}.", "{o}'s teeth are looked after by {v}.",
                  "{o} books cleanings with {v}."],
         question=["Who is {o}'s dentist?", "Which dentist does {o} see?",
                   "Who does {o} visit for toothaches?", "Who looks after {o}'s teeth?",
                   "With whom does {o} book cleanings?"]),
    dict(rel="favorite band", vfn=_plain,
         passage=["{o}'s favorite band is {v}.", "{o} never misses a {v} concert.",
                  "The band {o} loves is {v}.", "{o} plays {v} on repeat.",
                  "{o}'s go-to music is {v}."],
         question=["What is {o}'s favorite band?", "Which band does {o} love?",
                   "Whose concerts does {o} never miss?", "What band does {o} play on repeat?",
                   "What is {o}'s go-to music?"]),
    dict(rel="coach", vfn=_person,
         passage=["{o} is coached by {v}.", "{o}'s coach is {v}.", "{v} trains {o} at the gym.",
                  "{o} works out under {v}.", "The one who coaches {o} is {v}."],
         question=["Who is {o}'s coach?", "Who trains {o}?", "Under whom does {o} work out?",
                   "Who coaches {o} at the gym?", "Whose athlete is {o}?"]),
]


@dataclass
class StageAItem:
    passage_ids: torch.Tensor    # [T_p]  (the whole passage → memory)
    content_mask: torch.Tensor   # [T_p] bool — coined owner/value tokens in the whole passage
    key_ids: List[torch.Tensor]  # P × [T_k]  (a QUESTION — selects which fact)
    val_ids: List[torch.Tensor]  # P × [T_v]  (the short coined value + EOS)
    sent_ids: List[torch.Tensor]      # P × [T_s]  fact j's full SENTENCE (the key-addressed recon target)
    sent_cmask: List[torch.Tensor]    # P × [T_s] bool — coined owner/value tokens within sentence j
    meta: dict


class StageAKVDataset(IterableDataset):
    """Infinite stream of (passage, [(question, value)…]) items at a fixed ``n_pairs``."""

    def __init__(self, tokenizer, n_pairs: int = 4, seed: int = 0,
                 n_items: int = 1_000_000, max_passage_tok: int = 512,
                 rels_per_owner: int = 3, min_value_tok: int = 2, max_value_tok: int = 5):
        self.tok = tokenizer
        self.n_pairs = n_pairs
        self.seed = seed
        self.n_items = n_items
        self.max_passage_tok = max_passage_tok
        self.rels_per_owner = min(rels_per_owner, len(RELATIONS))
        self.min_value_tok = min_value_tok           # reject 1-token (guessable) values
        self.max_value_tok = max_value_tok           # …and cartoonishly long ones
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        n_owners = max(1, -(-n_pairs // self.rels_per_owner))
        cap = n_owners * len(RELATIONS)
        assert n_pairs <= cap, (f"n_pairs={n_pairs} exceeds capacity {cap} "
                                f"(raise rels_per_owner or add RELATIONS)")

    def _ids(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def _value(self, rng: random.Random, vfn) -> str:
        """Draw a coined value, resampling until it tokenizes into [min,max]_value_tok tokens
        — a genuinely multi-token, unguessable span (no 1-token real words, no absurd lengths)."""
        v = vfn(rng)
        for _ in range(30):
            if self.min_value_tok <= len(self._ids(v)) <= self.max_value_tok:
                return v
            v = vfn(rng)
        return v

    def _gen(self, rng: random.Random) -> StageAItem:
        n_owners = max(1, -(-self.n_pairs // self.rels_per_owner))  # ceil
        owners, seen = [], set()                                   # UNIQUE owners: duplicates would
        while len(owners) < n_owners:                              # shrink (owner,relation) capacity
            w = coined_word(rng, 2)                                 # 2-syll: owners recur a lot, keep short
            if w not in seen:
                seen.add(w); owners.append(w)
        facts, used, owner_count = [], set(), Counter()
        guard = 0
        while len(facts) < self.n_pairs and guard < 4000:
            guard += 1
            owner = rng.choice(owners)
            if owner_count[owner] >= self.rels_per_owner:           # balanced split across owners
                continue
            ri = rng.randrange(len(RELATIONS))
            if (owner, ri) in used:                                 # one (owner,relation) once
                continue
            used.add((owner, ri)); owner_count[owner] += 1
            rel = RELATIONS[ri]
            facts.append((owner, ri, self._value(rng, rel["vfn"])))
        assert len(facts) == self.n_pairs, f"guard tripped: {len(facts)}/{self.n_pairs} facts"

        # Tokenize WITH offsets → mark the memory-load-bearing positions (coined owner/value fills).
        # Plain English is predictable from the TF prefix; only the coined spans force the encoder to
        # actually store content. Used for BOTH the whole passage and each per-fact sentence.
        def _toks_content(text, coined):
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            spans = []
            for s in coined:
                k = text.find(s)
                while k >= 0:
                    spans.append((k, k + len(s))); k = text.find(s, k + 1)
            cm = [any(ts < ce and cs < te for (cs, ce) in spans) for (ts, te) in enc["offset_mapping"]]
            return enc["input_ids"], cm

        # Render each fact's declarative SENTENCE once (aligned to facts order); the passage is those
        # sentences in shuffled order. KEY-ADDRESSED recon: memory = whole passage (all facts); key j
        # selects fact j; the recon target is fact j's SENTENCE (its coined owner+value), not the passage.
        fact_sentences = [rng.choice(RELATIONS[facts[i][1]]["passage"]).format(o=facts[i][0], v=facts[i][2])
                          for i in range(len(facts))]
        order = list(range(len(facts)))
        rng.shuffle(order)                                          # interleave owners → mini-bios
        passage = " ".join(fact_sentences[i] for i in order)
        pid, content = _toks_content(passage, set(owners) | {f[2] for f in facts})
        assert len(pid) <= self.max_passage_tok, (                 # NO silent truncation
            f"passage {len(pid)} tok > max_passage_tok {self.max_passage_tok} at n_pairs={self.n_pairs}; "
            f"raise max_passage_tok")

        key_ids, val_ids, sent_ids, sent_cmask, meta_facts = [], [], [], [], []
        for i, (owner, ri, val) in enumerate(facts):
            question = rng.choice(RELATIONS[ri]["question"]).format(o=owner)
            key_ids.append(torch.tensor(self._ids(question), dtype=torch.long))
            val_ids.append(torch.tensor(self._ids(val) + [self.eos], dtype=torch.long))
            sid, scm = _toks_content(fact_sentences[i], {owner, val})   # fact j's sentence + its coined toks
            sent_ids.append(torch.tensor(sid, dtype=torch.long))
            sent_cmask.append(torch.tensor(scm, dtype=torch.bool))
            meta_facts.append((owner, RELATIONS[ri]["rel"], val, question))
        return StageAItem(torch.tensor(pid, dtype=torch.long),
                          torch.tensor(content, dtype=torch.bool), key_ids, val_ids,
                          sent_ids, sent_cmask,
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
    Ts = max(s.numel() for x in batch for s in x.sent_ids)
    passage = torch.full((B, Tp), pad_id, dtype=torch.long); p_mask = torch.zeros(B, Tp, dtype=torch.bool)
    p_content = torch.zeros(B, Tp, dtype=torch.bool)              # coined owner/value positions
    keys = torch.full((B, P, Tk), pad_id, dtype=torch.long); k_mask = torch.zeros(B, P, Tk, dtype=torch.bool)
    vals = torch.full((B, P, Tv), pad_id, dtype=torch.long); v_mask = torch.zeros(B, P, Tv, dtype=torch.bool)
    sents = torch.full((B, P, Ts), pad_id, dtype=torch.long); s_mask = torch.zeros(B, P, Ts, dtype=torch.bool)
    s_content = torch.zeros(B, P, Ts, dtype=torch.bool)          # coined toks WITHIN each fact's sentence
    for i, x in enumerate(batch):
        n = x.passage_ids.numel(); passage[i, :n] = x.passage_ids; p_mask[i, :n] = True
        p_content[i, :n] = x.content_mask
        for j in range(P):
            k, v, s = x.key_ids[j], x.val_ids[j], x.sent_ids[j]
            keys[i, j, :k.numel()] = k; k_mask[i, j, :k.numel()] = True
            vals[i, j, :v.numel()] = v; v_mask[i, j, :v.numel()] = True
            sents[i, j, :s.numel()] = s; s_mask[i, j, :s.numel()] = True
            s_content[i, j, :s.numel()] = x.sent_cmask[j]
    return {"passage": passage, "passage_mask": p_mask, "passage_content_mask": p_content,
            "keys": keys, "keys_mask": k_mask, "values": vals, "values_mask": v_mask,
            "sentences": sents, "sentences_mask": s_mask, "sentences_content_mask": s_content,
            "meta": [x.meta for x in batch]}


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
    # multi-fact, multi-owner
    x = next(iter(StageAKVDataset(tok, n_pairs=8, seed=5)))
    print(f"=== n_pairs=8 (owners={x.meta['owners']}) ===\nPASSAGE:", x.meta["passage"])
    for o, r, v, q in x.meta["facts"]:
        print(f"  Q: {q}  ->  {v!r}")
