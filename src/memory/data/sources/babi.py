"""bAbI source — question-agnostic stories (declarative facts) + a distractor sentence pool.

Source half of the old ``data/babi.py``: loads rows from HF ``Muennighoff/babi`` (offline fallback
synthesizes task-1), splits stories into fact sentences, and exposes a noise pool of other stories'
sentences for the qa task to pad with. The distractor-insertion / packing is the qa Task's job.

Data/build: HF ``Muennighoff/babi`` (1k), auto-downloaded; 10k ingest =
``scripts/data_build/ingest/babi_10k.py`` (TODO). See DATASETS.md / docs/DATA.md.
"""
from __future__ import annotations

import itertools
import random
import re
from typing import List

from .base import Source, QAItem

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Memory-focused task subset (default). 1/2/3 = one/two/three supporting facts, 7 = counting,
# 8 = lists/sets, 11/12/13 = coreference, 14 = time reasoning.
DEFAULT_TASKS = (1, 2, 3, 7, 8, 11, 12, 13, 14)

_FALLBACK_NAMES = ["Mary", "John", "Daniel", "Sandra", "Fred", "Julie", "Bill", "Jeff"]
_FALLBACK_PLACES = ["bathroom", "hallway", "kitchen", "garden", "bedroom",
                    "office", "school", "park", "cinema", "kitchen"]
_FALLBACK_MOVES = ["moved to", "went to", "journeyed to", "travelled to", "went back to"]

# bAbI locations are ANSWERS, not queried subjects → never renamed (a shared location across segments
# doesn't make "where is <segment-k's person>?" ambiguous).
_LOCATIONS = {"bathroom", "hallway", "kitchen", "garden", "bedroom", "office", "school", "park",
              "cinema", "pantry", "closet", "porch", "cellar", "attic", "study", "lounge"}

# object-transfer verbs → the noun after them is a portable OBJECT (used to derive bAbI's object vocab
# from the data, so renaming works for any task subset without a hardcoded object list).
_OBJ_VERB = re.compile(r"\b(?:picked up|put down|grabbed|took|got|left|dropped|discarded|"
                       r"handed|passed|received|gave)\s+(?:the\s+)?(\w+)")

# a capitalized word that is the SUBJECT of a bAbI ACTION verb (movement/transfer) is a PERSON. Derived
# from the data. ACTION verbs only — NOT state verbs (is/was): "Where was the milk" would otherwise grab
# the question word "Where". Every bAbI person moves at least once, so the action-verb vocab is complete.
_PERSON_SUBJ = re.compile(r"\b([A-Z][a-z]+)\s+(?:went|moved|journeyed|travelled|picked|grabbed|took|"
                          r"got|dropped|left|put|handed|passed|received|gave|discarded|carries|carried)\b")

# Disjoint replacement pools (NOT overlapping bAbI's own Mary/John/… or milk/apple/football) — large
# enough that ~20 co-packed segments each get globally unique people + objects with headroom.
_RENAME_NAMES = [
    "Aldric", "Beatrix", "Caspian", "Delphine", "Emeric", "Fenwick", "Guinevere", "Hadrian",
    "Isolde", "Jorah", "Katarina", "Leontes", "Mirabel", "Nikolai", "Ophelia", "Percival",
    "Quintus", "Rosalind", "Sebastian", "Theodora", "Ulric", "Vivienne", "Wendell", "Xanthe",
    "Yseult", "Zephyrine", "Alaric", "Brunhilda", "Cornelius", "Drusilla", "Evander", "Faustine",
    "Godfrey", "Henrietta", "Ignatius", "Jacinta", "Kenelm", "Lavinia", "Montague", "Nerissa",
    "Osric", "Philippa", "Quenby", "Reinhold", "Seraphina", "Tobias", "Ursula", "Valerian",
    "Winifred", "Xavier", "Yolanda", "Zebedee", "Anselm", "Bathsheba", "Cormac", "Dorothea",
    "Ephraim", "Ferdinand", "Gwendolyn", "Horatio", "Idris", "Jemima", "Konrad", "Lucasta",
    "Marius", "Nadezhda", "Octavian", "Perpetua", "Quillon", "Ruricius", "Sigismund", "Tatiana",
    "Urien", "Verena", "Wilhelmina", "Xerxes", "Ysabeau", "Zenobia", "Ambrose", "Berengaria",
    "Cyprian", "Desdemona", "Eberhard", "Florian", "Grimwald", "Hildegard", "Ithamar", "Jocasta",
    "Kasimir", "Lucretia", "Meinhard", "Nunzio", "Odovacar", "Prudence", "Querela", "Roderick",
    "Sidonia", "Thaddeus", "Ulfric", "Vespasia", "Waldemar", "Xiomara", "Ysolde", "Zephaniah",
    "Anastasia", "Balthasar", "Clementine", "Dagobert", "Euphemia", "Frideric", "Genevieve",
    "Hortensia", "Ivo", "Josceline", "Kunibert", "Ludmila", "Magnus", "Nicasius", "Ottoline",
    "Pelagia", "Radulf", "Sabinus", "Theobald", "Umberto", "Valentina", "Wolfram", "Xenophon",
    "Yorick", "Zelphina", "Archibald", "Bertrada", "Casimira", "Dietrich", "Estella", "Fulbert",
    "Gisela", "Hubertus", "Immaculata", "Jehanne", "Kilian", "Leopolda", "Morwenna", "Norbert",
    "Odalys", "Ptolemy", "Ravenna", "Sylvestra", "Torquil", "Ulyssa", "Vitalis", "Wenceslas",
    "Ximena", "Yorath", "Zosimus", "Adalbert", "Brünhild", "Cecilia", "Deodatus", "Eulalia",
]
_RENAME_OBJECTS = [
    "lantern", "kettle", "compass", "ledger", "trowel", "satchel", "goblet", "anvil", "quill",
    "abacus", "mallet", "flask", "sextant", "chisel", "tankard", "harp", "spindle", "cauldron",
    "beacon", "scroll", "brazier", "gauntlet", "censer", "astrolabe", "reliquary", "tuning-fork",
    "cudgel", "phial", "bellows", "grindstone", "hourglass", "loom", "manacle", "plumb-bob",
    "quiver", "rasp", "sledge", "torque", "vellum", "whetstone", "yardstick", "zither", "aiglet",
    "buckler", "crampon", "dulcimer", "ewer", "firkin", "grommet", "halberd", "inkhorn", "jerkin",
    "kazoo", "lyre", "matchlock", "nib", "oarlock", "pannier", "quern", "ratchet", "sabaton",
    "thurible", "urn", "vise", "wimple", "xylophone", "yoke", "zarf", "amphora", "bodkin",
    "caltrop", "distaff", "escutcheon", "fetlock", "gimlet", "holster", "ingot", "jubbah",
]


def _split_sents(text: str) -> List[str]:
    """Split a bAbI passage into individual fact SENTENCES (handles both the HF one-line form and
    the newline-separated fallback): flatten newlines → split after sentence-final punctuation."""
    flat = text.replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]



def _load_babi_rows(tasks, split: str):
    """Return list[(story, question, answer, task_int)] for the given tasks/split. Tries HF, then
    a programmatic task-1 fallback if offline. The story field is QUESTION-AGNOSTIC by construction."""
    task_set = set(tasks)
    if split == "train":
        hf_split = "train"
    elif split in ("validation", "val"):
        hf_split = "validation"
    else:
        raise ValueError(
            f"bAbI: unrecognized split {split!r} (expected 'train', 'validation', or 'val')")

    for name in ("Muennighoff/babi",):
        try:
            from datasets import load_dataset
            ds = load_dataset(name, split=hf_split)
            rows = []
            for ex in ds:
                t = int(ex["task"])
                if t not in task_set:
                    continue
                story = (ex["passage"] or "").strip()
                q = (ex["question"] or "").strip()
                a = (ex["answer"] or "").strip()
                if story and q and a:
                    rows.append((story, q, a, t))
            if rows:
                print(f"[babi] loaded {len(rows):,} rows from {name} "
                      f"(split={hf_split}, tasks={sorted(task_set)})", flush=True)
                return rows
        except Exception as e:  # pragma: no cover — network/offline path
            print(f"[babi] {name} unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  f"trying next source", flush=True)

    if task_set != {1}:
        raise RuntimeError(
            f"bAbI HF source unreachable and the offline fallback only synthesizes TASK-1 stories, but "
            f"the requested tasks are {sorted(task_set)} (≠ {{1}}). Silently training the default "
            f"multi-task set on task-1-only would corrupt the binding mix and misreport the arm (audit "
            f"blocker #3). Restore network access (Muennighoff/babi), or request exactly task 1 to use "
            f"the offline fallback.")
    is_val = split in ("validation", "val")
    print(f"[babi] HF bAbI unreachable — generating programmatic task-1 stories "
          f"(offline fallback, split={'val' if is_val else 'train'})", flush=True)
    gen = random.Random(5678 if is_val else 1234)
    rows = []
    for _ in range(4000):
        n_facts = gen.randint(2, 8)
        loc = {}
        lines = []
        for _ in range(n_facts):
            who = gen.choice(_FALLBACK_NAMES)
            where = gen.choice(_FALLBACK_PLACES)
            loc[who] = where
            lines.append(f"{who} {gen.choice(_FALLBACK_MOVES)} the {where}.")
        who_q = gen.choice(list(loc.keys()))
        rows.append(("\n".join(lines) + "\n", f"Where is {who_q}?", loc[who_q], 1))
    return rows


class BabiSource(Source):
    """Yields bAbI stories as QAItems. `pack_rename` tells the qa Task to co-pack MANY stories to fill
    the budget but give each a DISJOINT set of entities (people + objects renamed per segment): bAbI
    reuses a tiny name pool, so without renaming a distractor story's "Mary" collides with the gold's
    "Mary" (ambiguous supervision). Renaming turns bAbI into a fill-the-budget, retrieve-the-right-
    segment + bind-within-it task at a real compression ratio. Locations are answers → never renamed."""

    kind = "qa"
    pack_n_queries = (1, 3)            # query several of the co-packed segments (addressing pressure)
    pack_rename = True                 # co-packed bAbI segments must be entity-disjoint (see rename())

    def __init__(self, tokenizer, *, split: str = "train", tasks=DEFAULT_TASKS, seed: int = 0, **kw):
        self.tasks = tuple(tasks)
        self.rows = _load_babi_rows(self.tasks, split)
        if not self.rows:
            raise ValueError(f"bAbI: no rows for tasks={self.tasks} split={split}")
        # data-derived vocabularies: which tokens are PEOPLE (subjects of action verbs) and which are
        # portable OBJECTS (picked up / dropped / …). Renaming keys off these sets, so capitalized
        # non-names (Where/What/Following/Yesterday) and locations are never touched.
        self.people_vocab, self.object_vocab = set(), set()
        for story, q, _a, _t in self.rows:
            self.object_vocab.update(m.group(1) for m in _OBJ_VERB.finditer(story))
            self.people_vocab.update(m.group(1) for m in _PERSON_SUBJ.finditer(story + " " + q))

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            story, q, a, t = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(facts=_split_sents(story), question=q, answer=a, task_id=t,
                              meta={"dataset": "babi"}))    # per-dataset telemetry (task_family)
        return out

    def rename_pools(self, rng):
        """Fresh shuffled entity ITERATORS (name_iter, object_iter) for ONE episode — the qa Task pulls
        disjoint entities from these across all co-packed segments so no two segments share a person or
        object. Each iterator is the shuffled curated pool followed by an INEXHAUSTIBLE unique tail
        (``Person{i}``/``item{i}``), so co-packing arbitrarily many segments (BEAM-scale total_len) never
        exhausts the pool and never re-uses a name — the tail ids are episode-global (the iterator is
        shared across all rename() calls), so they can't collide across segments."""
        names, objs = _RENAME_NAMES[:], _RENAME_OBJECTS[:]
        rng.shuffle(names)
        rng.shuffle(objs)
        name_iter = itertools.chain(names, (f"Person{i}" for i in itertools.count()))
        obj_iter = itertools.chain(objs, (f"item{i}" for i in itertools.count()))
        return name_iter, obj_iter

    def rename(self, item: QAItem, name_iter, obj_iter) -> QAItem:
        """Return a copy of `item` with its people and portable objects replaced by fresh entities PULLED
        from the shared `name_iter` / `obj_iter` (so co-packed segments get globally disjoint entities).
        Locations are left untouched (answers, not subjects). The same map is applied to facts, question,
        AND answer, so a person/object answer stays consistent."""
        text = " ".join(item.facts) + " " + item.question + " " + item.answer
        toks = {w.strip(".,!?;:'\"") for w in text.split()}
        people = toks & self.people_vocab                 # data-derived: real names only (not Where/What/…)
        objects = toks & self.object_vocab
        rmap = {}
        for p in sorted(people):                          # sorted → deterministic given the shuffled pools
            rmap[p] = next(name_iter)
        for o in sorted(objects):
            rmap[o] = next(obj_iter)

        def sub(s: str) -> str:
            # whole-word substitution; longest-first avoids partial hits (none expected in bAbI)
            for ent in sorted(rmap, key=len, reverse=True):
                s = re.sub(rf"\b{re.escape(ent)}\b", rmap[ent], s)
            return s

        return QAItem(facts=[sub(f) for f in item.facts], question=sub(item.question),
                      answer=sub(item.answer), task_id=item.task_id, meta=dict(item.meta or {}))
