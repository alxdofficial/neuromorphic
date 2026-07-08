"""RULER-style NIAH source — procedural long-context needle-in-a-haystack (TRAINING fuel).

Procedural keyed/QA source implementing the RULER (Hsieh et al., 2024, NVIDIA) needle-in-a-haystack
family as unlimited, exact-match-verifiable TRAINING data: a HAYSTACK of innocuous, templated filler
sentences with one or more NEEDLES of the form "The special magic {key} is {value}." inserted at
random positions. ``{key}``/``{value}`` are un-guessable random alnum strings (``mqar.rand_alnum``
style, via the shared ``draw_keyed`` helper) — the only way to answer is to have actually stored the
inserted binding, never a language-model prior. Also the base for the future value-overwrite fork
(``ruler_overwrite`` already forks the sibling ``variable_tracking`` recipe the same way).

Three modes (``mode`` knob):
  - ``"niah"``      — SINGLE needle (n_needles forced to 1): the base needle-in-haystack probe.
  - ``"multikey"``  — MULTI needle (``n_needles`` knob, forced >=2): K distinct key/value needles
    are inserted and the question asks for ONE of them — harder than "niah" because the model must
    discriminate the queried key from the other, confusable, inserted needles (not just find *a* fact).
  - ``"vartrack"``  — VARIABLE-TRACKING chain (``n_needles`` knob, forced >=2, = chain length):
    ``X1 = value.``, ``X2 = X1.``, … ``Xn = X(n-1).`` scattered through the haystack (RULER
    ``variable_tracking``); the question asks for the value of the LAST variable, which requires
    multi-hop resolution back through the chain to the one fact that actually carries a value.

Haystack filler is PROCEDURAL (templated subject/verb/object combinatorics — no corpus dependency,
self-contained, unlimited) — deliberately a different vocabulary from the EVAL-ONLY RULER reader
(``src/memory/data/ruler.py``, ``RULERNIAHDataset``) so this training source and that held-out OOD
probe never share needle/filler text; the random key/value space also makes train/eval collision
probability ~0 (same reasoning as ``mqar``/``ruler_overwrite``).

Runtime-procedural: no offline build, nothing under ``data/ruler_niah/``. See DATASETS.md /
docs/data_arch_plan.md (Layer L1).
"""
from __future__ import annotations

from typing import List

from .base import QAItem, Source
from .mqar import draw_keyed, rand_alnum

MODES = ("niah", "multikey", "vartrack")

# Templated, combinatorial "innocuous filler" vocabulary — deliberately distinct wording from the
# eval-only RULER reader's filler bank so training never overlaps the held-out OOD probe's text.
_SUBJECTS = [
    "The morning fog", "A quiet stream", "The old bridge", "A passing cloud",
    "The corner store", "A tall pine", "The empty bench", "A soft rain",
    "The village square", "A narrow path", "The harbor wall", "A slow tide",
    "The garden gate", "A distant bell", "The market stall", "A worn ladder",
    "The stone well", "A gentle hill", "The open field", "A wooden fence",
]
_VERBS = [
    "settled quietly over", "drifted past", "stood beside", "faded into",
    "curved around", "opened onto", "led toward", "rested near",
    "stretched beyond", "circled back to", "gave way to", "bordered",
    "overlooked", "sloped down to", "wound past", "framed",
]
_OBJECTS = [
    "the empty courtyard", "the quiet lane", "the far hillside", "the old orchard",
    "the sleepy town", "the wide meadow", "the narrow alley", "the still pond",
    "the low wall", "the open market", "the shaded grove", "the river bend",
    "the town square", "the dusty road", "the small pier", "the stone steps",
]


def _haystack_sentence(rng) -> str:
    """One procedurally-templated, contentless filler sentence (subj x verb x object combinatorics)."""
    return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}."


def _scatter(rng, haystack: List[str], inserts: List[str]) -> List[str]:
    """Interleave ``inserts`` into ``haystack`` at DISTINCT random slot positions (haystack keeps its
    own relative order; inserts land in the order given). ``len(haystack)+1`` slots (before/between/
    after every haystack sentence), so ``len(inserts)`` must be <= ``len(haystack)+1`` (callers pad the
    haystack to guarantee this)."""
    n_slots = len(haystack) + 1
    positions = sorted(rng.sample(range(n_slots), len(inserts)))
    out: List[str] = []
    ii = 0
    for i in range(n_slots):
        while ii < len(positions) and positions[ii] == i:
            out.append(inserts[ii])
            ii += 1
        if i < len(haystack):
            out.append(haystack[i])
    return out


def _distinct_alnum(rng, n: int, chunk_len: int, label: str) -> List[str]:
    """``n`` DISTINCT short random alnum identifiers (shared entropy source with ``mqar.rand_alnum``;
    used for vartrack's variable names, which — unlike ``draw_keyed``'s keys — don't carry a value)."""
    out: List[str] = []
    seen: set = set()
    guard = 0
    while len(out) < n and guard < 100 * n + 100:
        guard += 1
        s = rand_alnum(rng, chunk_len)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n:
        raise ValueError(f"{label}: could not draw {n} distinct identifiers at chunk_len={chunk_len}")
    return out


class RulerNiahSource(Source):
    """Yields RULER-style long-context needle-in-a-haystack ``QAItem``s (niah / multikey / vartrack).

    ``facts`` = the haystack+needle sentences IN INSERTION ORDER (question-agnostic — the fact list
    alone doesn't reveal which needle will be queried, matching the ``QAItem`` contract). Answer is
    always the exact inserted value substring (or, for vartrack, the value the chain resolves to) —
    clean exact-match scoring, no echo/abstraction credit possible.
    """

    kind = "qa"
    pack_n_queries = (1, 3)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0,
                 n_needles: int = 1, haystack_sents: int = 40, mode: str = "niah",
                 key_len: int = 2, val_len: int = 2, **kw):
        if mode not in MODES:
            raise ValueError(f"ruler_niah: mode must be one of {MODES}, got {mode!r}")
        # Fully procedural/random ⇒ train/val streams are naturally disjoint (collision probability
        # ~0); the per-worker RNG the Task passes to `sample` supplies the entropy. tokenizer/split/
        # seed are kept for interface parity + reproducibility bookkeeping (mirrors mqar/ruler_overwrite).
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.n_needles = n_needles
        self.haystack_sents = haystack_sents
        self.mode = mode
        self.key_len = key_len
        self.val_len = val_len

    def _haystack(self, rng, min_len: int) -> List[str]:
        n = max(self.haystack_sents, min_len)
        return [_haystack_sentence(rng) for _ in range(n)]

    def _draw_niah(self, rng, k: int) -> QAItem:
        """``k`` distinct "The special magic {key} is {value}." needles scattered in filler; the
        question asks for ONE of them (k=1 ⇒ "niah", k>=2 ⇒ "multikey")."""
        haystack = self._haystack(rng, k - 1)
        needles_kv = draw_keyed(rng, k, self.key_len, self.val_len, "ruler_niah")
        needle_sents = [f"The special magic {kv.key_text} is {kv.value_text}." for kv in needles_kv]
        facts = _scatter(rng, haystack, needle_sents)
        target = needles_kv[rng.randrange(k)]
        question = f"What is the special magic {target.key_text}?"
        return QAItem(facts=facts, question=question, answer=target.value_text, task_id=0,
                      meta={"dataset": "ruler_niah", "mode": self.mode, "n_needles": k})

    def _draw_vartrack(self, rng) -> QAItem:
        """A chain ``X1 = value.``, ``X2 = X1.``, … ``Xn = X(n-1).`` scattered in filler; the question
        asks for the LAST variable's value — multi-hop: resolving it requires walking the whole chain
        back to the single fact that actually carries ``value``."""
        chain_len = max(2, self.n_needles)
        haystack = self._haystack(rng, chain_len - 1)
        names = _distinct_alnum(rng, chain_len, self.key_len, "ruler_niah/vartrack")
        value = rand_alnum(rng, self.val_len)
        chain_sents = [f"{names[0]} = {value}."]
        chain_sents += [f"{names[i]} = {names[i - 1]}." for i in range(1, chain_len)]
        facts = _scatter(rng, haystack, chain_sents)
        question = f"What is the value of {names[-1]}?"
        return QAItem(facts=facts, question=question, answer=value, task_id=0,
                      meta={"dataset": "ruler_niah", "mode": self.mode, "chain_len": chain_len})

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            if self.mode == "vartrack":
                out.append(self._draw_vartrack(rng))
            elif self.mode == "multikey":
                out.append(self._draw_niah(rng, max(2, self.n_needles)))
            else:  # "niah"
                out.append(self._draw_niah(rng, 1))
        return out
