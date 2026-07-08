"""Source base — raw item streams. A Source knows *where tokens come from*, nothing about
windows, queries, or distractors (that is the Task's job). It yields typed *items* that Tasks
consume. See ``docs/data_arch_plan.md`` (Layer L1).

Item kinds + the interface Tasks rely on:
  - CorpusItem : ``.tokens``                              (fineweb, pile, redpajama, code)
  - KeyedItem  : ``.key_text, .value_text, .value_subs``  (bio, mqar, ruler_overwrite)
  - QAItem     : ``.facts, .question, .answer``           (babi)  + ``Source.distractor_pool()``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]     # repo root, shared by every source that reads data/<name>/


@dataclass
class CorpusItem:
    """A span of natural text, already tokenized with the BACKBONE tokenizer."""
    tokens: Any                       # 1-D np.ndarray[int64]


@dataclass
class KeyedItem:
    """An entity rendered as key→value, with the load-bearing value substrings for the loss mask."""
    key_text: str
    value_text: str
    value_subs: list                  # list[str] — the exact fact-value substrings inside value_text
    name: str = ""                    # entity name (key-derivable → excluded from the loss span)
    given: str = ""                   # given/first name (also excluded)


@dataclass
class QAItem:
    """A question-agnostic story (declarative facts) + a question and its answer."""
    facts: list                       # list[str] — the gold fact sentences (order preserved)
    question: str
    answer: str
    task_id: int = 0
    meta: dict = field(default_factory=dict)


class Source(ABC):
    """Yields items of one ``kind`` ("corpus" | "keyed" | "qa"). Constructed once per split;
    ``sample`` draws fresh items each call (worker-seeded RNG passed by the Task).

    Per-source PACKING profile (item size varies by source, so how many to query does too — a single
    per-task constant can't be right for a union like qa_multi). The qa/reconstruction Tasks read these:
      - ``pack_n_queries`` : (min, max) facts to ask about per episode; the Task samples in-range and
        the packer feasibility-caps it (big-context sources can't fit the max). Fill is always
        budget-driven, so the SEGMENT count falls out of item size automatically.
      - ``pack_rename``    : co-packed items reuse entities (bAbI) → rename them disjoint per segment.
    """

    kind: str = ""
    pack_n_queries: tuple = (1, 1)     # (min, max) queries/episode — see class docstring
    pack_rename: bool = False          # co-packed segments need disjoint entity renaming (bAbI)

    @abstractmethod
    def sample(self, rng, n: int) -> list:
        """Draw ``n`` items (documents / entities / stories). May return < n only if the source
        is genuinely exhausted; procedural/corpus sources are effectively infinite."""
        raise NotImplementedError

    def distractor_pool(self) -> list:
        """Optional noise units (e.g. bAbI sentences) for Tasks that pad with interference.
        Default: none (keyed/corpus tasks draw distractors as extra items instead)."""
        return []
