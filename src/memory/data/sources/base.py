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
from typing import Any


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
    ``sample`` draws fresh items each call (worker-seeded RNG passed by the Task)."""

    kind: str = ""

    @abstractmethod
    def sample(self, rng, n: int) -> list:
        """Draw ``n`` items (documents / entities / stories). May return < n only if the source
        is genuinely exhausted; procedural/corpus sources are effectively infinite."""
        raise NotImplementedError

    def distractor_pool(self) -> list:
        """Optional noise units (e.g. bAbI sentences) for Tasks that pad with interference.
        Default: none (keyed/corpus tasks draw distractors as extra items instead)."""
        return []
