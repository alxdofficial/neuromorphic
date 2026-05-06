"""Shared helpers for passphrase corpus loading.

Provides ``_load_facts``, ``_split_train_heldout``, and ``_ExpandedFact``,
used by ``passphrase_chat_loader.py`` (Wave 3 chat-injected GRPO) and
any future passphrase-based eval / train script.

Originally lived in ``src/data/passphrase_loader.py`` alongside the
AR-TF teacher-forced loader. That loader was retired (see scope-B
cleanup, 2026-05-06) but these tiny helpers are still useful, so they
got hoisted out before the parent file was deleted.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _ExpandedFact:
    id: int
    topic: str
    fact: str
    paraphrases: list[str]
    questions: list[str]
    reference_answers: list[str]


def _load_facts(path: str | Path) -> list[_ExpandedFact]:
    """Load and validate ``expanded.json`` produced by ``build_user_facts.py``."""
    with Path(path).open() as f:
        raw = json.load(f)
    facts: list[_ExpandedFact] = []
    for entry in raw:
        facts.append(_ExpandedFact(
            id=entry["id"],
            topic=entry.get("topic", "misc"),
            fact=entry["fact"],
            paraphrases=entry["paraphrases"],
            questions=entry["questions"],
            reference_answers=entry["reference_answers"],
        ))
    return facts


def _split_train_heldout(
    facts: list[_ExpandedFact], n_heldout: int, seed: int = 42,
) -> tuple[list[_ExpandedFact], list[_ExpandedFact]]:
    """Deterministic split. ``n_heldout`` random facts are held out for
    evaluation; everything else goes to training."""
    rng = random.Random(seed)
    ids = sorted([f.id for f in facts])
    rng.shuffle(ids)
    heldout_ids = set(ids[:n_heldout])
    train_facts: list[_ExpandedFact] = []
    heldout_facts: list[_ExpandedFact] = []
    for f in facts:
        (heldout_facts if f.id in heldout_ids else train_facts).append(f)
    return train_facts, heldout_facts
