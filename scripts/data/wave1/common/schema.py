"""Shared schema for composite Wave 1 dataset.

Every task family's generator emits two JSONL streams that conform to
these row schemas. The trainer's RetrievalSampler doesn't care which
family produced a row — it just reads passages/questions and bundles
them per the `task_family` + `evidence_keys` fields.

Two JSONLs per task:
- passages.jsonl: one row per write-window-worth of text (one passage)
- questions.jsonl: one row per question, with evidence_keys pointing into
  passages.jsonl by passage_id

When all tasks are merged into the composite dataset:
- passage_id values must be globally unique across families (the task
  generator should prefix them with the task name, e.g. "bio_pi_0042_...")
- evidence_keys references stay valid

Validation helpers below enforce these invariants at emit time so the
trainer can trust the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ── Passage row ─────────────────────────────────────────────────────


@dataclass
class PassageRow:
    """One passage = one write-window-worth of text. The trainer feeds
    these directly to Llama via `passage_token_ids`.

    `passage_id` must be globally unique across the composite dataset.
    Prefix with task name when generating (e.g. `bio_pi_0001_sample_0`,
    `calendar_event_0042`, `boxes_op_0007`).

    `task_family` identifies which generator produced this passage.
    `passage_type` is a task-specific subtype (e.g. "Person",
    "calendar_event", "boxes_add_op") — used for distractor selection
    and per-subtype telemetry if desired.
    """
    task_family: str
    passage_id: str
    passage_type: str
    passage: str
    passage_token_ids: list[int]
    passage_token_count: int = 0
    # Optional task-specific extras (e.g. biographical's entity_type,
    # surface_names, outgoing_edges). Stored as a flat dict so the
    # downstream sampler / splitter can use them when present without
    # forcing every task to populate them.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.passage_token_count == 0 and self.passage_token_ids:
            self.passage_token_count = len(self.passage_token_ids)


# ── Question row ────────────────────────────────────────────────────


@dataclass
class QuestionRow:
    """One question = one (question text, answer text, evidence_keys).

    `evidence_keys` is the list of passage_id values whose passages must
    be in the chunk for the answer to be derivable. The sampler bundles
    them into a write-then-read chunk.

    `task_family` and `question_type` are used for per-family/per-type
    val telemetry — the trainer aggregates loss/acc keyed on these.
    """
    task_family: str
    question_type: str
    question_id: str
    evidence_keys: list[str]
    question: str
    answer: str
    target_value: str                       # short-form ground truth for accuracy scoring
    question_token_ids: list[int]
    answer_token_ids: list[int]
    question_token_count: int = 0
    answer_token_count: int = 0
    # Optional task-specific extras (e.g. biographical's relation_chain,
    # temporal_targets, predicate_attr_path).
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.question_token_count == 0 and self.question_token_ids:
            self.question_token_count = len(self.question_token_ids)
        if self.answer_token_count == 0 and self.answer_token_ids:
            self.answer_token_count = len(self.answer_token_ids)


# ── Validation ──────────────────────────────────────────────────────


def validate_passages(passages: list[dict]) -> None:
    """Sanity-check a list of passage rows (loaded from JSONL).

    Raises ValueError on:
    - duplicate passage_id
    - missing required fields
    """
    seen: set[str] = set()
    required = {"task_family", "passage_id", "passage_type", "passage",
                "passage_token_ids"}
    for i, p in enumerate(passages):
        missing = required - set(p.keys())
        if missing:
            raise ValueError(f"passage[{i}] missing fields: {missing}")
        pid = p["passage_id"]
        if pid in seen:
            raise ValueError(f"duplicate passage_id: {pid}")
        seen.add(pid)


def validate_questions(
    questions: list[dict], passage_ids: set[str] | None = None,
) -> None:
    """Sanity-check a list of question rows.

    Raises ValueError on:
    - missing required fields
    - duplicate question_id
    - evidence_keys referencing unknown passage_id (if passage_ids given)
    """
    seen: set[str] = set()
    required = {"task_family", "question_type", "question_id",
                "evidence_keys", "question", "answer", "target_value",
                "question_token_ids", "answer_token_ids"}
    for i, q in enumerate(questions):
        missing = required - set(q.keys())
        if missing:
            raise ValueError(f"question[{i}] missing fields: {missing}")
        qid = q["question_id"]
        if qid in seen:
            raise ValueError(f"duplicate question_id: {qid}")
        seen.add(qid)
        if passage_ids is not None:
            for ek in q["evidence_keys"]:
                if ek not in passage_ids:
                    raise ValueError(
                        f"question[{qid}] evidence_keys references unknown "
                        f"passage_id: {ek}"
                    )


# ── Row → JSON dict ─────────────────────────────────────────────────


def passage_to_dict(p: PassageRow) -> dict:
    """Convert PassageRow to JSON-serializable dict (extras flattened)."""
    d = asdict(p)
    extras = d.pop("extras", {}) or {}
    d.update(extras)
    return d


def question_to_dict(q: QuestionRow) -> dict:
    """Convert QuestionRow to JSON-serializable dict (extras flattened)."""
    d = asdict(q)
    extras = d.pop("extras", {}) or {}
    d.update(extras)
    return d
