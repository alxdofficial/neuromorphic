"""Pre-tokenization passage + question drafts, and the TaskGenerator
protocol that task-family generators implement.

The driver (common/driver.py) calls a TaskGenerator's hooks, takes back
PassageDraft/QuestionDraft objects, tokenizes them, and emits the final
PassageRow/QuestionRow to JSONL.

This separates GENERATION (task-specific) from EMISSION (shared) so each
task only has to declare its state model + templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class PassageDraft:
    """Pre-tokenization passage. Each draft becomes one row in passages.jsonl
    after the driver tokenizes and tags with task_family.

    `passage_id` must be globally unique across the composite dataset.
    Tasks should prefix with the task name (e.g. `bio_pi_0042_s0`,
    `triage_scen_0017_req_3`).

    `extras` is a free-form dict for task-specific debug/analysis fields
    (the driver flattens these to top-level keys when emitting JSON).
    """
    passage_id: str
    passage_type: str
    text: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionDraft:
    """Pre-tokenization question + answer.

    `evidence_keys` lists the passage_ids whose passages must be in the
    chunk for the answer to be derivable.

    `target_value` is the short-form answer string for accuracy scoring
    (e.g. just the number, or just the entity name).

    `extras` for task-specific fields (relation_chain, scenario_id, etc.).
    """
    question_type: str
    question_id: str
    evidence_keys: list[str]
    question_text: str
    answer_text: str
    target_value: str
    extras: dict[str, Any] = field(default_factory=dict)


# Task generator hooks.
# Each task family implements three functions matching these signatures
# and wires them into a TaskGenerator instance (below).

BuildScenarioFn = Callable[..., Any]
"""(rng: random.Random, scenario_idx: int, **task_kwargs) -> Scenario.
Scenario type is task-defined (anything with the right shape)."""

RenderPassagesFn = Callable[[Any, Any], Iterable[PassageDraft]]
"""(scenario, rng: random.Random) -> Iterable[PassageDraft]."""

EnumerateQuestionsFn = Callable[[Any, Any], Iterable[QuestionDraft]]
"""(scenario, rng: random.Random) -> Iterable[QuestionDraft]."""

VerifyFn = Callable[[Any, QuestionDraft], bool]
"""Optional: (scenario, question_draft) -> bool. Re-derives the answer
from scenario state and asserts it matches. Used to catch generator
bugs that produce wrong answers."""

ConfigSpaceFn = Callable[[], int]
"""() -> int. Back-of-envelope count of distinct scenario states this
generator can produce."""

SurfaceVariantsFn = Callable[[], int]
"""() -> int. Back-of-envelope count of distinct surface renderings
PER scenario state (product of passage templates × question templates
× phrasing variants)."""


@dataclass
class TaskGenerator:
    """The complete spec for a task family.

    The driver takes one of these + a count of scenarios to generate,
    and produces passages.jsonl + questions.jsonl.

    Each task creates a TaskGenerator instance and the driver does the rest.
    """
    task_family: str
    build_scenario: BuildScenarioFn
    render_passages: RenderPassagesFn
    enumerate_questions: EnumerateQuestionsFn
    config_space_size: ConfigSpaceFn
    surface_variant_count: SurfaceVariantsFn
    verify: VerifyFn | None = None
    # Task-specific kwargs forwarded to build_scenario each scenario.
    build_kwargs: dict[str, Any] = field(default_factory=dict)
