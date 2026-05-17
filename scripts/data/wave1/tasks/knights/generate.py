"""Knights & Knaves task — entry point.

Per scenario: 1 preamble passage + 1 passage per statement. Then ask the
identity of one randomly-chosen character.
"""

from __future__ import annotations

from scripts.data.wave1.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data.wave1.common.driver import default_argparser, generate_task
from scripts.data.wave1.tasks.knights.state import (
    KnightsScenario, build_scenario, config_space_size,
)
from scripts.data.wave1.tasks.knights.templates import (
    PREAMBLE_TEMPLATES, STATEMENT_TEMPLATES,
    QUESTION_TEMPLATES, ANSWER_TEMPLATES_KNIGHT, ANSWER_TEMPLATES_KNAVE,
    surface_variant_count,
)


TASK_FAMILY = "knights"


def _scenario_id(scen_idx: int) -> str:
    return f"knights_scen_{scen_idx:06d}"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"{_scenario_id(scen_idx)}_p{k:02d}"


def _question_id(scen_idx: int) -> str:
    return f"q_knights_{scen_idx:06d}"


def render_passages(scen: KnightsScenario, rng):
    """Emit one preamble passage + one passage per statement."""
    # Preamble.
    preamble = rng.choice(PREAMBLE_TEMPLATES)
    yield PassageDraft(
        passage_id=_passage_id(scen.scenario_idx, 0),
        passage_type="knights_preamble",
        text=preamble,
        extras={"scenario_id": _scenario_id(scen.scenario_idx)},
    )
    # Statement passages.
    for k, stmt in enumerate(scen.statements, start=1):
        tmpl_choices = STATEMENT_TEMPLATES[stmt.stmt_type]
        tmpl = rng.choice(tmpl_choices)
        text = tmpl.format(speaker=scen.names[stmt.speaker_idx],
                           target=scen.names[stmt.target_idx])
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, k),
            passage_type="knights_statement",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "speaker": scen.names[stmt.speaker_idx],
                    "target": scen.names[stmt.target_idx],
                    "stmt_type": stmt.stmt_type},
        )


def enumerate_questions(scen: KnightsScenario, rng):
    """Ask the identity of one character. Evidence = preamble + ALL statements
    (you need them all to solve the puzzle)."""
    target_idx = rng.randrange(scen.n_chars)
    target = scen.names[target_idx]
    is_knight = scen.roles[target_idx]

    q_tmpl = rng.choice(QUESTION_TEMPLATES)
    a_tmpl_list = ANSWER_TEMPLATES_KNIGHT if is_knight else ANSWER_TEMPLATES_KNAVE
    a_tmpl = rng.choice(a_tmpl_list)
    q_text = q_tmpl.format(target=target)
    a_text = a_tmpl.format(target=target)
    target_value = "knight" if is_knight else "knave"

    # All passages are evidence (puzzle requires every statement).
    all_pids = [_passage_id(scen.scenario_idx, k)
                for k in range(len(scen.statements) + 1)]

    yield QuestionDraft(
        question_type="identity_of",
        question_id=_question_id(scen.scenario_idx),
        evidence_keys=all_pids,
        question_text=q_text,
        answer_text=a_text,
        target_value=target_value,
        extras={"scenario_id": _scenario_id(scen.scenario_idx),
                "target_name": target,
                "true_roles": dict(zip(scen.names, scen.roles))},
    )


def verify(scen: KnightsScenario, q: QuestionDraft) -> bool:
    """Sanity check: target_value matches the unique consistent assignment."""
    target_name = q.extras["target_name"]
    idx = scen.names.index(target_name)
    expected = "knight" if scen.roles[idx] else "knave"
    return q.target_value == expected


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_chars_range": (3, 4)},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
