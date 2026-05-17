"""Revisions task — entry point.

Per scenario: emit one passage per (attr, version) step in the plan
(initial or revision). Then ask a question about an attribute's current
value or revision count.
"""

from __future__ import annotations

from scripts.data.wave1.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data.wave1.common.driver import default_argparser, generate_task
from scripts.data.wave1.common.text import count_word
from scripts.data.wave1.tasks.revisions.state import (
    RevisionsScenario, build_scenario, config_space_size,
)
from scripts.data.wave1.tasks.revisions.templates import (
    INITIAL_TEMPLATES, REVISION_TEMPLATES,
    CURRENT_VALUE_Q, CURRENT_VALUE_A,
    REVISION_COUNT_Q, REVISION_COUNT_A_ZERO,
    REVISION_COUNT_A_ONE, REVISION_COUNT_A_MANY,
    surface_variant_count,
)


TASK_FAMILY = "revisions"


def _scenario_id(scen_idx: int) -> str:
    return f"rev_scen_{scen_idx:06d}"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"{_scenario_id(scen_idx)}_p{k:02d}"


def _question_id(scen_idx: int, q_idx: int) -> str:
    return f"q_rev_{scen_idx:06d}_q{q_idx:02d}"


def render_passages(scen: RevisionsScenario, rng):
    """Emit per the scenario's pre-computed plan."""
    for k, (kind, attr_idx, version_idx) in enumerate(scen.passage_emit_plan):
        attr = scen.attributes[attr_idx]
        if kind == "init":
            tmpl = rng.choice(INITIAL_TEMPLATES)
            text = tmpl.format(project=scen.project_name, attr=attr.label,
                               value=attr.history[0])
            yield PassageDraft(
                passage_id=_passage_id(scen.scenario_idx, k),
                passage_type="revisions_initial",
                text=text,
                extras={"scenario_id": _scenario_id(scen.scenario_idx),
                        "attr_label": attr.label,
                        "version_idx": 0, "value": attr.history[0]},
            )
        else:  # revise
            tmpl = rng.choice(REVISION_TEMPLATES)
            text = tmpl.format(project=scen.project_name, attr=attr.label,
                               old_value=attr.history[version_idx - 1],
                               new_value=attr.history[version_idx])
            yield PassageDraft(
                passage_id=_passage_id(scen.scenario_idx, k),
                passage_type="revisions_update",
                text=text,
                extras={"scenario_id": _scenario_id(scen.scenario_idx),
                        "attr_label": attr.label,
                        "version_idx": version_idx,
                        "value": attr.history[version_idx]},
            )


def _evidence_for_attr(scen: RevisionsScenario, attr_idx: int) -> list[str]:
    """All passage_ids that touch the given attribute."""
    out = []
    for k, (kind, ai, _v) in enumerate(scen.passage_emit_plan):
        if ai == attr_idx:
            out.append(_passage_id(scen.scenario_idx, k))
    return out


def enumerate_questions(scen: RevisionsScenario, rng):
    """Two questions per scenario, sampled across types.

    De-duplicates by (q_type, attr_idx) so the same attribute can't be queried
    by the same question type twice within a scenario.
    """
    seen: set[tuple[str, int]] = set()
    emitted = 0
    # Up to 8 attempts to fill 2 distinct slots.
    for _ in range(8):
        if emitted >= 2:
            return
        q_type = rng.choice(["current_value", "current_value", "how_many_revisions"])
        attr_idx = rng.randrange(len(scen.attributes))
        key = (q_type, attr_idx)
        if key in seen:
            continue
        seen.add(key)
        attr = scen.attributes[attr_idx]
        evidence = _evidence_for_attr(scen, attr_idx)
        q_idx = emitted
        emitted += 1
        if q_type == "current_value":
            current = attr.history[-1]
            q_tmpl = rng.choice(CURRENT_VALUE_Q)
            a_tmpl = rng.choice(CURRENT_VALUE_A)
            q_text = q_tmpl.format(attr=attr.label, project=scen.project_name)
            a_text = a_tmpl.format(attr=attr.label, project=scen.project_name,
                                   value=current)
            target = current
        else:  # how_many_revisions
            n_rev = len(attr.history) - 1
            q_tmpl = rng.choice(REVISION_COUNT_Q)
            q_text = q_tmpl.format(attr=attr.label, project=scen.project_name)
            if n_rev == 0:
                a_text = rng.choice(REVISION_COUNT_A_ZERO).format(
                    attr=attr.label, project=scen.project_name)
                target = "zero"
            elif n_rev == 1:
                a_text = rng.choice(REVISION_COUNT_A_ONE).format(
                    attr=attr.label, project=scen.project_name)
                target = "one"
            else:
                a_text = rng.choice(REVISION_COUNT_A_MANY).format(
                    attr=attr.label, project=scen.project_name, n=count_word(n_rev))
                target = count_word(n_rev)
        yield QuestionDraft(
            question_type=q_type,
            question_id=_question_id(scen.scenario_idx, q_idx),
            evidence_keys=evidence,
            question_text=q_text,
            answer_text=a_text,
            target_value=target,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "attr_label": attr.label,
                    "attr_idx": attr_idx},
        )


def verify(scen: RevisionsScenario, q: QuestionDraft) -> bool:
    attr_idx = q.extras["attr_idx"]
    attr = scen.attributes[attr_idx]
    if q.question_type == "current_value":
        return q.target_value == attr.history[-1]
    if q.question_type == "how_many_revisions":
        n = len(attr.history) - 1
        expected = count_word(n) if n > 1 else ("zero" if n == 0 else "one")
        return q.target_value == expected
    return True


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_attrs_range": (2, 5), "max_revisions_per_attr": 2},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
