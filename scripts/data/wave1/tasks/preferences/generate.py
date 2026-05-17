"""Preferences task — entry point.

Per scenario: one statement passage per preference + optional one
cancellation passage. Then a question targeting one preference (biased
toward cancelled ones when present).
"""

from __future__ import annotations

from scripts.data.wave1.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data.wave1.common.driver import default_argparser, generate_task
from scripts.data.wave1.tasks.preferences.state import (
    PreferencesScenario, build_scenario, config_space_size,
)
from scripts.data.wave1.tasks.preferences.templates import (
    STATEMENT_TEMPLATES, CANCEL_TEMPLATES,
    QUESTION_TEMPLATES, ANSWER_TEMPLATES, surface_variant_count,
)


TASK_FAMILY = "preferences"


def _scenario_id(scen_idx: int) -> str:
    return f"pref_user_{scen_idx:06d}"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"{_scenario_id(scen_idx)}_p{k:02d}"


def _question_id(scen_idx: int) -> str:
    return f"q_pref_{scen_idx:06d}"


def render_passages(scen: PreferencesScenario, rng):
    """Emit initial statement for each pref + (optional) cancellation passages."""
    p_idx = 0
    for pref in scen.preferences:
        tmpl = rng.choice(STATEMENT_TEMPLATES)
        text = tmpl.format(user=scen.user_name, value=pref.initial_value,
                           domain=pref.domain)
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, p_idx),
            passage_type="preference_statement",
            text=text,
            extras={"user_id": _scenario_id(scen.scenario_idx),
                    "user_name": scen.user_name, "domain": pref.domain,
                    "stated_value": pref.initial_value},
        )
        p_idx += 1
    for pref in scen.preferences:
        if pref.cancelled and pref.revised_value:
            tmpl = rng.choice(CANCEL_TEMPLATES)
            text = tmpl.format(user=scen.user_name, domain=pref.domain,
                               old_value=pref.initial_value,
                               new_value=pref.revised_value)
            yield PassageDraft(
                passage_id=_passage_id(scen.scenario_idx, p_idx),
                passage_type="preference_cancellation",
                text=text,
                extras={"user_id": _scenario_id(scen.scenario_idx),
                        "user_name": scen.user_name, "domain": pref.domain,
                        "stated_value": pref.revised_value,
                        "cancelled_old_value": pref.initial_value},
            )
            p_idx += 1


def _final_value(pref) -> str:
    return pref.revised_value if pref.cancelled else pref.initial_value


def enumerate_questions(scen: PreferencesScenario, rng):
    if not scen.preferences:
        return
    # Bias 50% toward cancelled-domain queries if any exist.
    cancelled = [p for p in scen.preferences if p.cancelled]
    if cancelled and rng.random() < 0.5:
        pref = rng.choice(cancelled)
        q_type = "preference_cancelled"
    else:
        pref = rng.choice(scen.preferences)
        q_type = "preference_recall"
    value = _final_value(pref)

    # Build evidence: all passages that touch this domain.
    p_idx = 0
    evidence = []
    for p in scen.preferences:
        if p.domain == pref.domain:
            evidence.append(_passage_id(scen.scenario_idx, p_idx))
        p_idx += 1
    # Cancellation passages come after all statements.
    for p in scen.preferences:
        if p.cancelled and p.revised_value:
            if p.domain == pref.domain:
                evidence.append(_passage_id(scen.scenario_idx, p_idx))
            p_idx += 1

    q_tmpl = rng.choice(QUESTION_TEMPLATES)
    a_tmpl = rng.choice(ANSWER_TEMPLATES)
    q_text = q_tmpl.format(user=scen.user_name, domain=pref.domain)
    a_text = a_tmpl.format(user=scen.user_name, value=value, domain=pref.domain)
    yield QuestionDraft(
        question_type=q_type,
        question_id=_question_id(scen.scenario_idx),
        evidence_keys=evidence,
        question_text=q_text,
        answer_text=a_text,
        target_value=value,
        extras={"user_id": _scenario_id(scen.scenario_idx),
                "domain": pref.domain},
    )


def verify(scen: PreferencesScenario, q: QuestionDraft) -> bool:
    domain = q.extras["domain"]
    pref = next(p for p in scen.preferences if p.domain == domain)
    return q.target_value == _final_value(pref)


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_prefs_range": (3, 5), "cancellation_rate": 0.3},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
