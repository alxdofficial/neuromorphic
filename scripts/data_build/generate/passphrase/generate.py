"""Passphrase task — entry point.

Renders one carrier passage + N distractor passages per scenario, plus
one verbatim-recall question per scenario.

Usage:
    python -m scripts.data_build.generate.passphrase.generate \\
        --output-dir data/wave1/passphrase \\
        --n-scenarios 1000
"""

from __future__ import annotations

from scripts.data_build.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data_build.common.driver import default_argparser, generate_task
from scripts.data_build.generate.passphrase.state import (
    PassphraseScenario, build_scenario, config_space_size,
)
from scripts.data_build.generate.passphrase.templates import (
    PASSAGE_TEMPLATES, DISTRACTOR_TEMPLATES,
    QUESTION_TEMPLATES, ANSWER_TEMPLATES,
    surface_variant_count,
)


TASK_FAMILY = "passphrase"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"pp_{scen_idx:06d}_p{k:02d}"


def _question_id(scen_idx: int) -> str:
    return f"q_pp_{scen_idx:06d}"


def render_passages(scen: PassphraseScenario, rng):
    """Emit one carrier passage + N distractor passages."""
    # Carrier (the one the question is about).
    p_tmpl = rng.choice(PASSAGE_TEMPLATES)
    carrier_text = p_tmpl.format(speaker=scen.speaker, phrase=scen.phrase)
    yield PassageDraft(
        passage_id=_passage_id(scen.scenario_idx, 0),
        passage_type="passphrase_carrier",
        text=carrier_text,
        extras={"speaker": scen.speaker, "phrase": scen.phrase},
    )
    # Distractors (same speaker pool, unrelated content).
    for i, d_speaker in enumerate(scen.distractor_speakers, start=1):
        d_tmpl = rng.choice(DISTRACTOR_TEMPLATES)
        text = d_tmpl.format(speaker=d_speaker)
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, i),
            passage_type="distractor",
            text=text,
            extras={"speaker": d_speaker},
        )


def enumerate_questions(scen: PassphraseScenario, rng):
    """One verbatim-recall question per scenario."""
    q_tmpl = rng.choice(QUESTION_TEMPLATES)
    a_tmpl = rng.choice(ANSWER_TEMPLATES)
    yield QuestionDraft(
        question_type="verbatim_recall",
        question_id=_question_id(scen.scenario_idx),
        evidence_keys=[_passage_id(scen.scenario_idx, 0)],
        question_text=q_tmpl.format(speaker=scen.speaker),
        answer_text=a_tmpl.format(speaker=scen.speaker, phrase=scen.phrase),
        target_value=scen.phrase,
    )


def verify(scen: PassphraseScenario, q: QuestionDraft) -> bool:
    """Sanity check: the target_value should equal the scenario's phrase."""
    return q.target_value == scen.phrase


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_distractors": 2},
)


if __name__ == "__main__":
    ap = default_argparser(description=__doc__)
    args = ap.parse_args()
    generate_task(GEN, args)
