"""Biographical task — entry point.

Builds one fully-populated world per scenario (so n-scenarios=1 by default
gives one world with all its passages + questions). The world.py classes
do the heavy lifting; we wire them into the common driver.
"""

from __future__ import annotations

import argparse
import random

from scripts.data_gen.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data_gen.common.driver import default_argparser, generate_task
from scripts.data_gen.common.text import cap_first

from scripts.data_gen.tasks.biographical.state import (
    BiographicalScenario, build_scenario, config_space_size,
)
from scripts.data_gen.tasks.biographical.templates.passages import (
    render_passage, PERSONAS,
)
from scripts.data_gen.tasks.biographical.templates.questions_atomic import (
    generate_atomic_questions,
)
from scripts.data_gen.tasks.biographical.templates.questions_relational import (
    generate_relational_questions,
)
from scripts.data_gen.tasks.biographical.templates.questions_temporal import (
    generate_temporal_questions,
)
from scripts.data_gen.tasks.biographical.templates.questions_aggregation import (
    generate_aggregation_questions,
)


TASK_FAMILY = "biographical"
ENTITY_PASSAGE_ID_PREFIX = "bio"


def _passage_id(scenario_idx: int, entity_key: str, sample_idx: int) -> str:
    """Global passage_id for an entity's k-th rendering in scenario S."""
    return f"{ENTITY_PASSAGE_ID_PREFIX}_s{scenario_idx:03d}_{entity_key}_s{sample_idx}"


def _question_id(scenario_idx: int, q_idx: int) -> str:
    return f"q_bio_s{scenario_idx:03d}_{q_idx:08d}"


def render_passages(scen: BiographicalScenario, rng):
    """Emit one PassageDraft per (entity, sample_idx) pair.

    sample_idx=0 is the question target — pin it to biographical_paragraph,
    which is the only persona guaranteed to mention every queried attribute.
    Higher sample_idx values rotate through the other personas for paraphrase
    diversity during write-pass distractors.
    """
    for ent in scen.world.entities.values():
        for idx in range(scen.samples_per_entity):
            forced = "biographical_paragraph" if idx == 0 else None
            passage, persona = render_passage(scen.world, ent, rng, persona=forced)
            yield PassageDraft(
                passage_id=_passage_id(scen.scenario_idx, ent.key, idx),
                passage_type=ent.entity_type,
                text=passage,
                extras={
                    "entity_key": ent.key,
                    "sample_idx": idx,
                    "surface_names": list(ent.surface_names),
                    "attrs": dict(ent.attrs),
                    "outgoing_edges": [
                        {"rel": e.rel, "dst": e.dst}
                        for e in scen.world.get_edges(ent.key)
                    ],
                    "passage_persona": persona,
                },
            )


def enumerate_questions(scen: BiographicalScenario, rng):
    """Run all 4 question generators across the world, translating
    evidence_keys from entity_keys → passage_ids."""
    world = scen.world
    atomic = generate_atomic_questions(world, rng, samples_per_attribute=1)
    relational = generate_relational_questions(world, rng, max_hops=2)
    temporal = generate_temporal_questions(world, rng, samples=200)
    aggregation = generate_aggregation_questions(world, rng, samples=200)

    all_questions = atomic + relational + temporal + aggregation
    for q_idx, q in enumerate(all_questions):
        # Translate evidence_keys from entity_keys → passage_ids (use
        # sample_idx=0). Keep the original entity_keys in extras for
        # downstream debugging.
        translated = [
            _passage_id(scen.scenario_idx, ek, 0) for ek in q["evidence_keys"]
        ]
        yield QuestionDraft(
            question_type=q["question_type"],
            question_id=_question_id(scen.scenario_idx, q_idx),
            evidence_keys=translated,
            question_text=cap_first(q["question"]),
            answer_text=cap_first(q["answer"]),
            target_value=str(q.get("target_value", "")),
            extras={
                **{k: v for k, v in q.items()
                   if k not in {"question", "answer", "target_value",
                                "question_type", "question_id",
                                "evidence_keys", "question_token_ids",
                                "answer_token_ids", "question_token_count",
                                "answer_token_count"}},
                "evidence_entity_keys": list(q["evidence_keys"]),
            },
        )


def surface_variant_count() -> int:
    """6 personas × ~3 templates per persona × ~3 questions per attribute × ~2 answers."""
    return len(PERSONAS) * 3 * 3 * 2


def add_biographical_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--n-people", type=int, default=200)
    ap.add_argument("--n-public-figures", type=int, default=30)
    ap.add_argument("--n-orgs", type=int, default=60)
    ap.add_argument("--n-nations", type=int, default=20)
    ap.add_argument("--n-places", type=int, default=40)
    ap.add_argument("--n-events", type=int, default=30)
    ap.add_argument("--n-works", type=int, default=30)
    ap.add_argument("--samples-per-entity", type=int, default=1)


def main():
    ap = default_argparser(description=__doc__)
    add_biographical_args(ap)
    # biographical's natural unit IS one whole world. Default to a single
    # world; bumping n-scenarios produces multiple independent worlds.
    ap.set_defaults(n_scenarios=1)
    args = ap.parse_args()

    GEN = TaskGenerator(
        task_family=TASK_FAMILY,
        build_scenario=build_scenario,
        render_passages=render_passages,
        enumerate_questions=enumerate_questions,
        config_space_size=config_space_size,
        surface_variant_count=surface_variant_count,
        verify=None,    # too expensive at scale; pilots inspect manually
        build_kwargs={
            "n_people": args.n_people,
            "n_public_figures": args.n_public_figures,
            "n_orgs": args.n_orgs,
            "n_nations": args.n_nations,
            "n_places": args.n_places,
            "n_events": args.n_events,
            "n_works": args.n_works,
            "samples_per_entity": args.samples_per_entity,
        },
    )
    generate_task(GEN, args)


if __name__ == "__main__":
    main()
