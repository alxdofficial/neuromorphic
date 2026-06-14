"""Theory of Mind — generator entry point.

Per scenario: 1 preamble passage + 1 passage per event.
Question targets: belief vs. reality vs. witness.
"""

from __future__ import annotations

from scripts.data_gen.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data_gen.common.driver import default_argparser, generate_task
from scripts.data_gen.common.text import comma_list
from scripts.data_gen.tasks.theory_of_mind.state import (
    TomScenario, build_scenario, config_space_size,
)
from scripts.data_gen.tasks.theory_of_mind.templates import (
    PREAMBLE_TEMPLATES, PLACE_TEMPLATES, LEAVE_TEMPLATES,
    MOVE_TEMPLATES, RETURN_TEMPLATES,
    WHERE_BELIEF_Q, WHERE_BELIEF_A,
    WHERE_ACTUALLY_Q, WHERE_ACTUALLY_A,
    HAS_SEEN_Q, HAS_SEEN_A_YES, HAS_SEEN_A_NO,
    surface_variant_count,
)


TASK_FAMILY = "theory_of_mind"


def _scenario_id(scen_idx: int) -> str:
    return f"tom_scen_{scen_idx:06d}"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"{_scenario_id(scen_idx)}_p{k:02d}"


def _question_id(scen_idx: int, q_idx: int) -> str:
    return f"q_tom_{scen_idx:06d}_q{q_idx:02d}"


def render_passages(scen: TomScenario, rng):
    """Preamble + 1 passage per event."""
    preamble_tmpl = rng.choice(PREAMBLE_TEMPLATES)
    preamble_text = preamble_tmpl.format(chars_list=comma_list(list(scen.chars)))
    yield PassageDraft(
        passage_id=_passage_id(scen.scenario_idx, 0),
        passage_type="tom_preamble",
        text=preamble_text,
        extras={"scenario_id": _scenario_id(scen.scenario_idx)},
    )
    for k, evt in enumerate(scen.events, start=1):
        if evt.event_type == "place":
            text = rng.choice(PLACE_TEMPLATES).format(
                actor=evt.actor, obj=evt.obj, dst=evt.dst)
        elif evt.event_type == "leave":
            text = rng.choice(LEAVE_TEMPLATES).format(actor=evt.actor)
        elif evt.event_type == "move":
            text = rng.choice(MOVE_TEMPLATES).format(
                actor=evt.actor, obj=evt.obj, src=evt.src, dst=evt.dst)
        elif evt.event_type == "return":
            text = rng.choice(RETURN_TEMPLATES).format(actor=evt.actor)
        else:
            raise ValueError(f"Unknown event type: {evt.event_type}")
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, k),
            passage_type=f"tom_{evt.event_type}",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "actor": evt.actor,
                    "obj": evt.obj, "src": evt.src, "dst": evt.dst,
                    "witnesses": list(evt.witnesses)},
        )


def _saw_move(scen: TomScenario, char: str, obj: str) -> bool:
    for e in scen.events:
        if e.event_type == "move" and e.obj == obj and char in e.witnesses:
            return True
    return False


def enumerate_questions(scen: TomScenario, rng):
    """Two questions per scenario, distributed across types."""
    obj = scen.objects[0]
    # Find the character who has the false belief (was absent during the move)
    # and the one who has the true belief.
    move_evt = next(e for e in scen.events
                    if e.event_type == "move" and e.obj == obj)
    absent_char = next(c for c in scen.chars if c not in move_evt.witnesses)
    present_char = move_evt.actor   # mover always witnesses

    all_pids = [_passage_id(scen.scenario_idx, k)
                for k in range(len(scen.events) + 1)]

    # Question 1: false-belief probe (most interesting one).
    q1_char = absent_char
    q1_container = scen.belief[(q1_char, obj)]
    q1_q = rng.choice(WHERE_BELIEF_Q).format(char=q1_char, obj=obj)
    q1_a = rng.choice(WHERE_BELIEF_A).format(
        char=q1_char, obj=obj, container=q1_container)
    yield QuestionDraft(
        question_type="where_belief",
        question_id=_question_id(scen.scenario_idx, 0),
        evidence_keys=all_pids,
        question_text=q1_q,
        answer_text=q1_a,
        target_value=q1_container,
        extras={"scenario_id": _scenario_id(scen.scenario_idx),
                "char": q1_char, "obj": obj},
    )

    # Question 2: alternate between reality-check and witness-check.
    q2_type = rng.choice(["where_actually", "has_seen"])
    if q2_type == "where_actually":
        container = scen.final_location[obj]
        q_text = rng.choice(WHERE_ACTUALLY_Q).format(obj=obj)
        a_text = rng.choice(WHERE_ACTUALLY_A).format(obj=obj, container=container)
        target = container
        extras = {"scenario_id": _scenario_id(scen.scenario_idx), "obj": obj}
    else:  # has_seen — sample either char so yes/no targets are balanced.
        char = rng.choice([absent_char, present_char])
        saw = _saw_move(scen, char, obj)
        q_text = rng.choice(HAS_SEEN_Q).format(char=char, obj=obj)
        if saw:
            a_text = rng.choice(HAS_SEEN_A_YES).format(char=char, obj=obj)
            target = "yes"
        else:
            a_text = rng.choice(HAS_SEEN_A_NO).format(char=char, obj=obj)
            target = "no"
        extras = {"scenario_id": _scenario_id(scen.scenario_idx),
                  "char": char, "obj": obj}

    yield QuestionDraft(
        question_type=q2_type,
        question_id=_question_id(scen.scenario_idx, 1),
        evidence_keys=all_pids,
        question_text=q_text,
        answer_text=a_text,
        target_value=target,
        extras=extras,
    )


def verify(scen: TomScenario, q: QuestionDraft) -> bool:
    if q.question_type == "where_belief":
        return q.target_value == scen.belief[(q.extras["char"], q.extras["obj"])]
    if q.question_type == "where_actually":
        return q.target_value == scen.final_location[q.extras["obj"]]
    if q.question_type == "has_seen":
        saw = _saw_move(scen, q.extras["char"], q.extras["obj"])
        return q.target_value == ("yes" if saw else "no")
    return True


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_chars": 2, "n_objects": 1},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
