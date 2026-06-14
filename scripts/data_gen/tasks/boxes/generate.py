"""Boxes task — entry point.

Renders initial-state passages (one per box) + one passage per mutation
op, then asks a final-state question for one box.
"""

from __future__ import annotations

from scripts.data_gen.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data_gen.common.driver import default_argparser, generate_task
from scripts.data_gen.common.text import with_indef, indef_list, cap_first
from scripts.data_gen.tasks.boxes.state import (
    BoxesScenario, build_scenario, config_space_size,
)
from scripts.data_gen.tasks.boxes.templates import (
    INIT_TEMPLATES, ADD_TEMPLATES, REMOVE_TEMPLATES, MOVE_TEMPLATES,
    QUESTION_TEMPLATES, ANSWER_TEMPLATES_NONEMPTY, ANSWER_TEMPLATES_EMPTY,
    surface_variant_count,
)


TASK_FAMILY = "boxes"


def _scenario_id(scen_idx: int) -> str:
    return f"box_scen_{scen_idx:06d}"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"{_scenario_id(scen_idx)}_p{k:02d}"


def _question_id(scen_idx: int) -> str:
    return f"q_box_{scen_idx:06d}"


def render_passages(scen: BoxesScenario, rng):
    """Emit one initial-state passage per box, then one passage per op."""
    p_idx = 0
    # Initial state passages.
    for b in range(scen.n_boxes):
        tmpl = rng.choice(INIT_TEMPLATES)
        text = tmpl.format(n=b + 1, item=with_indef(scen.initial_items[b]))
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, p_idx),
            passage_type="boxes_init",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "op_type": "init", "boxes_touched": [b + 1]},
        )
        p_idx += 1
    # Op passages.
    for op in scen.ops:
        if op.op_type == "add":
            tmpl = rng.choice(ADD_TEMPLATES)
            text = cap_first(tmpl.format(item_a_an=with_indef(op.item), n=op.dst_box))
            touched = [op.dst_box]
        elif op.op_type == "remove":
            tmpl = rng.choice(REMOVE_TEMPLATES)
            text = cap_first(tmpl.format(item=op.item, n=op.src_box))
            touched = [op.src_box]
        else:  # move
            tmpl = rng.choice(MOVE_TEMPLATES)
            text = cap_first(tmpl.format(item=op.item, src=op.src_box, dst=op.dst_box))
            touched = [op.src_box, op.dst_box]
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, p_idx),
            passage_type=f"boxes_op_{op.op_type}",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "op_type": op.op_type, "boxes_touched": touched},
        )
        p_idx += 1


def enumerate_questions(scen: BoxesScenario, rng):
    """One question per scenario: final state of a randomly-chosen box."""
    # Prefer boxes that had at least one mutation op.
    op_boxes = [b for b in range(scen.n_boxes)
                if len(scen.per_box_touched_ops[b]) > 1]
    candidates = op_boxes if op_boxes else list(range(scen.n_boxes))
    b = rng.choice(candidates)
    final_contents = sorted(scen.final_state[b])
    q_tmpl = rng.choice(QUESTION_TEMPLATES)
    if final_contents:
        a_tmpl = rng.choice(ANSWER_TEMPLATES_NONEMPTY)
        a_text = a_tmpl.format(n=b + 1, contents=indef_list(final_contents))
        target = ", ".join(final_contents)
    else:
        a_tmpl = rng.choice(ANSWER_TEMPLATES_EMPTY)
        a_text = a_tmpl.format(n=b + 1)
        target = "empty"
    evidence = [_passage_id(scen.scenario_idx, k)
                for k in scen.per_box_touched_ops[b]]
    yield QuestionDraft(
        question_type="final_state",
        question_id=_question_id(scen.scenario_idx),
        evidence_keys=evidence,
        question_text=q_tmpl.format(n=b + 1),
        answer_text=a_text,
        target_value=target,
        extras={"scenario_id": _scenario_id(scen.scenario_idx),
                "box_number": b + 1},
    )


def verify(scen: BoxesScenario, q: QuestionDraft) -> bool:
    """Re-apply ops and confirm target matches our state-machine simulation."""
    box_num = q.extras["box_number"]
    final = sorted(scen.final_state[box_num - 1])
    if not final:
        return q.target_value == "empty"
    return q.target_value == ", ".join(final)


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_boxes_range": (3, 5), "max_ops": 5},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
