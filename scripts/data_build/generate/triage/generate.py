"""Triage task — entry point.

Emit one passage per request (in arrival order) + one passage per update.
Then enumerate questions about the current state of the queue.
"""

from __future__ import annotations

from scripts.data_build.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data_build.common.driver import default_argparser, generate_task
from scripts.data_build.common.text import comma_list, count_word
from scripts.data_build.generate.triage.state import (
    TriageScenario, Request, build_scenario, config_space_size,
    is_ready, blocking_deps,
)
from scripts.data_build.generate.triage.templates import (
    ARRIVAL_NO_DEP_TEMPLATES, ARRIVAL_DEP_TEMPLATES,
    DONE_UPDATE_TEMPLATES, DEPRIORITIZE_UPDATE_TEMPLATES,
    WHAT_UNBLOCKED_Q, WHAT_BLOCKS_Q, IS_READY_Q, NEXT_PRIORITY_Q,
    surface_variant_count,
)


TASK_FAMILY = "triage"
QUESTIONS_PER_SCENARIO = 2

PRIORITY_RANK = {"urgent": 0, "normal": 1, "low": 2}


def _scenario_id(scen_idx: int) -> str:
    return f"triage_scen_{scen_idx:06d}"


def _passage_id(scen_idx: int, k: int) -> str:
    return f"{_scenario_id(scen_idx)}_p{k:02d}"


def _question_id(scen_idx: int, q_idx: int) -> str:
    return f"q_triage_{scen_idx:06d}_q{q_idx:02d}"


def _deps_phrase(deps: tuple[str, ...]) -> str:
    if len(deps) == 1:
        return f"Request {deps[0]}"
    return "Requests " + comma_list(list(deps))


def render_passages(scen: TriageScenario, rng):
    p_idx = 0
    # Arrival passages.
    for req in scen.requests:
        if req.deps:
            tmpl = rng.choice(ARRIVAL_DEP_TEMPLATES)
            text = tmpl.format(label=req.label, priority=req.priority,
                               description=req.description,
                               deps_phrase=_deps_phrase(req.deps))
        else:
            tmpl = rng.choice(ARRIVAL_NO_DEP_TEMPLATES)
            text = tmpl.format(label=req.label, priority=req.priority,
                               description=req.description)
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, p_idx),
            passage_type="triage_arrival",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "label": req.label, "deps": list(req.deps),
                    "priority": req.priority},
        )
        p_idx += 1
    # Update passages.
    for u in scen.updates:
        if u.new_status == "done":
            tmpl = rng.choice(DONE_UPDATE_TEMPLATES)
        else:
            tmpl = rng.choice(DEPRIORITIZE_UPDATE_TEMPLATES)
        text = tmpl.format(label=u.label)
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, p_idx),
            passage_type=f"triage_update_{u.new_status}",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "label": u.label, "new_status": u.new_status},
        )
        p_idx += 1


def _label_to_passage_idx(scen: TriageScenario) -> dict[str, list[int]]:
    """For each label, list of passage indices that mention it."""
    idx: dict[str, list[int]] = {r.label: [i] for i, r in enumerate(scen.requests)}
    base = len(scen.requests)
    for k, u in enumerate(scen.updates):
        idx.setdefault(u.label, []).append(base + k)
    return idx


def _ready_now(scen: TriageScenario) -> list[Request]:
    """All requests in pending state with all deps done."""
    return [r for r in scen.requests if is_ready(scen, r.label)]


def enumerate_questions(scen: TriageScenario, rng):
    label_to_idx = _label_to_passage_idx(scen)
    all_pids = [_passage_id(scen.scenario_idx, k)
                for k in range(len(scen.requests) + len(scen.updates))]

    emitted = 0
    seen_qa: set[tuple[str, str]] = set()   # (question_type, target_value)
    # Try every q_type at most twice; stop after QUESTIONS_PER_SCENARIO.
    Q_TYPES = ["what_unblocked", "what_blocks", "is_ready", "next_priority"]
    type_pool = list(Q_TYPES)
    rng.shuffle(type_pool)
    type_pool = type_pool * 2          # allow 2 sweeps through types
    for q_type in type_pool:
        if emitted >= QUESTIONS_PER_SCENARIO:
            return
        result = _make_question(q_type, scen, label_to_idx, all_pids, rng)
        if result is None:
            continue
        q_text, a_text, target, evidence = result
        key = (q_type, target)
        if key in seen_qa:
            continue
        seen_qa.add(key)
        yield QuestionDraft(
            question_type=q_type,
            question_id=_question_id(scen.scenario_idx, emitted),
            evidence_keys=evidence,
            question_text=q_text,
            answer_text=a_text,
            target_value=target,
            extras={"scenario_id": _scenario_id(scen.scenario_idx)},
        )
        emitted += 1


def _make_question(q_type, scen, label_to_idx, all_pids, rng):
    if q_type == "what_unblocked":
        ready = _ready_now(scen)
        ready_labels = sorted(r.label for r in ready)
        q_tmpl = rng.choice(WHAT_UNBLOCKED_Q)
        if not ready_labels:
            ans = "Nothing is ready right now — everything pending is blocked or already done."
            target = "nothing"
        elif len(ready_labels) == 1:
            ans = f"Request {ready_labels[0]} is ready to start."
            target = ready_labels[0]
        else:
            ans = "Requests " + comma_list(ready_labels) + " are ready to start."
            target = ", ".join(ready_labels)
        # Evidence: all arrivals + all updates (the model needs the full picture).
        return q_tmpl, ans, target, list(all_pids)

    if q_type == "what_blocks":
        # Pick a pending request with at least one unresolved dep.
        candidates = [r for r in scen.requests
                      if scen.final_status[r.label] == "pending"
                      and blocking_deps(scen, r.label)]
        if not candidates:
            return None
        req = rng.choice(candidates)
        blockers = blocking_deps(scen, req.label)
        q_tmpl = rng.choice(WHAT_BLOCKS_Q)
        q_text = q_tmpl.format(label=req.label)
        if len(blockers) == 1:
            ans = (f"Request {req.label} is blocked by Request {blockers[0]}, "
                   f"which hasn't been completed yet.")
        else:
            ans = (f"Request {req.label} is blocked by Requests "
                   f"{comma_list(blockers)}.")
        target = ", ".join(blockers)
        # Evidence: the asked request + any updates on its deps.
        evidence = []
        evidence.extend(_passage_id(scen.scenario_idx, i)
                        for i in label_to_idx[req.label])
        for d in req.deps:
            evidence.extend(_passage_id(scen.scenario_idx, i)
                            for i in label_to_idx[d])
        # De-dupe while preserving order.
        seen = set(); evidence = [e for e in evidence if not (e in seen or seen.add(e))]
        return q_text, ans, target, evidence

    if q_type == "is_ready":
        req = rng.choice(scen.requests)
        ready = is_ready(scen, req.label)
        q_tmpl = rng.choice(IS_READY_Q)
        q_text = q_tmpl.format(label=req.label)
        status = scen.final_status[req.label]
        if status == "done":
            ans = f"Request {req.label} is already done."
            target = "done"
        elif status == "deprioritized":
            ans = f"Request {req.label} is deprioritized — we're not working on it now."
            target = "deprioritized"
        elif ready:
            ans = f"Yes, Request {req.label} is ready to start."
            target = "yes"
        else:
            blockers = blocking_deps(scen, req.label)
            ans = (f"No, Request {req.label} is still blocked by Request "
                   f"{comma_list(blockers)}.")
            target = "no"
        evidence = []
        evidence.extend(_passage_id(scen.scenario_idx, i)
                        for i in label_to_idx[req.label])
        for d in req.deps:
            evidence.extend(_passage_id(scen.scenario_idx, i)
                            for i in label_to_idx[d])
        seen = set(); evidence = [e for e in evidence if not (e in seen or seen.add(e))]
        return q_text, ans, target, evidence

    if q_type == "next_priority":
        ready = _ready_now(scen)
        q_tmpl = rng.choice(NEXT_PRIORITY_Q)
        if not ready:
            ans = "Nothing is ready right now."
            target = "nothing"
            return q_tmpl, ans, target, list(all_pids)
        # Sort by priority rank then label for determinism.
        ranked = sorted(ready, key=lambda r: (PRIORITY_RANK[r.priority], r.label))
        winner = ranked[0]
        ans = (f"Request {winner.label} ({winner.description}) is the "
               f"highest-priority unblocked request, with priority "
               f"{winner.priority}.")
        target = winner.label
        return q_tmpl, ans, target, list(all_pids)
    return None


def verify(scen: TriageScenario, q: QuestionDraft) -> bool:
    """Re-derive expected target from scenario state for every question type."""
    if q.question_type == "what_unblocked":
        ready = sorted(r.label for r in _ready_now(scen))
        if not ready:
            return q.target_value == "nothing"
        if len(ready) == 1:
            return q.target_value == ready[0]
        return q.target_value == ", ".join(ready)

    if q.question_type == "what_blocks":
        # target_value is comma-joined blocking deps; q's text mentions the
        # blocked request label. We can't recover the label from the rendered
        # question text reliably, so check that target IS a subset of some
        # pending request's blockers.
        for r in scen.requests:
            if scen.final_status[r.label] != "pending":
                continue
            blockers = blocking_deps(scen, r.label)
            if blockers and q.target_value == ", ".join(blockers):
                return True
        return False

    if q.question_type == "is_ready":
        # target ∈ {yes, no, done, deprioritized}; verify it's a legal label.
        if q.target_value not in {"yes", "no", "done", "deprioritized"}:
            return False
        # Cross-check: at least one request must be in a state matching target.
        for r in scen.requests:
            status = scen.final_status[r.label]
            if q.target_value == "done" and status == "done":
                return True
            if q.target_value == "deprioritized" and status == "deprioritized":
                return True
            if q.target_value == "yes" and status == "pending" and is_ready(scen, r.label):
                return True
            if q.target_value == "no" and status == "pending" and not is_ready(scen, r.label):
                return True
        return False

    if q.question_type == "next_priority":
        ready = _ready_now(scen)
        if not ready:
            return q.target_value == "nothing"
        ranked = sorted(ready, key=lambda r: (PRIORITY_RANK[r.priority], r.label))
        return q.target_value == ranked[0].label

    return True


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    # what_unblocked / next_priority need ALL passages as evidence. Cap total
    # at ≤ 8 so the composite sampler at chunk_size=8 doesn't truncate evidence.
    build_kwargs={"n_requests_range": (4, 6), "n_updates_range": (0, 2)},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
