"""Calendar task — entry point.

Per scenario: emit one passage per event. Generate 2-3 questions of
varying types (free_at, conflict_with, busy_count_on, next_event_on).
"""

from __future__ import annotations

from scripts.data.wave1.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)
from scripts.data.wave1.common.driver import default_argparser, generate_task
from scripts.data.wave1.common.text import (
    with_indef, hour_12h, duration_phrase, count_word,
)
from scripts.data.wave1.tasks.calendar.state import (
    CalendarScenario, CalendarEvent, DAYS, build_scenario, config_space_size,
)
from scripts.data.wave1.tasks.calendar.templates import (
    PASSAGE_TEMPLATES, surface_variant_count,
)


TASK_FAMILY = "calendar"
QUESTIONS_PER_SCENARIO = 3


def _scenario_id(scen_idx: int) -> str:
    return f"cal_scen_{scen_idx:06d}"


def _passage_id(scen_idx: int, ev_idx: int) -> str:
    return f"{_scenario_id(scen_idx)}_e{ev_idx:02d}"


def _question_id(scen_idx: int, q_idx: int) -> str:
    return f"q_cal_{scen_idx:06d}_q{q_idx:02d}"


def render_passages(scen: CalendarScenario, rng):
    """One passage per event."""
    for ev_idx, ev in enumerate(scen.events):
        tmpl = rng.choice(PASSAGE_TEMPLATES)
        text = tmpl.format(
            day=ev.day,
            time_str=hour_12h(ev.hour),
            event_a_an=with_indef(ev.name),
            duration=str(ev.duration),
            dur_phrase=duration_phrase(ev.duration),
        )
        yield PassageDraft(
            passage_id=_passage_id(scen.scenario_idx, ev_idx),
            passage_type="calendar_event",
            text=text,
            extras={"scenario_id": _scenario_id(scen.scenario_idx),
                    "event_day": ev.day, "event_hour": ev.hour,
                    "event_duration": ev.duration, "event_name": ev.name},
        )


def enumerate_questions(scen: CalendarScenario, rng):
    events = scen.events
    event_pids = [_passage_id(scen.scenario_idx, i) for i in range(len(events))]

    seen: set[tuple] = set()    # (q_type, frozen params) — per-scenario de-dup
    emitted = 0
    # Up to 4× attempts so de-dup'd scenarios still hit QUESTIONS_PER_SCENARIO.
    for _ in range(QUESTIONS_PER_SCENARIO * 4):
        if emitted >= QUESTIONS_PER_SCENARIO:
            return
        q_type = rng.choice([
            "free_at", "free_at",
            "conflict_with",
            "busy_count_on",
            "next_event_on",
        ])
        result = _make_question(q_type, scen, event_pids, rng)
        if result is None:
            continue
        question_text, answer_text, target, evidence, params = result
        key = (q_type, tuple(sorted(params.items())))
        if key in seen:
            continue
        seen.add(key)
        extras = {"scenario_id": _scenario_id(scen.scenario_idx)}
        extras.update(params)
        yield QuestionDraft(
            question_type=q_type,
            question_id=_question_id(scen.scenario_idx, emitted),
            evidence_keys=evidence,
            question_text=question_text,
            answer_text=answer_text,
            target_value=target,
            extras=extras,
        )
        emitted += 1


def _make_question(q_type, scen, event_pids, rng):
    events = scen.events
    if q_type == "free_at":
        if rng.random() < 0.5 and events:
            ev = rng.choice(events)
            day, hour = ev.day, ev.hour + rng.randint(0, ev.duration - 1)
        else:
            day = rng.choice(DAYS)
            hour = rng.randint(7, 21)
        busy = [e for e in events
                if e.day == day and e.hour <= hour < e.hour + e.duration]
        if busy:
            e = busy[0]
            ans = (f"No, I'm not free — I have {with_indef(e.name)} on "
                   f"{day} at {hour_12h(e.hour)}.")
            tgt = "no"
            evidence = [event_pids[events.index(e)]]
        else:
            ans = f"Yes, I'm free on {day} at {hour_12h(hour)}."
            tgt = "yes"
            evidence = list(event_pids)
        params = {"q_day": day, "q_hour": hour}
        return (f"Am I free on {day} at {hour_12h(hour)}?", ans, tgt, evidence, params)

    if q_type == "conflict_with":
        day = rng.choice(DAYS)
        hour = rng.randint(7, 21)
        duration = rng.choice([1, 2])
        conflicts = [e for e in events
                     if e.day == day and e.hour < hour + duration
                     and hour < e.hour + e.duration]
        if not conflicts:
            ans = (f"Nothing conflicts — I'm free on {day} from "
                   f"{hour_12h(hour)} for {duration_phrase(duration)}.")
            tgt = "nothing"
            evidence = list(event_pids)
        else:
            e = conflicts[0]
            ans = (f"That conflicts with {with_indef(e.name)} on "
                   f"{day} at {hour_12h(e.hour)}.")
            tgt = e.name
            evidence = [event_pids[events.index(e)]]
        q = (f"If I want to book something on {day} at {hour_12h(hour)} "
             f"for {duration_phrase(duration)}, would it conflict with anything?")
        params = {"q_day": day, "q_hour": hour, "q_duration": duration}
        return (q, ans, tgt, evidence, params)

    if q_type == "busy_count_on":
        day = rng.choice(DAYS)
        count = sum(1 for e in events if e.day == day)
        cw = count_word(count)
        if count == 0:
            ans = f"I have nothing on {day}."
        elif count == 1:
            ans = f"I have one event on {day}."
        else:
            ans = f"I have {cw} events on {day}."
        evidence = [event_pids[events.index(e)] for e in events if e.day == day]
        if not evidence:
            evidence = list(event_pids)
        params = {"q_day": day}
        return (f"How many events do I have on {day}?", ans, cw, evidence, params)

    if q_type == "next_event_on":
        day = rng.choice(DAYS)
        day_events = sorted([e for e in events if e.day == day],
                            key=lambda e: e.hour)
        if not day_events:
            return None
        e = day_events[0]
        ans = (f"My next event on {day} is {with_indef(e.name)} "
               f"at {hour_12h(e.hour)}.")
        params = {"q_day": day}
        return (f"What is the first event scheduled on {day}?",
                ans, e.name, [event_pids[events.index(e)]], params)
    return None


def verify(scen: CalendarScenario, q: QuestionDraft) -> bool:
    """Re-derive answer from scenario state + query params in extras."""
    events = scen.events
    if q.question_type == "free_at":
        day, hour = q.extras["q_day"], q.extras["q_hour"]
        busy = any(e.day == day and e.hour <= hour < e.hour + e.duration
                   for e in events)
        return q.target_value == ("no" if busy else "yes")
    if q.question_type == "conflict_with":
        day = q.extras["q_day"]
        hour = q.extras["q_hour"]
        duration = q.extras["q_duration"]
        conflicts = [e for e in events
                     if e.day == day and e.hour < hour + duration
                     and hour < e.hour + e.duration]
        return (q.target_value == "nothing" and not conflicts) or \
               (bool(conflicts) and q.target_value == conflicts[0].name)
    if q.question_type == "busy_count_on":
        day = q.extras["q_day"]
        count = sum(1 for e in events if e.day == day)
        return q.target_value == count_word(count)
    if q.question_type == "next_event_on":
        day = q.extras["q_day"]
        day_events = sorted([e for e in events if e.day == day],
                            key=lambda e: e.hour)
        return bool(day_events) and q.target_value == day_events[0].name
    return True


GEN = TaskGenerator(
    task_family=TASK_FAMILY,
    build_scenario=build_scenario,
    render_passages=render_passages,
    enumerate_questions=enumerate_questions,
    config_space_size=config_space_size,
    surface_variant_count=surface_variant_count,
    verify=verify,
    build_kwargs={"n_events": 6},
)


if __name__ == "__main__":
    args = default_argparser(description=__doc__).parse_args()
    generate_task(GEN, args)
