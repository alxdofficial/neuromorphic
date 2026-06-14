"""Calendar task — scenario data model.

A scenario is a set of N events placed on a 2D (day × hour) grid with
non-overlap constraint. Each event has a type, day, start_hour, and
duration. The model must reason over the grid for conflict/availability/
counting/ordering questions.
"""

from __future__ import annotations

from dataclasses import dataclass


DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# (event_name, default_duration_hours)
EVENT_TYPES = (
    ("team meeting", 1), ("doctor's appointment", 1), ("dentist visit", 1),
    ("client call", 1), ("yoga class", 1), ("tutoring session", 1),
    ("therapy session", 1), ("design review", 2), ("planning session", 2),
    ("interview", 1), ("haircut", 1), ("hiking trip", 4),
    ("dinner with family", 2), ("birthday party", 3), ("piano lesson", 1),
    ("study group", 2), ("book club", 2), ("workshop", 3),
    ("research call", 1), ("phone call with mother", 1),
    ("coffee with a friend", 1), ("personal training session", 1),
    ("guitar lesson", 1),
)


@dataclass
class CalendarEvent:
    day: str
    hour: int          # 24h, 0..23
    duration: int
    name: str


@dataclass
class CalendarScenario:
    scenario_idx: int
    events: list[CalendarEvent]


def _overlaps(e1: CalendarEvent, e2: CalendarEvent) -> bool:
    if e1.day != e2.day:
        return False
    return e1.hour < e2.hour + e2.duration and e2.hour < e1.hour + e1.duration


def build_scenario(
    rng, scenario_idx: int, *, n_events: int = 6,
) -> CalendarScenario | None:
    """Sample n_events non-conflicting events on a Mon-Sun week."""
    placed: list[CalendarEvent] = []
    attempts = 0
    while len(placed) < n_events and attempts < n_events * 30:
        attempts += 1
        day = rng.choice(DAYS)
        hour = rng.randint(7, 20)
        ev_name, duration = rng.choice(EVENT_TYPES)
        if hour + duration > 22:
            continue
        cand = CalendarEvent(day=day, hour=hour, duration=duration, name=ev_name)
        if any(_overlaps(cand, p) for p in placed):
            continue
        placed.append(cand)
    if not placed:
        return None
    return CalendarScenario(scenario_idx=scenario_idx, events=placed)


def config_space_size() -> int:
    """7 days × 14 valid hours × 23 event types per slot. Pick N=6 slots with
    non-overlap. Rough product (over-counts, ignores constraint)."""
    slot_count = 7 * 14 * len(EVENT_TYPES)   # ≈ 2,254
    # Approx: C(slot_count, 6) but most won't conflict given 7×14=98 hour slots.
    n = 1
    for k in range(6):
        n *= slot_count - k
    return n
