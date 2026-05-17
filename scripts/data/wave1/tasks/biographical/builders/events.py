"""Event entities (historical + life events merged into one type).

Local attrs:
- name (event title)
- year (int)
- decade
- outcome_descriptor
- century (string for passage rendering)

Edges added later:
- happened_in → Place
- primary_figure → Person
- involved → Organization (many)
"""

from __future__ import annotations

import random
from scripts.data.wave1.tasks.biographical.world import Entity, World
from scripts.data.wave1.tasks.biographical import pools


def build_events(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "ev",
) -> None:
    """Generate `n` Event entities — fictional historical events.

    Place entities must already be populated (events need a happening
    location). Person entities optional but recommended (events need a
    primary_figure for some questions to work).
    """
    rng = random.Random(seed)

    if not list(world.entities_of_type("Place")):
        raise ValueError("Event builder requires Place entities first")

    # HE_TYPE_TEMPLATES is a list of patterns like "the {Y} {N}" where
    # {Y} is a year and {N} is a noun (excavation, expedition, etc.).
    # We construct event names by filling these patterns.
    type_templates = list(pools.HE_TYPE_TEMPLATES)
    outcomes = list(pools.HE_OUTCOMES)
    lo, hi = pools.EVENT_YEAR_RANGE

    seen_names: set[str] = set()
    idx = 0
    attempts = 0
    while idx < n and attempts < n * 30:
        attempts += 1
        template = rng.choice(type_templates)
        year = rng.randint(lo, hi)
        decade = f"{(year // 10) * 10}"
        century = _century_word(year)

        # Templates use {town} or {institution} slots (v4 convention).
        try:
            name = template.format(
                town=rng.choice(list(world.entities_of_type("Place"))).attrs["name"],
                institution=rng.choice(pools.HE_INSTITUTIONS),
            )
        except (KeyError, IndexError):
            continue

        if name in seen_names:
            continue
        seen_names.add(name)

        outcome = rng.choice(outcomes)

        key = f"{key_prefix}_{idx:04d}_{_slug(name)}"
        ent = Entity(
            key=key,
            entity_type="Event",
            attrs={
                "name": name,
                "year": year,
                "decade": decade,
                "century": century,
                "outcome_descriptor": outcome,
            },
            surface_names=(name,),
        )
        world.add_entity(ent)
        idx += 1

    if idx < n:
        raise RuntimeError(
            f"Could not generate {n} unique event names in {attempts} attempts"
        )


def _century_word(year: int) -> str:
    """Render a year's century as words: 1907 → 'twentieth', 2017 → 'twenty-first'."""
    c = (year - 1) // 100 + 1
    names = {
        17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
        20: "twentieth", 21: "twenty-first",
    }
    return names.get(c, f"{c}th")


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "event"
