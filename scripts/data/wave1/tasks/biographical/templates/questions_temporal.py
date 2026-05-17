"""Temporal-ordering question generation.

Compares year-typed attributes across two entities. Each question
references both entities by surface name and asks which event came first
(or last, or by how many years).

API:
    generate_temporal_questions(world, rng, samples=N) -> list[dict]
"""

from __future__ import annotations

import random
from scripts.data.wave1.tasks.biographical.world import World


# Year-typed attributes per entity_type.
YEAR_ATTRS: dict[str, tuple[str, ...]] = {
    "Person": ("birth_year", "award_year"),
    "Organization": ("founding_year",),
    "Event": ("year",),
    "Work": ("year_released",),
    "Nation": ("founding_year",),
}


# Descriptor templates per (entity_type, year_attr).
# Used to render short noun phrases like "Maria's birth" or "the Battle of Y".
DESCRIPTORS: dict[tuple[str, str], str] = {
    ("Person", "birth_year"): "{name}'s birth",
    ("Person", "award_year"): "{name}'s major award",
    ("Organization", "founding_year"): "the founding of {name}",
    ("Nation", "founding_year"): "the founding of {name}",
    ("Event", "year"): "{name}",
    ("Work", "year_released"): "the release of {title}",
}


WHICH_FIRST_TEMPLATES = (
    ("Which happened first: {a_descriptor} or {b_descriptor}?",
     "{winner_descriptor} happened first, in {winner_year}; {loser_descriptor} occurred later, in {loser_year}."),
    ("Of these two, which came first: {a_descriptor} or {b_descriptor}?",
     "{winner_descriptor} came first ({winner_year}), before {loser_descriptor} ({loser_year})."),
)

WHICH_LAST_TEMPLATES = (
    ("Which happened more recently: {a_descriptor} or {b_descriptor}?",
     "{winner_descriptor} happened more recently, in {winner_year}; {loser_descriptor} occurred earlier, in {loser_year}."),
)

YEARS_BETWEEN_TEMPLATES = (
    ("How many years passed between {a_descriptor} and {b_descriptor}?",
     "{years} years passed between {a_descriptor} ({year_a}) and {b_descriptor} ({year_b})."),
)


def generate_temporal_questions(
    world: World,
    rng: random.Random,
    *,
    samples: int = 1000,
    variants: tuple[str, ...] = ("which_first", "which_last", "years_between"),
) -> list[dict]:
    """Sample `samples` temporal-comparison questions.

    Strategy:
    1. Build a flat list of (entity, year_attr, year_value, descriptor_str).
    2. Repeatedly sample two from this list with distinct entity keys and
       distinct years (or with arbitrary year diff for "years_between").
    3. Render question + answer.
    """
    flat: list[tuple] = []
    for ent in world.entities.values():
        attrs = YEAR_ATTRS.get(ent.entity_type, ())
        for attr in attrs:
            year = ent.attrs.get(attr)
            if not isinstance(year, int):
                continue
            descriptor = _make_descriptor(ent, attr)
            if descriptor is None:
                continue
            flat.append((ent.key, attr, year, descriptor))

    if len(flat) < 2:
        return []

    out: list[dict] = []
    attempts = 0
    while len(out) < samples and attempts < samples * 10:
        attempts += 1
        variant = rng.choice(variants)
        a, b = rng.sample(flat, 2)
        a_key, a_attr, year_a, desc_a = a
        b_key, b_attr, year_b, desc_b = b

        if variant in ("which_first", "which_last"):
            if year_a == year_b:
                continue
            winner_first = year_a < year_b
            winner_key, winner_year, winner_desc = (a_key, year_a, desc_a) if winner_first else (b_key, year_b, desc_b)
            loser_year, loser_desc = (year_b, desc_b) if winner_first else (year_a, desc_a)
            if variant == "which_last":
                # Flip — "more recent" winner.
                if winner_first:
                    winner_key, winner_year, winner_desc = b_key, year_b, desc_b
                    loser_year, loser_desc = year_a, desc_a
                else:
                    winner_key, winner_year, winner_desc = a_key, year_a, desc_a
                    loser_year, loser_desc = year_b, desc_b
            tmpls = WHICH_FIRST_TEMPLATES if variant == "which_first" else WHICH_LAST_TEMPLATES
            q_tmpl, a_tmpl = rng.choice(tmpls)
            slots = {
                "a_descriptor": desc_a, "b_descriptor": desc_b,
                "winner_descriptor": winner_desc, "winner_year": winner_year,
                "loser_descriptor": loser_desc, "loser_year": loser_year,
            }
            q = q_tmpl.format(**slots)
            a_text = a_tmpl.format(**slots)
            out.append({
                "question_id": f"q_temporal_{len(out):06d}",
                "question_type": f"temporal_{variant}",
                "evidence_keys": [a_key, b_key],
                "temporal_targets": [
                    {"key": a_key, "attr": a_attr, "year": year_a, "descriptor": desc_a},
                    {"key": b_key, "attr": b_attr, "year": year_b, "descriptor": desc_b},
                ],
                "target_value": winner_desc,
                "question": q,
                "answer": a_text,
            })
        elif variant == "years_between":
            diff = abs(year_a - year_b)
            if diff == 0:
                continue
            q_tmpl, a_tmpl = rng.choice(YEARS_BETWEEN_TEMPLATES)
            slots = {
                "a_descriptor": desc_a, "b_descriptor": desc_b,
                "year_a": year_a, "year_b": year_b, "years": diff,
            }
            q = q_tmpl.format(**slots)
            a_text = a_tmpl.format(**slots)
            out.append({
                "question_id": f"q_temporal_{len(out):06d}",
                "question_type": "temporal_years_between",
                "evidence_keys": [a_key, b_key],
                "temporal_targets": [
                    {"key": a_key, "attr": a_attr, "year": year_a, "descriptor": desc_a},
                    {"key": b_key, "attr": b_attr, "year": year_b, "descriptor": desc_b},
                ],
                "target_value": str(diff),
                "question": q,
                "answer": a_text,
            })
    return out


def _make_descriptor(ent, attr: str) -> str | None:
    tmpl = DESCRIPTORS.get((ent.entity_type, attr))
    if tmpl is None:
        return None
    name = ent.attrs.get("name") or ent.attrs.get("title", ent.key)
    return tmpl.format(name=name, title=name)
