"""Atomic question generation — one query per (entity, attribute).

Each generated row has:
- evidence_keys = [entity.key]
- target_value = entity.attrs[attribute]
- question + answer rendered from per-attribute template variants

API:
    generate_atomic_questions(world, rng, samples_per_attribute=N) -> list[dict]
"""

from __future__ import annotations

import random
from scripts.data.wave1.tasks.biographical.world import Entity, World


# ── Templates: per (entity_type, attribute), list of (q_tmpl, a_tmpl) pairs.
# Slots:
#   {name}       canonical entity name
#   {given}      Person.given_name (first name only)
#   {family}     Person.family_name
#   {v}          target value (the attr being asked about)
# Article handling ({a_v}) is post-processed.

PERSON_ATTR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "occupation": (
        ("What does {name} do for work?",
         "{name} works as {a_v}."),
        ("What is {name}'s occupation?",
         "{name}'s occupation is {v}."),
        ("What is {name}'s line of work?",
         "{name} is {a_v}."),
        ("In what profession does {name} work?",
         "{given} is {a_v}."),
    ),
    "signature_skill": (
        ("What is {name} known for in their work?",
         "{name} is known for {v}."),
        ("What is {given}'s area of specialty?",
         "{given} specializes in {v}."),
        ("What specific skill is {family} most associated with?",
         "{family} is most associated with {v}."),
    ),
    "birth_year": (
        ("In what year was {name} born?",
         "{name} was born in {v}."),
        ("When was {given} born?",
         "{given} was born in {v}."),
        ("What is {family}'s year of birth?",
         "{family}'s year of birth is {v}."),
    ),
    "hobby_gerund": (
        ("What is {name}'s hobby?",
         "{name}'s hobby is {v}."),
        ("Outside of work, what does {given} enjoy doing?",
         "{given} enjoys {v}."),
    ),
    "alma_mater_name": (
        ("Where did {name} study?",
         "{name} studied at {v}."),
        ("What is {given}'s alma mater?",
         "{given}'s alma mater is {v}."),
    ),
    "primary_field": (
        ("What is {name}'s primary field?",
         "{name}'s primary field is {v}."),
        ("In what area does {given} work?",
         "{given} works in {v}."),
    ),
    "signature_work": (
        ("What work is {name} best known for?",
         "{name} is best known for {v}."),
        ("What is {family}'s signature contribution?",
         "{family}'s signature contribution is {v}."),
    ),
    "famous_award": (
        ("What prize did {name} receive?",
         "{name} received {v}."),
        ("What major award is {family} associated with?",
         "{family} is associated with {v}."),
    ),
    "award_year": (
        ("In what year did {name} receive their major award?",
         "{name} received the award in {v}."),
    ),
}

ORG_ATTR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "founding_year": (
        ("When was {name} founded?",
         "{name} was founded in {v}."),
        ("In what year was {name} established?",
         "{name} was established in {v}."),
    ),
    "primary_activity": (
        ("What does {name} primarily do?",
         "{name} is primarily engaged in {v}."),
        ("What is {name}'s primary activity?",
         "{name}'s primary activity is {v}."),
    ),
    "org_type": (
        ("What kind of organization is {name}?",
         "{name} is {a_v}."),
    ),
    "notable_milestone": (
        ("What is {name} best known for?",
         "{name} is best known for {v}."),
        ("What is {name}'s notable achievement?",
         "{name}'s notable achievement is {v}."),
    ),
}

NATION_ATTR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "founding_year": (
        ("When was {name} founded?",
         "{name} was founded in {v}."),
    ),
    "official_language": (
        ("What is the official language of {name}?",
         "The official language of {name} is {v}."),
        ("What language is spoken officially in {name}?",
         "{name}'s official language is {v}."),
    ),
    "capital_name": (
        ("What is the capital of {name}?",
         "The capital of {name} is {v}."),
    ),
}

PLACE_ATTR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "region": (
        ("In what region is {name} located?",
         "{name} is in the {v} region."),
    ),
    "country_name": (
        ("In what country is {name}?",
         "{name} is in {v}."),
    ),
    "city_descriptor": (
        ("What kind of place is {name}?",
         "{name} is {a_v}."),
    ),
}

EVENT_ATTR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "year": (
        ("In what year did {name} occur?",
         "{name} occurred in {v}."),
        ("When did {name} take place?",
         "{name} took place in {v}."),
    ),
    "outcome_descriptor": (
        ("What was the outcome of {name}?",
         "The outcome of {name} was {v}."),
    ),
    "decade": (
        ("In what decade did {name} occur?",
         "{name} occurred in the {v}s."),
    ),
}

WORK_ATTR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "year_released": (
        ("When was {title} released?",
         "{title} was released in {v}."),
        ("In what year did {title} come out?",
         "{title} came out in {v}."),
    ),
    "work_type": (
        ("What kind of work is {title}?",
         "{title} is {a_v}."),
    ),
    "genre": (
        ("What genre does {title} belong to?",
         "{title} is {a_v} work."),
    ),
    "main_subject": (
        ("What is the main subject of {title}?",
         "The main subject of {title} is {v}."),
    ),
}


TEMPLATES_BY_TYPE = {
    "Person": PERSON_ATTR_TEMPLATES,
    "Organization": ORG_ATTR_TEMPLATES,
    "Nation": NATION_ATTR_TEMPLATES,
    "Place": PLACE_ATTR_TEMPLATES,
    "Event": EVENT_ATTR_TEMPLATES,
    "Work": WORK_ATTR_TEMPLATES,
}


def generate_atomic_questions(
    world: World,
    rng: random.Random,
    *,
    samples_per_attribute: int = 1,
) -> list[dict]:
    """For each entity, for each attribute we have templates for, emit
    `samples_per_attribute` rows."""
    out: list[dict] = []
    qid = 0
    for ent in world.entities.values():
        tmpls = TEMPLATES_BY_TYPE.get(ent.entity_type)
        if tmpls is None:
            continue
        for attr, variants in tmpls.items():
            if attr not in ent.attrs:
                continue
            value = ent.attrs[attr]
            if value is None or value == "":
                continue
            for _ in range(samples_per_attribute):
                q_tmpl, a_tmpl = rng.choice(variants)
                q, a = _render(ent, attr, value, q_tmpl, a_tmpl)
                out.append({
                    "question_id": f"q_atomic_{qid:06d}",
                    "question_type": "atomic",
                    "subject_entity": ent.key,
                    "target_attribute": attr,
                    "target_value": str(value),
                    "evidence_keys": [ent.key],
                    "question": q,
                    "answer": a,
                })
                qid += 1
    return out


def _render(ent: Entity, attr: str, value, q_tmpl: str, a_tmpl: str) -> tuple[str, str]:
    """Substitute slots in question/answer templates."""
    v = str(value)
    a_v = _with_indef(v)
    name = ent.attrs.get("name") or ent.attrs.get("title", ent.key)
    given = ent.attrs.get("given_name", name.split()[0])
    family = ent.attrs.get("family_name", name.split()[-1])
    title = ent.attrs.get("title", name)
    slots = {
        "name": name, "given": given, "family": family,
        "title": title, "v": v, "a_v": a_v,
    }
    return q_tmpl.format(**slots), a_tmpl.format(**slots)


def _with_indef(noun_phrase: str) -> str:
    if not noun_phrase:
        return noun_phrase
    first = noun_phrase[0].lower()
    article = "an" if first in "aeiou" else "a"
    return f"{article} {noun_phrase}"
