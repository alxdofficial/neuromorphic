"""Aggregation question generation — count/filter/predicate over a set.

Questions over a SET of evidence entities (typically 3-6) that require:
1. Reading all evidence passages from memory
2. Applying a predicate to each
3. Aggregating (count, all, any, which)

API:
    generate_aggregation_questions(world, rng, samples=N) -> list[dict]
"""

from __future__ import annotations

import random
from scripts.data_gen.tasks.biographical.world import World


# Predicate definitions — what attribute (possibly via a relation chain)
# to check on each evidence entity, and the human-readable descriptor.
#
# Each predicate is (entity_type, attr_or_chain, descriptor_template).
# attr_or_chain is either a string (direct attr) or a tuple of relation
# names + final attr (e.g. ("works_at", "headquartered_in", "name")).

# Each entry is (entity_type, attr_or_chain, descriptor_plural, descriptor_singular).
# - descriptor_plural    used in plural-subject contexts ("How many people work...")
# - descriptor_singular  used in singular-subject answers ("X works as Y")
# Both have a {value} slot for the predicate's target value.
PREDICATES = [
    # Person predicates
    ("Person", "occupation",
     "work as {value}", "works as {value}"),
    ("Person", ("works_at", "headquartered_in", "name"),
     "work at an organization headquartered in {value}",
     "works at an organization headquartered in {value}"),
    ("Person", ("works_at", "name"),
     "work at {value}", "works at {value}"),
    ("Person", ("alma_mater", "name"),
     "are alumni of {value}", "is an alumnus of {value}"),
    ("Person", ("born_in", "name"),
     "were born in {value}", "was born in {value}"),
    ("Person", ("born_in", "located_in", "name"),
     "were born in {value}", "was born in {value}"),
    # Organization predicates
    ("Organization", "org_type",
     "are {value}s", "is {a_value}"),
    ("Organization", ("headquartered_in", "name"),
     "are headquartered in {value}", "is headquartered in {value}"),
    # Work predicates
    ("Work", "work_type",
     "are {value}s", "is {a_value}"),
    ("Work", "genre",
     "are {value} works", "is {a_value} work"),
    # Event predicates
    ("Event", ("happened_in", "name"),
     "took place in {value}", "took place in {value}"),
]


COUNT_TEMPLATES = (
    ("How many of these {entity_plural} {predicate_descriptor}?",
     "{count} of these {entity_plural} {predicate_descriptor}."),
)

WHICH_TEMPLATES = (
    ("Which of these {entity_plural} {predicate_descriptor}?",
     "{matches_str} {predicate_descriptor_singular}."),
    ("Of these {entity_plural}, which one {predicate_descriptor_singular_q}?",
     "{matches_str} {predicate_descriptor_singular}."),
)


def generate_aggregation_questions(
    world: World,
    rng: random.Random,
    *,
    samples: int = 500,
    set_size_range: tuple[int, int] = (3, 6),
    variants: tuple[str, ...] = ("count_matching", "which_matches"),
) -> list[dict]:
    """Sample `samples` aggregation questions.

    For each sample:
    1. Pick a predicate (entity_type + attr_path + value).
    2. Sample N entities of that type (set size random in set_size_range).
    3. Compute ground-truth match count.
    4. Filter trivial outcomes (count == 0 or count == N) for most variants.
    5. Render question + answer.
    """
    out: list[dict] = []
    attempts = 0
    def _format_pred(tmpl: str, value: str) -> str:
        a_value = _with_indef(value)
        return tmpl.format(value=value, a_value=a_value)

    while len(out) < samples and attempts < samples * 30:
        attempts += 1
        pred_type, attr_path, descriptor_plural_tmpl, descriptor_singular_tmpl = rng.choice(PREDICATES)
        candidates = [e for e in world.entities_of_type(pred_type)
                      if _resolve_attr_path(world, e, attr_path) is not None]
        if len(candidates) < set_size_range[0]:
            continue

        # Sample possible target values for the predicate (drawn from
        # candidates' actual values, weighted toward those that have at
        # least 1 match in the world).
        value_counts: dict[str, int] = {}
        for c in candidates:
            v = _resolve_attr_path(world, c, attr_path)
            if v is not None:
                value_counts[str(v)] = value_counts.get(str(v), 0) + 1
        # Pick a value that has >=1 match.
        viable = [v for v, c in value_counts.items() if c >= 1]
        if not viable:
            continue
        target_value = rng.choice(viable)

        # Sample a set of evidence entities (mix of matching + non-matching).
        n_set = rng.randint(set_size_range[0], min(set_size_range[1], len(candidates)))
        evidence = rng.sample(candidates, n_set)
        match_keys = [
            e.key for e in evidence
            if str(_resolve_attr_path(world, e, attr_path)) == target_value
        ]
        match_count = len(match_keys)

        variant = rng.choice(variants)

        if variant == "count_matching":
            # Skip trivial all-or-nothing AND skip count==1 (subject-verb
            # agreement is fiddly — "one of these people works"). Only
            # emit when match_count is strictly between 2 and n_set-1.
            if match_count < 2 or match_count == n_set:
                continue
            q_tmpl, a_tmpl = rng.choice(COUNT_TEMPLATES)
            entity_plural = _plural(pred_type)
            predicate_descriptor = _format_pred(descriptor_plural_tmpl, target_value)
            count_word = _count_word(match_count)
            slots = {
                "entity_plural": entity_plural,
                "predicate_descriptor": predicate_descriptor,
                "count": count_word,
            }
            q = q_tmpl.format(**slots)
            a = a_tmpl.format(**slots)
            # Answer template starts with {count}; capitalize for sentence start.
            a = a[0].upper() + a[1:]
            out.append({
                "question_id": f"q_agg_{len(out):06d}",
                "question_type": "aggregation_count",
                "predicate_type": pred_type,
                "predicate_attr_path": list(attr_path) if isinstance(attr_path, tuple) else [attr_path],
                "predicate_value": target_value,
                "evidence_keys": [e.key for e in evidence],
                "match_count": match_count,
                "match_keys": match_keys,
                "target_value": _count_word(match_count),
                "question": q,
                "answer": a,
            })
        elif variant == "which_matches":
            # Only emit when exactly 1 match (clean answer).
            if match_count != 1:
                continue
            match_ent = world.entities[match_keys[0]]
            match_name = match_ent.attrs.get("name") or match_ent.attrs.get("title", match_ent.key)
            q_tmpl, a_tmpl = rng.choice(WHICH_TEMPLATES)
            entity_plural = _plural(pred_type)
            predicate_descriptor = _format_pred(descriptor_plural_tmpl, target_value)
            predicate_descriptor_singular = _format_pred(descriptor_singular_tmpl, target_value)
            # Convert "X works as Y" -> "works as Y" form by stripping
            # the leading verb if needed (the descriptor_singular starts
            # with the verb; for the question form we want it conjugated).
            predicate_descriptor_singular_q = predicate_descriptor_singular
            slots = {
                "entity_plural": entity_plural,
                "predicate_descriptor": predicate_descriptor,
                "predicate_descriptor_singular": predicate_descriptor_singular,
                "predicate_descriptor_singular_q": predicate_descriptor_singular_q,
                "matches_str": match_name,
            }
            q = q_tmpl.format(**slots)
            a = a_tmpl.format(**slots)
            out.append({
                "question_id": f"q_agg_{len(out):06d}",
                "question_type": "aggregation_which",
                "predicate_type": pred_type,
                "predicate_attr_path": list(attr_path) if isinstance(attr_path, tuple) else [attr_path],
                "predicate_value": target_value,
                "evidence_keys": [e.key for e in evidence],
                "match_count": match_count,
                "match_keys": match_keys,
                "target_value": match_name,
                "question": q,
                "answer": a,
            })
    return out


def _resolve_attr_path(world: World, ent, attr_or_chain):
    """Resolve either a direct attr name or a (rel_1, rel_2, ..., attr) chain."""
    if isinstance(attr_or_chain, str):
        return ent.attrs.get(attr_or_chain)
    # Tuple: walk the chain.
    *path, final_attr = attr_or_chain
    end = world.resolve_path(ent.key, tuple(path))
    if end is None or not hasattr(end, "attrs"):
        return None
    return end.attrs.get(final_attr)


_PLURALS = {
    "Person": "people",
    "Organization": "organizations",
    "Nation": "nations",
    "Place": "places",
    "Event": "events",
    "Work": "works",
}


def _plural(entity_type: str) -> str:
    return _PLURALS.get(entity_type, entity_type.lower() + "s")


_COUNT_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
)


def _count_word(n: int) -> str:
    if 0 <= n <= 10:
        return _COUNT_WORDS[n]
    return str(n)


def _with_indef(noun_phrase: str) -> str:
    if not noun_phrase:
        return noun_phrase
    first = noun_phrase[0].lower()
    article = "an" if first in "aeiou" else "a"
    return f"{article} {noun_phrase}"
