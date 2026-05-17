"""Person entities — local attrs only. Edges added later by edges.py.

Local attrs:
- name (canonical "First Last")
- given_name
- family_name
- birth_year (int)
- occupation
- domain
- signature_skill
- hobby_gerund
- recurring_habit
- alma_mater_name      (also referenced by `alma_mater` edge → Organization)
- family_background

For Public Figures (a Person subtype), extra attrs:
- flavor = "public_figure"
- primary_field
- signature_work
- famous_award
- award_year
"""

from __future__ import annotations

import random
from scripts.data.wave1.tasks.biographical.world import Entity, World
from scripts.data.wave1.tasks.biographical import pools


def build_people(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "pi",
) -> None:
    """Generate `n` private-individual Person entities.

    Each gets a unique (first, last) name combo. Constraints:
    - No name collisions across people in the world.
    - Occupation is drawn from the flat pool; domain & signature_skill
      are derived from the occupation for coherence.
    """
    existing_names = _collect_existing_person_names(world)
    rng = random.Random(seed)

    first_pool = list(pools.FIRST_NAMES_F) + list(pools.FIRST_NAMES_M)
    last_pool = list(pools.LAST_NAMES)
    lo, hi = pools.PERSON_BIRTH_YEAR_RANGE

    idx = 0
    attempts = 0
    while idx < n and attempts < n * 30:
        attempts += 1
        first = rng.choice(first_pool)
        last = rng.choice(last_pool)
        full = f"{first} {last}"
        if full in existing_names:
            continue
        existing_names.add(full)

        occupation = rng.choice(pools.OCCUPATIONS)
        domain = pools.OCCUPATION_DOMAIN.get(occupation, "general")
        skills = pools.SIGNATURE_SKILLS.get(occupation) or ["specialized work"]
        signature_skill = rng.choice(skills)
        hobby_tuple = rng.choice(pools.HOBBIES)
        hobby_gerund = hobby_tuple[0] if isinstance(hobby_tuple, tuple) else hobby_tuple
        recurring_habit = rng.choice(pools.RECURRING_HABITS)
        recurring_habit = (
            recurring_habit[0] if isinstance(recurring_habit, tuple)
            else recurring_habit
        )
        alma_mater_name = rng.choice(pools.ALMA_MATERS)
        family_bg = rng.choice(pools.FAMILY_BACKGROUNDS)
        birth_year = rng.randint(lo, hi)

        key = f"{key_prefix}_{idx:04d}_{_slug(full)}"
        ent = Entity(
            key=key,
            entity_type="Person",
            attrs={
                "name": full,
                "given_name": first,
                "family_name": last,
                "birth_year": birth_year,
                "occupation": occupation,
                "domain": domain,
                "signature_skill": signature_skill,
                "hobby_gerund": hobby_gerund,
                "recurring_habit": recurring_habit,
                "alma_mater_name": alma_mater_name,
                "family_background": family_bg,
                "flavor": "private",
            },
            surface_names=_surface_names(first, last),
        )
        world.add_entity(ent)
        idx += 1

    if idx < n:
        raise RuntimeError(
            f"Could not generate {n} unique person names in {attempts} attempts; "
            "expand FIRST_NAMES/LAST_NAMES pools."
        )


def build_public_figures(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "pf",
) -> None:
    """Generate `n` public-figure Person entities. Same entity_type as
    private people but with extra attrs (primary_field, signature_work,
    famous_award, award_year)."""
    existing_names = _collect_existing_person_names(world)
    rng = random.Random(seed)

    first_pool = list(pools.FIRST_NAMES_F) + list(pools.FIRST_NAMES_M)
    last_pool = list(pools.LAST_NAMES)
    lo, hi = pools.PERSON_BIRTH_YEAR_RANGE

    idx = 0
    attempts = 0
    while idx < n and attempts < n * 30:
        attempts += 1
        first = rng.choice(first_pool)
        last = rng.choice(last_pool)
        full = f"{first} {last}"
        if full in existing_names:
            continue
        existing_names.add(full)

        field = rng.choice(pools.PUBLIC_FIGURE_FIELDS)
        sig_work_pool = pools.SIGNATURE_WORKS.get(field) or ["unspecified contribution"]
        sig_work = rng.choice(sig_work_pool)
        famous_award = rng.choice(pools.FAMOUS_AWARDS)
        # For public figures, force an older birth year so they have time
        # to have a "career" and an award by 2024.
        pf_lo, pf_hi = lo, min(hi, 1990)
        birth_year = rng.randint(pf_lo, pf_hi)
        # Award typically given mid-to-late career.
        award_lo = max(birth_year + 30, 1970)
        award_hi = min(birth_year + 70, 2024)
        if award_hi < award_lo:
            award_hi = award_lo  # degenerate guard
        award_year = rng.randint(award_lo, award_hi)
        alma_mater = rng.choice(pools.PF_ALMA_MATERS)

        key = f"{key_prefix}_{idx:04d}_{_slug(full)}"
        ent = Entity(
            key=key,
            entity_type="Person",
            attrs={
                "name": full,
                "given_name": first,
                "family_name": last,
                "birth_year": birth_year,
                "primary_field": field,
                "signature_work": sig_work,
                "famous_award": famous_award,
                "award_year": award_year,
                "alma_mater_name": alma_mater,
                "flavor": "public_figure",
            },
            surface_names=_surface_names(first, last),
        )
        world.add_entity(ent)
        idx += 1

    if idx < n:
        raise RuntimeError(
            f"Could not generate {n} unique public-figure names in {attempts} attempts"
        )


def _collect_existing_person_names(world: World) -> set[str]:
    """All canonical "First Last" strings already in the world."""
    return {
        ent.attrs["name"] for ent in world.entities_of_type("Person")
        if "name" in ent.attrs
    }


def _surface_names(first: str, last: str) -> tuple[str, ...]:
    """All surface forms a passage might use for this person.

    Includes the full name, last-name only, first-name only, and
    title-prefixed variants. The name-disjoint splitter uses any overlap
    here to union-find entities with shared names.
    """
    return (
        f"{first} {last}",
        last,
        first,
        f"Dr. {last}",
        f"Prof. {last}",
        f"Ms. {last}",
        f"Mr. {last}",
    )


def _slug(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "person"
