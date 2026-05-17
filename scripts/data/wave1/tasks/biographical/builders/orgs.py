"""Organization entities — local attrs only.

Local attrs:
- name (canonical surface form)
- short_name (informal/abbreviated)
- org_type      (university | museum | trust | press | research_institute | ...)
- primary_activity
- founding_year (int)
- founding_decade
- notable_milestone
- headquarters_city_name (also referenced by `headquartered_in` edge → Place)

Built from v4's ORG_NAME_{PREFIXES,STEMS,SUFFIXES} pools.
"""

from __future__ import annotations

import random
from scripts.data.wave1.tasks.biographical.world import Entity, World
from scripts.data.wave1.tasks.biographical import pools


def build_orgs(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "org",
) -> None:
    """Generate `n` Organization entities."""
    places = list(world.entities_of_type("Place"))
    if not places:
        raise ValueError("Organization builder requires Place entities to exist first")

    rng = random.Random(seed)
    seen_names: set[str] = set()
    lo, hi = pools.ORG_FOUNDING_YEAR_RANGE
    idx = 0
    attempts = 0
    while idx < n and attempts < n * 10:
        attempts += 1
        # Compose a name from prefix + stem + suffix. Each fragment list is
        # smallish so collisions occur — retry until we have n unique names.
        prefix = rng.choice(pools.ORG_NAME_PREFIXES)
        stem = rng.choice(pools.ORG_NAME_STEMS)
        suffix = rng.choice(pools.ORG_NAME_SUFFIXES)
        name = f"{prefix} {stem} {suffix}"
        if name in seen_names:
            continue
        seen_names.add(name)

        # Derived short name: drop "the", keep significant words. Light heuristic.
        short_words = [w for w in name.replace(",", "").split()
                       if w.lower() not in {"the", "for", "of", "and", "a", "an"}]
        short_name = " ".join(short_words[:3])

        # Plausible org_type derived from the suffix word.
        org_type = _classify_org_type(suffix)

        # Pick a headquarters city — must be a real Place.
        hq = rng.choice(places)

        founding_year = rng.randint(lo, hi)
        founding_decade = f"{(founding_year // 10) * 10}"
        primary_activity = rng.choice(pools.ORG_ACTIVITIES)
        milestone = rng.choice(pools.ORG_MILESTONES)

        key = f"{key_prefix}_{idx:04d}_{_slug(short_name)}"
        ent = Entity(
            key=key,
            entity_type="Organization",
            attrs={
                "name": name,
                "short_name": short_name,
                "org_type": org_type,
                "primary_activity": primary_activity,
                "founding_year": founding_year,
                "founding_decade": founding_decade,
                "notable_milestone": milestone,
                "headquarters_city_name": hq.attrs["name"],
            },
            surface_names=(name, short_name),
        )
        world.add_entity(ent)
        idx += 1

    if idx < n:
        raise RuntimeError(
            f"Could not generate {n} unique org names in {attempts} attempts; "
            "expand ORG_NAME_* pools or reduce n."
        )


def _classify_org_type(suffix: str) -> str:
    """Heuristic mapping from suffix word to org_type tag."""
    s = suffix.lower()
    if "university" in s or "college" in s or "academy" in s or "school" in s:
        return "university"
    if "museum" in s:
        return "museum"
    if "trust" in s or "foundation" in s or "society" in s:
        return "trust"
    if "press" in s or "publish" in s:
        return "press"
    if "institute" in s or "laboratory" in s or "centre" in s or "center" in s:
        return "research_institute"
    if "co." in s or "company" in s or "corp" in s or "guild" in s:
        return "company"
    if "hospital" in s or "clinic" in s:
        return "healthcare"
    if "bureau" in s or "council" in s or "office" in s:
        return "bureau"
    if "guild" in s or "association" in s or "cooperative" in s:
        return "association"
    return "association"   # safe fallback (consonant-initial for "a {type}")


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "org"
