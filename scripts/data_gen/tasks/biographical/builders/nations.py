"""Nation entities — local attrs only.

Local attrs:
- name
- founding_year (int)
- official_language
- capital_name  (also referenced by `capital` edge → Place)

Edges (added later in edges.py):
- head_of_government → Person
- capital → Place
- bordered_by → Nation
"""

from __future__ import annotations

import random
from scripts.data_gen.tasks.biographical.world import Entity, World
from scripts.data_gen.tasks.biographical import pools


def build_nations(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "nt",
) -> None:
    """Generate `n` Nation entities."""
    pool_names = list(pools.NATION_NAMES)
    if n > len(pool_names):
        raise ValueError(
            f"requested {n} nations but pool has only {len(pool_names)} unique names"
        )

    rng = random.Random(seed)
    rng.shuffle(pool_names)
    chosen = pool_names[:n]

    lo, hi = pools.NATION_FOUNDING_YEAR_RANGE
    for i, name in enumerate(chosen):
        key = f"{key_prefix}_{i:04d}_{_slug(name)}"
        founding_year = rng.randint(lo, hi)
        # Local-only attrs: capital_name is a hint that the `capital` edge
        # (added later) will point to a Place with the same name. We keep
        # the string here as a fallback for rendering passages BEFORE
        # populate_relationships runs the edge logic.
        capital_name = rng.choice(pools.NATION_CAPITAL_NAMES)
        official_language = rng.choice(pools.NATION_LANGUAGES)

        ent = Entity(
            key=key,
            entity_type="Nation",
            attrs={
                "name": name,
                "founding_year": founding_year,
                "official_language": official_language,
                "capital_name": capital_name,
            },
            surface_names=(name,),
        )
        world.add_entity(ent)


def _slug(s: str) -> str:
    """Filesystem/key-safe slug: lowercase, alphanumerics + underscores only."""
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")
