"""Place entities — NEW in v5 (in v4 cities were free strings).

Local attrs:
- name
- city_descriptor (e.g. "coastal town", "industrial port", "historic district")
- region (e.g. "northern coast")
- country_name  (also via `located_in` edge → Nation)

Edges (added later):
- located_in → Nation

Built from v4's TOWNS pool (which is a list of (name, descriptor, region)
tuples). Nation entities MUST already be populated.
"""

from __future__ import annotations

import random
from scripts.data_build.generate.bio.world import Entity, World
from scripts.data_build.generate.bio import pools


def build_places(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "place",
) -> None:
    """Generate `n` Place entities by sampling from TOWNS.

    Nation entities must already be in `world`. Each place gets a
    `country_name` attr — a random Nation's name. The `located_in` edge
    is added later in edges.py.
    """
    nations = list(world.entities_of_type("Nation"))
    if not nations:
        raise ValueError("Place builder requires Nation entities to exist first")

    pool = list(pools.TOWNS)
    if n > len(pool):
        raise ValueError(
            f"requested {n} places but pool has only {len(pool)} unique towns"
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:n]

    for i, (name, descriptor, region) in enumerate(chosen):
        key = f"{key_prefix}_{i:04d}_{_slug(name)}"
        # Assign a country (a Nation) at random — will be confirmed by the
        # `located_in` edge later.
        country = rng.choice(nations)
        ent = Entity(
            key=key,
            entity_type="Place",
            attrs={
                "name": name,
                "city_descriptor": descriptor,
                "region": region,
                "country_name": country.attrs["name"],
            },
            surface_names=(name,),
        )
        world.add_entity(ent)


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")
