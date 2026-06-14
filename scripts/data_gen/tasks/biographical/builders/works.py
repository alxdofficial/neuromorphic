"""Work entities (cultural works — novels, films, paintings, etc.).

Local attrs:
- title
- work_type (novel | film | opera | painting | monograph | ...)
- year_released (int)
- release_decade
- genre
- main_subject
- reception (short critical-opinion summary)

Edges added later:
- created_by → Person
- published_by → Organization
"""

from __future__ import annotations

import random
from scripts.data_gen.tasks.biographical.world import Entity, World
from scripts.data_gen.tasks.biographical import pools


def build_works(
    world: World,
    n: int,
    *,
    seed: int = 0,
    key_prefix: str = "cw",
) -> None:
    """Generate `n` Work entities. People must already be populated
    (works need a created_by edge to a real Person, added later)."""
    rng = random.Random(seed)
    if not list(world.entities_of_type("Person")):
        raise ValueError("Work builder requires Person entities first")

    # CW_TYPE_TITLES is dict[work_type -> list of title fragments].
    work_types = list(pools.CW_TYPE_TITLES.keys())
    genres = list(pools.CW_GENRES)
    subjects = list(pools.CW_SUBJECTS)
    receptions = list(pools.CW_RECEPTION)
    lo, hi = pools.WORK_YEAR_RANGE

    seen_titles: set[str] = set()
    idx = 0
    attempts = 0
    while idx < n and attempts < n * 30:
        attempts += 1
        work_type = rng.choice(work_types)
        title_fragments = list(pools.CW_TYPE_TITLES[work_type])
        title = rng.choice(title_fragments)
        if title in seen_titles:
            continue
        seen_titles.add(title)

        year = rng.randint(lo, hi)
        decade = f"{(year // 10) * 10}"
        genre = rng.choice(genres)
        subject = rng.choice(subjects)
        reception = rng.choice(receptions)

        key = f"{key_prefix}_{idx:04d}_{_slug(title)}"
        ent = Entity(
            key=key,
            entity_type="Work",
            attrs={
                "title": title,
                "name": title,         # alias for uniform passage rendering
                "work_type": work_type,
                "year_released": year,
                "release_decade": decade,
                "genre": genre,
                "main_subject": subject,
                "reception": reception,
            },
            surface_names=(title,),
        )
        world.add_entity(ent)
        idx += 1

    if idx < n:
        raise RuntimeError(
            f"Could not generate {n} unique work titles in {attempts} attempts"
        )


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "work"
