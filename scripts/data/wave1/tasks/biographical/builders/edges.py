"""Cross-entity edges — added AFTER all entity types are populated.

Every cross-reference that v4 stored as a free string is now a typed
Edge in the world graph. This is what makes compositional questions
("X's mentor's workplace") possible: the question generator can walk
the graph at generation time, and the trainer's evidence_keys list is
exactly the entities visited along the walk.

Each `add_edges_*` function:
- Takes a `World` (must already contain the entity types it links)
- Takes a `seed` for determinism
- Takes a `coverage` rate (0.0-1.0) for how many src entities get this edge
- Mutates `world` in place

`populate_relationships(world, seed)` orchestrates them in dependency order.
"""

from __future__ import annotations

import random
from scripts.data.wave1.tasks.biographical.world import Edge, World


# ── Place / Nation structural edges ─────────────────────────────────


def add_edges_located_in(
    world: World, *, seed: int = 0, coverage: float = 1.00,
) -> None:
    """Place → Nation. Coverage=1.0: every place is in a nation.

    The Nation chosen is whichever has the matching `country_name` attr;
    if no match (e.g. the place's country_name was randomly assigned from
    a different generator), fall back to a random Nation.
    """
    rng = random.Random(seed)
    nations = list(world.entities_of_type("Nation"))
    if not nations:
        return
    nations_by_name = {n.attrs["name"]: n for n in nations}
    for place in list(world.entities_of_type("Place")):
        if rng.random() > coverage:
            continue
        target = nations_by_name.get(place.attrs.get("country_name", ""))
        if target is None:
            target = rng.choice(nations)
        world.add_edge(Edge(src=place.key, rel="located_in", dst=target.key))


def add_edges_capital(
    world: World, *, seed: int = 0, coverage: float = 1.00,
) -> None:
    """Nation → Place. Each Nation gets one capital Place.

    Tries to match the Nation's `capital_name` attr to a Place's name;
    otherwise picks a random Place from the same `located_in` country
    when possible, else any Place.
    """
    rng = random.Random(seed)
    places = list(world.entities_of_type("Place"))
    if not places:
        return
    places_by_name = {p.attrs["name"]: p for p in places}

    # Index places by their country.
    places_by_country: dict[str, list] = {}
    for p in places:
        country_key = None
        for e in world.get_edges(p.key, "located_in"):
            country_key = e.dst
            break
        if country_key:
            places_by_country.setdefault(country_key, []).append(p)

    for nation in list(world.entities_of_type("Nation")):
        if rng.random() > coverage:
            continue
        # Capital MUST be a Place located_in this nation (logical
        # consistency). If no such Place exists, skip rather than break
        # the invariant.
        candidates = places_by_country.get(nation.key, [])
        if not candidates:
            continue
        target = rng.choice(candidates)
        world.add_edge(Edge(src=nation.key, rel="capital", dst=target.key))


def add_edges_bordered_by(
    world: World, *, seed: int = 0, coverage: float = 0.60,
    min_neighbors: int = 1, max_neighbors: int = 4,
) -> None:
    """Nation → Nation (many-cardinality)."""
    rng = random.Random(seed)
    nations = list(world.entities_of_type("Nation"))
    if len(nations) < 2:
        return
    for nation in nations:
        if rng.random() > coverage:
            continue
        n_neighbors = rng.randint(min_neighbors, min(max_neighbors, len(nations) - 1))
        # Sample n_neighbors others (no self, no duplicates).
        others = [n for n in nations if n.key != nation.key]
        picked = rng.sample(others, n_neighbors)
        for nbr in picked:
            world.add_edge(Edge(src=nation.key, rel="bordered_by", dst=nbr.key))


# ── Organization structural edges ───────────────────────────────────


def add_edges_headquartered_in(
    world: World, *, seed: int = 0, coverage: float = 1.00,
) -> None:
    """Organization → Place. Coverage=1.0 — every org has a HQ city."""
    rng = random.Random(seed)
    places = list(world.entities_of_type("Place"))
    if not places:
        return
    places_by_name = {p.attrs["name"]: p for p in places}
    for org in list(world.entities_of_type("Organization")):
        if rng.random() > coverage:
            continue
        hq_name = org.attrs.get("headquarters_city_name")
        target = places_by_name.get(hq_name) if hq_name else None
        if target is None:
            target = rng.choice(places)
        world.add_edge(Edge(src=org.key, rel="headquartered_in", dst=target.key))


def add_edges_subsidiary_of(
    world: World, *, seed: int = 0, coverage: float = 0.10,
) -> None:
    """Organization → Organization. Rare but enables 3-hop questions."""
    rng = random.Random(seed)
    orgs = list(world.entities_of_type("Organization"))
    if len(orgs) < 2:
        return
    # No cycles: smaller-idx orgs are parents, larger-idx are subsidiaries.
    sorted_orgs = sorted(orgs, key=lambda o: o.key)
    for i, sub in enumerate(sorted_orgs[1:], start=1):
        if rng.random() > coverage:
            continue
        parent = rng.choice(sorted_orgs[:i])
        world.add_edge(Edge(src=sub.key, rel="subsidiary_of", dst=parent.key))


# ── Person structural edges ─────────────────────────────────────────


def add_edges_born_in(
    world: World, *, seed: int = 0, coverage: float = 0.70,
) -> None:
    """Person → Place."""
    rng = random.Random(seed)
    places = list(world.entities_of_type("Place"))
    if not places:
        return
    for person in list(world.entities_of_type("Person")):
        if rng.random() > coverage:
            continue
        target = rng.choice(places)
        world.add_edge(Edge(src=person.key, rel="born_in", dst=target.key))


def add_edges_works_at(
    world: World, *, seed: int = 0, coverage: float = 0.80,
) -> None:
    """Person → Organization. Tries to match Person's domain to Org's type
    so the assignment is at least loosely coherent (a marine biologist
    doesn't work at a bakery)."""
    rng = random.Random(seed)
    orgs = list(world.entities_of_type("Organization"))
    if not orgs:
        return
    for person in list(world.entities_of_type("Person")):
        if rng.random() > coverage:
            continue
        # Lightweight coherence: prefer same domain if any match, else any.
        # (We don't enforce strictly to avoid pool starvation.)
        target = rng.choice(orgs)
        world.add_edge(Edge(src=person.key, rel="works_at", dst=target.key))


def add_edges_alma_mater(
    world: World, *, seed: int = 0, coverage: float = 0.60,
) -> None:
    """Person → Organization (where org_type prefers 'university').

    Sets Person.attrs['alma_mater_name'] to the target org's name so passage
    rendering remains consistent.
    """
    rng = random.Random(seed)
    universities = [
        o for o in world.entities_of_type("Organization")
        if o.attrs.get("org_type") == "university"
    ]
    fallback = list(world.entities_of_type("Organization"))
    if not fallback:
        return

    for person in list(world.entities_of_type("Person")):
        if rng.random() > coverage:
            continue
        pool = universities if universities else fallback
        target = rng.choice(pool)
        world.add_edge(Edge(src=person.key, rel="alma_mater", dst=target.key))


def add_edges_mentor(
    world: World, *, seed: int = 0, coverage: float = 0.30,
) -> None:
    """Person → Person. Constraints:
    - mentor must be ≥ 10 years older than src
    - no self-mentoring
    - no cycles (mentor's mentor chain may not contain src)
    """
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if len(people) < 2:
        return
    # Index by birth year so we can quickly find people ≥ 10 years older.
    people_sorted = sorted(people, key=lambda p: p.attrs.get("birth_year", 0))

    # Mentor map (for cycle detection).
    mentor_of: dict[str, str] = {}
    for person in people:
        if rng.random() > coverage:
            continue
        src_birth = person.attrs.get("birth_year", 0)
        candidates = [
            p for p in people_sorted
            if p.key != person.key
            and p.attrs.get("birth_year", 0) <= src_birth - 10
        ]
        if not candidates:
            continue
        target = rng.choice(candidates)
        # Cycle check: walk target's mentor chain.
        if _would_create_cycle(mentor_of, src=person.key, dst=target.key):
            continue
        mentor_of[person.key] = target.key
        world.add_edge(Edge(src=person.key, rel="mentor", dst=target.key))


def add_edges_spouse(
    world: World, *, seed: int = 0, coverage: float = 0.40,
) -> None:
    """Person ↔ Person (symmetric edge — add both directions).

    Constraints:
    - no self-spouse
    - each person has at most 1 spouse
    - mentor and spouse cannot overlap (a mentor isn't your spouse here,
      to keep questions clean)
    """
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if len(people) < 2:
        return
    available = {p.key for p in people}
    rng.shuffle(people)

    # Index existing mentor edges so we can avoid mentor=spouse overlap.
    mentor_pairs: set[tuple[str, str]] = set()
    for e in world.edges:
        if e.rel == "mentor" and isinstance(e.dst, str):
            mentor_pairs.add((e.src, e.dst))
            mentor_pairs.add((e.dst, e.src))

    for person in people:
        if person.key not in available:
            continue
        if rng.random() > coverage:
            continue
        candidates = [
            p for p in people
            if p.key in available
            and p.key != person.key
            and (person.key, p.key) not in mentor_pairs
        ]
        if not candidates:
            continue
        partner = rng.choice(candidates)
        world.add_edge(Edge(src=person.key, rel="spouse", dst=partner.key))
        world.add_edge(Edge(src=partner.key, rel="spouse", dst=person.key))
        available.discard(person.key)
        available.discard(partner.key)


def add_edges_parent(
    world: World, *, seed: int = 0, coverage: float = 0.20,
) -> None:
    """Person → Person (0-2 cardinality).

    Constraints:
    - parent must be ≥ 18 years older
    - no cycles
    """
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if len(people) < 2:
        return
    parent_of: dict[str, list[str]] = {}
    for person in people:
        if rng.random() > coverage:
            continue
        src_birth = person.attrs.get("birth_year", 0)
        candidates = [
            p for p in people
            if p.key != person.key
            and p.attrs.get("birth_year", 0) <= src_birth - 18
        ]
        if not candidates:
            continue
        # Maybe 1 or 2 parents.
        n_parents = rng.choices([1, 2], weights=[0.7, 0.3])[0]
        if n_parents > len(candidates):
            n_parents = len(candidates)
        picked = rng.sample(candidates, n_parents)
        for parent in picked:
            if _would_create_cycle(
                {k: vs[0] for k, vs in parent_of.items() if vs},
                src=person.key, dst=parent.key,
            ):
                continue
            parent_of.setdefault(person.key, []).append(parent.key)
            world.add_edge(Edge(src=person.key, rel="parent", dst=parent.key))


# ── Organization "founded_by" — requires People ──────────────────────


def add_edges_founded_by(
    world: World, *, seed: int = 0, coverage: float = 0.50,
) -> None:
    """Organization → Person.

    The founder must plausibly be alive at the time of founding. We
    require founder.birth_year ≤ org.founding_year - 25; if no candidate
    in the pool satisfies that, **skip the edge entirely** (better to
    have no founded_by than an anachronistic one).
    """
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if not people:
        return
    for org in list(world.entities_of_type("Organization")):
        if rng.random() > coverage:
            continue
        founding_year = org.attrs.get("founding_year", 1950)
        candidates = [
            p for p in people
            if p.attrs.get("birth_year", 0) <= founding_year - 25
        ]
        if not candidates:
            continue                # skip rather than anachronism
        founder = rng.choice(candidates)
        world.add_edge(Edge(src=org.key, rel="founded_by", dst=founder.key))


# ── Nation head_of_government ───────────────────────────────────────


def add_edges_head_of_government(
    world: World, *, seed: int = 0, coverage: float = 1.00,
) -> None:
    """Nation → Person."""
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if not people:
        return
    for nation in list(world.entities_of_type("Nation")):
        if rng.random() > coverage:
            continue
        head = rng.choice(people)
        world.add_edge(Edge(src=nation.key, rel="head_of_government", dst=head.key))


# ── Event edges ─────────────────────────────────────────────────────


def add_edges_event_happened_in(
    world: World, *, seed: int = 0, coverage: float = 1.00,
) -> None:
    """Event → Place."""
    rng = random.Random(seed)
    places = list(world.entities_of_type("Place"))
    if not places:
        return
    for ev in list(world.entities_of_type("Event")):
        if rng.random() > coverage:
            continue
        target = rng.choice(places)
        world.add_edge(Edge(src=ev.key, rel="happened_in", dst=target.key))


def add_edges_event_primary_figure(
    world: World, *, seed: int = 0, coverage: float = 0.80,
) -> None:
    """Event → Person."""
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if not people:
        return
    for ev in list(world.entities_of_type("Event")):
        if rng.random() > coverage:
            continue
        year = ev.attrs.get("year", 1950)
        # Figures alive at the event (born before, not too long after).
        candidates = [
            p for p in people
            if p.attrs.get("birth_year", 0) <= year - 18
        ]
        if not candidates:
            continue        # skip rather than anachronism
        figure = rng.choice(candidates)
        world.add_edge(Edge(src=ev.key, rel="primary_figure", dst=figure.key))


def add_edges_event_involved(
    world: World, *, seed: int = 0, coverage: float = 0.50,
    max_orgs: int = 2,
) -> None:
    """Event → Organization (many-cardinality)."""
    rng = random.Random(seed)
    orgs = list(world.entities_of_type("Organization"))
    if not orgs:
        return
    for ev in list(world.entities_of_type("Event")):
        if rng.random() > coverage:
            continue
        n = rng.randint(1, min(max_orgs, len(orgs)))
        picked = rng.sample(orgs, n)
        for o in picked:
            world.add_edge(Edge(src=ev.key, rel="involved", dst=o.key))


# ── Work edges ──────────────────────────────────────────────────────


def add_edges_work_created_by(
    world: World, *, seed: int = 0, coverage: float = 1.00,
) -> None:
    """Work → Person."""
    rng = random.Random(seed)
    people = list(world.entities_of_type("Person"))
    if not people:
        return
    for w in list(world.entities_of_type("Work")):
        if rng.random() > coverage:
            continue
        year = w.attrs.get("year_released", 1950)
        candidates = [
            p for p in people
            if p.attrs.get("birth_year", 0) <= year - 20
        ]
        if not candidates:
            continue        # skip rather than anachronism
        creator = rng.choice(candidates)
        world.add_edge(Edge(src=w.key, rel="created_by", dst=creator.key))


def add_edges_work_published_by(
    world: World, *, seed: int = 0, coverage: float = 0.70,
) -> None:
    """Work → Organization."""
    rng = random.Random(seed)
    orgs = list(world.entities_of_type("Organization"))
    if not orgs:
        return
    # Prefer presses / publishers when available, else any org.
    publishers = [o for o in orgs if o.attrs.get("org_type") in {"press", "company"}]
    pool = publishers if publishers else orgs
    for w in list(world.entities_of_type("Work")):
        if rng.random() > coverage:
            continue
        target = rng.choice(pool)
        world.add_edge(Edge(src=w.key, rel="published_by", dst=target.key))


# ── Orchestrator ────────────────────────────────────────────────────


def populate_relationships(world: World, *, seed: int = 0) -> None:
    """Apply all edge generators in dependency order, then sync the
    "shadow" attrs that name what a related entity is (capital_name,
    headquarters_city_name, alma_mater_name, country_name) so atomic
    questions and relational-1hop questions return the same answer.
    """
    add_edges_located_in(world, seed=seed + 1000)
    add_edges_capital(world, seed=seed + 1010)
    add_edges_bordered_by(world, seed=seed + 1020)
    add_edges_headquartered_in(world, seed=seed + 1030)
    add_edges_subsidiary_of(world, seed=seed + 1050)
    add_edges_born_in(world, seed=seed + 1070)
    add_edges_works_at(world, seed=seed + 1080)
    add_edges_alma_mater(world, seed=seed + 1090)
    add_edges_mentor(world, seed=seed + 1100)
    add_edges_spouse(world, seed=seed + 1110)
    add_edges_parent(world, seed=seed + 1120)
    add_edges_founded_by(world, seed=seed + 1040)
    add_edges_head_of_government(world, seed=seed + 1060)
    add_edges_event_happened_in(world, seed=seed + 1130)
    add_edges_event_primary_figure(world, seed=seed + 1140)
    add_edges_event_involved(world, seed=seed + 1150)
    add_edges_work_created_by(world, seed=seed + 1160)
    add_edges_work_published_by(world, seed=seed + 1170)

    _sync_shadow_attrs(world)


def _sync_shadow_attrs(world: World) -> None:
    """Reconcile the per-entity "_name" hint attrs with the resolved edge
    targets. Without this, atomic and relational questions can return
    inconsistent answers (e.g. Nation.attrs['capital_name']='Asgardholm'
    while world.neighbor(nation, 'capital').attrs['name']='Eskbridge').
    """
    # Entity is frozen dataclass with frozen=True — its attrs dict can still
    # be mutated since dict mutation isn't blocked by @frozen.
    for ent in world.entities.values():
        if ent.entity_type == "Nation":
            cap = world.neighbor(ent.key, "capital")
            if cap is not None:
                ent.attrs["capital_name"] = cap.attrs["name"]
        elif ent.entity_type == "Organization":
            hq = world.neighbor(ent.key, "headquartered_in")
            if hq is not None:
                ent.attrs["headquarters_city_name"] = hq.attrs["name"]
        elif ent.entity_type == "Person":
            am = world.neighbor(ent.key, "alma_mater")
            if am is not None:
                ent.attrs["alma_mater_name"] = am.attrs["name"]
        elif ent.entity_type == "Place":
            country = world.neighbor(ent.key, "located_in")
            if country is not None:
                ent.attrs["country_name"] = country.attrs["name"]


# ── Internal helpers ────────────────────────────────────────────────


def _would_create_cycle(
    mentor_of: dict[str, str], *, src: str, dst: str, max_depth: int = 20,
) -> bool:
    """Check if adding (src → dst) would create a cycle in the mentor chain.

    Walks mentor_of from `dst` upward; if we encounter `src`, we'd form a cycle.
    """
    cur = dst
    for _ in range(max_depth):
        if cur == src:
            return True
        cur = mentor_of.get(cur)
        if cur is None:
            return False
    return False
