"""Biographical task — scenario data model.

The biographical "scenario" is the entire world graph (entities + edges).
Unlike per-instance tasks (boxes, calendar), one biographical scenario
yields many passages + questions because each entity becomes one
passage and many question types are walked across the whole graph.

build_scenario() builds the world once. The driver iterates n_scenarios
times — each iteration with a different rng — to produce independent
worlds (e.g. for train vs val name pools).
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.data_gen.tasks.biographical.world import World
from scripts.data_gen.tasks.biographical.builders import (
    build_nations, build_places, build_orgs, build_people, build_events,
    build_works, populate_relationships,
)
from scripts.data_gen.tasks.biographical.builders.people import (
    build_public_figures,
)


@dataclass
class BiographicalScenario:
    """One whole world. Passes through to the existing world.py classes."""
    scenario_idx: int
    world: World
    samples_per_entity: int


def build_scenario(
    rng, scenario_idx: int, *,
    n_people: int = 200, n_public_figures: int = 30,
    n_orgs: int = 60, n_nations: int = 20, n_places: int = 40,
    n_events: int = 30, n_works: int = 30,
    samples_per_entity: int = 1,
) -> BiographicalScenario:
    """Build one fully-populated world."""
    # Translate the driver's rng to per-builder seeds for determinism.
    # We use the rng to derive seeds (don't pass rng directly into
    # builders — they use random.Random instances internally).
    base_seed = rng.randint(0, 2**31 - 1)
    world = World()
    build_nations(world, n=n_nations, seed=base_seed)
    build_places(world, n=n_places, seed=base_seed)
    build_orgs(world, n=n_orgs, seed=base_seed)
    build_people(world, n=n_people, seed=base_seed)
    build_public_figures(world, n=n_public_figures, seed=base_seed + 5000)
    build_events(world, n=n_events, seed=base_seed)
    build_works(world, n=n_works, seed=base_seed)
    populate_relationships(world, seed=base_seed)
    return BiographicalScenario(
        scenario_idx=scenario_idx,
        world=world,
        samples_per_entity=samples_per_entity,
    )


def config_space_size() -> int:
    """Approximate distinct worlds.

    Per scenario: ~470 entities, each drawing from pools of size hundreds.
    The combinatorial space of distinct worlds is astronomical at the
    seed level alone (each builder uses a random.Random seed and shuffles
    its pool). Conservative lower bound: assume ~10K independent random
    decisions per world × ~10 choices each.
    """
    return 10 ** 30   # effectively unbounded
