"""Entity + edge population.

Each builder takes a `World` (initially empty or partially populated) and
adds entities of a specific type. Edges between entities are added via
`edges.py` *after* all entity-type pools are populated, so cross-type
edges (Person → Organization, etc.) can be drawn from real entity keys.

Call order (see `generate.py`):
    1. build_nations(world, n=40)
    2. build_places(world, n=80)            # cities populated; located_in edges
    3. build_orgs(world, n=80)              # organizations; headquartered_in edges
    4. build_people(world, n=300)           # people
    5. build_events(world, n=50)
    6. build_works(world, n=50)
    7. populate_relationships(world, ...)   # cross-edges: mentor, spouse,
                                            #              works_at, founded_by, etc.
"""

from scripts.data_build.generate.bio.builders.people import build_people
from scripts.data_build.generate.bio.builders.orgs import build_orgs
from scripts.data_build.generate.bio.builders.nations import build_nations
from scripts.data_build.generate.bio.builders.places import build_places
from scripts.data_build.generate.bio.builders.events import build_events
from scripts.data_build.generate.bio.builders.works import build_works
from scripts.data_build.generate.bio.builders.edges import populate_relationships

__all__ = [
    "build_people", "build_orgs", "build_nations", "build_places",
    "build_events", "build_works", "populate_relationships",
]
