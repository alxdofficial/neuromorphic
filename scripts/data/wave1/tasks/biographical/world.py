"""World — typed graph of entities with cross-references.

Wave 1 v5 represents the entity universe as a graph (Entity nodes,
typed Edge relations) instead of v4's flat dict-of-strings. This lets
us generate compositional questions ("X's mentor's workplace") by
walking the graph at question-generation time.

Schema:
- Entity: local attrs (no cross-refs), surface names (for split)
- Edge:   typed relation (src, rel, dst) where dst may be another
          entity_key or a literal (year, etc.)
- World:  container + indexes + path-resolution helpers

See `docs/wave1_v5_protocol.md` §3 for the full schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Iterator


# ── Entity / Edge ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Entity:
    """A node in the world graph.

    Attributes (`attrs`) are local — they describe THIS entity only.
    All cross-entity references live in edges, not attrs.

    `surface_names` is the closure of human-readable names this entity
    is rendered as (e.g. ["Maria Halverson", "Maria", "Halverson", "Dr. Halverson"]).
    Used by the name-disjoint train/val splitter.
    """
    key: str
    entity_type: str                              # "Person" | "Organization" | "Nation" | "Place" | "Event" | "Work"
    attrs: dict[str, Any] = field(default_factory=dict)
    surface_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Edge:
    """A typed directed relation from one entity to another (or to a literal).

    `dst` is either:
    - an entity_key string (for entity-to-entity edges like "mentor")
    - a primitive literal (int year, str descriptor) for entity-to-literal
      edges. Most literal-valued data goes in `Entity.attrs` instead;
      literal edges are reserved for relations that need to be queried
      via the graph (e.g. an event's `year_value` for temporal-ordering
      compose).
    """
    src: str
    rel: str
    dst: Any                                      # str (entity_key) | int | str (literal) | None


# ── World container ─────────────────────────────────────────────────


class World:
    """Container for entities + edges, with O(1) neighbor lookup and
    path resolution for compositional question generation.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.edges: list[Edge] = []
        self._edges_by_src: dict[str, list[Edge]] = defaultdict(list)
        self._edges_by_dst: dict[str, list[Edge]] = defaultdict(list)

    # ── mutation ────────────────────────────────────────────────────

    def add_entity(self, ent: Entity) -> None:
        if ent.key in self.entities:
            raise ValueError(f"duplicate entity_key: {ent.key}")
        self.entities[ent.key] = ent

    def add_edge(self, edge: Edge) -> None:
        # Caller is responsible for ensuring src exists. dst may be a
        # literal so we don't validate it.
        if edge.src not in self.entities:
            raise ValueError(f"edge src {edge.src!r} not in world")
        self.edges.append(edge)
        self._edges_by_src[edge.src].append(edge)
        if isinstance(edge.dst, str) and edge.dst in self.entities:
            self._edges_by_dst[edge.dst].append(edge)

    # ── graph queries ───────────────────────────────────────────────

    def get_edges(self, src: str, rel: str | None = None) -> list[Edge]:
        """Outgoing edges from `src`, optionally filtered by relation."""
        edges = self._edges_by_src.get(src, [])
        if rel is None:
            return list(edges)
        return [e for e in edges if e.rel == rel]

    def neighbor(self, src: str, rel: str) -> Entity | None:
        """First neighbor via `rel`, or None if missing.

        For many-cardinality relations (parent, bordered_by, involved),
        this returns the first edge's target — useful for deterministic
        question generation. Use `neighbors()` to get the full list.
        """
        es = self.get_edges(src, rel)
        if not es:
            return None
        dst = es[0].dst
        if isinstance(dst, str) and dst in self.entities:
            return self.entities[dst]
        return None

    def neighbors(self, src: str, rel: str) -> list[Entity]:
        """All neighbors via `rel` (for many-cardinality relations)."""
        out: list[Entity] = []
        for e in self.get_edges(src, rel):
            if isinstance(e.dst, str) and e.dst in self.entities:
                out.append(self.entities[e.dst])
        return out

    def resolve_path(
        self, src: str, path: tuple[str, ...],
    ) -> Entity | Any | None:
        """Walk a relation path starting from entity `src`. Returns the
        terminal entity, or None if any hop is missing.

        Path is a tuple of relation names; e.g. ("mentor", "works_at").
        Each step calls neighbor() so paths must use 0-or-1 cardinality
        relations.
        """
        cur: Entity | None = self.entities.get(src)
        for rel in path:
            if cur is None:
                return None
            cur = self.neighbor(cur.key, rel)
        return cur

    def resolve_path_to_attr(
        self, src: str, path: tuple[str, ...], attr: str,
    ) -> Any | None:
        """Convenience: walk a path, then read an attr from the terminal
        entity. Returns None on any miss."""
        end = self.resolve_path(src, path)
        if end is None or not isinstance(end, Entity):
            return None
        return end.attrs.get(attr)

    def find_paths_of_length(
        self, src: str, length: int,
    ) -> list[tuple[tuple[str, ...], Entity]]:
        """Enumerate all paths of exactly `length` hops starting from `src`,
        returning (relation_tuple, terminal_entity) pairs.

        Used by relational-question generators to discover valid 1/2/3-hop
        question targets without hard-coding the relation chains.
        """
        if length <= 0:
            ent = self.entities.get(src)
            return [((), ent)] if ent is not None else []
        out: list[tuple[tuple[str, ...], Entity]] = []
        for e in self.get_edges(src):
            if not isinstance(e.dst, str) or e.dst not in self.entities:
                continue
            for sub_path, end in self.find_paths_of_length(e.dst, length - 1):
                out.append(((e.rel,) + sub_path, end))
        return out

    # ── iteration ────────────────────────────────────────────────────

    def entities_of_type(self, entity_type: str) -> Iterator[Entity]:
        for ent in self.entities.values():
            if ent.entity_type == entity_type:
                yield ent

    def __len__(self) -> int:
        return len(self.entities)
