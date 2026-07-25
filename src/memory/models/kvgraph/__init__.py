"""KV-graph: an entity/event graph over the KV cache. See README.md and docs/design/kvgraph/."""
from .schema import DISCOURSE_RELATIONS, STATIVE_RELATIONS, Edge, Graph, Mention, Node, NodeKind, Relation

__all__ = ["Graph", "Node", "Edge", "Mention", "NodeKind", "Relation", "DISCOURSE_RELATIONS", "STATIVE_RELATIONS"]
