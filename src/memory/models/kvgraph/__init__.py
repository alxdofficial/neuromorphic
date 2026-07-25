"""KV-graph: an entity/event graph over the KV cache. See README.md and docs/design/kvgraph/."""
from .schema import DISCOURSE_ROLES, STATIVE_ROLES, Edge, Graph, Mention, Node, NodeKind, Role

__all__ = ["Graph", "Node", "Edge", "Mention", "NodeKind", "Role", "DISCOURSE_ROLES", "STATIVE_ROLES"]
