"""Types shared by every stage of the KV-graph pipeline.

The pipeline is a chain of narrow transformations, and this module is the vocabulary all of them speak:

    text  --parse-->  [Mention]  --build-->  [Node]  --edges-->  Graph  --ground-->  Graph with KV
                                                                       --merge-->   persistent Graph

Nothing here touches torch or spaCy on purpose. These are plain data, so a graph can be built, printed,
diffed, and asserted about in a test without loading a model. `Graph.pretty()` exists because the whole
design rests on the memory being INSPECTABLE — if you cannot read the graph, you cannot tell a binding
failure from a parsing failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(str, Enum):
    ENTITY = "entity"   # a particular individual: Maria, the farm, the letter
    EVENT = "event"     # a particular event or state, REIFIED as a node: the sale, the delivery


class Relation(str, Enum):
    """The canonical relation inventory — deliberately small and CLOSED.

    Size is the point: `K = len(Relation)` is the free-bits constraint on the edge channel, exactly as the node
    budget is on the node channel. A small K forces relations to be REUSED across the corpus, and reuse is
    what makes the topology carry information a flat bank does not have. If every verb were allowed to invent
    its own label ("payload", "merchandise" — the same relation under two names) the vocabulary would grow with
    the corpus, reuse would never happen, and the typed-operator bank could not be shared.

    Polarity, tense and quantity are NOT relations. They are attributes on a node (see `Node.attrs`), because
    they modify one node rather than relating two.
    """

    # These names are BORROWED, not invented: they are the standard thematic/semantic relations used by
    # PropBank, FrameNet, VerbNet and AMR. Every one answers a plain question, given in its comment.

    # --- WHO IS INVOLVED IN THIS EVENT, AND HOW? (event hub -> entity) --------------------------------
    AGENT = "agent"                # who DID it.            Sale --agent--> Brother
    THEME = "theme"                # what it was done TO.   Sale --theme--> farm
    RECIPIENT = "recipient"        # who GOT it.            Delivery --recipient--> Maria
    EXPERIENCER = "experiencer"    # who FELT/THOUGHT it — an agent that isn't doing anything, just
    #                                undergoing a mental state.  Believe --experiencer--> Maria.
    #                                A nicety; collapsing it into AGENT would lose little.
    INSTRUMENT = "instrument"      # what it was done WITH.
    SOURCE = "source"              # where it came FROM.
    GOAL = "goal"                  # where it went TO.
    CONTENT = "content"            # WHAT WAS SAID/BELIEVED/RUMOURED — the argument is a whole
    #                                proposition, not a thing.  Rumour --content--> Sale.
    #                                This single relation is what makes "the rumour THAT he sold it"
    #                                representable; without it the graph asserts the sale as
    #                                world-fact. See THESIS.md §3.3. THE load-bearing relation.

    # --- UNDER WHAT CIRCUMSTANCES? (event hub -> entity) ----------------------------------------------
    MANNER = "manner"              # HOW.    Only when there is a noun to point at ("sold by auction");
    #                                a bare adverb ("secretly") is an attribute, not an edge.
    TIME = "time"                  # WHEN.
    LOCATION = "location"          # WHERE.
    PURPOSE = "purpose"            # WHAT FOR.
    BENEFICIARY = "beneficiary"    # FOR WHOSE BENEFIT.

    # --- FACTS RELATING TWO THINGS, NO EVENT NEEDED (entity -> entity) --------------------------------
    # These never need a hub: binary, and nothing ever points at them.
    PART_OF = "part_of"            # X is part of Y.
    MEMBER_OF = "member_of"        # X belongs to group Y.
    OWNS = "owns"                  # X possesses Y.         Maria --owns--> brother ("her brother")
    LOCATED_IN = "located_in"      # X sits inside Y.
    ATTRIBUTE = "attribute"        # X has property Y.

    # --- HOW DO TWO EVENTS RELATE? (event hub -> event hub) -------------------------------------------
    # Only expressible because events are nodes. Both endpoints must be EVENTs.
    BECAUSE = "because"            # one caused the other.
    ALTHOUGH = "although"          # one happened despite the other.
    CONTRAST = "contrast"          # the two are set against each other.
    THEN = "then"                  # one followed the other.
    SUPERSEDES = "supersedes"      # fact-update: the newer assertion RETRACTS the older one.
    #                                Written by the merge stage, not by the parser.


#: Relations whose endpoints must both be EVENT nodes. Checked by `Graph.validate()`.
DISCOURSE_RELATIONS = frozenset({
    Relation.BECAUSE, Relation.ALTHOUGH, Relation.CONTRAST, Relation.THEN, Relation.SUPERSEDES})

#: Relations that connect two ENTITY nodes directly, with no hub.
STATIVE_RELATIONS = frozenset({
    Relation.PART_OF, Relation.MEMBER_OF, Relation.OWNS, Relation.LOCATED_IN, Relation.ATTRIBUTE})


@dataclass
class Mention:
    """One surface occurrence. Many mentions collapse into one Node; that collapse IS the compression.

    Character offsets come from the parser; token indices are filled in by `align.py` and are what the
    KV pooling actually indexes. Both are kept because a mismatch between them is the pipeline's most
    dangerous silent failure — see `align.py`.
    """

    char_start: int
    char_end: int
    text: str
    kind: NodeKind
    head_lemma: str                       # syntactic head of the span: "farm" in "the family farm"
    token_start: int = -1                 # inclusive; -1 until align.py runs
    token_end: int = -1                   # exclusive
    head_token: int = -1                  # the head's own token index — carries most of the phrase meaning
    ner_type: str | None = None
    sent_idx: int = -1
    cluster_id: int | None = None         # coref cluster; mentions sharing one become a single Node

    @property
    def token_span(self) -> range:
        if self.token_start < 0:
            raise ValueError(f"mention {self.text!r} was never aligned to tokens — run align.py first")
        return range(self.token_start, self.token_end)


@dataclass
class Node:
    """An entity or a reified event. Its identity is a KEY; its facts live on its EDGES.

    Deliberately NOT "the pooled meaning of all mentions": pooling every mention's content into the node
    makes it membership-only (the pool-then-address trap). An entity node's vector being a generic "Maria"
    is CORRECT — the specific propositions live in the event nodes attached to it.
    """

    node_id: int
    kind: NodeKind
    label: str                                     # head/predicate lemma, for humans and for debugging
    mentions: list[Mention] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)      # polarity, tense, ner_type, recency, ...
    #: Filled by ground.py — pooled pre-RoPE K and V, one entry per layer. Kept off the dataclass's
    #: __repr__ path deliberately; these are large tensors and a graph must stay printable.
    kv: object | None = None

    @property
    def token_positions(self) -> list[int]:
        """Union of every mention's tokens. This is the index set ground.py pools KV over."""
        return sorted({t for m in self.mentions for t in m.token_span})

    @property
    def first_token(self) -> int:
        """First-mention token index. The SORT KEY for RoPE position assignment — note the injected
        position is the node's RANK in this order, never this index itself (see inject.py)."""
        return min(m.token_start for m in self.mentions)

    def __repr__(self) -> str:  # keep tensors out of reprs so graphs stay printable in tests
        return f"Node({self.node_id}, {self.kind.value}, {self.label!r}, {len(self.mentions)} mentions)"


@dataclass
class Edge:
    """Directed and typed. For relation edges the source is ALWAYS the event hub — direction is uniform and
    semantically empty, and the RELATION carries the argument structure (log2 K bits, versus the 1 bit that
    edge orientation carries when verbs are edges). See THESIS.md §3.3."""

    src: int
    dst: int
    #: The discrete grammatical relation. 1-of-K, from the closed inventory. This is what the mixer's
    #: operator bank is indexed by, and what the edge ablation collapses.
    relation: Relation
    #: The continuous EDGE VECTOR: everything the discrete relation throws away. `theme` says the farm was
    #: acted on; the edge vector is what carries "merchandise-ness". Attention-pooled from the tokens that
    #: licensed the arc plus the heads of both endpoints, then refined by the mixer's edge-update.
    #: BOUNDED by construction (low-rank, zero-init, decay) — if it is free to carry everything, the
    #: relation goes decorative and we have rebuilt the loss-neutrality wall in a new idiom.
    vec: object | None = None
    #: The token indices that licensed this arc. Where `vec` is pooled from, and debugging provenance.
    licensing_tokens: tuple[int, ...] = ()
    provenance: str = ""    # the dependency arc / rule this came from. Debugging only, never a feature.


@dataclass
class Graph:
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> Node:
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, src: int, dst: int, relation: Relation, provenance: str = "",
                 licensing_tokens: tuple[int, ...] = ()) -> Edge:
        edge = Edge(src, dst, relation, licensing_tokens=licensing_tokens, provenance=provenance)
        self.edges.append(edge)
        return edge

    def neighbors(self, node_id: int) -> list[tuple[Edge, Node]]:
        """Outgoing only. Traversal walks the SYMMETRIZED graph (see mixer/retrieval); direction and relation
        are for computation, not for reachability, which is why no inverse edges are ever stored."""
        return [(e, self.nodes[e.dst]) for e in self.edges if e.src == node_id]

    def by_kind(self, kind: NodeKind) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind is kind]

    def validate(self) -> list[str]:
        """Structural invariants. Returns problems rather than raising, so a whole corpus can be swept and
        the error RATE measured — extraction quality is a number we need, not an exception we dodge."""
        problems = []
        for e in self.edges:
            if e.src not in self.nodes or e.dst not in self.nodes:
                problems.append(f"edge {e.relation.value} references a missing node: {e.src}->{e.dst}")
                continue
            src, dst = self.nodes[e.src], self.nodes[e.dst]
            if e.relation in DISCOURSE_RELATIONS and not (src.kind is NodeKind.EVENT and dst.kind is NodeKind.EVENT):
                problems.append(f"discourse edge {e.relation.value} must join two events: {src} -> {dst}")
            if e.relation in STATIVE_RELATIONS and src.kind is NodeKind.EVENT:
                problems.append(f"stative edge {e.relation.value} should not leave an event hub: {src}")
            # NB: CONTENT is deliberately unconstrained in its target kind. Both are legitimate:
            #   rumour --content--> (sell)     an event  ("the rumour THAT he sold it")
            #   letter --content--> rumour     an entity ("a letter ABOUT it")
            # What makes CONTENT well-formed is that the SOURCE is information-bearing, which is a
            # lexical property we cannot check structurally — so we do not pretend to.
        for n in self.nodes.values():
            if not n.mentions:
                problems.append(f"{n} has no mentions — nothing to pool KV from")
        return problems

    def pretty(self) -> str:
        """Human-readable dump. The design commits to a decodable, inspectable memory; this is where that
        commitment is cashed in during debugging."""
        lines = []
        for n in sorted(self.nodes.values(), key=lambda x: x.first_token):
            marker = "()" if n.kind is NodeKind.EVENT else "[]"
            surface = " | ".join(m.text for m in n.mentions)
            lines.append(f"{marker[0]}{n.label}{marker[1]:<18} #{n.node_id:<3} tok{n.token_positions[:6]}"
                         f"  <- {surface}")
            for edge, dst in self.neighbors(n.node_id):
                lines.append(f"      --{edge.relation.value:<12}-> #{dst.node_id} {dst.label}")
        return "\n".join(lines)
