"""Stage 2+3 — parse output -> a `Graph` of nodes and typed edges.

Two collapses happen here, and they are the whole of "how a paragraph becomes a graph":

  1. mention level -> node level.  One entity node per within-window coref cluster; one event node per
     predicate. **Node count is data-dependent** — we never choose K. This is where compression first
     appears: N mentions of Maria become one node, which is the deduplication language structurally cannot
     do (every mention must be re-uttered).
  2. dependency arcs -> canonical relations, with every predicate reified as a HUB whose spokes are
     role-labelled. Direction is uniform (hub -> participant) and semantically empty; the relation carries
     the argument structure.

Token indices are attached here (via align.py) because everything downstream — pooling, eviction, injection —
indexes KV by token position, and a node with no aligned tokens has nothing to pool.
"""
from __future__ import annotations

from .align import tokens_for_span
from .edges import is_attribute, is_merge_signal, relation_for_argument
from .parse import ParseResult
from .schema import Graph, Mention, Node, NodeKind, Relation


def _attach_tokens(m: Mention, offsets) -> bool:
    """Fill token_start/token_end/head_token (all LM-token indices). -> False if the mention covers no
    token, in which case it is dropped — a node with nothing to pool is worse than no node."""
    toks = tokens_for_span(offsets, m.char_start, m.char_end)
    if not toks:
        return False
    m.token_start, m.token_end = toks[0], toks[-1] + 1
    if m.head_char is not None:
        head_toks = tokens_for_span(offsets, *m.head_char)
        m.head_token = head_toks[-1] if head_toks else m.token_end - 1
    else:
        # Fallback: the LAST token of the span. English NPs are head-final ("the family farm"), so this is
        # usually right, but head_char is set by the parser and should be preferred.
        m.head_token = m.token_end - 1
    return True


def build_graph(parse: ParseResult, offsets: list[tuple[int, int]], *,
                first_node_id: int = 0) -> tuple[Graph, dict[int, int]]:
    """-> (graph, mention_head_token -> node_id).

    The returned map is what merge.py needs to link this window's nodes to the persistent graph, and what
    the caller needs if it wants to trace a node back to its surface form.
    """
    g = Graph()
    nid = first_node_id
    # Keyed by the PARSER's token index, because that is the index space dependency arcs are stated in.
    # Keying this by LM token index instead silently drops every edge — the two tokenizations disagree.
    tok2node: dict[int, int] = {}

    # ── entity nodes: one per coref cluster; unclustered mentions get their own node ─────────────
    by_cluster: dict[int, list[Mention]] = {}
    singletons: list[Mention] = []
    for m in parse.mentions:
        if not _attach_tokens(m, offsets):
            continue
        (by_cluster.setdefault(m.cluster_id, []) if m.cluster_id is not None else singletons).append(m)  # type: ignore[union-attr]

    for group in list(by_cluster.values()) + [[m] for m in singletons]:
        # Label from the longest non-pronoun mention: "Maria" is a better handle than "she", and the
        # longest surface form is the most informative one the cluster offers.
        named = [m for m in group if m.head_lemma not in _PRONOUNS] or group
        label = max(named, key=lambda m: m.char_end - m.char_start).head_lemma
        node = Node(node_id=nid, kind=NodeKind.ENTITY, label=label, mentions=list(group))
        ner = next((m.ner_type for m in group if m.ner_type), None)
        if ner:
            node.attrs["ner_type"] = ner
        g.add_node(node)
        for m in group:
            if m.parser_token >= 0:
                tok2node[m.parser_token] = nid
        nid += 1

    # ── event nodes: one per predicate, no within-window event coref ─────────────────────────────
    for m in parse.predicates:
        if not _attach_tokens(m, offsets):
            continue
        g.add_node(Node(node_id=nid, kind=NodeKind.EVENT, label=m.head_lemma, mentions=[m]))
        if m.parser_token >= 0:
            tok2node[m.parser_token] = nid
        nid += 1

    # ── edges ────────────────────────────────────────────────────────────────────────────────────
    # A modifier becomes an EDGE only when its target is independently referable — it appears elsewhere as
    # a standalone mention, or carries an NER type. Otherwise it is a string attribute. Budget: a node per
    # adjective would be eaten by things nothing ever points at.
    referable = {t for t, n in tok2node.items()
                 if n in g.nodes and (len(g.nodes[n].mentions) > 1 or g.nodes[n].attrs.get("ner_type"))}

    for arc in parse.arcs:
        src = tok2node.get(arc.head_token)
        dst = tok2node.get(arc.child_token)
        if src is None:
            continue

        if is_merge_signal(arc.dep):
            # Apposition asserts identity. Merge rather than relate — but only ENTITY/ENTITY, since
            # "the sale, a disaster" apposes an event to a description and fusing them would be a
            # category error.
            if dst is not None and dst != src and \
                    g.nodes[src].kind is g.nodes[dst].kind is NodeKind.ENTITY:
                _absorb(g, keep=src, drop=dst, tok2node=tok2node)
            continue

        attr = is_attribute(arc.dep, modifier_is_referable=(arc.child_token in referable))
        if attr is not None:
            # Store the surface text, not a node: `polarity="not"`, `manner="secretly"`.
            g.nodes[src].attrs[attr] = _child_text(parse, arc)
            continue

        if dst is None or dst == src:
            continue
        rel = relation_for_argument(
            arc.dep, head_lemma=g.nodes[src].label, head_is_passive=arc.head_is_passive,
            prep_lemma=arc.marker if arc.dep in ("prep", "pobj", "nmod") else None,
            obj_ent_type=arc.obj_ent_type,
            mark_lemma=arc.marker if arc.dep == "advcl" else None)
        if rel is None:
            continue
        # Discourse relations join two EVENT hubs by definition; a parse that hands us an entity endpoint
        # is a parse error, and emitting the edge anyway would trip Graph.validate downstream.
        if rel in _DISCOURSE and not (g.nodes[src].kind is g.nodes[dst].kind is NodeKind.EVENT):
            continue
        # `licensing_tokens` feeds ground.pool_hidden, which indexes the LM's KV — so translate out of
        # parser index space here. Getting this wrong pools the edge vector from unrelated tokens.
        lic = tuple(sorted({t for n in (src, dst) for m in g.nodes[n].mentions
                            for t in (m.head_token,) if t >= 0}))
        g.add_edge(src, dst, rel, provenance=arc.dep, licensing_tokens=lic)
    return g, tok2node


_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "him", "her", "them", "me", "us",
             "my", "your", "his", "its", "our", "their", "this", "that", "these", "those"}
_DISCOURSE = {Relation.BECAUSE, Relation.ALTHOUGH, Relation.CONTRAST, Relation.THEN}


def _child_text(parse: ParseResult, arc) -> str:
    """Surface text of an arc's child, for attribute values. Falls back to the dep label when the child is
    not a tracked mention (bare adverbs and negations usually are not)."""
    for m in parse.mentions + parse.predicates:
        if m.parser_token == arc.child_token:      # parser index space, like every other arc lookup
            return m.text
    # Bare modifiers ("secretly", "not", "the") are not mentions, so fall back to the token's own SURFACE
    # TEXT. Falling through to the dependency label stored manner="advmod" instead of manner="secretly",
    # which is not an attribute value at all.
    return parse.token_text.get(arc.child_token) or arc.marker or arc.dep


def _absorb(g: Graph, *, keep: int, drop: int, tok2node: dict[int, int]) -> None:
    """Fold node `drop` into `keep`: mentions unite, edges rewire, attrs fill gaps.

    This is COREF MERGE (identity — content unified, nothing archived), which is a different operation from
    contraction in merge.py (budget — content archived behind a pointer). Never conflate them: contraction
    must not make identity claims, because over-merging produces wrong answers where under-merging only
    produces missing ones.
    """
    if drop not in g.nodes or keep not in g.nodes:
        return
    kn, dn = g.nodes[keep], g.nodes[drop]
    kn.mentions.extend(dn.mentions)
    for k, v in dn.attrs.items():
        kn.attrs.setdefault(k, v)
    for e in g.edges:
        if e.src == drop:
            e.src = keep
        if e.dst == drop:
            e.dst = keep
    g.edges = [e for e in g.edges if e.src != e.dst]
    for t, n in list(tok2node.items()):
        if n == drop:
            tok2node[t] = keep
    del g.nodes[drop]
