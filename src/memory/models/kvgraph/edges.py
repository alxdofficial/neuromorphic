"""Edge creation: dependency arcs -> canonical relations.

This is the whole of the v0 edge rule, and it is deliberately DETERMINISTIC. Every edge comes from a
licensed parser arc, so `E` is O(N) — the sparse typed topology the mixer wants is handed to us rather
than induced. Nothing here is learned; a later Gumbel head may retype/add/prune edges, and when it does,
these tables become both its initialisation and its ablation control.

The construction is neo-Davidsonian (THESIS.md §3.3): each predicate becomes an EVENT hub and every
argument hangs off it by a relation-labelled spoke. Direction is uniform (hub -> participant) and carries no
meaning; the RELATION carries the argument structure.

Accuracy of these edges is bounded by the parse, and the two known systematic gaps are worth stating up
front rather than discovering later:

  * VOICE. Dependency labels are syntactic, not semantic. "The farm was sold by her brother" gives
    nsubjpass=farm and agent=brother — the naive nsubj->AGENT mapping would make the FARM the seller.
    Handled below; it is not optional.
  * CONTROL / RAISING. "Maria wanted to call him" has no nsubj on `call`; its subject is inherited from
    `want`. A dependency parse will not recover it and we simply lose that argument. SRL/AMR does recover
    it, which is the concrete reason AMR is the upgrade path rather than a nicety.
"""
from __future__ import annotations

from .schema import Relation

# ---------------------------------------------------------------------------------------------------
# Direct arc -> relation. Applies when the head is an EVENT hub and the child is one of its arguments.
# ---------------------------------------------------------------------------------------------------
DEP_TO_RELATION: dict[str, Relation] = {
    "nsubj": Relation.AGENT,          # overridden to EXPERIENCER for mental predicates, see below
    "dobj": Relation.THEME,
    "obj": Relation.THEME,            # UD-style label, in case a UD parser is swapped in
    "dative": Relation.RECIPIENT,
    "iobj": Relation.RECIPIENT,
    "attr": Relation.THEME,           # copula complement: "the farm IS an asset"
    "oprd": Relation.THEME,
    "acomp": Relation.ATTRIBUTE,      # "the farm is LARGE"

    # Clausal complements: the proposition-as-argument link. `acl`/`relcl` is the one that makes
    # "the rumour THAT her brother sold the farm" representable -- Rumour --content--> Sale.
    # Without this single arc the graph asserts the sale as world-fact. THESIS.md §3.3.
    "ccomp": Relation.CONTENT,
    "xcomp": Relation.CONTENT,
    "acl": Relation.CONTENT,
    "relcl": Relation.CONTENT,

    "npadvmod": Relation.TIME,        # bare temporal NP: "he called YESTERDAY"
    # NOTE: `advmod` is deliberately absent. A bare adverb ("secretly") is not a particular, so there is
    # no node for a MANNER edge to point at -- it becomes an ATTRIBUTE on the event instead. MANNER fires
    # only for prepositional phrases with a nominal object ("sold BY AUCTION"), handled below.
}

# Entity -> entity, no hub needed: binary, and nothing ever points at them.
STATIVE_DEP_TO_RELATION: dict[str, Relation] = {
    "poss": Relation.OWNS,            # possessor -> possessed ("her brother" => Maria --owns--> brother)
    "amod": Relation.ATTRIBUTE,
    "compound": Relation.ATTRIBUTE,
}

# Arcs that are NOT edges. Recording them explicitly stops them being silently dropped.
ATTRIBUTE_DEPS: dict[str, str] = {
    "neg": "polarity",            # -> node attr polarity="-", NOT an edge. Ground facts need no scope region.
    "advmod": "manner",           # bare adverb: "sold SECRETLY". No particular to point at -> attribute.
    "aux": "tense",
    "auxpass": "voice",
    "det": "definiteness",        # "the sale" vs "a sale" -- a coref cue, not a relation
}

#: Modifier arcs are edges ONLY when their target is independently referable; otherwise they collapse into
#: a string attribute on the head. Rationale: budget. Making a node for every adjective would blow the node
#: count on things nothing ever points at -- but a compound like "the FAMILY farm" deserves a node if
#: "the family" is mentioned on its own elsewhere, because then a later fact can attach to it.
#: `modifier_is_referable` is decided by the caller (build.py), which is the stage that knows the full
#: mention list and NER types for the window.
MODIFIER_DEPS: frozenset[str] = frozenset({"amod", "compound", "nmod"})
#: Apposition asserts IDENTITY ("Maria, the owner, ..."), so it merges two nodes instead of relating them.
MERGE_DEPS: frozenset[str] = frozenset({"appos"})

# ---------------------------------------------------------------------------------------------------
# Prepositional phrases: the preposition lemma selects the relation of its object.
# ---------------------------------------------------------------------------------------------------
PREP_TO_RELATION: dict[str, Relation] = {
    "with": Relation.INSTRUMENT,
    "without": Relation.INSTRUMENT,
    "from": Relation.SOURCE,
    "to": Relation.GOAL,
    "into": Relation.GOAL,
    "toward": Relation.GOAL,
    "towards": Relation.GOAL,
    "for": Relation.PURPOSE,
    "of": Relation.PART_OF,
    "about": Relation.CONTENT,        # "a letter ABOUT it" -- often points at an event
    "by": Relation.AGENT,             # only inside a passive `agent` arc; else MANNER (handled in code)
}
#: in/at/on/during/after/before are ambiguous between place and time; resolved by the object, not the prep.
_SPATIOTEMPORAL_PREPS: frozenset[str] = frozenset({"in", "at", "on", "during", "after", "before", "over"})

#: spaCy NER labels that mark a temporal object, used to split TIME from LOCATION.
_TEMPORAL_ENT_TYPES: frozenset[str] = frozenset({"DATE", "TIME"})

# ---------------------------------------------------------------------------------------------------
# Discourse: `advcl` joins two clause heads, and the subordinator says which relation it is.
# Only expressible because both endpoints are event hubs -- with verbs-as-edges there is nothing to join.
# ---------------------------------------------------------------------------------------------------
MARK_TO_DISCOURSE: dict[str, Relation] = {
    "because": Relation.BECAUSE,
    "since": Relation.BECAUSE,        # AMBIGUOUS: also temporal. Causal is the commoner reading; accepted noise.
    "as": Relation.BECAUSE,           # AMBIGUOUS: also temporal/manner.
    "so": Relation.BECAUSE,
    "although": Relation.ALTHOUGH,
    "though": Relation.ALTHOUGH,
    "whereas": Relation.ALTHOUGH,
    "while": Relation.ALTHOUGH,       # AMBIGUOUS: also temporal.
    "after": Relation.THEN,
    "before": Relation.THEN,
    "when": Relation.THEN,
    "once": Relation.THEN,
}

#: Mental-state predicates take an EXPERIENCER, not an AGENT. Small closed class, cheap to special-case,
#: and it matters here: `believe` is exactly the verb whose subject is not a doer.
MENTAL_PREDICATES: frozenset[str] = frozenset({
    "believe", "know", "think", "feel", "want", "wish", "hope", "fear", "doubt", "suspect",
    "remember", "forget", "see", "hear", "notice", "realize", "realise", "understand", "like", "love",
})


def relation_for_argument(dep: str, *, head_lemma: str, head_is_passive: bool,
                      prep_lemma: str | None = None, obj_ent_type: str | None = None,
                      mark_lemma: str | None = None) -> Relation | None:
    """Map one dependency arc to a canonical relation, or None if it licenses no edge.

    `head_is_passive` flips the two arcs whose meaning inverts under passivisation. `prep_lemma`,
    `obj_ent_type` and `mark_lemma` are the extra context needed for the three context-sensitive cases
    (prepositions, spatiotemporal ambiguity, subordinators).
    """
    # -- passive voice inverts the mapping; get this wrong and the theme becomes the agent -------------
    if dep in ("nsubjpass", "nsubj:pass"):
        return Relation.THEME
    if dep == "agent":                       # the "by" phrase of a passive IS the agent
        return Relation.AGENT
    if dep == "nsubj":
        if head_is_passive:                  # a parser that mislabels the passive subject as plain nsubj
            return Relation.THEME
        return Relation.EXPERIENCER if head_lemma in MENTAL_PREDICATES else Relation.AGENT

    # -- prepositional phrases -------------------------------------------------------------------------
    if dep in ("prep", "pobj", "nmod"):
        if prep_lemma is None:
            return None
        if prep_lemma in _SPATIOTEMPORAL_PREPS:
            return Relation.TIME if (obj_ent_type in _TEMPORAL_ENT_TYPES) else Relation.LOCATION
        if prep_lemma == "by" and not head_is_passive:
            return Relation.MANNER               # "sold BY auction" is manner, not agent
        return PREP_TO_RELATION.get(prep_lemma)

    # -- discourse -------------------------------------------------------------------------------------
    if dep == "advcl":
        return MARK_TO_DISCOURSE.get(mark_lemma or "", Relation.THEN)   # bare advcl => loose temporal link

    return DEP_TO_RELATION.get(dep) or STATIVE_DEP_TO_RELATION.get(dep)


def is_attribute(dep: str, *, modifier_is_referable: bool = False) -> str | None:
    """-> the attribute name this arc sets on the head node, or None if it should be an edge instead.

    `modifier_is_referable` lets build.py promote a modifier to a real edge when its target is a thing the
    rest of the document can point at (it appears as a standalone mention, or carries an NER type).
    "the family farm" becomes an ATTRIBUTE edge to a `family` node if "the family" is mentioned elsewhere,
    and a plain string attribute otherwise.
    """
    if dep in MODIFIER_DEPS:
        return None if modifier_is_referable else "modifier"
    return ATTRIBUTE_DEPS.get(dep)


def is_merge_signal(dep: str) -> bool:
    """-> True if this arc asserts that two mentions are the SAME particular (apposition)."""
    return dep in MERGE_DEPS
