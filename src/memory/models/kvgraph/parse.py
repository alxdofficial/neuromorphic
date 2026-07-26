"""Stage 1 — text -> mentions, predicates, dependency arcs.

Off-the-shelf and TRAINING-FREE. The parser supplies GROUPING DECISIONS only (which spans are one entity,
which span is an event, what role each argument plays); all content comes from pooled KV. That is what keeps
this a graph over hidden states rather than a text knowledge graph.

Two levels, escalate only when a measurement says to (BUILD.md §8 / README):
  v0  spaCy dependency parse + noun chunks  (+ optional coref)  <- here
  v1  AMR / SRL for control-raising and proper predicate senses
  v2  an LLM extractor, offline, as a distillation teacher

Coreference is optional at import time on purpose. `fastcoref` pulls a transformer and a large dependency
tree; without it we fall back to a deterministic string/lemma matcher, which is close to sufficient on
`factconsolidation` (templated, named entities repeated near-verbatim) and clearly insufficient on dialogue.
`Parser.coref_backend` reports which one ran so a result can never be silently attributed to the wrong one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Mention, NodeKind

#: Nominalisations that denote events ("the sale", "her refusal"). A noun in this set becomes an EVENT node,
#: which is what lets `Rumour --content--> Sale` attach to a noun. Deliberately small and lexical: a learned
#: eventiveness classifier is a phase-2 upgrade, and getting it wrong costs a node, not a wrong answer.
_EVENTIVE_NOUN_SUFFIXES = ("tion", "sion", "ment", "ance", "ence", "al", "ure", "age", "ing")
#: Relativizers: spaCy attaches them as an extra object of the relative clause's verb.
_RELATIVIZERS = {"that", "which", "who", "whom", "whose"}
_EVENTIVE_NOUNS = {"sale", "purchase", "delivery", "meeting", "call", "visit", "death", "birth",
                   "arrival", "departure", "change", "update", "transfer", "gift", "loan", "trip"}


@dataclass
class Arc:
    """One dependency arc, already resolved to the mention/predicate indices it connects."""

    head_token: int
    child_token: int
    dep: str
    #: preposition lemma for `prep`/`pobj` arcs, subordinator lemma for `advcl`; None otherwise.
    marker: str | None = None
    #: NER label of the arc's object, used to split TIME from LOCATION on in/at/on.
    obj_ent_type: str | None = None
    head_is_passive: bool = False


@dataclass
class ParseResult:
    text: str
    mentions: list[Mention] = field(default_factory=list)
    #: Predicate mentions (verbs + eventive nominals). Kept separate from `mentions` because they become
    #: EVENT nodes and the two are built differently in build.py.
    predicates: list[Mention] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    #: token index -> index into `mentions` or `predicates`, for resolving arcs to nodes.
    head_token_to_mention: dict[int, int] = field(default_factory=dict)
    head_token_to_predicate: dict[int, int] = field(default_factory=dict)
    coref_backend: str = "none"


def _is_eventive_noun(lemma: str) -> bool:
    return lemma in _EVENTIVE_NOUNS or any(lemma.endswith(s) for s in _EVENTIVE_NOUN_SUFFIXES)


class Parser:
    """Wraps spaCy (+ optional coref). Constructed once and reused — model load is the expensive part.

    `require_coref=True` turns a missing `fastcoref` into an error instead of a silent downgrade. Use it for
    any run whose result will be reported, because string-match coref and neural coref are not the same
    experiment.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", use_coref: bool = True,
                 require_coref: bool = False):
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "kvgraph.parse needs spaCy:  pip install spacy && python -m spacy download en_core_web_sm"
            ) from exc
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as exc:  # pragma: no cover - environment-dependent
            raise OSError(f"spaCy model {spacy_model!r} not installed: "
                          f"python -m spacy download {spacy_model}") from exc

        self._coref = None
        self.coref_backend = "string_match"
        if use_coref:
            try:
                from fastcoref import FCoref
                self._coref = FCoref(device="cpu")
                self.coref_backend = "fastcoref"
            except Exception as exc:  # noqa: BLE001 - any import/load failure degrades the same way
                if require_coref:
                    raise RuntimeError(
                        "require_coref=True but fastcoref is unavailable: pip install fastcoref"
                    ) from exc

    # ------------------------------------------------------------------ coref

    def _coref_clusters(self, text: str, doc) -> dict[tuple[int, int], int]:
        """-> {(char_start, char_end): cluster_id}. Empty dict means "no cluster", not "no coref"."""
        if self._coref is not None:
            preds = self._coref.predict(texts=[text])[0]
            out = {}
            for cid, cluster in enumerate(preds.get_clusters(as_strings=False)):
                for span in cluster:
                    out[(int(span[0]), int(span[1]))] = cid
            return out
        # Fallback: lemma of the head noun. Groups "the farm"/"a farm", never resolves a pronoun, and will
        # happily fuse two different people named John — which is the OVER-merge error, the one that
        # produces wrong answers rather than missing ones. Hence require_coref for reportable runs.
        by_lemma: dict[str, int] = {}
        out = {}
        for chunk in doc.noun_chunks:
            head = chunk.root
            if head.pos_ == "PRON":
                continue
            key = head.lemma_.lower()
            out[(chunk.start_char, chunk.end_char)] = by_lemma.setdefault(key, len(by_lemma))
        return out

    # ----------------------------------------------------------------- public

    def parse(self, text: str) -> ParseResult:
        doc = self.nlp(text)
        clusters = self._coref_clusters(text, doc)
        res = ParseResult(text=text, coref_backend=self.coref_backend)

        # ── entity mentions: noun chunks (covers "the family farm") + bare pronouns ────────────────
        covered: set[int] = set()
        for chunk in doc.noun_chunks:
            head = chunk.root
            kind = NodeKind.EVENT if _is_eventive_noun(head.lemma_.lower()) else NodeKind.ENTITY
            if head.pos_ == "PRON" and head.lemma_.lower() in _RELATIVIZERS:
                # "the rumour THAT he sold it": spaCy makes the relativizer a second dobj of the verb.
                # As a node it is pure noise that competes with the real theme.
                covered.update(range(chunk.start, chunk.end))
                continue
            m = Mention(char_start=chunk.start_char, char_end=chunk.end_char, text=chunk.text, kind=kind,
                        head_lemma=head.lemma_.lower(), ner_type=head.ent_type_ or None,
                        sent_idx=list(doc.sents).index(chunk.sent) if chunk.sent is not None else -1,
                        cluster_id=clusters.get((chunk.start_char, chunk.end_char)),
                        parser_token=head.i, head_char=(head.idx, head.idx + len(head.text)))
            covered.update(range(chunk.start, chunk.end))
            if kind is NodeKind.EVENT:
                res.head_token_to_predicate[head.i] = len(res.predicates)
                res.predicates.append(m)
            else:
                res.head_token_to_mention[head.i] = len(res.mentions)
                res.mentions.append(m)
        for tok in doc:
            if tok.i in covered or tok.pos_ != "PRON":
                continue
            if tok.lemma_.lower() in _RELATIVIZERS:
                continue
            m = Mention(char_start=tok.idx, char_end=tok.idx + len(tok.text), text=tok.text,
                        kind=NodeKind.ENTITY, head_lemma=tok.lemma_.lower(), sent_idx=-1,
                        cluster_id=clusters.get((tok.idx, tok.idx + len(tok.text))),
                        parser_token=tok.i, head_char=(tok.idx, tok.idx + len(tok.text)))
            res.head_token_to_mention[tok.i] = len(res.mentions)
            res.mentions.append(m)

        # ── predicates: verbs (nominalisations were caught above) ─────────────────────────────────
        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX") or tok.dep_ in ("aux", "auxpass"):
                continue
            m = Mention(char_start=tok.idx, char_end=tok.idx + len(tok.text), text=tok.text,
                        kind=NodeKind.EVENT, head_lemma=tok.lemma_.lower(), sent_idx=-1,
                        parser_token=tok.i, head_char=(tok.idx, tok.idx + len(tok.text)))
            res.head_token_to_predicate[tok.i] = len(res.predicates)
            res.predicates.append(m)

        # ── arcs ──────────────────────────────────────────────────────────────────────────────────
        for tok in doc:
            head = tok.head
            if head is tok:
                continue
            passive = any(c.dep_ == "auxpass" for c in head.children)
            marker = None
            obj_ent = None
            child_token = tok.i

            if tok.dep_ == "prep":
                # Resolve prep -> its object in one arc, so edges.py sees (prep lemma, object NER type).
                objs = [c for c in tok.children if c.dep_ == "pobj"]
                if not objs:
                    continue
                marker, child_token = tok.lemma_.lower(), objs[0].i
                obj_ent = objs[0].ent_type_ or None
            elif tok.dep_ == "advcl":
                marks = [c.lemma_.lower() for c in tok.children if c.dep_ == "mark"]
                marker = marks[0] if marks else None

            res.arcs.append(Arc(head_token=head.i, child_token=child_token, dep=tok.dep_,
                                marker=marker, obj_ent_type=obj_ent, head_is_passive=passive))
        return res
