"""QA task — story/passage → question → answer, with other packed passages as distractors.

A QAItem becomes a `Unit(write=<facts>, query=<question>, answer=<answer>)`. The task packs `n_queries`
(sampled from the SOURCE's `pack_n_queries`, so item-size drives it) gold facts + other items'
contexts as distractors, placed by `query_lag`, via the shared streaming packer. The
answer-un-guessability filter rejects any distractor whose text contains a query answer, so the answer
lives in exactly one packed context (subsumes bAbI's old entity-disjoint distractor filtering).

`pack_rename` sources (bAbI) take a different fill path: co-pack MANY tiny segments to fill the budget,
each entity-renamed disjoint (so no name collision), and DON'T answer-filter (shared locations aren't
leaks — the unique renamed subject disambiguates). Consumes a `qa`-kind source.
"""
from __future__ import annotations

from .base import Task
from ._pack import Unit, pack_streaming_episode
from ..schedule import EpisodeSpec


class QATask(Task):
    accepts = ("qa",)

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        lo, hi = getattr(source, "pack_n_queries", (1, max(1, spec.n_queries)))
        nq = rng.randint(lo, hi)                             # per-source addressing pressure (1..max)
        rename = getattr(source, "pack_rename", False)

        if rename:
            # bAbI: co-pack many tiny segments to fill budget; rename entities disjoint per segment.
            # ~total_len/75 segments (bAbI stories ~75–90 tok) with a little headroom for the fill loop.
            items = source.sample(rng, max(nq + 1, spec.total_len // 75))
            names, objs = source.rename_pools(rng)
            items = [source.rename(it, names, objs) for it in items]
        else:
            # over-sample a distractor pool scaled with total_len (small items need more candidates).
            items = source.sample(rng, nq + max(spec.n_inputs, spec.total_len // 24))
        if len(items) < nq:
            return None

        def _unit(it, *, gold: bool):
            write = "".join(f.rstrip("\n") + "\n" for f in it.facts)      # one fact per line
            if not gold:
                return Unit(write=write)                                 # distractor: never queried
            refs = [it.answer] + list((it.meta or {}).get("aliases", []))
            return Unit(write=write, query=it.question, answer=it.answer,
                        answer_spans=(), refs=tuple(refs))                # whole (short) answer scored

        query_units = [_unit(it, gold=True) for it in items[:nq]]
        filler_units = [_unit(it, gold=False) for it in items[nq:]]
        # Top up the residual budget with the source's distractor POOL (cross-example noise sentences/
        # paragraphs) AFTER the whole-item distractors — this both uses the pool the sources build and
        # cuts the inherent padding of big-context sources (hotpot/musique fill their ~15% tail). Bounded
        # sample; the packer's fill loop + un-guessability filter apply to these too. (rename sources have
        # no useful pool — their fill is the renamed segments.)
        if not rename:
            pool = source.distractor_pool()
            if pool:
                filler_units += [Unit(write=s.rstrip("\n") + "\n")
                                 for s in rng.sample(pool, min(len(pool), 48))]

        # Un-guessability filter: no distractor may contain a queried answer verbatim. Skipped for
        # rename sources — bAbI answers are shared locations (kitchen), NOT leaks: the unique renamed
        # subject ("where is Ophelia?") disambiguates, so filtering them would needlessly underfill.
        filler_ok = None
        if not rename:
            answers = [u.answer.lower() for u in query_units if u.answer]
            def filler_ok(u):
                w = u.write.lower()
                return not any(a and a in w for a in answers)

        fam = (items[0].meta or {}).get("dataset", "qa")     # per-dataset telemetry label (task_family)
        return pack_streaming_episode(query_units, filler_units, spec, tok, pad_token_id,
                                      task_family=fam, rng=rng, filler_ok=filler_ok)
