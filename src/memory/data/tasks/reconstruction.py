"""Reconstruction task — conditioned key→value reconstruction (biographical / MQAR-style).

Packs `key = value` facts from a keyed source and asks about particular keys, scoring loss only on
the un-guessable fact-VALUE spans (not the entity name / template scaffolding). Now a thin adapter
over the shared streaming packer (`_pack.pack_streaming_episode`): a KeyedItem becomes a
`Unit(write="key = value", query="key =", answer=value, answer_spans=value_subs,
answer_exclude=(name, given))`, and the packer handles fill-to-budget, `query_lag` placement, the
value-span mask, multi-query cues, and causality. Consumes a `keyed`-kind source.
"""
from __future__ import annotations

from .base import Task
from ._pack import Unit, pack_streaming_episode
from ..schedule import EpisodeSpec


class ReconstructionTask(Task):
    accepts = ("keyed",)

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        lo, hi = getattr(source, "pack_n_queries", (1, max(1, spec.n_queries)))
        nq = rng.randint(lo, hi)                           # per-source addressing pressure (1..max)
        items = source.sample(rng, spec.n_inputs)
        if len(items) < nq:
            return None
        units = [Unit(write=f"{it.key_text} = {it.value_text}\n",
                      query=f"{it.key_text} =",
                      answer=it.value_text,
                      answer_spans=tuple(it.value_subs),
                      answer_exclude=tuple(x for x in (it.name, it.given) if x),
                      refs=(it.value_text,))
                 for it in items]
        return pack_streaming_episode(units[:nq], units[nq:], spec, tok, pad_token_id,
                                      task_family="conditioned_reconstruction_bio", rng=rng)
