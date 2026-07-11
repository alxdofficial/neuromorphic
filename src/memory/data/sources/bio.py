"""Biographical source — a world of entities rendered as key→value KeyedItems.

Source half of the old ``data/bio.py``: builds one procedural world (``build_scenario`` over the
``_WORLD`` per-type counts), enforces the train/val entity firewall, and yields ``KeyedItem``s
(a short identifying KEY phrase + a fact-dense natural-sentence VALUE, plus the load-bearing
fact-value substrings for the loss mask). It does NO packing / windowing / query placement — that
is the ReconstructionTask's job.

A **key** is a short identifying phrase for a world entity; a **value** is a fact-dense natural
sentence packing several of that entity's *other* random attributes (see ``bio_render.py``). The
key's disambiguator attribute is excluded from the value, so a wrong-entity (shuffled) memory
cannot reconstruct the sentence.

Train/val disjointness: train and val build worlds from different ``world_seed`` values AND val
drops any entity whose canonical name collides with a train name (entity-level firewall).

Generator: ``scripts/data_build/generate/bio/`` (``build_scenario``); render templates in
``bio_render.py``; composite store: ``data/bio/``. See DATASETS.md / docs/history/docs/history/data_arch_plan.md.
"""
from __future__ import annotations

import random
from typing import List

from .base import Source, KeyedItem
from ..bio_render import render_key, render_value

# bio world builder + lexical helper (build/generate layer: scripts/data_build/generate/bio/)
from scripts.data_build.generate.bio.state import build_scenario
from scripts.data_build.generate.bio.pools import year_as_words

# default per-type entity counts (≈410 entities → supports n_inputs up to ~32)
_WORLD = dict(n_people=200, n_public_figures=30, n_orgs=60, n_nations=20,
              n_places=40, n_events=30, n_works=30)


def _canon(ent) -> str:
    """Canonical name used for the train/val firewall."""
    return ent.attrs.get("name") or ent.attrs.get("title") or ent.key


def _train_names(world_seed: int) -> set:
    scen = build_scenario(random.Random(world_seed), 0, **_WORLD)
    return {_canon(e) for e in scen.world.entities.values()}


class BioSource(Source):
    """Yields biographical entities as key→value ``KeyedItem``s from one procedural world."""

    kind = "keyed"
    pack_n_queries = (1, 3)            # tiny unique facts → high addressing pressure (query up to 3 keys)

    def __init__(self, tokenizer, *, split: str = "train", world_seed: int = 0,
                 n_facts: int = 3, seed: int = 0, **kw):
        self.tok = tokenizer
        self.n_facts = n_facts
        self.seed = seed

        # val builds a DIFFERENT world AND drops any entity whose canonical name also exists
        # in the train world — a REAL name-disjoint firewall (the seed offset alone is NOT
        # disjoint; both worlds draw from the same finite pools). sweep bug #1.
        ws = world_seed if split == "train" else world_seed + 10_000
        exclude = None if split == "train" else _train_names(world_seed)

        scen = build_scenario(random.Random(ws), 0, **_WORLD)
        ents = list(scen.world.entities.values())
        # Real train/val firewall: drop val entities whose canonical name collides with
        # a train name (seed-only separation is NOT disjoint — same finite pools). Because
        # every value sentence contains the entity name, canonical-name disjointness also
        # makes the rendered values disjoint, killing the cross-split leak (sweep bug #1).
        if exclude:
            ents = [e for e in ents if _canon(e) not in exclude]
        self.entities = ents
        print(f"[bio] world_seed={ws} (split={split}): {len(self.entities)} entities "
              f"(firewall dropped {len(scen.world.entities) - len(self.entities)}); "
              f"n_facts={n_facts}", flush=True)

    def _render_pair(self, ent, rng):
        key, excl = render_key(ent, rng, year_as_words)
        vsubs: List[str] = []
        val = render_value(ent, rng, year_as_words, n_facts=self.n_facts,
                           exclude=excl, value_out=vsubs)
        name = ent.attrs.get("name") or ent.attrs.get("title") or ent.key
        given = ent.attrs.get("given_name") or name
        return key, val, name, given, vsubs

    def sample(self, rng, n: int) -> list:
        if n > len(self.entities):
            raise ValueError(f"bio world has {len(self.entities)} entities < requested {n} "
                             f"(after firewall drop)")
        # Draw n DISTINCT entities (rng.sample, no replacement) so a batch never re-writes the
        # same entity. For each render key→value; retry on a bare-name key collision within the
        # draw (two disambiguator-free keys would be indistinguishable). Matches bio.py's
        # ``_render_pair`` retry.
        pool = rng.sample(self.entities, n)
        out: List[KeyedItem] = []
        keys: List[str] = []
        for e in pool:
            k, v, nm, gv, vs = self._render_pair(e, rng)
            tries = 0
            while k in keys and tries < 5:                 # avoid a bare-name key collision
                k, v, nm, gv, vs = self._render_pair(e, rng)
                tries += 1
            keys.append(k)
            out.append(KeyedItem(key_text=k, value_text=v, value_subs=vs, name=nm, given=gv))
        return out
