"""Biographical conditioned reconstruction — natural-sentence key→value reconstruction.

The natural-language sibling of ``data_conditioned_reconstruction.py`` (which is
random-word MQAR). Here a **key** is a short identifying phrase for a world entity
and a **value** is a fact-dense natural sentence packing several of that entity's
*other* random attributes (see ``bio_render.py``). The encoder ingests N
``key = value`` lines → memory; the decoder reproduces a queried key's value
sentence verbatim, conditioned on the key. Loss only on the un-guessable
fact-value spans (not the entity name or template scaffolding).

Emits the exact per-sample dict ``data_qa.collate_qa`` consumes, so the whole
``compute_loss`` path + REAL/SHUF/OFF gate are reused unchanged — only the
key/value *content* differs from ``data_conditioned_reconstruction.py``.

Train/val disjointness: train and val build worlds from different ``world_seed``
values → entirely different entity names/attrs (entity-level firewall).

Generator: ``scripts/data_build/generate/bio/`` (``build_scenario``); render
templates in ``bio_render.py``; composite store: ``data/bio/``. See DATASETS.md.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import IterableDataset, DataLoader

from .common import collate_qa
from .bio_render import render_key, render_value

# bio world builder + lexical helper (build/generate layer: scripts/data_build/generate/bio/)
from scripts.data_build.generate.bio.state import build_scenario
from scripts.data_build.generate.bio.pools import year_as_words

# default per-type entity counts (≈410 entities → supports n_pairs up to ~32)
_WORLD = dict(n_people=200, n_public_figures=30, n_orgs=60, n_nations=20,
              n_places=40, n_events=30, n_works=30)


def _canon(ent) -> str:
    """Canonical name used for the train/val firewall."""
    return ent.attrs.get("name") or ent.attrs.get("title") or ent.key


def _train_names(world_seed: int) -> set:
    scen = build_scenario(random.Random(world_seed), 0, **_WORLD)
    return {_canon(e) for e in scen.world.entities.values()}


class ConditionedReconstructionBioDataset(IterableDataset):
    """Infinite stream of biographical conditioned-reconstruction examples from one world."""

    def __init__(self, tokenizer, context_len: int, n_pairs: int = 16,
                 n_query: int = 1, n_facts: int = 3, world_seed: int = 0,
                 stream_seed: int = 0, n_items: int = 1_000_000,
                 pad_token_id: int = 128_001, relational: bool = False,
                 exclude_names: set = None, window_size: int = None,
                 query_window: int = None):
        self.tok = tokenizer
        self.context_len = context_len
        self.n_pairs = n_pairs
        self.n_query = n_query
        self.n_facts = n_facts
        self.stream_seed = stream_seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id
        # STREAMING-WRITE retention control: with the encoder chunking the context into
        # `window_size` windows, `query_window` pins the queried key→value pair into a chosen
        # window so the other pairs act as distractors between it and the end-of-context query.
        # query_window=0 → evidence in the FIRST window (max retention lag); -1 → last window
        # (recency baseline); None → any window (current behaviour). See project_streaming_write.
        self.window_size = window_size
        self.query_window = query_window
        assert 1 <= n_query <= n_pairs

        scen = build_scenario(random.Random(world_seed), 0, **_WORLD)
        ents = list(scen.world.entities.values())
        # Real train/val firewall: drop val entities whose canonical name collides with
        # a train name (seed-only separation is NOT disjoint — same finite pools). Because
        # every value sentence contains the entity name, canonical-name disjointness also
        # makes the rendered values disjoint, killing the cross-split leak (sweep bug #1).
        if exclude_names:
            ents = [e for e in ents if _canon(e) not in exclude_names]
        self.entities = ents
        if len(self.entities) < n_pairs:
            raise ValueError(f"world has {len(self.entities)} entities < n_pairs={n_pairs} "
                             f"(after firewall drop of {len(exclude_names or ())} train names)")
        print(f"[conditioned_reconstruction_bio] world_seed={world_seed}: {len(self.entities)} entities "
              f"(firewall dropped {len(scen.world.entities) - len(self.entities)}); "
              f"n_pairs={n_pairs} n_query={n_query} n_facts={n_facts} "
              f"context_len={context_len}", flush=True)

    def _ids(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def _value_ids_content(self, value: str, name: str, given: str, lead_space: bool,
                           value_subs=()):
        """Tokenize a value and mark content=True on ONLY the fact-VALUE character spans
        (the substrings render_value actually packed via ``value_out``), excluding the entity
        name and all template/persona scaffolding.

        Scoring loss on the whole sentence charged ~90% of the CE mass to key-independent
        connectives ('recognized for', 'a graduate of', 'In brief —') → SHUF−REAL≈0. Restricting
        to the un-guessable fact spans concentrates the objective where a wrong-entity memory
        actually fails (value-span mask). Falls back to the old name-excluded mask when no value
        spans are given/matched, so a name-only value never yields an all-False (loss-less) mask."""
        s = (" " + value) if lead_space else value
        enc = self.tok(s, add_special_tokens=False, return_offsets_mapping=True)

        def _spans_of(needles):
            out = []
            for nd in {x for x in needles if x}:
                start = 0
                while True:
                    i = s.find(nd, start)
                    if i < 0:
                        break
                    out.append((i, i + len(nd)))
                    start = i + 1
            return out

        name_spans = _spans_of((name, given))
        val_spans = _spans_of(value_subs)

        def _overlaps(spans, a, b):
            return any(not (b <= x or a >= y) for x, y in spans)

        if val_spans:
            content = [_overlaps(val_spans, a, b) and not _overlaps(name_spans, a, b)
                       for (a, b) in enc.offset_mapping]
            if any(content):
                return enc.input_ids, content
        # no value spans matched (name-only value / degenerate draw) → old name-excluded mask.
        content = [not _overlaps(name_spans, a, b) for (a, b) in enc.offset_mapping]
        return enc.input_ids, content

    def _render_pair(self, ent, rng):
        key, excl = render_key(ent, rng, year_as_words)
        vsubs: List[str] = []
        val = render_value(ent, rng, year_as_words, n_facts=self.n_facts,
                           exclude=excl, value_out=vsubs)
        name = ent.attrs.get("name") or ent.attrs.get("title") or ent.key
        given = ent.attrs.get("given_name") or name
        return key, val, name, given, vsubs

    def _gen(self, rng: random.Random) -> dict:
        # Fill-to-budget: pack (key = value) lines until the next would exceed context_len,
        # so the encoder input is ~context_len (= compression_ratio × the 128-tok memory) and
        # never overflows. n_pairs is the MAX pool; the actual count varies with sentence length.
        pool = rng.sample(self.entities, self.n_pairs)
        keys: List[str] = []
        values: List[str] = []
        names: List[str] = []
        givens: List[str] = []
        valsubs: List[List[str]] = []
        cum = 0
        line_lens: List[int] = []
        budget = self.context_len - 8                          # margin for BPE line-join effects
        for e in pool:
            k, v, nm, gv, vs = self._render_pair(e, rng)
            tries = 0
            while k in keys and tries < 5:                     # avoid a bare-name key collision
                k, v, nm, gv, vs = self._render_pair(e, rng)
                tries += 1
            line_len = len(self._ids(f"{k} = {v}\n"))
            if cum + line_len > budget and len(keys) >= max(self.n_query, 1):
                break                                          # budget full → stop packing
            keys.append(k); values.append(v); names.append(nm); givens.append(gv)
            valsubs.append(vs)
            line_lens.append(line_len); cum += line_len

        context_str = "".join(f"{k} = {v}\n" for k, v in zip(keys, values))
        ctx = self._ids(context_str)[: self.context_len]       # clamp (margin makes this a no-op)
        valid = len(ctx)
        ctx = ctx + [self.pad_token_id] * (self.context_len - valid)

        # Only query keys whose full "k = v" line is entirely inside the clamped
        # context. To guarantee n_query the loop may over-pack past budget when few
        # keys exist; those trailing lines can be truncated by the clamp, and
        # querying a truncated value would ask the decoder for absent tokens.
        cum_end, queryable, starts = 0, [], []
        for i, ll in enumerate(line_lens):
            starts.append(cum_end)                            # token offset where pair i's line begins
            cum_end += ll
            if cum_end <= valid:
                queryable.append(i)
        if len(queryable) < self.n_query:
            # Unlucky draw: this example's first n_query rendered sentences were long
            # enough that fewer than n_query fully fit the clamped context. Signal the
            # caller to resample rather than crash the whole run; __iter__ distinguishes
            # an occasional bad draw (retry) from a fundamentally-too-small config (raise).
            return None
        # STREAMING retention: restrict the query to pairs whose line sits in the target
        # window (evidence placement = the lag axis). Prefer pairs FULLY inside the window
        # (their whole binding is written in that one window); fall back to pairs that start
        # there, then to any queryable, so an unlucky pack never crashes the run.
        qcands = queryable
        if self.query_window is not None and self.window_size:
            ws = self.window_size
            tgt = self.query_window if self.query_window >= 0 else max(0, (valid - 1) // ws)
            win = lambda i: (starts[i] // ws, (starts[i] + line_lens[i] - 1) // ws)
            full = [i for i in queryable if win(i) == (tgt, tgt)]
            starts_in = [i for i in queryable if win(i)[0] == tgt]
            qcands = full or starts_in or queryable
        if len(qcands) >= self.n_query:
            qi = rng.sample(qcands, self.n_query)
        else:                                                  # target window under-populated: top up
            extra = [i for i in queryable if i not in qcands]
            qi = qcands + rng.sample(extra, self.n_query - len(qcands))
        # Condition with the "key =" form the context uses ("k = v\n") so the decoder
        # predicts the value IN-DISTRIBUTION (a bare key makes the model — esp. the
        # full-context ceiling — predict " =" instead of the value: the inverted-band bug).
        question_ids = self._ids(keys[qi[0]] + " =")

        answer_ids: List[int] = []
        content: List[bool] = []
        for n, j in enumerate(qi):
            if n == 0:
                # space-prefixed value (matches "= v" in the context), like the multi-query cue below.
                v_ids, v_content = self._value_ids_content(values[j], names[j], givens[j], True, valsubs[j])
            else:
                cue = self._ids(f" {keys[j]} =")          # inline cue for the next pair (given)
                answer_ids += cue
                content += [False] * len(cue)
                v_ids, v_content = self._value_ids_content(values[j], names[j], givens[j], True, valsubs[j])
            answer_ids += v_ids
            content += v_content                          # loss only on the un-guessable fact-value spans

        return {
            "context_ids": torch.tensor(ctx, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (self.context_len - valid),
                                         dtype=torch.bool),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_content_mask_list": content,
            "task_family": "conditioned_reconstruction_bio",
            "question_type": "single" if self.n_query == 1 else f"multi{self.n_query}",
            "answer_refs": [values[qi[0]]],
        }

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.stream_seed + (wi.id if wi is not None else 0))
        _MAX_RETRY = 50                       # tolerate unlucky long-sentence draws
        for _ in range(self.n_items):
            ex = None
            for _try in range(_MAX_RETRY):
                ex = self._gen(rng)
                if ex is not None:
                    break
            if ex is None:
                # Every one of _MAX_RETRY draws under-filled → the config is genuinely
                # too small for n_query, not just an unlucky sentence-length draw.
                raise ValueError(
                    f"context_len={self.context_len} too small for n_query={self.n_query}: "
                    f"{_MAX_RETRY} consecutive draws had fewer than n_query facts fully fit "
                    f"the context. Raise --cond-recon-bio context length or lower n_query.")
            yield ex


def make_conditioned_reconstruction_bio_dataloader(tokenizer, context_len: int, batch_size: int,
                             n_pairs: int = 16, n_query: int = 1, n_facts: int = 3,
                             split: str = "train", world_seed: int = 0,
                             stream_seed: int = 0, pad_token_id: int = None,
                             num_workers: int = 2, relational: bool = False,
                             window_size: int = None, query_window: int = None) -> DataLoader:
    if pad_token_id is None:                                  # LLM-agnostic default
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # val builds a DIFFERENT world AND drops any entity whose canonical name also exists
    # in the train world — a REAL name-disjoint firewall (the seed offset alone is NOT
    # disjoint; both worlds draw from the same finite pools). sweep bug #1.
    ws = world_seed if split == "train" else world_seed + 10_000
    exclude = None if split == "train" else _train_names(world_seed)
    ds = ConditionedReconstructionBioDataset(tokenizer, context_len=context_len, n_pairs=n_pairs,
                        n_query=n_query, n_facts=n_facts, world_seed=ws,
                        stream_seed=stream_seed, pad_token_id=pad_token_id,
                        relational=relational, exclude_names=exclude,
                        window_size=window_size, query_window=query_window)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id))


if __name__ == "__main__":  # smoke: render a few conditioned_reconstruction_bio examples end-to-end
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from transformers import AutoTokenizer
    from src.memory.config import ReprConfig

    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    for nq in (1, 3):
        ds = ConditionedReconstructionBioDataset(tok, context_len=2048, n_pairs=12, n_query=nq, n_facts=3,
                            world_seed=0, stream_seed=1)
        s = next(iter(ds))
        ctx = tok.decode([t for t in s["context_ids"].tolist() if t != ds.pad_token_id])
        val_toks = [tok.decode([a]) for a in s["answer_ids"].tolist()]
        cm = s["answer_content_mask_list"]
        scored = tok.decode([a for a, m in zip(s["answer_ids"].tolist(), cm) if m])
        print(f"\n===== n_query={nq} =====")
        print("CONTEXT (key = value lines, valid only):")
        print("  " + ctx.replace("\n", "\n  "))
        print(f"QUESTION (queried key): {tok.decode(s['question_ids'].tolist())!r}")
        print(f"ANSWER   (full)       : {tok.decode(s['answer_ids'].tolist())!r}")
        print(f"SCORED   (content=True): {scored!r}")
        print(f"ctx_tokens={int(s['context_mask'].sum())}  answer_tokens={len(s['answer_ids'])}  "
              f"scored={sum(cm)}")
