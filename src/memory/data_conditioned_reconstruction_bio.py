"""Biographical conditioned reconstruction — natural-sentence key→value reconstruction.

The natural-language sibling of ``data_conditioned_reconstruction.py`` (which is
random-word MQAR). Here a **key** is a short identifying phrase for a world entity
and a **value** is a fact-dense natural sentence packing several of that entity's
*other* random attributes (see ``conditioned_reconstruction_bio_templates.py``).
The encoder ingests N
``key = value`` lines → memory; the decoder reproduces a queried key's value
sentence verbatim, conditioned on the key. Loss only on the value span.

Emits the exact per-sample dict ``data_qa.collate_qa`` consumes, so the whole
``compute_loss`` path + REAL/SHUF/OFF gate are reused unchanged — only the
key/value *content* differs from ``data_conditioned_reconstruction.py``.

Train/val disjointness: train and val build worlds from different ``world_seed``
values → entirely different entity names/attrs (entity-level firewall).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import IterableDataset, DataLoader

from src.memory.data_qa import collate_qa
from src.memory.conditioned_reconstruction_bio_templates import render_key, render_value

# bio world builder + lexical helper (worldspec files restored from git)
from scripts.data_gen.tasks.biographical.state import build_scenario
from scripts.data_gen.tasks.biographical.pools import year_as_words

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
                 exclude_names: set = None):
        self.tok = tokenizer
        self.context_len = context_len
        self.n_pairs = n_pairs
        self.n_query = n_query
        self.n_facts = n_facts
        self.stream_seed = stream_seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id
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

    def _value_ids_content(self, value: str, name: str, given: str, lead_space: bool):
        """Tokenize a value and mark content=False on the entity-name tokens (which are
        given verbatim in the key → key-derivable dead weight that dilutes the gate, sweep
        bug #2). Uses char offsets so it catches the name wherever the persona places it."""
        s = (" " + value) if lead_space else value
        enc = self.tok(s, add_special_tokens=False, return_offsets_mapping=True)
        spans = []
        for nm in {x for x in (name, given) if x}:
            start = 0
            while True:
                i = s.find(nm, start)
                if i < 0:
                    break
                spans.append((i, i + len(nm)))
                start = i + 1
        def _is_name(a, b):
            return any(not (b <= x or a >= y) for x, y in spans)
        content = [not _is_name(a, b) for (a, b) in enc.offset_mapping]
        return enc.input_ids, content

    def _render_pair(self, ent, rng):
        key, excl = render_key(ent, rng, year_as_words)
        val = render_value(ent, rng, year_as_words, n_facts=self.n_facts, exclude=excl)
        name = ent.attrs.get("name") or ent.attrs.get("title") or ent.key
        given = ent.attrs.get("given_name") or name
        return key, val, name, given

    def _gen(self, rng: random.Random) -> dict:
        # Fill-to-budget: pack (key = value) lines until the next would exceed context_len,
        # so the encoder input is ~context_len (= compression_ratio × the 128-tok memory) and
        # never overflows. n_pairs is the MAX pool; the actual count varies with sentence length.
        pool = rng.sample(self.entities, self.n_pairs)
        keys: List[str] = []
        values: List[str] = []
        names: List[str] = []
        givens: List[str] = []
        cum = 0
        line_lens: List[int] = []
        budget = self.context_len - 8                          # margin for BPE line-join effects
        for e in pool:
            k, v, nm, gv = self._render_pair(e, rng)
            tries = 0
            while k in keys and tries < 5:                     # avoid a bare-name key collision
                k, v, nm, gv = self._render_pair(e, rng)
                tries += 1
            line_len = len(self._ids(f"{k} = {v}\n"))
            if cum + line_len > budget and len(keys) >= max(self.n_query, 1):
                break                                          # budget full → stop packing
            keys.append(k); values.append(v); names.append(nm); givens.append(gv)
            line_lens.append(line_len); cum += line_len

        context_str = "".join(f"{k} = {v}\n" for k, v in zip(keys, values))
        ctx = self._ids(context_str)[: self.context_len]       # clamp (margin makes this a no-op)
        valid = len(ctx)
        ctx = ctx + [self.pad_token_id] * (self.context_len - valid)

        # Only query keys whose full "k = v" line is entirely inside the clamped
        # context. To guarantee n_query the loop may over-pack past budget when few
        # keys exist; those trailing lines can be truncated by the clamp, and
        # querying a truncated value would ask the decoder for absent tokens.
        cum_end, queryable = 0, []
        for i, ll in enumerate(line_lens):
            cum_end += ll
            if cum_end <= valid:
                queryable.append(i)
        if len(queryable) < self.n_query:
            # Unlucky draw: this example's first n_query rendered sentences were long
            # enough that fewer than n_query fully fit the clamped context. Signal the
            # caller to resample rather than crash the whole run; __iter__ distinguishes
            # an occasional bad draw (retry) from a fundamentally-too-small config (raise).
            return None
        qi = rng.sample(queryable, self.n_query)
        question_ids = self._ids(keys[qi[0]])

        answer_ids: List[int] = []
        content: List[bool] = []
        for n, j in enumerate(qi):
            if n == 0:
                v_ids, v_content = self._value_ids_content(values[j], names[j], givens[j], False)
            else:
                cue = self._ids(f" {keys[j]} =")          # inline cue for the next pair (given)
                answer_ids += cue
                content += [False] * len(cue)
                v_ids, v_content = self._value_ids_content(values[j], names[j], givens[j], True)
            answer_ids += v_ids
            content += v_content                          # value facts load-bearing; name tokens excluded

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
                             num_workers: int = 2, relational: bool = False) -> DataLoader:
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
                        relational=relational, exclude_names=exclude)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id))


if __name__ == "__main__":  # smoke: render a few conditioned_reconstruction_bio examples end-to-end
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
