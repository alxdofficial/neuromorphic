"""Reconstruction task — conditioned key→value reconstruction (biographical / MQAR-style).

Task half of the old ``data/bio.py``'s ``_gen``: fill-to-budget pack ``key = value`` lines from a
keyed source, place the queried pair per ``query_lag`` (streaming-window retention), and score loss
only on the un-guessable fact-VALUE spans (not the entity name or template scaffolding). The
decoder reproduces a queried key's value sentence verbatim, conditioned on the key.

Emits the exact per-sample dict ``common.collate_qa`` consumes, so the whole ``compute_loss`` path
+ REAL/SHUF/OFF gate are reused unchanged. Consumes a ``keyed``-kind source (``.sample`` KeyedItems
with ``.key_text / .value_text / .value_subs / .name / .given``).
"""
from __future__ import annotations

from typing import List

import torch

from .base import Task
from ..schedule import EpisodeSpec


class ReconstructionTask(Task):
    accepts = ("keyed",)

    def _value_ids_content(self, tok, value: str, name: str, given: str, lead_space: bool,
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
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)

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

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        context_len = spec.total_len
        n_query = spec.n_queries

        def ids(s: str) -> List[int]:
            return tok(s, add_special_tokens=False).input_ids

        # Fill-to-budget: pack (key = value) lines until the next would exceed context_len,
        # so the encoder input is ~context_len (= compression_ratio × the memory) and never
        # overflows. n_inputs is the MAX pool; the actual count varies with sentence length.
        items = source.sample(rng, spec.n_inputs)
        keys: List[str] = []
        values: List[str] = []
        names: List[str] = []
        givens: List[str] = []
        valsubs: List[List[str]] = []
        cum = 0
        line_lens: List[int] = []
        budget = context_len - 8                                # margin for BPE line-join effects
        for it in items:
            k, v, nm, gv, vs = it.key_text, it.value_text, it.name, it.given, it.value_subs
            line_len = len(ids(f"{k} = {v}\n"))
            if cum + line_len > budget and len(keys) >= max(n_query, 1):
                break                                          # budget full → stop packing
            keys.append(k); values.append(v); names.append(nm); givens.append(gv)
            valsubs.append(vs)
            line_lens.append(line_len); cum += line_len

        context_str = "".join(f"{k} = {v}\n" for k, v in zip(keys, values))
        ctx = ids(context_str)[: context_len]                  # clamp (margin makes this a no-op)
        valid = len(ctx)
        ctx = ctx + [pad_token_id] * (context_len - valid)

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
        if len(queryable) < n_query:
            # Unlucky draw: this example's first n_query rendered sentences were long
            # enough that fewer than n_query fully fit the clamped context. Signal the
            # caller to resample rather than crash the whole run (TaskDataset retries).
            return None
        # STREAMING retention: restrict the query to pairs whose line sits in the target
        # window (evidence placement = the lag axis). query_lag "early" → window 0 (max
        # retention lag); "recent" → last window (recency baseline); "any" → no restriction.
        # Prefer pairs FULLY inside the window (their whole binding is written in that one
        # window); fall back to pairs that start there, then to any queryable, so an unlucky
        # pack never crashes the run. See project_streaming_write.
        qcands = queryable
        if spec.query_lag != "any" and spec.window_size:
            ws = spec.effective_window
            query_window = (0 if spec.query_lag == "early"
                            else -1 if spec.query_lag == "recent"
                            else int(spec.query_lag))
            tgt = query_window if query_window >= 0 else max(0, (valid - 1) // ws)
            win = lambda i: (starts[i] // ws, (starts[i] + line_lens[i] - 1) // ws)
            full = [i for i in queryable if win(i) == (tgt, tgt)]
            starts_in = [i for i in queryable if win(i)[0] == tgt]
            qcands = full or starts_in or queryable
        if len(qcands) >= n_query:
            qi = rng.sample(qcands, n_query)
        else:                                                  # target window under-populated: top up
            extra = [i for i in queryable if i not in qcands]
            qi = qcands + rng.sample(extra, n_query - len(qcands))
        # Condition with the "key =" form the context uses ("k = v\n") so the decoder
        # predicts the value IN-DISTRIBUTION (a bare key makes the model — esp. the
        # full-context ceiling — predict " =" instead of the value: the inverted-band bug).
        question_ids = ids(keys[qi[0]] + " =")

        answer_ids: List[int] = []
        content: List[bool] = []
        for n, j in enumerate(qi):
            if n == 0:
                # space-prefixed value (matches "= v" in the context), like the multi-query cue below.
                v_ids, v_content = self._value_ids_content(tok, values[j], names[j], givens[j], True, valsubs[j])
            else:
                cue = ids(f" {keys[j]} =")                 # inline cue for the next pair (given)
                answer_ids += cue
                content += [False] * len(cue)
                v_ids, v_content = self._value_ids_content(tok, values[j], names[j], givens[j], True, valsubs[j])
            answer_ids += v_ids
            content += v_content                          # loss only on the un-guessable fact-value spans

        return {
            "context_ids": torch.tensor(ctx, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (context_len - valid),
                                         dtype=torch.bool),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_content_mask_list": content,
            "task_family": "conditioned_reconstruction_bio",
            "question_type": "single" if n_query == 1 else f"multi{n_query}",
            "answer_refs": [values[qi[0]]],
        }
