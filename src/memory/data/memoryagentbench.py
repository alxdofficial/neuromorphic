"""MemoryAgentBench — judge-free co-primary memory benchmark (EVAL-ONLY).

Raw-TEXT accessor for the Phase-2 API baseline runners (mirrors `longmemeval.load_longmemeval_text`).
HF dataset `ai-hyz/MemoryAgentBench` (MIT). Each dataset row = one long `context` + a LIST of questions,
each with a LIST of acceptable paraphrase golds ("inject once, query many"); we expand to one item per
question. The sub-dataset (`metadata.source`) fixes the deterministic metric + competency AND the exact
per-competency prompt (verbatim from the benchmark's `utils/templates.py`, commit 455306d — ICL "output only
the label", factconsolidation "newer fact has larger serial number", etc.). The two LLM-judged subsets
(`longmemeval_*`, `infbench_sum_*`) and `recsys` (recall@5, not deterministically scoreable) are DROPPED here
so no API spend is wasted on unscoreable questions. See `docs/baselines/MEMORYAGENTBENCH_SCHEMA.md`.
"""
from __future__ import annotations

from typing import Optional

_SPLITS = ("Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution")
_SMOKE_TARGET_SOURCES = 12   # a bounded sample aims to span ~this many source variants (per-source cap sizing)

# Shared system prompt + per-sub-dataset query templates, VERBATIM from MemoryAgentBench utils/templates.py
# (long_context_agent variant — the canonical one for a full-context / RAG panel). `{question}` is substituted
# by str.replace in build_messages (NOT .format), so the literal `{label}` in the ICL template survives.
MAB_SYSTEM = "You are a helpful assistant that can read the context and memorize it for future retrieval."
MAB_CONTEXT_HEADER = "# Context"

_QUERY_TEMPLATES = {
    "ruler": ("Answer the question based on the memorized documents. Only give me the answer and do not "
              "output any other words. \n\nQuestion: {question} \n\n Answer:"),
    "eventqa": ("Based on the context you memorized, complete the task below:\n\n{question}\n\n The event "
                "that happens next is:"),
    "icl": ('Use the provided mapping from the context to numerical label to assign a numerical label to the '
            'context. Only output "label: {label}" and nothing else. \n\n{question} \n\n label:'),
    "detective": ("Based on the context you memorized, answer the question below. You are required to answer "
                  "the question based on the strict output format.\n\n {question} \n\n"),
    "factconsolidation": (
        "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a "
        "serial number at the beginning, and the newer fact has larger serial number. \n You need to solve "
        "the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. "
        "You need to answer a question based on this rule. You should give a very concise answer without "
        "saying other words for the question **only** from the knowledge pool you have memorized rather than "
        "the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the "
        "provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump "
        "\n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:"),
}


def _query_template(source: str) -> str:
    s = (source or "").lower()
    for prefix, key in (("ruler", "ruler"), ("eventqa", "eventqa"), ("icl", "icl"),
                        ("detective", "detective"), ("factconsolidation", "factconsolidation")):
        if s.startswith(prefix):
            return _QUERY_TEMPLATES[key]
    return ""   # unknown → build_messages falls back to the default question block


def _metric_competency(source: str) -> tuple[Optional[str], Optional[str]]:
    """Map a sub-dataset source → (deterministic metric, competency). None metric ⇒ drop (judged/unscoreable)."""
    s = (source or "").lower()
    if s.startswith(("longmemeval", "infbench_sum")):
        return None, None                                    # LLM-judged → dropped
    if s.startswith("recsys"):
        return None, None                                    # recall@5 not scored here → DROP (no API waste)
    if s.startswith(("ruler", "eventqa")):
        return "substring_exact_match", "Accurate_Retrieval"
    if s.startswith("factconsolidation"):
        return "substring_exact_match", "Conflict_Resolution"
    if s.startswith("icl"):
        return "exact_match", "Test_Time_Learning"
    if s.startswith("detective"):
        return "exact_match", "Long_Range_Understanding"
    return None, None


def _chunk(context: str, chunk_chars: int = 2000) -> list[str]:
    """Split a long context into ~chunk_chars passages (retrieval units for RAG). Splits on paragraph
    boundaries, and HARD-SPLITS any single paragraph longer than chunk_chars — so a context with no
    newlines (or one giant paragraph) still becomes many chunks instead of one useless giant chunk."""
    paras = [p for p in (context or "").split("\n") if p.strip()]
    pieces: list[str] = []
    for p in paras:
        if len(p) <= chunk_chars:
            pieces.append(p)
        else:
            pieces.extend(p[i:i + chunk_chars] for i in range(0, len(p), chunk_chars))
    chunks, cur, n = [], [], 0
    for p in pieces:
        if n + len(p) > chunk_chars and cur:
            chunks.append("\n".join(cur)); cur, n = [], 0
        cur.append(p); n += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [context or ""]


def _round_robin_stratified(by_comp_source: dict, max_examples: Optional[int]) -> list:
    """Deterministic stratified sample: round-robin across the 4 COMPETENCIES first, then across sources
    within each. A bounded --max-examples must span every competency PRESENT — not just whichever has the
    most distinct `source` strings (Accurate_Retrieval alone has ruler_qa1_*, ruler_qa2_*, eventqa_*, …), or
    a small smoke sample degenerates to '100% Accurate_Retrieval' — the exact first-N failure this prevents,
    one level up. No RNG (round-robin is deterministic)."""
    per_comp: dict = {}
    for comp, srcs in by_comp_source.items():          # interleave sources WITHIN a competency
        buckets = list(srcs.values())
        interleaved, depth = [], 0
        while any(depth < len(b) for b in buckets):
            for b in buckets:
                if depth < len(b):
                    interleaved.append(b[depth])
            depth += 1
        per_comp[comp] = interleaved
    if max_examples is None:
        return [it for lst in per_comp.values() for it in lst]
    out, depth, comps = [], 0, list(per_comp.values())   # then round-robin ACROSS competencies
    while len(out) < max_examples and any(depth < len(c) for c in comps):
        for c in comps:
            if depth < len(c):
                out.append(c[depth])
                if len(out) >= max_examples:
                    break
        depth += 1
    return out


def load_memoryagentbench_text(variant: str = "s", max_examples: Optional[int] = None, *,
                               sources: Optional[list[str]] = None,
                               max_context_chars: Optional[int] = None,
                               splits: tuple[str, ...] = _SPLITS,
                               chunk_chars: int = 2000) -> list[dict]:
    """Return per-question items: {question_id, question, answer (list[str] golds), question_type (=competency),
    competency, source, metric, system, question_template, context_header, full_history (context),
    sessions (chunked context)}.

    `variant` is accepted for signature parity with the LongMemEval loader (ignored here).
    `sources`: keep only these `metadata.source` values (substring match). `max_context_chars`: skip rows
    whose context exceeds this (keeps API cost/time bounded — several sources are multi-M chars).
    RAISES if 0 items load; WARNS (does not silently continue) if a split fails → results would be partial."""
    del variant
    from datasets import load_dataset

    import math
    by_comp_source: dict[str, dict[str, list]] = {}
    skipped_size: dict[str, int] = {}
    loaded_splits: list[str] = []
    failed_splits: dict[str, str] = {}
    # per-SOURCE soft cap when sampling — so a bounded --max-examples spans MANY source variants, not just
    # the first source of each competency (a per-competency cap let ruler_qa1 alone fill Accurate_Retrieval,
    # starving ruler_qa2/eventqa). But when the caller has already NARROWED via `sources`, don't sub-cap (a
    # single-source filter must still fill max_examples, not ceil(max_examples/N)). None ⇒ full load.
    if max_examples is None:
        per_source_cap = None
    elif sources:
        per_source_cap = max_examples                        # user narrowed → let the selected sources fill
    else:
        per_source_cap = max(2, math.ceil(max_examples / _SMOKE_TARGET_SOURCES))

    for split in splits:
        try:
            ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)
        except Exception as e:  # noqa: BLE001 — record the failure; do NOT silently pretend the split is empty
            failed_splits[split] = f"{type(e).__name__}: {e}"
            print(f"[data mab] WARN: split {split} FAILED to load ({type(e).__name__}: {e})")
            continue
        loaded_splits.append(split)
        for ri, row in enumerate(ds):
            meta = row.get("metadata", {}) or {}
            source = str(meta.get("source", ""))
            metric, competency = _metric_competency(source)
            if metric is None:
                continue
            if sources and not any(s in source for s in sources):
                continue
            if per_source_cap is not None and \
                    len(by_comp_source.get(competency, {}).get(source, [])) >= per_source_cap:
                continue                                     # this SOURCE already has enough for the sample
            context = row.get("context", "") or ""
            if max_context_chars is not None and len(context) > max_context_chars:
                skipped_size[source] = skipped_size.get(source, 0) + 1   # don't silently vanish sub-sources
                continue
            questions = row.get("questions", []) or []
            answers = row.get("answers", []) or []
            qids = meta.get("question_ids") or []
            chunks = _chunk(context, chunk_chars)
            qtmpl = _query_template(source)
            src_bucket = by_comp_source.setdefault(competency, {}).setdefault(source, [])
            for qi, q in enumerate(questions):
                gold = answers[qi] if qi < len(answers) else []
                if not isinstance(gold, (list, tuple)):
                    gold = [gold]
                qid = str(qids[qi]) if qi < len(qids) else f"{source}_{ri}_{qi}"
                src_bucket.append({
                    "question_id": qid,
                    "question": str(q),
                    "answer": [str(g) for g in gold],
                    "question_type": competency,       # for the runner's grouping
                    "competency": competency,
                    "source": source,
                    "metric": metric,
                    "system": MAB_SYSTEM,
                    "question_template": qtmpl,
                    "context_header": MAB_CONTEXT_HEADER,
                    "full_history": context,
                    "sessions": chunks,
                })
                if per_source_cap is not None and len(src_bucket) >= per_source_cap:
                    break

    if skipped_size:
        print(f"[data mab] skipped {sum(skipped_size.values())} row(s) over max_context_chars="
              f"{max_context_chars} (entire sub-sources may be absent): {skipped_size}")
    if failed_splits:
        print(f"[data mab] WARN: {len(failed_splits)} split(s) FAILED {list(failed_splits)} — any results are "
              f"PARTIAL (missing whole competencies). Fix connectivity before trusting competency numbers.")

    items = _round_robin_stratified(by_comp_source, max_examples)

    if not items:
        raise RuntimeError(
            "MemoryAgentBench: 0 items loaded. "
            f"splits_ok={loaded_splits} splits_failed={list(failed_splits)} "
            f"sources_filter={sources} max_context_chars={max_context_chars}. "
            "Check HF reachability / that the filters are not too strict.")

    per_comp = {c: sum(len(v) for v in srcs.values()) for c, srcs in by_comp_source.items()}
    print(f"[data mab] loaded {len(items)} items (splits_ok={loaded_splits}); per-competency available={per_comp}")
    return items
