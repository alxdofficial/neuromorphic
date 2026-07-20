#!/usr/bin/env python3
"""Phase-2 2b baseline: agent-memory frameworks (A-MEM / MemoryOS) over LongMemEval / MemoryAgentBench.

API-based, **NO GPU** — runs on any box (this one): the agent-memory orchestration + a small local
sentence-embedder run on CPU, and the reader LLM runs over OpenRouter (the SAME panel as run_api_eval.py, so
the reader is shared with Tier-1). The 2b "agent-memory paradigm" reference (docs/baselines/PHASE2_BASELINES.md
§2.5): an external memory store + retrieval bolted onto a frozen chat LLM.

  --method a-mem     A-MEM (Xu et al., NeurIPS'25; github.com/WujiangXu/A-mem, MIT). `AgenticMemorySystem`:
                     add_note(text) ingests (LLM-generates keywords/tags + evolves links), find_related_memories
                     (q, k) -> (context_str, indices) retrieves; WE generate the answer via OpenRouter.
  --method memoryos  MemoryOS (Kang et al., EMNLP'25; pip memoryos-pro). `Memoryos`: add_memory then
                     get_response(q) which retrieves AND generates internally. ⚠ UNTESTED here (needs its pip
                     package); get_response may mutate state, so per-context reuse is a POD-VERIFY for it.

PER-CONTEXT REUSE (docs/baselines/TIER2_HOSTING.md, via src/memory/eval/tier2_common.run_grouped): build the
memory store ONCE per distinct context and answer every question sharing it — the local prompt-cache analog.
MAB = 36 ingests for 3,071 Q (retrieval is read-only, safe to reuse); LongMemEval = ingest per question
(unique histories). Same deterministic scorers + JSON shape as Tier-1.

`--help` works without the framework installed (lazy import). Running needs the cloned A-mem repo
(`--repo-dir`, default ~/tier2_repos/A-mem) + `OPENROUTER_API_KEY`. RESUMABLE + crash-safe.

Example:  OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --dataset longmemeval --max-examples 5
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_LLM = "meta-llama/llama-3.1-8b-instruct"              # reader LLM (share with the Tier-1 panel)
_DEFAULT_EMBED = "all-MiniLM-L6-v2"
_DEFAULT_REPO_DIR = str(REPO.parent / "baselines" / "A-mem")  # local master/baselines; git clone github.com/WujiangXu/A-mem


def _openrouter_chat(model: str, messages: list[dict], api_key: str, max_tokens: int):
    """Synchronous OpenRouter chat (the answer step for A-MEM), with the Tier-1 safeguards (audit #7): retry
    on 429/5xx with backoff, treat a choice-level error or terminal finish_reason (error/content_filter) as
    an ERROR not an empty answer, and surface a length cutoff. Returns (text, error, finish_reason)."""
    import time
    import httpx
    last = "exhausted retries"
    for attempt in range(5):
        try:
            r = httpx.post(f"{_OPENROUTER_BASE}/chat/completions",
                           headers={"Authorization": f"Bearer {api_key}"},
                           json={"model": model, "messages": messages, "max_tokens": max_tokens,
                                 "temperature": 0.0}, timeout=120.0)
        except Exception as e:  # noqa: BLE001 — pre-response: retryable
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:120]}"
            elif r.status_code != 200:
                return "", f"HTTP {r.status_code}: {r.text[:200]}", None   # permanent client error, no retry
            else:
                d = r.json()
                if d.get("error") or not d.get("choices"):
                    return "", str(d.get("error") or "200 with no choices"), None
                choice = d["choices"][0]
                fr = choice.get("finish_reason")
                if choice.get("error") or fr in ("error", "content_filter"):
                    return "", str(choice.get("error") or f"terminal finish_reason={fr}"), fr
                return ((choice.get("message") or {}).get("content") or ""), None, fr
        if attempt < 4:
            time.sleep(2 ** attempt)
    return "", last, None


def _quiet(verbose: bool):
    """Suppress A-MEM's very chatty internal print()s (per-note evolution JSONs) unless --verbose — a 500-item
    run prints GBs of them otherwise. Our own run_grouped progress lines print OUTSIDE this, so stay visible."""
    import contextlib
    import os
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def run_a_mem(args, items, api_key, repo_dir, store, dataset, meta_out) -> None:
    """A-MEM with per-context reuse: ingest a context's sessions ONCE (Zettelkasten notes with link-evolution),
    then per question retrieve top-k and generate via OpenRouter. Retrieval is read-only → safe to reuse."""
    sys.path.insert(0, repo_dir)
    from memory_layer import AgenticMemorySystem
    from src.memory.eval.tier2_common import format_query, group_by_context, run_grouped
    from src.memory.eval.baselines import build_messages

    def _ingest_units(ctx, first_item):
        """What to ingest as A-MEM notes. LongMemEval → its natural dated sessions. MAB → the DOCUMENT context
        re-aggregated into `--ingest-chunk-chars` notes: A-MEM runs 1-3 LLM calls PER note (keyword/tag + link
        evolution), and MAB's 2k-char RAG chunks make ~600 notes/context (985k-char AR) → ~10 hr of ingest.
        A-MEM was designed for dozens of sessions, not hundreds of doc chunks; a coarser granularity is both
        more faithful to that design and tractable. DEVIATION to disclose in the paper."""
        if dataset == "longmemeval":
            return first_item.get("sessions") or [ctx]
        from src.memory.data.memoryagentbench import _chunk
        return _chunk(ctx, chunk_chars=args.ingest_chunk_chars)

    def encode_ctx(ctx, first_item):
        mem = AgenticMemorySystem(model_name=args.embed_model, llm_backend="openai",
                                  llm_model=args.llm_model, api_key=api_key, api_base=_OPENROUTER_BASE)
        # Pass each LongMemEval session's REAL date ("[Session X — DATE]") so temporal reasoning isn't broken
        # by wall-clock defaults; MAB chunks carry no such marker → default time.
        with _quiet(args.verbose):
            for sess in _ingest_units(ctx, first_item):
                mt = re.search(r"—\s*([^\]]+)\]", sess)
                mem.add_note(sess, time=mt.group(1).strip()) if mt else mem.add_note(sess)
        return mem

    def answer(mem, it):
        # retrieve with the RAW question (semantic embedding match); generate with the possibly-templated query
        # (MAB competency instruction / LongMemEval date anchor).
        with _quiet(args.verbose):
            ctx_str, _idx = mem.find_related_memories(it["question"], k=args.retrieve_k)
        msgs, _ = build_messages("full_context", question=format_query(it, dataset),
                                 full_history=ctx_str or "", char_budget=10 ** 9)
        hyp, err, fr = _openrouter_chat(args.llm_model, msgs, api_key, args.max_new_tokens)
        if err:
            raise RuntimeError(err)                 # let run_grouped record it as an error (retryable)
        return hyp, (fr or "stop")                  # fr='length' → excluded from scoring + retried

    # audit #10: A-MEM runs ~2 LLM calls per note (metadata + link-evolution) — estimate up front so a full
    # run's cost/time is known (it is the SLOWEST baseline). Resume is PER-CONTEXT: a context whose questions
    # are all done is skipped (not re-ingested); a mid-context crash re-ingests that one context on rerun.
    groups = group_by_context(items)
    n_units = sum(len(_ingest_units(ctx, its[0])) for ctx, its in groups.items())
    est_calls = n_units * 2 + len(items)
    meta_out.update({"n_contexts": len(groups), "n_ingest_units": n_units, "est_llm_calls": est_calls})
    print(f"[run_agentmem] A-MEM estimate: {len(groups)} contexts, ~{n_units} ingest notes → ~{est_calls} LLM "
          f"calls (~2/note + {len(items)} answers). Slowest baseline; resume is per-context.", flush=True)

    run_grouped(items, encode_ctx, answer, store, "[run_agentmem a-mem]")


def run_memoryos(args, items, api_key, repo_dir, store, dataset, meta_out) -> None:
    """MemoryOS: ingest, then get_response (retrieves + generates internally). ⚠ UNTESTED here (needs
    memoryos-pro). audit #1: get_response MUTATES memory with every query+answer, so reusing one instance
    across a context's questions would let later questions SEE earlier benchmark Q&A → contamination. We
    therefore build a FRESH instance PER QUESTION (no cross-question reuse), and detect the upstream
    "Error: Could not get response from LLM." string (which it returns instead of raising) → recorded as an
    error, NOT frozen as a valid answer."""
    del meta_out
    from memoryos import Memoryos   # pip install memoryos-pro
    from src.memory.eval.tier2_common import format_query, make_record

    done = store.done_ids()
    for i, it in enumerate(items):
        if str(it["question_id"]) in done:
            continue
        try:
            state_dir = REPO / "outputs" / "baselines" / "memoryos_state" / str(it["question_id"])
            if state_dir.exists():
                shutil.rmtree(state_dir, ignore_errors=True)
            memo = Memoryos(user_id=f"q_{it['question_id']}", assistant_id="assistant",
                            openai_api_key=api_key, openai_base_url=_OPENROUTER_BASE,
                            llm_model=args.llm_model, embedding_model_name=args.embed_model,
                            data_storage_path=str(state_dir))
            for sess in (it.get("sessions") or [it["full_history"]]):
                memo.add_memory(user_input=sess, agent_response="")     # fresh memory, this question only
            hyp = memo.get_response(query=format_query(it, dataset)) or ""
            if hyp.strip().startswith("Error: Could not get response"):
                store.append(make_record(it, error="memoryos: upstream 'Could not get response from LLM'"))
            else:
                store.append(make_record(it, hyp=hyp, finish_reason="stop"))
        except Exception as e:  # noqa: BLE001 — crash-safe per question
            print(f"[run_agentmem memoryos] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(make_record(it, error=f"{type(e).__name__}: {e}"))
        if (i + 1) % 25 == 0:
            print(f"[run_agentmem memoryos] {i + 1}/{len(items)} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["a-mem", "memoryos"], required=True)
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--repo-dir", default=_DEFAULT_REPO_DIR, help="cloned A-mem repo (on sys.path)")
    ap.add_argument("--llm-model", default=_DEFAULT_LLM, help="reader LLM (OpenRouter id; share with Tier-1)")
    ap.add_argument("--embed-model", default=_DEFAULT_EMBED, help="local sentence-embedder (CPU-fine)")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--retrieve-k", type=int, default=10, help="(a-mem) top-k memories to retrieve")
    ap.add_argument("--ingest-chunk-chars", type=int, default=8000,
                    help="(MAB only) coarser note size for ingest — A-MEM runs 1-3 LLM calls/note, so the 2k "
                         "RAG chunks (~600 notes/context) are intractable; 8000 → ~120 notes. LongMemEval "
                         "ignores this (uses its natural sessions).")
    ap.add_argument("--verbose", action="store_true", help="show A-MEM's internal evolution logging (very noisy)")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("[run_agentmem] set OPENROUTER_API_KEY (the reader LLM runs over OpenRouter).")
    # A-MEM's `openai` backend (OpenAIController) does NOT accept api_base → route via the openai SDK's
    # OPENAI_BASE_URL env fallback (set OPENAI_API_KEY too) so its internal metadata/evolution calls hit
    # OpenRouter, not the default OpenAI endpoint (which would 401 the key). MemoryOS forwards its own base_url.
    os.environ["OPENAI_BASE_URL"] = _OPENROUTER_BASE
    os.environ["OPENAI_API_KEY"] = api_key

    repo_dir = str(Path(args.repo_dir).expanduser())
    from src.memory.eval.tier2_common import git_commit, load_items, build_tag, finalize
    from src.memory.eval.results import ResultStore

    print(f"[run_agentmem] method={args.method} dataset={args.dataset} llm={args.llm_model} "
          f"variant={args.variant} max_examples={args.max_examples}")
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_agentmem] {len(items)} items; types={types}")

    commit = git_commit(REPO)
    tag = build_tag(args.dataset, args.method, args.llm_model.split("/")[-1], args.variant, len(items),
                    f"k{args.retrieve_k}", args.max_new_tokens, 0, commit)
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_agentmem] resume: {n_done}/{len(items)} already done — answering the rest")

    meta_out: dict = {}
    runner = run_a_mem if args.method == "a-mem" else run_memoryos
    runner(args, items, api_key, repo_dir, store, args.dataset, meta_out)

    finalize(args.dataset, args.method, args.llm_model, items, store, use_bem=not args.no_bem,
             extra_meta={"variant": args.variant, "retrieve_k": args.retrieve_k,
                         "max_new_tokens": args.max_new_tokens, "commit": commit,
                         "upstream_commit": git_commit(repo_dir), **meta_out},
             out_dir=out_dir, tag=tag, log_prefix="[run_agentmem]")


if __name__ == "__main__":
    main()
