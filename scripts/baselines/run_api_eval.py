#!/usr/bin/env python3
"""Phase-2 Tier-1 reference baselines over the OpenRouter API (no local GPU).

Runs the reference conditions (floor / full-context / RAG-bm25 / RAG-dense) for a panel of chat models on a
memory benchmark, scores them with the repo's DETERMINISTIC scorer (no LLM judge), and writes per-(model,mode)
JSON + a summary with exact $ cost. `--dry-run` builds every prompt and ESTIMATES cost from char counts
without spending a cent — use it to preview cost and to smoke the pipeline without a live key.

Examples:
  # preview cost + prompt shapes, no API calls, no key needed:
  python scripts/baselines/run_api_eval.py --dataset longmemeval --dry-run
  # real run, all defaults:
  python scripts/baselines/run_api_eval.py --dataset longmemeval --max-examples 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.memory.data.longmemeval import load_longmemeval_text          # noqa: E402
from src.memory.data.memoryagentbench import load_memoryagentbench_text  # noqa: E402
from src.memory.eval import score_longmemeval, score_memoryagentbench  # noqa: E402
from src.memory.eval.api_client import OpenRouterClient, DEFAULT_MODELS, PRICING, cost_usd  # noqa: E402
from src.memory.eval.baselines import build_messages, char_budget_for, MODES, DenseRetriever  # noqa: E402
from src.memory.eval.results import ResultStore, store_path  # noqa: E402

# dataset -> (text-item loader, scorer). Both scorers share the (records, use_bem=) signature.
DATASETS = {
    "longmemeval": (load_longmemeval_text, score_longmemeval),
    "memoryagentbench": (load_memoryagentbench_text, score_memoryagentbench),
}
_BIG_FIELDS = ("full_history", "sessions", "context")   # dropped from scored records (kept only for prompts)
_CHARS_PER_TOKEN = 3.6          # rough, for dry-run cost estimation only
_PROMPT_CODE_VERSION = "2026-07-20"   # bump when build_messages/prompt logic changes; recorded in meta (audit #6)
_SIG_ANCHOR_PROMPT = "2026-07-20"     # sigs at this prompt version OMIT it from the hash (audit #1): the store
#                                     # keys GENERATION-affecting knobs, and the existing caches were generated
#                                     # under this prompt (only SCORING changed, which is re-applied on rescore,
#                                     # not cached) → they stay valid. A LATER prompt/reserve is added to the
#                                     # sig → a distinct store, so a changed generation path never silently reuses.
_RESERVE_TOKENS = 16000         # headroom kept free in the served window: completion (~2k) + system/question/
#                               # template (~1k) + a safety margin for the provider tokenizer counting ~8%
#                               # higher than our reference tokenizer on the largest histories (131k-ctx llama
#                               # still 400'd at 6k/10k reserve; 16k gives budget ~115k, comfortably under 131k).


def model_context_lengths(models: list[str]) -> dict[str, int]:
    """Fetch each model's context window from OpenRouter /models (public metadata; no valid key needed)."""
    import os
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
        # top_provider.context_length = the window actually SERVED by the default provider (can be << the
        # models-list max, e.g. qwen-2.5-7b lists 131k but serves 32k) → budget by this to avoid over-long prompts.
        by_id = {m["id"]: ((m.get("top_provider") or {}).get("context_length")
                           or m.get("context_length") or 131072) for m in data}
    except Exception as e:  # noqa: BLE001 — fall back to a safe default
        print(f"[run_api_eval] WARN: could not fetch model metadata ({e}); assuming 131072 ctx")
        by_id = {}
    return {m: by_id.get(m, 131072) for m in models}


def _est_tokens(messages: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in messages) / _CHARS_PER_TOKEN)


def _config_sig(args, mode: str) -> str:
    """Short hash of the knobs that change what gets generated for THIS mode — so a rerun with different
    settings uses a fresh store rather than stale cached answers. Mode-aware: bm25_topk only affects rag."""
    import hashlib
    parts = [f"mt{args.max_tokens}", f"var{args.variant}"]
    if args.max_context_chars:
        parts.append(f"mcc{args.max_context_chars}")
    if args.sources:
        parts.append("src" + ",".join(sorted(args.sources)))
    if mode.startswith("rag"):
        parts.append(f"k{args.bm25_topk}")
    if mode == "full_context" and args.pin_provider:
        parts.append(f"prov{args.pin_provider}")   # pinned provider can serve subtly different text → own store
    # audit #1: capture GENERATION-affecting policy so a changed run doesn't silently reuse stale answers.
    # Gated to the anchor so existing (still-valid) caches are preserved; any future change adds to the sig.
    if _RESERVE_TOKENS != 16000:
        parts.append(f"rsv{_RESERVE_TOKENS}")
    if _PROMPT_CODE_VERSION != _SIG_ANCHOR_PROMPT:
        parts.append(f"pv{_PROMPT_CODE_VERSION}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:8]


def build_all(items, mode, budget, bm25_topk, dense):
    """Build messages for every item (dry-run only); return (message_lists, n_truncated). Char-budgeted
    (fast, no tokenizer load) — the live path is token-accurate, so dry-run truncation is an estimate."""
    msgs, n_trunc = [], 0
    for it in items:
        m, info = build_messages(mode, question=it["question"], full_history=it.get("full_history", ""),
                                 sessions=it.get("sessions"), char_budget=budget,
                                 bm25_topk=bm25_topk, dense=dense, question_date=it.get("question_date"),
                                 system=it.get("system"), question_template=it.get("question_template"),
                                 context_header=it.get("context_header", "# Conversation history"))
        msgs.append(m)
        n_trunc += int(info["truncated"])
    return msgs, n_trunc


def _valid_for_scoring(r) -> bool:
    """A record counts toward accuracy only if it carries a real, COMPLETE model answer. EXCLUDE API errors
    and ANY answer cut off at the token cap (finish_reason='length' — empty OR partial), PLUS provider
    failures that arrive as a terminal finish_reason (error / content_filter) even when no `error` field was
    recorded (audit #2: OpenRouter can return content_filter with blank/partial content — a refusal, not a
    wrong answer). These are harness artifacts; counting them would deflate accuracy. Surfaced as coverage."""
    if r.get("error"):
        return False
    if r.get("finish_reason") in ("length", "error", "content_filter"):
        return False
    return True


def _proj(r) -> dict:
    """Project a store record into the shape both scorers read via .get() (LongMemEval + MemoryAgentBench)."""
    return {"question": r.get("question"), "answer": r.get("gold"), "hypothesis": r.get("hypothesis"),
            "question_type": r.get("question_type"), "question_id": r.get("question_id"),
            "source": r.get("source"), "metric": r.get("metric"), "competency": r.get("competency")}


async def run_one(client, model, mode, items, token_budget, char_budget, bm25_topk, dense, max_tokens,
                  store, scorer, use_bem, concurrency=32, provider=None, provider_slug=None,
                  warm_by_context=False):
    """RESUMABLE: skip questions already answered in `store`, request only the rest, appending each result
    the moment it returns (crash-safe). Then score ONLY the currently-selected items — NOT the whole store,
    which may hold answers from a larger earlier run (`--max-examples` is a proper scoping knob) — and fold
    verdicts back in. Meta counts (coverage, cost, cutoffs) are likewise over the current selection.

    `provider` pins an OpenRouter backend (dict, e.g. {"order":["deepseek"],"allow_fallbacks":False}) so a
    repeated context keeps hitting ONE instance's prefix cache. `warm_by_context` (full_context only): send
    the FIRST question of each distinct context alone to populate the cache, THEN blast the rest concurrently
    — otherwise the first `concurrency` requests on a cold context all race as cache-misses (paying full
    input price ~85x the cached rate) before the cache is written."""
    want = {str(it["question_id"]) for it in items}
    done = store.done_ids()
    pending = [it for it in items if str(it["question_id"]) not in done]

    async def one(it):
        msgs, info = build_messages(mode, question=it["question"], full_history=it.get("full_history", ""),
                                    sessions=it.get("sessions"), token_budget=token_budget,
                                    char_budget=char_budget,
                                    bm25_topk=bm25_topk, dense=dense, question_date=it.get("question_date"),
                                    system=it.get("system"), question_template=it.get("question_template"),
                                    context_header=it.get("context_header", "# Conversation history"))
        r = await client.chat(model, msgs, max_tokens=max_tokens, provider=provider)
        store.append({
            "question_id": str(it["question_id"]), "question": it["question"], "gold": it["answer"],
            "question_type": it.get("question_type"), "competency": it.get("competency"),
            "source": it.get("source"), "metric": it.get("metric"),
            "mode": mode, "model": model, "hypothesis": r.text, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "cached_tokens": r.cached_tokens,
            "provider": provider_slug, "error": r.error, "finish_reason": r.finish_reason,
            "truncated": info["truncated"], "retrieved_idx": info["retrieved_idx"],
            "correct": None, "score_method": None,
        })

    # WORKER-POOL (producer/consumer), NOT a batch barrier. `concurrency` workers each pull the next pending
    # item from a shared iterator, process it, and append the moment it returns — so a slow (e.g. reasoning-
    # heavy) call ties up ONE worker while the others keep draining the queue. The old code gathered a whole
    # batch and awaited ALL of it before starting the next, so one straggler stalled ~batch finished answers
    # (bursty writes that looked like a hang). In-flight stays bounded at `concurrency`, so prompt RAM
    # residency is still capped (the only reason the old code batched). Mirrors a DataLoader's num_workers.
    # `next(item_iter)` is atomic under the single-threaded event loop (no await between get and use), so no
    # lock is needed across workers.
    async def _drain(item_iter):
        while True:
            try:
                it = next(item_iter)
            except StopIteration:
                return
            await one(it)

    if warm_by_context:
        # group pending by distinct context; warm each context with ONE call (populate the prefix cache),
        # then drain the remainder through the worker pool (~1 fresh + N cached per context).
        groups: dict = {}
        for it in pending:
            groups.setdefault(it.get("full_history", ""), []).append(it)
        for _, gitems in groups.items():
            await one(gitems[0])                                   # warm the prefix cache
            rest_iter = iter(gitems[1:])
            await asyncio.gather(*[_drain(rest_iter) for _ in range(concurrency)])
    else:
        pending_iter = iter(pending)
        await asyncio.gather(*[_drain(pending_iter) for _ in range(concurrency)])

    # score ONLY this run's selection (the cache may carry extra records from a bigger earlier run)
    sel = [r for r in store.all_records() if str(r.get("question_id")) in want]
    agg = scorer([_proj(r) for r in sel if _valid_for_scoring(r)], use_bem=use_bem)
    store.merge_verdicts(agg.get("details", []))
    store.compact()

    pin = sum(r.get("prompt_tokens", 0) or 0 for r in sel)
    pout = sum(r.get("completion_tokens", 0) or 0 for r in sel)
    pcached = sum(r.get("cached_tokens", 0) or 0 for r in sel)
    n_valid = sum(1 for r in sel if _valid_for_scoring(r))
    # n_gen_cutoff = answers cut off at the token cap (empty/partial content) — QC signal to raise --max-tokens.
    n_cutoff = sum(1 for r in sel if r.get("finish_reason") == "length")
    n_err = sum(1 for r in sel if r.get("error"))
    meta = {"n": len(sel), "n_pending_this_run": len(pending),
            # coverage = fraction of selected items that produced a scorable answer. <1.0 ⇒ errors/cutoffs
            # were EXCLUDED from accuracy (not counted wrong); rerun to fill them before trusting the number.
            "n_scored": n_valid, "coverage": round(n_valid / len(sel), 4) if sel else None,
            "n_errors": n_err,
            "n_input_truncated": sum(1 for r in sel if r.get("truncated")),
            "n_gen_cutoff": n_cutoff,
            "prompt_tokens": pin, "completion_tokens": pout, "cached_tokens": pcached,
            "cache_hit_frac": round(pcached / pin, 4) if pin else None, "provider": provider_slug,
            "cost_usd": round(cost_usd(model, pin, pout, pcached, provider_slug), 4)}
    return {k: v for k, v in agg.items() if k != "details"}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="longmemeval")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--variant", default="s")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32,
                    help="concurrent API calls — only affects wall-time, not cost; client retries 429s")
    ap.add_argument("--bm25-topk", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="completion cap — must be HIGH for reasoning models (they spend it on chain-of-thought "
                         "before the answer; too low ⇒ empty content, finish_reason=length, spurious 0 score)")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build prompts + estimate cost; no API calls")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="(memoryagentbench) keep only these metadata.source substrings")
    ap.add_argument("--max-context-chars", type=int, default=None,
                    help="(memoryagentbench) skip rows whose context exceeds this many chars")
    ap.add_argument("--pin-provider", default=None,
                    help="(full_context) pin this OpenRouter provider slug (e.g. 'deepseek') + warm each "
                         "context once so repeated big contexts hit its prefix cache at the cache-read rate")
    ap.add_argument("--revision", default=None,
                    help="(memoryagentbench) pin the HF dataset revision for reproducibility; recorded in meta")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    loader, scorer = DATASETS[args.dataset]
    print(f"[run_api_eval] dataset={args.dataset} variant={args.variant} max_examples={args.max_examples}")
    lkw = dict(variant=args.variant, max_examples=args.max_examples)
    if args.dataset == "memoryagentbench":
        if args.sources:
            lkw["sources"] = args.sources
        if args.max_context_chars:
            lkw["max_context_chars"] = args.max_context_chars
        if args.revision:
            lkw["revision"] = args.revision   # LongMemEval loader has no revision param yet → MAB-only for now
    items = loader(**lkw)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_api_eval] {len(items)} items; types={types}")

    ctx_len = model_context_lengths(args.models)
    unpriced = [m for m in args.models if m not in PRICING]
    if unpriced:
        print(f"[run_api_eval] WARN: no PRICING entry for {unpriced} → their cost read-out will be $0.00 "
              "(add them to api_client.PRICING for an accurate cost).")
    dense = DenseRetriever() if "rag_dense" in args.modes else None
    out_dir = REPO / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("\n=== DRY RUN — projected cost (no API calls) ===")
        grand = 0.0
        for model in args.models:
            budget = char_budget_for(ctx_len[model])
            for mode in args.modes:
                msgs, n_trunc = build_all(items, mode, budget, args.bm25_topk, dense)
                est_in = sum(_est_tokens(m) for m in msgs)
                est = cost_usd(model, est_in, len(items) * 40)   # ~40 completion tok/answer
                grand += est
                print(f"  {model:38} {mode:13} in≈{est_in/1e6:6.2f}M tok  ~${est:6.2f}  "
                      f"trunc={n_trunc}/{len(items)}")
        print(f"\n  PROJECTED TOTAL ≈ ${grand:.2f}  (ctx windows: "
              f"{ {m: ctx_len[m] for m in args.models} })")
        return

    async def run_all():
        async with OpenRouterClient(concurrency=args.concurrency) as client:
            grand = {}
            for model in args.models:
                # token-accurate budget: send everything that fits the SERVED window minus reserve (no
                # char-heuristic over-truncation of histories that actually fit — audit#2 finding 2).
                token_budget = max(0, ctx_len[model] - _RESERVE_TOKENS)
                # per-model CHAR fallback (used only if the reference tokenizer is unavailable) — must be
                # sized to THIS model's window, not the flat 440k default, else small-window models overflow.
                char_budget = char_budget_for(ctx_len[model])
                for mode in args.modes:
                    # rag conditions carry their retrieval budget (k) in the label so a k-sweep (top-5 vs
                    # top-15) produces DISTINCT report columns/files instead of colliding on one "rag_bm25".
                    report_mode = f"{mode}_k{args.bm25_topk}" if mode.startswith("rag") else mode
                    # audit #6: config_sig in the AGGREGATE filename too, so a different provider/budget/etc
                    # doesn't silently OVERWRITE a previous aggregate JSON (the store already keys by it).
                    sig = _config_sig(args, mode)
                    tag = f"{args.dataset}__{model.split('/')[-1]}__{report_mode}__{sig}"
                    store = ResultStore(store_path(out_dir, args.dataset, model, mode, sig))
                    n_done = len(store.done_ids())
                    if n_done:
                        print(f"[resume] {tag}: {n_done}/{len(items)} already answered — requesting the rest")
                    # pin+warm ONLY for full_context (where the same big context is re-sent many times); floor
                    # and rag send tiny/no context, so caching buys nothing and default routing is fine.
                    pin_prov = args.pin_provider if mode == "full_context" else None
                    provider = ({"order": [pin_prov], "allow_fallbacks": False} if pin_prov else None)
                    agg, meta = await run_one(client, model, mode, items, token_budget, char_budget,
                                              args.bm25_topk, dense, args.max_tokens, store, scorer,
                                              not args.no_bem, concurrency=args.concurrency,
                                              provider=provider, provider_slug=pin_prov,
                                              warm_by_context=bool(pin_prov))
                    print(f"\n=== {tag} ===")
                    _sec = (f"abstention={agg.get('abstention_accuracy')}" if "abstention_accuracy" in agg
                            else f"n_scored={agg.get('n_scored')} n_skipped={agg.get('n_skipped')}")
                    print(f"  overall_acc={agg.get('overall_accuracy', float('nan')):.3f}  {_sec}  meta={meta}")
                    cov = meta.get("coverage")
                    if cov is not None and cov < 1.0:
                        print(f"  ⚠ COVERAGE {cov:.1%}: {meta['n'] - meta['n_scored']}/{meta['n']} items NOT "
                              f"scored (errors={meta['n_errors']}, gen_cutoffs={meta['n_gen_cutoff']}) — "
                              f"EXCLUDED from accuracy, not counted wrong. Rerun to fill before trusting it.")
                    # aggregate JSON (per-item records live in the resumable store cache/<tag>.jsonl). `mode`
                    # carries report_mode (audit #7: report.py keys columns by it — else k5/k15 collide).
                    # Record the reserve policy + prompt-code version for traceability (audit #6).
                    meta = {**meta, "config_sig": sig, "reserve_tokens": _RESERVE_TOKENS,
                            "prompt_code_version": _PROMPT_CODE_VERSION, "served_ctx_len": ctx_len[model],
                            "dataset_revision": args.revision, "bem_threshold": (None if args.no_bem else 0.85)}
                    (out_dir / f"{tag}.json").write_text(json.dumps(
                        {"dataset": args.dataset, "model": model, "mode": report_mode,
                         "meta": meta, "aggregate": agg, "store": str(store.path)}, indent=1))
                    grand[tag] = {"aggregate": agg, "meta": meta}
            (out_dir / f"{args.dataset}_api_summary.json").write_text(json.dumps(grand, indent=1))
            total = sum(v["meta"]["cost_usd"] for v in grand.values())
            print(f"\n[run_api_eval] wrote {out_dir}/{args.dataset}_api_summary.json  total_cost=${total:.2f}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
