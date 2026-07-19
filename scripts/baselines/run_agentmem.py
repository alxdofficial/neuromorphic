#!/usr/bin/env python3
"""Phase-2 2b baseline: agent-memory frameworks (A-MEM / MemoryOS) over LongMemEval — API-based, NO GPU.

The 2b "agent-memory paradigm" reference (docs/baselines/PHASE2_BASELINES.md §2.5): an external memory store
+ retrieval bolted onto a FROZEN chat LLM. Both frameworks run over an OpenAI-compatible endpoint → we point
them at OpenRouter (the same panel as run_api_eval.py), so the reader LLM is shared with Tier-1 and only a
small local sentence-embedder runs (CPU-fine). Per LongMemEval item (private haystack): spin up a FRESH memory
system, ingest each session, then answer the question. Scored with the SAME deterministic scorer.

  --method a-mem     A-MEM (Xu et al., NeurIPS'25; github.com/WujiangXu/A-mem, MIT). `AgenticMemorySystem`:
                     add_note(session) to ingest, find_related_memories(q, k) to retrieve; WE generate the
                     answer from the retrieved context via OpenRouter (its class doesn't do the QA step).
  --method memoryos  MemoryOS (Kang et al., EMNLP'25; github.com/BAI-LAB/MemoryOS, Apache-2.0; pip
                     memoryos-pro). `Memoryos`: add_memory(user,assistant) to ingest, get_response(q) which
                     retrieves AND generates internally (same OpenRouter model).

Add exactly ONE for the paper (A-MEM default). `--help` works without the framework installed (lazy import);
running needs `pip install` of the chosen framework + `OPENROUTER_API_KEY`. RESUMABLE per-item.

Example:  OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --max-examples 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.memory.eval.baselines import build_messages          # noqa: E402  (ours, no heavy deps)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_LLM = "meta-llama/llama-3.1-8b-instruct"              # reader LLM (share with the Tier-1 panel)
_DEFAULT_EMBED = "all-MiniLM-L6-v2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def _openrouter_chat(model: str, messages: list[dict], api_key: str, max_tokens: int) -> tuple[str, str | None]:
    """Minimal synchronous OpenRouter chat (the answer step for A-MEM). Returns (text, error)."""
    import httpx
    try:
        r = httpx.post(f"{_OPENROUTER_BASE}/chat/completions",
                       headers={"Authorization": f"Bearer {api_key}"},
                       json={"model": model, "messages": messages, "max_tokens": max_tokens,
                             "temperature": 0.0}, timeout=120.0)
        if r.status_code != 200:
            return "", f"HTTP {r.status_code}: {r.text[:200]}"
        choice = r.json()["choices"][0]
        return (choice.get("message", {}).get("content") or ""), None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"


def _record(it, hyp="", error=None):
    return {"question_id": it["question_id"], "question": it["question"], "answer": it["answer"],
            "hypothesis": hyp, "question_type": it["question_type"],
            "finish_reason": "error" if error else "stop", "error": error}


def run_a_mem(args, items, api_key, store) -> None:
    """A-MEM: ingest sessions as Zettelkasten notes, retrieve top-k, then WE generate via OpenRouter."""
    from memory_layer import AgenticMemorySystem   # POD/env: pip install the A-mem repo (github.com/WujiangXu/A-mem)

    done = store.done_ids()
    for it in items:
        if str(it["question_id"]) in done:
            continue
        try:
            mem = AgenticMemorySystem(model_name=args.embed_model, llm_backend="openai",
                                      llm_model=args.llm_model, api_key=api_key, api_base=_OPENROUTER_BASE)
            for sess in (it["sessions"] or [it["full_history"]]):
                mem.add_note(sess)
            ctx, _idx = mem.find_related_memories(it["question"], k=args.retrieve_k)
            q = it["question"]
            if it.get("question_date"):
                q = f"Current Date: {it['question_date']}\n{q}"
            msgs, _ = build_messages("full_context", question=q, full_history=ctx or "", char_budget=10 ** 9)
            hyp, err = _openrouter_chat(args.llm_model, msgs, api_key, args.max_new_tokens)
            store.append(_record(it, hyp=hyp, error=err))
        except Exception as e:  # noqa: BLE001
            print(f"[run_agentmem] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(_record(it, error=f"{type(e).__name__}: {e}"))


def run_memoryos(args, items, api_key, store) -> None:
    """MemoryOS: ingest session turn-pairs, then get_response (retrieves + generates internally)."""
    from memoryos import Memoryos   # POD/env: pip install memoryos-pro

    done = store.done_ids()
    for i, it in enumerate(items):
        if str(it["question_id"]) in done:
            continue
        try:
            memo = Memoryos(user_id=f"lme_{it['question_id']}", assistant_id="lme_assistant",
                            openai_api_key=api_key, openai_base_url=_OPENROUTER_BASE,
                            llm_model=args.llm_model, embedding_model_name=args.embed_model,
                            data_storage_path=str(REPO / "outputs" / "baselines" / "memoryos_state" / str(i)))
            for sess in (it["sessions"] or [it["full_history"]]):
                memo.add_memory(user_input=sess, agent_response="")     # session text as the user turn
            q = it["question"]
            if it.get("question_date"):
                q = f"Current Date: {it['question_date']}\n{q}"
            hyp = memo.get_response(query=q)
            store.append(_record(it, hyp=hyp))
        except Exception as e:  # noqa: BLE001
            print(f"[run_agentmem] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(_record(it, error=f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["a-mem", "memoryos"], required=True)
    ap.add_argument("--llm-model", default=_DEFAULT_LLM, help="reader LLM (OpenRouter id; share with Tier-1)")
    ap.add_argument("--embed-model", default=_DEFAULT_EMBED, help="local sentence-embedder (CPU-fine)")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--retrieve-k", type=int, default=10, help="(a-mem) top-k memories to retrieve")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("[run_agentmem] set OPENROUTER_API_KEY (the reader LLM runs over OpenRouter).")

    from src.memory.data.longmemeval import load_longmemeval_text
    from src.memory.eval import score_longmemeval
    from src.memory.eval.results import ResultStore

    print(f"[run_agentmem] method={args.method} llm={args.llm_model} variant={args.variant} "
          f"max_examples={args.max_examples}")
    items = load_longmemeval_text(variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_agentmem] {len(items)} items; types={types}")

    commit = _git_commit()
    tag = (f"longmemeval__{args.method}__{args.llm_model.split('/')[-1]}__{args.variant}"
           f"__n{len(items)}__{commit}")
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_agentmem] resume: {n_done}/{len(items)} already done — answering the rest")

    (run_a_mem if args.method == "a-mem" else run_memoryos)(args, items, api_key, store)

    records = [r for r in store.all_records() if not r.get("error")]
    agg = score_longmemeval(records, use_bem=not args.no_bem)
    store.merge_verdicts(agg.get("details", [])); store.compact()
    n_err = sum(1 for r in store.all_records() if r.get("error"))
    print(f"\n[run_agentmem] overall_acc={agg.get('overall_accuracy', float('nan')):.3f}  "
          f"task_avg={agg.get('task_averaged_accuracy', float('nan')):.3f}  "
          f"abstention={agg.get('abstention_accuracy')}  n={agg.get('n_nonabstention')}  errors={n_err}")

    payload = {
        "dataset": "longmemeval", "method": args.method, "model": args.llm_model,
        "meta": {"n": len(records), "n_errors": n_err, "variant": args.variant, "commit": commit,
                 "coverage": round(len(records) / len(items), 4) if items else None},
        "aggregate": {k: v for k, v in agg.items() if k != "details"},
        "store": str(store.path),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"[run_agentmem] wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
