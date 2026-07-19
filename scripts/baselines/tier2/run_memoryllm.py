#!/usr/bin/env python3
"""Phase-2 Tier-2 GPU baseline: MemoryLLM / M+ (parametric memory) over LongMemEval (POD-ONLY).

MemoryLLM (Wang et al., ICML'24) injects context into a fixed-size in-weights memory pool via
`inject_memory(ids, update_memory=True)`, then answers with a plain `generate()` — the memory is
already fused into hidden states, no prompt-side context at query time. M+ (Wang et al. 2025) adds a
CPU-offloaded long-term memory (LTM) store on top.

Base model = **M+ (`YuWangX/mplus-8b`), NOT MemoryLLM-8B** — MemoryLLM-8B's useful retention caps at
~20k tokens (random Ebbinghaus-style eviction of its 12,800 mem-tok/layer pool), which cannot hold
LongMemEval-S's ~115k-token haystack; M+'s LTM (153,600 tok, co-trained retriever) can. See
`docs/baselines/TIER2_GPU_INTEGRATION.md` #3 for the full writeup, VRAM (~16GB weights + ~3.3GB pool,
plausibly fits a 24GB card but unverified — authors used an H100-80GB) and pod plan.

No official LongMemEval runner exists for this repo — this file hand-writes the write-then-query loop:
chunk each item's full_history into its sessions -> `inject_memory` each session (merging any session
under MemoryLLM's >16-token hard minimum) -> `generate()` for the question. Scores with the SAME
deterministic scorer as Tier-1 (`src/memory/eval/score_longmemeval`). RESUMABLE + crash-safe: each
answer is appended to a per-run JSONL store immediately.

STATE RESET: each LongMemEval question has its OWN private haystack, so per-item memory state must NOT
leak. Both the short-term memory pool (`model.memory`) AND the M+ long-term store (LTM contents/keys/
ages/frequencies/dropped-caches/update_step) mutate under `inject_memory`. We snapshot ALL of them once
after load and restore before every item. See `_snapshot_memory_state` — it PRINTS exactly which buffers
it captured and HARD-WARNS if the LTM store was not found, so state leakage cannot pass silently.

`--help` works without torch/transformers/the MemoryLLM repo installed — every heavy import is lazy,
inside `main()` (or a helper called from it), so plain argparse always succeeds.

Example (on the pod, after `scripts/baselines/tier2/README.md`'s setup):
  python scripts/baselines/tier2/run_memoryllm.py --max-examples 5     # smoke test
  python scripts/baselines/tier2/run_memoryllm.py --max-examples 500
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_DEFAULT_MODEL = "YuWangX/mplus-8b"          # M+ — the only checkpoint that can hold the full ~115k history
_DEFAULT_REPO_DIR = "~/tier2_repos/MemoryLLM"  # `git clone git@github.com:wangyu-ustc/MemoryLLM.git`

# Candidate attribute names for M+'s long-term-memory state (short-term `model.memory` is handled separately).
# Bounded/explicit — NOT a dir() scan — so we never accidentally snapshot the 8B weight tensors. If the real
# names differ, `_snapshot_memory_state` warns loudly listing what it DID find, prompting a modeling_mplus.py
# check. Cross-reference the auditor's list: LTM contents, keys, ages, retrieval frequencies, dropped caches.
_LTM_CANDIDATE_ATTRS = (
    "ltm", "long_term_memory", "ltm_memory", "ltm_keys", "ltm_values", "ltm_ages", "ltm_age",
    "memory_ages", "retrieval_frequencies", "retrieval_freq", "dropped_memory", "dropped_memories",
    "update_step", "ltm_numpy", "ltm_key", "ltm_value",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def _seed_everything(seed: int) -> None:
    """M+ randomly DROPS memory during injection — seed py/np/torch/cuda so a rerun reproduces the same result."""
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _copy_buf(v):
    import copy
    import numpy as np
    import torch
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, (list, tuple)):
        return type(v)(_copy_buf(x) for x in v)
    if isinstance(v, dict):
        return {k: _copy_buf(x) for k, x in v.items()}
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return copy.deepcopy(v)


def _is_nonempty(v) -> bool:
    """True if v holds real state (not None / not a 0-length tensor/array/container). A candidate attr that
    is empty at snapshot time (e.g. LTM lazily populated on first inject) preserves NOTHING useful, so it
    must NOT count as 'captured' — otherwise the leak warning is silenced while state still leaks."""
    if v is None:
        return False
    n = getattr(v, "numel", None)
    if callable(n):
        return v.numel() > 0
    sz = getattr(v, "size", None)
    if isinstance(sz, int):
        return sz > 0
    try:
        return len(v) > 0
    except TypeError:
        return True          # scalar-like (e.g. update_step int) → treat as real state


def _snapshot_memory_state(model):
    """Deep-snapshot the short-term pool + all LTM buffers found. Returns (snapshot dict, captured names)."""
    snap = {"model.memory": model.memory.data.clone()}
    captured, empty = [], []
    for attr in _LTM_CANDIDATE_ATTRS:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                snap[attr] = _copy_buf(val)
                (captured if _is_nonempty(val) else empty).append(attr)
            except Exception as e:  # noqa: BLE001
                print(f"[run_memoryllm] WARN: could not snapshot model.{attr}: {type(e).__name__}: {e}")
    if not captured:
        print("[run_memoryllm] ⚠ HARD WARNING: no NON-EMPTY M+ long-term-memory buffer was found among "
              f"{_LTM_CANDIDATE_ATTRS} (empty-at-load attrs seen: {empty or 'none'}). Only model.memory "
              "(short-term pool) is guaranteed reset between items — the LTM store may LEAK across "
              "LongMemEval questions and CORRUPT results (an attr that is empty at load but fills during "
              "inject_memory would NOT be reset to its filled-then-cleared state). Inspect modeling_mplus.py "
              "on the pod, add/confirm the real LTM attribute name(s) in _LTM_CANDIDATE_ATTRS, and re-run "
              "BEFORE trusting any number. (See docs/baselines/TIER2_GPU_INTEGRATION.md #3.)")
    else:
        print(f"[run_memoryllm] per-item reset will restore: model.memory + LTM buffers {captured}"
              + (f" (also snapshotting empty-at-load: {empty})" if empty else ""))
    return snap, captured


def _restore_memory_state(model, snap) -> None:
    import torch
    model.memory.data.copy_(snap["model.memory"])
    for attr, saved in snap.items():
        if attr == "model.memory":
            continue
        cur = getattr(model, attr, None)
        if isinstance(cur, torch.Tensor) and isinstance(saved, torch.Tensor):
            cur.data.copy_(saved)            # in-place for parameters/buffers
        else:
            setattr(model, attr, _copy_buf(saved))


def _chunk_min_tokens(tok, sessions: list[str], min_tokens: int) -> list[str]:
    """Merge consecutive rendered sessions so every chunk we hand to `inject_memory` clears
    MemoryLLM's hard >16-token minimum (injecting fewer "disturbs" the memory, per the model
    README). Order-preserving forward merge; any short tail is folded into the previous chunk."""
    chunks: list[str] = []
    buf = ""
    for s in sessions:
        buf = f"{buf}\n{s}" if buf else s
        if len(tok(buf, add_special_tokens=False)["input_ids"]) >= min_tokens:
            chunks.append(buf)
            buf = ""
    if buf:
        if chunks and len(tok(buf, add_special_tokens=False)["input_ids"]) < min_tokens:
            chunks[-1] = f"{chunks[-1]}\n{buf}"
        else:
            chunks.append(buf)
    return chunks


def _record(it, hyp="", error=None, finish_reason=None):
    return {"question_id": it["question_id"], "question": it["question"], "answer": it["answer"],
            "hypothesis": hyp, "question_type": it["question_type"],
            "finish_reason": finish_reason or ("error" if error else "stop"), "error": error}


def _finish_reason(gen_ids, max_new_tokens, eos_id) -> str:
    """'length' if generation hit the cap without EOS (answer may be truncated → retryable), else 'stop'."""
    n = gen_ids.shape[0] if hasattr(gen_ids, "shape") else len(gen_ids)
    if n >= max_new_tokens and (eos_id is None or int(gen_ids[-1]) != eos_id):
        return "length"
    return "stop"


def run_memoryllm(args, items, model_name: str, repo_dir: str, store) -> None:
    """Write-then-query loop. Exact call sequence per docs/baselines/TIER2_GPU_INTEGRATION.md #3
    (cross-checked against the MemoryLLM repo README's "How to use the model" section). Appends per item."""
    sys.path.insert(0, repo_dir)
    import torch
    # POD-ONLY: requires MemoryLLM cloned to `repo_dir` (see README.md) — `modeling_mplus.py` is a repo
    # file (not a pip package); the class name is `MPlus` (MemoryLLM-8B/-chat would be `MemoryLLM` from
    # `modeling_memoryllm.py` instead — swap the import + from_pretrained call if benchmarking those).
    from modeling_mplus import MPlus
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = MPlus.from_pretrained(model_name, attn_implementation="flash_attention_2",
                                  torch_dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)  # re-cast per README: from_pretrained alone leaves rotary_emb.inv_freq fp32
    model.put_ltm_to_numpy()          # move LTM off-GPU to a numpy store for inference (README-mandated)
    model = model.cuda()
    model.eval()

    # snapshot the pristine post-load state (short-term pool + LTM buffers); restore before EVERY item.
    snap, _ = _snapshot_memory_state(model)

    done = store.done_ids()
    for it in items:
        if str(it["question_id"]) in done:
            continue
        try:
            _restore_memory_state(model, snap)          # reset before this item's private haystack

            sessions = it["sessions"] or [it["full_history"]]
            chunks = _chunk_min_tokens(tok, sessions, min_tokens=args.min_inject_tokens)
            for chunk_text in chunks:
                ctx_ids = tok(chunk_text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                model.inject_memory(ctx_ids, update_memory=True)

            # Pretrained-model prompt template (mplus-8b currently ships pretrained-only, per README "we
            # only have the pretrained version"). Anchor temporal questions to their date, like Tier-1.
            q = it["question"]
            if it.get("question_date"):
                q = f"Current Date: {it['question_date']} {q}"
            q_ids = tok(f"Question: {q} Answer:", return_tensors="pt",
                        add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                out = model.generate(input_ids=q_ids, max_new_tokens=args.max_new_tokens)
            gen = out[0][q_ids.shape[1]:]
            hyp = tok.decode(gen, skip_special_tokens=True)
            store.append(_record(it, hyp=hyp,
                                  finish_reason=_finish_reason(gen, args.max_new_tokens, tok.eos_token_id)))
        except Exception as e:  # noqa: BLE001 — crash-safe: record, continue, resume later
            print(f"[run_memoryllm] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(_record(it, error=f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=_DEFAULT_MODEL,
                    help="HF repo id. Default = M+ (only checkpoint that fits the full ~115k history); "
                         "MemoryLLM-8B/-chat cap at ~20k tokens (see module docstring).")
    ap.add_argument("--repo-dir", default=_DEFAULT_REPO_DIR, help="path to the cloned MemoryLLM repo")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--min-inject-tokens", type=int, default=17,
                    help="merge sessions until each inject_memory() chunk has >= this many tokens "
                         "(model's hard minimum is 'larger than 16')")
    ap.add_argument("--seed", type=int, default=0, help="seed py/np/torch/cuda (M+ randomly drops memory)")
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (EM+containment only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    repo_dir = str(Path(args.repo_dir).expanduser())

    # --- everything below needs torch/transformers/the MemoryLLM repo — lazy on purpose, see docstring ---
    _seed_everything(args.seed)
    from src.memory.data.longmemeval import load_longmemeval_text
    from src.memory.eval import score_longmemeval
    from src.memory.eval.results import ResultStore

    print(f"[run_memoryllm] model={args.model} repo_dir={repo_dir} variant={args.variant} "
          f"max_examples={args.max_examples} seed={args.seed}")
    items = load_longmemeval_text(variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_memoryllm] {len(items)} items; types={types}")

    commit = _git_commit()
    tag = (f"longmemeval__memoryllm__{args.model.split('/')[-1]}__{args.variant}"
           f"__n{len(items)}__g{args.max_new_tokens}__seed{args.seed}__{commit}")
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_memoryllm] resume: {n_done}/{len(items)} already done — generating the rest")

    run_memoryllm(args, items, args.model, repo_dir, store)

    records = [r for r in store.all_records() if not r.get("error")]
    agg = score_longmemeval(records, use_bem=not args.no_bem)
    store.merge_verdicts(agg.get("details", [])); store.compact()
    n_err = sum(1 for r in store.all_records() if r.get("error"))
    print(f"\n[run_memoryllm] overall_acc={agg.get('overall_accuracy', float('nan')):.3f}  "
          f"task_avg={agg.get('task_averaged_accuracy', float('nan')):.3f}  "
          f"abstention={agg.get('abstention_accuracy')}  n={agg.get('n_nonabstention')}  errors={n_err}")

    payload = {
        "dataset": "longmemeval", "method": "memoryllm", "model": args.model,
        "meta": {"n": len(records), "n_errors": n_err, "variant": args.variant, "seed": args.seed,
                 "min_inject_tokens": args.min_inject_tokens, "max_new_tokens": args.max_new_tokens,
                 "commit": commit, "coverage": round(len(records) / len(items), 4) if items else None},
        "aggregate": {k: v for k, v in agg.items() if k != "details"},
        "store": str(store.path),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"[run_memoryllm] wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
