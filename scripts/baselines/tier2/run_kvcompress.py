#!/usr/bin/env python3
"""Phase-2 Tier-2 GPU baseline: KV-cache compression over LongMemEval (POD-ONLY).

Two method families behind one CLI, both compress the KV cache built from the FULL ~115k-token
rendered chat history before answering — the thing Tier-1 (API, prompt-level truncation/RAG) cannot
do, and our matched-decoder model does architecturally instead of via KV eviction:

  --method snapkv | h2o   KVCache-Factory (github.com/Zefan-Cai/KVCache-Factory, MIT). Monkey-patches
                          HF's Llama attention `.forward` to evict KV entries during prefill/decode.
  --method kvzip          KVzip (github.com/snu-mllab/KVzip, MIT, NeurIPS'25). Query-agnostic:
                          prefill + importance-score the full context once, prune, then answer.

Both need real weights + a GPU with the method's custom code on `sys.path` — cannot run via API and
cannot run on this machine. Peak VRAM ~35-45GB (KVCache-Factory) / ~33-38GB (KVzip) because the FULL
115k-token KV cache is materialized before compression (needs a >24GB card, e.g. a 48GB RunPod A6000).
See `docs/baselines/TIER2_GPU_INTEGRATION.md` (source of truth for the exact entry points below) and
`scripts/baselines/tier2/README.md` for pod bring-up (repo clones, pinned deps, launch commands).

Scores with the SAME deterministic scorer as Tier-1 (`src/memory/eval/score_longmemeval`) so numbers
are directly comparable. RESUMABLE + crash-safe: each answer is appended to a per-run JSONL store the
moment it is generated (a late OOM never loses the whole run), and a rerun skips finished questions.

Gotchas baked in below (see the integration doc for the full writeup):
  - KVCache-Factory's 7,500-tok cap lives ONLY in ITS OWN `run_longbench.py`'s `model2maxlen` dict —
    we never import that file; we build the prompt via the model's own chat template so all ~115k tokens
    flow through the monkey-patched attention unmodified.
  - KVCache-Factory supports Llama + Mistral ONLY (no Qwen). Always pass
    `attn_implementation="flash_attention_2"` — eager/sdpa prefill on 115k tokens balloons VRAM further.
  - KVzip's `prefill()` chunks the context internally (16k blocks) so it handles the full 115k without
    a separate truncation step on our side.

`--help` works without torch/transformers/either external repo installed — every heavy import is
lazy, inside `main()` (or a helper called from it), so plain argparse always succeeds.

Examples (on the pod, after `scripts/baselines/tier2/README.md`'s setup):
  python scripts/baselines/tier2/run_kvcompress.py --method snapkv --max-examples 5   # smoke test
  python scripts/baselines/tier2/run_kvcompress.py --method kvzip --ratio 0.3 --max-examples 500
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Default base model per method (KVCache-Factory: Llama/Mistral only; KVzip: Llama3/Qwen2.5/Qwen3/Gemma3).
_DEFAULT_MODEL = {
    "snapkv": "meta-llama/Llama-3.1-8B-Instruct",
    "h2o": "meta-llama/Llama-3.1-8B-Instruct",
    "kvzip": "Qwen/Qwen2.5-7B-Instruct-1M",
}
# Where each method's repo is expected to be cloned on the pod (see README.md); override with --repo-dir.
_DEFAULT_REPO_DIR = {
    "snapkv": "~/tier2_repos/KVCache-Factory",
    "h2o": "~/tier2_repos/KVCache-Factory",
    "kvzip": "~/tier2_repos/KVzip",
}


def _git_commit(path=REPO) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def _seed_everything(seed: int) -> None:
    """Seed py/np/torch/cuda so eviction/generation are reproducible across reruns."""
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _valid(r) -> bool:
    # exclude API/gen errors AND length-truncated answers (a cut-off answer isn't a wrong answer — matches
    # the Tier-1 _valid_for_scoring policy).
    return not r.get("error") and r.get("finish_reason") != "length"


def _record(it, hyp="", error=None, finish_reason=None):
    """Store/score record shape `score_longmemeval` + `ResultStore` both read."""
    return {"question_id": it["question_id"], "question": it["question"], "answer": it["answer"],
            "hypothesis": hyp, "question_type": it["question_type"],
            "finish_reason": finish_reason or ("error" if error else "stop"), "error": error}


def _finish_reason(gen_ids, max_new_tokens, eos_id) -> str:
    """'length' if generation ran to the cap without emitting EOS (answer may be truncated → retryable via
    ResultStore.done_ids), else 'stop'. gen_ids = the NEW tokens only (after the prompt)."""
    n = gen_ids.shape[0] if hasattr(gen_ids, "shape") else len(gen_ids)
    if n >= max_new_tokens and (eos_id is None or int(gen_ids[-1]) != eos_id):
        return "length"
    return "stop"


def run_kvcache_factory(args, items, model_name: str, repo_dir: str, store) -> None:
    """SnapKV / H2O via KVCache-Factory's monkey-patch. Exact call sequence per
    docs/baselines/TIER2_GPU_INTEGRATION.md #1. Appends each answer to `store` as it is produced."""
    sys.path.insert(0, repo_dir)
    import torch
    # POD-ONLY: requires KVCache-Factory cloned to `repo_dir` (see README.md) — its `pyramidkv` package
    # is not a pip-installable dependency, hence the sys.path.insert above instead of `pip install`.
    from pyramidkv.monkeypatch import replace_llama
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.memory.eval.baselines import build_messages

    print(f"[run_kvcompress] monkey-patching Llama attention for method={args.method!r} ...")
    replace_llama(args.method)  # patches transformers.models.llama.modeling_llama.Llama{Attention,
                                 # FlashAttention2,SdpaAttention}.forward in-place, process-global.

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # REQUIRED — eager/sdpa prefill on ~115k tok balloons VRAM
        device_map="cuda",
    )
    model.eval()

    # Per-layer eviction config — exact attribute names + KVCache-Factory defaults, lifted from its
    # run_longbench.py (window_size/kernel_size/pooling are the paper's SnapKV/H2O defaults; only
    # max_capacity_prompt is meant to be swept).
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        cfg = model.model.layers[i].self_attn.config
        cfg.window_size = 8                                    # uncompressed "observation" window (recent tok)
        cfg.max_capacity_prompt = args.max_capacity_prompt      # total per-layer KV budget in tokens (incl. window)
        cfg.kernel_size = 7                                     # pooling kernel for the eviction importance score
        cfg.pooling = "maxpool"
        cfg.merge = None
        cfg.floor = None

    done = store.done_ids()
    for it in items:
        if str(it["question_id"]) in done:
            continue
        try:
            # Instruct chat template (system + full history + dated question) — NOT a hand-concatenated
            # string (which skips the model's turn formatting and drops question_date). char_budget huge:
            # the FULL history must flow through so the compressed KV is built from all ~115k tokens.
            msgs, _ = build_messages("full_context", question=it["question"],
                                     full_history=it["full_history"], char_budget=10 ** 9,
                                     question_date=it.get("question_date"))
            input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = out[0][input_ids.shape[1]:]
            hyp = tok.decode(gen, skip_special_tokens=True)
            store.append(_record(it, hyp=hyp,
                                  finish_reason=_finish_reason(gen, args.max_new_tokens, tok.eos_token_id)))
        except Exception as e:  # noqa: BLE001 — crash-safe: record the failure, keep going, resume later
            print(f"[run_kvcompress] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(_record(it, error=f"{type(e).__name__}: {e}"))


def run_kvzip(args, items, model_name: str, repo_dir: str, store) -> None:
    """KVzip compress-then-query. Exact call sequence per docs/baselines/TIER2_GPU_INTEGRATION.md #2.
    Appends each answer to `store` as it is produced."""
    sys.path.insert(0, repo_dir)
    # POD-ONLY: `model.py` does `from model import ModelKVzip` — its relative imports assume CWD is the
    # KVzip repo root (see README.md), on top of needing repo_dir on sys.path. Requires the pinned
    # CUDA 12.1 / py3.10 / flash-attn==2.7.4.post1 (--no-build-isolation) + `make i` custom-kernel build.
    from model import ModelKVzip

    m = ModelKVzip(model_name)
    done = store.done_ids()
    for it in items:
        if str(it["question_id"]) in done:
            continue
        try:
            # prefill() chunks internally (16k blocks) -> handles the full ~115k-token history without a
            # separate truncation step here. do_score=True scores importance during prefill (no reuse across
            # queries since LongMemEval haystacks are per-question, unlike KVzip's multi-query benchmarks).
            kv = m.prefill(it["full_history"], load_score=False, do_score=True)
            kv.prune(ratio=args.ratio)  # ratio = fraction of KV RETAINED (0.3 -> evict 70%)
            # anchor the question to its date (temporal questions), same as the Tier-1 panel.
            q = it["question"]
            if it.get("question_date"):
                q = f"Current Date: {it['question_date']}\n{q}"
            query_ids = m.apply_template(q)
            # #17: honor --max-new-tokens. KVzip's canonical call (integration doc) is generate(ids, kv=kv)
            # with no cap kwarg; if this build's signature doesn't accept it, fall back rather than error
            # every item (a silent 100%-failure run). POD-VERIFY: if generate() accepts **kwargs and SILENTLY
            # IGNORES max_new_tokens, this pass-through is a no-op — confirm the cap is actually honored (else
            # patch KVzip's generate to thread it into its decode loop) before trusting the numbers.
            try:
                hyp = m.generate(query_ids, kv=kv, max_new_tokens=args.max_new_tokens)
            except TypeError:
                print("[run_kvcompress] WARN: KVzip.generate rejected max_new_tokens= ; using its default cap")
                hyp = m.generate(query_ids, kv=kv)
            store.append(_record(it, hyp=hyp))
        except Exception as e:  # noqa: BLE001 — crash-safe
            print(f"[run_kvcompress] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(_record(it, error=f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["snapkv", "h2o", "kvzip"], required=True)
    ap.add_argument("--model", default=None, help=f"default per method: {_DEFAULT_MODEL}")
    ap.add_argument("--repo-dir", default=None,
                    help="path to the cloned method repo; default per method (see README.md), "
                         f"e.g. {_DEFAULT_REPO_DIR}")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-capacity-prompt", type=int, default=2048,
                    help="snapkv/h2o: total retained KV tokens per layer (KVCache-Factory arg name)")
    ap.add_argument("--ratio", type=float, default=0.3,
                    help="kvzip: fraction of KV cache RETAINED after prune (0.3 == evict 70%%)")
    ap.add_argument("--seed", type=int, default=0, help="seed py/np/torch/cuda for reproducibility")
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (EM+containment only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    model_name = args.model or _DEFAULT_MODEL[args.method]
    repo_dir = str(Path(args.repo_dir or _DEFAULT_REPO_DIR[args.method]).expanduser())

    # --- everything below needs torch/transformers/the method repo — lazy on purpose, see module docstring ---
    _seed_everything(args.seed)
    from src.memory.data.longmemeval import load_longmemeval_text
    from src.memory.eval import score_longmemeval
    from src.memory.eval.results import ResultStore

    print(f"[run_kvcompress] method={args.method} model={model_name} repo_dir={repo_dir} "
          f"variant={args.variant} max_examples={args.max_examples} seed={args.seed}")
    items = load_longmemeval_text(variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_kvcompress] {len(items)} items; types={types}")

    # artifact name carries everything needed to tell two runs apart: method, model, variant, sample size,
    # the swept knob (capacity or ratio), generation cap, and the code commit.
    knob = f"cap{args.max_capacity_prompt}" if args.method != "kvzip" else f"ratio{args.ratio}"
    commit = _git_commit()
    tag = (f"longmemeval__{args.method}__{model_name.split('/')[-1]}__{args.variant}"
           f"__n{len(items)}__{knob}__g{args.max_new_tokens}__seed{args.seed}__{commit}")
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_kvcompress] resume: {n_done}/{len(items)} already done — generating the rest")

    if args.method in ("snapkv", "h2o"):
        run_kvcache_factory(args, items, model_name, repo_dir, store)
    else:
        run_kvzip(args, items, model_name, repo_dir, store)

    records = [r for r in store.all_records() if _valid(r)]
    agg = score_longmemeval(records, use_bem=not args.no_bem)
    store.merge_verdicts(agg.get("details", [])); store.compact()
    n_err = sum(1 for r in store.all_records() if r.get("error"))
    print(f"\n[run_kvcompress] overall_acc={agg.get('overall_accuracy', float('nan')):.3f}  "
          f"task_avg={agg.get('task_averaged_accuracy', float('nan')):.3f}  "
          f"abstention={agg.get('abstention_accuracy')}  n={agg.get('n_nonabstention')}  errors={n_err}")

    payload = {
        "dataset": "longmemeval", "method": args.method, "model": model_name,
        "meta": {"n": len(records), "n_errors": n_err, "variant": args.variant, "seed": args.seed,
                 "max_new_tokens": args.max_new_tokens, "commit": commit, "upstream_commit": _git_commit(repo_dir), "scorer": "score_longmemeval (deterministic, negation-guarded)",
                 "coverage": round(len(records) / len(items), 4) if items else None,
                 "max_capacity_prompt": args.max_capacity_prompt if args.method != "kvzip" else None,
                 "ratio": args.ratio if args.method == "kvzip" else None},
        "aggregate": {k: v for k, v in agg.items() if k != "details"},
        "store": str(store.path),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"[run_kvcompress] wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
