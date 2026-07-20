#!/usr/bin/env python3
"""Phase-2 Tier-2 GPU baseline: KV-cache compression over LongMemEval / MemoryAgentBench (POD-ONLY).

Two method families behind one CLI, both compress the KV cache built from the FULL context before answering
— the thing Tier-1 (API, prompt-level truncation/RAG) cannot do, and our matched-decoder model does
architecturally instead of via KV eviction:

  --method snapkv | h2o   KVCache-Factory (github.com/Zefan-Cai/KVCache-Factory, MIT). Monkey-patches HF's
                          Llama attention `.forward` to evict KV entries during prefill/decode. QUERY-AWARE:
                          eviction depends on the question, so the compressed cache CANNOT be reused across
                          questions → LongMemEval only (each question has its own history anyway). Refused on
                          MemoryAgentBench (would re-prefill the same context ~85×, both slow AND degraded —
                          see KVzip's multi-query analysis). Use --method kvzip for MAB.
  --method kvzip          KVzip (github.com/snu-mllab/KVzip, MIT, NeurIPS'25). QUERY-AGNOSTIC: prefill +
                          importance-score a context ONCE, prune, then answer any number of questions from the
                          single reusable compressed cache. This is the correct KV baseline for MAB's
                          inject-once/query-many structure (docs/baselines/TIER2_HOSTING.md).

Per-CONTEXT reuse (the local analog of the Tier-1 prompt-cache win) is handled by `tier2_common.run_grouped`:
it groups questions by distinct context and calls our `encode_ctx` ONCE per context. MAB → 36 encodes for
3,071 Q; LongMemEval → 1-item groups (encode per question). Scores with the SAME deterministic scorers as
Tier-1 so numbers are directly comparable. RESUMABLE + crash-safe (per-question ResultStore).

`--help` works without torch/transformers/either external repo (every heavy import is lazy inside a helper).

Examples (on the pod, after scripts/baselines/tier2/README.md's setup):
  python scripts/baselines/tier2/run_kvcompress.py --method kvzip  --dataset memoryagentbench --max-examples 20
  python scripts/baselines/tier2/run_kvcompress.py --method snapkv --dataset longmemeval --max-capacity-prompt 2048
"""
from __future__ import annotations

import argparse
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
_BASELINES = REPO.parent / "baselines"   # local master/baselines; pod passes --repo-dir explicitly
_DEFAULT_REPO_DIR = {
    "snapkv": str(_BASELINES / "KVCache-Factory"),
    "h2o": str(_BASELINES / "KVCache-Factory"),
    "kvzip": str(_BASELINES / "KVzip"),
}


def _cuda_release() -> None:
    """Free the previous context's compressed KV before encoding the next — the pruned cache is large
    (a compressed 115k-token KV) and 36 of them would accumulate on MAB otherwise. Called by run_grouped
    AFTER the previous mem's ref is dropped, so empty_cache can actually reclaim it."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_kvcache_factory(args, items, model_name, repo_dir, store, dataset, meta_out) -> None:
    """SnapKV / H2O via KVCache-Factory's monkey-patch. QUERY-AWARE → no cross-question reuse: `encode_ctx`
    is a no-op returning the context, and `answer` re-prefills the (context+question) prompt per question
    with eviction. On LongMemEval (unique histories) that is the only correct behavior; refused on MAB."""
    meta_out["gen_cap_enforced"] = True                # HF generate() honors max_new_tokens
    meta_out["gen_finish_reason_available"] = True     # we compute finish_reason_of() from the token stream
    sys.path.insert(0, repo_dir)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.memory.eval.baselines import build_messages
    from src.memory.eval.tier2_common import format_query, finish_reason_of, run_grouped

    # audit #11: KVCache-Factory patches Llama and Mistral SEPARATELY — a Mistral --model needs replace_mistral,
    # NOT replace_llama (which would leave Mistral attention unpatched → no eviction, silently wrong).
    is_mistral = "mistral" in model_name.lower()
    if is_mistral:
        from pyramidkv.monkeypatch import replace_mistral as _replace
    else:
        from pyramidkv.monkeypatch import replace_llama as _replace
    print(f"[run_kvcompress] monkey-patching {'Mistral' if is_mistral else 'Llama'} attention "
          f"for method={args.method!r} ...")
    _replace(args.method)       # process-global patch of the model family's *Attention.forward

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        # sdpa by default (80GB fits a 115k-tok prefill; avoids the flash-attn build). NOTE: if KVCache-Factory
        # only monkey-patches the flash-attn attention class, snapkv/h2o eviction won't apply under sdpa —
        # the smoke test's KV-size check catches that; pass --attn-impl flash_attention_2 (built) if so.
        attn_implementation=args.attn_impl,
        device_map="cuda")
    model.eval()

    # Per-layer eviction config (KVCache-Factory defaults from its run_longbench.py; only max_capacity_prompt
    # is meant to be swept). window_size/kernel_size/pooling are the paper's SnapKV/H2O defaults.
    for i in range(model.config.num_hidden_layers):
        cfg = model.model.layers[i].self_attn.config
        cfg.window_size = 8
        cfg.max_capacity_prompt = args.max_capacity_prompt
        cfg.kernel_size = 7
        cfg.pooling = "maxpool"
        cfg.merge = None
        cfg.floor = None

    def encode_ctx(ctx, first_item):
        return ctx                                   # eviction is query-dependent → nothing to precompute

    def answer(ctx, it):
        # Instruct chat template (system + full context + dated/templated question); char_budget huge so the
        # FULL context flows through the patched attention and the compressed KV is built from all of it.
        msgs, _ = build_messages("full_context", question=format_query(it, dataset),
                                 full_history=ctx, char_budget=10 ** 9)
        input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[0][input_ids.shape[1]:]
        return (tok.decode(gen, skip_special_tokens=True),
                finish_reason_of(gen, args.max_new_tokens, tok.eos_token_id))

    run_grouped(items, encode_ctx, answer, store, f"[run_kvcompress {args.method}]")


def run_kvzip(args, items, model_name, repo_dir, store, dataset, meta_out) -> None:
    """KVzip compress-then-query WITH per-context reuse. `encode_ctx` prefills + importance-scores + prunes a
    context ONCE (query-agnostic → the pruned cache is reusable); `answer` decodes each question against that
    single cache. This is the whole point of KVzip and the reason it — not SnapKV — is the MAB KV baseline."""
    sys.path.insert(0, repo_dir)
    import inspect
    from model import ModelKVzip
    from src.memory.eval.tier2_common import format_query, run_grouped

    m = ModelKVzip(model_name)

    # audit #3: KVzip's generate() may NOT accept max_new_tokens (→ it silently uses its own hard-coded cap,
    # e.g. 512). DETECT it up front instead of relying on a per-call TypeError, and RECORD whether our cap is
    # actually enforced so the artifact doesn't claim g{max_new_tokens} while a different cap ran.
    try:
        _cap_ok = "max_new_tokens" in inspect.signature(m.generate).parameters
    except (ValueError, TypeError):
        _cap_ok = False
    meta_out["gen_cap_enforced"] = _cap_ok
    meta_out["gen_finish_reason_available"] = False   # KVzip returns a string, no finish signal → can't detect length
    if not _cap_ok:
        # audit #1: upstream generate() ignores max_new_tokens and uses KVzip's hard-coded 512 (model/wrapper.py).
        # Record the ACTUAL cap so the artifact isn't mislabeled, and — for a FAIR comparison with the
        # 64-capped API/other baselines (a longer generation has more chances to contain the gold under
        # substring scoring) — POD-PATCH wrapper.py's `max_new_tokens=512` default to `args.max_new_tokens`
        # (or set the wrapper attribute) before the real run. Until patched, KVzip runs at 512.
        meta_out["gen_cap_actual"] = 512
        meta_out["gen_cap_note"] = ("upstream cap 512 NOT overridden; POD-PATCH KVzip model/wrapper.py to honor "
                                    f"--max-new-tokens={args.max_new_tokens} for a fair vs-64 comparison")
        print(f"[run_kvcompress] ⚠ KVzip.generate() ignores max_new_tokens → runs at upstream 512, NOT "
              f"{args.max_new_tokens}. UNFAIR vs 64-capped baselines under substring scoring. POD-PATCH "
              "model/wrapper.py before trusting KVzip numbers. (meta records gen_cap_actual=512.)")

    def encode_ctx(ctx, first_item):
        # prefill() chunks internally (16k blocks) → handles a full ~115k-tok (or longer, truncated to window)
        # context without a separate truncation step. do_score=True scores KV importance during prefill.
        kv = m.prefill(ctx, load_score=False, do_score=True)
        kv.prune(ratio=args.ratio)                    # ratio = fraction of KV RETAINED (0.3 → evict 70%)
        return kv

    def answer(kv, it):
        query_ids = m.apply_template(format_query(it, dataset))
        # POD-VERIFY: KVzip is query-agnostic and its own multi-query benchmark reuses ONE pruned cache across
        # queries, so generate() must treat `kv` read-only. If answers degrade AFTER the first question in a
        # group (i.e. the cache is mutated by decode), clone the cache per question here instead.
        hyp = (m.generate(query_ids, kv=kv, max_new_tokens=args.max_new_tokens) if _cap_ok
               else m.generate(query_ids, kv=kv))
        return hyp, "stop"                            # KVzip gives no finish signal → cannot detect a cutoff

    run_grouped(items, encode_ctx, answer, store, "[run_kvcompress kvzip]", release=_cuda_release)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["snapkv", "h2o", "kvzip"], required=True)
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--model", default=None, help=f"default per method: {_DEFAULT_MODEL}")
    ap.add_argument("--repo-dir", default=None,
                    help=f"path to the cloned method repo; default per method: {_DEFAULT_REPO_DIR}")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                    help="attention backend for snapkv/h2o (sdpa fits 80GB; flash_attention_2 if built). "
                         "kvzip uses its own loader and ignores this.")
    ap.add_argument("--max-capacity-prompt", type=int, default=2048,
                    help="snapkv/h2o: total retained KV tokens per layer (KVCache-Factory arg name)")
    ap.add_argument("--ratio", type=float, default=0.3,
                    help="kvzip: fraction of KV cache RETAINED after prune (0.3 == evict 70%%)")
    ap.add_argument("--seed", type=int, default=0, help="seed py/np/torch/cuda for reproducibility")
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (LongMemEval only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    if args.method in ("snapkv", "h2o") and args.dataset == "memoryagentbench":
        sys.exit(f"[run_kvcompress] {args.method} is QUERY-AWARE and cannot reuse a compressed cache across "
                 "MemoryAgentBench's ~85 questions/context (it would re-prefill each, slow AND degraded). "
                 "Use --method kvzip for --dataset memoryagentbench.")

    model_name = args.model or _DEFAULT_MODEL[args.method]
    repo_dir = str(Path(args.repo_dir or _DEFAULT_REPO_DIR[args.method]).expanduser())

    # --- everything below needs torch/transformers/the method repo — lazy on purpose ---
    from src.memory.eval.tier2_common import (seed_everything, git_commit, load_items, build_tag, finalize)
    from src.memory.eval.results import ResultStore
    seed_everything(args.seed)

    print(f"[run_kvcompress] method={args.method} dataset={args.dataset} model={model_name} "
          f"repo_dir={repo_dir} variant={args.variant} max_examples={args.max_examples} seed={args.seed}")
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_kvcompress] {len(items)} items; types={types}")

    knob = f"ratio{args.ratio}" if args.method == "kvzip" else f"cap{args.max_capacity_prompt}"
    commit = git_commit(REPO)
    tag = build_tag(args.dataset, args.method, model_name.split("/")[-1], args.variant, len(items),
                    knob, args.max_new_tokens, args.seed, commit)
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_kvcompress] resume: {n_done}/{len(items)} already done — generating the rest")

    meta_out: dict = {}
    if args.method in ("snapkv", "h2o"):
        run_kvcache_factory(args, items, model_name, repo_dir, store, args.dataset, meta_out)
    else:
        run_kvzip(args, items, model_name, repo_dir, store, args.dataset, meta_out)

    finalize(args.dataset, args.method, model_name, items, store, use_bem=not args.no_bem,
             extra_meta={"variant": args.variant, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                         "commit": commit, "upstream_commit": git_commit(repo_dir),
                         "max_capacity_prompt": args.max_capacity_prompt if args.method != "kvzip" else None,
                         "ratio": args.ratio if args.method == "kvzip" else None, **meta_out},
             out_dir=out_dir, tag=tag, log_prefix="[run_kvcompress]")


if __name__ == "__main__":
    main()
