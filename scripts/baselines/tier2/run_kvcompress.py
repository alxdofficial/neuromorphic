#!/usr/bin/env python3
"""Phase-2 Tier-2 GPU baseline: KV-cache compression over LongMemEval / MemoryAgentBench.

Three method integrations behind one CLI compress or bound the KV cache while processing the FULL context
— the thing Tier-1 (API, prompt-level truncation/RAG) cannot do, and our matched-decoder model does
architecturally instead of via KV eviction:

  --method snapkv         KVCache-Factory (github.com/Zefan-Cai/KVCache-Factory, MIT). Monkey-patches HF's
                          Llama attention `.forward` to compress the prompt KV. QUERY-AWARE:
                          eviction depends on the question, so the compressed cache CANNOT be reused across
                          questions → LongMemEval only (each question has its own history anyway). Refused on
                          MemoryAgentBench (would re-prefill the same context ~85×, both slow AND degraded —
                          see KVzip's multi-query analysis). Use --method kvzip for MAB.
  --method h2o            First-party Llama-3.1/GQA adapter of the official FMInference/H2O infinite-streaming
                          policy. Raw keys are re-rotated at compact cache positions, so logical streams can exceed
                          the pretrained context window. MAB contexts are prefetched once and forked per question.
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

# Default base model per method (SnapKV: Llama/Mistral only; H2O adapter: Llama; KVzip: several families).
_DEFAULT_MODEL = {
    "snapkv": "meta-llama/Llama-3.1-8B-Instruct",
    "h2o": "meta-llama/Llama-3.1-8B-Instruct",
    "kvzip": "Qwen/Qwen2.5-7B-Instruct-1M",
}
_LLAMA31_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
_BASELINES = REPO.parent / "baselines"   # local master/baselines; pod passes --repo-dir explicitly
_DEFAULT_REPO_DIR = {
    "snapkv": str(_BASELINES / "KVCache-Factory"),
    "kvzip": str(_BASELINES / "KVzip"),
}


def _chat_input_ids(tokenizer, messages):
    """Normalize Transformers 4.x tensor and 5.x BatchEncoding return types."""
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"chat template returned unexpected input_ids shape: {getattr(input_ids, 'shape', None)}")
    return input_ids


def _benchmark_messages(build_messages, item, context: str, dataset: str):
    """Build the benchmark-native full-context prompt without double-wrapping MAB templates."""
    if dataset == "memoryagentbench":
        return build_messages(
            "full_context",
            question=item["question"],
            full_history=context,
            char_budget=10 ** 9,
            system=item.get("system"),
            question_template=item.get("question_template"),
            context_header=item.get("context_header") or "# Context",
        )[0]
    return build_messages(
        "full_context",
        question=item["question"],
        question_date=item.get("question_date"),
        full_history=context,
        char_budget=10 ** 9,
    )[0]


def _common_aligned_prefix(left, right, alignment: int) -> int:
    """Longest equal token prefix, rounded down so later chunk boundaries remain identical."""
    limit = min(left.shape[1], right.shape[1])
    unequal = (left[0, :limit] != right[0, :limit]).nonzero()
    common = limit if unequal.numel() == 0 else int(unequal[0, 0])
    common = min(common, limit - 1)  # every query must retain a non-empty suffix
    return max(0, common // alignment * alignment)


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
    """SnapKV via KVCache-Factory's monkey-patch. QUERY-AWARE → no cross-question reuse: `encode_ctx`
    is a no-op returning the context, and `answer` re-prefills the (context+question) prompt per question
    with eviction. On LongMemEval (unique histories) that is the only correct behavior; refused on MAB."""
    meta_out["gen_cap_enforced"] = True                # HF generate() honors max_new_tokens
    meta_out["gen_finish_reason_available"] = True     # we compute finish_reason_of() from the token stream
    sys.path.insert(0, repo_dir)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.memory.eval.baselines import build_messages
    from src.memory.eval.tier2_common import finish_reason_of, format_query, run_grouped

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
        # only monkey-patches the flash-attn attention class, SnapKV eviction won't apply under sdpa —
        # the smoke test's KV-size check catches that; pass --attn-impl flash_attention_2 (built) if so.
        attn_implementation=args.attn_impl,
        device_map="cuda")
    model.eval()

    # Per-layer eviction config (KVCache-Factory defaults from its run_longbench.py; only max_capacity_prompt
    # is meant to be swept). window_size/kernel_size/pooling are the paper's SnapKV defaults.
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


def _h2o_budgets(args) -> tuple[int, int]:
    heavy = args.h2o_heavy_size
    recent = args.h2o_recent_size
    if heavy is None and recent is None:
        heavy = args.max_capacity_prompt // 2
        recent = args.max_capacity_prompt - heavy
    elif heavy is None:
        heavy = args.max_capacity_prompt - recent
    elif recent is None:
        recent = args.max_capacity_prompt - heavy
    if heavy <= 0 or recent <= 0 or heavy + recent != args.max_capacity_prompt:
        raise ValueError(
            "H2O heavy/recent sizes must be positive and sum to --max-capacity-prompt "
            f"({heavy} + {recent} != {args.max_capacity_prompt})"
        )
    return heavy, recent


def _h2o_knob(args) -> str:
    heavy_size, recent_size = _h2o_budgets(args)
    revision = (args.model_revision or "floating")[:8]
    return (f"rolling-cap{args.max_capacity_prompt}-hh{heavy_size}-recent{recent_size}-"
            f"chunk{args.prefill_chunk_size}-{args.h2o_head_mode}-rev{revision}")


def run_h2o(args, items, model_name, store, dataset, meta_out) -> None:
    """Infinite-streaming H2O with exact per-context snapshot reuse on MAB."""
    from src.memory.eval.baselines import build_messages
    from src.memory.eval.h2o_llama import load_llama_engine
    from src.memory.eval.tier2_common import group_by_context, run_grouped

    heavy_size, recent_size = _h2o_budgets(args)
    engine, tok = load_llama_engine(
        model_name,
        heavy_size=heavy_size,
        recent_size=recent_size,
        prefill_chunk_size=args.prefill_chunk_size,
        head_mode=args.h2o_head_mode,
        revision=args.model_revision,
    )

    eos_ids = getattr(engine.model.generation_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tok.eos_token_id

    meta_out.update({
        "gen_cap_enforced": True,
        "gen_finish_reason_available": True,
        "h2o_implementation": "neuromorphic_infinite_streaming_llama31",
        "h2o_official_repo": "https://github.com/FMInference/H2O",
        "h2o_official_reference_commit": "ac75c2a8a9e76832b2a4139b9363373b56336bfb",
        "model_revision_requested": args.model_revision,
        "model_revision_loaded": getattr(engine.model.config, "_commit_hash", None),
        "h2o_heavy_size": heavy_size,
        "h2o_recent_size": recent_size,
        "h2o_head_mode": args.h2o_head_mode,
        "h2o_position_mode": "rolling",
        "prefill_chunk_size": args.prefill_chunk_size,
        "h2o_prefill_note": (
            "official infinite-streaming position rolling with raw retained keys; chunked prefill is required "
            "for bounded attention and incoming chunks temporarily extend KV before post-attention pruning"
        ),
        "h2o_query_visibility_note": (
            "the question is at the prompt tail; it can update surviving H2O scores but cannot recover "
            "context tokens evicted by earlier streaming-prefill chunks"
        ),
        "h2o_mab_reuse": dataset == "memoryagentbench",
    })
    timing = {"n": 0, "prompt_tokens": 0, "query_suffix_tokens": 0, "generated_tokens": 0,
              "prefill_seconds": 0.0, "snapshot_fork_seconds": 0.0, "decode_seconds": 0.0}
    shared_timing = {"contexts": 0, "tokens": 0, "seconds": 0.0}
    grouped_items = group_by_context(items)

    def prompt_ids(ctx, it):
        return _chat_input_ids(tok, _benchmark_messages(build_messages, it, ctx, dataset))

    def encode_ctx(ctx, first_item):
        if dataset != "memoryagentbench":
            return ctx
        group = grouped_items[ctx]
        if len(group) < 2:
            return {"context": ctx, "prefix_ids": None, "snapshot": None}

        left = prompt_ids(ctx, group[0])
        right = prompt_ids(ctx, group[1])
        prefix_length = _common_aligned_prefix(left, right, args.prefill_chunk_size)
        if prefix_length == 0:
            raise RuntimeError("MAB prompts unexpectedly have no reusable aligned token prefix")
        prefix_ids = left[:, :prefix_length].contiguous()
        snapshot = engine.prefill(prefix_ids)
        shared_timing["contexts"] += 1
        shared_timing["tokens"] += prefix_length
        shared_timing["seconds"] += snapshot.prefill_seconds
        return {"context": ctx, "prefix_ids": prefix_ids, "snapshot": snapshot}

    def answer(memory, it):
        ctx = memory["context"] if isinstance(memory, dict) else memory
        input_ids = prompt_ids(ctx, it)
        if isinstance(memory, dict) and memory["snapshot"] is not None:
            prefix_ids = memory["prefix_ids"]
            prefix_length = prefix_ids.shape[1]
            if input_ids.shape[1] <= prefix_length or not input_ids[:, :prefix_length].equal(prefix_ids):
                raise RuntimeError("question prompt does not match the cached MAB token prefix")
            result = engine.generate_from_snapshot(
                memory["snapshot"],
                input_ids[:, prefix_length:],
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_ids,
                total_prompt_tokens=input_ids.shape[1],
            )
        else:
            result = engine.generate(input_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids)
        peak = result.diagnostics.get("peak_vram_bytes")
        if peak is not None:
            meta_out["peak_vram_bytes_max"] = max(int(peak), int(meta_out.get("peak_vram_bytes_max", 0)))
        timing["n"] += 1
        for key in ("prompt_tokens", "generated_tokens", "prefill_seconds", "decode_seconds"):
            timing[key] += result.diagnostics[key]
        timing["query_suffix_tokens"] += result.diagnostics["query_suffix_tokens"]
        timing["snapshot_fork_seconds"] += result.diagnostics.get("snapshot_fork_seconds", 0.0)
        meta_out["h2o_last_diagnostics"] = result.diagnostics
        return tok.decode(result.token_ids[0], skip_special_tokens=True), result.finish_reason

    run_grouped(items, encode_ctx, answer, store, "[run_kvcompress h2o]", release=_cuda_release)
    if timing["n"]:
        timing["prefill_tokens_per_second"] = timing["query_suffix_tokens"] / timing["prefill_seconds"]
        timing["decode_tokens_per_second"] = timing["generated_tokens"] / timing["decode_seconds"]
        streamed = shared_timing["tokens"] + timing["query_suffix_tokens"]
        prefill_wall = shared_timing["seconds"] + timing["prefill_seconds"] + timing["snapshot_fork_seconds"]
        timing["unique_streamed_tokens"] = streamed
        timing["prefill_and_fork_seconds"] = prefill_wall
        timing["effective_streamed_tokens_per_second"] = streamed / prefill_wall
    meta_out["timing_current_process"] = timing
    meta_out["h2o_shared_prefill_timing"] = shared_timing


def run_kvzip(args, items, model_name, repo_dir, store, dataset, meta_out) -> None:
    """KVzip compress-then-query WITH per-context reuse. `encode_ctx` prefills + importance-scores + prunes a
    context ONCE (query-agnostic → the pruned cache is reusable); `answer` decodes each question against that
    single cache. This is the whole point of KVzip and the reason it — not SnapKV — is the MAB KV baseline."""
    sys.path.insert(0, repo_dir)
    import time

    import torch
    from model import ModelKVzip

    from src.memory.eval.tier2_common import format_query, run_grouped

    m = ModelKVzip(model_name)

    # Upstream exposes generation controls through this public dictionary rather than generate() kwargs.
    # Set it directly so the Phase-2-wide cap is genuinely enforced without patching model logic.
    if not isinstance(getattr(m, "gen_kwargs", None), dict):
        raise RuntimeError("KVzip ModelKVzip.gen_kwargs is unavailable; cannot enforce generation policy")
    m.gen_kwargs["max_new_tokens"] = args.max_new_tokens
    meta_out["gen_cap_enforced"] = True
    meta_out["gen_cap_actual"] = args.max_new_tokens
    meta_out["gen_finish_reason_available"] = False   # KVzip returns a string, no finish signal → can't detect length
    meta_out["model_revision_loaded"] = getattr(m.model.config, "_commit_hash", None)
    meta_out["kvzip_prefill_chunk_size"] = args.kvzip_prefill_chunk_size
    meta_out["kvzip_scoring_chunk_size"] = 2000       # upstream ModelKVzip.self_task default; paper setting
    meta_out["kvzip_allocation"] = "pair_nonuniform"
    meta_out["kvzip_precision"] = str(m.dtype)
    timing = {"contexts": 0, "questions": 0, "compression_seconds": 0.0, "decode_seconds": 0.0}

    def encode_ctx(ctx, first_item):
        # prefill() chunks internally (16k blocks) → handles a full ~115k-tok (or longer, truncated to window)
        # context without a separate truncation step. do_score=True scores KV importance during prefill.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        kv = m.prefill(ctx, prefill_chunk_size=args.kvzip_prefill_chunk_size,
                       load_score=False, do_score=True)
        kv.prune(ratio=args.ratio)                    # ratio = fraction of KV RETAINED (0.3 → evict 70%)
        timing["contexts"] += 1
        timing["compression_seconds"] += time.perf_counter() - started
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            meta_out["peak_vram_bytes_max"] = max(int(peak), int(meta_out.get("peak_vram_bytes_max", 0)))
        return kv

    def answer(kv, it):
        query_ids = m.apply_template(format_query(it, dataset))
        # POD-VERIFY: KVzip is query-agnostic and its own multi-query benchmark reuses ONE pruned cache across
        # queries, so generate() must treat `kv` read-only. If answers degrade AFTER the first question in a
        # group (i.e. the cache is mutated by decode), clone the cache per question here instead.
        started = time.perf_counter()
        hyp = m.generate(query_ids, kv=kv, update_cache=False)
        timing["questions"] += 1
        timing["decode_seconds"] += time.perf_counter() - started
        return hyp, "stop"                            # KVzip gives no finish signal → cannot detect a cutoff

    run_grouped(items, encode_ctx, answer, store, "[run_kvcompress kvzip]", release=_cuda_release)
    meta_out["timing_current_process"] = timing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["snapkv", "h2o", "kvzip"], required=True)
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--model", default=None, help=f"default per method: {_DEFAULT_MODEL}")
    ap.add_argument("--model-revision", default=None,
                    help="H2O Hugging Face revision; default model is pinned to its audited commit")
    ap.add_argument("--repo-dir", default=None,
                    help=f"path to the cloned method repo (snapkv/kvzip only); defaults: {_DEFAULT_REPO_DIR}")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                    help="attention backend for snapkv; h2o/kvzip use their own attention paths")
    ap.add_argument("--max-capacity-prompt", type=int, default=2048,
                    help="snapkv/h2o: total retained KV tokens per layer")
    ap.add_argument("--h2o-heavy-size", type=int, default=None,
                    help="H2O heavy-hitter slots; default half of --max-capacity-prompt")
    ap.add_argument("--h2o-recent-size", type=int, default=None,
                    help="H2O recent-token slots; default remaining half of --max-capacity-prompt")
    ap.add_argument("--h2o-head-mode", choices=["query_head", "kv_head"], default="query_head",
                    help="query_head is closest to original H2O; kv_head is a lower-memory GQA adaptation")
    ap.add_argument("--prefill-chunk-size", type=int, default=512,
                    help="H2O streaming-prefill chunk (must be <= recent size)")
    ap.add_argument("--ratio", type=float, default=0.3,
                    help="kvzip: fraction of KV cache RETAINED after prune (0.3 == evict 70%%)")
    ap.add_argument("--kvzip-prefill-chunk-size", type=int, default=16_000,
                    help="kvzip: upstream prefill chunk size; scoring stays at the paper's fixed 2,000 tokens")
    ap.add_argument("--seed", type=int, default=0, help="seed py/np/torch/cuda for reproducibility")
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (LongMemEval only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    if args.kvzip_prefill_chunk_size <= 0:
        ap.error("--kvzip-prefill-chunk-size must be positive")

    if args.method == "snapkv" and args.dataset == "memoryagentbench":
        sys.exit("[run_kvcompress] snapkv has no reusable per-context cache path across "
                 "MemoryAgentBench's ~85 questions/context. Use --method h2o or kvzip instead.")

    model_name = args.model or _DEFAULT_MODEL[args.method]
    if args.method == "h2o" and args.model_revision is None and model_name == _DEFAULT_MODEL["h2o"]:
        args.model_revision = _LLAMA31_REVISION
    repo_dir = None
    if args.method != "h2o":
        repo_dir = str(Path(args.repo_dir or _DEFAULT_REPO_DIR[args.method]).expanduser())

    # --- everything below needs torch/transformers/the method repo — lazy on purpose ---
    from src.memory.eval.results import ResultStore
    from src.memory.eval.tier2_common import build_tag, finalize, git_commit, load_items, seed_everything
    seed_everything(args.seed)

    print(f"[run_kvcompress] method={args.method} dataset={args.dataset} model={model_name} "
          f"repo_dir={repo_dir} variant={args.variant} max_examples={args.max_examples} seed={args.seed}")
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_kvcompress] {len(items)} items; types={types}")

    if args.method == "kvzip":
        knob = f"ratio{args.ratio}-prefill{args.kvzip_prefill_chunk_size}"
    elif args.method == "h2o":
        knob = _h2o_knob(args)
    else:
        knob = f"cap{args.max_capacity_prompt}"
    commit = git_commit(REPO)
    tag = build_tag(args.dataset, args.method, model_name.split("/")[-1], args.variant, len(items),
                    knob, args.max_new_tokens, args.seed, commit)
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_kvcompress] resume: {n_done}/{len(items)} already done — generating the rest")

    meta_out: dict = {}
    if args.method == "snapkv":
        run_kvcache_factory(args, items, model_name, repo_dir, store, args.dataset, meta_out)
    elif args.method == "h2o":
        run_h2o(args, items, model_name, store, args.dataset, meta_out)
    else:
        run_kvzip(args, items, model_name, repo_dir, store, args.dataset, meta_out)

    finalize(args.dataset, args.method, model_name, items, store, use_bem=not args.no_bem,
             extra_meta={"variant": args.variant, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                         "commit": commit, "upstream_commit": git_commit(repo_dir) if repo_dir else None,
                         "max_capacity_prompt": args.max_capacity_prompt if args.method != "kvzip" else None,
                         "ratio": args.ratio if args.method == "kvzip" else None, **meta_out},
             out_dir=out_dir, tag=tag, log_prefix="[run_kvcompress]")


if __name__ == "__main__":
    main()
