#!/usr/bin/env python3
"""Phase-2 Tier-2 GPU baseline: MemoryLLM / M+ over LongMemEval / MemoryAgentBench (POD-ONLY).

MemoryLLM (Wang et al., ICML'24) injects context into a fixed-size in-weights memory pool via
`inject_memory(ids, update_memory=True)`, then answers with a plain `generate()` — the memory is already
fused into hidden states, no prompt-side context at query time. M+ (Wang et al. 2025) adds a CPU-offloaded
long-term memory (LTM) store. Base model = **M+ (`YuWangX/mplus-8b`)**: MemoryLLM-8B's retention caps ~20k
tokens, too small for the ~115k haystacks; M+'s LTM (153,600 tok) holds them. See
`docs/baselines/TIER2_GPU_INTEGRATION.md` #3.

PER-CONTEXT REUSE (the local prompt-cache analog, docs/baselines/TIER2_HOSTING.md): MemoryLLM's design is
already "write once, query many" — we exploit it. `encode_ctx` clears memory, injects a context ONCE, and
snapshots the post-injection state; `answer` restores that snapshot (undoing any per-question LTM mutation)
and generates. MAB → 36 injections for 3,071 Q; LongMemEval → inject per question (unique histories).

STATE ISOLATION: each context's memory must NOT leak into the next. Both the short-term pool (`model.memory`)
AND the M+ LTM store (contents/keys/ages/frequencies/dropped-caches/update_step) mutate under
`inject_memory`. We snapshot ALL of them once after load (`_snapshot_memory_state`, which PRINTS what it
captured and HARD-WARNS if the LTM store wasn't found), restore to pristine before each context injection,
and restore the post-injection snapshot before each question.

`--help` works without torch/transformers/the MemoryLLM repo (heavy imports are lazy).

Example (on the pod, after scripts/baselines/tier2/README.md's setup):
  python scripts/baselines/tier2/run_memoryllm.py --dataset memoryagentbench --max-examples 20
  python scripts/baselines/tier2/run_memoryllm.py --dataset longmemeval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_DEFAULT_MODEL = "YuWangX/mplus-8b"
_DEFAULT_REPO_DIR = str(REPO.parent / "baselines" / "MemoryLLM")  # local master/baselines; pod passes --repo-dir

# Candidate attribute names for M+'s long-term-memory state (short-term `model.memory` handled separately).
# Bounded/explicit — NOT a dir() scan — so we never accidentally snapshot the 8B weight tensors.
_LTM_CANDIDATE_ATTRS = (
    "ltm_recall_frequencies", "cached_dropped_memories", "cached_dropped_memory_ages", "cached_dropped_keys",
    "ltm", "long_term_memory", "ltm_memory", "ltm_keys", "ltm_values", "ltm_ages", "ltm_age",
    "memory_ages", "retrieval_frequencies", "retrieval_freq", "dropped_memory", "dropped_memories",
    "update_step", "ltm_numpy", "ltm_key", "ltm_value",
)


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
    """True if v holds real state (not None / not a 0-length tensor/array/container) — so an attr that is
    empty at snapshot time doesn't silence the leak warning while state still leaks."""
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
        return True


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
              "(short-term pool) is guaranteed reset between contexts — the LTM store may LEAK and CORRUPT "
              "results. Inspect modeling_mplus.py on the pod, add the real LTM attr name(s), re-run BEFORE "
              "trusting any number. (docs/baselines/TIER2_GPU_INTEGRATION.md #3.)")
    else:
        print(f"[run_memoryllm] per-context reset will restore: model.memory + LTM buffers {captured}"
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
            cur.data.copy_(saved)
        else:
            setattr(model, attr, _copy_buf(saved))


def _inject_blocks(tok, full_history: str, block_tokens: int, min_tokens: int) -> list[list[int]]:
    """Split the context into FIXED `block_tokens`-token id blocks — the granularity M+ was evaluated at
    upstream (LongBench injects 512-token blocks). A short (<min_tokens) tail block is merged into the
    previous one (sub-16-token injects 'disturb' memory)."""
    ids = tok(full_history, add_special_tokens=False)["input_ids"]
    blocks = [ids[i:i + block_tokens] for i in range(0, len(ids), block_tokens)]
    if len(blocks) >= 2 and len(blocks[-1]) < min_tokens:
        blocks[-2] = blocks[-2] + blocks[-1]
        blocks.pop()
    return blocks


def run_memoryllm(args, items, model_name, repo_dir, store, dataset) -> dict:
    """Encode-once/reuse loop over contexts via tier2_common.run_grouped. Returns a timing dict (GPU-synced
    inject vs generate seconds + counts) so the block-size sweep can read the cost lever off the artifact."""
    sys.path.insert(0, repo_dir)
    import time

    import torch
    # POD-ONLY: `modeling_mplus.py` is a repo file (not a pip package); class name `MPlus` (MemoryLLM-8B/-chat
    # would be `MemoryLLM` from `modeling_memoryllm.py` — swap import + from_pretrained if benchmarking those).
    from modeling_mplus import MPlus
    from transformers import AutoTokenizer
    from src.memory.eval.tier2_common import format_query, finish_reason_of, run_grouped

    tok = AutoTokenizer.from_pretrained(model_name)
    # attn_implementation: sdpa by default (memory-efficient prefill of a ~115k-token context fits on an 80GB
    # H100, and it avoids the from-source flash-attn build). Pass --attn-impl flash_attention_2 if built.
    model = MPlus.from_pretrained(model_name, attn_implementation=args.attn_impl,
                                  torch_dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)  # re-cast per README: from_pretrained leaves rotary_emb.inv_freq fp32
    model.put_ltm_to_numpy()          # move LTM off-GPU to a numpy store for inference (README-mandated)
    model = model.cuda()
    model.eval()

    # Log the M+ memory geometry at runtime (audit rec): confirms the LTM retriever is present + capacities
    # match the paper (STM = num_tokens×num_blocks/layer; retrieve num_tokens×num_ltm_blocks/layer from LTM).
    cfg = model.config
    print(f"[run_memoryllm] M+ geometry: num_tokens={cfg.num_tokens} num_blocks={cfg.num_blocks} "
          f"→ STM {cfg.num_tokens * cfg.num_blocks} tok/layer · num_ltm_blocks={cfg.num_ltm_blocks} "
          f"→ retrieve {cfg.num_tokens * cfg.num_ltm_blocks} tok/layer · add_selector={getattr(cfg, 'add_selector', None)}")

    pristine, _ = _snapshot_memory_state(model)   # empty post-load state; restore before each context inject

    # GPU-synced timing (excludes the ~1-2 min model load, which the sweep amortizes separately): inject time
    # is THE block-size cost lever (bigger blocks → fewer sequential inject_memory calls → less overhead), and
    # M+ is overhead-bound so this is what moves. Kept local to this runner (no shared-harness edit).
    stats = {"encode_s": 0.0, "n_injects": 0, "answer_s": 0.0, "n_answers": 0}

    def encode_ctx(ctx, first_item):
        _restore_memory_state(model, pristine)                      # clear to pristine
        blocks = _inject_blocks(tok, ctx, args.inject_block_tokens, args.min_inject_tokens)
        t0 = time.perf_counter()
        for block in blocks:
            model.inject_memory(torch.tensor([block], dtype=torch.long).cuda(), update_memory=True)
        torch.cuda.synchronize()
        stats["encode_s"] += time.perf_counter() - t0
        stats["n_injects"] += len(blocks)
        return _snapshot_memory_state(model)[0]                     # reusable post-injection memory

    def answer(post_snap, it):
        _restore_memory_state(model, post_snap)                     # undo any prior question's LTM mutation
        q = format_query(it, dataset)
        # pretrained mplus-8b (no chat template): LongMemEval → "Question: … Answer:"; MAB template is
        # self-contained (already carries its own task instruction + answer cue) → feed as-is.
        prompt = f"Question: {q} Answer:" if dataset == "longmemeval" else q
        q_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            # GREEDY — mplus-8b's generation_config.json ships do_sample=True/temp=0.6/top_p=0.9, but every
            # authors' eval forces greedy (longbench_pred.py: num_beams=1, do_sample=False). Pin it so results
            # are deterministic + faithful to the paper (our seed only makes SAMPLING reproducible, not greedy).
            out = model.generate(input_ids=q_ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1)
        torch.cuda.synchronize()
        stats["answer_s"] += time.perf_counter() - t0
        stats["n_answers"] += 1
        gen = out[0][q_ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        # mplus-8b is a BASE model (no chat template / EOS discipline): it emits the answer, then rambles
        # into hallucinated follow-up "Question: … Answer: …" turns, so it reaches the token cap without an
        # EOS and finish_reason_of() would mark every item "length" → excluded from scoring (coverage 0).
        # Take the FIRST line as the answer (the standard base-model QA convention) and report a clean stop
        # so the answer IS scored. (The raw multi-line text is still in the record for audit.)
        answer_text = text.split("\n")[0].strip() or text.strip()
        return answer_text, "stop"

    def release():
        torch.cuda.empty_cache()   # run_grouped drops the old snapshot's ref FIRST → this reclaims its ~2.5GB

    run_grouped(items, encode_ctx, answer, store, "[run_memoryllm]", release=release)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=_DEFAULT_MODEL,
                    help="HF repo id. Default = M+ (only checkpoint that fits ~115k history).")
    ap.add_argument("--repo-dir", default=_DEFAULT_REPO_DIR, help="path to the cloned MemoryLLM repo")
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--attn-impl", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"],
                    help="M+ REQUIRES flash_attention_2: its eager/sdpa attention classes return 4 values but "
                         "the decoder layer unpacks 5 (encoder_retriever_weights) → sdpa/eager crash. Repo pins "
                         "flash-attn too. Default flash_attention_2; sdpa/eager only if the repo is patched.")
    ap.add_argument("--inject-block-tokens", type=int, default=512,
                    help="inject_memory() block size in tokens (512 = upstream LongBench granularity)")
    ap.add_argument("--min-inject-tokens", type=int, default=17,
                    help="a trailing block shorter than this is merged into the previous one (hard min > 16)")
    ap.add_argument("--seed", type=int, default=0, help="seed py/np/torch/cuda (M+ randomly drops memory)")
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (LongMemEval only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    repo_dir = str(Path(args.repo_dir).expanduser())

    from src.memory.eval.tier2_common import (seed_everything, git_commit, load_items, build_tag, finalize)
    from src.memory.eval.results import ResultStore
    seed_everything(args.seed)

    print(f"[run_memoryllm] model={args.model} dataset={args.dataset} repo_dir={repo_dir} "
          f"variant={args.variant} max_examples={args.max_examples} seed={args.seed}")
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_memoryllm] {len(items)} items; types={types}")

    commit = git_commit(REPO)
    tag = build_tag(args.dataset, "memoryllm", args.model.split("/")[-1], args.variant, len(items),
                    f"blk{args.inject_block_tokens}", args.max_new_tokens, args.seed, commit)
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_memoryllm] resume: {n_done}/{len(items)} already done — generating the rest")

    stats = run_memoryllm(args, items, args.model, repo_dir, store, args.dataset)
    timing = {**stats, "inject_block_tokens": args.inject_block_tokens,
              "s_per_inject": round(stats["encode_s"] / max(stats["n_injects"], 1), 4),
              "s_per_answer": round(stats["answer_s"] / max(stats["n_answers"], 1), 4)}
    print(f"[run_memoryllm] timing: {stats['n_injects']} injects in {stats['encode_s']:.1f}s "
          f"({timing['s_per_inject']:.3f}s/inject) · {stats['n_answers']} answers in {stats['answer_s']:.1f}s "
          f"({timing['s_per_answer']:.3f}s/answer)")

    finalize(args.dataset, "memoryllm", args.model, items, store, use_bem=not args.no_bem,
             extra_meta={"variant": args.variant, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                         "min_inject_tokens": args.min_inject_tokens, "commit": commit,
                         "upstream_commit": git_commit(repo_dir), "timing": timing},
             out_dir=out_dir, tag=tag, log_prefix="[run_memoryllm]")


if __name__ == "__main__":
    main()
