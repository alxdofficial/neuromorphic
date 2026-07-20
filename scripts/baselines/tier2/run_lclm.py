#!/usr/bin/env python3
"""Phase-2 Tier-2 baseline: LCLM — Latent Context Language Models over LongMemEval / MemoryAgentBench (POD-ONLY).

LCLM (Li, McLeish et al., "End-to-End Context Compression at Scale", arXiv:2606.09659) is our closest
concurrent competitor: an encoder (Qwen3-Embedding-0.6B) pools token chunks into SOFT TOKENS, an adapter
projects them, and a 4B decoder (Qwen3-4B) consumes them as latent context. Released weights:
`latent-context/0.6b-4b-LCLM-{4x,8x,16x}`. Unlike our FROZEN-decoder design, LCLM trains the decoder
end-to-end (see docs/baselines/PHASE2_BASELINES.md §2.5).

⚠ LICENSE: none stated on the repo/HF org as of 2026-07-18 — clearance is the user's call. ⚠ Checkpoints are
NOT loadable via vanilla transformers/vllm — clone github.com/LeonLixyz/LCLM and put it on PYTHONPATH
(`--repo-dir`); the context MUST be wrapped in `<|memory_start|> … <|memory_end|>`.

PER-CONTEXT REUSE (docs/baselines/TIER2_HOSTING.md): LCLM encodes the context to soft tokens BEFORE decoder
prefill, so in principle we encode a context ONCE and decode every question against the cached latents. This
runner is STRUCTURED for that (per-context via tier2_common.run_grouped) but currently uses the correct
per-question path (encode+decode together) — the encode-latents-once/reuse API must be wired against the
repo's inference/hf.py on the pod. See `_encode_memory` POD-VERIFY. For LongMemEval (unique histories) there
is nothing to reuse; for MAB, wiring the reuse is what keeps LCLM inside the pod time budget.

`--help` works without torch/the LCLM repo (lazy imports). VRAM ~9–10GB bf16 → fits a 24GB GPU, inference-only.
  python scripts/baselines/tier2/run_lclm.py --dataset longmemeval --checkpoint latent-context/0.6b-4b-LCLM-16x
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_DEFAULT_CHECKPOINT = "latent-context/0.6b-4b-LCLM-16x"   # highest compression; 16x→8x/4x trades fidelity
_DEFAULT_REPO_DIR = str(REPO.parent / "baselines" / "LCLM")  # local master/baselines; pod passes --repo-dir


def _generate_new_tokens(model, dec_tok, processor, prompt: str, max_new_tokens: int,
                         device="cuda"):
    """Mirror LCLM inference/hf.generate_text's flow but decode ROBUSTLY. LCLM generates via `inputs_embeds`
    (memory positions become latent embeddings), so HF `generate` returns ONLY the new tokens; we decode the
    full returned sequence UNLESS it is longer than the input (a prefixed regime), then slice the prefix.
    Greedy (do_sample=False) for determinism. Returns (text, finish_reason)."""
    import torch
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    proc = processor.process_wrapped_batch(prompts=[formatted], targets=None, padding="longest",
                                           truncation=True, return_tensors="pt")
    input_ids = proc["input_ids"].to(device)
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=proc["attention_mask"].to(device),
                             memory_token_ids=proc["memory_token_ids"], memory_positions=proc["memory_positions"],
                             latent_counts=proc["latent_counts"], max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=dec_tok.pad_token_id, eos_token_id=dec_tok.eos_token_id)
    seq = out[0]
    gen = seq[input_ids.shape[1]:] if seq.shape[0] > input_ids.shape[1] else seq   # slice ONLY if prefixed
    text = dec_tok.decode(gen, skip_special_tokens=True).strip()
    eos = dec_tok.eos_token_id
    truncated = gen.shape[0] >= max_new_tokens and (eos is None or int(gen[-1]) != eos)
    return text, ("length" if truncated else "stop")


def _encode_memory(model, processor, ctx: str):
    """POD-VERIFY (MAB reuse optimization). Encode a context to reusable soft-token latents ONCE, so the ~85
    questions sharing it don't each re-run the encoder. Requires an entry point in the LCLM repo's
    inference/hf.py that (a) runs the 0.6B encoder + adapter on the memory span and (b) lets `generate`
    consume the cached latents. Until wired on the pod, we return None → the runner falls back to the correct
    per-question encode+decode path. Wiring this is what brings LCLM-on-MAB inside the time budget."""
    return None


def run_lclm(args, items, checkpoint, repo_dir, store, dataset) -> None:
    sys.path.insert(0, repo_dir)
    from inference.hf import load_model
    from src.memory.eval.tier2_common import format_query, run_grouped

    model, dec_tok, processor = load_model(checkpoint, device="cuda", dtype="bf16")

    def encode_ctx(ctx, first_item):
        cached = _encode_memory(model, processor, ctx)      # None until reuse is wired on the pod
        return {"ctx": ctx, "latents": cached}

    def answer(mem, it):
        q = format_query(it, dataset)
        if mem["latents"] is not None:
            # POD-VERIFY: decode against cached latents (reuse path) — wire against the repo's API alongside
            # _encode_memory. Must produce the same (text, finish_reason) contract as _generate_new_tokens.
            raise NotImplementedError("LCLM latent-reuse decode not wired — see _encode_memory POD-VERIFY")
        # correct fallback: encode+decode together (per question). Context wrapped in the encoder markers.
        prompt = f"<|memory_start|>{mem['ctx']}<|memory_end|> {q}"
        return _generate_new_tokens(model, dec_tok, processor, prompt, args.max_new_tokens)

    def release():
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # audit #8: on MAB, _encode_memory returns None → we re-encode per question (no latent reuse), so the
    # advertised ~85x MAB speedup is NOT realized until the reuse is wired against the repo on the pod. Warn
    # loudly so it isn't mistaken for the fast path.
    if dataset == "memoryagentbench":
        print("[run_lclm] ⚠ MAB latent-reuse NOT wired (_encode_memory returns None) → the context is "
              "re-encoded for EVERY question (~85x the encoder work). Wire _encode_memory on the pod for the "
              "2-hour budget; LongMemEval is unaffected. (docs/baselines/TIER2_HOSTING.md, audit #8.)")
    run_grouped(items, encode_ctx, answer, store, "[run_lclm]", release=release)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT,
                    help=f"HF checkpoint (default {_DEFAULT_CHECKPOINT}); 4x/8x/16x = compression ratio")
    ap.add_argument("--repo-dir", default=_DEFAULT_REPO_DIR, help="path to the cloned LCLM repo (PYTHONPATH)")
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bem", action="store_true", help="skip BEM paraphrase scoring (LongMemEval only)")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    repo_dir = str(Path(args.repo_dir).expanduser())

    from src.memory.eval.tier2_common import (seed_everything, git_commit, load_items, build_tag, finalize)
    from src.memory.eval.results import ResultStore
    seed_everything(args.seed)

    print(f"[run_lclm] checkpoint={args.checkpoint} dataset={args.dataset} repo_dir={repo_dir} "
          f"variant={args.variant} max_examples={args.max_examples} seed={args.seed}")
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_lclm] {len(items)} items; types={types}")

    commit = git_commit(REPO)
    tag = build_tag(args.dataset, "lclm", args.checkpoint.split("/")[-1], args.variant, len(items),
                    "cmp", args.max_new_tokens, args.seed, commit)
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_lclm] resume: {n_done}/{len(items)} already done — generating the rest")

    run_lclm(args, items, args.checkpoint, repo_dir, store, args.dataset)

    finalize(args.dataset, "lclm", args.checkpoint, items, store, use_bem=not args.no_bem,
             extra_meta={"variant": args.variant, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                         "commit": commit, "upstream_commit": git_commit(repo_dir)},
             out_dir=out_dir, tag=tag, log_prefix="[run_lclm]")


if __name__ == "__main__":
    main()
