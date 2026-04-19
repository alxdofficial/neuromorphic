#!/usr/bin/env python3
"""Measure vanilla HF LM's native accuracy on the passphrase-retrieval task.

No memory, no training — just load a pretrained host, build passphrase
prompts from the same generator used for Stage-0 data, generate
continuations, and score them with `src.pretrained.rewards`.

Output per (host, tier) cell:
    - n examples
    - mean passphrase_ratio (0..1)
    - mean order_correctness (0..1)
    - mean exact_match (0..1)
    - fraction of runs with score >= 0.8

Decides the starting tier for Stage-0 training: whichever difficulty
the chosen dev host already fails at. If the baseline is already > 0.9
on tiers 0-3 at practical context lengths, switch to a weaker host or
push to harder tiers.

Usage:
    python scripts/bench_llama_passphrase_baseline.py \\
        --host llama-1b,tinyllama,smollm2-360m \\
        --tiers 0,1,2,3,4 --num-examples 32 \\
        --out bench/baseline_passphrase.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pretrained.passphrase_data import PHRASE_POOL
from src.pretrained.rewards import (
    score_exact_match,
    score_order_correctness,
    score_passphrase_ratio,
)
from scripts.build_passphrase_data import build_example


# Tier definitions for the sweep. Each tier is (n_phrases, target_length).
# Extend as needed; anything past tier 4 saturates our 60-phrase pool for
# distinctness.
TIERS: dict[int, tuple[int, int]] = {
    0: (1, 128),     # trivial — single phrase in short filler
    1: (2, 256),     # easy — two phrases
    2: (3, 512),     # medium — three phrases (Stage-0 default)
    3: (5, 1024),    # hard — five phrases, 1K context
    4: (8, 2048),    # stretch — eight phrases, 2K context
}

# HF model IDs per host name. Mirrors PretrainedConfig factory methods
# but avoids loading the memory-wrapper (we only need the bare model).
HOST_MODEL_ID: dict[str, str] = {
    "llama-1b":    "meta-llama/Llama-3.2-1B",
    "llama-3b":    "meta-llama/Llama-3.2-3B",
    "tinyllama":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
}


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_host(host_name: str, dtype: torch.dtype = torch.bfloat16):
    """Load just the HF model + tokenizer. No wrapper."""
    if host_name not in HOST_MODEL_ID:
        known = ", ".join(sorted(HOST_MODEL_ID))
        raise ValueError(f"unknown host {host_name!r}; known: {known}")
    model_id = HOST_MODEL_ID[host_name]
    print(f"  loading {model_id} in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to("cuda")
    model.train(False)  # inference mode — pytorch term, no dropout etc.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    gen_length: int,
    temperature: float,
    seed: int,
) -> str:
    """Generate `gen_length` tokens from the prompt and decode.

    Uses HF's `generate` with KV cache — no manual unroll needed for a
    no-training baseline.
    """
    torch.manual_seed(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    do_sample = temperature > 0.0
    out = model.generate(
        input_ids,
        max_new_tokens=gen_length,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=0.95 if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Slice off the prompt.
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _score_one(pred_text: str, phrases: list[str], answer_text: str) -> dict:
    """Run all three rewards on one prediction."""
    return {
        "ratio": score_passphrase_ratio(pred_text, phrases),
        "order": score_order_correctness(pred_text, phrases),
        "exact": score_exact_match(pred_text, answer_text),
    }


def _measure_cell(
    model,
    tokenizer,
    host_name: str,
    tier: int,
    n_phrases: int,
    target_length: int,
    num_examples: int,
    rng_seed: int,
    temperature: float,
) -> dict:
    """Build `num_examples` prompts at this (tier, context), generate, score."""
    rng = random.Random(rng_seed + tier * 7919)
    gen_length_budget = max(32, 16 * n_phrases)  # enough room for "<p>. "*N

    scores = {"ratio": [], "order": [], "exact": []}
    samples: list[dict] = []
    t0 = time.time()

    for i in range(num_examples):
        ex = build_example(
            tokenizer, rng,
            n_phrases=n_phrases,
            target_length=target_length,
        )
        pred_text = _generate(
            model, tokenizer,
            prompt_ids=ex["prompt_ids"],
            gen_length=gen_length_budget,
            temperature=temperature,
            seed=rng_seed + tier * 7919 + i,
        )
        s = _score_one(pred_text, ex["passphrases"], ex["answer_text"])
        for k, v in s.items():
            scores[k].append(v)
        # Keep a couple of qualitative examples for eyeballing.
        if i < 3:
            samples.append({
                "passphrases": ex["passphrases"],
                "gt_answer": ex["answer_text"],
                "pred": pred_text[:300],
                "scores": s,
            })

    ms_per_ex = (time.time() - t0) * 1000 / max(1, num_examples)

    def _stats(xs):
        if not xs:
            return {"mean": None, "solved_frac": None}
        mean = sum(xs) / len(xs)
        solved = sum(1 for x in xs if x >= 0.8) / len(xs)
        return {"mean": round(mean, 4), "solved_frac": round(solved, 4)}

    return {
        "host": host_name,
        "tier": tier,
        "n_phrases": n_phrases,
        "target_length": target_length,
        "num_examples": num_examples,
        "temperature": temperature,
        "ms_per_example": round(ms_per_ex, 1),
        "ratio":   _stats(scores["ratio"]),
        "order":   _stats(scores["order"]),
        "exact":   _stats(scores["exact"]),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="llama-1b,llama-3b",
                        help=f"Comma-separated from: {', '.join(sorted(HOST_MODEL_ID))}")
    parser.add_argument("--tiers", type=str, default="0,1,2,3,4",
                        help="Comma-separated tier ids from TIERS")
    parser.add_argument("--num-examples", type=int, default=32,
                        help="Examples per (host, tier) cell")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy; >0 = sampled (reported stat is noisier)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--out", type=Path, default=Path("bench/baseline_passphrase.jsonl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; this benchmark requires a GPU.")

    hosts = [m.strip() for m in args.host.split(",") if m.strip()]
    tiers = _parse_int_list(args.tiers)
    for t in tiers:
        if t not in TIERS:
            raise ValueError(f"unknown tier {t}; valid: {sorted(TIERS)}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    assert max(n for (n, _) in TIERS.values()) <= len(PHRASE_POOL), (
        f"PHRASE_POOL has {len(PHRASE_POOL)} phrases; "
        f"max tier N={max(n for (n, _) in TIERS.values())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")
    print(f"Hosts: {hosts}  Tiers: {tiers}  "
          f"Examples/cell: {args.num_examples}  Temp: {args.temperature}")
    print()
    print(f"{'host':>14}  {'tier':>4}  {'N':>2}  {'T_ctx':>6}  "
          f"{'ratio':>6}  {'order':>6}  {'exact':>6}  {'solved@ratio':>12}  {'ms/ex':>6}")
    print("-" * 92)

    with args.out.open("w") as f:
        for host_name in hosts:
            model, tokenizer = _load_host(host_name, dtype=dtype)
            try:
                for tier in tiers:
                    n, L = TIERS[tier]
                    r = _measure_cell(
                        model, tokenizer,
                        host_name=host_name, tier=tier,
                        n_phrases=n, target_length=L,
                        num_examples=args.num_examples,
                        rng_seed=args.seed,
                        temperature=args.temperature,
                    )
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(f"{host_name:>14}  {tier:>4}  {n:>2}  {L:>6}  "
                          f"{r['ratio']['mean']:>6.3f}  "
                          f"{r['order']['mean']:>6.3f}  "
                          f"{r['exact']['mean']:>6.3f}  "
                          f"{r['ratio']['solved_frac']:>12.2f}  "
                          f"{r['ms_per_example']:>6.0f}")
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
