#!/usr/bin/env python3
"""Vanilla Llama-3.2-1B AR decode throughput on this GPU.

Pure inference (no trajectory memory, no reward, no training, no grad).
Same prompt-length + max_new + BS as our GRPO bench so the numbers
are directly comparable.

Reports tok/s aggregate at BS in {1, 8, 32, 64} with HF DynamicCache.
This is the apples-to-apples upper bound for what our GRPO step could
do if all the side-car / training / reward overhead vanished.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def bench_bs(model, BS: int, prompt_len: int = 2048,
             max_new: int = 256, n_iter: int = 3) -> dict:
    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (BS, prompt_len), device=device)

    with torch.no_grad():
        cache = DynamicCache()
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(8):
            out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            cache = DynamicCache()
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(max_new - 1):
                out = model(
                    input_ids=next_tok, past_key_values=cache, use_cache=True,
                )
                cache = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    peak = torch.cuda.max_memory_allocated() / 1e9
    rollout_tok = BS * max_new
    return {
        "BS": BS, "s_per_step": dt, "peak_gb": peak,
        "tok_per_sec": rollout_tok / dt,
        "per_sample_s": dt / BS,
    }


def main():
    print("Loading Llama-3.2-1B (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", dtype=torch.bfloat16,
    ).to("cuda")
    model.train(False)

    print(f"\n{'BS':>4} {'s/step':>8} {'tok/s':>10} {'peak GB':>9} {'per-sample s':>14}")
    print("-" * 50)
    for BS in [1, 8, 32, 64]:
        try:
            r = bench_bs(model, BS)
            print(f"{r['BS']:>4} {r['s_per_step']:>8.2f} {r['tok_per_sec']:>10.0f} "
                  f"{r['peak_gb']:>9.2f} {r['per_sample_s']:>14.3f}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{BS:>4}   OOM  ({str(e)[:60]})", flush=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
