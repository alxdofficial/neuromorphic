"""Throughput bench: vanilla Llama forward (no grad) vs memory graph alone.

Measures the max per-step forward throughput to answer "does squeezing
memory graph performance matter if Llama itself is the bottleneck during
phase-2 autoregressive rollouts?"
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM


def bench_llama(model_name: str, bs: int, T: int, dtype: torch.dtype,
                warmup: int = 3, measure: int = 10) -> tuple[float, float]:
    """Parallel forward throughput — one shot over T tokens, no grad."""
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype).to(device)
    model.train(False)
    vocab = model.config.vocab_size

    def step():
        ids = torch.randint(0, vocab, (bs, T), device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            _ = model(input_ids=ids).logits
        torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    t0 = time.time()
    for _ in range(measure):
        step()
    dt = time.time() - t0
    tokens = bs * T * measure
    del model
    torch.cuda.empty_cache()
    return tokens / dt, dt / measure * 1000


def bench_llama_autoregressive(model_name: str, bs: int, T_pre: int,
                                T_gen: int, dtype: torch.dtype,
                                warmup: int = 1, measure: int = 3) -> tuple[float, float]:
    """Autoregressive gen with KV cache — the actual phase-2 rollout shape.
    Returns (tok/s measured over generated tokens, ms/step)."""
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype).to(device)
    model.train(False)
    vocab = model.config.vocab_size

    def gen_one():
        ids = torch.randint(0, vocab, (bs, T_pre), device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            current = ids[:, -1:]
            for _ in range(T_gen):
                out = model(input_ids=current, past_key_values=past, use_cache=True)
                past = out.past_key_values
                # Greedy for simplicity (doesn't affect throughput meaningfully)
                current = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()

    for _ in range(warmup):
        gen_one()
    t0 = time.time()
    for _ in range(measure):
        gen_one()
    dt = time.time() - t0
    # Report throughput over generated tokens only (the interesting bit).
    tokens = bs * T_gen * measure
    del model
    torch.cuda.empty_cache()
    return tokens / dt, dt / measure * 1000


def bench_memory(bs: int, T: int) -> tuple[float, float]:
    """Memory graph alone — approximate no-grad forward throughput."""
    import sys
    sys.path.insert(0, ".")
    from src.model.config import Config as MemoryConfig
    from src.model.memory import MemoryGraph

    device = torch.device("cuda")
    cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    mem = MemoryGraph(cfg).to(device)
    mem.train(False)
    vocab = 128256

    # Need an adapter-like object for the surprise signal. Use a zero-bias
    # Linear for proj_down + the Llama norm. Simplest: use a toy adapter.
    class _ToyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2048, vocab, bias=False).to(device).to(torch.bfloat16)
            self.proj_down = None
            class _Norm:
                def __init__(s):
                    s.weight = torch.ones(2048, device=device, dtype=torch.bfloat16)
                    s.bias = None
            self.ln_final = _Norm()
        def mem_head_logits(self, x):
            return self.lm_head(x.to(torch.bfloat16))
    toy = _ToyLM()

    def step():
        ids = torch.randint(0, vocab, (bs, T), device=device)
        H = torch.randn(bs, T, 2048, device=device, dtype=torch.bfloat16)
        mem.initialize_states(bs, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = mem.forward_segment(H, ids, toy, use_rmsnorm=True,
                                     rms_eps=1e-5, phase="phase1")
        torch.cuda.synchronize()

    for _ in range(3):
        step()
    t0 = time.time()
    N = 10
    for _ in range(N):
        step()
    dt = time.time() - t0
    return bs * T * N / dt, dt / N * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["1b", "3b", "both"], default="both")
    args = p.parse_args()

    models = (["meta-llama/Llama-3.2-1B"] if args.model == "1b"
              else ["meta-llama/Llama-3.2-3B"] if args.model == "3b"
              else ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"])
    dtype = torch.bfloat16

    header = f"{'Model':<22s} {'mode':<24s} {'BS':>4s} {'T':>6s} {'tok/s':>10s} {'ms/step':>10s}"
    print(header); print("-" * len(header))

    for m in models:
        name = m.split("/")[-1]
        # Parallel — single-shot forward of BS=16 × T=128. This is what the
        # phase-1 training loop sees per step.
        try:
            toks, ms = bench_llama(m, bs=16, T=128, dtype=dtype)
            print(f"{name:<22s} {'parallel (BS=16,T=128)':<24s} {16:>4d} {128:>6d} "
                  f"{toks/1000:>9.1f}K {ms:>9.1f}")
        except Exception as e:
            print(f"{name} parallel: ERROR {type(e).__name__}: {e}")
        # Autoregressive gen with KV cache — what phase-2 rollouts look like.
        try:
            toks, ms = bench_llama_autoregressive(
                m, bs=8, T_pre=128, T_gen=128, dtype=dtype)
            print(f"{name:<22s} {'AR gen K=8 pre=128 gen=128':<24s} {8:>4d} {128:>6d} "
                  f"{toks/1000:>9.1f}K {ms:>9.1f}")
        except Exception as e:
            print(f"{name} AR gen: ERROR {type(e).__name__}: {e}")

    print()
    print(header); print("-" * len(header))
    try:
        toks, ms = bench_memory(bs=16, T=128)
        print(f"{'Memory graph (tier_a)':<22s} {'parallel (BS=16,T=128)':<24s} "
              f"{16:>4d} {128:>6d} {toks/1000:>9.1f}K {ms:>9.1f}")
    except Exception as e:
        print(f"Memory parallel: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
