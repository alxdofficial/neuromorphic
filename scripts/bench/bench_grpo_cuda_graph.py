#!/usr/bin/env python3
"""A/B bench: DynamicCache vs StaticCache vs StaticCache+CUDA-graph.

Runs Phase 2 GRPO step_batched at BS_outer in {1, 4, 8}, K=8, max_new=256
under three configurations to isolate the gain from each optimization.

Output: outputs/bench_grpo_cuda_graph.log
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training.phase2 import Phase2Trainer  # noqa: E402


def load_prompts(path: str, n: int, min_prompt_len: int, tokenizer):
    table = pq.read_table(path)
    rows = [r for r in table.to_pylist() if len(r["prompt_ids"]) >= min_prompt_len]
    rows = rows[:n]
    assert len(rows) == n, f"need {n} prompts, got {len(rows)}"
    prompts = [torch.tensor(r["prompt_ids"], dtype=torch.int64).unsqueeze(0).to("cuda") for r in rows]
    metas = [
        {
            "reward_kind": r["reward_kind"],
            "gold": tokenizer.decode(r["gold_ids"], skip_special_tokens=True),
            "meta": r.get("meta", {}) or {},
        } for r in rows
    ]
    return prompts, metas


def bench_one(model, optim, tokenizer, prompts_all, metas_all,
              BS_outer: int, *, use_static_cache: bool, use_cuda_graph: bool,
              num_samples: int = 8, max_new_tokens: int = 256) -> dict:
    trainer = Phase2Trainer(
        model, optim, grad_clip=1.0, clip_eps=0.2, kl_coef=0.0,
        use_static_cache=use_static_cache,
        use_cuda_graph=use_cuda_graph,
    )
    M = BS_outer
    K = num_samples

    # Warmup (first step pays capture cost when use_cuda_graph=True)
    t_warm = time.time()
    prompts = prompts_all[:M]; metas = metas_all[:M]
    if M == 1:
        trainer.step(prompts[0], num_samples=K, max_new_tokens=max_new_tokens,
                     reward_kind=metas[0]["reward_kind"], gold=metas[0]["gold"],
                     meta=metas[0]["meta"], tokenizer=tokenizer)
    else:
        trainer.step_batched(prompts, metas, num_samples=K,
                             max_new_tokens=max_new_tokens, tokenizer=tokenizer)
    torch.cuda.synchronize()
    t_warm = time.time() - t_warm

    torch.cuda.reset_peak_memory_stats()
    n_iter = 3
    t0 = time.time()
    for i in range(n_iter):
        offset = M * (i + 1)
        prompts_i = prompts_all[offset:offset + M]
        metas_i = metas_all[offset:offset + M]
        if M == 1:
            trainer.step(prompts_i[0], num_samples=K, max_new_tokens=max_new_tokens,
                         reward_kind=metas_i[0]["reward_kind"], gold=metas_i[0]["gold"],
                         meta=metas_i[0]["meta"], tokenizer=tokenizer)
        else:
            trainer.step_batched(prompts_i, metas_i, num_samples=K,
                                 max_new_tokens=max_new_tokens, tokenizer=tokenizer)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    peak = torch.cuda.max_memory_allocated() / 1e9
    rollout_tok = M * K * max_new_tokens
    return {
        "BS_outer": M, "s_per_step": dt, "peak_gb": peak,
        "warmup_s": t_warm,
        "tok_per_sec": rollout_tok / dt,
        "per_sample_s": dt / (M * K),
    }


def main():
    torch.manual_seed(0)
    cfg = TrajMemConfig.medium()
    print(f"Config: N={cfg.N}, D_concept={cfg.D_concept}, J={cfg.J}, "
          f"effective_lm_context={cfg.effective_lm_context}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print("Loading model...", flush=True)
    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B").to("cuda")
    model.train(True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )

    print("Loading prompts...", flush=True)
    prompts, metas = load_prompts(
        "data/wave3/narrativeqa.train.parquet", n=64,
        min_prompt_len=2048, tokenizer=tokenizer,
    )

    print()
    configs = [
        ("dyn_cache", False, False),       # Baseline: DynamicCache, no CUDA graph
        ("static_cache", True, False),     # StaticCache only
        ("static_cache+cg", True, True),   # StaticCache + CUDA graph
    ]
    print(f"{'BS_outer':>9} {'mode':>16} {'warm s':>8} {'s/step':>8} "
          f"{'tok/s':>10} {'peak GB':>9} {'per-sample':>12}")
    print("-" * 84)
    for M in [1, 4, 8]:
        for name, usc, ucg in configs:
            try:
                r = bench_one(
                    model, optim, tokenizer, prompts, metas, M,
                    use_static_cache=usc, use_cuda_graph=ucg,
                )
                print(f"{r['BS_outer']:>9} {name:>16} "
                      f"{r['warmup_s']:>8.2f} {r['s_per_step']:>8.2f} "
                      f"{r['tok_per_sec']:>10.0f} {r['peak_gb']:>9.2f} "
                      f"{r['per_sample_s']:>12.3f}", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"{M:>9} {name:>16}  OOM ({str(e)[:50]})", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{M:>9} {name:>16}  ERROR: {type(e).__name__}: "
                      f"{str(e)[:80]}", flush=True)
                torch.cuda.empty_cache()
                gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
