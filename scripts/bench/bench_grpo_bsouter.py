#!/usr/bin/env python3
"""Bench Phase 2 GRPO at BS_outer ∈ {1, 2, 4, 8} on real narrativeqa prompts.

For each BS_outer, runs:
  - 1 warmup step (compile/cold-start)
  - 3 timed steps
Then reports:
  - mean s/step (cuda.synchronize at boundaries)
  - peak VRAM
  - rollout tok/sec ≈ M*K*max_new_tokens / s
  - rough GPU util via parallel `nvidia-smi --query-gpu=utilization.gpu` poll

Uses kl_coef=0 (no reference policy, our new default).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training.phase2 import Phase2Trainer  # noqa: E402


class GpuUtilSampler:
    """Background thread polling nvidia-smi every ~100ms during a context."""

    def __init__(self, interval_s: float = 0.1):
        self.interval = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                val = int(out.stdout.strip().split("\n")[0])
                self.samples.append(val)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._stop.clear()
        self.samples = []
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)

    @property
    def mean(self) -> float:
        return sum(self.samples) / max(len(self.samples), 1)

    @property
    def median(self) -> float:
        s = sorted(self.samples)
        if not s:
            return 0.0
        return s[len(s) // 2]


def load_prompts_and_meta(
    parquet_path: str, n: int, min_prompt_len: int, tokenizer,
) -> tuple[list, list]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    rows = [r for r in rows if len(r["prompt_ids"]) >= min_prompt_len]
    rows = rows[:n]
    assert len(rows) >= n, f"only got {len(rows)} rows >= {min_prompt_len} tokens"

    prompts, metas = [], []
    for r in rows:
        pid = torch.tensor(r["prompt_ids"], dtype=torch.int64).unsqueeze(0).to("cuda")
        prompts.append(pid)
        gold_str = tokenizer.decode(r["gold_ids"], skip_special_tokens=True)
        metas.append({
            "reward_kind": r["reward_kind"],
            "gold": gold_str,
            "meta": r.get("meta", {}) or {},
        })
    return prompts, metas


def bench_one(bs_outer: int, model, optim, tokenizer, prompts_all, metas_all,
              num_samples: int = 8, max_new_tokens: int = 256) -> dict:
    trainer = Phase2Trainer(
        model, optim, grad_clip=1.0, clip_eps=0.2, kl_coef=0.0,
    )
    M = bs_outer
    K = num_samples

    # Warmup
    print(f"  warmup (BS_outer={M}) ...", flush=True)
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
    print(f"  warmup took {time.time()-t_warm:.1f}s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    # Steady-state
    n_iter = 3
    t0 = time.time()
    with GpuUtilSampler(interval_s=0.1) as sampler:
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
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    rollout_tok = M * K * max_new_tokens
    return {
        "BS_outer": M,
        "s_per_step": dt,
        "peak_vram_gb": peak_gb,
        "rollout_tok_per_step": rollout_tok,
        "rollout_tok_per_sec": rollout_tok / dt,
        "gpu_util_mean": sampler.mean,
        "gpu_util_median": sampler.median,
        "gpu_samples": len(sampler.samples),
    }


def main():
    torch.manual_seed(0)
    cfg = TrajMemConfig.medium()  # now defaults D=1024
    print(f"Config: N={cfg.N}, D_concept={cfg.D_concept}, J={cfg.J}, "
          f"effective_lm_context={cfg.effective_lm_context}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print("Loading model...", flush=True)
    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B")
    ck_path = "outputs/wave2_v2/ckpt.pt"
    if Path(ck_path).exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        # Wave2 ckpt was trained with D=256; D=1024 is now default, so old
        # ckpt may not load cleanly. strict=False — we just need a warm enough
        # model that AR rollouts produce sensible text.
        # If shape mismatch: skip ckpt entirely (cold-start the trajectory
        # memory; Llama is still pretrained).
        try:
            missing, unexpected = model.load_state_dict(
                ck["model_state_dict"], strict=False,
            )
            print(f"  loaded ckpt (missing={len(missing)}, unexpected={len(unexpected)})",
                  flush=True)
        except Exception as e:
            print(f"  ckpt load failed ({type(e).__name__}); cold-start", flush=True)
    model = model.to("cuda")
    model.train(True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )

    # Need enough prompts for BS_outer=8 with 3 timed steps + 1 warmup = 4*8=32
    print("Loading prompts...", flush=True)
    prompts, metas = load_prompts_and_meta(
        "data/wave3/narrativeqa.train.parquet", n=64,
        min_prompt_len=2048, tokenizer=tokenizer,
    )
    print(f"  got {len(prompts)} prompts >= 2048 tokens", flush=True)

    print()
    print(f"{'BS_outer':>9} {'s/step':>8} {'tok/s':>10} {'peak GB':>9} "
          f"{'GPU%avg':>8} {'GPU%med':>8} {'samples':>8}")
    print("-" * 72)
    results = []
    for M in [1, 2, 4, 8]:
        try:
            r = bench_one(M, model, optim, tokenizer, prompts, metas)
            results.append(r)
            print(f"{r['BS_outer']:>9} {r['s_per_step']:>8.2f} "
                  f"{r['rollout_tok_per_sec']:>10.0f} {r['peak_vram_gb']:>9.2f} "
                  f"{r['gpu_util_mean']:>8.1f} {r['gpu_util_median']:>8.0f} "
                  f"{r['gpu_samples']:>8d}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{M:>9}   OOM  ({str(e)[:60]})", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
