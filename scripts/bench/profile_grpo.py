#!/usr/bin/env python3
"""Profile a real GRPO step. Identifies where the 77%-idle time goes.

Sets up Phase2Trainer like train_wave3 does, loads a few real
narrativeqa prompts, runs a warmup step, then profiles 3 steady-state
steps with torch.profiler. Prints per-op CPU/CUDA time tables.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import pyarrow.parquet as pq
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trajectory_memory.config import TrajMemConfig  # noqa
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa
from src.trajectory_memory.training.phase2 import Phase2Trainer  # noqa


def main():
    print("=== GRPO step profiler ===", flush=True)

    # 1. Setup the model
    cfg = TrajMemConfig.medium()  # NB: now defaults to D_concept=1024
    # Override back to D=256 for fair comparison to our earlier benches.
    cfg.D_concept = 256
    cfg.validate()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print("Loading model + Wave 2 ckpt...", flush=True)
    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B")
    ck = torch.load("outputs/wave2_v2/ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to("cuda")
    model.train(True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )
    trainer = Phase2Trainer(
        model, optim, grad_clip=1.0,
        clip_eps=0.2, kl_coef=0.0,  # kl_coef=0 to skip reference policy
    )

    # 2. Load a few prompts from narrativeqa
    table = pq.read_table("data/wave3/narrativeqa.train.parquet").slice(0, 8)
    rows = table.to_pylist()
    prompts = []
    metas = []
    for r in rows:
        pid = torch.tensor(r["prompt_ids"], dtype=torch.int64).unsqueeze(0).to("cuda")
        prompts.append(pid)
        gold_str = tokenizer.decode(r["gold_ids"], skip_special_tokens=True)
        metas.append({
            "reward_kind": r["reward_kind"],
            "gold": gold_str,
            "meta": {},  # narrativeqa meta_json may not be needed for bert_cosine
        })

    # 3. Warmup (compile cold-start)
    print("Warmup step (compile)...", flush=True)
    t0 = time.time()
    metrics = trainer.step(
        prompts[0], num_samples=8, max_new_tokens=256,
        reward_kind=metas[0]["reward_kind"], gold=metas[0]["gold"],
        meta=metas[0]["meta"], tokenizer=tokenizer,
    )
    torch.cuda.synchronize()
    mean_r = sum(metrics.rewards) / max(len(metrics.rewards), 1)
    print(f"  warmup step took {time.time()-t0:.1f}s, mean reward={mean_r:.3f}",
          flush=True)

    # 4. Profile 3 steady-state steps
    print("\nProfiling 3 steady-state steps...", flush=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for i in range(3):
            with record_function(f"grpo_step_{i}"):
                trainer.step(
                    prompts[(i+1) % len(prompts)],
                    num_samples=8, max_new_tokens=256,
                    reward_kind=metas[(i+1) % len(metas)]["reward_kind"],
                    gold=metas[(i+1) % len(metas)]["gold"],
                    meta=metas[(i+1) % len(metas)]["meta"],
                    tokenizer=tokenizer,
                )
            torch.cuda.synchronize()

    # 5. Reports
    print("\n=== Top 25 ops by CUDA time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))

    print("\n=== Top 25 ops by CPU time ===")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=25, max_name_column_width=60))

    # 6. Save chrome trace
    out_trace = "outputs/grpo_profile.json"
    Path(out_trace).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(out_trace)
    print(f"\nChrome trace saved: {out_trace}")
    print(f"  Open at: chrome://tracing  or  https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
