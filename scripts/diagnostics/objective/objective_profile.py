"""Objective×arm profiling sweep (2026-07-03) — is everything implemented efficiently?

Times the REAL training step (bf16 autocast + grad checkpointing + AdamW) for every
{plain, contrastive, trajectory} × {lm4, custom} cell on realistic batch shapes
(MAE B=8 T=1024 → 4 write windows; QA B=8 ctx=1024), with a per-segment breakdown:

  enc_fwd          encoder-only pass (GradCache pass 0)
  full_fwd         plain mode's single fused encoder+decoder forward
  dec_pass_nograd  no-grad decoder scoring passes (GradCache pass 1 + GRPO rollout reads)
  dec_pass_grad    grad decoder passes (GradCache pass 2 forwards)
  bwd+misc         residual: backwards, W assembly, rollout building, optimizer

Peak VRAM per cell. Median of 3 timed steps after 2 warmups.

Usage: .venv/bin/python scripts/diagnostics/objective/objective_profile.py
"""
from __future__ import annotations

import statistics
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.memory.config import ReprConfig  # noqa: E402
from src.memory.model import ReprLearningModel  # noqa: E402
from src.memory.training import _grad_cached_objective_step  # noqa: E402

DEV = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"
VOCAB = 49152
B = 8


def mk_model(arm):
    cfg = ReprConfig()
    cfg.llama_model = BACKBONE
    cfg.d_llama = 576
    cfg.pad_token_id = 0
    cfg.use_llama_lora = True
    cfg.slotgraph3_n_nodes = 16
    cfg.slotgraph3_gate_ids = True
    cfg.slotgraph3_st_leak = True
    cfg.slotgraph3_route_key = "node"
    cfg.slotgraph3_edge_state = "matrix"
    cfg.slotgraph3_write = arm
    if arm == "lm":
        cfg.slotgraph3_write_layers = 4
    cfg.objective_coef = 0.5
    m = ReprLearningModel(cfg, variant="slotgraph3_baseline", llama_model=None).to(DEV)
    m.train()
    return m


def mae_batch(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return types.SimpleNamespace(
        context_ids=torch.randint(1, VOCAB, (B, 1024), generator=g).to(DEV),
        context_mask=torch.ones(B, 1024, dtype=torch.bool, device=DEV), k_slots=None)


def qa_batch(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return types.SimpleNamespace(
        context_ids=torch.randint(1, VOCAB, (B, 1024), generator=g).to(DEV),
        context_mask=torch.ones(B, 1024, dtype=torch.bool, device=DEV),
        question_ids=torch.randint(1, VOCAB, (B, 16), generator=g).to(DEV),
        question_mask=torch.ones(B, 16, dtype=torch.bool, device=DEV),
        answer_ids=torch.randint(1, VOCAB, (B, 8), generator=g).to(DEV),
        answer_mask=torch.ones(B, 8, dtype=torch.bool, device=DEV),
        answer_content_mask=torch.ones(B, 8, dtype=torch.bool, device=DEV),
    )


class SegTimer:
    def __init__(self):
        self.acc = defaultdict(float)

    @contextmanager
    def seg(self, name):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.acc[name] += time.perf_counter() - t0


def wrap(model, timer):
    orig = model.compute_loss

    def wrapped(batch, **kw):
        if kw.get("encoder_only"):
            name = "enc_fwd"
        elif kw.get("memory_override") is not None:
            name = "dec_pass_nograd" if not torch.is_grad_enabled() else "dec_pass_grad"
        else:
            name = "full_fwd"
        with timer.seg(name):
            return orig(batch, **kw)
    model.compute_loss = wrapped
    return orig


def one_step(model, opt, batch, mode, cfg):
    opt.zero_grad(set_to_none=True)
    if mode == "plain":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(batch, window_size=1024)
        out["loss"].backward()
    else:
        cfg.objective_mode = mode
        _grad_cached_objective_step(model, batch, cfg, 1024)
    opt.step()


def profile_cell(model, mode, task, batch_fn):
    cfg = model.cfg
    model.task_mode = task
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    timer = SegTimer()
    orig = wrap(model, timer)
    try:
        for i in range(2):                                    # warmup
            one_step(model, opt, batch_fn(seed=100 + i), mode, cfg)
        torch.cuda.reset_peak_memory_stats()
        totals = []
        for i in range(3):
            timer.acc.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_step(model, opt, batch_fn(seed=i), mode, cfg)
            torch.cuda.synchronize()
            tot = time.perf_counter() - t0
            totals.append((tot, dict(timer.acc)))
        peak = torch.cuda.max_memory_allocated() / 2**30
    finally:
        model.compute_loss = orig
        del opt
        torch.cuda.empty_cache()
    med_tot, med_segs = sorted(totals, key=lambda x: x[0])[1]
    resid = med_tot - sum(med_segs.values())
    segs = {k: v * 1e3 for k, v in med_segs.items()}
    segs["bwd+misc"] = resid * 1e3
    return med_tot * 1e3, segs, peak


def main():
    torch.manual_seed(0)
    print(f"{'cell':34s} {'ms/step':>9s} {'peak GB':>8s}   segments (ms)")
    print("-" * 110)
    results = {}
    for arm in ("lm", "custom"):
        model = mk_model(arm)
        for mode in ("plain", "contrastive", "trajectory"):
            for task, bf in (("masked_reconstruction", mae_batch), ("qa", qa_batch)):
                if mode == "trajectory" and arm == "custom" and task == "qa":
                    pass  # keep all cells — trajectory works on both arms
                tag = f"{mode:12s} × {arm:6s} × {'mae' if task.startswith('m') else 'qa '}"
                try:
                    tot, segs, peak = profile_cell(model, mode, task, bf)
                except RuntimeError as e:
                    print(f"{tag:34s} FAILED: {str(e)[:60]}")
                    continue
                seg_str = "  ".join(f"{k}={v:7.1f}" for k, v in sorted(segs.items()) if v > 0.5)
                print(f"{tag:34s} {tot:9.1f} {peak:8.2f}   {seg_str}")
                results[(mode, arm, task)] = (tot, segs, peak)
        del model
        torch.cuda.empty_cache()

    # efficiency ratios that matter
    print("\nratios:")
    for arm in ("lm", "custom"):
        for task in ("masked_reconstruction", "qa"):
            p = results.get(("plain", arm, task))
            c = results.get(("contrastive", arm, task))
            t = results.get(("trajectory", arm, task))
            if p and c:
                print(f"  {arm:6s}/{task[:3]}: contrastive/plain = {c[0]/p[0]:.2f}x"
                      + (f", trajectory/plain = {t[0]/p[0]:.2f}x" if t else "")
                      + f"   (theory: contrastive ≈ (enc + {B}·dec_ng + {B}·dec_g·~2)/(enc+dec·~3))")


if __name__ == "__main__":
    main()
