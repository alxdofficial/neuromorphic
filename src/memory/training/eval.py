"""Validation: per-task mixed eval, single-set eval with the REAL/SHUF/OFF binding gate,
and the continuation early-token loss.

Extracted verbatim from ``scripts/train/train.py`` (harness reorg phase 2). The ONLY change from
the original bodies: ``run_mixed_val`` routes task_mode via ``mixes.task_mode(t)`` instead of the
deleted module-level ``MIXED_TASK_MODE`` dict (same value).
"""
from __future__ import annotations

import torch

from src.memory.data import mixes

from .utils import to_device

CONT_EARLY_TOKENS = 16   # continuation early-token loss = mean over the first N predict positions


def run_mixed_val(model, mixed_tasks, val_sets, device, n_batches, window_size,
                  gate_batches: int = 0) -> dict:
    """Evaluate ALL mixed val sets, switching model.task_mode per task so each
    routes to its own loss path. Returns {task: per-task metric dict}. The
    continuation set additionally reports an early-token loss (mean CE over the
    first CONT_EARLY_TOKENS predict positions, since the late positions are
    increasingly local-autoregressive and dilute the memory signal)."""
    prev_mode = getattr(model, "task_mode", None)
    per_task = {}
    for t in mixed_tasks:
        model.task_mode = mixes.task_mode(t)
        # gate_batches>0 enables the REAL/SHUF/OFF binding gate (example-specificity diagnostic) on the
        # first `gate_batches` val batches; 0 (default) skips it (it triples eval cost). Set via --mixed-gate-batches.
        vm = run_val(model, val_sets[t], device, n_batches, window_size, gate_batches=gate_batches)
        if t == "continuation":
            vm["val_cont_early_loss"] = _continuation_early_loss(
                model, val_sets[t], device, n_batches, window_size)
        per_task[t] = vm
    model.task_mode = prev_mode
    return per_task


@torch.no_grad()
def _continuation_early_loss(model, val_set, device, n_batches, window_size) -> float:
    """Mean CE over the FIRST CONT_EARLY_TOKENS answer positions of each continuation
    example (predict_len=64; the early tokens depend most on the compressed memory).
    Implemented by zeroing answer_content_mask past the early window before the
    generic compute_loss, so loss_recon is restricted to those positions."""
    import dataclasses
    model.train(False)
    losses = []
    for i, batch in enumerate(val_set):
        if i >= n_batches:
            break
        torch.manual_seed(20260527 + i)
        batch = to_device(batch, device)
        # restrict the content mask to the first CONT_EARLY_TOKENS valid positions
        cm = batch.answer_content_mask.clone()
        cm[:, CONT_EARLY_TOKENS:] = False
        early = dataclasses.replace(batch, answer_content_mask=cm)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(early, window_size=window_size)
        losses.append(float(out["loss_recon"]))
    model.train(True)
    return sum(losses) / max(len(losses), 1)


def run_val(model, val_set, device, n_batches: int, window_size: int,
            gate_batches: int = 8) -> dict:
    """Eval on a fixed val_set (list of batches from materialize_val_set).

    n_batches caps the iteration in case val_set has more batches than we want
    to spend time on (e.g., subsampling for a fast in-training eval). To use the
    full set, pass n_batches >= len(val_set).

    gate_batches: on the FIRST `gate_batches` val batches, additionally compute
    the SHUF (rolled-batch memory) and OFF (zero memory) controls so each eval
    reports the binding gate REAL ≪ SHUF ≪ OFF. Capped because each control is a
    full extra encode+decode (3× cost). SHUF is skipped for a B==1 batch.
    """
    model.train(False)
    losses, accs, per_fam_stats = [], [], {}
    babi_em_hits, babi_em_n = 0.0, 0   # bAbI exact-match (teacher-forced answer span)
    # REAL/SHUF/OFF binding gate, accumulated over the first gate_batches.
    shuf_abs, shuf_gap, off_abs, off_gap = [], [], [], []
    last_graph_eval: dict[str, float] = {}  # graph read+write collapse canaries
    last_biomem_eval: dict[str, float] = {}  # biomem edge/state saturation + leak/eta canaries
    last_slotgraph_eval: dict[str, float] = {}  # slotgraph edge-frac / src-dst entropy / mem-rank canaries
    last_vqicae_eval: dict[str, float] = {}  # vqicae codebook perplexity / active-code / batch-used canaries
    # Deterministic-eval seed: the soft_pointer_graph chunk-fresh init samples
    # fresh noise per call → seeding torch RNG per batch makes eval reproducible.
    # In eval mode (model.train(False)) dropout is off so this is safe.
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    for i, batch in enumerate(val_set):
        if i >= n_batches:
            break
        torch.manual_seed(20260527 + i)  # batch-dependent deterministic seed
        batch = to_device(batch, device)
        # no_grad: huge memory win (~18→4 GiB at B=12). Val never backprops,
        # so the autograd graph was being built for nothing under autocast.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(batch, window_size=window_size)
        losses.append(float(out["loss_recon"]))
        accs.append(float(out["top1_acc"]))
        # REAL/SHUF/OFF binding gate on the first gate_batches (3× cost → capped).
        # Positive gap = memory helps (control loss higher than REAL).
        if i < gate_batches:
            real_l = float(out["loss_recon"])
            # REAL/OFF/SHUF must differ ONLY in memory, NOT in the random MAE
            # mask. The eval path's sole RNG draw is the mask (torch.rand in
            # compute_masked_reconstruction_loss); re-seeding to the SAME value before each control
            # makes all three predict the IDENTICAL masked positions [fix B].
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(20260527 + i)
                off = model.compute_loss(batch, window_size=window_size,
                                            zero_memory=True)
                off_abs.append(float(off["loss_recon"]))
                off_gap.append(float(off["loss_recon"]) - real_l)
                if batch.context_ids.shape[0] > 1:
                    torch.manual_seed(20260527 + i)
                    shuf = model.compute_loss(batch, window_size=window_size,
                                                 shuffle_memory=True)
                    shuf_abs.append(float(shuf["loss_recon"]))
                    shuf_gap.append(float(shuf["loss_recon"]) - real_l)
        for k, v in out.items():
            if v is None:
                continue
            sink = last_graph_eval if k.startswith("graph_") else (
                last_biomem_eval if k.startswith("biomem_") else (
                last_slotgraph_eval if k.startswith("slotgraph_") else (
                last_vqicae_eval if k.startswith("vqicae_") else None)))
            if sink is None:
                continue
            if isinstance(v, (int, float)):
                sink[k] = float(v)
            elif torch.is_tensor(v) and v.numel() == 1:
                sink[k] = float(v.detach())
        # Per-family: use per-row loss instead of batch-wide mean (a 2-row
        # batch with rows from families X and Y was previously credited
        # the same mean to both, hiding genuine per-family differences).
        if "per_example_loss" in out:
            per_ex = out["per_example_loss"].detach().cpu().tolist()
        else:
            per_ex = [float(out["loss_recon"])] * len(batch.task_family)
        for fam, l in zip(batch.task_family, per_ex):
            d = per_fam_stats.setdefault(fam, {"n": 0, "loss": 0.0})
            d["n"] += 1
            d["loss"] += float(l)
        # bAbI is scored by exact-match answer accuracy. per_example_em is the
        # per-row teacher-forced argmax EM over the (all-content) answer span;
        # for list tasks (task 8) the answer is comma-joined so token-exact
        # match == set-match (same token sequence). Aggregate over babi rows.
        if "per_example_em" in out:
            ems = out["per_example_em"].detach().cpu().tolist()
            for fam, em in zip(batch.task_family, ems):
                if fam == "babi":
                    babi_em_hits += float(em)
                    babi_em_n += 1
    model.train(True)
    # Restore training RNG state so eval doesn't perturb the training stream.
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    n = max(len(losses), 1)
    fam_summary = {f: {"n": v["n"], "mean_loss": v["loss"] / max(v["n"], 1)}
                   for f, v in per_fam_stats.items()}
    result = {
        "val_loss_recon": sum(losses) / n,
        "val_top1_acc": sum(accs) / n,
        "val_n_batches": len(losses),
        "val_per_family": fam_summary,
    }
    # REAL/SHUF/OFF binding gate (subset means + per-batch gap vs REAL).
    if off_abs:
        result["val_loss_recon_off"] = sum(off_abs) / len(off_abs)
        result["val_off_minus_real"] = sum(off_gap) / len(off_gap)
    if shuf_abs:
        result["val_loss_recon_shuf"] = sum(shuf_abs) / len(shuf_abs)
        result["val_shuf_minus_real"] = sum(shuf_gap) / len(shuf_gap)
    if babi_em_n:
        result["val_babi_em"] = babi_em_hits / babi_em_n
    for k, v in last_graph_eval.items():
        result[f"val_{k}"] = v
    for k, v in last_biomem_eval.items():
        result[f"val_{k}"] = v
    for k, v in last_slotgraph_eval.items():
        result[f"val_{k}"] = v
    for k, v in last_vqicae_eval.items():
        result[f"val_{k}"] = v
    return result
