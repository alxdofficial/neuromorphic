"""Checkpoint save + identity metadata + per-group grad-norm helper.

Extracted verbatim from ``scripts/train/train.py`` (harness reorg phase 2). No logic changes.
"""
from __future__ import annotations

from pathlib import Path

import torch


def _grad_group_norm(params) -> float:
    """L2 norm of the grads of a parameter group (post-backward, pre-clip).
    Returns 0.0 if the group has no grads. Used to split memory-mechanism vs
    LoRA gradient magnitude for diagnosing gradient starvation."""
    norms = [p.grad.detach().norm() for p in params if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms)))


def _ckpt_metadata(model) -> dict:
    """Identity metadata pinned into every ckpt.

    Captures backbone, chat-scaffold tokens (hash, date, lengths), tokenizer
    eos/pad ids, and the cfg dataclass as a dict. Eval/resume code can verify
    these match the current environment and abort on drift. Without this,
    base-Llama-trained encoders can silently be evaluated against Instruct
    Llama with chat scaffold (the audit issue #1).
    """
    import dataclasses, hashlib
    cfg = model.cfg
    # cfg_dict = the DECLARED dataclass fields. cfg_all = EVERY attribute on the instance, including the
    # ~20 baseline-critical fields cli.py attaches DYNAMICALLY (memoryllm_lora_rank, titans_mem_hidden,
    # kl_coef, slotgraph_*, gisting_*, …) that are NOT declared on ReprConfig. dataclasses.asdict() drops
    # those silently, so a reloaded checkpoint would rebuild the arm with WRONG hyperparameters (audit #3).
    # vars(cfg) captures both (dataclass instances store all attrs in __dict__), JSON-safe-filtered.
    def _jsonable(v):
        return isinstance(v, (int, float, str, bool, type(None))) or (
            isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v))
    cfg_all = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in vars(cfg).items() if _jsonable(v)}
    meta = {
        "backbone_model": cfg.llama_model,
        "cfg_dict": dataclasses.asdict(cfg),          # declared fields only (back-compat)
        "cfg_all": cfg_all,                            # declared + dynamically-attached (the full truth)
        "system_intro_for_memory": cfg.system_intro_for_memory,
    }
    ct = getattr(model, "chat_template", None)
    if ct is not None:
        # Concatenate scaffold token ids and hash — any drift in tokens or
        # date will produce a different digest.
        scaffold_bytes = b"".join([
            ct.pre_memory_ids.numpy().tobytes(),
            ct.post_memory_ids.numpy().tobytes(),
            ct.post_question_ids.numpy().tobytes(),
            ct.eot_id.to_bytes(8, "little", signed=False) if isinstance(ct.eot_id, int) else b"",
        ])
        meta["chat_scaffold_hash"] = hashlib.sha256(scaffold_bytes).hexdigest()
        meta["chat_scaffold_pre_len"] = int(len(ct.pre_memory_ids))
        meta["chat_scaffold_post_mem_len"] = int(len(ct.post_memory_ids))
        meta["chat_scaffold_post_q_len"] = int(len(ct.post_question_ids))
        meta["chat_scaffold_eot_id"] = int(ct.eot_id)
        meta["chat_scaffold_date_string"] = ct.date_string
    return meta


def save_checkpoint(model, opt, step, path: Path, **extras):
    """Persist model + opt state. `extras` lets callers stash auxiliary
    tracking fields (best_val_recon / best_val_step) so resume can pick
    them up; without that, resume starts best_val_recon=inf and the very
    first val of the resumed run overwrites the genuine prior best.pt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save TRAINABLE params + small buffers only; DROP frozen backbone weights —
    # the decoder's Llama AND each encoder's own frozen base copy (ICAE/CCM/Beacon
    # option-A). Keying on requires_grad (not a name prefix) is required because
    # the CCM/Beacon adapters live INSIDE encoder.base.* (they wrap the base
    # linears). Without dropping the frozen encoder base, every ckpt persists a
    # 2nd frozen 1B (~GBs) → kills overnight sweeps on disk/I-O. Frozen weights are
    # reconstructed from the pretrained backbone on load (load is strict=False).
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    def keep(k: str) -> bool:
        if k in trainable:
            return True
        if k.startswith("decoder.llama.") or k.startswith("encoder.base."):
            return False
        return True

    # RNG state (python/numpy/torch/cuda): without it, --resume re-seeds to the INITIAL seed (cli.py does
    # this at startup) and the resumed run REDRAWS the exact early data/augmentation sequence it already
    # trained on, while the optimizer/LR continue mid-schedule (audit #5). Stashing + restoring the RNG on
    # resume lets the stochastic stream continue from the crash point instead of replaying.
    import random
    import numpy as _np
    rng_state = {
        "python": random.getstate(),
        "numpy": _np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    payload = {
        "step": step,
        # Normalize torch.compile's `._orig_mod.` prefix out of the keys → a compile-AGNOSTIC checkpoint.
        # Without this, a run with --compile-decoder saves the read-LoRA as decoder.llama.model._orig_mod.*
        # and an uncompiled eval/reload can't match it (with the eval's missing-trainable-key guard it
        # ABORTS; without it, silently runs an unadapted decoder). No-op when nothing is compiled.
        "model_state_dict": {
            k.replace("._orig_mod.", "."): v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
        "metadata": _ckpt_metadata(model),
        "rng_state": rng_state,
    }
    payload.update(extras)
    # Atomic write: torch.save to a temp sibling, then atomically rename. A crash
    # mid-write (OOM kill, timeout, Ctrl-C) otherwise leaves a truncated .pt that
    # makes --resume fail hard. Over a multi-hour, 7-variant sweep there are many
    # save windows; POSIX rename(2) is atomic so resume always sees a complete file.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
