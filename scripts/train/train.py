#!/usr/bin/env python3
"""Memory training harness: every memory variant × objective, with teacher-forced CE.

Default objective = masked_reconstruction (sentence-pair MAE); other objectives
(qa, conditioned_reconstruction[_bio], continuation) select via --task.

For each variant:
 - Encoder ingests a context via streaming writes → memory tokens.
 - Decoder forward on [memory, question, answer]. The original context
   tokens are NOT visible to the decoder — only memory carries that info.
 - TF-CE on the answer's content-mask positions (load-bearing tokens; the
   rest of the answer span is filler).

Per-variant outputs in outputs/memory/<out_tag>_<variant>/:
  jsonl/<variant>.jsonl   — per-step training metrics
  ckpts/<variant>.last.pt — encoder + decoder.mask_embed weights
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.config import ReprConfig
from src.memory.data_qa import make_qa_dataloader, make_mixed_qa_dataloader, collate_qa
from src.memory.data_continuation import make_continuation_dataloader
from src.memory.data_conditioned_reconstruction_bio import make_conditioned_reconstruction_bio_dataloader
from src.memory.data_babi import make_babi_dataloader, DEFAULT_TASKS as BABI_DEFAULT_TASKS
from src.memory.data_masked_reconstruction import make_long_passage_mae_dataloader
from src.memory.decoder import load_frozen_llama
from src.memory.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
COMPOSITE_TRAIN_P = REPO / "data/wave1/composite_v1/train/passages.jsonl"
COMPOSITE_TRAIN_Q = REPO / "data/wave1/composite_v1/train/questions.jsonl"
COMPOSITE_VAL_P   = REPO / "data/wave1/composite_v1/val/passages.jsonl"
COMPOSITE_VAL_Q   = REPO / "data/wave1/composite_v1/val/questions.jsonl"


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)


def to_device(batch, device):
    for f in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        setattr(batch, f, getattr(batch, f).to(device, non_blocking=True))
    return batch


@torch.no_grad()
def materialize_val_set(val_dl, n_batches: int) -> list:
    """Drain the streaming val_dl ONCE into a fixed list of batches.

    Fixes the val sampling variance bug (#614). Previously each `run_val` call
    advanced the val DataLoader iterator → different random batches per eval →
    0.4-0.7 nat noise between streaming-best and final-eval on the SAME model.
    Materializing the set once means every eval is on the same data → numbers
    become directly comparable and the streaming-best vs final-eval gap closes
    to ~zero (modulo checkpoint differences).

    Memory cost: ~2 MB for 30 batches × 2 examples × 4096 tokens × int64. Trivial.
    """
    fixed = []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        fixed.append(batch)
    return fixed


# ── Mixed multi-task harness ────────────────────────────────────────────────
# A single model per architecture trained on an equal round-robin of these tasks,
# evaluated PER-TASK. Each entry: task key → (data task_family, model.task_mode).
# task_mode routes compute_loss: "masked_reconstruction" → the MAE infill path;
# anything else → the generic QA-path compute_loss (used by babi + continuation).
MIXED_TASKS_DEFAULT = ("mae", "babi", "continuation", "condrecon_bio")
# model.task_mode per mixed task. MAE uses the infill path; babi/continuation/condrecon_bio
# use the generic compute_loss (a non-MAE sentinel string routes there).
MIXED_TASK_MODE = {
    "mae": "masked_reconstruction",
    "babi": "babi",
    "continuation": "continuation",
    "condrecon_bio": "conditioned_reconstruction_bio",
}
CONT_EARLY_TOKENS = 16   # continuation early-token loss = mean over the first N predict positions
# BIOGRAPHICAL conditioned-reconstruction (natural fact-dense value sentences, NOT random words):
# fill-to-budget packs key=value lines to ~ctx_len; n_pairs is the MAX entity pool (the world
# supports ~32 after the train/val name-firewall), n_facts = attributes packed per value sentence.
MIXED_CONDRECON_BIO_N_PAIRS = 24
MIXED_CONDRECON_BIO_N_FACTS = 3


def make_mixed_train_dataloaders(mixed_tasks, tokenizer, cfg, *, ctx_len: int,
                                 m_slots: int, mae_src_tok: str, babi_tasks,
                                 predict_len: int, num_workers: int = 1) -> dict:
    """One TRAIN dataloader per mixed task, all on the uniform interface
    (context_len/chunk = ctx_len, M = m_slots). Homogeneous batches; the
    round-robin alternates which loader is pulled each step."""
    _pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    dls = {}
    for t in mixed_tasks:
        if t == "mae":
            dls[t] = make_long_passage_mae_dataloader(
                tokenizer, batch_size=cfg.batch_size, src_tokenizer_name=mae_src_tok,
                split="train", ctx_len=ctx_len, m_slots=m_slots, seed=42,
                pad_token_id=_pad, num_workers=num_workers)
        elif t == "babi":
            dls[t] = make_babi_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                split="train", seed=42, pad_token_id=_pad, tasks=babi_tasks,
                num_workers=num_workers)
        elif t == "continuation":
            dls[t] = make_continuation_dataloader(
                tokenizer, batch_size=cfg.batch_size, compress_len=ctx_len,
                predict_len=predict_len, split="train", seed=42, pad_token_id=_pad,
                objective="continuation", src_tokenizer_name=mae_src_tok,
                num_workers=num_workers)
        elif t == "condrecon_bio":
            dls[t] = make_conditioned_reconstruction_bio_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                n_pairs=MIXED_CONDRECON_BIO_N_PAIRS, n_query=1, n_facts=MIXED_CONDRECON_BIO_N_FACTS,
                split="train", world_seed=0, stream_seed=42, pad_token_id=_pad, num_workers=num_workers)
        else:
            raise ValueError(f"unknown mixed task {t!r} (expected mae/babi/continuation/condrecon_bio)")
    return dls


def make_mixed_val_sets(mixed_tasks, tokenizer, cfg, val_batches, *, ctx_len: int,
                        m_slots: int, mae_src_tok: str, babi_tasks,
                        predict_len: int) -> dict:
    """One materialized (fixed) VAL set per mixed task — disjoint val seed so the
    eval data never overlaps the training stream."""
    _pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    sets = {}
    for t in mixed_tasks:
        if t == "mae":
            dl = make_long_passage_mae_dataloader(
                tokenizer, batch_size=cfg.batch_size, src_tokenizer_name=mae_src_tok,
                split="val", ctx_len=ctx_len, m_slots=m_slots, seed=7,
                pad_token_id=_pad, num_workers=0)
        elif t == "babi":
            dl = make_babi_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                split="validation", seed=7, pad_token_id=_pad, tasks=babi_tasks,
                num_workers=0)
        elif t == "continuation":
            dl = make_continuation_dataloader(
                tokenizer, batch_size=cfg.batch_size, compress_len=ctx_len,
                predict_len=predict_len, split="validation", seed=7, pad_token_id=_pad,
                objective="continuation", src_tokenizer_name=mae_src_tok, num_workers=0)
        elif t == "condrecon_bio":
            dl = make_conditioned_reconstruction_bio_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                n_pairs=MIXED_CONDRECON_BIO_N_PAIRS, n_query=1, n_facts=MIXED_CONDRECON_BIO_N_FACTS,
                split="validation", world_seed=0, stream_seed=7, pad_token_id=_pad, num_workers=0)
        sets[t] = materialize_val_set(dl, val_batches)
    return sets


def run_mixed_val(model, mixed_tasks, val_sets, device, n_batches, window_size) -> dict:
    """Evaluate ALL mixed val sets, switching model.task_mode per task so each
    routes to its own loss path. Returns {task: per-task metric dict}. The
    continuation set additionally reports an early-token loss (mean CE over the
    first CONT_EARLY_TOKENS predict positions, since the late positions are
    increasingly local-autoregressive and dilute the memory signal)."""
    prev_mode = getattr(model, "task_mode", None)
    per_task = {}
    for t in mixed_tasks:
        model.task_mode = MIXED_TASK_MODE[t]
        # gate_batches=0: the REAL/SHUF/OFF controls aren't needed for the
        # mixed per-task dashboard and triple the eval cost.
        vm = run_val(model, val_sets[t], device, n_batches, window_size, gate_batches=0)
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
    meta = {
        "backbone_model": cfg.llama_model,
        "cfg_dict": dataclasses.asdict(cfg),
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

    payload = {
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
        "metadata": _ckpt_metadata(model),
    }
    payload.update(extras)
    # Atomic write: torch.save to a temp sibling, then atomically rename. A crash
    # mid-write (OOM kill, timeout, Ctrl-C) otherwise leaves a truncated .pt that
    # makes --resume fail hard. Over a multi-hour, 7-variant sweep there are many
    # save windows; POSIX rename(2) is atomic so resume always sees a complete file.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def train_one_variant(
    variant: str, llama, tokenizer, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path, chunk_size: int, window_size: int,
    passages_per_chunk: int, resume: bool = False,
    use_hotpot: bool = True, use_narrative: bool = True,
    use_musique: bool = False, use_babilong: bool = False,
    babilong_config: str = "4k",
    mix_weights: tuple = (0.5, 0.25, 0.25, 0.0, 0.0),
    composite_task_weights: Optional[dict[str, float]] = None,
    patience: int = 5,                       # eval-points without improvement → stop
    min_step_for_stop: int = 2000,           # don't stop during warmup-noise era
    min_delta: float = 0.01,                 # min val_recon drop to count as a real improvement
    task: str = "masked_reconstruction",     # "masked_reconstruction" = sentence-pair MAE (active); "qa" = composite-QA mix; "conditioned_reconstruction" = key→value; "continuation" = gist-LM
    cond_recon_n_pairs: int = 64, cond_recon_n_query: int = 1, cond_recon_value_len: int = 1,
    cond_recon_bio_n_facts: int = 3, cond_recon_bio_world_seed: int = 0,
    babi_tasks: tuple = BABI_DEFAULT_TASKS,
    compress_len: int = 2048, predict_len: int = 64,
    mae_src_tok: str = "meta-llama/Llama-3.2-1B",
) -> dict:
    device = "cuda"
    # LoRA wraps Llama's q/v in place (LoRALinear). The base `llama` is shared
    # across variants, so when LoRA is on each variant must get its OWN fresh
    # frozen Llama (pass None → the decoder self-loads one) — otherwise variant N
    # would double-wrap variant N-1's LoRALinear and inherit its trained adapter.
    # LoRA off → keep sharing the single read-only frozen Llama (fast).
    llama_arg = None if cfg.use_llama_lora else llama
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
    model.task_mode = task   # explicit dispatch (MAE→masked_reconstruction path; else generic compute_loss)
    n_trainable = model.n_trainable_params()
    # A variant is eval-only iff it has no trainable params. With LoRA-all on, the
    # two vanilla references DO train (their ~1.7M LoRA) — they become the LoRA'd
    # floor (no context) and ceiling (full context), not frozen reference points.
    # vanilla_full_context is kept FROZEN/eval-only (the ceiling): it re-forwards
    # the full ~8192-token context per question (~40x the decoder tokens of the
    # memory arms), so LoRA-training it OOMs under backward — and a frozen
    # full-context Llama is already a valid upper bound. Its zero-init LoRA, if
    # built, is an identity no-op. vanilla_llama (the floor) still LoRA-trains.
    is_eval_only = (n_trainable == 0) or (
        variant == "vanilla_full_context" and task != "masked_reconstruction")
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps)")
    print(f"{'='*78}")

    # vanilla_* reference points:
    #   - vanilla_llama: no memory at all (LoRA'd loss FLOOR)
    #   - vanilla_full_context: passes raw context embeddings as memory (LoRA'd CEILING)
    # With LoRA off they have 0 trainable params → eval-only. With LoRA-all on they
    # train their shared ~1.7M adapter exactly like every other arm.
    if not is_eval_only:
        opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )
    else:
        opt = None

    # qa (composite multi-hop mix) — the only single-task path; the 4-task `mixed`
    # benchmark has its own trainer (train_mixed_variant). Same encoder→memory→
    # prepend→CE path; only the DATA differs.
    train_dl = None if is_eval_only else make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=COMPOSITE_TRAIN_P,
        composite_questions_path=COMPOSITE_TRAIN_Q,
        use_hotpot=use_hotpot, use_narrative=use_narrative,
        use_musique=use_musique, use_babilong=use_babilong,
        babilong_config=babilong_config,
        split="train",
        chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
        weights=mix_weights, composite_task_weights=composite_task_weights,
        num_workers=2, seed=42,
    )
    val_dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=COMPOSITE_VAL_P,
        composite_questions_path=COMPOSITE_VAL_Q,
        use_hotpot=use_hotpot, use_narrative=use_narrative,
        use_musique=use_musique, use_babilong=use_babilong,
        babilong_config=babilong_config,
        split="validation",
        chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
        weights=mix_weights, composite_task_weights=composite_task_weights,
        num_workers=2, seed=7,
    )
    # Fixes #614: drain val_dl ONCE into a fixed list so every run_val call
    # sees the same batches. Without this, in-training val and final-eval got
    # different random draws → 0.4-0.7 nat noise → streaming-best was biased
    # low (cherry-picked from many noisy draws) and final-eval was a one-shot
    # gamble. With a fixed set: streaming val and final eval directly comparable.
    print(f"  Materializing fixed val set ({val_batches} batches)...")
    val_set = materialize_val_set(val_dl, val_batches)
    print(f"  Fixed val set ready: {len(val_set)} batches.")

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    best_ckpt_path = out_dir / f"ckpts/{variant}.best.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # Best-checkpoint tracking. Only consider vals from step >= BEST_MIN_STEP
    # (10% of total steps, min 1000) to filter early-warmup flukes — LR is
    # still ramping and val noise dominates real signal at low steps.
    best_val_recon = float("inf")
    best_val_step = -1
    # 10% of total (min 1000) to skip early-warmup flukes, but never exceed the run
    # itself — short smokes (e.g. 600 steps) must still save a best after warmup [merge #5]
    BEST_MIN_STEP = min(1000, max(val_every, n_steps // 10))
    # Patience-based early stopping (best.pt staleness criterion). Counts
    # eval points since the last best.pt update. Stop only when no new
    # global best has been seen for `patience` consecutive evals past
    # min_step_for_stop. Previous "smoothed-mean" criterion triggered on
    # volatility and could fire on the same step that just produced a new
    # best.pt — invalidated several runs. Best-staleness avoids that
    # pathology entirely: the moment a true improvement lands, the counter
    # resets, so we never stop within `patience * val_every` steps of an
    # actual improvement.
    early_stopped = False
    stopped_at_step = n_steps
    evals_since_best = 0

    if is_eval_only:
        # Eval-only path. Skip any prior jsonl and run a single final-val pass.
        if jsonl_path.exists():
            jsonl_path.unlink()
        t_start = time.time()
        final_val = run_val(model, val_set, device, val_batches, window_size)
        elapsed = time.time() - t_start
        with open(jsonl_path, "w") as fp:
            fp.write(json.dumps({
                "phase": "val", "step": 0, "variant": variant,
                "final": True, "eval_only": True, **final_val,
            }) + "\n")
        print(f"  [eval-only] {variant} val_recon={final_val['val_loss_recon']:.4f}  "
              f"top1={final_val['val_top1_acc']*100:.1f}%  ({elapsed:.1f}s)", flush=True)
        summary = {
            "variant": variant,
            "trainable_params": n_trainable,
            "n_steps": 0,
            "elapsed_s": elapsed,
            "final_val_loss_recon": final_val["val_loss_recon"],
            "final_val_top1_acc": final_val["val_top1_acc"],
            "final_val_per_family": final_val["val_per_family"],
            "eval_only": True,
        }
        # Save a loadable ckpt so the per-family eval harness can load this
        # eval-only arm (e.g. vanilla_full_context) uniformly — load_variant
        # requires a ckpt. No optimizer state (nothing trained); cfg_dict +
        # mask_embed + zero-init LoRA are all eval needs.
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        _keep = lambda k: (not k.startswith("decoder.llama.")) or ("lora_" in k)
        torch.save({
            "step": 0,
            "model_state_dict": {k: v for k, v in model.state_dict().items() if _keep(k)},
            "metadata": _ckpt_metadata(model),
            "eval_only": True,
        }, best_ckpt_path)
        print(f"  [eval-only] saved ckpt → {best_ckpt_path}", flush=True)
        del model
        torch.cuda.empty_cache()
        return summary

    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # objective/backbone drift check — a resume that silently switches task or
        # backbone would corrupt the run (the metadata's whole reason for existing).
        _meta = sd.get("metadata", {}) or {}
        _ck_cfg = _meta.get("cfg_dict", {}) or {}
        for _field, _now in (("backbone_model", model.cfg.llama_model),
                             ("task_mode", getattr(model.cfg, "task_mode", None))):
            _was = _meta.get(_field, _ck_cfg.get(_field))
            if _was is not None and _was != _now:
                raise RuntimeError(
                    f"[resume] {_field} drift: checkpoint has {_was!r} but this run "
                    f"is {_now!r}. Resuming would mix objectives/backbones — refusing. "
                    f"Start a fresh --out-tag or match the original setting.")
        _res = model.load_state_dict(sd["model_state_dict"], strict=False)
        _bad = [k for k in (_res.missing_keys + _res.unexpected_keys)
                if "llama" not in k.lower()
                and not k.startswith("encoder.base.")      # frozen own-base (dropped by keep())
                and not k.startswith("decoder.llama.")]
        if _bad:
            raise RuntimeError(
                "[resume] checkpoint/arch mismatch — non-Llama missing/unexpected "
                f"keys would silently leave params random: {_bad[:12]}")
        opt.load_state_dict(sd["optimizer_state_dict"])
        start_step = int(sd.get("step", 0)) + 1
        # Restore best-tracking so we don't overwrite the prior best.pt with
        # a worse first-val from the resumed run. Older ckpts didn't store
        # these — fall back to inf and warn so the user knows the prior best
        # is at risk of being clobbered on the next improvement check.
        if "best_val_recon" in sd and "best_val_step" in sd:
            best_val_recon = float(sd["best_val_recon"])
            best_val_step = int(sd["best_val_step"])
            evals_since_best = int(sd.get("evals_since_best", 0))
            print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1} "
                  f"(prior best={best_val_recon:.4f} @ step {best_val_step}, "
                  f"evals_since_best={evals_since_best})")
        else:
            print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1} "
                  f"(WARN: no best_val_recon in ckpt — first val will define "
                  f"a new best.pt; prior best.pt may be overwritten)")
    else:
        if jsonl_path.exists():
            jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    last_print_step, last_print_time = start_step, t_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # per-variant peak-VRAM tracking

    # Resume: use a local step counter rather than skipping batches from the
    # iterator. The dataloader is seeded but stochastic per worker; replaying
    # start_step batches is wasted compute and the resumed run sees a
    # different stream than the original anyway. Just resume the optimizer
    # state and start consuming fresh batches at start_step.
    step = start_step
    last_completed = start_step - 1  # for the final save (= last step whose body finished)
    for batch in train_dl:
        if step >= n_steps:
            break

        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = to_device(batch, device)

        opt.zero_grad(set_to_none=True)
        # bf16 autocast for ~40% speed + ~50% activation memory reduction.
        # Encoders have `enabled=False` blocks around numerically-sensitive ops
        # (entropy, normalization) so those remain fp32 even under the outer
        # autocast — the codebase was designed for this pattern.
        _do_contrast = getattr(cfg, "contrastive_shuf_coef", 0.0) > 0
        # REAL and SHUF must see the SAME random mask, else the contrastive term
        # absorbs mask-difficulty noise (the masking is drawn fresh each compute_loss
        # call). Capture the RNG before REAL and restore it before SHUF so SHUF redraws
        # the identical mask — REAL/SHUF then differ only in memory identity. [R1-M3]
        if _do_contrast:
            _rng = torch.cuda.get_rng_state()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(batch, window_size=window_size)
        loss = out["loss"]
        if _do_contrast:
            # contrastive binding pressure: gradient flows through BOTH branches
            # (lower REAL loss AND raise SHUF loss — both require doc-identity in
            # the state to be expressed by the read). Primary CE on REAL keeps
            # the help-on-match direction anchored against poison-on-mismatch.
            torch.cuda.set_rng_state(_rng)            # SHUF redraws the SAME mask as REAL
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_shuf = model.compute_loss(batch, window_size=window_size,
                                                 shuffle_memory=True)
            contrast = torch.nn.functional.softplus(out["loss"] - out_shuf["loss"])
            loss = loss + cfg.contrastive_shuf_coef * contrast
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        # Per-module grad-norm split BEFORE clipping: separates the memory
        # mechanism (encoder/graph write+read) from the decoder LoRA so we can
        # SEE if the memory params are gradient-starved relative to the adapter
        # (a global norm hides this — the #1 graph training-failure mode).
        _gn_enc = _grad_group_norm(
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("decoder.llama."))
        _gn_lora = _grad_group_norm(
            p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("decoder.llama."))
        # Per-MECHANISM gradient flow (graph variant): is any sub-module being
        # gradient-starved / ignored (the model learning to route around it)?
        # Splits write (obs/blocks/endpoint-heads/edge-head/slots) and read
        # (q_in/blocks/out/gate/tag-role) so the sweep can see WHICH path carries
        # the learning signal. (VQ codebook is EMA — no grad — intentionally absent.)
        graph_gn = {}
        if variant == "graph_baseline":
            _gmech = {
                "p_obs": ("encoder.parser.obs_proj",),
                "p_blocks": ("encoder.parser.blocks",),
                "p_ptr": ("encoder.parser.q_src", "encoder.parser.q_dst", "encoder.parser.bank_key",
                          "encoder.parser.log_temp"),   # incl. the pointer sharpness param
                "p_edge": ("encoder.parser.edge_head",),
                "p_bank": ("encoder.parser.node_bank", "encoder.parser.node_role_avail"),
                "p_slots": ("encoder.parser.init_graph", "encoder.parser.role",
                            "encoder.parser.tag", "encoder.parser.part"),
                # Perceiver read: latent queries + role/tag KV features + cross/self
                # blocks (each incl. its pre-LN) + FFN(+final norm) + out projection.
                "r_queries": ("encoder.reader.queries",),
                "r_kv": ("encoder.reader.role_emb", "encoder.reader.tag"),
                "r_cross": tuple(p for i in range(cfg.graph_read_layers)
                                 for p in (f"encoder.reader.blocks.{i}.cross",
                                           f"encoder.reader.blocks.{i}.cn")),
                "r_self": tuple(p for i in range(cfg.graph_read_layers)
                                for p in (f"encoder.reader.blocks.{i}.slf",
                                          f"encoder.reader.blocks.{i}.sn")),
                "r_ffn": tuple(p for i in range(cfg.graph_read_layers)
                               for p in (f"encoder.reader.blocks.{i}.ff",
                                         f"encoder.reader.blocks.{i}.fn")) + ("encoder.reader.norm",),
                "r_out": ("encoder.reader.out",),
            }
            for _lab, _prefs in _gmech.items():
                graph_gn[f"graph_gn_{_lab}"] = _grad_group_norm(
                    p for n, p in model.named_parameters()
                    if p.requires_grad and any(n.startswith(x) for x in _prefs))
        # PER-GROUP clipping (audit fix): a single global clip is dominated by the
        # LoRA group (>99% of the norm), silently shrinking the memory side's step
        # whenever LoRA spikes. Clip memory and LoRA separately at the same bound.
        _mem_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("decoder.llama.")]
        _lora_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("decoder.llama.")]
        gn_mem = torch.nn.utils.clip_grad_norm_(_mem_params, cfg.grad_clip)
        gn_lora = torch.nn.utils.clip_grad_norm_(_lora_params, cfg.grad_clip)
        gn = (gn_mem ** 2 + gn_lora ** 2) ** 0.5
        opt.step()

        row = {
            "step": step,
            "variant": variant,
            "loss": float(out["loss"].detach()),
            "loss_recon": float(out["loss_recon"]),
            "loss_aux": float(out["loss_aux"]),               # load_balance only (unweighted)
            "top1_acc": float(out["top1_acc"]),
            "n_content_positions": int(out["n_content_positions"]),
            "grad_norm": float(gn),
            "grad_norm_memory": _gn_enc,    # encoder/graph write+read params
            "grad_norm_lora": _gn_lora,     # decoder rank-16 q/v adapter
            "lr": lr,
            "memory_M": out["memory_shape"][1],
        }
        # flat_baseline codebook health (codes_active = #live codes).
        for key in ("codes_active", "routing_entropy"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Per-component aux breakdown (unweighted). Lets us see WHICH aux
        # term is exploding when total aux spikes — previously we only had
        # the sum, and a load_balance=1392 spike was indistinguishable from
        # an orth_loss=1392 spike. Both have very different root causes.
        for key in ("loss_orth", "loss_z"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Splat-variant sublosses (only present when variant == splat_baseline)
        for key in ("splat_aux", "splat_L_pin", "splat_L_prop",
                    "splat_L_adj", "splat_L_sat"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Graph-variant telemetry (only present when variant == graph_baseline).
        # The relational parser emits the selection/vocabulary/relation/read canaries
        # (pointer entropy, distinct nodes used, eff-ranks, read-gate amplitude).
        _graph_scalar_keys = (
            # relational-parser canaries (selection → vocabulary → relation → read)
            "graph_ptr_entropy", "graph_nodes_used", "graph_bank_effrank",
            "graph_edge_effrank", "graph_read_effrank", "graph_read_gate",
        )
        for key in _graph_scalar_keys:
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        row.update(graph_gn)         # per-mechanism gradient flow (graph variant)
        # soft_pointer_graph telemetry (variant == soft_pointer_graph_baseline).
        # Write-side gate means + fact norm are emitted every step; the read-side
        # health probes (rezero/state_effect/collapse/entropy/active_frac) are
        # eval-only (computed in finalize_memory when not self.training), so on
        # train steps only the first three are present — `if key in out` handles it.
        _spg_scalar_keys = (
            "spg_node_gate_mean_avg", "spg_edge_gate_mean_avg",
            "spg_fact_norm",
            "spg_rezero_scale_eff", "spg_state_effect",
            "spg_node_collapse_cos",
            "spg_read_src_entropy", "spg_read_dst_entropy",
            "spg_node_active_frac",
        )
        for key in _spg_scalar_keys:
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # hlvocab telemetry (variant == hlvocab_baseline) — emitted every step.
        for key, value in out.items():
            if not key.startswith("hlvocab_") or value is None:
                continue
            if isinstance(value, (int, float)):
                row[key] = float(value)
            elif torch.is_tensor(value) and value.numel() == 1:
                row[key] = float(value.detach())
        # Per-window breakdown: graph_g_mean_w0..w3, graph_frac_*_w0..w3
        for k, v in out.items():
            if v is None: continue
            if (k.startswith("graph_g_mean_w") or
                (k.startswith("graph_frac_") and k.endswith(("_w0", "_w1", "_w2", "_w3")))):
                row[k] = float(v)
        jsonl_fp.write(json.dumps(row) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            # Display the variant-specific signal that's actually informative:
            # - splat_aux: weighted aux contributing to loss
            # - graph: u/pick/overwrite signals (no aux loss in v3 — graph_aux
            #   is always 0, so displaying it tells you nothing)
            # - else: plain loss_aux (load_balance+orth+z)
            extra_field = ""
            if variant == "soft_pointer_graph_baseline":
                # show node/edge gate means (anchor-init ≈ 0.38 / 0.27).
                g_n = float(out.get("spg_node_gate_mean_avg", 0.0) or 0.0)
                g_e = float(out.get("spg_edge_gate_mean_avg", 0.0) or 0.0)
                extra_field = f"g_n={g_n:.2f} g_e={g_e:.2f}"
                aux_tag, aux_display = "aux", float(out["loss_aux"])
            elif variant == "graph_baseline":
                # live collapse canaries: nodes used, pointer entropy, read eff-rank,
                # read gate. Watch nodes→1 (vocabulary collapse), ent high (pointer not
                # sharpening), rd_er→1 (membership-not-binding), gate→0 (read ignored).
                extra_field = (f"nodes={float(out.get('graph_nodes_used',0)):.0f} "
                               f"ent={float(out.get('graph_ptr_entropy',0)):.2f} "
                               f"rd_er={float(out.get('graph_read_effrank',0)):.1f} "
                               f"gate={float(out.get('graph_read_gate',0)):.3f}")
                aux_tag, aux_display = "ent", float(out.get("graph_ptr_entropy", 0.0))
            elif "splat_aux" in out and out["splat_aux"] is not None:
                aux_display = float(out["splat_aux"])
                aux_tag = "s_aux"
            else:
                aux_display = float(out["loss_aux"])
                aux_tag = "aux"
            print(f"  step {step:6d}/{n_steps}  recon={float(out['loss_recon']):.4f}  "
                  f"top1={float(out['top1_acc'])*100:5.1f}%  "
                  f"{aux_tag}={aux_display:.3f}  "
                  f"{extra_field + '  ' if extra_field else ''}"
                  f"gnorm={float(gn):6.2f}  lr={lr:.2e}  ({sps:.1f} step/s)",
                  flush=True)

        if step > 0 and step % val_every == 0:
            vm = run_val(model, val_set, device, val_batches, window_size)
            val_row = {"phase": "val", "step": step, "variant": variant, **vm}
            jsonl_fp.write(json.dumps(val_row) + "\n")
            extra_parts = []
            if "val_off_minus_real" in vm:
                extra_parts.append(f"OFF-REAL={vm['val_off_minus_real']:+.3f}")
            if "val_shuf_minus_real" in vm:
                extra_parts.append(f"SHUF-REAL={vm['val_shuf_minus_real']:+.3f}")
            if "val_babi_em" in vm:
                extra_parts.append(f"babi_EM={vm['val_babi_em']*100:.1f}%")
            extra = ("  " + "  ".join(extra_parts)) if extra_parts else ""
            print(f"    [val @ {step}]  recon={vm['val_loss_recon']:.4f}  "
                  f"top1={vm['val_top1_acc']*100:.1f}%{extra}",
                  flush=True)
            # Best-checkpoint save: only past the warmup-fluke window.
            # This is also the patience-reset signal — if best.pt updates,
            # the model is still improving; clear the staleness counter.
            improved = (step >= BEST_MIN_STEP
                        and vm["val_loss_recon"] < best_val_recon - min_delta)
            if improved:
                best_val_recon = vm["val_loss_recon"]
                best_val_step = step
                evals_since_best = 0
                # best.pt carries its own best-tracking metadata so a --resume from
                # best.pt restores it (resume normally uses last.pt, which already
                # stashes these; this keeps best.pt self-describing too).
                save_checkpoint(
                    model, opt, step, best_ckpt_path,
                    best_val_recon=best_val_recon, best_val_step=best_val_step,
                    evals_since_best=evals_since_best,
                )
                print(f"    [best ckpt @ {step}]  val_recon={best_val_recon:.4f}",
                      flush=True)
            else:
                # No improvement. Only count it against patience past
                # min_step_for_stop (warmup eval points don't trigger stops).
                if step >= min_step_for_stop:
                    evals_since_best += 1
                    if patience > 0 and evals_since_best >= patience:
                        print(f"    [early stop @ {step}]  {evals_since_best} "
                              f"evals without best.pt update "
                              f"(best val_recon={best_val_recon:.4f} "
                              f"@ step {best_val_step})", flush=True)
                        early_stopped = True
                        stopped_at_step = step
                        # Audit fix #9: stamp last_completed BEFORE break
                        # so the final .last.pt save records the correct
                        # step (otherwise it'd be one save_every behind).
                        last_completed = step
                        break

        if step > 0 and step % save_every == 0:
            # `step` here is the last completed step. Resume reads N → starts at N+1.
            # Stash best tracking in last.pt so resume preserves prior best.pt
            # rather than overwriting it with a worse first-val.
            save_checkpoint(
                model, opt, step, ckpt_path,
                best_val_recon=best_val_recon,
                best_val_step=best_val_step,
                evals_since_best=evals_since_best,
            )

        last_completed = step
        step += 1

    # Final save: `step` has been incremented past the last completed iter
    # (or equals start_step if the loop never ran). Persist last_completed so
    # resume's `+ 1` lands on the correct next step.
    save_checkpoint(
        model, opt, last_completed, ckpt_path,
        best_val_recon=best_val_recon,
        best_val_step=best_val_step,
        evals_since_best=evals_since_best,
    )

    # Final eval on BEST.PT, not on the last training step's weights.
    # Previously we evaluated whatever weights were in memory at end of
    # training — which is .last.pt, NOT .best.pt. With the patience
    # criterion, .last.pt is by definition a checkpoint that did NOT beat
    # the best. So final-eval was systematically biased high (worse) vs
    # the model's actual peak. Loading best.pt before the final eval gives
    # the trustworthy "what's the best this variant achieved" number.
    if best_ckpt_path.exists():
        sd = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        _res = model.load_state_dict(sd["model_state_dict"], strict=False)
        _bad = [k for k in (_res.missing_keys + _res.unexpected_keys)
                if "llama" not in k.lower()
                and not k.startswith("encoder.base.")      # frozen own-base (dropped by keep())
                and not k.startswith("decoder.llama.")]
        if _bad:
            raise RuntimeError(f"[final-best] checkpoint/arch mismatch — non-Llama keys: {_bad[:12]}")
        print(f"  [loaded best.pt @ step {best_val_step} for final eval]", flush=True)
    final_val = run_val(model, val_set, device, val_batches, window_size)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, "evaluated_on_best": best_ckpt_path.exists(),
        "best_step": best_val_step, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
    src = f"best.pt @ {best_val_step}" if best_ckpt_path.exists() else f"last weights"
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_recon={final_val['val_loss_recon']:.4f} "
          f"top1={final_val['val_top1_acc']*100:.1f}% ({src})  "
          f"peak_vram={peak_vram_gb:.1f}GB", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "n_steps": step,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "final_val_loss_recon": final_val["val_loss_recon"],
        "final_val_top1_acc": final_val["val_top1_acc"],
        "final_val_per_family": final_val["val_per_family"],
        "best_val_loss_recon": (best_val_recon if best_val_step >= 0
                                else final_val["val_loss_recon"]),
        "best_val_step": best_val_step,
        "early_stopped": early_stopped,
        "stopped_at_step": stopped_at_step,
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def train_mixed_variant(
    variant: str, llama, tokenizer, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path, window_size: int,
    mixed_tasks: tuple, mixed_ctx: int, mixed_M: int,
    babi_tasks: tuple, predict_len: int, mae_src_tok: str,
    resume: bool = False,
) -> dict:
    """ONE model per architecture, trained on an EQUAL round-robin of mixed_tasks,
    evaluated PER-TASK. Homogeneous batches; tasks alternate step-to-step. Before
    each compute_loss the model's task_mode is set from the batch's task so MAE
    routes to the infill path and babi/continuation to the generic QA path.

    Distinct from train_one_variant (single-task) so the single-task path stays
    untouched. Per-task val rows are logged to the run jsonl, each tagged with
    `task` + `step`; per-task best_step/best_metric are tracked independently."""
    device = "cuda"
    llama_arg = None if cfg.use_llama_lora else llama
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
    n_trainable = model.n_trainable_params()
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps, "
          f"MIXED={'+'.join(mixed_tasks)})")
    print(f"{'='*78}")
    if n_trainable == 0:
        raise SystemExit(
            f"mixed training requires trainable params; {variant} has none "
            f"(vanilla floor/ceiling are single-task eval-only references).")

    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        fused=torch.cuda.is_available())

    train_dls = make_mixed_train_dataloaders(
        mixed_tasks, tokenizer, cfg, ctx_len=mixed_ctx, m_slots=mixed_M,
        mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len)
    print(f"  Materializing fixed per-task val sets ({val_batches} batches each)...")
    val_sets = make_mixed_val_sets(
        mixed_tasks, tokenizer, cfg, val_batches, ctx_len=mixed_ctx, m_slots=mixed_M,
        mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len)
    print(f"  Val sets ready: {{ {', '.join(f'{t}:{len(val_sets[t])}' for t in mixed_tasks)} }}")

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    best_ckpt_path = out_dir / f"ckpts/{variant}.best.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-task best tracking — each task plateaus on its own schedule, so we keep
    # an independent best_step/best_metric per task (the primary metric is val_loss).
    best = {t: {"metric": float("inf"), "step": -1} for t in mixed_tasks}
    # Aggregate (mean val_loss across tasks) best — the SINGLE selection metric backed by a saved
    # .best.pt. One model serves all tasks, so a per-task best can't each own weights; the aggregate
    # is the principled selection criterion (per-task best_metric stays as informational telemetry).
    agg_best = {"metric": float("inf"), "step": -1}

    # ── resume: restore model + optimizer + step + best tracking from .last.pt ──
    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        _meta = sd.get("metadata", {}) or {}
        _ck_cfg = _meta.get("cfg_dict", {}) or {}
        for _field, _now in (("backbone_model", model.cfg.llama_model),
                             ("task_mode", getattr(model.cfg, "task_mode", None))):
            _was = _meta.get(_field, _ck_cfg.get(_field))
            if _was is not None and _was != _now:
                raise RuntimeError(
                    f"[resume] {_field} drift: checkpoint has {_was!r} but this run is {_now!r}. "
                    f"Refusing to mix objectives/backbones — use a fresh --out-tag.")
        _res = model.load_state_dict(sd["model_state_dict"], strict=False)
        _bad = [k for k in (_res.missing_keys + _res.unexpected_keys)
                if "llama" not in k.lower() and not k.startswith("encoder.base.")
                and not k.startswith("decoder.llama.")]
        if _bad:
            raise RuntimeError(f"[resume] checkpoint/arch mismatch — would leave params random: {_bad[:12]}")
        opt.load_state_dict(sd["optimizer_state_dict"])
        start_step = int(sd.get("step", 0)) + 1
        if "mixed_best" in sd:
            best = {t: dict(sd["mixed_best"].get(t, {"metric": float("inf"), "step": -1}))
                    for t in mixed_tasks}
        if "mixed_agg_best" in sd:
            agg_best = dict(sd["mixed_agg_best"])
        print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1} "
              f"(agg_best={agg_best['metric']:.4f} @ {agg_best['step']})")
    elif jsonl_path.exists() and not resume:
        jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    last_print_step, last_print_time = start_step, t_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # One persistent iterator per task; round-robin pulls the next task in rotation.
    iters = {t: iter(dl) for t, dl in train_dls.items()}

    def next_batch(t):
        try:
            return next(iters[t])
        except StopIteration:                # streaming loaders are ~infinite; re-arm defensively
            iters[t] = iter(train_dls[t])
            return next(iters[t])

    rotation = []   # diagnostic: the realized task sequence (capped for the smoke print)
    for step in range(start_step, n_steps):
        task = mixed_tasks[step % len(mixed_tasks)]   # equal round-robin
        if len(rotation) < 24:
            rotation.append(task)
        model.task_mode = MIXED_TASK_MODE[task]       # per-batch dispatch (E/D)
        batch = next_batch(task)

        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(batch, window_size=window_size)
        loss = out["loss"]
        if not torch.isfinite(loss):
            # Skip a rare non-finite batch instead of dying (grads are still zeroed from above, so the
            # model is NOT poisoned). A SYSTEMATIC divergence would skip every step → visible in the log.
            print(f"  [step {step}] non-finite loss = {float(loss.detach())} (task={task}) — skipping batch")
            opt.zero_grad(set_to_none=True)
            continue
        # Debug: from --anomaly-from onward, run backward under anomaly detection so the FIRST non-finite
        # gradient halts with a traceback to the exact forward op that produced it (catches backward NaNs
        # that the forward-loss skip-guard above can't see).
        import contextlib as _ctxlib
        _afrom = int(getattr(cfg, "anomaly_from", -1))
        _anom = (torch.autograd.detect_anomaly()
                 if (_afrom >= 0 and step >= _afrom) else _ctxlib.nullcontext())
        with _anom:
            loss.backward()
        _mem_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("decoder.llama.")]
        _lora_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("decoder.llama.")]
        gn_mem = torch.nn.utils.clip_grad_norm_(_mem_params, cfg.grad_clip)
        gn_lora = torch.nn.utils.clip_grad_norm_(_lora_params, cfg.grad_clip)
        gn = (gn_mem ** 2 + gn_lora ** 2) ** 0.5
        # Skip the optimizer step on a non-finite GRADIENT (the forward-loss check above misses backward
        # NaNs). This is the standard bf16/AMP handling — PyTorch's GradScaler skips inf/nan-grad steps —
        # and is the right fix for a STOCHASTIC mixed-precision NaN (a rare borderline overflow that fires
        # on different steps each run and vanishes under anomaly mode), vs poisoning the model.
        if not (torch.isfinite(gn_mem) and torch.isfinite(gn_lora)):
            print(f"  [step {step}] non-finite GRADIENT ({float(gn)}, task={task}) — skipping opt.step")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        # Per-step TRAIN row, tagged with the task that produced this batch.
        jsonl_fp.write(json.dumps({
            "step": step, "variant": variant, "task": task, "phase": "train",
            "loss": float(out["loss"].detach()),
            "loss_recon": float(out["loss_recon"]),
            "top1_acc": float(out["top1_acc"]),
            "grad_norm": float(gn),
            "memory_M": out["memory_shape"][1],
            "lr": lr,
        }) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            print(f"  step {step:6d}/{n_steps}  [{task:12}]  "
                  f"recon={float(out['loss_recon']):.4f}  "
                  f"top1={float(out['top1_acc'])*100:5.1f}%  "
                  f"M={out['memory_shape'][1]}  gnorm={float(gn):6.2f}  "
                  f"lr={lr:.2e}  ({sps:.1f} step/s)", flush=True)

        if step > 0 and step % val_every == 0:
            per_task = run_mixed_val(model, mixed_tasks, val_sets, device,
                                     val_batches, window_size)
            # one PER-TASK val row each, tagged with task + step (E/E).
            parts = []
            for t in mixed_tasks:
                vm = per_task[t]
                row = {"phase": "val", "step": step, "variant": variant, "task": t,
                       "val_loss": vm["val_loss_recon"], "top1": vm["val_top1_acc"]}
                if "val_babi_em" in vm:
                    row["val_babi_em"] = vm["val_babi_em"]
                if "val_cont_early_loss" in vm:
                    row["val_cont_early_loss"] = vm["val_cont_early_loss"]
                for _k, _v in vm.items():               # graph read+write collapse canaries
                    if _k.startswith(("val_graph_", "val_biomem_", "val_slotgraph_", "val_vqicae_")):
                        row[_k] = _v
                # per-task best on val_loss
                if vm["val_loss_recon"] < best[t]["metric"]:
                    best[t]["metric"] = vm["val_loss_recon"]
                    best[t]["step"] = step
                row["best_step"] = best[t]["step"]
                row["best_metric"] = best[t]["metric"]
                jsonl_fp.write(json.dumps(row) + "\n")
                tag = f"{t}={vm['val_loss_recon']:.3f}"
                if "val_babi_em" in vm:
                    tag += f"(EM={vm['val_babi_em']*100:.0f}%)"
                if "val_cont_early_loss" in vm:
                    tag += f"(early={vm['val_cont_early_loss']:.3f})"
                if "val_graph_edge_cos" in vm:          # write/read collapse at a glance
                    tag += (f"[ec={vm['val_graph_edge_cos']:.2f}"
                            f" nu={int(vm.get('val_graph_nodes_used', 0))}"
                            f" isns={vm.get('val_graph_input_sens', 0):.1f}"
                            f" mr={vm.get('val_graph_mem_effrank', 0):.1f}]")
                if "val_vqicae_perplexity" in vm:       # vqicae codebook-usage canary at a glance
                    tag += (f"[ppl={vm['val_vqicae_perplexity']:.0f}"
                            f" act={int(vm.get('val_vqicae_active_codes', 0))}]")
                if "val_slotgraph_edge_frac" in vm:     # slotgraph structure canary at a glance
                    tag += (f"[ef={vm['val_slotgraph_edge_frac']:.2f}"
                            f" se={vm.get('val_slotgraph_src_entropy', 0):.1f}"
                            f" mr={vm.get('val_slotgraph_mem_effrank', 0):.1f}]")
                parts.append(tag)
            print(f"    [val @ {step}]  " + "  ".join(parts), flush=True)
            # Aggregate selection: mean val_loss across tasks → save the single .best.pt.
            agg = sum(per_task[t]["val_loss_recon"] for t in mixed_tasks) / len(mixed_tasks)
            if agg < agg_best["metric"]:
                agg_best["metric"], agg_best["step"] = agg, step
                save_checkpoint(model, opt, step, best_ckpt_path,
                                mixed_best=best, mixed_agg_best=agg_best)
                print(f"    [best @ {step}]  agg_val={agg:.4f} → saved {best_ckpt_path.name}", flush=True)

        if step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, ckpt_path,
                            mixed_best=best, mixed_agg_best=agg_best)

    save_checkpoint(model, opt, max(n_steps - 1, 0), ckpt_path,
                    mixed_best=best, mixed_agg_best=agg_best)
    final = run_mixed_val(model, mixed_tasks, val_sets, device, val_batches, window_size)
    for t in mixed_tasks:
        vm = final[t]
        row = {"phase": "val", "step": n_steps, "variant": variant, "task": t,
               "final": True, "val_loss": vm["val_loss_recon"], "top1": vm["val_top1_acc"],
               "best_step": best[t]["step"], "best_metric": best[t]["metric"]}
        if "val_babi_em" in vm:
            row["val_babi_em"] = vm["val_babi_em"]
        if "val_cont_early_loss" in vm:
            row["val_cont_early_loss"] = vm["val_cont_early_loss"]
        for _k, _v in vm.items():                       # graph/biomem/slotgraph/arrival collapse canaries
            if _k.startswith(("val_graph_", "val_biomem_", "val_slotgraph_", "val_vqicae_")):
                row[_k] = _v
        jsonl_fp.write(json.dumps(row) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
    print(f"  DONE (mixed): {n_steps} steps in {elapsed/60:.1f} min  "
          f"peak_vram={peak_vram_gb:.1f}GB", flush=True)
    print(f"  rotation[:{len(rotation)}] = {rotation}", flush=True)
    summary = {
        "variant": variant, "trainable_params": n_trainable, "n_steps": n_steps,
        "elapsed_s": elapsed, "peak_vram_gb": peak_vram_gb, "mixed": True,
        "mixed_tasks": list(mixed_tasks),
        "per_task_final": {t: {"val_loss": final[t]["val_loss_recon"],
                               "top1": final[t]["val_top1_acc"],
                               "best_step": best[t]["step"],
                               "best_metric": best[t]["metric"]} for t in mixed_tasks},
        "agg_best_metric": agg_best["metric"], "agg_best_step": agg_best["step"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def probe_bs(variants, llama, tokenizer, cfg, args):
    """Per-arm max-batch-size VRAM probe on the conditioned-reconstruction path (no training, no checkpoints).
    For each arm: push BS up the --probe-bs-list until OOM; report the largest fitting BS,
    its peak VRAM, and throughput. Builds each arm with the SAME production cfg as a real
    run, so the reported max-BS is directly usable as that arm's --batch-size."""
    import time
    from dataclasses import replace as _dc_replace
    from src.memory.data_conditioned_reconstruction_bio import ConditionedReconstructionBioDataset
    device = "cuda"
    _pad = cfg.pad_token_id if cfg.pad_token_id is not None else 128_001
    ds = ConditionedReconstructionBioDataset(tokenizer, context_len=args.chunk_size, n_pairs=args.cond_recon_n_pairs,
                     n_query=args.cond_recon_n_query, n_facts=args.cond_recon_bio_n_facts,
                     world_seed=0, stream_seed=0, pad_token_id=_pad)
    pool, it = [], iter(ds)
    for _ in range(max(args.probe_bs_list)):
        pool.append(next(it))

    _TF = ("context_ids", "context_mask", "question_ids", "question_mask",
           "answer_ids", "answer_mask", "answer_content_mask")

    def _batch(bs):
        qb = collate_qa(pool[:bs])
        return _dc_replace(qb, **{f: getattr(qb, f).to(device) for f in _TF})

    rows = []
    for variant in variants:
        llama_arg = None if cfg.use_llama_lora else llama
        try:
            model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{variant:24}] OOM at model build", flush=True)
            torch.cuda.empty_cache(); rows.append((variant, 0, 0.0, 0.0)); continue
        if model.n_trainable_params() == 0:
            print(f"  [{variant:24}] eval-only — skipped", flush=True)
            del model; torch.cuda.empty_cache(); continue
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
        model.train()
        best = (0, 0.0, 0.0)
        for bs in args.probe_bs_list:
            try:
                batch = _batch(bs)
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):                                   # warmup (compile/alloc)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model.compute_loss(batch, window_size=args.window_size)
                    out["loss"].backward(); opt.step()
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model.compute_loss(batch, window_size=args.window_size)
                    out["loss"].backward(); opt.step()
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 5
                peak = torch.cuda.max_memory_allocated() / 2**30
                best = (bs, peak, bs / dt)
                print(f"  [{variant:24}] BS={bs:4d}  peak={peak:5.1f}GB  {bs/dt:6.1f} samp/s  "
                      f"{dt*1e3:6.0f}ms/step", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{variant:24}] BS={bs:4d}  OOM", flush=True)
                torch.cuda.empty_cache(); break
        rows.append((variant, *best))
        del model, opt; torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"\n==== conditioned-reconstruction max-BS per arm (N={args.cond_recon_n_pairs}, {total:.0f}GB GPU) ====", flush=True)
    print(f"  {'arm':<24}{'maxBS':>7}{'peakGB':>8}{'headGB':>8}{'samp/s':>9}")
    for v, bs, pk, sps in rows:
        print(f"  {v:<24}{bs:>7}{pk:>8.1f}{total-pk:>8.1f}{sps:>9.1f}")


def main():
    # allow_abbrev=False: stop `--out <path>` from prefix-matching `--out-tag`,
    # which silently baked a full relative path into the tag and re-nested every
    # run under outputs/memory/outputs/memory/.
    ap = argparse.ArgumentParser(allow_abbrev=False)
    # Active suite: latest graph + published closed-book compressor baselines.
    # Retired graph/plastic/splat and older flat/continuous/MT/Mamba variants
    # remain selectable via explicit --variants if needed.
    # hlvocab_baseline + soft_pointer_graph_baseline are ABANDONED (2026-06-15) —
    # still selectable via explicit --variants for reproduction, out of the default.
    ap.add_argument("--variants", nargs="+", default=[
        "slotgraph_baseline",         # emergent-topology slot memory (supersedes graph_baseline)
        "icae_baseline",              # ICAE (ICLR'24)
        "ccm_baseline",               # CCM (ICLR'24)
        "autocompressor_baseline",    # AutoCompressor/RMT-style recurrent summary
        "beacon_baseline",            # Activation Beacon
        "vanilla_llama",              # MAE loss FLOOR (band lower bound)
        "vanilla_full_context",       # MAE loss CEILING (band upper bound)
    ])
    ap.add_argument("--steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="Override cfg.learning_rate (default 1e-4). Scale with "
                         "BS — e.g. sqrt rule: 1e-4×sqrt(BS/2) → BS=16 ≈ 2.5e-4.")
    ap.add_argument("--warmup", type=int, default=500,
                    help="LR warmup steps (default 500). Recurrent ports "
                         "(autocompressor) need a longer warmup for stability.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=32,
                    help="Number of batches in the fixed val set. With the "
                         "composite mix sampling 9 families + 3 external sources, "
                         "10 batches ≈ 1 example per family (high per-family "
                         "noise). 32 batches × BS=2 = 64 examples ≈ ~5 per family. "
                         "Bumped from old default of 10.")
    # Default chunk_size 4096→8192 (2026-05-28 tranche-3 protocol; hard datasets
    # need the larger window to fit evidence + distractors).
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--mem-tokens", type=int, default=144,
                    help="Matched MEMORY budget: M memory tokens × d_llama, "
                         "matched across ICAE/CCM/AutoCompressor/Beacon "
                         "(and soft_pointer_graph if selected). Derives icae_n_slots, "
                         "ccm_n_comp, autocompressor_n_slots, and Beacon's α.")
    ap.add_argument("--passages-per-chunk", type=int, default=0,
                    help="composite_v1 passages sampled per chunk. 0 = auto: "
                         "scales with chunk_size (~75 per 1024 tokens). "
                         "Manual override accepted as positive int.")
    ap.add_argument("--task", type=str, default="mixed",
                    choices=["mixed", "qa"],
                    help="mixed = ONE model trained on an equal round-robin of --mixed-tasks "
                         "(mae babi continuation condrecon_bio), evaluated per-task (default, active); "
                         "qa = composite multi-hop QA mix. (Standalone single-task entry points were "
                         "removed; the per-task loaders live on as components of --task mixed.)")
    ap.add_argument("--mixed-tasks", nargs="+", default=list(MIXED_TASKS_DEFAULT),
                    choices=list(MIXED_TASK_MODE),
                    help="mixed: tasks in the equal round-robin (default: mae babi continuation "
                         "condrecon_bio). mae = long-passage contiguous MAE compression; babi = "
                         "relational; continuation = next-token prediction; condrecon_bio = "
                         "biographical key→value closed-book recall. Used only when --task mixed.")
    ap.add_argument("--mixed-ctx", type=int, default=1024,
                    help="mixed: uniform context_len/chunk for ALL tasks (default 1024).")
    ap.add_argument("--mixed-M", type=int, default=32,
                    help="mixed: uniform memory budget M (slots/edges) for ALL tasks "
                         "(default 32 → ~32:1 compression at ctx=1024).")
    ap.add_argument("--mae-mask-ratio", type=float, default=0.85,
                    help="mae: fraction of answer tokens replaced by <mask> in the forward.")
    ap.add_argument("--hlvocab-emit", choices=["edge_query", "slotattn"], default="edge_query",
                    help="hlvocab emit read-out: edge_query (independent sharp-softmax) "
                         "| slotattn (Slot-Attention competition — slots partition candidates).")
    ap.add_argument("--cond-recon-n-pairs", type=int, default=64,
                    help="conditioned_reconstruction: number of key→value pairs packed into the context (capacity).")
    ap.add_argument("--cond-recon-n-query", type=int, default=1,
                    help="conditioned_reconstruction: keys recalled per example. 1 = single; >1 = multi.")
    ap.add_argument("--cond-recon-value-len", type=int, default=1,
                    help="conditioned_reconstruction: words per value (1 = single-token value).")
    ap.add_argument("--cond-recon-bio-n-facts", type=int, default=3,
                    help="conditioned_reconstruction_bio: random facts packed per value sentence (2-4).")
    ap.add_argument("--backbone", type=str, default=None,
                    help="override cfg.llama_model (e.g. HuggingFaceTB/SmolLM2-135M for "
                         "the compression line). Auto-sets d_llama from the config.")
    ap.add_argument("--src-tokenizer", type=str, default="meta-llama/Llama-3.2-1B",
                    help="tokenizer that produced the FineWeb-EDU parquet ids (for "
                         "decode→retokenize in the sentence loader).")
    ap.add_argument("--contrastive-shuf-coef", type=float, default=0.0,
                    help="add coef*softplus(L_real - L_shuf) to the loss: makes the "
                         "binding gate ITSELF a training objective (2x step cost; "
                         "the sanctioned aux-loss fallback after the architectural "
                         "ladder, 2026-06-12). Needs batch>1 for the roll.")
    ap.add_argument("--cond-recon-bio-world-seed", type=int, default=0,
                    help="conditioned_reconstruction_bio: world-build seed (train uses this; val uses +10000 → disjoint).")
    ap.add_argument("--compress-len", type=int, default=1024,
                    help="continuation/ae/mae: # natural-text tokens compressed into the 128-token "
                         "memory (then dropped). 1024 = 8x compression (the aligned default).")
    ap.add_argument("--predict-len", type=int, default=64,
                    help="continuation: # next tokens to predict from memory only (closed-book). "
                         "Default 64 isolates the memory signal (less local-autoregression dilution).")
    ap.add_argument("--babi-tasks", type=int, nargs="+", default=list(BABI_DEFAULT_TASKS),
                    help="babi: which bAbI task ids to pool (default = memory-focused subset "
                         "1/2/3/7/8/11/12/13/14: supporting facts, counting, lists, coreference, time).")
    ap.add_argument("--graph-d-graph", type=int, default=0,
                    help="graph: override the graph/vocabulary width d_graph (0 = task default). "
                         "576 matches d_llama → full-rank read tokens (removes the rank handicap).")
    ap.add_argument("--graph-n-nodes", type=int, default=0,
                    help="graph: override the node-bank size N (0 = task default 1024). "
                         "Smaller (384/512) = a tighter vocabulary (the '1024 is oversized' lever); "
                         "barely affects params (bank is N×d_graph).")
    ap.add_argument("--graph-entmax-alpha", type=float, default=1.0,
                    help="graph: node-selection sparsity. 1.0 = softmax (dense blend, default); "
                         "1.5 = entmax (sparse, commits to a few nodes); 2.0 = sparsemax.")
    ap.add_argument("--anomaly-from", type=int, default=-1,
                    help="debug: from this step on, run loss.backward() under torch.autograd.detect_anomaly "
                         "so the first non-finite GRADIENT halts with a traceback to the exact forward op.")
    ap.add_argument("--slotgraph-no-structure", action="store_true",
                    help="slotgraph: disable the MP-read structure = plain prepend of the id-tagged slots "
                         "('id-tagged ICAE'; true pure-ICAE = the icae_baseline variant). "
                         "Ablation: does the message-passing read add anything over id-tagged slots?")
    ap.add_argument("--graph-encoder-lora-rank", type=int, default=0,
                    help="graph: LoRA-adapt the encoder forward like the baselines (0=frozen tap). "
                         "Evens the encoder footing (the graph historically read a frozen tap).")
    ap.add_argument("--graph-read-final", action="store_true",
                    help="graph: read the FINAL hidden (full forward) instead of the mid tap.")
    ap.add_argument("--graph-free-endpoints", action="store_true",
                    help="graph: regress FREE src/dst vectors (drop the bank/selection/topology).")
    ap.add_argument("--beacon-param", nargs="+", default=None,
                    help="Beacon capacity knob: which projections get a trainable copy "
                         "(default q k v ≈ 102M). e.g. --beacon-param v shrinks toward ~17M.")
    ap.add_argument("--beacon-wrap-layers", nargs="+", type=int, default=None,
                    help="Beacon capacity knob: which Llama layer indices to wrap "
                         "(default all 16). e.g. --beacon-wrap-layers 0 1 2 3 → ~4 layers.")
    ap.add_argument("--port-lora-rank", type=int, default=None,
                    help="Capacity knob for ICAE/CCM/AutoCompressor: override their LoRA rank "
                         "(defaults 32/8/32 ≈ 4–6M). e.g. 256 pushes them to ~27–55M (above "
                         "Beacon's ~10M binding floor) to test capacity-vs-mechanism on conditioned-reconstruction.")
    ap.add_argument("--probe-bs", action="store_true",
                    help="Per-arm max-batch-size VRAM probe (no training). For each --variants "
                         "arm: push BS up until OOM, report max-fitting BS + peak VRAM + samp/s, "
                         "then exit. Uses the production cfg + the conditioned-reconstruction data path.")
    ap.add_argument("--probe-bs-list", nargs="+", type=int,
                    default=[8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                    help="BS values to try in --probe-bs (ascending; stops at first OOM).")
    ap.add_argument("--out-tag", type=str, default="v1h")
    ap.add_argument("--resume", action="store_true")
    # Per-window activation checkpointing on the encoder streaming write. With the
    # FlashAttention encoder path (packed windows drop the mask) most variants fit
    # without it, so default ON is a safety net you can disable for full speed.
    ap.add_argument("--grad-ckpt-stream", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--no-hotpot", action="store_true",
                    help="Disable HotpotQA source (default: enabled)")
    # 2026-05-28: hard-only protocol enables narrative + musique by default;
    # use --no-narrative/--no-musique to disable (action='store_false').
    ap.add_argument("--narrative", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable NarrativeQA source (default: ENABLED for "
                         "tranche-3 hard-only protocol). Uses random window "
                         "(oracle-centering removed in post-audit fix).")
    ap.add_argument("--musique", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable MuSiQue-Ans source (default: ENABLED for "
                         "tranche-3 hard-only protocol). Contamination-controlled "
                         "2-4 hop QA — complements HotpotQA by eliminating "
                         "shortcut reasoning.")
    ap.add_argument("--babilong", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable BABILong source (default: ENABLED for the "
                         "v2.1 joint sweep — train + held-out eval, fine-tuned "
                         "small-model track only). Synthetic state-tracking, "
                         "pre-formatted at the config length (4k/8k/16k). "
                         "Use --no-babilong to disable.")
    ap.add_argument("--babilong-config", type=str, default="auto",
                    help="BABILong length config. 'auto' picks the closest "
                         "config below chunk_size (e.g. 4k for chunk=4096, "
                         "8k for chunk=8192). Manual: 0k, 1k, 2k, 4k, 8k, "
                         "16k, 32k, 64k, 128k.")
    ap.add_argument("--mix-weights", nargs="+", type=float,
                    default=[0.2, 0.2, 0.2, 0.2, 0.2],
                    metavar="W",
                    help="Sampling weights for (composite, hotpot, narrative, "
                         "musique, babilong). v2.1 joint-sweep default: equal "
                         "0.2 each across the 5 sources (composite restricted "
                         "to biographical via --composite-task-weights). Equal-"
                         "by-source is the least-gameable fair-head-to-head mix. "
                         "Older 3-tuple callers still work; missing entries "
                         "default to 0.")
    ap.add_argument("--composite-task-weights", nargs="+",
                    default=["biographical:1.0"],
                    metavar="FAMILY:W",
                    help="Per-family weights inside composite_v1. v2.1 joint-"
                         "sweep default: 'biographical:1.0' — composite is "
                         "restricted to the biographical family only (the "
                         "hardest/most-relational family; atomic+relational+"
                         "temporal+aggregation question types over a controlled "
                         "entity-relation world). Unlisted families get weight 0 "
                         "(filtered out). Pass e.g. '' or list families to "
                         "override; 'biographical:2.0 calendar:1.0' for ratios.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Stop training when best.pt hasn't updated for this "
                         "many consecutive val evals past --min-step-for-stop. "
                         "Best-staleness criterion (was previously smoothed "
                         "rolling mean — that one triggered on volatility "
                         "and could fire on the same step a new best landed). "
                         "0 disables. Default 5 (≈ 2500-step plateau at "
                         "val_every=500).")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.01,
                    help="Min val_recon drop to count as a real improvement "
                         "(resets the patience counter). Was hardcoded 1e-4 — "
                         "~200x below val noise (~0.02), so sub-noise drift kept "
                         "resetting patience and runs ground to the step cap. "
                         "0.01 is a meaningful-improvement threshold above noise.")
    ap.add_argument("--min-step-for-stop", type=int, default=3000,
                    help="Don't trigger early-stop before this step. Skips "
                         "warmup-noise era where val is bouncy. Bumped 2000→"
                         "3000 after tranche 1 v2: flat_baseline was still "
                         "improving past step 5000 when patience fired at 5k. "
                         "Slow learners need more runway before plateau check.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global RNG seed (torch/numpy/random). Wired for reproducibility.")
    ap.add_argument("--allow-unmatched-backbone", action="store_true",
                    help="Permit masked_reconstruction on a non-d=576 backbone "
                         "(param-matched ranks are calibrated for SmolLM2-135M).")
    args = ap.parse_args()

    # ── reproducibility: wire the seed (was an unused cfg field) ─────────────
    import random as _random
    import numpy as _np
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ── fail-fast guards (cheap, before any model/data construction) ─────────
    if "hlvocab_baseline" in args.variants and args.task != "masked_reconstruction" \
            and args.chunk_size > 1024:
        raise SystemExit(
            f"hlvocab_baseline builds an [L,L] STDP kernel guarded at L<=1024, but "
            f"--task {args.task} --chunk-size {args.chunk_size} yields L>1024. Use "
            f"--task masked_reconstruction, or --chunk-size <=1024 (single window).")
    if args.contrastive_shuf_coef > 0 and args.batch_size < 2:
        raise SystemExit(
            f"--contrastive-shuf-coef {args.contrastive_shuf_coef} needs batch_size>=2 "
            f"(SHUF rolls memory along the batch dim; B==1 would leave REAL memory).")

    if args.task == "mixed":
        # mixed: the uniform interface — ALL tasks share context_len = mixed_ctx and
        # M = mixed_M. Drive chunk_size/window/compress_len/predict_len from it so the
        # downstream capacity block + per-task loaders all see one consistent length.
        args.chunk_size = args.mixed_ctx
        args.window_size = min(args.window_size, args.chunk_size)
        args.compress_len = args.mixed_ctx          # continuation compress span
        print(f"[auto] mixed: tasks={args.mixed_tasks}  ctx={args.mixed_ctx}  "
              f"M={args.mixed_M}  window_size={args.window_size}  predict_len={args.predict_len}")

    if "/" in args.out_tag:
        ap.error(
            f"--out-tag must be a bare tag, not a path (got {args.out_tag!r}). "
            f"Outputs go to outputs/memory/<out_tag>_<variant>/ automatically; "
            f"pass e.g. --out-tag tranche5_mamba_canonical"
        )

    if "v21" in args.variants:
        raise SystemExit("v21 is not supported in v1h yet.")

    # Flag/weight consistency. The flags toggle source *availability*; the
    # weights control *sampling*. A source with weight=0 is never sampled,
    # so enabling the flag without bumping the weight is a no-op that
    # silently loads ~570MB of data (HotpotQA) or downloads NarrativeQA
    # while contributing nothing. Surface this immediately.
    # Pad mix_weights to length 5 for the 5-source schema.
    padded_weights = list(args.mix_weights) + [0.0] * (5 - len(args.mix_weights))
    args.mix_weights = padded_weights

    if args.narrative and args.mix_weights[2] <= 0:
        raise SystemExit(
            "--narrative is set but mix_weights[2] (NarrativeQA) is 0. "
            "Either drop --narrative or pass --mix-weights with a positive "
            "third value (e.g. --mix-weights 0.5 0.3 0.2)."
        )
    if (not args.no_hotpot) and args.mix_weights[1] <= 0:
        # HotpotQA is on by default; if user explicitly zeros the weight
        # they likely meant to disable the source entirely.
        raise SystemExit(
            "HotpotQA is enabled but mix_weights[1] is 0. Either pass "
            "--no-hotpot or set --mix-weights with a positive second value."
        )
    if args.mix_weights[0] <= 0:
        raise SystemExit(
            "Composite (mix_weights[0]) is 0. composite_v1 is the primary "
            "source and cannot be disabled."
        )

    # Parse --composite-task-weights "family:weight" pairs into a dict.
    composite_task_weights = None
    if args.composite_task_weights:
        composite_task_weights = {}
        for item in args.composite_task_weights:
            if ":" not in item:
                raise SystemExit(
                    f"--composite-task-weights expects 'family:weight', got {item!r}"
                )
            fam, w = item.split(":", 1)
            composite_task_weights[fam.strip()] = float(w)
        print(f"[composite] per-family weights: {composite_task_weights}")
    if args.musique and args.mix_weights[3] <= 0:
        raise SystemExit(
            "--musique is set but mix_weights[3] (MuSiQue) is 0. Either drop "
            "--musique or pass --mix-weights with a positive fourth value "
            "(e.g. --mix-weights 0.4 0.2 0.2 0.2)."
        )
    if args.babilong and args.mix_weights[4] <= 0:
        raise SystemExit(
            "--babilong is set but mix_weights[4] (BABILong) is 0. Either drop "
            "--babilong or pass --mix-weights with a positive fifth value "
            "(e.g. --mix-weights 0.35 0.15 0.15 0.15 0.2)."
        )

    # Base config. Memory-token count + per-variant LoRA ranks/slots are set
    # below (matched-budget block + masked_reconstruction override). LoRA-all:
    # every arm gets the SAME decoder LoRA on the frozen backbone, so the decoder
    # budget is identical and only the memory mechanism differs.
    cfg = ReprConfig(
        batch_size=args.batch_size,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        use_llama_lora=True,
        grad_checkpoint_stream=args.grad_ckpt_stream,
        **({"learning_rate": args.lr} if args.lr is not None else {}),
    )

    # ── backbone resolution (MUST precede budget reporting) ──────────────────
    # Resolve --backbone d_llama/vocab/pad BEFORE the budget block so the printed
    # decoder-read float budget uses the real d_llama (was printing the 2048
    # default even on SmolLM2 d=576) [fix I].
    if args.backbone is not None:
        cfg.llama_model = args.backbone
        from transformers import AutoConfig as _AC, AutoTokenizer as _AT
        _bc = _AC.from_pretrained(args.backbone)
        cfg.d_llama = _bc.hidden_size
        cfg.llama_vocab_size = _bc.vocab_size
        _bt = _AT.from_pretrained(args.backbone)
        # LLM-AGNOSTIC pad + sep derived from the active backbone tokenizer (the old
        # 128001/198 defaults are Llama-only → out of range on SmolLM2 etc.).
        from src.memory.common import resolve_special_ids as _rsi
        cfg.pad_token_id, cfg.sep_token_id = _rsi(_bt)
        print(f"[backbone] {args.backbone}  d_llama={cfg.d_llama}  "
              f"vocab={cfg.llama_vocab_size}  pad={cfg.pad_token_id}  sep={cfg.sep_token_id}")

    # ── Matched MEMORY budget (decoder-read M × d_llama) ────────────────
    # mem_tokens is the single knob; the prepend conditioned-reconstruction arms all emit ~M tokens at
    # d_llama, so the decoder reads the SAME float budget from each — only the
    # memory MECHANISM differs. Beacon (concat) derives α = chunk//M so its total
    # ≈ M. Trainable params are NOT matched (LoRA ports vs the graph substrate
    # differ by design, ~2.5M–48M–100M) — they are reported, not equated.
    M = args.mem_tokens
    cfg.icae_n_slots = M
    cfg.ccm_n_comp = M
    cfg.autocompressor_n_slots = M
    cfg.n_flat_codes = M             # flat/continuous/MT prepend M too (was 192 -> mismatch)
    cfg.beacon_ratio = max(1, args.chunk_size // M)
    if args.beacon_param is not None:
        cfg.beacon_param = tuple(args.beacon_param)
    if args.beacon_wrap_layers is not None:
        cfg.beacon_wrap_layers = tuple(args.beacon_wrap_layers)
    if args.port_lora_rank is not None:
        cfg.icae_lora_rank = args.port_lora_rank
        cfg.ccm_lora_rank = args.port_lora_rank
        cfg.autocompressor_lora_rank = args.port_lora_rank
        print(f"[capacity] ICAE/CCM/AutoCompressor LoRA rank → {args.port_lora_rank}")
    cfg.mae_mask_ratio = args.mae_mask_ratio
    _ceil = lambda a, b: -(-a // b)
    _beacon_M = (_ceil(args.chunk_size, args.window_size)
                 * _ceil(args.window_size, cfg.beacon_ratio))
    print(f"[memory budget] mem_tokens={M} × d_llama={cfg.d_llama} = "
          f"{M * cfg.d_llama:,} prepend decoder-read floats/arm")
    for _a, _m in (("icae", M), ("ccm", M), ("autocompressor", M), ("beacon", _beacon_M)):
        print(f"   {_a:<18} M={_m:<4} → {_m * cfg.d_llama:,} floats")
    if args.task == "mixed":
        # mixed ignores --mem-tokens: the override below sets a FIXED M (mixed_M). The
        # QA-shaped budget figures above (M, _beacon_M) do NOT describe what it emits —
        # skip the multi-window beacon assertion.
        print(f"   [{args.task}] the above --mem-tokens budget is IGNORED; the "
              f"compression-line override below sets the fixed memory budget.")
    elif abs(_beacon_M - M) > max(1, M // 10):
        raise SystemExit(
            f"[memory budget] beacon M={_beacon_M} is >10% off mem_tokens={M} "
            f"(α={cfg.beacon_ratio}); adjust --mem-tokens / chunk / window so the "
            f"matched memory budget holds before launching.")

    # ── compression line: param-matched baselines (backbone resolved above) ───
    if args.task == "mixed":
        # MIXED multi-task: ONE model per arch on a round-robin of mae+babi+
        # continuation, per-task eval. UNIFORM interface: context_len = mixed_ctx
        # and a FIXED M = mixed_M for EVERY task (override the per-task ceil(ctx/30)
        # / k-bucket logic). compute_loss is dispatched per batch (MAE → infill;
        # babi/continuation → generic), so we set BOTH the MAE compressor slots AND
        # the generic prepend budget to the same fixed M. cfg.task_mode = "mixed"
        # is metadata only — the trainer sets model.task_mode per batch.
        cfg.task_mode = "mixed"
        cfg.use_llama_lora = True
        cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
        _M = args.mixed_M                                # FIXED budget (no ceil(chunk/30))
        # MAE-calibrated, param-matched ranks (verified to hold at M=32 within ~0.1M:
        # icae 6.01M / ccm 6.01M / ac 6.03M / beacon 6.08M memory; graph ~6.9M (d_graph=256) —
        # M barely affects params. See scripts/diagnostics/param_count.py.)
        cfg.n_flat_codes = _M
        cfg.icae_n_slots = _M; cfg.icae_lora_rank = 104; cfg.icae_lora_alpha = 208
        cfg.ccm_n_comp = _M; cfg.ccm_lora_rank = 52; cfg.ccm_lora_alpha = 104
        cfg.autocompressor_n_slots = _M
        cfg.autocompressor_lora_rank = 52; cfg.autocompressor_lora_alpha = 104
        cfg.beacon_ratio = max(1, args.mixed_ctx // _M)  # ratio honors the uniform ctx:M (32:1)
        from transformers import AutoConfig as _ACLM
        from src.memory.common import beacon_wrap_layers as _bwlm
        _nlayersm = _ACLM.from_pretrained(cfg.llama_model).num_hidden_layers
        cfg.beacon_wrap_layers = _bwlm(_nlayersm, 11)
        # slotgraph (icae-write + fixed partition + RMSNorm-bounded MP read): own frozen base +
        # encoder-LoRA + endpoint heads + the MP read modules (msg/update ≈1.0M). Encoder-LoRA rank
        # TRIMMED to r85 (from icae's r104) to offset the MP params → total ≈ icae's ~6.9M.
        cfg.slotgraph_n_slots = _M
        cfg.slotgraph_n_nodes = _M // 2          # FIXED partition: half nodes, half edges
        cfg.slotgraph_lora_rank = 85; cfg.slotgraph_lora_alpha = 170   # +MP modules → params ≈ icae
        # vqicae (icae + VQ-discretized slots): encoder-LoRA r96 + projns + EMA codebook (a buffer,
        # not gradient-trained) → ~7.0M trainable, matched to icae. Large codebook K=8192.
        cfg.vqicae_n_slots = _M
        cfg.vqicae_lora_rank = 100; cfg.vqicae_lora_alpha = 200
        cfg.vqicae_codebook_size = 8192; cfg.vqicae_d_code = 256
        print(f"[capacity] mixed: FIXED M={_M}, beacon_ratio={cfg.beacon_ratio} (ctx "
              f"{args.mixed_ctx}:M = {args.mixed_ctx // _M}:1); baselines icae r104 / "
              f"ccm r52 / ac r52 / beacon 11L / slotgraph r85+MP-read / "
              f"vqicae r100+K{cfg.vqicae_codebook_size} (param-matched ~6.9-7.0M).")
        if cfg.d_llama != 576 and not args.allow_unmatched_backbone:
            raise SystemExit(
                f"mixed param-matched ranks are calibrated for SmolLM2-135M (d=576); "
                f"got d_llama={cfg.d_llama} (backbone={cfg.llama_model}). Pass "
                f"--backbone HuggingFaceTB/SmolLM2-135M, or --allow-unmatched-backbone "
                f"to override (capacity match will be off).")
    elif args.task == "qa":
        # composite multi-hop QA: 30:1 COMPRESSION + ~13M memory params. Memory budget
        # M = ceil(context / 30); params matched to the graph anchor (d_graph=384,
        # write=3/read=2, N=1024 ≈ 13.0M) on SmolLM2-135M (d=576). Ranks calibrated
        # for d=576 — capacity match is approximate on other backbones.
        # (See scripts/diagnostics/param_count.py.)
        cfg.use_llama_lora = True
        cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
        _ctx = args.compress_len if args.task == "continuation" else args.chunk_size
        _M = max(1, -(-_ctx // 30))                      # ceil(ctx/30) — the 30:1 budget
        cfg.n_flat_codes = _M
        cfg.icae_n_slots = _M; cfg.icae_lora_rank = 223; cfg.icae_lora_alpha = 446
        cfg.ccm_n_comp = _M; cfg.ccm_lora_rank = 111; cfg.ccm_lora_alpha = 222
        cfg.autocompressor_n_slots = _M
        cfg.autocompressor_lora_rank = 110; cfg.autocompressor_lora_alpha = 220
        cfg.beacon_ratio = 30                            # condensing ratio = the 30:1 compression
        from transformers import AutoConfig as _ACL2
        from src.memory.common import beacon_wrap_layers as _bwl2
        _nlayers2 = _ACL2.from_pretrained(cfg.llama_model).num_hidden_layers
        cfg.beacon_wrap_layers = _bwl2(_nlayers2, 23)
        # graph: ~13M via WIDER node/edge vectors (d_graph 256→384); E = M (same 30:1 budget)
        cfg.graph_d_graph = 384; cfg.graph_write_layers = 3; cfg.graph_read_layers = 2
        cfg.graph_n_nodes = 1024; cfg.graph_n_edges = _M
        print(f"[capacity] {args.task}: 30:1 compression → M={_M} (ctx {_ctx}); "
              f"~13M memory params (graph d_graph=384, E={_M}, baselines icae r223 / "
              f"ccm r111 / ac r110 / beacon 23L).")
    cfg.contrastive_shuf_coef = args.contrastive_shuf_coef
    # graph experiment overrides (win over the task defaults above): wider node/edge
    # vectors (removes the read-token rank handicap) + sparse node selection (entmax).
    if args.graph_d_graph > 0:
        cfg.graph_d_graph = args.graph_d_graph
        print(f"[graph override] d_graph = {cfg.graph_d_graph}")
    if args.graph_n_nodes > 0:
        cfg.graph_n_nodes = args.graph_n_nodes
        print(f"[graph override] n_nodes = {cfg.graph_n_nodes}")
    if args.graph_entmax_alpha > 1.0:
        cfg.graph_entmax_alpha = args.graph_entmax_alpha
        print(f"[graph override] node selection = entmax α={cfg.graph_entmax_alpha}")
    if args.graph_encoder_lora_rank > 0:
        cfg.graph_encoder_lora_rank = args.graph_encoder_lora_rank
        print(f"[graph override] encoder-LoRA rank = {cfg.graph_encoder_lora_rank}")
    if args.graph_read_final:
        cfg.graph_read_final = True
        print("[graph override] reading FINAL hidden (full forward)")
    if args.graph_free_endpoints:
        cfg.graph_free_endpoints = True
        print("[graph override] FREE endpoints (no bank/selection)")
    if args.slotgraph_no_structure:
        cfg.slotgraph_use_structure = False
        print("[slotgraph override] structure OFF = plain prepend of id-tagged slots (id-tagged ICAE; "
              "true pure-ICAE is the icae_baseline variant)")
    cfg.task_mode = args.task        # accurate ckpt metadata (dispatch still keys on this)
    cfg.seed = args.seed             # record the actual seed in ckpt metadata
    cfg.anomaly_from = args.anomaly_from   # debug: backward anomaly detection from this step (-1 = off)

    # Auto-pick BABILong config to match chunk_size (audit fix #10).
    if args.babilong_config == "auto":
        # Map chunk_size to nearest BABILong config at or below it.
        cs = args.chunk_size
        if cs >= 16384:
            args.babilong_config = "16k"
        elif cs >= 8192:
            args.babilong_config = "8k"
        elif cs >= 4096:
            args.babilong_config = "4k"
        elif cs >= 2048:
            args.babilong_config = "2k"
        elif cs >= 1024:
            args.babilong_config = "1k"
        else:
            args.babilong_config = "0k"
        if args.babilong:
            print(f"[auto] babilong_config = {args.babilong_config} "
                  f"(scaled for chunk_size={args.chunk_size})")

    # Auto-scale composite passages_per_chunk with chunk_size if user passed 0.
    # composite_v1 passages average ~13 tokens; we target ~75 passages per
    # 1024 chunk tokens so the chunk fills to ~95% even after rejecting
    # over-long candidates.
    if args.passages_per_chunk <= 0:
        args.passages_per_chunk = max(75, (args.chunk_size // 1024) * 75)
        print(f"[auto] composite passages_per_chunk = {args.passages_per_chunk} "
              f"(scaled for chunk_size={args.chunk_size})")

    print(f"config: chunk={args.chunk_size}, window={args.window_size}, "
          f"passages_per_chunk={args.passages_per_chunk}")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")

    print(f"\nLoading tokenizer {cfg.llama_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading Llama (shared across variants, frozen)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    if args.probe_bs:
        print(f"\n[probe-bs] per-arm max batch size, conditioned-reconstruction N={args.cond_recon_n_pairs}, "
              f"chunk={args.chunk_size}, mem_tokens={M}")
        probe_bs(args.variants, llama, tokenizer, cfg, args)
        return

    summaries = []
    # MIXED multi-task: vanilla floor/ceiling have no trainable params in the mixed
    # round-robin (they are single-task eval-only references) — skip them.
    if args.task == "mixed":
        mixed_variants = [v for v in args.variants if not v.startswith("vanilla_")]
        skipped = [v for v in args.variants if v.startswith("vanilla_")]
        if skipped:
            print(f"[mixed] skipping eval-only vanilla references: {skipped}")
        for variant in mixed_variants:
            out_dir = REPO / f"outputs/memory/{args.out_tag}_{variant}"
            out_dir.mkdir(parents=True, exist_ok=True)
            s = train_mixed_variant(
                variant=variant, llama=llama, tokenizer=tokenizer, cfg=cfg,
                n_steps=args.steps, log_every=args.log_every,
                val_every=args.val_every, save_every=args.save_every,
                val_batches=args.val_batches, out_dir=out_dir,
                window_size=args.window_size,
                mixed_tasks=tuple(args.mixed_tasks), mixed_ctx=args.mixed_ctx,
                mixed_M=args.mixed_M, babi_tasks=tuple(args.babi_tasks),
                predict_len=args.predict_len, mae_src_tok=args.src_tokenizer,
                resume=args.resume,
            )
            summaries.append(s)
    else:
        for variant in args.variants:
            out_dir = REPO / f"outputs/memory/{args.out_tag}_{variant}"
            out_dir.mkdir(parents=True, exist_ok=True)
            s = train_one_variant(
                variant=variant, llama=llama, tokenizer=tokenizer, cfg=cfg,
                n_steps=args.steps, log_every=args.log_every,
                val_every=args.val_every, save_every=args.save_every,
                val_batches=args.val_batches, out_dir=out_dir,
                chunk_size=args.chunk_size, window_size=args.window_size,
                passages_per_chunk=args.passages_per_chunk,
                resume=args.resume,
                use_hotpot=not args.no_hotpot,
                use_narrative=args.narrative,
                use_musique=args.musique,
                use_babilong=args.babilong,
                babilong_config=args.babilong_config,
                mix_weights=tuple(args.mix_weights),
                composite_task_weights=composite_task_weights,
                patience=args.patience,
                min_step_for_stop=args.min_step_for_stop,
                min_delta=args.early_stop_min_delta,
                task=args.task,
                cond_recon_n_pairs=args.cond_recon_n_pairs, cond_recon_n_query=args.cond_recon_n_query,
                cond_recon_value_len=args.cond_recon_value_len,
                cond_recon_bio_n_facts=args.cond_recon_bio_n_facts, cond_recon_bio_world_seed=args.cond_recon_bio_world_seed,
                babi_tasks=tuple(args.babi_tasks),
                compress_len=args.compress_len, predict_len=args.predict_len,
                mae_src_tok=args.src_tokenizer,
            )
            summaries.append(s)

    print("\n" + "=" * 78)
    print("v1h SUMMARY")
    print("=" * 78)
    if args.task == "mixed":
        print(f"  {'variant':<25}{'params':>12}   per-task final val_loss")
        print("  " + "-" * 70)
        for s in summaries:
            pt = "  ".join(f"{t}={s['per_task_final'][t]['val_loss']:.3f}"
                           for t in s["mixed_tasks"])
            print(f"  {s['variant']:<25}{s['trainable_params']:>12,}   {pt}")
    else:
        print(f"  {'variant':<25}{'params':>12}{'final_recon':>13}{'top1':>8}{'time(min)':>12}")
        print("  " + "-" * 70)
        for s in summaries:
            print(f"  {s['variant']:<25}{s['trainable_params']:>12,}"
                  f"{s['final_val_loss_recon']:>13.4f}"
                  f"{s['final_val_top1_acc']*100:>7.1f}%"
                  f"{s['elapsed_s']/60:>12.1f}")

    summary_path = REPO / f"outputs/memory/{args.out_tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
