"""Per-family AR-decode EM/F1 evaluation across trained variants.

For each task family (biographical, hotpot_qa, narrative_qa, musique) we:
  1. Collect N samples from the corresponding val dataset
  2. For each variant: encode chunks → memory → greedy AR-decode the answer
  3. Score predictions with SQuAD-style normalized EM/F1 (max over refs)
  4. Output a per-family × per-variant table + JSONL dump

The same (context, question) is held fixed across variants so the only
varying factor is the encoder. Frozen Llama-3.2-1B is shared across all
variants to keep GPU memory bounded.

Usage:
    # Default: auto-discover tranche-3 ckpts, 64 samples per family
    python scripts/repr_learning/eval_per_family.py

    # Custom sample count + variants
    python scripts/repr_learning/eval_per_family.py \\
        --n-per-family 128 --variants graph_v5_baseline flat_baseline

    # Custom ckpt locations
    python scripts/repr_learning/eval_per_family.py \\
        --ckpt-pattern 'outputs/repr_learning/outputs/repr_learning/tranche3_{v}_{v}/ckpts/{v}.best.pt'
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/alex/code/neuromorphic")
sys.path.insert(0, str(ROOT))

from src.repr_learning.config import ReprConfig                        # noqa: E402
from src.repr_learning.model import ReprLearningModel                  # noqa: E402
from src.repr_learning.data_qa import (                                # noqa: E402
    HotpotQADataset, MuSiQueDataset, NarrativeQADataset, QADataset,
    BABILongDataset, RULERNIAHDataset, LoCoMoQADataset,
    collate_qa,
)

COMPOSITE_VAL_P = ROOT / "data/wave1/composite_v1/val/passages.jsonl"
COMPOSITE_VAL_Q = ROOT / "data/wave1/composite_v1/val/questions.jsonl"

# v2.1 joint sweep — all 7 arms scored with identical decode params so the
# floor/ceiling are produced in the SAME run as the comparison arms (EM/
# Containment/Judge), not only on reconstruction NLL.
DEFAULT_VARIANTS = [
    "graph_v6_baseline",     # primary
    "flat_baseline",
    "continuous_baseline",
    "recurrent_baseline",
    "memorizing_baseline",
    "vanilla_llama",         # floor (no context)
    "vanilla_full_context",  # ceiling (full evidence)
]

# Doubly-nested path comes from `--out outputs/repr_learning/tranche4_<v>`
# becoming `--out-tag outputs/repr_learning/tranche4_<v>` and then prepending
# `outputs/repr_learning/` again in train_repr_qa.py.
DEFAULT_CKPT_PATTERN = "outputs/repr_learning/outputs/repr_learning/tranche4_{v}_{v}/ckpts/{v}.best.pt"

FAMILY_TO_BUILDER = {
    "biographical": "composite",
    "hotpot_qa":    "hotpot",
    "narrative_qa": "narrative",
    "musique":      "musique",
    "babilong":     "babilong",
    # OOD (eval-only, never trained on):
    #   ruler_niah — synthetic needle-in-haystack, contamination-free, 8k.
    #   locomo — real very-long-term dialogue (~9–26k tok); run at a LARGER
    #            --chunk-size (e.g. 32768) so the streaming encoder windows the
    #            whole conversation into the O(1) memory (length-gen probe).
    "ruler_niah":   "ruler_niah",
    "locomo":       "locomo",
}


# ═══════════════════════════════════════════════════════════════════════════
# SQuAD-style normalization + EM/F1 (adapted from official SQuAD eval script)
# ═══════════════════════════════════════════════════════════════════════════

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower, strip articles, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = _ARTICLES_RE.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)  # both empty → 1.0, else 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_score(pred: str, gold: str) -> float:
    """Token recall — fraction of gold tokens present in pred (the recall half of
    F1). Verbosity-robust: extra pred tokens don't penalize. The capped generation
    length guards against gaming by text-dumping."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not g_tokens:
        return float(not p_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    return sum(common.values()) / len(g_tokens)


def containment_score(pred: str, gold: str) -> float:
    """HEADLINE correctness: 1.0 iff the normalized gold answer appears as a
    contiguous run of WHOLE tokens in the normalized prediction. Verbosity-robust
    (ignores surrounding framing) AND factually strict — whole-word, so '4' does NOT
    match inside '34' and 'cab' does not match 'cabinet'. See feedback-qa-correctness-metric."""
    g = normalize_answer(gold).split()
    if not g:
        return 0.0
    p = normalize_answer(pred).split()
    n = len(g)
    return float(any(p[i:i + n] == g for i in range(len(p) - n + 1)))


def max_over_refs(pred: str, refs: list[str], fn) -> float:
    if not refs:
        return 0.0
    return max(fn(pred, r) for r in refs)


# ═══════════════════════════════════════════════════════════════════════════
# Sample collection — per-family, fixed across variants
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalSample:
    family: str
    context_ids: torch.Tensor       # [T_ctx]
    context_mask: torch.Tensor      # [T_ctx]
    question_ids: torch.Tensor      # [T_q]
    answer_ids: torch.Tensor        # [T_a] — used for max_new_tokens budget
    answer_refs: list[str]          # 1+ reference answer strings


def _build_dataset(builder: str, *, tokenizer, chunk_size: int, cfg: ReprConfig,
                   seed: int, passages_per_chunk: int,
                   composite_family: Optional[str] = None):
    """Construct the underlying val dataset for a single family."""
    if builder == "composite":
        # task_weights restricts CompositeSampler to a single family so we
        # don't burn scans on off-family questions. Matches the training
        # protocol (--composite-task-weights biographical:1.0).
        task_weights = ({composite_family: 1.0} if composite_family
                         else None)
        return QADataset(
            COMPOSITE_VAL_P, COMPOSITE_VAL_Q,
            chunk_size=chunk_size,
            passages_per_chunk=passages_per_chunk,
            sep_token_id=cfg.sep_token_id,
            pad_token_id=cfg.pad_token_id,
            task_weights=task_weights,
            seed=seed,
        )
    if builder == "hotpot":
        return HotpotQADataset(
            split="validation", tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
        )
    if builder == "narrative":
        return NarrativeQADataset(
            split="validation", tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
        )
    if builder == "musique":
        return MuSiQueDataset(
            split="validation", tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
        )
    if builder == "babilong":
        # OOD: eval-only (not in the train mix). Length config matches chunk_size.
        return BABILongDataset(
            split="validation", tokenizer=tokenizer, chunk_size=chunk_size,
            config_name=f"{max(1, chunk_size // 1024)}k",
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
        )
    if builder == "ruler_niah":
        # OOD: synthetic multi-key needle-in-haystack, never trained on.
        return RULERNIAHDataset(
            split="validation", tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
        )
    if builder == "locomo":
        # OOD: real very-long-term dialogue. Native length ~9–26k > 8k — run
        # this family at a larger --chunk-size so the streaming encoder windows
        # the whole conversation into the fixed-footprint memory.
        return LoCoMoQADataset(
            split="validation", tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
        )
    raise ValueError(f"unknown builder: {builder}")


def collect_samples(families: list[str], n_per_family: int, *,
                    tokenizer, cfg: ReprConfig, chunk_size: int,
                    passages_per_chunk: int, max_scan_mult: int = 50,
                    ) -> list[EvalSample]:
    """Pull `n_per_family` samples from each requested family, in order."""
    out: list[EvalSample] = []
    for fam in families:
        builder = FAMILY_TO_BUILDER[fam]
        print(f"[collect] {fam} (builder={builder}, n={n_per_family})")
        composite_family = fam if builder == "composite" else None
        # Stable cross-process seed: Python's hash() is salted per-process
        # unless PYTHONHASHSEED is set. Use a hashlib digest so re-runs and
        # re-launches see the same samples.
        import hashlib
        fam_seed = int(hashlib.md5(fam.encode()).hexdigest()[:4], 16)
        ds = _build_dataset(
            builder, tokenizer=tokenizer, chunk_size=chunk_size, cfg=cfg,
            seed=fam_seed, passages_per_chunk=passages_per_chunk,
            composite_family=composite_family,
        )
        it = iter(ds)
        scanned = 0
        added = 0
        max_scan = n_per_family * max_scan_mult
        while added < n_per_family and scanned < max_scan:
            try:
                s = next(it)
            except StopIteration:
                break
            scanned += 1
            # The underlying datasets sometimes yield off-family rows when
            # builder maps 1:1 (eg composite). Filter to be safe. BABILong
            # tags per-task ("babilong_qa1"), so accept the "<fam>_*" prefix too.
            tf = s.get("task_family", "")
            if tf != fam and not tf.startswith(fam + "_"):
                continue
            refs = s.get("answer_refs") or []
            if not refs:
                # Fallback: decode the answer_ids tensor we'd train on
                refs = [tokenizer.decode(s["answer_ids"].tolist(),
                                         skip_special_tokens=True).strip()]
            out.append(EvalSample(
                family=fam,
                context_ids=s["context_ids"],
                context_mask=s["context_mask"],
                question_ids=s["question_ids"],
                answer_ids=s["answer_ids"],
                answer_refs=[r.strip() for r in refs if r and r.strip()],
            ))
            added += 1
        if added < n_per_family:
            print(f"[collect]   WARN: only {added}/{n_per_family} samples "
                  f"after {scanned} scans for {fam}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-variant model loading + AR-decode
# ═══════════════════════════════════════════════════════════════════════════


# Variant → allowed-unexpected state-dict keys for strict-ish load. (graph_v5
# is retired/deleted; no live arm needs an allowance.)
VARIANT_ALLOW_UNEXPECTED: dict[str, set] = {}


def _verify_ckpt_metadata(variant: str, sd: dict, cfg: ReprConfig, model):
    """Verify ckpt's saved identity metadata matches the current env.

    Permissive: if ckpt has no metadata (legacy ckpts pre-tranche-4), warn
    but allow load. If metadata present, abort on backbone or scaffold-hash
    mismatch — those would silently invalidate results.
    """
    meta = sd.get("metadata")
    if meta is None:
        print(f"   [warn] {variant}: ckpt has no metadata (likely pre-tranche-4) "
              f"— cannot verify backbone/scaffold match")
        return
    ckpt_backbone = meta.get("backbone_model")
    if ckpt_backbone and ckpt_backbone != cfg.llama_model:
        raise RuntimeError(
            f"{variant}: backbone mismatch — ckpt trained against "
            f"{ckpt_backbone!r} but eval cfg uses {cfg.llama_model!r}. "
            f"Refusing to silently mix backbones."
        )
    ct = getattr(model, "chat_template", None)
    if ct is not None:
        ckpt_hash = meta.get("chat_scaffold_hash")
        if ckpt_hash:
            import hashlib
            scaffold_bytes = b"".join([
                ct.pre_memory_ids.numpy().tobytes(),
                ct.post_memory_ids.numpy().tobytes(),
                ct.post_question_ids.numpy().tobytes(),
                ct.eot_id.to_bytes(8, "little", signed=False) if isinstance(ct.eot_id, int) else b"",
            ])
            now_hash = hashlib.sha256(scaffold_bytes).hexdigest()
            if now_hash != ckpt_hash:
                raise RuntimeError(
                    f"{variant}: chat-scaffold hash mismatch — ckpt "
                    f"{ckpt_hash[:12]}... vs current {now_hash[:12]}.... "
                    f"Date drift? system_intro change? Refusing to evaluate "
                    f"under different scaffold than training."
                )


def _frozen_base_key(k: str) -> bool:
    """A frozen-Llama BASE weight (not saved in ckpt): decoder.llama.* without
    a LoRA adapter in the name. LoRA keys (decoder.llama.*lora*) ARE saved and
    must match, so they are NOT treated as freely-missing."""
    return k.startswith("decoder.llama.") and "lora" not in k.lower()


def load_variant(variant: str, ckpt_path: Path, base_cfg: ReprConfig, llama):
    """Load one variant, rebuilding its ReprConfig from the checkpoint's OWN
    pinned cfg_dict so eval sizing + LoRA exactly match what that arm trained
    with. The eval-side cfg must NOT be hardcoded — it drifts from the trainer
    and silently crash-skips every fixed-footprint baseline (audit flat-01)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{variant}: ckpt missing at {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = (sd.get("metadata") or {}).get("cfg_dict")
    if cfg_dict:
        valid = {f.name for f in fields(ReprConfig)}
        cfg = ReprConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    else:
        print(f"   [warn] {variant}: ckpt has no metadata.cfg_dict — falling back "
              f"to eval default cfg; sizing may mismatch. Retrain to embed cfg_dict.")
        cfg = base_cfg
    # LoRA-all: each variant self-loads a FRESH frozen Llama (passing None) so the
    # shared module isn't LoRA-wrapped in place across variants (double-wrap). Only
    # share the base module when this variant has no LoRA.
    llama_arg = None if getattr(cfg, "use_llama_lora", False) else llama
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg)
    _verify_ckpt_metadata(variant, sd, cfg, model)
    state = sd["model_state_dict"]
    allow = VARIANT_ALLOW_UNEXPECTED.get(variant, set())
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Frozen base weights are legitimately absent (not saved); LoRA + memory keys
    # must match. Anything else missing/unexpected is real ckpt drift → abort.
    bad_missing = [k for k in missing if not _frozen_base_key(k)]
    bad_unexpected = [k for k in unexpected
                      if k not in allow and not _frozen_base_key(k)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"{variant} ckpt drift:\n"
            f"  missing[:5]={bad_missing[:5]}\n"
            f"  unexpected[:5]={bad_unexpected[:5]}"
        )
    model.train(False)
    return model, int(sd.get("step", -1))


def _stream_encode_batch(model, batch_samples: list[EvalSample],
                          device, window_size: int) -> tuple[torch.Tensor, dict]:
    """Streaming encode of a B-batch of samples → memory tokens."""
    enc = model.encoder
    embed = model.decoder.llama.get_input_embeddings()

    # Stack context: all samples in a family share chunk_size.
    ctx = torch.stack([s.context_ids for s in batch_samples]).to(device)
    mask = torch.stack([s.context_mask for s in batch_samples]).to(device)
    B, T = ctx.shape

    with torch.no_grad():
        token_embeds = embed(ctx)
        state = enc.init_streaming_state(B, device=device,
                                          dtype=token_embeds.dtype)
        for s in range(0, T, window_size):
            e = min(s + window_size, T)
            state, _ = enc.streaming_write(
                state, token_embeds[:, s:e, :],
                attention_mask=mask[:, s:e],
                chunk_offset=s,
            )
        # v5.6: hand the question to the encoder (graph_v5 reads it for the
        # question-conditioned readout). NullEncoder/Mamba use a non-dict state
        # → guard the whole stash (also skips wasted work for those variants).
        if isinstance(state, dict):
            T_q = max(int(bs.question_ids.shape[0]) for bs in batch_samples)
            qc_ids = torch.full((B, T_q), cfg_pad_id(model), dtype=torch.long, device=device)
            qc_mask = torch.zeros((B, T_q), dtype=torch.bool, device=device)
            for i, bs in enumerate(batch_samples):
                tq = int(bs.question_ids.shape[0])
                qc_ids[i, :tq] = bs.question_ids.to(device)
                qc_mask[i, :tq] = True
            state["question_embeds"] = embed(qc_ids)
            state["question_mask"] = qc_mask
        memory, finalize_aux = enc.finalize_memory(state)
        mt_bank = finalize_aux.get("mt_bank")
        if mt_bank is not None:
            # MT retrieves per-question — needs question embeds.
            # Pad-stack questions for the batch.
            T_q = max(int(s.question_ids.shape[0]) for s in batch_samples)
            q_ids = torch.full((B, T_q), cfg_pad_id(model), dtype=torch.long,
                                device=device)
            q_mask = torch.zeros((B, T_q), dtype=torch.bool, device=device)
            for i, s in enumerate(batch_samples):
                tq = int(s.question_ids.shape[0])
                q_ids[i, :tq] = s.question_ids.to(device)
                q_mask[i, :tq] = True
            q_emb = embed(q_ids)
            memory, _ = enc.retrieve_for_query(
                mt_bank, q_emb, q_mask, K=model.cfg.n_flat_codes,
            )
    return memory, finalize_aux


def cfg_pad_id(model) -> int:
    return int(model.cfg.pad_token_id)


# Per-family max_new_tokens budget. Short-answer families (hotpot, musique)
# only need ~16 tokens; templated biographical answers need ~32; narrative
# answers can be longer. Keeping it tight reduces ramble fodder.
MAX_NEW_TOKENS_PER_FAMILY = {
    "biographical": 32,
    "hotpot_qa":    16,
    "narrative_qa": 48,
    "musique":      16,
    "babilong":     12,   # bAbI answers are 1-2 words
    "ruler_niah":   12,   # a 7-digit number
    "locomo":       32,   # dates / names / short phrases; some 1-sentence
}


def _truncate_at_natural_end(text: str) -> str:
    """Aggressively cut model output at the first 'this is clearly done' signal.
    Order of cuts (whichever fires first):
      1. First newline (model often emits Q/A pairs after the answer)
      2. First sentence-end period followed by whitespace+capital (continues
         into a new sentence — model has finished the answer and moved on)
      3. First detected 4-gram repetition cycle (collapse loop)
    """
    # 1. Newline cut
    if "\n" in text:
        text = text.split("\n", 1)[0]
    # 2. Sentence-end cut: ". X" where X is uppercase letter starting a new
    #    sentence. Keep the trailing period of the first sentence.
    m = re.search(r"\.\s+[A-Z]", text)
    if m is not None:
        text = text[:m.start() + 1]  # keep the period
    # 3. Repetition-cycle cut: detect any 4-token n-gram that repeats ≥2× in
    #    the output. If found, truncate to the first occurrence's end.
    toks = text.split()
    if len(toks) >= 8:
        for i in range(len(toks) - 7):
            gram = tuple(toks[i:i+4])
            for j in range(i + 4, len(toks) - 3):
                if tuple(toks[j:j+4]) == gram:
                    # Cycle detected; truncate to end of first occurrence
                    cut_words = toks[:i + 4]
                    text = " ".join(cut_words)
                    return text.strip()
    return text.strip()


@torch.no_grad()
def generate_answers(llama, tokenizer, memory: torch.Tensor,
                      batch_samples: list[EvalSample],
                      max_new_tokens: int, device,
                      memory_mask: Optional[torch.Tensor] = None,
                      adaptive_budget: bool = True,
                      chat_template=None,   # ChatTemplate or None
                      inject=None,          # graph_v6 per-token read: {encoder,facts,layer_idx}
                      ) -> tuple[list[str], list[str]]:
    """Greedy AR-decode per sample. When chat_template is set, prefix is:
       [pre_mem; memory; post_mem; question; post_q]
    Otherwise the legacy [memory; question] concat is used.

    `llama` MUST be the per-variant LoRA-wrapped decoder (model.decoder.llama),
    not the shared base — otherwise the trained adapter is dropped. For graph_v6,
    `inject` carries the per-token MemInject hook spec so the read mechanism is
    actually exercised at decode (mirrors model.compute_qa_loss); without it the
    primary arm decodes with ZERO memory (audit graph-1).

    Returns (raw_texts, cleaned_texts) — raw kept for audit, cleaned scored.
    """
    embed = llama.get_input_embeddings()

    # Pre-embed chat scaffold tokens once (constant per template).
    if chat_template is not None:
        pre_mem_embeds = embed(chat_template.pre_memory_ids.to(device))   # [L_pre, d]
        post_mem_embeds = embed(chat_template.post_memory_ids.to(device))  # [L_post_mem, d]
        post_q_embeds = embed(chat_template.post_question_ids.to(device))  # [L_post_q, d]
        # Use the chat-template EOT as stop signal (Llama-3 Instruct eos and
        # eot share id 128009; Qwen/etc differ — chat_template.eot_id is the
        # authoritative one).
        stop_token_id = chat_template.eot_id
    else:
        pre_mem_embeds = post_mem_embeds = post_q_embeds = None
        stop_token_id = tokenizer.eos_token_id

    raw_outs: list[str] = []
    clean_outs: list[str] = []
    for i, s in enumerate(batch_samples):
        # Adaptive per-family budget caps the rope the model gets.
        mnt = (MAX_NEW_TOKENS_PER_FAMILY.get(s.family, max_new_tokens)
               if adaptive_budget else max_new_tokens)
        q = s.question_ids.to(device).unsqueeze(0)
        q_emb = embed(q)

        parts = []   # list of [1, L_x, d] tensors to concat along dim=1
        masks = []   # corresponding [1, L_x] long masks
        if chat_template is not None:
            parts.append(pre_mem_embeds.unsqueeze(0).to(q_emb.dtype))
            masks.append(torch.ones(1, pre_mem_embeds.shape[0],
                                     dtype=torch.long, device=device))

        if memory.shape[1] > 0:
            m_i = memory[i:i+1].to(q_emb.dtype)
            parts.append(m_i)
            if memory_mask is not None:
                masks.append(memory_mask[i:i+1, :m_i.shape[1]].to(torch.long).to(device))
            else:
                masks.append(torch.ones(m_i.shape[:2], dtype=torch.long, device=device))

        if chat_template is not None:
            parts.append(post_mem_embeds.unsqueeze(0).to(q_emb.dtype))
            masks.append(torch.ones(1, post_mem_embeds.shape[0],
                                     dtype=torch.long, device=device))

        parts.append(q_emb)
        masks.append(torch.ones(q_emb.shape[:2], dtype=torch.long, device=device))

        if chat_template is not None:
            parts.append(post_q_embeds.unsqueeze(0).to(q_emb.dtype))
            masks.append(torch.ones(1, post_q_embeds.shape[0],
                                     dtype=torch.long, device=device))

        full = torch.cat(parts, dim=1)
        attn = torch.cat(masks, dim=1)

        # graph_v6: install the SAME per-token MemInject pre-hook used in
        # compute_qa_loss so the read injects this sample's facts at every
        # decode position. Per-sample facts slice [i:i+1]; removed in finally.
        hook_handle = None
        if inject is not None:
            enc_ref = inject["encoder"]
            # facts is a dict of [B,...] tensors (e.g. {"value": ...}); slice to
            # this sample. Non-tensor entries pass through unchanged.
            facts_i = {k: (v[i:i+1] if torch.is_tensor(v) else v)
                       for k, v in inject["facts"].items()}
            lidx = inject["layer_idx"]

            def _pre_hook(module, args, kwargs, _enc=enc_ref, _f=facts_i):
                if not args:
                    return None
                hs = args[0]
                return (_enc.inject(hs, _f),) + args[1:], kwargs

            hook_handle = llama.model.layers[lidx].register_forward_pre_hook(
                _pre_hook, with_kwargs=True)
        try:
            gen = llama.generate(
                inputs_embeds=full, attention_mask=attn,
                max_new_tokens=mnt, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=stop_token_id,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        # `generate` with inputs_embeds returns only NEW tokens (no prefix).
        raw = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True).strip()
        clean = _truncate_at_natural_end(raw)
        raw_outs.append(raw)
        clean_outs.append(clean)
    return raw_outs, clean_outs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    help="variants to evaluate (must have matching ckpts)")
    ap.add_argument("--families", nargs="+",
                    default=["biographical", "hotpot_qa", "narrative_qa", "musique"],
                    choices=list(FAMILY_TO_BUILDER.keys()))
    ap.add_argument("--n-per-family", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="encode + decode batch size (lower if OOM)")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--chunk-size", type=int, default=8192,
                    help="must match training chunk_size for these ckpts")
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--passages-per-chunk", type=int, default=0,
                    help="composite passages_per_chunk (0 = auto-scale)")
    ap.add_argument("--ckpt-pattern", default=DEFAULT_CKPT_PATTERN,
                    help="format string with {v} placeholder for variant name")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs/repr_learning/eval_per_family")
    ap.add_argument("--tag", default="tranche3",
                    help="output filename prefix")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-scale composite passages to fill the chunk (matches train_repr_qa.py).
    if args.passages_per_chunk <= 0:
        args.passages_per_chunk = max(75, (args.chunk_size // 1024) * 75)
        print(f"[auto] composite passages_per_chunk = {args.passages_per_chunk}")

    # Match training-time config (tranche-3 / v5.5 — see train_repr_qa.py).
    cfg = ReprConfig(
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=128,
        d_continuous=1398, d_concept_baseline=1398,
        d_mt_value=1398, d_recurrent=1398,
        graph_v5_K_node=128, graph_v5_K_edge=196, graph_v5_K_proposal=196,
        graph_v5_d_node=384,
        graph_v5_d_state=384,
        graph_v5_d_updater=640,
        graph_v5_updater_layers=5,
        graph_v5_n_message_rounds=6,
        graph_v5_mp_d_hidden=1024,
        d_enc=768,
        enc_n_layers=4,   # MUST match train_repr_qa.py (4 → ~49M baselines, matched)
        enc_n_heads=12,
        enc_ffn_dim=3072,
        d_mamba=1280,   # MUST match train_repr_qa.py v5.5 override (matched to graph ~48.6M)
        edge_token_packing="fused",
    )
    # NOTE (2026-05-29): this override block DUPLICATES train_repr_qa.py's — they
    # drifted (d_mamba 1792 here vs 1280 there) and silently broke ckpt loading.
    # TODO: extract a single shared config builder (audit Major #3).

    print(f"[cfg] chunk={args.chunk_size}, window={args.window_size}, "
          f"M={cfg.n_flat_codes}, families={args.families}, "
          f"n_per_family={args.n_per_family}")

    print(f"\n[tokenizer] loading {cfg.llama_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Collect samples (shared across variants) ─────────────────────────
    t0 = time.time()
    samples = collect_samples(
        args.families, args.n_per_family,
        tokenizer=tokenizer, cfg=cfg,
        chunk_size=args.chunk_size,
        passages_per_chunk=args.passages_per_chunk,
    )
    print(f"[collect] {len(samples)} total samples in {time.time()-t0:.1f}s")

    # ── Load frozen Llama once, share across variants ────────────────────
    # Shared frozen Llama is only a FALLBACK for non-LoRA variants; under LoRA-all
    # every variant self-loads its own (bf16) decoder in load_variant. Load bf16 to
    # match the training/val numerical regime (was fp32 → train/eval dtype mismatch).
    print(f"\n[llama] loading {cfg.llama_model} (shared fallback, frozen, bf16)")
    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.bfloat16)
    llama.train(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama = llama.to(device)

    # ── Per-variant evaluation ────────────────────────────────────────────
    # Structure: results[variant][family] = list of dicts {pred, refs, em, f1}
    results: dict[str, dict[str, list[dict]]] = {}
    variant_steps: dict[str, int] = {}

    for variant in args.variants:
        ckpt_path = ROOT / args.ckpt_pattern.format(v=variant)
        print(f"\n══ variant: {variant}")
        print(f"   ckpt: {ckpt_path}")
        try:
            model, step = load_variant(variant, ckpt_path, cfg, llama)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"   SKIP: {e}")
            continue
        model = model.to(device)
        variant_steps[variant] = step
        print(f"   loaded @ step {step}")
        results[variant] = {fam: [] for fam in args.families}

        # Batch by family (so chunk_size + memory_mask logic is uniform per fam)
        by_family: dict[str, list[EvalSample]] = defaultdict(list)
        for s in samples:
            by_family[s.family].append(s)

        for fam, fam_samples in by_family.items():
            n = len(fam_samples)
            for start in range(0, n, args.batch_size):
                batch = fam_samples[start:start + args.batch_size]
                memory, finalize_aux = _stream_encode_batch(
                    model, batch, device, args.window_size,
                )
                mem_mask = finalize_aux.get("memory_mask")
                # graph_v6 reads via a per-token inject hook (not prepend) — pass
                # the spec so decode exercises the actual read mechanism.
                inject = None
                facts = finalize_aux.get("graph_v6_facts")
                if facts is not None:
                    inject = {"encoder": model.encoder, "facts": facts,
                              "layer_idx": model.encoder.inject_layer_idx}
                # Decode with THIS variant's LoRA-wrapped Llama (not the shared
                # base) so the trained rank-16 adapter is applied.
                raw_preds, clean_preds = generate_answers(
                    model.decoder.llama, tokenizer, memory.detach(), batch,
                    args.max_new_tokens, device, memory_mask=mem_mask,
                    chat_template=getattr(model, "chat_template", None),
                    inject=inject,
                )
                for i, s in enumerate(batch):
                    raw = raw_preds[i]
                    clean = clean_preds[i]
                    em = max_over_refs(clean, s.answer_refs, em_score)
                    contain = max_over_refs(clean, s.answer_refs, containment_score)
                    recall = max_over_refs(clean, s.answer_refs, recall_score)
                    f1 = max_over_refs(clean, s.answer_refs, f1_score)
                    results[variant][fam].append({
                        "raw": raw,
                        "pred": clean,
                        "refs": s.answer_refs,
                        "em": em,
                        "contain": contain,
                        "recall": recall,
                        "f1": f1,
                    })
            n_scored = len(results[variant][fam])

            def _avg(k):
                return sum(r[k] for r in results[variant][fam]) / max(1, n_scored)
            print(f"   {fam:14s}  n={n_scored:3d}  "
                  f"CORR={_avg('contain')*100:5.1f}  EM={_avg('em')*100:5.1f}  "
                  f"REC={_avg('recall')*100:5.1f}  F1={_avg('f1')*100:5.1f}"
                  f"   (CORR=containment=headline)")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"PER-FAMILY CORRECTNESS=containment / EM  (n={args.n_per_family} per "
          f"family, greedy AR-decode, max_new_tokens={args.max_new_tokens})")
    print("  headline=containment (verbosity-robust + factually strict); "
          "macro=mean containment over families")
    print("=" * 80)

    # Header
    fam_cols = "  ".join(f"{fam:>20s}" for fam in args.families)
    print(f"\n  {'variant':25s}  {'step':>7s}  {'macro':>7s}  {fam_cols}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*7}  "
          + "  ".join("-" * 20 for _ in args.families))

    for variant in args.variants:
        if variant not in results:
            continue
        step = variant_steps.get(variant, -1)
        cells = []
        con_per_fam = []
        for fam in args.families:
            rs = results[variant][fam]
            if not rs:
                cells.append(f"{'—':>20s}")
                continue
            con = sum(r["contain"] for r in rs) / len(rs)
            em = sum(r["em"] for r in rs) / len(rs)
            con_per_fam.append(con)
            cells.append(f"  C={con*100:5.1f}  EM={em*100:5.1f}")
        macro = sum(con_per_fam) / len(con_per_fam) if con_per_fam else 0.0
        print(f"  {variant:25s}  {step:>7d}  {macro*100:6.1f}  "
              + "  ".join(cells))
    print()

    # ── Persist outputs ───────────────────────────────────────────────────
    summary = {
        "tag": args.tag,
        "chunk_size": args.chunk_size,
        "window_size": args.window_size,
        "n_per_family": args.n_per_family,
        "max_new_tokens": args.max_new_tokens,
        "families": args.families,
        "variants": list(results.keys()),
        "variant_steps": variant_steps,
        "headline_metric": "containment",
        "by_variant": {
            v: {
                fam: {
                    "n": len(rs),
                    "containment": sum(r["contain"] for r in rs) / max(1, len(rs)),
                    "em": sum(r["em"] for r in rs) / max(1, len(rs)),
                    "recall": sum(r["recall"] for r in rs) / max(1, len(rs)),
                    "f1": sum(r["f1"] for r in rs) / max(1, len(rs)),
                }
                for fam, rs in fam_results.items()
            }
            for v, fam_results in results.items()
        },
        "macro_containment": {
            v: (
                sum(sum(r["contain"] for r in rs) / max(1, len(rs))
                    for rs in fam_results.values() if rs)
                / max(1, sum(1 for rs in fam_results.values() if rs))
            )
            for v, fam_results in results.items()
        },
    }
    summary_path = args.out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[output] summary  → {summary_path}")

    # Per-sample JSONL for re-analysis
    per_sample_path = args.out_dir / f"{args.tag}_per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for variant, fam_results in results.items():
            for fam, rs in fam_results.items():
                for r in rs:
                    f.write(json.dumps({
                        "variant": variant,
                        "family": fam,
                        **r,
                    }) + "\n")
    print(f"[output] per-sample → {per_sample_path}")


if __name__ == "__main__":
    main()
