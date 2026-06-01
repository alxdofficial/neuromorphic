"""[LEGACY — its VARIANTS list + ckpt paths target graph_v5/tranche-era runs
(now deleted). For the current v2.1/graph_v6 sweep, AR-decode eval is done by
eval_per_family.py. Kept for reference only.]
Autoregressive QA decode across multiple trained variants.

For each variant we hold the **same** input chunks fixed and ask:
  - Encode chunk → per-variant memory tokens
  - Prepend memory + question to Llama, greedy-generate the answer
  - Compare side by side

Variants that retrieve per-query (memorizing_baseline) call
encoder.retrieve_for_query(question_embeds) after streaming, mirroring
compute_qa_loss.

Filters to a single task_family if --family is passed (e.g. biographical).

Run:
    python scripts/repr_learning/decode_qa_multibaseline.py \\
        --family biographical --chunks 4 --max-new-tokens 40
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/alex/code/neuromorphic")
sys.path.insert(0, str(ROOT))

from src.repr_learning.config import ReprConfig                        # noqa: E402
from src.repr_learning.model import ReprLearningModel                  # noqa: E402
from src.repr_learning.data_qa import (                                # noqa: E402
    HotpotQADataset, MixedQADataset, QADataset, collate_qa,
)

COMPOSITE_VAL_P = ROOT / "data/wave1/composite_v1/val/passages.jsonl"
COMPOSITE_VAL_Q = ROOT / "data/wave1/composite_v1/val/questions.jsonl"

# Variant → (ckpt path, allowed-unexpected keys for strict load).
VARIANTS: dict[str, tuple[Path, set[str]]] = {
    "graph_v5_baseline":   (ROOT / "outputs/repr_learning/v5_4_first_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt",
                            {"encoder.soft_pointer.W_v.weight"}),
    "recurrent_baseline":  (ROOT / "outputs/repr_learning/v1h_t4k_v3_recurrent_baseline/ckpts/recurrent_baseline.best.pt",
                            set()),
    "continuous_baseline": (ROOT / "outputs/repr_learning/v1h_t4k_v3_continuous_baseline/ckpts/continuous_baseline.best.pt",
                            set()),
    "memorizing_baseline": (ROOT / "outputs/repr_learning/v1h_t4k_v3_memorizing_baseline/ckpts/memorizing_baseline.best.pt",
                            set()),
    "flat_baseline":       (ROOT / "outputs/repr_learning/v1h_t4k_v3_flat_baseline/ckpts/flat_baseline.best.pt",
                            set()),
}


def load_model(variant: str, cfg: ReprConfig, llama, allow_unexpected: set[str]):
    """Build ReprLearningModel(variant) and load encoder weights from ckpt.
    Pass in a shared llama so we only materialize the 1B model once."""
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama)
    ckpt_path = VARIANTS[variant][0]
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("decoder.llama.")]
    bad_unexpected = [k for k in unexpected if k not in allow_unexpected
                       and not k.startswith("decoder.llama.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"{variant} ckpt drift: missing={bad_missing[:5]} "
            f"unexpected={bad_unexpected[:5]}"
        )
    model.train(False)
    return model, sd.get("step", -1)


def encode_memory(model, batch, device, window_size: int = 1024):
    """Run model.encoder over the streamed chunk and return memory tokens.
    For memorizing_baseline, also performs per-question retrieval."""
    enc = model.encoder
    embed = model.decoder.llama.get_input_embeddings()
    with torch.no_grad():
        token_embeds = embed(batch.context_ids.to(device))
        T = token_embeds.shape[1]
        state = enc.init_streaming_state(token_embeds.shape[0],
                                          device=device, dtype=token_embeds.dtype)
        for s in range(0, T, window_size):
            e = min(s + window_size, T)
            state, _ = enc.streaming_write(
                state, token_embeds[:, s:e, :],
                attention_mask=batch.context_mask[:, s:e].to(device),
                chunk_offset=s,
            )
        # v5.6: hand the question to graph_v5's question-conditioned readout
        # (dict-state variants only; others ignore the extra keys).
        if isinstance(state, dict):
            state["question_embeds"] = embed(batch.question_ids.to(device))
            state["question_mask"] = batch.question_mask.to(device)
        memory, finalize_aux = enc.finalize_memory(state)
        # MT: retrieve per-query using the question embeddings.
        mt_bank = finalize_aux.get("mt_bank")
        if mt_bank is not None:
            q_embeds = embed(batch.question_ids.to(device))
            memory, _ = enc.retrieve_for_query(
                mt_bank, q_embeds, batch.question_mask.to(device),
                K=model.cfg.n_flat_codes,
            )
    return memory  # [B, M, d_llama]; may be [B, 0, d_llama] for vanilla


def generate_qa_answer(llama, tokenizer, memory: torch.Tensor,
                        question_ids: list[int], max_new_tokens: int,
                        device, chat_template=None) -> str:
    """Single-sample generation. When chat_template is set the prefix is
       [pre_mem; memory; post_mem; question; post_q]
    matching tranche-4+ training. Otherwise legacy [memory; question].
    """
    embed = llama.get_input_embeddings()
    q = torch.tensor(question_ids, dtype=torch.long, device=device).unsqueeze(0)
    q_emb = embed(q)
    parts = []
    if chat_template is not None:
        parts.append(embed(chat_template.pre_memory_ids.to(device)).unsqueeze(0).to(q_emb.dtype))
    if memory.shape[1] > 0:
        parts.append(memory.to(q_emb.dtype))
    if chat_template is not None:
        parts.append(embed(chat_template.post_memory_ids.to(device)).unsqueeze(0).to(q_emb.dtype))
    parts.append(q_emb)
    if chat_template is not None:
        parts.append(embed(chat_template.post_question_ids.to(device)).unsqueeze(0).to(q_emb.dtype))
    full = torch.cat(parts, dim=1) if parts else q_emb
    attn = torch.ones(full.shape[:2], dtype=torch.long, device=device)
    stop_token_id = (chat_template.eot_id if chat_template is not None
                      else tokenizer.eos_token_id)
    with torch.no_grad():
        gen = llama.generate(
            inputs_embeds=full, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=stop_token_id,
        )
    return tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=list(VARIANTS.keys()),
                    help="variants to evaluate")
    ap.add_argument("--family", default=None,
                    help="filter samples to this task_family "
                         "(e.g. biographical, hotpot_qa, boxes)")
    ap.add_argument("--source", choices=["hotpot", "mixed"], default="mixed")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/plots/qa_multibaseline.txt")
    args = ap.parse_args()

    # Match the training-time config for v1h_t4k_v3 (see train_repr_qa.py
    # around line 688). These overrides set per-variant param shapes that
    # would otherwise cause state_dict size mismatches on load.
    cfg = ReprConfig(
        fixed_window_size=1024,
        max_window_size=args.chunk_size,    # 4096 for tranche-1
        d_node_state=128,
        n_edges=68,
        n_flat_codes=36,
        edge_token_packing="fused",
        d_mamba=768,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the data source once.
    if args.source == "mixed":
        comp = QADataset(
            COMPOSITE_VAL_P, COMPOSITE_VAL_Q,
            chunk_size=args.chunk_size, passages_per_chunk=300,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=0,
        )
        hp = HotpotQADataset(
            split="validation", tokenizer=tokenizer, chunk_size=args.chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=1,
        )
        ds = MixedQADataset(sources=[comp, hp], weights=[0.7, 0.3], seed=0)
    else:
        ds = HotpotQADataset(tokenizer=tokenizer, split="validation",
                              chunk_size=args.chunk_size,
                              pad_token_id=tokenizer.pad_token_id or 128_001)

    # Collect samples, filtering by family if requested.
    samples = []
    it = iter(ds)
    max_scan = args.chunks * 200
    scanned = 0
    while len(samples) < args.chunks and scanned < max_scan:
        s = next(it)
        scanned += 1
        if args.family is None or s.get("task_family") == args.family:
            samples.append(s)
    if len(samples) < args.chunks:
        print(f"[warn] only collected {len(samples)} of {args.chunks} "
              f"samples matching family={args.family!r} after {scanned} scans")
    args.chunks = len(samples)
    batch = collate_qa(samples, pad_token_id=tokenizer.pad_token_id or 128_001)
    print(f"[data] {args.chunks} samples (filter={args.family or 'any'})")

    # Load Llama ONCE; share across variants (the frozen base is identical).
    print("[llama] loading frozen Llama-3.2-1B once for all variants")
    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.float32)
    llama.train(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama = llama.to(device)

    # Per-variant generations.
    per_variant_gens: dict[str, list[str]] = OrderedDict()
    for variant in args.variants:
        print(f"[variant] {variant}")
        model, step = load_model(variant, cfg, llama,
                                  allow_unexpected=VARIANTS[variant][1])
        model = model.to(device)
        memory = encode_memory(model, batch, device).detach()
        print(f"  ckpt step={step}  memory shape={tuple(memory.shape)}")
        per_variant_gens[variant] = []
        ct = getattr(model, "chat_template", None)
        for b in range(args.chunks):
            q_ids = samples[b]["question_ids"].tolist()
            gen = generate_qa_answer(
                llama, tokenizer, memory[b:b+1],
                q_ids, args.max_new_tokens, device,
                chat_template=ct,
            )
            per_variant_gens[variant].append(gen.strip())
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Per-chunk side-by-side report.
    lines: list[str] = []
    summary_hits = {v: 0 for v in args.variants}
    for b in range(args.chunks):
        q = tokenizer.decode(samples[b]["question_ids"].tolist()).strip()
        a = tokenizer.decode(samples[b]["answer_ids"].tolist()).strip()
        family = samples[b].get("task_family", "?")
        # Context preview
        ctx_ids = samples[b]["context_ids"][samples[b]["context_mask"].bool()][:400]
        ctx = tokenizer.decode(ctx_ids.tolist()).replace("\n", " ")[:800]
        lines.append(f"\n{'=' * 80}")
        lines.append(f"CHUNK {b}  ({family})")
        lines.append(f"  context: {ctx}…")
        lines.append(f"  Q:    {q}")
        lines.append(f"  Gold: {a}")
        lines.append("")
        for variant in args.variants:
            pred = per_variant_gens[variant][b]
            hit = a.lower() in pred.lower()
            if hit:
                summary_hits[variant] += 1
            mark = "✓" if hit else "·"
            # Show first ~120 chars; loops compress naturally
            pred_short = pred[:160].replace("\n", " ")
            lines.append(f"  {mark} {variant:22s}: {pred_short}")
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SUMMARY (substring match of gold in prediction, n={args.chunks}):")
    for v, hits in summary_hits.items():
        lines.append(f"  {v:22s}: {hits}/{args.chunks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\n[output] wrote {args.out}")
    print("\n".join(lines[-(2 + len(args.variants)):]))


if __name__ == "__main__":
    main()
