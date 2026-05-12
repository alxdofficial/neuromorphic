#!/usr/bin/env python3
"""bench_em_accuracy.py — TF-EM accuracy + per-distance CE on needle val.

Supports both vanilla Llama-3.2-1B and our IntegratedLM checkpoint.

Per doc:
- Tokenize the full doc (or use pre-tokenized input_ids from parquet)
- Process chunks of 1024 tokens sequentially:
    vanilla mode: Llama with rolling KV cache (2K cap — older context flushed)
    ours mode:    IntegratedLM.forward_window with state threading across windows
- At each answer-span token position, check if argmax(logits) == gold token
- Per-doc EM = ALL answer-span tokens argmax-correct (string-exact-match upper bound)
- Stratify by needle→query distance bin

Usage:
    PYTHONPATH=. python scripts/diagnostics/bench_em_accuracy.py \\
        --model-type vanilla \\
        --max-docs 250 \\
        --output outputs/em_vanilla.json

    PYTHONPATH=. python scripts/diagnostics/bench_em_accuracy.py \\
        --model-type ours \\
        --ckpt outputs/wave1/ckpt.best.pt \\
        --max-docs 100 \\
        --output outputs/em_ours.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.trajectory_memory.config import TrajMemConfig


def distance_bin(d: int) -> str:
    if d < 2000:
        return "0-2K"
    if d < 5000:
        return "2K-5K"
    if d < 12000:
        return "5K-12K"
    if d < 24000:
        return "12K-24K"
    return "24K+"


def aggregate(per_doc: list[dict]) -> dict:
    """Aggregate stats by distance bin + answer-type."""
    by_bin = defaultdict(list)
    by_kind = defaultdict(list)
    for d in per_doc:
        by_bin[distance_bin(d["target_distance"])].append(d)
        by_kind[d.get("answer_type", "?")].append(d)
    out = {"per_distance": {}, "per_type": {}, "overall": {}}

    def stats(items: list[dict]) -> dict:
        if not items:
            return {}
        n = len(items)
        em = sum(d["em_correct"] for d in items) / n
        ce_full = sum(d["full_ce_sum"] for d in items) / max(
            sum(d["full_token_count"] for d in items), 1
        )
        ce_ans = sum(d["answer_ce_sum"] for d in items) / max(
            sum(d["answer_token_count"] for d in items), 1
        )
        token_acc = sum(d["answer_correct_tokens"] for d in items) / max(
            sum(d["answer_token_count"] for d in items), 1
        )
        return {
            "n_docs": n,
            "em_accuracy": em,
            "answer_token_accuracy": token_acc,
            "ntp_ce_full": ce_full,
            "ntp_ce_answer": ce_ans,
        }

    for b, items in by_bin.items():
        out["per_distance"][b] = stats(items)
    for k, items in by_kind.items():
        out["per_type"][k] = stats(items)
    out["overall"] = stats(per_doc)
    return out


def classify_answer(answer: str) -> str:
    """Heuristic answer-type classification."""
    import re
    if len(answer) == 6 and re.fullmatch(r"[A-Z0-9]+", answer):
        return "alphanum_short"
    if re.fullmatch(r"[A-Za-z0-9-]{12,}", answer.replace(" ", "")):
        return "alphanum_long"
    if "-" in answer and not any(c.isdigit() for c in answer):
        return "phrase"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer):
        return "date"
    if re.fullmatch(r"[\d,]+ [A-Z]{3}", answer):
        return "currency"
    if re.fullmatch(r"id-[a-z0-9]+", answer):
        return "url_id"
    if "," in answer and any(c.isalpha() for c in answer):
        return "location"
    if re.fullmatch(r"[A-Z][a-z]+/[A-Z][a-z_]+", answer):
        return "timezone"
    if any(c.isupper() for c in answer) and " " in answer:
        return "name"
    if re.fullmatch(r"\d+", answer):
        return "number"
    if answer in {"Mandarin", "Swahili", "Tamil", "Quechua", "Welsh",
                  "Icelandic", "Yoruba", "Esperanto", "Vietnamese", "Tagalog",
                  "Norwegian", "Hebrew", "Portuguese", "Hungarian", "Korean"}:
        return "language"
    return "other"


def run_vanilla(args) -> list[dict]:
    """Vanilla Llama: process each doc with rolling KV cache, check argmax at answer span."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map=args.device,
    )
    model.train(False)

    table = pq.read_table(args.val_parquet)
    rows = table.to_pylist()[: args.max_docs]

    per_doc = []
    t0 = time.time()
    for i, row in enumerate(rows):
        ids_full = torch.tensor(row["input_ids"], dtype=torch.int64, device=args.device)
        ans_start = row["answer_start_token"]
        ans_end = row["answer_end_token"]
        T = ids_full.shape[0]
        if ans_end >= T:
            continue  # malformed metadata
        # Optionally truncate to last context-cap tokens before ans_end+1.
        # Lets us test "vanilla Llama with 2K context" — the scenario where
        # memory was supposed to win.
        if args.context_cap > 0:
            context_start = max(0, (ans_end + 1) - args.context_cap)
            ids_cur = ids_full[context_start : ans_end + 1]
            ans_start_local = ans_start - context_start
            ans_end_local = ans_end - context_start
        else:
            ids_cur = ids_full
            ans_start_local = ans_start
            ans_end_local = ans_end
        with torch.no_grad():
            out = model(input_ids=ids_cur.unsqueeze(0), use_cache=False)
        logits = out.logits[0]                                # [T_cur, V]
        # Update answer-position indices to use the local frame
        ans_start, ans_end = ans_start_local, ans_end_local
        ids_full = ids_cur
        # logits[t] predicts token[t+1]
        # We want logits at positions ans_start-1 ... ans_end-1 predicting tokens ans_start ... ans_end
        ans_token_ids = ids_full[ans_start : ans_end + 1]
        # Only convert the answer-span slice to float for CE (full-doc fp32
        # logits OOM for long docs at 128K vocab).
        pred_logits = logits[ans_start - 1 : ans_end].float()
        pred_argmax = pred_logits.argmax(dim=-1)
        correct = (pred_argmax == ans_token_ids).int()
        ans_ce = torch.nn.functional.cross_entropy(
            pred_logits, ans_token_ids, reduction="sum",
        ).item()

        per_doc.append({
            "doc_idx": i,
            "target_distance": row["target_distance"],
            "answer": row["answer"],
            "answer_type": classify_answer(row["answer"]),
            "answer_correct_tokens": int(correct.sum().item()),
            "answer_token_count": int(correct.numel()),
            "em_correct": int(correct.all().item()),
            "answer_ce_sum": ans_ce,
            "full_ce_sum": 0.0,             # skipped (OOM on long docs)
            "full_token_count": 0,
        })
        if (i + 1) % 25 == 0:
            agg = aggregate(per_doc)["overall"]
            print(f"  doc {i+1}/{len(rows)}: EM={agg['em_accuracy']:.3f}, "
                  f"tok-acc={agg['answer_token_accuracy']:.3f}, "
                  f"answer-CE={agg['ntp_ce_answer']:.3f}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return per_doc


def run_ours(args) -> list[dict]:
    """Our model: process each doc window-by-window with state threading."""
    import sys
    sys.path.insert(0, ".")
    from src.trajectory_memory.integrated_lm import IntegratedLM

    cfg = TrajMemConfig.medium()
    cfg.d_lm = 2048  # Llama-3.2-1B

    print(f"Loading IntegratedLM from {args.ckpt}...")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = IntegratedLM(cfg, model_name=args.model_name, llama_dtype="bf16")
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.to(args.device)
    model.train(False)

    if getattr(args, "ablate_scale_zero", False):
        with torch.no_grad():
            model._mem_inject_layer().scale_raw.zero_()
        print("  [ablate] scale_raw zeroed → injection is exact identity")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    table = pq.read_table(args.val_parquet)
    rows = table.to_pylist()[: args.max_docs]

    T_window = cfg.T_window
    per_doc = []
    t0 = time.time()
    for i, row in enumerate(rows):
        ids_full = torch.tensor(row["input_ids"], dtype=torch.int64, device=args.device)
        ans_start = row["answer_start_token"]
        ans_end = row["answer_end_token"]
        T = ids_full.shape[0]
        if ans_end >= T:
            continue

        # Pad to a multiple of T_window for clean chunking
        n_windows = (T + T_window - 1) // T_window
        T_pad = n_windows * T_window
        ids_padded = torch.full(
            (T_pad,), tokenizer.pad_token_id or tokenizer.eos_token_id,
            dtype=torch.int64, device=args.device,
        )
        ids_padded[:T] = ids_full
        ids_padded = ids_padded.unsqueeze(0)  # [1, T_pad]

        prev_states = model.manifold.reset_states(batch_size=1)
        prev_hiddens = None
        kv = None
        cache_pos = 0

        # Save logits for windows that cover the prediction positions.
        # We need predictions at positions ans_start-1 ... ans_end (to
        # predict tokens ans_start ... ans_end). Identify the window
        # range that contains [ans_start-1, ans_end].
        pred_lo = ans_start - 1
        pred_hi = ans_end                       # inclusive
        save_w_lo = max(0, pred_lo // T_window)
        save_w_hi = pred_hi // T_window         # inclusive
        saved_start = save_w_lo * T_window
        saved_logits = []
        with torch.no_grad():
            for w in range(n_windows):
                window_ids = ids_padded[:, w * T_window : (w + 1) * T_window]
                out = model.forward_window(
                    window_ids,
                    prev_window_hiddens=prev_hiddens,
                    prev_states=prev_states,
                    use_kv_cache=True,
                    past_key_values=kv,
                    cache_abs_pos=cache_pos,
                    hard_routing=False,
                )
                if save_w_lo <= w <= save_w_hi:
                    saved_logits.append(out["logits"][0].float().cpu())
                prev_states = out["new_states"]
                prev_hiddens = out["current_hiddens"]
                kv = out.get("new_past_key_values", None)
                cache_pos = int(out.get("new_cache_abs_pos", cache_pos + T_window))

        # Stitch saved logits + extract answer-span positions.
        # Note logits_cat covers [saved_start, saved_start + len].
        logits_cat = torch.cat(saved_logits, dim=0).to(args.device)
        # logits[t] predicts token[t+1]. To predict tokens [ans_start..ans_end],
        # use logits at positions [ans_start-1 .. ans_end-1], slice [a-1:b].
        local_pred_start = pred_lo - saved_start         # = (ans_start-1) - saved_start
        local_pred_end = (pred_hi + 1) - 1 - saved_start  # = ans_end - saved_start (exclusive bound)
        ans_token_ids = ids_full[ans_start : ans_end + 1]
        pred_logits = logits_cat[local_pred_start : local_pred_end]
        pred_argmax = pred_logits.argmax(dim=-1)
        correct = (pred_argmax == ans_token_ids).int()
        ans_ce = torch.nn.functional.cross_entropy(
            pred_logits, ans_token_ids, reduction="sum",
        ).item()

        per_doc.append({
            "doc_idx": i,
            "target_distance": row["target_distance"],
            "answer": row["answer"],
            "answer_type": classify_answer(row["answer"]),
            "answer_correct_tokens": int(correct.sum().item()),
            "answer_token_count": int(correct.numel()),
            "em_correct": int(correct.all().item()),
            "answer_ce_sum": ans_ce,
            "full_ce_sum": 0.0,             # skipped
            "full_token_count": 0,
        })
        if (i + 1) % 10 == 0:
            agg = aggregate(per_doc)["overall"]
            print(f"  doc {i+1}/{len(rows)}: EM={agg['em_accuracy']:.3f}, "
                  f"tok-acc={agg['answer_token_accuracy']:.3f}, "
                  f"answer-CE={agg['ntp_ce_answer']:.3f}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return per_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["vanilla", "ours"], required=True)
    ap.add_argument("--ckpt", type=Path, help="ckpt path (required for --model-type ours)")
    ap.add_argument("--val-parquet", default="data/wave1/needle.val.parquet")
    ap.add_argument("--max-docs", type=int, default=250)
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--context-cap", type=int, default=0,
                    help="Cap Llama's input context to last N tokens (0=full). "
                         "Use 2048 to test memory-vs-shortened-Llama story.")
    ap.add_argument("--ablate-scale-zero", action="store_true",
                    help="Zero scale_raw before eval. Makes injection exact "
                         "identity → measures what Llama-via-our-scaffold "
                         "achieves WITHOUT memory.")
    args = ap.parse_args()

    if args.model_type == "ours" and args.ckpt is None:
        raise SystemExit("--ckpt required for --model-type ours")

    if args.model_type == "vanilla":
        per_doc = run_vanilla(args)
    else:
        per_doc = run_ours(args)

    agg = aggregate(per_doc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "ckpt": str(args.ckpt) if args.ckpt else None,
            "n_docs_processed": len(per_doc),
            "aggregated": agg,
            "per_doc": per_doc,
        }, f, indent=2)

    print(f"\n=== Overall ({len(per_doc)} docs) ===")
    o = agg["overall"]
    print(f"  EM accuracy:           {o['em_accuracy']:.3f}")
    print(f"  Answer token accuracy: {o['answer_token_accuracy']:.3f}")
    print(f"  Answer-only CE:        {o['ntp_ce_answer']:.4f}")
    print(f"  Full-doc CE:           {o['ntp_ce_full']:.4f}")
    print(f"\n=== Per distance ===")
    for b in ["0-2K", "2K-5K", "5K-12K", "12K-24K", "24K+"]:
        if b in agg["per_distance"]:
            s = agg["per_distance"][b]
            print(f"  {b:>8}: n={s['n_docs']:3d}, EM={s['em_accuracy']:.3f}, "
                  f"tok-acc={s['answer_token_accuracy']:.3f}, "
                  f"ans-CE={s['ntp_ce_answer']:.3f}")
    print(f"\n=== Per answer type ===")
    for k, s in sorted(agg["per_type"].items(), key=lambda x: -x[1]["n_docs"]):
        print(f"  {k:<18}: n={s['n_docs']:3d}, EM={s['em_accuracy']:.3f}, "
              f"tok-acc={s['answer_token_accuracy']:.3f}, "
              f"ans-CE={s['ntp_ce_answer']:.3f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
