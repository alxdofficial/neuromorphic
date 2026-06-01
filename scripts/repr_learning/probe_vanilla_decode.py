"""Vanilla-Llama AR-decode sanity probe.

Before scoring our trained variants with EM/F1, verify that vanilla
Llama-3.2-1B itself can produce coherent QA answers — both without
context (closed-book) and with full 8192-token context (open-book).

What this tells us:
  1. Does vanilla Llama emit EOS naturally at the end of a QA answer,
     or does it ramble until max_new_tokens? (If it rambles, then our
     trained variants inheriting this behavior is partly a foundation
     issue, not purely a training-time EOS-supervision bug.)
  2. Is the output extractable by simple post-processing (first
     sentence / first newline / first punctuation), or is it
     fundamentally non-answer-shaped (e.g., reformatting the question,
     hallucinating, looping)?
  3. How much does adding the 8192 context help — is the gap large
     enough to justify the full-context retrieval baseline?

Run AFTER the sweep finishes:
    python scripts/repr_learning/probe_vanilla_decode.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/alex/code/neuromorphic")
sys.path.insert(0, str(ROOT))

from src.repr_learning.config import ReprConfig                        # noqa: E402
from scripts.repr_learning.eval_per_family import (                    # noqa: E402
    collect_samples, EvalSample,
)


@torch.no_grad()
def decode_no_context(llama, tokenizer, question_ids: torch.Tensor,
                       max_new_tokens: int, device) -> tuple[str, int, bool]:
    """Greedy decode given JUST the question — closed-book."""
    q = question_ids.to(device).unsqueeze(0)
    gen = llama.generate(
        input_ids=q,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # input_ids form: gen includes the prompt, slice it off
    new_tokens = gen[0, q.shape[1]:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    n_new = len(new_tokens)
    hit_eos = (tokenizer.eos_token_id is not None
                and tokenizer.eos_token_id in new_tokens)
    return text, n_new, hit_eos


@torch.no_grad()
def decode_with_context(llama, tokenizer, context_ids: torch.Tensor,
                         context_mask: torch.Tensor,
                         question_ids: torch.Tensor,
                         max_new_tokens: int, device) -> tuple[str, int, bool]:
    """Greedy decode given [context; question] — open-book."""
    # Strip padding from context for tighter prefix
    valid_len = int(context_mask.sum().item())
    ctx = context_ids[:valid_len].to(device)
    q = question_ids.to(device)
    full = torch.cat([ctx, q]).unsqueeze(0)
    attn = torch.ones_like(full)
    gen = llama.generate(
        input_ids=full, attention_mask=attn,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, full.shape[1]:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    n_new = len(new_tokens)
    hit_eos = (tokenizer.eos_token_id is not None
                and tokenizer.eos_token_id in new_tokens)
    return text, n_new, hit_eos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=5,
                    help="how many examples to show per family")
    ap.add_argument("--families", nargs="+",
                    default=["biographical", "hotpot_qa", "narrative_qa", "musique"])
    ap.add_argument("--max-new-tokens", type=int, default=80,
                    help="leave generous so we can observe ramble vs EOS")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs/repr_learning/eval_per_family/vanilla_probe.txt")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = ReprConfig(
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,
        d_node_state=128, n_edges=68, n_flat_codes=128,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[tok] eos_token_id={tokenizer.eos_token_id} "
          f"pad_token_id={tokenizer.pad_token_id}")

    passages_per_chunk = max(75, (args.chunk_size // 1024) * 75)
    samples = collect_samples(
        args.families, args.n_per_family,
        tokenizer=tokenizer, cfg=cfg,
        chunk_size=args.chunk_size,
        passages_per_chunk=passages_per_chunk,
    )

    print(f"\n[llama] loading {cfg.llama_model} ({args.dtype})")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=dtype)
    llama.train(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama = llama.to(device)

    lines: list[str] = []
    lines.append(f"VANILLA-LLAMA AR-DECODE SANITY PROBE")
    lines.append(f"model={cfg.llama_model}  dtype={args.dtype}  "
                 f"max_new_tokens={args.max_new_tokens}")
    lines.append(f"eos_token_id={tokenizer.eos_token_id}")
    lines.append("")

    eos_hit_no_ctx = 0
    eos_hit_full_ctx = 0
    total = 0

    for s in samples:
        q_text = tokenizer.decode(s.question_ids.tolist(),
                                   skip_special_tokens=True).strip()
        refs_str = " | ".join(s.answer_refs[:2])
        # Closed-book
        nc_text, nc_len, nc_eos = decode_no_context(
            llama, tokenizer, s.question_ids, args.max_new_tokens, device,
        )
        # Open-book
        fc_text, fc_len, fc_eos = decode_with_context(
            llama, tokenizer, s.context_ids, s.context_mask,
            s.question_ids, args.max_new_tokens, device,
        )
        eos_hit_no_ctx += int(nc_eos)
        eos_hit_full_ctx += int(fc_eos)
        total += 1

        # Substring hit (loose check, case-insensitive)
        ref_lower = s.answer_refs[0].lower() if s.answer_refs else ""
        nc_hit = "✓" if (ref_lower and ref_lower in nc_text.lower()) else "·"
        fc_hit = "✓" if (ref_lower and ref_lower in fc_text.lower()) else "·"

        lines.append("=" * 80)
        lines.append(f"family={s.family}")
        lines.append(f"  Q:    {q_text}")
        lines.append(f"  Gold: {refs_str}")
        lines.append(f"  [no-ctx]   {nc_hit} len={nc_len:>3d} eos={nc_eos}")
        lines.append(f"    decoded: {nc_text[:200]!r}{'…' if len(nc_text) > 200 else ''}")
        lines.append(f"  [full-ctx] {fc_hit} len={fc_len:>3d} eos={fc_eos}")
        lines.append(f"    decoded: {fc_text[:200]!r}{'…' if len(fc_text) > 200 else ''}")
        lines.append("")

    lines.append("=" * 80)
    lines.append(f"SUMMARY (n={total})")
    lines.append(f"  no-context   EOS hit rate: {eos_hit_no_ctx}/{total} = {eos_hit_no_ctx/max(1,total)*100:.0f}%")
    lines.append(f"  full-context EOS hit rate: {eos_hit_full_ctx}/{total} = {eos_hit_full_ctx/max(1,total)*100:.0f}%")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - EOS hit rate >50% → vanilla Llama is reasonable, ours is what broke EOS")
    lines.append("  - EOS hit rate <20% → vanilla Llama itself rambles; we need eval-time")
    lines.append("    truncation regardless of how we train")
    lines.append("  - Hit rates similar between no-ctx and full-ctx → format-driven, not")
    lines.append("    context-driven (Llama just doesn't know to stop after a QA answer)")

    out_text = "\n".join(lines)
    args.out.write_text(out_text)
    print(f"\n[output] wrote {args.out}")
    print(out_text[-700:])  # show summary in stdout


if __name__ == "__main__":
    main()
