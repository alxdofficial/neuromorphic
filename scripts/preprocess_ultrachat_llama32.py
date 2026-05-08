"""Pretokenize UltraChat-200k -> flat int32 file for Wave 2 training.

Same streaming + memmap pattern as the FineWeb-edu preprocessor. The
extra wrinkle: each example is a list-of-turns and we have to apply the
Llama-3.2-Instruct chat template per example before encoding.

Two side effects:
1. Forces HF to download UltraChat-200k locally (currently only a 36 KB
   metadata stub is in cache). First run will pull ~4 GB of parquet.
2. Bakes the chat-templated string into the token stream, so the Wave 2
   loader doesn't need to re-template each batch.

Output (under ``--out`` prefix):
  ``{prefix}.bin``        flat int32 with EOS separators between conversations
  ``{prefix}.mask.bin``   flat uint8 (0/1) — assistant-only loss mask aligned
                          with ``.bin``. 1 = supervised token (assistant
                          message body), 0 = ignored (user/system message,
                          chat-template headers, EOS separators between
                          conversations). The Wave 2 loader passes this
                          mask through ``F.cross_entropy(ignore_index=-100)``
                          so user/system positions don't contribute to the
                          LM loss or to the walker's plasticity signal.
  ``{prefix}.meta.json``  metadata

Default budget: 200M tokens (matches Wave 2's training_strategy.md
allocation). Streams from HF until target hit.

Memory ceiling: a single chat-templated string + one batch of token ids
+ the memmap (which lives on disk).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/preprocess_ultrachat_llama32.py \\
      --out data/phase_B/ultrachat_llama32 --target-tokens 200_000_000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", default="HuggingFaceH4/ultrachat_200k",
    )
    ap.add_argument(
        "--split", default="train_sft",
    )
    ap.add_argument(
        "--out", default="data/phase_B/ultrachat_llama32",
    )
    ap.add_argument(
        "--tokenizer", default="meta-llama/Llama-3.2-1B-Instruct",
        help="Use the -Instruct tokenizer for its chat_template",
    )
    ap.add_argument(
        "--target-tokens", type=int, default=200_000_000,
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(".bin")
    mask_path = out_path.parent / (out_path.name + ".mask.bin")
    meta_path = out_path.with_suffix(".meta.json")

    print(f"[preprocess] tokenizer = {args.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("tokenizer.eos_token_id is None")
    if tok.chat_template is None:
        raise SystemExit(
            f"{args.tokenizer} has no chat_template — need an "
            "Instruct-tuned tokenizer for chat templating"
        )
    print(f"[preprocess] eos_id = {eos_id}, vocab_size = {tok.vocab_size}",
          flush=True)

    print(f"[preprocess] streaming {args.dataset} split={args.split} ...",
          flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    print(f"[preprocess] allocating memmap {bin_path} "
          f"({args.target_tokens * 4 / 1e9:.2f} GB on disk)", flush=True)
    out = np.memmap(
        bin_path, dtype=np.int32, mode="w+",
        shape=(args.target_tokens,),
    )
    print(f"[preprocess] allocating mask memmap {mask_path} "
          f"({args.target_tokens / 1e9:.2f} GB on disk)", flush=True)
    out_mask = np.memmap(
        mask_path, dtype=np.uint8, mode="w+",
        shape=(args.target_tokens,),
    )

    n_filled = 0
    n_convs = 0
    n_skipped = 0
    t0 = time.perf_counter()
    last_log = t0
    last_flush = t0

    for example in ds:
        messages = example.get("messages") or example.get("conversation")
        if not messages:
            n_skipped += 1
            continue
        # Tokenize the full conversation once for the token stream, then
        # tokenize each prefix to recover where assistant message bodies
        # land. The mask covers ONLY assistant message bodies — chat
        # headers (`<|start_header_id|>assistant<|end_header_id|>`),
        # user/system bodies, and inter-message glue stay 0.
        try:
            full_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            n_skipped += 1
            continue
        ids = tok.encode(full_text, add_special_tokens=False)
        if not ids:
            n_skipped += 1
            continue

        # Build the assistant-only mask by templating cumulative prefixes
        # at "answer-time" (with `add_generation_prompt=True`) to find
        # where each assistant body STARTS, then templating with the
        # full message included to find where it ENDS.
        ex_mask = [0] * len(ids)
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            try:
                start_text = tok.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True,
                )
                end_text = tok.apply_chat_template(
                    messages[: i + 1], tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                continue
            start_tok = len(tok.encode(start_text, add_special_tokens=False))
            end_tok = len(tok.encode(end_text, add_special_tokens=False))
            for k in range(start_tok, min(end_tok, len(ex_mask))):
                ex_mask[k] = 1

        n = len(ids)
        remaining = args.target_tokens - n_filled - 1
        if remaining <= 0:
            break
        if n > remaining:
            n = remaining
            ids = ids[:n]
            ex_mask = ex_mask[:n]
        out[n_filled:n_filled + n] = ids
        out_mask[n_filled:n_filled + n] = ex_mask
        n_filled += n
        out[n_filled] = eos_id
        out_mask[n_filled] = 0   # EOS separator: not supervised
        n_filled += 1
        n_convs += 1

        now = time.perf_counter()
        if now - last_log > 10.0:
            elapsed = now - t0
            rate = n_filled / max(elapsed, 1e-9)
            print(f"[preprocess] {n_filled:,} tokens, {n_convs:,} convs, "
                  f"{n_skipped:,} skipped, {rate / 1e3:.1f}k tok/s, "
                  f"{elapsed:.1f}s", flush=True)
            last_log = now
        if now - last_flush > 60.0:
            out.flush()
            last_flush = now
        if n_filled >= args.target_tokens:
            break

    elapsed = time.perf_counter() - t0
    n_supervised = int(out_mask[:n_filled].sum())
    print(f"[preprocess] done: {n_filled:,} tokens "
          f"({n_supervised:,} supervised, "
          f"{100 * n_supervised / max(n_filled, 1):.1f}%), "
          f"{n_convs:,} convs, {n_skipped:,} skipped, "
          f"{elapsed:.1f}s", flush=True)

    out.flush()
    del out
    out_mask.flush()
    del out_mask

    if n_filled < args.target_tokens:
        print(f"[preprocess] truncating to {n_filled:,} tokens", flush=True)
        src = np.memmap(bin_path, dtype=np.int32, mode="r",
                        shape=(args.target_tokens,))
        truncated = np.array(src[:n_filled])
        del src
        bin_path.unlink()
        truncated.tofile(bin_path)
        del truncated
        src_mask = np.memmap(mask_path, dtype=np.uint8, mode="r",
                             shape=(args.target_tokens,))
        truncated_mask = np.array(src_mask[:n_filled])
        del src_mask
        mask_path.unlink()
        truncated_mask.tofile(mask_path)
        del truncated_mask

    meta = {
        "tokens": int(n_filled),
        "supervised_tokens": int(n_supervised),
        "conversations": int(n_convs),
        "skipped": int(n_skipped),
        "tokenizer": args.tokenizer,
        "source_dataset": args.dataset,
        "split": args.split,
        "eos_id": int(eos_id),
        "dtype": "int32",
        "mask_dtype": "uint8",
        "has_assistant_mask": True,
        "elapsed_s": round(elapsed, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] wrote {bin_path} ({n_filled * 4 / 1e9:.2f} GB)",
          flush=True)
    print(f"[preprocess] wrote {mask_path} ({n_filled / 1e9:.2f} GB)",
          flush=True)
    print(f"[preprocess] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
