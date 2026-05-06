"""Pretokenize FineWeb-edu parquet -> flat int32 file for Wave 1 training.

Best-practice streaming preprocessor:
- **Streaming parquet read** via ``pyarrow.parquet.ParquetFile.iter_batches``
  (don't hold the entire 2 GB parquet table in RAM).
- **Memory-mapped output** via ``numpy.memmap`` with ``mode="w+"``: token
  stream lives on disk; only a small write window stays in RAM.
- **Single-pass write** with periodic ``flush()`` so the OS can checkpoint
  pages without us materializing the full output array.

Memory ceiling at runtime: a single row-batch of parquet text + the
tokenizer's working buffers. The output shard is on disk from the first
write; no full-corpus tensor ever exists in process memory.

Output layout (under ``--out`` prefix):
  ``{prefix}.bin``        flat int32 token stream with EOS separators
  ``{prefix}.meta.json``  metadata (token count, tokenizer, source)

The companion loader update teaches ``fineweb_edu_phase1_iter`` to
detect ``{prefix}.bin`` and read with ``numpy.memmap`` (zero-copy
random-window slices).

Why int32 (4 bytes/token) and not int16 (2 bytes/token): Llama-3.2's
vocab is 128256, exceeds the int16 ceiling of 65535.

Default budget: 200M tokens for Wave 1 -- 2x Wave 1's ~100M-token
budget so we have margin without materializing the full ~1.2B-token
parquet (~4.8 GB on disk).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/preprocess_fineweb_edu_llama32.py \\
      --parquet data/phase_B/fineweb_edu.parquet \\
      --out data/phase_B/fineweb_edu_llama32 \\
      --target-tokens 200_000_000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet", default="data/phase_B/fineweb_edu.parquet",
        help="Source parquet with a `text` column",
    )
    ap.add_argument(
        "--out", default="data/phase_B/fineweb_edu_llama32",
        help="Output prefix (.bin + .meta.json appended)",
    )
    ap.add_argument(
        "--tokenizer", default="meta-llama/Llama-3.2-1B",
        help="HF tokenizer name",
    )
    ap.add_argument(
        "--target-tokens", type=int, default=200_000_000,
        help="Stop once this many tokens are accumulated (incl. EOS markers)",
    )
    ap.add_argument(
        "--row-batch", type=int, default=1024,
        help="Parquet row batch size (streaming chunk)",
    )
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(f"parquet not found: {parquet_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(".bin")
    meta_path = out_path.with_suffix(".meta.json")

    print(f"[preprocess] tokenizer = {args.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("tokenizer.eos_token_id is None")
    print(f"[preprocess] eos_id = {eos_id}, vocab_size = {tok.vocab_size}",
          flush=True)

    # Allocate the output as a memmap'd int32 array on disk. Writing into
    # this array writes directly to the file via the OS page cache; the
    # full array never sits in process RAM. We size to target_tokens; if
    # the source parquet runs out before target, we'll truncate at the
    # end via a second smaller memmap.
    print(f"[preprocess] allocating memmap {bin_path} "
          f"({args.target_tokens * 4 / 1e9:.2f} GB on disk)", flush=True)
    out = np.memmap(
        bin_path, dtype=np.int32, mode="w+",
        shape=(args.target_tokens,),
    )

    pf = pq.ParquetFile(parquet_path)
    n_filled = 0
    n_docs = 0
    t0 = time.perf_counter()
    last_log = t0
    last_flush = t0
    done = False

    print(f"[preprocess] streaming {parquet_path} in row-batches of "
          f"{args.row_batch}", flush=True)
    for record_batch in pf.iter_batches(
        batch_size=args.row_batch, columns=["text"],
    ):
        if done:
            break
        # to_pylist gives us a python list of strings for this batch — we
        # avoid materializing the full table column at once.
        texts = record_batch.column("text").to_pylist()
        for text in texts:
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            n = len(ids)
            # Cap at remaining budget (account for the EOS we'll append).
            remaining = args.target_tokens - n_filled - 1
            if remaining <= 0:
                done = True
                break
            if n > remaining:
                n = remaining
                ids = ids[:n]
            out[n_filled:n_filled + n] = ids
            n_filled += n
            out[n_filled] = eos_id
            n_filled += 1
            n_docs += 1

            now = time.perf_counter()
            if now - last_log > 10.0:
                elapsed = now - t0
                rate = n_filled / max(elapsed, 1e-9)
                print(f"[preprocess] {n_filled:,} tokens, {n_docs:,} docs, "
                      f"{rate / 1e3:.1f}k tok/s, {elapsed:.1f}s", flush=True)
                last_log = now
            # Flush dirty pages to disk every minute so a process kill
            # doesn't lose the recent work and so the OS doesn't blow up
            # its dirty-page cache.
            if now - last_flush > 60.0:
                out.flush()
                last_flush = now
            if n_filled >= args.target_tokens:
                done = True
                break

    elapsed = time.perf_counter() - t0
    print(f"[preprocess] done streaming: {n_filled:,} tokens, {n_docs:,} docs, "
          f"{elapsed:.1f}s, {n_filled / max(elapsed, 1e-9) / 1e3:.1f}k tok/s",
          flush=True)

    out.flush()
    del out  # closes the memmap

    # If we ran out before target, truncate the file to the actual size.
    if n_filled < args.target_tokens:
        # Re-open in r+ to read, then write a smaller file.
        print(f"[preprocess] truncating to {n_filled:,} tokens "
              f"({n_filled * 4 / 1e9:.2f} GB)", flush=True)
        # Simplest: read the prefix as memmap, write a new file from it.
        src = np.memmap(bin_path, dtype=np.int32, mode="r",
                        shape=(args.target_tokens,))
        truncated = np.array(src[:n_filled])  # one alloc of the actual size
        del src
        bin_path.unlink()
        truncated.tofile(bin_path)
        del truncated

    meta = {
        "tokens": int(n_filled),
        "docs": int(n_docs),
        "tokenizer": args.tokenizer,
        "source": str(parquet_path),
        "eos_id": int(eos_id),
        "dtype": "int32",
        "elapsed_s": round(elapsed, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] wrote {bin_path} ({n_filled * 4 / 1e9:.2f} GB on disk)",
          flush=True)
    print(f"[preprocess] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
