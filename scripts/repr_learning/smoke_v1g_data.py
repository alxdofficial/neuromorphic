#!/usr/bin/env python3
"""Quick smoke test for v1g sentence-chunk data pipeline.

Loads tokenizer + a few FineWeb chunks, prints sentence statistics,
verifies mask/reveal shapes and that "still-masked" positions are
non-empty per queried sentence. Should run in < 60s on cold start.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_sentence import make_sentence_chunk_dataloader


REPO = Path(__file__).resolve().parents[2]
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"


def main():
    cfg = ReprConfig(batch_size=2)
    print(f"Loading tokenizer {cfg.llama_model}...")
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)

    dl = make_sentence_chunk_dataloader(
        cfg,
        fineweb_path=FINEWEB_VAL,
        tokenizer=tok,
        chunk_size=4096,
        n_queries=3,
        mask_ratio=0.8,
        reveal_lo=0.0,
        reveal_hi=0.9,
        sentence_min_len=8,
        sentence_max_len=80,
        num_workers=0,
        seed=42,
        batch_size=2,
    )

    it = iter(dl)
    for step in range(3):
        batch = next(it)
        B, T = batch.input_ids.shape
        _, K, T_sent = batch.query_input_ids.shape

        print(f"\n=== step {step} ===")
        print(f"input_ids       : {tuple(batch.input_ids.shape)}")
        print(f"attention_mask  : {tuple(batch.attention_mask.shape)}")
        print(f"n_sentences     : {batch.n_sentences.tolist()}")
        print(f"query_starts    : {batch.query_starts.tolist()}")
        print(f"query_lengths   : {batch.query_lengths.tolist()}")
        print(f"query_input_ids : {tuple(batch.query_input_ids.shape)}")
        print(f"mask_positions  : {tuple(batch.mask_positions.shape)}")
        print(f"reveal_positions: {tuple(batch.reveal_positions.shape)}")

        # Per-queried-sentence: count masked and revealed
        for b in range(B):
            for k in range(K):
                L = int(batch.query_lengths[b, k].item())
                n_mask = int(batch.mask_positions[b, k, :L].sum().item())
                n_reveal = int(batch.reveal_positions[b, k, :L].sum().item())
                still_masked = int(
                    (batch.mask_positions[b, k, :L]
                     & ~batch.reveal_positions[b, k, :L]).sum().item()
                )
                # Sanity: revealed ⊆ masked, still_masked = mask − reveal
                rev_in_mask = bool(
                    (batch.reveal_positions[b, k, :L]
                     & ~batch.mask_positions[b, k, :L]).any().item()
                )
                if rev_in_mask:
                    print(f"  ✗ b={b} k={k}: reveal not subset of mask")
                else:
                    pct = 100.0 * still_masked / max(L, 1)
                    print(f"  b={b} k={k}: L={L:2d}  mask={n_mask:2d}  "
                          f"reveal={n_reveal:2d}  still_masked={still_masked:2d} "
                          f"({pct:.0f}% of sentence)")

        # Decode one queried sentence to show what it looks like
        if step == 0:
            b, k = 0, 0
            L = int(batch.query_lengths[b, k].item())
            ids = batch.query_input_ids[b, k, :L].tolist()
            text = tok.decode(ids)
            print(f"\n  Sample queried sentence (b=0, k=0):")
            print(f"    {text!r}")
            mask = batch.mask_positions[b, k, :L].tolist()
            reveal = batch.reveal_positions[b, k, :L].tolist()
            still = [m and not r for m, r in zip(mask, reveal)]
            tokens_view = []
            for tid, m, r, s in zip(ids, mask, reveal, still):
                t = tok.decode([tid])
                if s:
                    tokens_view.append(f"[MASK]")
                elif r:
                    tokens_view.append(f"<{t.strip()}>")  # revealed (treated as predicted)
                else:
                    tokens_view.append(t.strip() or "_")
            print(f"  Decoder input view (still=[MASK], revealed=<tok>, visible=tok):")
            print(f"    {' '.join(tokens_view)}")

    print("\n[smoke v1g data] OK — pipeline produces sane batches.")


if __name__ == "__main__":
    main()
