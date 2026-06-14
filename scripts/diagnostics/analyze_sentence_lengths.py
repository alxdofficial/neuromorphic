"""Sentence-length distribution for the compression-objective design.

Streams documents from a corpus, segments into sentences (and 1-2 sentence
units, since the compression unit may be either), tokenizes with the target
backbone's tokenizer (length in TOKENS is what sets the compression ratio), and
reports percentiles + a proposed bucket/ratio scheme. Run between design steps;
cheap (a few thousand docs).
"""
import re
import sys
import argparse
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--dataset", default="EleutherAI/the_pile_deduplicated")
ap.add_argument("--n-docs", type=int, default=2000)
ap.add_argument("--ratio", type=float, default=8.0)
ap.add_argument("--min-len", type=int, default=24)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)

# lightweight sentence splitter: break on .!? + whitespace + capital/quote/digit.
# Good enough for a distribution (not production segmentation).
_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\d])')


def sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]


single_lens, pair_lens = [], []
from datasets import load_dataset
ds = load_dataset(args.dataset, split="train", streaming=True)
it = iter(ds)
try:
    for _ in range(args.n_docs):
        ex = next(it)
        sents = sentences(ex.get("text", ""))
        toks = [len(tok.encode(s, add_special_tokens=False)) for s in sents]
        single_lens.extend(toks)
        for i in range(len(toks) - 1):
            pair_lens.append(toks[i] + toks[i + 1])
finally:
    del it, ds   # clean shutdown (streaming generator core-dumps at finalize)

single = np.array(single_lens)
pair = np.array(pair_lens)


def report(name, arr, ratio, min_len):
    keep = arr[arr >= min_len]
    pct = lambda p: int(np.percentile(arr, p))
    print(f"\n=== {name}  (n={len(arr):,}) ===")
    print(f"  token length pctiles: p10={pct(10)} p25={pct(25)} p50={pct(50)} "
          f"p75={pct(75)} p90={pct(90)} p95={pct(95)} p99={pct(99)} max={arr.max()}")
    print(f"  mean={arr.mean():.1f}  >= min_len({min_len}): "
          f"{100*len(keep)/len(arr):.1f}% kept ({len(keep):,})")
    slots = np.ceil(keep / ratio).astype(int)
    print(f"  at ratio {ratio:g}: slot counts (of kept) — "
          f"p10={int(np.percentile(slots,10))} p50={int(np.percentile(slots,50))} "
          f"p90={int(np.percentile(slots,90))} p99={int(np.percentile(slots,99))} "
          f"max={slots.max()}")
    u, c = np.unique(slots, return_counts=True)
    dist = {int(k): f"{100*v/len(slots):.0f}%" for k, v in zip(u[:10], c[:10])}
    print(f"  slot-count distribution (kept): {dist}")
    eff_ratio = keep.sum() / slots.sum()
    print(f"  mean achieved ratio (kept tokens / kept slots): {eff_ratio:.2f}")


report("SINGLE sentence", single, args.ratio, args.min_len)
report("SENTENCE PAIR", pair, args.ratio, args.min_len)
print(f"\n[tokenizer] {args.tokenizer}  [corpus] {args.dataset}  [docs] {args.n_docs}")
