#!/usr/bin/env python3
"""Wave 3 passphrase eval — BERT-cosine recall on held-out facts.

For each fact in the held-out split:
1. Construct a prompt the same way training does (filler + fact +
   filler + question), but DO NOT include the reference answer.
2. AR-generate the answer with `autoregressive_rollout` (real
   generation, no teacher forcing).
3. Encode generated answer + each reference answer with
   `sentence-transformers/all-mpnet-base-v2`.
4. Compute cosine similarity for each reference; take the MAX.

Reports:
- per-fact: max cosine, generated text vs best reference
- aggregate: recall@0.7, recall@0.5, mean similarity

Usage:
    PYTHONPATH=. .venv/bin/python scripts/eval_passphrase.py \\
        --ckpt outputs/wave3/ckpt_final.pt \\
        --expanded data/passphrase/expanded.json \\
        --filler-parquet data/phase_B/fineweb_edu.parquet
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.passphrase_loader import _FillerPool, _load_facts, _split_train_heldout
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rollout import autoregressive_rollout


def _bert_cosine_matrix(texts_a: list[str], texts_b: list[str], encoder) -> np.ndarray:
    """Return cosine-similarity matrix [len(texts_a), len(texts_b)]."""
    emb_a = encoder.encode(texts_a, convert_to_numpy=True, normalize_embeddings=True)
    emb_b = encoder.encode(texts_b, convert_to_numpy=True, normalize_embeddings=True)
    return emb_a @ emb_b.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to a wrapper checkpoint .pt file.")
    ap.add_argument("--expanded", default="data/passphrase/expanded.json")
    ap.add_argument("--filler-parquet", default="data/phase_B/fineweb_edu.parquet")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--d-mem", type=int, default=256)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--bert-encoder",
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--T-pre", type=int, default=512)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--filler-mid-tokens", type=int, default=300,
                    help="Filler-mid length to test at (constant during eval).")
    ap.add_argument("--n-heldout", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filler-pool-size", type=int, default=2_000_000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # Load wrapper from ckpt
    print(f"[eval] loading wrapper from {args.ckpt}...")
    cfg = PretrainedGWConfig.llama_1b(
        model_name=args.model, inject_layer=args.inject_layer,
        d_mem=args.d_mem, T=args.T, bs=1,
    )
    wrapper = GraphWalkerPretrainedLM(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    wrapper.load_state_dict(ckpt["wrapper"])
    wrapper.train(False)
    print(f"[eval] loaded ckpt at step {ckpt.get('step', '?')}")

    # Load BERT encoder
    print(f"[eval] loading BERT encoder {args.bert_encoder}...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.bert_encoder, device=str(device))

    # Load facts + held-out split
    facts = _load_facts(args.expanded)
    _, heldout = _split_train_heldout(facts, n_heldout=args.n_heldout, seed=args.seed)
    print(f"[eval] {len(heldout)} held-out facts")

    # Filler pool
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    filler_pool = _FillerPool(args.filler_parquet, tokenizer,
                              target_tokens=args.filler_pool_size, seed=args.seed)
    rng = random.Random(args.seed + 7)

    # Per-fact generation + scoring
    per_fact: list[dict] = []
    for fact in heldout:
        # Build prompt (no answer included)
        fact_text = rng.choice(fact.paraphrases)
        question_text = rng.choice(fact.questions)

        eos_id = tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id
        fact_ids = tokenizer.encode(" " + fact_text, add_special_tokens=False)
        q_ids = tokenizer.encode("\n\nQuestion: " + question_text + "\n\nAnswer:",
                                  add_special_tokens=False)
        fixed_tokens = 1 + len(fact_ids) + args.filler_mid_tokens + len(q_ids)
        filler_pre_len = max(0, args.T_pre - fixed_tokens)
        filler_pre = filler_pool.sample(filler_pre_len) if filler_pre_len > 0 else []
        filler_mid = filler_pool.sample(args.filler_mid_tokens)
        prefix_ids = [bos_id] + filler_pre + fact_ids + filler_mid + q_ids
        prefix_tensor = torch.tensor(
            [prefix_ids], dtype=torch.long, device=device,
        )

        # AR generate. `update_plasticity(None)` will be called by no-op
        # rollout path; we pass `phase="phase1"` to skip GRPO log_pi capture.
        out = autoregressive_rollout(
            wrapper, prefix_tensor,
            gen_length=args.gen_length,
            temperature=args.temperature, top_p=args.top_p,
            phase="phase1", grad_during_prefix=False, grad_during_gen=False,
        )
        gen_ids = out.new_tokens[0].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Score with BERT cosine
        sims = _bert_cosine_matrix([gen_text], fact.reference_answers, encoder)[0]
        best_idx = int(sims.argmax())
        max_sim = float(sims[best_idx])

        per_fact.append({
            "id": fact.id,
            "fact": fact.fact,
            "question": question_text,
            "generated": gen_text,
            "best_reference": fact.reference_answers[best_idx],
            "max_cosine": max_sim,
            "all_cosines": [float(s) for s in sims],
        })
        print(f"  [fact {fact.id:>3}] cos={max_sim:.3f} | gen='{gen_text[:80]}' | "
              f"ref='{fact.reference_answers[best_idx][:60]}'")

    # Aggregate
    sims_arr = np.asarray([p["max_cosine"] for p in per_fact])
    recall_70 = (sims_arr >= 0.7).mean()
    recall_50 = (sims_arr >= 0.5).mean()
    mean_sim = sims_arr.mean()
    print()
    print("=" * 60)
    print(f"  Held-out facts:    {len(heldout)}")
    print(f"  Mean BERT-cosine:  {mean_sim:.3f}")
    print(f"  Recall@0.7:        {recall_70:.1%}  ({(sims_arr >= 0.7).sum()}/{len(heldout)})")
    print(f"  Recall@0.5:        {recall_50:.1%}  ({(sims_arr >= 0.5).sum()}/{len(heldout)})")
    print("=" * 60)

    # Save per-fact details
    out_path = Path(args.ckpt).parent / "passphrase_eval.json"
    with out_path.open("w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "filler_mid_tokens": args.filler_mid_tokens,
            "T_pre": args.T_pre,
            "n_heldout": len(heldout),
            "mean_cosine": float(mean_sim),
            "recall_70": float(recall_70),
            "recall_50": float(recall_50),
            "per_fact": per_fact,
        }, f, indent=2)
    print(f"[eval] full results saved to {out_path}")


if __name__ == "__main__":
    main()
