#!/usr/bin/env python3
"""Diagnostic B-1: linear-probe answer recoverability — is the answer actually IN the
compressed memory? Separates WRITE quality (did the encoder store it?) from READ quality
(did the decoder surface it?).

For each arm: encode N (chunk, question, answer) samples → pool the memory to one vector.
Embed the gold answer via Llama's token table. Fit a ridge linear map memory_pool → answer_emb
on a train split, then on the held-out split measure:
  retr@1  — is the predicted answer-embedding nearest (cosine) to the TRUE answer among the
            test answers?  (chance = 1/N_test).  High ⇒ the answer is LINEARLY present in memory.
  cos     — mean cosine(predicted, true) on test.

retr@1 ≫ chance ⇒ info is stored (write OK) → the bottleneck is the READ/decoder.
retr@1 ≈ chance ⇒ the answer was never written into the memory → fix the WRITE.

Usage: python scripts/repr_learning/probe_recoverability.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as E

ARMS = [("graph_v6_baseline", "v6_1"), ("flat_baseline", "v2_1"),
        ("continuous_baseline", "v2_1"), ("recurrent_baseline", "v2_1"),
        ("memorizing_baseline", "v2_1")]


def ridge_fit(X, Y, lam=1.0):
    # W = (XᵀX + λI)⁻¹ XᵀY
    d = X.shape[1]
    A = X.t() @ X + lam * torch.eye(d, device=X.device)
    return torch.linalg.solve(A, X.t() @ Y)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ReprConfig(fixed_window_size=1024, max_window_size=8192)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    samples = E.collect_samples(["biographical", "hotpot_qa"], 48, tokenizer=tok, cfg=cfg,
                                chunk_size=8192, passages_per_chunk=600)
    print(f"{'arm':22s} {'N':>4s} {'d_mem':>6s} {'retr@1':>8s} {'chance':>7s} {'cos':>7s}")
    print("-" * 64)
    for variant, tag in ARMS:
        ck = ROOT / f"outputs/repr_learning/{tag}_{variant}/ckpts/{variant}.best.pt"
        if not ck.exists():
            ck = ROOT / f"outputs/repr_learning/{tag}_{variant}/ckpts/{variant}.last.pt"
        try:
            model, _ = E.load_variant(variant, ck, cfg, None)
        except Exception as e:
            print(f"{variant:22s}  LOAD FAIL: {str(e)[:40]}"); continue
        model = model.to(dev)
        embed = model.decoder.llama.get_input_embeddings()
        Mem, Ans = [], []
        for s0 in range(0, len(samples), 4):
            batch = samples[s0:s0 + 4]
            with torch.no_grad():
                mem, faux = E._stream_encode_batch(model, batch, dev, 1024)
                obj = mem if mem.shape[1] > 0 else (
                    faux["graph_v6_facts"]["value"] if faux.get("graph_v6_facts") is not None else None)
                if obj is None:
                    break
                pool = obj.float().mean(1)                          # [B, d_mem]
                for b, s in enumerate(batch):
                    ans = s.answer_refs[0] if s.answer_refs else ""
                    ids = tok(ans, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                    if ids.numel() == 0:
                        continue
                    ae = embed(ids).float().mean(1).squeeze(0)      # [d_llama]
                    Mem.append(pool[b].cpu()); Ans.append(ae.cpu())
        if len(Mem) < 8:
            print(f"{variant:22s}  (no memory)"); del model; torch.cuda.empty_cache(); continue
        X = torch.stack(Mem); Y = torch.stack(Ans)
        X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
        n = X.shape[0]; ntr = n // 2
        W = ridge_fit(X[:ntr], Y[:ntr])
        Yh = X[ntr:] @ W                                            # predicted answer emb
        Yt = Y[ntr:]
        Yhn = torch.nn.functional.normalize(Yh, dim=-1)
        Ytn = torch.nn.functional.normalize(Yt, dim=-1)
        sim = Yhn @ Ytn.t()                                        # [te, te]
        retr1 = (sim.argmax(1) == torch.arange(sim.shape[0])).float().mean().item()
        cos = (Yhn * Ytn).sum(-1).mean().item()
        print(f"{variant:22s} {n:>4d} {X.shape[1]:>6d} {100*retr1:7.1f}% "
              f"{100/sim.shape[0]:6.1f}% {cos:7.3f}")
        del model; torch.cuda.empty_cache()
    print("\nretr@1 ≫ chance ⇒ answer is in memory (write OK, fix READ); ≈ chance ⇒ fix WRITE")


if __name__ == "__main__":
    main()
