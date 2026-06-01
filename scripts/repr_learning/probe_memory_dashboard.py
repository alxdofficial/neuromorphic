#!/usr/bin/env python3
"""Cross-model memory-health dashboard (diagnostic suite A). For each memory arm,
encode N samples and report, in one comparable table:

  M, d        — memory shape (slots × dim)
  mem_norm    — mean per-slot L2 norm of the memory  (prepend arms inject this into
                Llama's token stream; compare to text_norm ≈ Llama embed norm ≈ 0.93)
  norm_ratio  — mem_norm / text_norm  (≫1 ⇒ out-of-distribution "loud" memory; the
                49× bug.  N/A across spaces for graph, which reads facts, not tokens)
  within_cos  — mean pairwise cosine of the M slots WITHIN one input (→1 ⇒ slot collapse:
                the "192 tokens" are one vector repeated)
  cross_cos   — mean cosine of the per-input mean-memory ACROSS different inputs
                (→1 ⇒ CONSTANT memory: same regardless of context = carries no info)
  eff_rank%   — participation ratio of all collected slots, as % of dim (low ⇒ low-rank/
                collapsed; high ⇒ memory spans many directions)

Reads each arm's best/last ckpt. graph_v6 uses its fact-tokens as the memory object
(different vector space, so mem_norm/norm_ratio are not cross-comparable for it — the
RELATIVE metrics within_cos/cross_cos/eff_rank ARE comparable across all arms).

Usage: python scripts/repr_learning/probe_memory_dashboard.py
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as E

# (variant, tag) — graph_v6 from the v6_1 retrain, the rest from the v2_1 sweep.
ARMS = [
    ("graph_v6_baseline", "v6_1"),
    ("flat_baseline", "v2_1"),
    ("continuous_baseline", "v2_1"),
    ("recurrent_baseline", "v2_1"),
    ("memorizing_baseline", "v2_1"),
]


def participation_ratio(X):
    # X: [n, d] — (Σλ)² / Σλ²  of the covariance eigenvalues, via singular values.
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X.float())
    l = s ** 2
    return (l.sum() ** 2 / (l ** 2).sum().clamp_min(1e-12)).item()


def mean_pairwise_cos(M):  # M: [m, d] within-input slots
    Mn = torch.nn.functional.normalize(M.float(), dim=-1)
    C = Mn @ Mn.t()
    m = M.shape[0]
    off = C[~torch.eye(m, dtype=torch.bool, device=C.device)]
    return off.mean().item() if off.numel() else float("nan")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ReprConfig(fixed_window_size=1024, max_window_size=8192)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    # reference: mean Llama token-embedding norm ≈ 0.93 (measured)
    fams = ["biographical", "hotpot_qa"]
    samples = E.collect_samples(fams, 24, tokenizer=tok, cfg=cfg, chunk_size=8192,
                                passages_per_chunk=600)
    print(f"{'arm':22s} {'M':>4s} {'d':>5s} {'mem_norm':>9s} {'ratio':>7s} "
          f"{'within_cos':>11s} {'cross_cos':>10s} {'eff_rank%':>10s}")
    print("-" * 92)
    TXT = 0.93  # measured Llama-3.2-1B embed mean L2
    for variant, tag in ARMS:
        ck = ROOT / f"outputs/repr_learning/{tag}_{variant}/ckpts/{variant}.best.pt"
        if not ck.exists():
            ck = ROOT / f"outputs/repr_learning/{tag}_{variant}/ckpts/{variant}.last.pt"
        if not ck.exists():
            print(f"{variant:22s}  (no ckpt)"); continue
        try:
            model, _ = E.load_variant(variant, ck, cfg, None)
        except Exception as e:
            print(f"{variant:22s}  LOAD FAIL: {str(e)[:50]}"); continue
        model = model.to(dev)
        per_input_mean, all_slots, within = [], [], []
        for s0 in range(0, len(samples), 4):
            batch = samples[s0:s0 + 4]
            with torch.no_grad():
                mem, faux = E._stream_encode_batch(model, batch, dev, 1024)
            if mem.shape[1] > 0:
                obj = mem                          # [B, M, d_llama] prepend memory
            elif faux.get("graph_v6_facts") is not None:
                obj = faux["graph_v6_facts"]["value"]   # [B, K_edge, d_read] graph facts
            else:
                obj = None
            if obj is None:
                continue
            obj = obj.float().cpu()
            for b in range(obj.shape[0]):
                Mb = obj[b]                         # [M, d]
                per_input_mean.append(Mb.mean(0))
                all_slots.append(Mb)
                within.append(mean_pairwise_cos(Mb))
        if not all_slots:
            print(f"{variant:22s}  (no memory)"); continue
        S = torch.cat(all_slots, 0)               # [N*M, d]
        Mn, d = all_slots[0].shape
        mem_norm = S.norm(dim=-1).mean().item()
        pim = torch.stack(per_input_mean)         # [N, d]
        pim_n = torch.nn.functional.normalize(pim, dim=-1)
        Cx = pim_n @ pim_n.t()
        n = pim.shape[0]
        cross = Cx[~torch.eye(n, dtype=torch.bool)].mean().item()
        within_m = sum(within) / len(within)
        er = 100 * participation_ratio(S) / d
        ratio = f"{mem_norm/TXT:6.1f}" if variant != "graph_v6_baseline" else "  (n/a)"
        print(f"{variant:22s} {Mn:>4d} {d:>5d} {mem_norm:9.2f} {ratio:>7s} "
              f"{within_m:11.4f} {cross:10.4f} {er:10.1f}")
        del model
        torch.cuda.empty_cache()
    print(f"\nreference: Llama token-embedding norm ≈ {TXT};  within_cos/cross_cos →1 = collapse/constant; eff_rank% low = low-rank")


if __name__ == "__main__":
    main()
