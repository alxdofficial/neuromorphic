#!/usr/bin/env python3
"""Decompose WHERE Mamba's rank-2 memory comes from: the per-token hidden-state
trajectory (Mamba's nature) vs the adaptive-avg-pool readout (our choice).
  pre_pool : eff-rank of the full [T, d_recurrent] Mamba state sequence
  post_pool: eff-rank of the 128 avg-pooled bins (pre Llama-projection)
  memory   : eff-rank of the final 128 memory tokens (what we measured = ~2)
If pre_pool >> post_pool, the avg-pool readout is destroying rank (fixable, fairness
lever). If pre_pool is also ~2, Mamba's state is inherently low-rank (its nature)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

cfg = ReprConfig(
    n_flat_codes=128, d_continuous=1398, d_concept_baseline=1398, d_mt_value=1398, d_recurrent=1398,
    d_enc=768, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3072,
    d_mamba=1280, max_window_size=8192, fixed_window_size=1024,
)
device = "cuda"
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.float32).to(device)
llama.train(False)
mamba, _ = EPF.load_variant(
    "recurrent_baseline",
    Path("outputs/repr_learning/tranche6_mamba_1280_recurrent_baseline/ckpts/recurrent_baseline.best.pt"),
    cfg, llama)
mamba.to(device).train(False)
enc = mamba.encoder
embed = llama.get_input_embeddings()


def pr(X):  # participation ratio of singular values of centered X [N, d]
    Xc = X.float() - X.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    return (s.sum() ** 2 / (s ** 2).sum()).item()


samples = EPF.collect_samples(["biographical", "musique"], 3, tokenizer=tok, cfg=cfg,
                              chunk_size=8192, passages_per_chunk=600)
print(f"\n{'family':13s} {'T_ctx':>6} {'pre_pool':>9} {'post_pool':>10} {'memory(128)':>12}")
print("-" * 56)
with torch.no_grad():
    for s in samples:
        ctx = s.context_ids[s.context_mask.bool()].to(device)
        emb = embed(ctx.unsqueeze(0)).to(enc.in_proj.weight.dtype)   # [1,T,d_llama]
        h = enc.in_proj(emb)
        for norm, block in zip(enc.mamba_norms, enc.mamba_blocks):
            h = h + block(norm(h))
        h = enc.norm(h)
        h = enc.bottleneck(h)                                        # [1,T,d_recurrent]
        T = h.shape[1]
        pre = pr(h[0])
        pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), 128).transpose(1, 2)  # [1,128,d_rec]
        post = pr(pooled[0])
        mem = enc.proj_to_llama(pooled)                              # [1,128,d_llama]
        memr = pr(mem[0])
        print(f"{s.family:13s} {T:>6d} {pre:>9.1f} {post:>10.1f} {memr:>12.1f}")
