#!/usr/bin/env python3
"""HOW GOOD IS LLAMA AT READING EACH MEMORY — fair version (v2).

Fair design = AGNOSTIC STORE + QUERY-CONDITIONED READ:
  - Every mechanism builds a QUESTION-AGNOSTIC 128-token store (graph_v5's
    question-conditioned readout is turned OFF here too, so no model gets a
    head start tailoring its memory to the question).
  - A single shared QUERY-CONDITIONED read head (the probe) retrieves from each
    store using the QUESTION as its attention query — identical for all models.
    This is the fair "given a query-aware reader, which substrate retains the
    answer?" and it implements "the read module sees the question" for everyone.

Decodable@k = top-k retrieval of the gold answer from the frozen store by the
query-conditioned reader (held-out, k-fold CV → mean±std). Compare to each
mechanism's Llama-read (containment) from the rescore:
  Decodable >> Llama-read  => read bottleneck (info there, frozen Llama wastes it)
  Decodable ~ chance       => not retrievable even with a query-aware reader
                              => genuine write/storage loss.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

ROOT = Path(__file__).resolve().parents[2]
CKPTS = {
    "graph_v5_baseline":   ROOT / "outputs/repr_learning/v56_graph_v5_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt",
    "recurrent_baseline":  ROOT / "outputs/repr_learning/tranche6_mamba_1280_recurrent_baseline/ckpts/recurrent_baseline.best.pt",
    "flat_baseline":       ROOT / "outputs/repr_learning/tranche6_baselines_flat_baseline/ckpts/flat_baseline.best.pt",
    "continuous_baseline": ROOT / "outputs/repr_learning/tranche6_baselines_continuous_baseline/ckpts/continuous_baseline.best.pt",
    "memorizing_baseline": ROOT / "outputs/repr_learning/tranche6_baselines_memorizing_baseline/ckpts/memorizing_baseline.best.pt",
}
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--family", default="biographical")
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--batch-size", type=int, default=4)
args = ap.parse_args()
device = "cuda"

cfg = ReprConfig(
    fixed_window_size=1024, max_window_size=8192,
    d_node_state=128, n_edges=68, n_flat_codes=128,
    d_continuous=1398, d_concept_baseline=1398, d_mt_value=1398, d_recurrent=1398,
    graph_v5_K_node=128, graph_v5_K_edge=196, graph_v5_K_proposal=196,
    graph_v5_d_node=384, graph_v5_d_state=384, graph_v5_d_updater=640,
    graph_v5_updater_layers=5, graph_v5_n_message_rounds=6, graph_v5_mp_d_hidden=1024,
    d_enc=768, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3072,
    d_mamba=1280, edge_token_packing="fused",
)
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.bfloat16).to(device)
llama.train(False)
embed = llama.get_input_embeddings()
d = llama.config.hidden_size

samples = EPF.collect_samples([args.family], args.n, tokenizer=tok, cfg=cfg,
                              chunk_size=8192, passages_per_chunk=600)
samples = [s for s in samples if s.answer_refs]
N = len(samples)
print(f"[probe v2] {N} {args.family} samples — AGNOSTIC store + QUERY-CONDITIONED read, {args.folds}-fold")


def masked_mean_embed(ids, mask=None):
    e = embed(ids.unsqueeze(0).to(device))[0].float()      # [L,d]
    if mask is not None:
        m = mask.to(device).bool()
        e = e[m] if m.any() else e
    return e.mean(0)


with torch.no_grad():
    gold = torch.stack([masked_mean_embed(
        tok(s.answer_refs[0], return_tensors="pt", add_special_tokens=False).input_ids[0]) for s in samples])
    qemb = torch.stack([masked_mean_embed(
        s.question_ids, getattr(s, "question_mask", None)) for s in samples])      # [N,d]
mu = gold.mean(0, keepdim=True)
goldc = F.normalize(gold - mu, dim=-1)                       # centered + normed targets


def encode_agnostic(model, sams):
    """Question-AGNOSTIC streaming encode (no question stash) — graph_v5's
    finalize sees no question_embeds → q_vec=None → readout is question-blind."""
    enc = model.encoder
    mems = []
    with torch.no_grad():
        for i in range(0, len(sams), args.batch_size):
            b = sams[i:i + args.batch_size]
            ctx = torch.stack([s.context_ids for s in b]).to(device)
            msk = torch.stack([s.context_mask for s in b]).to(device)
            te = embed(ctx)
            state = enc.init_streaming_state(ctx.shape[0], device=device, dtype=te.dtype)
            for s0 in range(0, ctx.shape[1], 1024):
                e0 = min(s0 + 1024, ctx.shape[1])
                state, _ = enc.streaming_write(state, te[:, s0:e0, :],
                                               attention_mask=msk[:, s0:e0], chunk_offset=s0)
            mem, _ = enc.finalize_memory(state)
            mems.append(mem.float().cpu())
    return torch.cat(mems, 0)                                # [N,128,d]


class QCProbe(nn.Module):
    """Query-conditioned read head: the QUESTION is the attention query."""
    def __init__(self, d):
        super().__init__()
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, mem, q):                               # mem [B,128,d], q [B,d]
        query = self.wq(q)
        sc = torch.einsum("bld,bd->bl", self.wk(mem), query) / (mem.shape[-1] ** 0.5)
        a = sc.softmax(-1)
        pooled = torch.einsum("bl,bld->bd", a, self.wv(mem))
        return self.out(pooled)


g = torch.Generator().manual_seed(0)
perm = torch.randperm(N, generator=g)
folds = [perm[i::args.folds] for i in range(args.folds)]     # interleaved folds

print(f"\n{'mechanism':22s} {'decodable@1':>16s} {'decodable@5':>16s}")
print("-" * 58)
for variant in CKPTS:
    model, _ = EPF.load_variant(variant, CKPTS[variant], cfg, llama)
    model = model.to(device).train(False)
    mem_all = encode_agnostic(model, samples).to(device)     # [N,128,d]
    del model
    torch.cuda.empty_cache()
    t1s, t5s, chances = [], [], []
    for fk in range(args.folds):
        va = folds[fk]
        tr = torch.cat([folds[j] for j in range(args.folds) if j != fk])
        probe = QCProbe(d).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-3)
        for step in range(args.steps):
            idx = tr[torch.randint(0, len(tr), (64,))]
            pred = F.normalize(probe(mem_all[idx], qemb[idx]), dim=-1)
            loss = (1 - (pred * goldc[idx]).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        probe.train(False)
        with torch.no_grad():
            pv = F.normalize(probe(mem_all[va], qemb[va]), dim=-1)
            sim = pv @ goldc[va].T
            rank = sim.argsort(-1, descending=True)
            tgt = torch.arange(len(va), device=device)
            t1s.append((rank[:, 0] == tgt).float().mean().item() * 100)
            t5s.append((rank[:, :5] == tgt[:, None]).any(-1).float().mean().item() * 100)
            chances.append(100.0 / len(va))
    import statistics as st
    print(f"{variant:22s} {st.mean(t1s):7.1f} ± {st.pstdev(t1s):4.1f}   "
          f"{st.mean(t5s):7.1f} ± {st.pstdev(t5s):4.1f}    (chance@1≈{st.mean(chances):.1f})", flush=True)
    del mem_all
    torch.cuda.empty_cache()
print("\nAgnostic store + query-conditioned read, identical for all. Above chance =>")
print("answer retrievable from that substrate by a query-aware reader; compare to")
print("frozen-Llama containment (rescore) to locate read vs write bottleneck per model.")
