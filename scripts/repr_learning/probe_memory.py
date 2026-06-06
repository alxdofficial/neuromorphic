"""Linear-probe diagnostic — is the ANSWER SPAN retrievable from the (frozen) passage memory given the KEY?

Vocabulary (locked): passage = whole input (all facts) -> memory; fact = (key, value); key = the question;
value = the fact's sentence; answer span = the coined token(s) inside the value (the only memory-load-bearing part).

Per arm, two phases:
  A) train the ENCODER via our key->value recon objective (frozen Llama decoder), exactly as in the experiments.
  B) FREEZE the encoder; train a small multi-head cross-attention PROBE to predict the answer-span vector from
     (key, passage-memory), with a within-batch contrastive loss. Eval REAL vs SHUFFLE retrieval accuracy.

Read: REAL >> SHUF  => the answer span IS retrievable from the memory -> binding PRESENT -> it's a READ problem.
      REAL ~= SHUF (~chance) => not retrievable even by a probe trained to do it -> binding ABSENT -> WRITE problem.
The probe can't cheat off the key: the answer span (coined) is NOT in the key, so success requires reading memory.
"""
import argparse, sys, time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a, make_stage_a_loader
from scripts.repr_learning.train_stage_a import stage_a_cfg
from scripts.repr_learning.train_stage_a_qa import to_recon_batch

DEV = "cuda"
D_ENC = {"vqvae_baseline": 1600, "memorizing_transformer_baseline": 1536,
         "flat_baseline": 1600, "memorizing_baseline": 1536}   # +old aliases; others default 1408


def _to(d):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}


@torch.no_grad()
def get_memory(model, ids, mask):
    """Encode the passage -> the memory SOURCE. For MT this is the (query-dependent) KV bank dict —
    its finalize_memory returns an EMPTY [B,0,d] placeholder and stashes the real bank in aux, so we
    must hand the probe the bank and let it retrieve_for_query by the key. Every other arm's memory is
    query-independent, so the source IS the [B, M, d_llama] prepend tensor."""
    embed = model.decoder.llama.get_input_embeddings()
    te = embed(ids).float()
    st = model.encoder.init_streaming_state(ids.size(0), ids.device, te.dtype)
    st, _ = model.encoder.streaming_write(st, te, mask, chunk_offset=0)
    mem, aux = model.encoder.finalize_memory(st)
    if aux.get("mt_bank") is not None:                             # MT: real memory is retrieved per query
        return aux["mt_bank"]
    return mem.float()                                            # [B, M, d_llama]


def _roll_src(src):
    """SHUF: give each row a DIFFERENT passage's memory source (roll across the batch)."""
    if isinstance(src, dict):
        return {k: (v.roll(1, 0) if torch.is_tensor(v) else v) for k, v in src.items()}
    return src.roll(1, 0)


@torch.no_grad()
def _materialize(model, src, key_ids, key_mask, embed, K):
    """Memory source -> the [B, M, d_llama] tensor the probe cross-attends. MT retrieves from its KV
    bank using the KEY as the query (its native, query-addressed read); all other arms are
    query-independent and returned unchanged."""
    if isinstance(src, dict):                                     # MT bank
        qe = embed(key_ids).float()                              # [B, Tk, d_llama]
        mem, _ = model.encoder.retrieve_for_query(src, qe, key_mask, K)
        return mem.float()
    return src


def span_target(embed, span_ids, span_mask):
    """Masked-mean embedding of the ANSWER SPAN (the coined token(s)) -> one target vector per fact."""
    with torch.no_grad():
        e = embed(span_ids).float()                                # [B, Ts, d]
    m = span_mask.float().unsqueeze(-1)
    return (e * m).sum(1) / m.sum(1).clamp_min(1.0)                # [B, d]


class Probe(nn.Module):
    """key tokens -> learned-pool to one query -> N multi-head cross-attn layers over memory (each
    locate-then-read) -> predict answer-span vector. N>=2 so it can 'find the fact, then read its value'."""
    def __init__(self, d_llama, d=256, heads=4, layers=2):
        super().__init__()
        self.k_in = nn.Linear(d_llama, d)
        self.m_in = nn.Linear(d_llama, d)
        self.kq = nn.Parameter(torch.randn(1, 1, d) * 0.02)        # learned query that pools the key tokens
        self.kpool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.xattn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Linear(d, d_llama)

    def forward(self, key_emb, key_mask, mem):
        K = self.k_in(key_emb)                                      # [B, Tk, d]
        M = self.m_in(mem)                                          # [B, Mtok, d]
        kpad = ~key_mask.bool()
        r, _ = self.kpool(self.kq.expand(K.size(0), -1, -1), K, K, key_padding_mask=kpad)   # [B,1,d] pooled key
        for xa, ln1, ffn, ln2 in zip(self.xattn, self.ln1, self.ffn, self.ln2):
            a, _ = xa(r, M, M)                                      # cross-attend memory
            r = ln1(r + a)
            r = ln2(r + ffn(r))
        return self.out(r.squeeze(1))                              # [B, d_llama] predicted answer-span vector


def probe_batch(d, gen=None):
    """Pick one fact j per passage; return (key, answer-span, passage) for that fact."""
    B, P, _ = d["keys"].shape
    j = torch.randint(P, (B,), device=d["keys"].device, generator=gen)
    bi = torch.arange(B, device=d["keys"].device)
    return (d["keys"][bi, j], d["keys_mask"][bi, j],               # key
            d["values"][bi, j], d["values_mask"][bi, j],           # answer span (= `values` in code)
            d["passage"], d["passage_mask"])                       # passage -> memory


def contrastive(probe, embed, mem, key_ids, key_mask, span_ids, span_mask, temp=0.07):
    pred = F.normalize(probe(embed(key_ids).float(), key_mask, mem), dim=-1)   # [B,d]
    tgt = F.normalize(span_target(embed, span_ids, span_mask), dim=-1)         # [B,d]
    logits = pred @ tgt.t() / temp                                            # [B,B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(1) == labels).float().mean()
    return loss, float(acc)


def run(variant, phaseA=800, phaseB=600, bs=8, n_pairs=8, oracle=False):
    cfg = replace(stage_a_cfg("nc8"),
                  graph_v6_d_updater=384, graph_v6_updater_layers=3, graph_v6_read_ffn_mult=1,
                  d_enc=D_ENC.get(variant, 1408), d_mamba=1408,
                  use_llama_lora=False,
                  b_diversity_scale=50.0, mt_diversity_scale=50.0)   # anti-collapse so the probe is fair
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    model = ReprLearningModel(cfg, variant=variant, llama_model=None).to(DEV)
    embed = model.decoder.llama.get_input_embeddings()
    K_RET = getattr(cfg, "n_flat_codes", 108)   # MT retrieval budget (~= bank length, retrieve ~all)

    def _mpool(ids, mask):
        e = embed(ids).float()                                     # [...,T,d]
        m = mask.float().unsqueeze(-1)
        return (e * m).sum(-2) / m.sum(-2).clamp_min(1.0)          # [...,d]

    def memory_of(d):
        # ORACLE positive control: a GENUINELY-bound KV store. One slot per fact = (key address) + (answer-span
        # payload). The key MUST be able to address its slot and the answer span IS in it -> a correct probe
        # MUST score REAL >> SHUF. Raw passage embeddings would NOT work: the key->value binding is positional,
        # and a bag of token embeddings drops it (that's what the first oracle run revealed).
        if oracle:
            return _mpool(d["keys"], d["keys_mask"]) + _mpool(d["values"], d["values_mask"])   # [B,P,d_llama]
        return get_memory(model, d["passage"], d["passage_mask"])

    loader = make_stage_a_loader(tok, batch_size=bs, n_pairs=n_pairs)
    it = iter(loader)
    # fixed val set for REAL/SHUF eval
    valit = iter(StageAKVDataset(tok, n_pairs=n_pairs, seed=999))
    val = [_to(collate_stage_a([next(valit) for _ in range(bs)])) for _ in range(8)]
    vgen = torch.Generator(device=DEV)

    # ---- Phase A: train the encoder via key->value recon (frozen Llama). Skipped for oracle. ----
    if not oracle:
        params = [p for p in model.trainable_parameters()]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95))
        for step in range(phaseA):
            d = _to(next(it))
            opt.zero_grad(set_to_none=True)
            out = model.compute_qa_loss(to_recon_batch(d))
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 300 == 0 or step == phaseA - 1:
                print(f"  [{variant}] phaseA step {step:4d}  recon_loss {float(out['loss']):.3f}", flush=True)

    # ---- Phase B: FREEZE encoder; train the probe (key + memory -> answer-span vector) ----
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    probe = Probe(cfg.d_llama).to(DEV)
    popt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.01)

    def evaluate():
        probe.eval()
        real = shuf = 0.0
        with torch.no_grad():
            for d in val:
                k, km, s, sm, pid, pm = probe_batch(d, _seed(vgen))
                src = memory_of(d)                                  # memory SOURCE (MT: bank dict; else tensor)
                real = real + contrastive(probe, embed, _materialize(model, src, k, km, embed, K_RET), k, km, s, sm)[1]
                shuf = shuf + contrastive(probe, embed, _materialize(model, _roll_src(src), k, km, embed, K_RET), k, km, s, sm)[1]
        probe.train()
        return real / len(val), shuf / len(val)

    for step in range(phaseB):
        d = _to(next(it))
        k, km, s, sm, pid, pm = probe_batch(d)
        src = memory_of(d)                                          # frozen encoder source (or bound-KV slots if oracle)
        mem = _materialize(model, src, k, km, embed, K_RET)        # MT retrieves by key; others key-independent
        popt.zero_grad(set_to_none=True)
        loss, acc = contrastive(probe, embed, mem, k, km, s, sm)
        loss.backward()
        popt.step()
        if step % 200 == 0:
            r, sh = evaluate()
            print(f"  [{variant}] probe step {step:4d}  train_acc {acc:.3f} | "
                  f"val REAL {r:.3f}  SHUF {sh:.3f}  (chance {1.0/bs:.3f})", flush=True)
    r, sh = evaluate()
    print(f"[{variant}] FINAL  probe-retrieval  REAL {r:.3f}  SHUF {sh:.3f}  (chance {1.0/bs:.3f})  "
          f"=> {'BINDING PRESENT (read problem)' if r - sh > 0.15 else 'binding ABSENT (write problem)'}",
          flush=True)


def _seed(gen):
    gen.manual_seed(0)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graph_v6_baseline")
    ap.add_argument("--phaseA", type=int, default=800)
    ap.add_argument("--phaseB", type=int, default=600)
    ap.add_argument("--oracle", action="store_true", help="positive control: probe the RAW passage embeddings")
    args = ap.parse_args()
    run(args.variant, args.phaseA, args.phaseB, oracle=args.oracle)


if __name__ == "__main__":
    main()
