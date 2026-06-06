"""Stage-A retrieval through the FROZEN Llama decoder (ReprLearningModel.compute_qa_loss).

Tests whether routing the coined Stage-A data through the real pretrained decoder — instead
of the from-scratch read in stage_a_read.py (which collapses) — FIXES the read-ignores-memory
collapse. QA-only (conditioned, compression-friendly), CLOSED-BOOK (decoder sees question+memory,
never the passage — guaranteed by compute_qa_loss), ratio-1 (nc8 budget), with the load-bearing
controls + effective-rank telemetry.

PASS = REAL top1/loss beats OFF (zero-memory) AND SHUFFLE, and erank_cross does NOT collapse
toward ~1 (the continuous failure signature). Decoder stays frozen Llama (use_llama_lora=False).

  python scripts/repr_learning/train_stage_a_qa.py --variant graph_v6_baseline
  python scripts/repr_learning/train_stage_a_qa.py --variant slot_attention_baseline
"""
import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer

from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a, make_stage_a_loader
from scripts.repr_learning.train_stage_a import stage_a_cfg

DEV = "cuda"


@dataclass
class QABatchLike:                    # attribute-access shape compute_qa_loss expects (data_qa.py:40)
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    question_ids: torch.Tensor
    question_mask: torch.Tensor
    answer_ids: torch.Tensor
    answer_mask: torch.Tensor
    answer_content_mask: torch.Tensor


def _to(d):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}


def to_qabatch(d, gen=None):
    """One random (question,value) pair per passage → B distinct-passage rows. Avoids
    re-encoding the passage P times (compute_qa_loss has no write-once/read-many hook) AND
    makes the shuffle control land a strictly different passage (every row is a distinct one)."""
    B, P, Tk = d["keys"].shape
    j = torch.randint(P, (B,), device=d["keys"].device, generator=gen)
    bi = torch.arange(B, device=d["keys"].device)
    return QABatchLike(
        context_ids=d["passage"], context_mask=d["passage_mask"],     # passage → memory (closed-book)
        question_ids=d["keys"][bi, j], question_mask=d["keys_mask"][bi, j],
        answer_ids=d["values"][bi, j], answer_mask=d["values_mask"][bi, j],
        answer_content_mask=d["values_mask"][bi, j],                  # every value tok is load-bearing
    )


def to_recon_batch(d, gen=None):
    """KEY-ADDRESSED reconstruction. Memory = the WHOLE passage (all facts). The KEY for a selected
    fact j conditions the decoder; the target is fact j's SENTENCE — the input-text sentence, e.g.
    'Dax's mentor is Zylo Praxis.' — NOT the whole passage and NOT the bare short value. Loss on the
    coined owner+value tokens of that sentence. The key genuinely SELECTS which of the passage's
    facts to reproduce (so REAL≪SHUFFLE = memory read AND addressed by the key)."""
    B, P, Tk = d["keys"].shape
    j = torch.randint(P, (B,), device=d["keys"].device, generator=gen)   # pick one fact per passage
    bi = torch.arange(B, device=d["keys"].device)
    return QABatchLike(
        context_ids=d["passage"], context_mask=d["passage_mask"],             # whole passage → memory
        question_ids=d["keys"][bi, j], question_mask=d["keys_mask"][bi, j],   # the KEY selects fact j
        answer_ids=d["sentences"][bi, j], answer_mask=d["sentences_mask"][bi, j],   # target = sentence j
        answer_content_mask=d["sentences_content_mask"][bi, j],               # loss on coined toks in sent j
    )


@torch.no_grad()
def copy_score(model, ctx_ids, ctx_mask):
    """Is the prepended memory just a COPY of the passage token embeddings? For each memory
    token, the MAX cosine to any passage embedding, averaged. ~1 = near-verbatim copy (mamba's
    trivial ratio-1 win — memory ≈ the passage); low = a genuine re-encoding (graph's structured
    facts, which route through the node-bank bottleneck and cannot copy). 0 = nothing prepended."""
    embed = model.decoder.llama.get_input_embeddings()
    te = embed(ctx_ids).float()                                  # [B, T, d] passage embeddings
    st = model.encoder.init_streaming_state(ctx_ids.size(0), ctx_ids.device, te.dtype)
    st, _ = model.encoder.streaming_write(st, te, ctx_mask, chunk_offset=0)
    mem, _ = model.encoder.finalize_memory(st)                   # [B, M, d_llama] = what gets prepended
    if mem.shape[1] == 0:
        return 0.0                                               # inject-mode graph: no prepend tokens
    Mn = mem.float() / mem.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    En = te / te.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cos = torch.einsum("bmd,btd->bmt", Mn, En)                   # [B, M, T] mem-token ↔ passage-token cos
    cos = cos.masked_fill(~ctx_mask.bool().unsqueeze(1), -1.0)   # ignore padded passage positions
    return float(cos.max(dim=-1).values.mean())                 # per mem token: best match to a passage tok


@torch.no_grad()
def erank(model, ctx_ids, ctx_mask, variant):
    """Within/cross effective rank of the WRITTEN memory (collapse monitor). Reads the right
    object per arm: graph_v6 fact-tokens from aux, continuous slots from finalize_memory."""
    embed = model.decoder.llama.get_input_embeddings()
    te = embed(ctx_ids).float()
    st = model.encoder.init_streaming_state(ctx_ids.size(0), ctx_ids.device, te.dtype)
    st, _ = model.encoder.streaming_write(st, te, ctx_mask, chunk_offset=0)
    mem, aux = model.encoder.finalize_memory(st)
    if variant == "graph_v6_baseline":
        mem = aux["graph_v6_facts"]["value"]
    elif variant in ("memorizing_baseline", "memorizing_transformer_baseline"):  # MT stores a bank; finalize returns empty
        mem = aux["mt_bank"]["values"]              # [B, N, d] — the values that get retrieved
    mem = mem.float()
    N, M = mem.shape[0], mem.shape[1]

    def er(X):
        s = torch.linalg.svdvals(X); s = s[s > 1e-8]
        if s.numel() == 0:
            return 0.0
        p = s / s.sum()
        return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))

    within = sum(er(mem[i]) for i in range(min(N, 8))) / min(N, 8)
    cross = er(mem.reshape(N, -1))
    return within, cross, M


def build_val(tok, n_pairs, n_items, bs, seed=999):
    it = iter(StageAKVDataset(tok, n_pairs=n_pairs, seed=seed))
    items = [next(it) for _ in range(n_items)]
    return [_to(collate_stage_a(items[i:i + bs])) for i in range(0, n_items, bs)]


def first_tok_mask(cm):
    """True only at the FIRST content position per row — the token that can ONLY come from
    memory CONTENT (no intra-word syllable-continuation bypass). REAL vs SHUF here is the
    decisive content-retrieval signal that teacher-forced full-answer top1 washes out."""
    return cm & (cm.cumsum(1) == 1)


@torch.no_grad()
def run_val(model, val, variant, gen, oracle=False, batch_fn=None):
    """oracle=True: REAL/SHUF use the raw-passage-embedding memory (lossless ceiling)
    instead of the encoder's memory — measures the FROZEN decoder's capability to
    locate-and-read from a faithful memory, the ceiling for content retrieval.
    batch_fn: overrides the batch builder (recon mode passes to_recon_batch)."""
    model.eval()
    agg = {"REAL": [0., 0., 0.], "OFF": [0., 0., 0.], "SHUF": [0., 0., 0.]}   # [loss, top1, ft_top1]
    wi = cr = 0.0
    for d in val:
        qb = batch_fn(d, gen) if batch_fn is not None else to_qabatch(d, gen)
        qb_ft = replace(qb, answer_content_mask=first_tok_mask(qb.answer_content_mask))
        for tag, kw in [("REAL", {}), ("OFF", {"zero_memory": True}), ("SHUF", {"shuffle_memory": True})]:
            if oracle and tag in ("REAL", "SHUF"):
                kw = {**kw, "oracle_memory": True}                 # lossless passage memory
            out = model.compute_qa_loss(qb, **kw)
            ft = model.compute_qa_loss(qb_ft, **kw)                 # top1 on first content token only
            agg[tag][0] += float(out["loss_recon"]); agg[tag][1] += float(out["top1_acc"])
            agg[tag][2] += float(ft["top1_acc"])
        if not oracle:                                              # erank reads the encoder mem, N/A for oracle
            w, c, _ = erank(model, qb.context_ids, qb.context_mask, variant); wi += w; cr += c
    cs = 0.0 if oracle else copy_score(model, val[0]["passage"], val[0]["passage_mask"])
    model.train()
    n = len(val)
    return {t: (v[0] / n, v[1] / n, v[2] / n) for t, v in agg.items()} | {"erank": (wi / n, cr / n), "copy": cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graph_v6_baseline",
                    choices=["graph_v6_baseline", "slot_attention_baseline", "mamba_baseline",
                             "vqvae_baseline", "memorizing_transformer_baseline",
                             # back-compat aliases
                             "continuous_baseline", "recurrent_baseline", "flat_baseline",
                             "memorizing_baseline"])
    ap.add_argument("--oracle", action="store_true",
                    help="eval-only ceiling: bypass the encoder, feed raw passage embeddings as memory. "
                         "Reports ORACLE(REAL) vs SHUF — does the FROZEN decoder read a faithful memory by content?")
    ap.add_argument("--recon", action="store_true",
                    help="RECONSTRUCTION objective: value:=passage. Teacher-force the whole passage "
                         "through the frozen decoder. Reports REAL/OFF/SHUF recon loss + grad-norm-to-encoder.")
    ap.add_argument("--graph-inject", action="store_true",
                    help="revert graph_v6 to the per-decode-token INJECT read (main-branch design). "
                         "Default (this branch): PREPEND the FiLM fact-tokens like the baselines.")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA-all: rank-16 q/v LoRA on the (otherwise frozen) Llama for EVERY arm "
                         "(main's recipe) — lets the decoder LEARN to query the memory. Also restores "
                         "the diversity losses (b/mt=50) so the prepend-baselines don't collapse.")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--val-items", type=int, default=64)
    args = ap.parse_args()

    # ~60M param-match across arms — the MEMORY bottleneck (108×2048) is unchanged; only the
    # internal COMPUTE width changes: shrink graph's updater+read (nc8 had it at 450M), inflate
    # continuous's bi-transformer + Mamba's d_model toward 60M. (My nc8 had bumped d_updater/d_enc
    # to 2048 but left d_mamba at 1280 — that inconsistency was the param spread.)
    # per-variant d_enc equalizes trainable params (others default to 1408)
    D_ENC = {"vqvae_baseline": 1600, "memorizing_transformer_baseline": 1536,
             "flat_baseline": 1600, "memorizing_baseline": 1536}   # +old aliases
    cfg = replace(stage_a_cfg("nc8"),                                     # ratio-1; inject_layer 13
                  graph_v6_d_updater=384, graph_v6_updater_layers=3, graph_v6_read_ffn_mult=1,
                  d_enc=D_ENC.get(args.variant, 1408), d_mamba=1408,
                  graph_v6_prepend_read=not args.graph_inject,        # PREPEND fact-tokens (new-branch design)
                  use_llama_lora=args.lora,                           # --lora: decoder learns to query the memory
                  b_diversity_scale=50.0 if args.lora else 0.0,       # …and restore main's anti-collapse
                  mt_diversity_scale=50.0 if args.lora else 0.0)      # losses so baselines don't collapse
    # per-variant d_enc equalizes trainable params: flat 59 / continuous 59 / MT 61 / Mamba 60 / graph 56M.
    # memory bottleneck 108×2048 identical; graph deliberately UNDER the baselines (no param artifact).
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    model = ReprLearningModel(cfg, variant=args.variant, llama_model=None).to(DEV)   # self-loads frozen Llama
    params = [p for p in model.trainable_parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    B = cfg.graph_v6_K_edge * cfg.graph_v6_d_read
    print(f"[stage-a-qa] variant={args.variant} llama={cfg.llama_model} FROZEN-decoder "
          f"trainable={sum(p.numel() for p in params) / 1e6:.0f}M  B={B:,}  lora={cfg.use_llama_lora}",
          flush=True)

    loader = make_stage_a_loader(tok, batch_size=args.batch_size, n_pairs=args.n_pairs)
    val = build_val(tok, args.n_pairs, args.val_items, args.batch_size)
    val_gen = torch.Generator(device=DEV)

    if args.oracle:                       # eval-only ceiling — encoder bypassed, nothing to train
        val_gen.manual_seed(0)
        r = run_val(model, val, args.variant, val_gen, oracle=True)
        print(f"  ORACLE | REAL[ft {r['REAL'][2]:.3f} all {r['REAL'][1]:.3f}] "
              f"OFF[ft {r['OFF'][2]:.3f} all {r['OFF'][1]:.3f}] "
              f"SHUF[ft {r['SHUF'][2]:.3f} all {r['SHUF'][1]:.3f}]  "
              f"(raw-passage-embed memory = frozen-decoder ceiling)", flush=True)
        return

    # RECON mode: KEY-conditioned, answer:=the selected fact's SENTENCE (to_recon_batch picks one key
    # per passage; loss on that sentence's coined owner+value tokens). NOT the whole passage.
    batch_fn = to_recon_batch if args.recon else None
    if args.recon:
        print("[recon] KEY-conditioned (the KV key), answer:=fact's sentence (coined-token loss), "
              f"decoder={'LoRA' if cfg.use_llama_lora else 'frozen'}, monitor=enc grad-norm", flush=True)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]

    t0 = time.time(); step = 0; enc_gn = 0.0
    for d in loader:
        d = _to(d)
        qb = batch_fn(d) if batch_fn is not None else to_qabatch(d)
        opt.zero_grad(set_to_none=True)
        out = model.compute_qa_loss(qb)
        out["loss"].backward()
        enc_gn = sum(float(p.grad.detach().float().norm()) ** 2
                     for p in enc_params if p.grad is not None) ** 0.5   # ||grad|| reaching the WRITE
        torch.nn.utils.clip_grad_norm_(params, getattr(cfg, "grad_clip", 1.0))
        opt.step()
        if step % args.eval_every == 0:
            val_gen.manual_seed(0)
            r = run_val(model, val, args.variant, val_gen, batch_fn=batch_fn)
            if args.recon:
                # PASS = REAL loss drops AND REAL ≪ SHUF/OFF (memory read + content-specific);
                # erank_c high (no collapse); enc_gn > 0 (write is being trained).
                print(f"  step {step:4d} | RECON loss REAL {r['REAL'][0]:.3f} OFF {r['OFF'][0]:.3f} "
                      f"SHUF {r['SHUF'][0]:.3f} | top1 REAL {r['REAL'][1]:.3f} SHUF {r['SHUF'][1]:.3f} | "
                      f"erank_c {r['erank'][1]:.1f} copy {r['copy']:.2f} enc_gn {enc_gn:.1e}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            else:
                # ft = first-content-token top1 (the decisive content-retrieval signal); all = full-answer
                print(f"  step {step:4d} | "
                      f"REAL[ft {r['REAL'][2]:.3f} all {r['REAL'][1]:.3f}] "
                      f"OFF[ft {r['OFF'][2]:.3f} all {r['OFF'][1]:.3f}] "
                      f"SHUF[ft {r['SHUF'][2]:.3f} all {r['SHUF'][1]:.3f}] "
                      f"erank_c {r['erank'][1]:.1f}  ({time.time() - t0:.0f}s)", flush=True)
        step += 1
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
