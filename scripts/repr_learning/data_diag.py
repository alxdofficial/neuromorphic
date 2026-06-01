#!/usr/bin/env python3
"""Data-side diagnostic: WHY does memory matter for biographical but ~not for
hotpot/musique/narrative (ΔMP +19.5 vs ~0)?

Phase A (default, no Llama): sanity-sample rows per family + measure
  - answer-in-context rate (is the gold answer even present in the chunk?)
  - answer-in-question rate (trivial leakage)
  - context / question / answer lengths
Phase B (--closed-book): closed-book F1 per family = Llama answering with
  EMPTY memory (just the chat scaffold + question). If real-world families
  score high closed-book, Llama already KNOWS them -> memory isn't used there
  -> explains why message-passing config is irrelevant for them.

Run:  python -m scripts.repr_learning.data_diag            # phase A
      python -m scripts.repr_learning.data_diag --closed-book
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from collections import defaultdict
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

ap = argparse.ArgumentParser()
ap.add_argument("--closed-book", action="store_true", help="Phase B: Llama F1 with empty memory")
ap.add_argument("--n", type=int, default=40, help="samples/family for rates")
ap.add_argument("--show", type=int, default=3, help="rows to print/family")
args = ap.parse_args()

cfg = ReprConfig(
    n_flat_codes=128, d_continuous=1398, d_concept_baseline=1398,
    d_mt_value=1398, d_recurrent=1398,
    d_enc=768, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3072,
    d_node_state=128, n_edges=68,
    graph_v5_K_node=128, graph_v5_K_edge=196, graph_v5_K_proposal=196,
    graph_v5_d_node=384, graph_v5_d_state=384, graph_v5_d_updater=640,
    graph_v5_updater_layers=5, graph_v5_n_message_rounds=6, graph_v5_mp_d_hidden=1024,
    d_mamba=1280, edge_token_packing="fused",
    max_window_size=8192, fixed_window_size=1024,
)
FAMS = ["biographical", "hotpot_qa", "musique", "narrative_qa"]

print("loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

samples = EPF.collect_samples(FAMS, args.n, tokenizer=tok, cfg=cfg,
                              chunk_size=8192, passages_per_chunk=600)
by_fam = defaultdict(list)
for s in samples:
    by_fam[s.family].append(s)


def decode_ctx(s):
    ids = s.context_ids[s.context_mask.bool()].tolist()
    return tok.decode(ids, skip_special_tokens=True)


def in_text(refs, text):
    nt = EPF.normalize_answer(text)
    return any(EPF.normalize_answer(r) and EPF.normalize_answer(r) in nt for r in refs)


# ════════════ PHASE A: data inspection (no Llama) ════════════
print("\n" + "=" * 78)
print("PHASE A — sanity samples + answer-presence rates")
print("=" * 78)
for fam in FAMS:
    rows = by_fam.get(fam, [])
    if not rows:
        print(f"\n### {fam}: NO SAMPLES")
        continue
    print(f"\n### {fam}  (n={len(rows)})")
    for s in rows[:args.show]:
        ctx = decode_ctx(s)
        q = tok.decode(s.question_ids.tolist(), skip_special_tokens=True)
        inc = in_text(s.answer_refs, ctx)
        print(f"  ── Q: {q.strip()[:160]}")
        print(f"     A: {s.answer_refs}")
        print(f"     ctx[{len(ctx)}ch] HEAD: {ctx[:340].strip()!r}")
        print(f"     ctx        TAIL: {ctx[-220:].strip()!r}")
        print(f"     answer-in-context: {inc}")
    # rates over all rows
    inc_ctx = sum(in_text(s.answer_refs, decode_ctx(s)) for s in rows) / len(rows)
    inc_q = sum(in_text(s.answer_refs, tok.decode(s.question_ids.tolist(), skip_special_tokens=True))
                for s in rows) / len(rows)
    ctx_toks = sum(int(s.context_mask.sum()) for s in rows) / len(rows)
    ans_len = sum(len(r.split()) for s in rows for r in s.answer_refs[:1]) / len(rows)
    print(f"  >>> answer-in-CONTEXT rate: {inc_ctx:.2f}   answer-in-QUESTION rate: {inc_q:.2f}")
    print(f"  >>> mean ctx tokens: {ctx_toks:.0f}   mean answer words: {ans_len:.1f}")

if not args.closed_book:
    print("\n(Phase B closed-book skipped — rerun with --closed-book)")
    sys.exit(0)

# ════════════ PHASE B: closed-book Llama F1 (empty memory) ════════════
print("\n" + "=" * 78)
print("PHASE B — closed-book Llama F1 (EMPTY memory, just question + scaffold)")
print("=" * 78)
from transformers import AutoModelForCausalLM
device = "cuda"
llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.float32).to(device)
llama.train(False)
ckpt = Path("outputs/repr_learning/outputs/repr_learning/"
            "tranche4_graph_v5_baseline_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt")
model, _ = EPF.load_variant("graph_v5_baseline", ckpt, cfg, llama)   # only for chat_template
chat_template = getattr(model, "chat_template", None)
print(f"chat_template={chat_template is not None}", flush=True)

print(f"\n  {'family':<14} {'CB F1':>7} {'CB EM':>7} {'%F1>.8':>7}   (%F1>.8 = guessable fraction)")
guessed = defaultdict(list)
with torch.no_grad():
    for fam in FAMS:
        rows = by_fam[fam]
        f1s, ems = [], []
        for i in range(0, len(rows), 4):
            batch = rows[i:i + 4]
            empty_mem = torch.zeros(len(batch), 0, cfg.d_llama, device=device)
            _, clean = EPF.generate_answers(
                llama, tok, empty_mem, batch, 40, device, chat_template=chat_template)
            for j, s in enumerate(batch):
                f1 = EPF.max_over_refs(clean[j], s.answer_refs, EPF.f1_score)
                em = EPF.max_over_refs(clean[j], s.answer_refs, EPF.em_score)
                f1s.append(f1)
                ems.append(em)
                if f1 > 0.8 and len(guessed[fam]) < 3:
                    q = tok.decode(s.question_ids.tolist(), skip_special_tokens=True)
                    guessed[fam].append((q.strip()[:75], s.answer_refs, clean[j].strip()[:45]))
        n = len(f1s)
        frac_hi = 100 * sum(1 for x in f1s if x > 0.8) / n
        print(f"  {fam:<14} {100*sum(f1s)/n:>7.1f} {100*sum(ems)/n:>7.1f} {frac_hi:>7.0f}")
print("\n  Distribution check: low %F1>.8 => only a small slice is guessable (filter those);")
print("  high %F1>.8 => dataset broadly known to Llama (would need replacing).")
for fam in FAMS:
    if guessed[fam]:
        print(f"\n  [{fam}] closed-book CORRECT (Llama already knew):")
        for q, refs, pred in guessed[fam]:
            print(f"    Q:{q!r}  gold:{refs}  pred:{pred!r}")
