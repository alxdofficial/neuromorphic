#!/usr/bin/env python3
"""Verbosity-robust re-score: span-F1 punishes verbose-but-correct outputs
(found in the oracle spot-check). Re-score oracle (full raw context, zero-shot) vs
graph_v5 (v5.6) vs Mamba (tranche6) on all families with:
  - F1      : squad F1 (current metric; verbosity-SENSITIVE)
  - Recall  : squad recall component (verbosity-ROBUST: verbose gen not penalized)
  - Contain : gold span appears verbatim (normalized) in the generation (binary)
De-confounds "memory beats full context" (format) from "graph trails Mamba on musique" (real).
"""
import sys, re, string
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

device = "cuda"
FAMS = ["biographical", "hotpot_qa", "musique", "narrative_qa"]
NPF = 32
GRAPH_CKPT = Path("outputs/repr_learning/v56_graph_v5_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt")
MAMBA_CKPT = Path("outputs/repr_learning/tranche6_mamba_1280_recurrent_baseline/ckpts/recurrent_baseline.best.pt")

cfg = ReprConfig(
    n_flat_codes=128, d_continuous=1398, d_concept_baseline=1398, d_mt_value=1398, d_recurrent=1398,
    d_enc=768, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3072,
    d_node_state=128, n_edges=68,
    graph_v5_K_node=128, graph_v5_K_edge=196, graph_v5_K_proposal=196,
    graph_v5_d_node=384, graph_v5_d_state=384, graph_v5_d_updater=640,
    graph_v5_updater_layers=5, graph_v5_n_message_rounds=6, graph_v5_mp_d_hidden=1024,
    d_mamba=1280, edge_token_packing="fused",
    max_window_size=8192, fixed_window_size=1024,
)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def norm(s):
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def score(gen, refs):
    g = norm(gen); gt = g.split()
    bf = br = bc = 0.0
    for r in refs:
        rn = norm(r); rt = rn.split()
        bc = max(bc, 1.0 if rn and rn in g else 0.0)
        if not rt:
            continue
        ns = sum((Counter(gt) & Counter(rt)).values())
        rec = ns / len(rt)
        prec = ns / len(gt) if gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        bf = max(bf, f1); br = max(br, rec)
    return bf, br, bc


print("loading llama + tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.bfloat16).to(device)
llama.train(False)

graph, gstep = EPF.load_variant("graph_v5_baseline", GRAPH_CKPT, cfg, llama)
graph.to(device).train(False)
mamba, mstep = EPF.load_variant("recurrent_baseline", MAMBA_CKPT, cfg, llama)
mamba.to(device).train(False)
print(f"graph_v5 @ {gstep}, mamba @ {mstep}", flush=True)

samples = EPF.collect_samples(FAMS, NPF, tokenizer=tok, cfg=cfg, chunk_size=8192, passages_per_chunk=600)
by_fam = defaultdict(list)
for s in samples:
    by_fam[s.family].append(s)


def mem_scores(model, sams):
    ct = getattr(model, "chat_template", None)
    acc = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for i in range(0, len(sams), 4):
            b = sams[i:i + 4]
            mem, aux = EPF._stream_encode_batch(model, b, device, 1024)
            _, clean = EPF.generate_answers(llama, tok, mem.detach(), b, 40, device,
                                            memory_mask=aux.get("memory_mask"), chat_template=ct)
            for j, s in enumerate(b):
                for k, val in enumerate(score(clean[j], s.answer_refs)):
                    acc[k] += val
    return [100 * a / len(sams) for a in acc]


def oracle_scores(sams):
    acc = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for s in sams:
            ctx = tok.decode(s.context_ids[s.context_mask.bool()].tolist())
            q = tok.decode(s.question_ids.tolist()).strip()
            p = tok.apply_chat_template([{"role": "user", "content": ctx + "\n\n" + q}],
                                        add_generation_prompt=True, tokenize=False)
            ids = tok(p, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=8800).to(device)
            out = llama.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
            gen = EPF._truncate_at_natural_end(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            for k, val in enumerate(score(gen, s.answer_refs)):
                acc[k] += val
    return [100 * a / len(sams) for a in acc]


print(f"\n{'family':<14} {'model':<10} {'F1':>6} {'Recall':>7} {'Contain':>8}")
print("-" * 50)
for fam in FAMS:
    v = by_fam[fam]
    rows = [("oracle", oracle_scores(v)), ("graph_v5", mem_scores(graph, v)), ("mamba", mem_scores(mamba, v))]
    for name, (f1, rec, con) in rows:
        print(f"{fam:<14} {name:<10} {f1:>6.1f} {rec:>7.1f} {con:>8.1f}", flush=True)
    print("-" * 50)
