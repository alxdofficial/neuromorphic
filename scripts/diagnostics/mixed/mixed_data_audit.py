"""Comprehensive data audit for the mixed 4-task benchmark (mae / babi / continuation / condrecon_bio).

Draws real batches from the EXACT loaders train.py uses (val via make_mixed_val_sets; train via the
same factory fns with split=train) and checks, per task:

  A. STRUCTURE   shapes/dtypes; mask⇔pad consistency (context/answer/content); fill fractions;
                 answer + content-mask length stats.
  B. SAMPLES     2 decoded examples (context head/tail, question, answer) for eyeball review.
  C. INVARIANTS  task-specific semantic checks:
                   babi          — answer in context (per task family), answer NOT leaked in question
                   continuation  — answer NOT in context (no leak); tail+head contiguity printout
                   condrecon_bio — answer VERBATIM in context; queried entity present in context
                   mae           — answer==context alignment (MAE reconstructs the passage)
                 + duplicate context rows within the val set.
  D. FIREWALLS   train/val disjointness: context-row hash overlap (mae/continuation/babi);
                 condrecon entity-world overlap (val world must be a different entity world).

Usage: python scripts/diagnostics/mixed/mixed_data_audit.py
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from src.memory.config import ReprConfig
from src.memory.training import make_mixed_val_sets
from src.memory.data.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS
from src.memory.data.mae import make_long_passage_mae_dataloader
from src.memory.data.babi import make_babi_dataloader
from src.memory.data.continuation import make_continuation_dataloader
from src.memory.data.bio import make_conditioned_reconstruction_bio_dataloader

TASKS = ["mae", "babi", "continuation", "condrecon_bio"]
N_BATCHES = 4                    # 4×8 = 32 examples audited per task/split
CTX, M, PLEN = 1024, 32, 64
SRC_TOK = "meta-llama/Llama-3.2-1B"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def note(task, name, ok, detail=""):
    tag = PASS if ok is True else (FAIL if ok is False else WARN)
    results.append((task, name, tag, detail))
    print(f"    [{tag}] {name}" + (f" — {detail}" if detail else ""))


def rowhash(ids_row, mask_row):
    return hashlib.md5(ids_row[mask_row.bool()].numpy().tobytes()).hexdigest()


def struct_audit(task, batches, tok, pad):
    b = batches[0]
    print(f"  shapes: ctx{tuple(b.context_ids.shape)} q{tuple(b.question_ids.shape)} "
          f"a{tuple(b.answer_ids.shape)} | families={sorted(set(b.task_family))[:4]}")
    ok_pad_c = all(bool((bt.context_ids[~bt.context_mask.bool()] == pad).all()) for bt in batches)
    note(task, "context pad⇔mask", ok_pad_c)
    ok_pad_a = all(bool((bt.answer_ids[~bt.answer_mask.bool()] == pad).all()) for bt in batches)
    note(task, "answer pad⇔mask", ok_pad_a)
    ok_cm = all(bool((bt.answer_content_mask.bool() & ~bt.answer_mask.bool()).sum() == 0) for bt in batches)
    note(task, "content_mask ⊆ answer_mask", ok_cm)
    fills = torch.cat([bt.context_mask.float().mean(1) for bt in batches])
    alens = torch.cat([bt.answer_mask.sum(1).float() for bt in batches])
    clens = torch.cat([bt.answer_content_mask.sum(1).float() for bt in batches])
    print(f"    ctx fill: mean {fills.mean():.3f}  min {fills.min():.3f} | answer len: mean {alens.mean():.1f} "
          f"[{int(alens.min())},{int(alens.max())}] | content len: mean {clens.mean():.1f}")
    note(task, "context mostly filled", bool(fills.mean() > 0.9), f"mean fill {fills.mean():.3f}")
    note(task, "content mask nonzero", bool(clens.min() > 0), f"min {int(clens.min())}")


def show_samples(task, batches, tok, n=2):
    b = batches[0]
    for i in range(n):
        ctx = tok.decode(b.context_ids[i][b.context_mask[i].bool()])
        q = tok.decode(b.question_ids[i][b.question_mask[i].bool()]) if b.question_mask[i].any() else "(empty)"
        a = tok.decode(b.answer_ids[i][b.answer_mask[i].bool()])
        print(f"    ── sample {i} ({b.task_family[i]}/{b.question_type[i]}) ──")
        print(f"    CTX[:300]: {ctx[:300]!r}")
        print(f"    CTX[-200:]: {ctx[-200:]!r}")
        print(f"    Q: {q[:200]!r}")
        print(f"    A: {a[:300]!r}")


def invariants(task, batches, tok):
    in_ctx, in_q, per_fam = [], [], defaultdict(list)
    for b in batches:
        for i in range(b.context_ids.shape[0]):
            ctx = tok.decode(b.context_ids[i][b.context_mask[i].bool()])
            q = tok.decode(b.question_ids[i][b.question_mask[i].bool()]) if b.question_mask[i].any() else ""
            a = tok.decode(b.answer_ids[i][b.answer_mask[i].bool()]).strip()
            a_core = a.split("\n")[0].strip()
            if task == "continuation" and len(a_core) < 20:   # 1-char/short first lines ('.', '-') match anywhere
                a_core = a.strip()[:80]                        # use a real span for the leak check instead
            hit_ctx = a_core.lower() in ctx.lower() if a_core else False
            in_ctx.append(hit_ctx); in_q.append(bool(a_core and a_core.lower() in q.lower()))
            per_fam[b.task_family[i]].append(hit_ctx)
    pc, pq = 100 * sum(in_ctx) / len(in_ctx), 100 * sum(in_q) / len(in_q)
    if task == "babi":
        note(task, "answer in context", pc > 60,
             f"{pc:.0f}% overall | per family: " + " ".join(f"{f}:{100*sum(v)/len(v):.0f}%" for f, v in sorted(per_fam.items())))
        note(task, "answer NOT leaked in question", pq == 0, f"{pq:.0f}% leak")
    elif task == "continuation":
        note(task, "answer NOT in context (no leak)", pc == 0, f"{pc:.0f}% found in ctx")
        b = batches[0]
        tail = tok.decode(b.context_ids[0][b.context_mask[0].bool()])[-120:]
        head = tok.decode(b.answer_ids[0][b.answer_mask[0].bool()])[:120]
        print(f"    contiguity eyeball: ...{tail!r} ++ {head!r}")
    elif task == "condrecon_bio":
        note(task, "answer VERBATIM in context", pc >= 97, f"{pc:.0f}%")
        note(task, "answer NOT in question", pq == 0, f"{pq:.0f}% leak")
        qents = []
        for b in batches:
            for i in range(b.context_ids.shape[0]):
                q = tok.decode(b.question_ids[i][b.question_mask[i].bool()])
                ctx = tok.decode(b.context_ids[i][b.context_mask[i].bool()])
                words = [w for w in q.replace("?", " ").split() if w[:1].isupper()]
                qents.append(bool(words) and all(w in ctx for w in words[:2]))
        note(task, "queried entity present in context", 100 * sum(qents) / len(qents) >= 97,
             f"{100*sum(qents)/len(qents):.0f}%")
    elif task == "mae":
        b = batches[0]
        same = all(bool(torch.equal(b.context_ids[i][b.context_mask[i].bool()],
                                    b.answer_ids[i][b.answer_mask[i].bool()])) for i in range(4))
        note(task, "answer == context (MAE reconstruction target)", same
             if same else None, "exact" if same else "answer≠context — check objective wiring")
    # duplicates within the val set
    hs = [rowhash(b.context_ids[i], b.context_mask[i]) for b in batches for i in range(b.context_ids.shape[0])]
    dupes = sum(c - 1 for c in Counter(hs).values() if c > 1)
    note(task, "no duplicate contexts in val draw", dupes == 0, f"{dupes} dupes / {len(hs)}")


def firewall(task, val_batches, train_batches, tok):
    vh = {rowhash(b.context_ids[i], b.context_mask[i]) for b in val_batches for i in range(b.context_ids.shape[0])}
    th = {rowhash(b.context_ids[i], b.context_mask[i]) for b in train_batches for i in range(b.context_ids.shape[0])}
    inter = len(vh & th)
    note(task, "train∩val context overlap", inter == 0, f"{inter} shared rows")
    if task == "condrecon_bio":
        def ents(batches):
            # FULL entity name = the question prefix before '=' (drop parentheticals) — single capitalized
            # words would false-alarm on the generator's shared template vocabulary (Academy, Coastal, ...)
            s = set()
            for b in batches:
                for i in range(b.context_ids.shape[0]):
                    q = tok.decode(b.question_ids[i][b.question_mask[i].bool()])
                    s.add(q.split("=")[0].split("(")[0].strip())
            return s
        ev, et = ents(val_batches), ents(train_batches)
        inter_e = ev & et
        note(task, "entity-world firewall (val∩train FULL names)", len(inter_e) == 0,
             f"{len(inter_e)} shared: {sorted(inter_e)[:6]}")


def main():
    cfg = ReprConfig()
    cfg.llama_model = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    cfg.batch_size = 8

    print("building VAL sets (exact make_mixed_val_sets call)...")
    vs = make_mixed_val_sets(TASKS, tok, cfg, N_BATCHES, ctx_len=CTX, m_slots=M,
                             mae_src_tok=SRC_TOK, babi_tasks=BABI_DEFAULT_TASKS, predict_len=PLEN)
    print("building TRAIN loaders (split=train, run seed 42)...")
    tl = {
        "mae": make_long_passage_mae_dataloader(tok, batch_size=8, src_tokenizer_name=SRC_TOK, split="train",
                                                ctx_len=CTX, m_slots=M, seed=42, pad_token_id=pad, num_workers=0),
        "babi": make_babi_dataloader(tok, context_len=CTX, batch_size=8, split="train", seed=42,
                                     pad_token_id=pad, tasks=BABI_DEFAULT_TASKS, num_workers=0),
        "continuation": make_continuation_dataloader(tok, batch_size=8, compress_len=CTX, predict_len=PLEN,
                                                     split="train", seed=42, pad_token_id=pad,
                                                     objective="continuation", src_tokenizer_name=SRC_TOK, num_workers=0),
        "condrecon_bio": make_conditioned_reconstruction_bio_dataloader(
            tok, context_len=CTX, batch_size=8, n_pairs=24, n_query=1, n_facts=3, split="train",
            world_seed=0, stream_seed=42, pad_token_id=pad, num_workers=0),
    }

    for task in TASKS:
        print(f"\n{'='*90}\n{task.upper()}\n{'='*90}")
        vb = vs[task][:N_BATCHES]
        it = iter(tl[task]); tb = [next(it) for _ in range(N_BATCHES)]
        struct_audit(task, vb, tok, pad)
        show_samples(task, vb, tok)
        invariants(task, vb, tok)
        firewall(task, vb, tb, tok)

    print(f"\n{'='*90}\nSUMMARY\n{'='*90}")
    fails = [r for r in results if r[2] == FAIL]
    warns = [r for r in results if r[2] == WARN]
    print(f"{len(results)} checks: {len(results)-len(fails)-len(warns)} PASS, {len(warns)} WARN, {len(fails)} FAIL")
    for t, n, tag, d in fails + warns:
        print(f"  [{tag}] {t}: {n} — {d}")


if __name__ == "__main__":
    main()
