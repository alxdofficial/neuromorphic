#!/usr/bin/env python3
"""Composite val-set evaluation across four modes.

Modes compared (all on the same chunks for paired comparison):
  - v2          : trained ckpt with full memory side-car (write 8 passages, read at QA)
  - v2_no_mem   : trained ckpt, but manifold edge_state zeroed before QA read
                  (ablation: is the manifold actually doing work, or only the adapter?)
  - vanilla_nc  : frozen Llama-3.2-1B, prompt = question only, no passage context
  - vanilla_fc  : frozen Llama-3.2-1B, prompt = all 8 passages + question (full ctx)

Metric: mean NLL per answer token. Lower is better. Per-task breakdown reported.

Usage:
  python scripts/eval/eval_composite_v1.py \\
    --ckpt outputs/wave1_v2/ckpt.10000.pt \\
    --val-dir data/wave1/composite_v1/val \\
    --num-chunks 800 --batch-size 8 \\
    --output outputs/wave1_v2/eval_compare.json
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.wave1.common.sampler import CompositeRetrievalAdapter
from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2.trainer import Phase1RetrievalTrainerV2


NEWLINE = 198
BOS = 128000
EOS = 128001


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--num-chunks", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", nargs="+",
                    default=["v2", "v2_no_mem", "vanilla_nc", "vanilla_fc"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def answer_nll_llama(llama, full_ids, attn_mask, answer_starts, answer_lens):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = llama(input_ids=full_ids, attention_mask=attn_mask)
        logits = out.logits
    nlls = []
    for b in range(full_ids.shape[0]):
        s, n = answer_starts[b], answer_lens[b]
        if n == 0:
            nlls.append(float("nan"))
            continue
        pred = logits[b, s - 1 : s - 1 + n].float()
        tgt = full_ids[b, s : s + n]
        nll = F.cross_entropy(pred, tgt, reduction="mean").item()
        nlls.append(nll)
    return nlls


def build_vanilla_batch(chunks, mode, device, max_len=4096):
    rows, a_starts, a_lens, families = [], [], [], []
    for c in chunks:
        if mode == "vanilla_nc":
            ctx = [BOS] + list(c["question_token_ids"]) + [NEWLINE]
        elif mode == "vanilla_fc":
            ctx = [BOS]
            for p in c["fact_passages_token_ids"]:
                ctx.extend(p)
                ctx.append(NEWLINE)
            ctx.extend(c["question_token_ids"])
            ctx.append(NEWLINE)
        else:
            raise ValueError(mode)
        ans = list(c["answer_token_ids"])
        ids = ctx + ans
        if len(ids) > max_len:
            keep_from = len(ids) - max_len
            ctx_kept = ctx[keep_from:]
            ids = ctx_kept + ans
        rows.append(ids)
        a_starts.append(len(ids) - len(ans))
        a_lens.append(len(ans))
        families.append(c["metadata"]["target_entity_class"])

    L = max(len(r) for r in rows)
    full = torch.zeros((len(rows), L), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), L), dtype=torch.bool, device=device)
    for i, r in enumerate(rows):
        full[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        mask[i, :len(r)] = True
    return full, mask, a_starts, a_lens, families


def aggregate(results, task_families, nlls, answer_lens):
    for fam, nll, n in zip(task_families, nlls, answer_lens):
        if not (nll == nll):
            continue
        r = results.setdefault(fam, {"sum_nll_x_tokens": 0.0, "n_tokens": 0, "n_samples": 0})
        r["sum_nll_x_tokens"] += nll * n
        r["n_tokens"] += n
        r["n_samples"] += 1


def finalize(results):
    out = {}
    total_nllxt, total_t = 0.0, 0
    for fam, r in results.items():
        if r["n_tokens"] == 0: continue
        out[fam] = {
            "nll_per_token": r["sum_nll_x_tokens"] / r["n_tokens"],
            "n_samples": r["n_samples"], "n_tokens": r["n_tokens"],
        }
        total_nllxt += r["sum_nll_x_tokens"]; total_t += r["n_tokens"]
    out["__overall__"] = {
        "nll_per_token": total_nllxt / max(total_t, 1),
        "n_samples": sum(r["n_samples"] for r in results.values()),
        "n_tokens": total_t,
    }
    return out


def eval_v2(trainer, batch, reset_memory):
    cfg = trainer.model.cfg
    M = len(batch)
    device = next(trainer.model.parameters()).device

    tensors = trainer._build_tensors(batch, device)
    passages = tensors["passages"]
    qa = tensors["qa"]
    answer_mask = tensors["answer_mask"]
    question_ids = tensors["question_ids"]
    question_lens = tensors["question_lens"]

    snap_eval = trainer.model.manifold.snapshot_edge_state()
    try:
        q_hiddens = trainer._compute_question_hiddens(question_ids, question_lens)
        q_mask_read = (question_ids != trainer.pad_token_id)

        if reset_memory:
            # No-memory baseline: empty manifold + skip passage writes entirely.
            # This isolates the "would the trained adapter do anything useful
            # WITHOUT the memory side-car?" question. Same model params, but
            # the read trajectory walks a fully empty graph.
            trainer.model.manifold.reset_edge_memory()
            prev = None
        else:
            prev = None
            for i in range(cfg.n_facts_per_chunk):
                p_ids = passages[:, i]
                p_mask = (p_ids != trainer.pad_token_id)
                out = trainer.model.forward_window(
                    lm_input_ids=p_ids, attention_mask=p_mask,
                    prev_window_hiddens=prev,
                    read_conditioning_hiddens=q_hiddens,
                    read_conditioning_mask=q_mask_read,
                    hard_routing=False, write_mode="passage",
                )
                prev = out.get("current_hiddens")

        qa_mask = (qa != trainer.pad_token_id)
        out_qa = trainer.model.forward_window(
            lm_input_ids=qa, attention_mask=qa_mask,
            prev_window_hiddens=prev,
            read_conditioning_hiddens=q_hiddens,
            read_conditioning_mask=q_mask_read,
            hard_routing=False, write_mode="qa",
        )
        logits = out_qa["logits"]

        nlls, alens, fams = [], [], []
        for m in range(M):
            mask_row = answer_mask[m]
            if mask_row.sum() == 0:
                nlls.append(float("nan")); alens.append(0)
                fams.append(batch[m]["metadata"]["target_entity_class"])
                continue
            tgt = qa[m, 1:]
            ans_msk = mask_row[1:]
            pred = logits[m, :-1].float()
            tok_nll = F.cross_entropy(pred[ans_msk], tgt[ans_msk], reduction="mean").item()
            nlls.append(tok_nll)
            alens.append(int(ans_msk.sum().item()))
            fams.append(batch[m]["metadata"]["target_entity_class"])
    finally:
        trainer.model.manifold.restore_edge_state(snap_eval)

    return nlls, alens, fams


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    val_sampler = CompositeRetrievalAdapter(
        args.val_dir / "passages.jsonl",
        args.val_dir / "questions.jsonl",
        chunk_size=8, seed=args.seed,
    )
    print(f"Val sampler: {len(val_sampler.facts)} questions", flush=True)

    print("Loading model...", flush=True)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    model = IntegratedLMV2(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.train(mode=False)
    trainer = Phase1RetrievalTrainerV2(
        model=model, optimizer=None, pad_token_id=EOS,
    )
    print(f"Loaded {args.ckpt.name} (step={ckpt.get('step', '?')})", flush=True)

    print(f"Sampling {args.num_chunks} val chunks (paired across modes)...", flush=True)
    all_chunks = val_sampler.sample_batch(args.num_chunks)

    results = {}
    for mode in args.modes:
        print(f"\n=== Running mode: {mode} ===", flush=True)
        t_start = time.time()
        mode_results = {}
        for b in range(0, len(all_chunks), args.batch_size):
            batch = all_chunks[b : b + args.batch_size]
            if not batch: break
            if mode == "v2":
                nlls, alens, fams = eval_v2(trainer, batch, reset_memory=False)
            elif mode == "v2_no_mem":
                nlls, alens, fams = eval_v2(trainer, batch, reset_memory=True)
            elif mode in ("vanilla_nc", "vanilla_fc"):
                full, mask, a_starts, a_lens, fams = build_vanilla_batch(
                    batch, mode, args.device,
                )
                nlls = answer_nll_llama(model.llama, full, mask, a_starts, a_lens)
                alens = a_lens
            else:
                raise ValueError(mode)
            aggregate(mode_results, fams, nlls, alens)
            done = b + len(batch)
            if done % 200 == 0:
                print(f"  {done}/{len(all_chunks)}  ({time.time()-t_start:.1f}s)", flush=True)

        results[mode] = finalize(mode_results)
        overall = results[mode]["__overall__"]
        print(f"  {mode} overall NLL/tok: {overall['nll_per_token']:.4f} "
              f"({overall['n_samples']} samples, {overall['n_tokens']} tokens, "
              f"{time.time() - t_start:.1f}s)", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("COMPARISON: NLL per answer token (lower = better)", flush=True)
    print("=" * 100, flush=True)
    all_tasks = sorted(set().union(*[set(r.keys()) for r in results.values()]) - {"__overall__"})
    modes = list(results.keys())
    header = f"{'task':<42}" + "".join(f"{m:>13}" for m in modes)
    print(header)
    print("-" * len(header))
    for task in all_tasks + ["__overall__"]:
        row = f"{task:<42}"
        for m in modes:
            v = results[m].get(task, {}).get("nll_per_token")
            row += f"{v:>13.4f}" if v is not None else f"{'—':>13}"
        print(row)

    payload = {
        "ckpt": str(args.ckpt),
        "val_dir": str(args.val_dir),
        "num_chunks": args.num_chunks,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "modes": args.modes,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
