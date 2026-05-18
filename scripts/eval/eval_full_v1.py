#!/usr/bin/env python3
"""Comprehensive eval for v1 trajectory-memory architecture (V1.5 or earlier).

Mirrors scripts/eval/eval_full.py from v2 but uses v1's IntegratedLM +
Phase1RetrievalTrainer. Computes paired with-mem vs no-mem NLL on
composite_v1 val for apples-to-apples comparison with V2.13.

Usage:
  python scripts/eval/eval_full_v1.py \\
    --ckpt outputs/v1.5/ckpt.final.pt \\
    --val-dir data/wave1/composite_v1/val \\
    --num-chunks 800 --batch-size 8 \\
    --output outputs/v1.5/eval_full.json \\
    --markdown outputs/v1.5/eval_full.md \\
    --version-label V1.5
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.wave1.common.sampler import CompositeRetrievalAdapter
from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.phase1_retrieval import Phase1RetrievalTrainer


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
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--version-label", type=str, default="V1.x")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def answer_nll_llama(model, full_ids, attn_mask, answer_starts, answer_lens):
    # Find the mem_inject layer in the wrapped Llama and install a zero
    # memory_fn so the assertion against silent bypass doesn't fire.
    mi = None
    for m in model.modules():
        if m.__class__.__name__ == "MemInjectLayer":
            mi = m
            break
    saved_fn = None
    if mi is not None:
        saved_fn = getattr(mi, "memory_fn", None)
        d_mem = mi.d_mem
        def zero_readout(h_mem):
            return torch.zeros(h_mem.shape[0], h_mem.shape[1], d_mem,
                               device=h_mem.device, dtype=h_mem.dtype)
        mi.memory_fn = zero_readout
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.llama(input_ids=full_ids, attention_mask=attn_mask)
            logits = out.logits
    finally:
        if mi is not None:
            mi.memory_fn = saved_fn
    res = []
    for b in range(full_ids.shape[0]):
        s, n = answer_starts[b], answer_lens[b]
        if n == 0:
            res.append((float("nan"), float("nan"))); continue
        pred = logits[b, s - 1 : s - 1 + n].float()
        tgt = full_ids[b, s : s + n]
        nll = F.cross_entropy(pred, tgt, reduction="mean").item()
        first_nll = F.cross_entropy(pred[0:1], tgt[0:1], reduction="mean").item()
        res.append((nll, first_nll))
    return res


def build_vanilla_batch(chunks, mode, device, max_len=4096):
    rows, a_starts, a_lens, families = [], [], [], []
    for c in chunks:
        if mode == "vanilla_nc":
            ctx = [BOS] + list(c["question_token_ids"]) + [NEWLINE]
        elif mode == "vanilla_fc":
            ctx = [BOS]
            for p in c["fact_passages_token_ids"]:
                ctx.extend(p); ctx.append(NEWLINE)
            ctx.extend(c["question_token_ids"]); ctx.append(NEWLINE)
        else:
            raise ValueError(mode)
        ans = list(c["answer_token_ids"])
        ids = ctx + ans
        if len(ids) > max_len:
            ids = ctx[len(ids) - max_len:] + ans
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


@torch.no_grad()
def eval_v1(trainer, batch, no_mem, zero_readout=False):
    """Run v1 on a batch. Returns per-row (content_nll, first_content_nll, family).
    If no_mem=True: skip the 8 passage writes; QA reads from fresh (state_init) states.
    If zero_readout=True: also force mem_inject.memory_fn to return zeros — true
       no-memory baseline that bypasses the bridge entirely.
    """
    cfg = trainer.model.cfg
    M = len(batch)
    device = next(trainer.model.parameters()).device

    passages, qa, answer_mask, full_answer_mask, question_ids, question_lengths = (
        trainer._build_tensors(batch, device)
    )

    # Fresh per-cell state at start of chunk
    prev_state = trainer.model.manifold.reset_states(batch_size=M)
    prev_hiddens = None

    if not no_mem:
        # Process 8 passage writes — state evolves window by window
        for i in range(cfg.n_facts_per_chunk if hasattr(cfg, "n_facts_per_chunk") else 8):
            out = trainer.model.forward_window(
                lm_input_ids=passages[:, i, :],
                prev_window_hiddens=prev_hiddens,
                prev_states=prev_state,
                target_mask=None,
                hard_routing=False,
                use_kv_cache=False,
                write_only_grad=False,
            )
            prev_state = out["new_states"]
            prev_hiddens = out.get("current_hiddens")

    # Compute question hiddens (memory disabled) for question-conditioned read
    q_hiddens = trainer._compute_question_hiddens(question_ids, question_lengths) \
        if hasattr(trainer, "_compute_question_hiddens") else None

    # QA window — read from final state (populated if memory active, fresh if not)
    fw_kwargs = dict(
        lm_input_ids=qa,
        prev_window_hiddens=prev_hiddens,
        prev_states=prev_state,
        target_mask=None,
        hard_routing=False,
        use_kv_cache=False,
        write_only_grad=False,
    )
    # V1.5 added read_conditioning_hiddens; older v1 didn't
    if q_hiddens is not None:
        try:
            out_qa = trainer.model.forward_window(
                read_conditioning_hiddens=q_hiddens,
                read_conditioning_mask=(question_ids != trainer.pad_token_id),
                **fw_kwargs,
            )
        except TypeError:
            out_qa = trainer.model.forward_window(**fw_kwargs)
    else:
        out_qa = trainer.model.forward_window(**fw_kwargs)

    logits = out_qa["logits"]
    read_visited = out_qa.get("read_visited_ids")

    # zero_readout post-hoc: if needed, re-run with memory_fn forced to zero
    if zero_readout:
        mi = None
        for m in trainer.model.modules():
            if m.__class__.__name__ == "MemInjectLayer":
                mi = m; break
        saved_fn = mi.memory_fn if mi else None
        d_mem = mi.d_mem
        def zero_fn(h_mem):
            return torch.zeros(h_mem.shape[0], h_mem.shape[1], d_mem,
                               device=h_mem.device, dtype=h_mem.dtype)
        mi.memory_fn = zero_fn
        try:
            out_qa = trainer.model.forward_window(**fw_kwargs)
            logits = out_qa["logits"]
        finally:
            mi.memory_fn = saved_fn

    res = []
    for m in range(M):
        mask_row = answer_mask[m]  # content-only in V1.5; full-answer in older v1
        fam = batch[m]["metadata"]["target_entity_class"]
        if mask_row.sum() == 0:
            res.append((float("nan"), float("nan"), fam)); continue
        tgt = qa[m, 1:]
        ans_msk = mask_row[1:]
        pred = logits[m, :-1].float()
        ans_logits = pred[ans_msk]
        ans_tgts = tgt[ans_msk]
        tok_nll = F.cross_entropy(ans_logits, ans_tgts, reduction="mean").item()
        first_nll = F.cross_entropy(ans_logits[0:1], ans_tgts[0:1], reduction="mean").item()
        res.append((tok_nll, first_nll, fam))

    return res, read_visited


def aggregate(samples):
    by_task = defaultdict(lambda: {"sum_nll": 0.0, "sum_first": 0.0, "n": 0})
    for nll, first, fam in samples:
        if not (nll == nll): continue
        d = by_task[fam]
        d["sum_nll"] += nll
        d["sum_first"] += first if first == first else 0
        d["n"] += 1
    out, total_n, total_sum, total_first = {}, 0, 0.0, 0.0
    for fam, d in by_task.items():
        if d["n"] == 0: continue
        out[fam] = {
            "mean_nll": d["sum_nll"] / d["n"],
            "mean_first_nll": d["sum_first"] / d["n"],
            "n_samples": d["n"],
        }
        total_n += d["n"]; total_sum += d["sum_nll"]; total_first += d["sum_first"]
    out["__overall__"] = {
        "mean_nll": total_sum / max(total_n, 1),
        "mean_first_nll": total_first / max(total_n, 1),
        "n_samples": total_n,
    }
    return out


def cross_question_divergence(read_visited_list):
    if not read_visited_list:
        return None
    import random
    random.seed(0)
    all_visited = []
    for rv in read_visited_list:
        if rv is None: continue
        for b in range(rv.shape[0]):
            cells = set(rv[b].flatten().tolist())
            all_visited.append(cells)
    if len(all_visited) < 2: return None
    n = len(all_visited)
    pairs = [(random.randint(0, n - 1), random.randint(0, n - 1)) for _ in range(200)]
    jaccards = []
    for i, j in pairs:
        if i == j: continue
        a, b = all_visited[i], all_visited[j]
        if not a or not b: continue
        inter = len(a & b); union = len(a | b)
        if union == 0: continue
        jaccards.append(inter / union)
    if not jaccards: return None
    return {"mean_jaccard": sum(jaccards) / len(jaccards), "n_pairs": len(jaccards)}


def emit_markdown(results, version_label, ckpt_path, args):
    md = []
    md.append(f"## {version_label} — full eval ({ckpt_path.name})\n")
    md.append(f"_Sampled {args.num_chunks} val chunks (paired across modes), BS={args.batch_size}_\n")
    md.append("\n### Memory contribution probes\n")
    v1 = results["modes"]["v1"]["__overall__"]
    no_mem = results["modes"]["v1_no_mem"]["__overall__"]
    nc = results["modes"]["vanilla_nc"]["__overall__"]
    fc = results["modes"]["vanilla_fc"]["__overall__"]
    md.append("| Probe | Mean NLL/tok (content) | First-token NLL |")
    md.append("|---|---:|---:|")
    md.append(f"| v1 (memory active) | {v1['mean_nll']:.4f} | {v1['mean_first_nll']:.4f} |")
    md.append(f"| v1_no_mem (skip writes, fresh state) | {no_mem['mean_nll']:.4f} | {no_mem['mean_first_nll']:.4f} |")
    md.append(f"| vanilla Llama no-ctx | {nc['mean_nll']:.4f} | {nc['mean_first_nll']:.4f} |")
    md.append(f"| vanilla Llama full-ctx | {fc['mean_nll']:.4f} | {fc['mean_first_nll']:.4f} |")
    md.append("")
    md.append(f"- **Memory contribution: v1 − v1_no_mem = {no_mem['mean_nll'] - v1['mean_nll']:+.4f} nat** (negative = memory helps)")
    md.append(f"- **Gap to vanilla no-ctx: v1 − vanilla_nc = {v1['mean_nll'] - nc['mean_nll']:+.4f} nat**")
    md.append(f"- **Gap to vanilla full-ctx: v1 − vanilla_fc = {v1['mean_nll'] - fc['mean_nll']:+.4f} nat**")
    md.append(f"- First-token-only memory contribution: {no_mem['mean_first_nll'] - v1['mean_first_nll']:+.4f} nat")
    md.append("")
    if results.get("cross_q_divergence"):
        cq = results["cross_q_divergence"]
        md.append(f"### Cross-question read divergence: Jaccard = {cq['mean_jaccard']:.3f} (n={cq['n_pairs']} pairs; lower = more divergent)\n")
    md.append("### Per-task NLL/tok (content tokens only)\n")
    md.append("| Task | v1 | v1_no_mem | vanilla_nc | vanilla_fc |")
    md.append("|---|---:|---:|---:|---:|")
    all_tasks = sorted(set().union(*[set(m.keys()) for m in results["modes"].values()]) - {"__overall__"})
    for task in all_tasks:
        row = [task]
        for mode in ["v1", "v1_no_mem", "v1_zero_readout", "vanilla_nc", "vanilla_fc"]:
            v = results["modes"][mode].get(task, {}).get("mean_nll")
            row.append(f"{v:.3f}" if v is not None else "—")
        md.append("| " + " | ".join(row) + " |")
    overall_row = ["**__overall__**"]
    for mode in ["v1", "v1_no_mem", "v1_zero_readout", "vanilla_nc", "vanilla_fc"]:
        v = results["modes"][mode]["__overall__"]["mean_nll"]
        overall_row.append(f"**{v:.3f}**")
    md.append("| " + " | ".join(overall_row) + " |")
    md.append("")
    return "\n".join(md)


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
    if isinstance(cfg, dict):
        cfg = TrajMemConfig(**cfg)
    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.train(mode=False)
    trainer = Phase1RetrievalTrainer(model=model, optimizer=None, pad_token_id=EOS)
    print(f"Loaded {args.ckpt.name} (step={ckpt.get('step', '?')})", flush=True)

    print(f"Sampling {args.num_chunks} val chunks (paired)...", flush=True)
    all_chunks = val_sampler.sample_batch(args.num_chunks)

    modes_results = {}
    read_visited_list = []

    for mode in ["v1", "v1_no_mem", "v1_zero_readout", "vanilla_nc", "vanilla_fc"]:
        print(f"\n=== Mode: {mode} ===", flush=True)
        t_start = time.time()
        samples = []
        for b in range(0, len(all_chunks), args.batch_size):
            batch = all_chunks[b : b + args.batch_size]
            if not batch: break
            if mode == "v1":
                res, rv = eval_v1(trainer, batch, no_mem=False)
                if rv is not None and b < args.batch_size * 30:
                    read_visited_list.append(rv.cpu())
                samples.extend(res)
            elif mode == "v1_no_mem":
                res, _ = eval_v1(trainer, batch, no_mem=True)
                samples.extend(res)
            elif mode == "v1_zero_readout":
                # Run with writes happening + manifold populated, but force the
                # memory_fn output to zero. Decouples "memory is harmful as a
                # readout" from "memory_fn is producing useful signal".
                res, _ = eval_v1(trainer, batch, no_mem=False, zero_readout=True)
                samples.extend(res)
            else:
                full, mask, a_starts, a_lens, fams = build_vanilla_batch(batch, mode, args.device)
                nlls = answer_nll_llama(model, full, mask, a_starts, a_lens)
                for (nll, fnll), fam in zip(nlls, fams):
                    samples.append((nll, fnll, fam))
            done = b + len(batch)
            if done % 200 == 0:
                print(f"  {done}/{len(all_chunks)}  ({time.time()-t_start:.1f}s)", flush=True)
        agg = aggregate(samples)
        modes_results[mode] = agg
        overall = agg["__overall__"]
        print(f"  {mode} overall NLL/tok: {overall['mean_nll']:.4f}  "
              f"first_tok: {overall['mean_first_nll']:.4f}  "
              f"({overall['n_samples']} samples, {time.time()-t_start:.1f}s)",
              flush=True)

    cross_q = cross_question_divergence(read_visited_list)
    if cross_q:
        print(f"\nCross-question Jaccard: {cross_q['mean_jaccard']:.3f}", flush=True)

    payload = {
        "ckpt": str(args.ckpt),
        "version_label": args.version_label,
        "num_chunks": args.num_chunks,
        "batch_size": args.batch_size,
        "modes": modes_results,
        "cross_q_divergence": cross_q,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {args.output}", flush=True)

    if args.markdown:
        args.markdown.write_text(emit_markdown(payload, args.version_label, args.ckpt, args))
        print(f"Markdown → {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
