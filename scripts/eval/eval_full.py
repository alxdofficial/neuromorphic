#!/usr/bin/env python3
"""eval_full.py — comprehensive diagnostic eval for a single ckpt.

Emits the full Section 1 metric set from docs/baseline_numbers.md (headline
NLL probes + routing/memory diagnostics; skips generation EM + decode probe
for cost reasons).

Modes:
    v2          — trained model with full memory side-car (writes 8 passages, reads at QA)
    v2_no_mem   — trained model BUT manifold reset to empty AND no passage writes
                  (true ablation — does memory contribute anything?)
    vanilla_nc  — frozen Llama-3.2-1B, prompt = question only (no passage context)
    vanilla_fc  — frozen Llama-3.2-1B, prompt = 8 passages + question (full context)

For each mode, computes:
    - Per-task answer NLL (mean per token)
    - First-token-only NLL (kills teacher-forced AR leak)
    - Overall NLL aggregated

For v2 mode only, also:
    - Memory readout norm + variance (hooks mem_inject.memory_fn)
    - Cross-question read divergence (Jaccard of read trajectories)
    - Lifetime cell utilization (% cells ever written / read)
    - Manifold state stats (edge counts, norms, ages, specificity)
    - Per-module gradient norms (read from final training JSONL)
    - All Section 1B-1H training-time metrics extracted from JSONL

Usage:
    python scripts/eval/eval_full.py \\
        --ckpt outputs/wave1_v2/ckpt.10000.pt \\
        --val-dir data/wave1/composite_v1/val \\
        --train-jsonl outputs/wave1_v2/train.jsonl \\
        --num-chunks 800 --batch-size 8 \\
        --output outputs/wave1_v2/eval_full.json \\
        --markdown outputs/wave1_v2/eval_full.md
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

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
    ap.add_argument("--train-jsonl", type=Path, default=None,
                    help="Optional. Final-step metrics will be extracted.")
    ap.add_argument("--num-chunks", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, default=None,
                    help="Optional: also emit a markdown summary.")
    ap.add_argument("--version-label", type=str, default="V2.13",
                    help="Label for the markdown report (e.g. V1.5, V2.13).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def answer_nll_llama(llama, full_ids, attn_mask, answer_starts, answer_lens):
    """Vanilla-Llama answer-NLL helper. Returns per-row (mean_nll, first_token_nll)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = llama(input_ids=full_ids, attention_mask=attn_mask)
        logits = out.logits
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


def eval_v2(trainer, batch, reset_memory, mem_log=None):
    """Run v2 on a batch. Returns per-row (mean_nll, first_token_nll, family).
    If reset_memory=True: empty manifold + skip writes (the no-mem ablation).
    If mem_log is a list, append per-sample memory readout norm.
    """
    cfg = trainer.model.cfg
    M = len(batch)
    device = next(trainer.model.parameters()).device

    tensors = trainer._build_tensors(batch, device)
    passages = tensors["passages"]; qa = tensors["qa"]
    answer_mask = tensors["answer_mask"]
    question_ids = tensors["question_ids"]; question_lens = tensors["question_lens"]

    snap_eval = trainer.model.manifold.snapshot_edge_state()
    try:
        q_hiddens = trainer._compute_question_hiddens(question_ids, question_lens)
        q_mask_read = (question_ids != trainer.pad_token_id)

        if reset_memory:
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

        # Optional: install a probe on mem_inject.memory_fn to track readout norms
        mem_inject = trainer.model._mem_inject_layer()
        if mem_log is not None and mem_inject.memory_fn is not None:
            orig_fn = mem_inject.memory_fn
            def logging_fn(h_mem):
                out = orig_fn(h_mem)
                mem_log.append({
                    "norm": out.float().norm().item(),
                    "mean_abs": out.float().abs().mean().item(),
                    "shape": tuple(out.shape),
                })
                return out
            mem_inject.memory_fn = logging_fn

        qa_mask = (qa != trainer.pad_token_id)
        out_qa = trainer.model.forward_window(
            lm_input_ids=qa, attention_mask=qa_mask,
            prev_window_hiddens=prev,
            read_conditioning_hiddens=q_hiddens,
            read_conditioning_mask=q_mask_read,
            hard_routing=False, write_mode="qa",
        )
        logits = out_qa["logits"]

        # Capture read trajectory for cross-question divergence
        read_visited = out_qa.get("read_visited_ids")

        res = []
        for m in range(M):
            mask_row = answer_mask[m]
            fam = batch[m]["metadata"]["target_entity_class"]
            if mask_row.sum() == 0:
                res.append((float("nan"), float("nan"), fam)); continue
            tgt = qa[m, 1:]; ans_msk = mask_row[1:]
            pred = logits[m, :-1].float()
            ans_logits = pred[ans_msk]
            ans_tgts = tgt[ans_msk]
            tok_nll = F.cross_entropy(ans_logits, ans_tgts, reduction="mean").item()
            first_nll = F.cross_entropy(ans_logits[0:1], ans_tgts[0:1], reduction="mean").item()
            res.append((tok_nll, first_nll, fam))
    finally:
        trainer.model.manifold.restore_edge_state(snap_eval)

    return res, read_visited


def aggregate(samples):
    """samples = list of (nll, first_nll, family). Returns per-task + overall."""
    by_task = defaultdict(lambda: {"sum_nll": 0.0, "sum_first_nll": 0.0, "n": 0})
    for nll, first_nll, fam in samples:
        if not (nll == nll): continue
        d = by_task[fam]
        d["sum_nll"] += nll
        d["sum_first_nll"] += first_nll if first_nll == first_nll else 0
        d["n"] += 1
    out = {}
    total_n = 0; total_sum = 0.0; total_first = 0.0
    for fam, d in by_task.items():
        if d["n"] == 0: continue
        out[fam] = {
            "mean_nll": d["sum_nll"] / d["n"],
            "mean_first_nll": d["sum_first_nll"] / d["n"],
            "n_samples": d["n"],
        }
        total_n += d["n"]; total_sum += d["sum_nll"]; total_first += d["sum_first_nll"]
    out["__overall__"] = {
        "mean_nll": total_sum / max(total_n, 1),
        "mean_first_nll": total_first / max(total_n, 1),
        "n_samples": total_n,
    }
    return out


def cross_question_divergence(read_visited_list):
    """read_visited_list = list of [B, J, K] tensors (one per batch).
    Returns mean pairwise Jaccard distance across all read trajectories.
    High divergence = trajectories differ per question (good).
    """
    if not read_visited_list:
        return None
    all_visited = []
    for rv in read_visited_list:
        # rv shape [B, J, K]  → flatten to per-sample sets
        if rv is None: continue
        for b in range(rv.shape[0]):
            cells = set(rv[b].flatten().tolist())
            all_visited.append(cells)
    if len(all_visited) < 2:
        return None
    # Sample up to 100 pairs for efficiency
    import random
    random.seed(0)
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
    return {
        "mean_jaccard": sum(jaccards) / len(jaccards),
        "n_pairs": len(jaccards),
    }


def manifold_state_snapshot(model):
    """Capture manifold edge stats."""
    m = model.manifold
    active = m.edge_active
    n_active = int(active.sum().item())
    return {
        "n_active_edges": n_active,
        "edge_active_fraction": n_active / float(active.numel()),
        "mean_edge_state_norm": float(m.edge_state[active].float().norm(dim=-1).mean().item()) if n_active > 0 else 0.0,
        "mean_visit_count": float(m.visit_count[active].float().mean().item()) if n_active > 0 else 0.0,
        "mean_specificity": float(m.specificity[active].float().mean().item()) if n_active > 0 else 0.0,
        "mean_edge_age": float((m.step_counter.float() - m.last_visit[active].float()).mean().item()) if n_active > 0 else 0.0,
    }


def extract_training_metrics(jsonl_path):
    """Read final train + last val record from training JSONL."""
    if not jsonl_path or not jsonl_path.exists():
        return None
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    train = [r for r in rows if r.get("phase") == "train"]
    val = [r for r in rows if r.get("phase") == "val"]
    if not train: return None
    last_train = train[-1]
    # Average over last 100 train records (MA)
    last_100 = train[-100:]
    keys = ["loss", "answer_loss", "answer_acc", "grad_norm",
            "aux_lb", "aux_z", "l_contrast_entry", "l_contrast_per_step",
            "rw_overlap_entry", "rw_overlap_hop", "rw_overlap_all", "rw_overlap_target",
            "w_unique_per_window", "r_unique_per_window",
            "w_unique_per_traj", "r_unique_per_traj",
            "read_entry_entropy", "write_entry_entropy", "entry_logits_max",
            "mean_edge_state_norm", "mean_edge_specificity", "mean_visit_count",
            "mean_edge_age", "mean_fan_out", "n_active_edges", "edge_active_fraction",
            "concept_ids_norm_mean", "concept_ids_norm_cv", "concept_ids_pairwise_cos",
            "grad_norm_concept_ids", "grad_norm_entry_proj", "grad_norm_lambda_edge",
            "grad_norm_mem_inject", "grad_norm_read", "grad_norm_write",
            "grad_norm_read_attn", "step_s",
    ]
    train_ma = {}
    for k in keys:
        vals = [r.get(k) for r in last_100 if r.get(k) is not None]
        if vals:
            train_ma[k] = sum(vals) / len(vals)
    # Spike count
    spike_count = sum(1 for r in last_100 if r.get("grad_norm", 0) > 50)
    train_ma["spike_count_per_1000"] = 10 * spike_count
    return {
        "final_step": last_train["step"],
        "train_ma_last100": train_ma,
        "best_val_loss": min((v.get("loss", 99) for v in val), default=None),
        "final_val_loss": val[-1].get("loss") if val else None,
        "final_val_acc": val[-1].get("answer_acc") if val else None,
        "n_val_records": len(val),
    }


def emit_markdown(results, version_label, ckpt_path, args):
    """Generate a markdown summary suitable for baseline_numbers.md."""
    md = []
    md.append(f"## {version_label} — full eval ({ckpt_path.name})\n")
    md.append(f"_Sampled {args.num_chunks} val chunks (paired across modes), BS={args.batch_size}_\n")

    # Memory-contribution headline
    md.append("\n### Memory contribution probes ⚠️\n")
    v2 = results["modes"]["v2"]["__overall__"]
    no_mem = results["modes"]["v2_no_mem"]["__overall__"]
    nc = results["modes"]["vanilla_nc"]["__overall__"]
    fc = results["modes"]["vanilla_fc"]["__overall__"]
    md.append("| Probe | Mean NLL/tok | First-token NLL |")
    md.append("|---|---:|---:|")
    md.append(f"| v2 (memory active) | {v2['mean_nll']:.4f} | {v2['mean_first_nll']:.4f} |")
    md.append(f"| v2_no_mem (empty manifold, no writes) | {no_mem['mean_nll']:.4f} | {no_mem['mean_first_nll']:.4f} |")
    md.append(f"| vanilla Llama no-ctx | {nc['mean_nll']:.4f} | {nc['mean_first_nll']:.4f} |")
    md.append(f"| vanilla Llama full-ctx | {fc['mean_nll']:.4f} | {fc['mean_first_nll']:.4f} |")
    md.append("")
    md.append(f"- **Memory contribution: v2 − v2_no_mem = {no_mem['mean_nll'] - v2['mean_nll']:+.4f} nat** (negative = memory helps)")
    md.append(f"- **Gap to vanilla no-ctx: v2 − vanilla_nc = {v2['mean_nll'] - nc['mean_nll']:+.4f} nat**")
    md.append(f"- **Gap to vanilla full-ctx: v2 − vanilla_fc = {v2['mean_nll'] - fc['mean_nll']:+.4f} nat**")
    md.append(f"- First-token-only memory contribution: {no_mem['mean_first_nll'] - v2['mean_first_nll']:+.4f} nat")
    md.append("")

    # Memory readout
    if results.get("memory_readout"):
        mr = results["memory_readout"]
        md.append("### Memory readout stats (mem_inject output)\n")
        md.append(f"- norm: mean={mr['mean_norm']:.3f}, stddev={mr['std_norm']:.3f}")
        md.append(f"- mean_abs: {mr['mean_abs']:.4f}")
        md.append(f"- across {mr['n_calls']} forward calls")
        md.append("")

    # Cross-question divergence
    if results.get("cross_q_divergence"):
        cq = results["cross_q_divergence"]
        md.append(f"### Cross-question read divergence: Jaccard = {cq['mean_jaccard']:.3f} (n={cq['n_pairs']} pairs; lower = more divergent)\n")

    # Manifold state
    if results.get("manifold_state"):
        ms = results["manifold_state"]
        md.append("### Manifold state\n")
        md.append(f"- n_active_edges: {ms['n_active_edges']:,d} ({100*ms['edge_active_fraction']:.1f}% of cap)")
        md.append(f"- mean_edge_state_norm: {ms['mean_edge_state_norm']:.2f}")
        md.append(f"- mean_visit_count: {ms['mean_visit_count']:.1f}")
        md.append(f"- mean_edge_age: {ms['mean_edge_age']:.0f} steps")
        md.append("")

    # Per-task break-down
    md.append("### Per-task NLL/tok\n")
    md.append("| Task | v2 | v2_no_mem | vanilla_nc | vanilla_fc |")
    md.append("|---|---:|---:|---:|---:|")
    all_tasks = sorted(set().union(*[set(m.keys()) for m in results["modes"].values()]) - {"__overall__"})
    for task in all_tasks:
        row = [task]
        for mode in ["v2", "v2_no_mem", "vanilla_nc", "vanilla_fc"]:
            v = results["modes"][mode].get(task, {}).get("mean_nll")
            row.append(f"{v:.3f}" if v is not None else "—")
        md.append("| " + " | ".join(row) + " |")
    overall_row = ["**__overall__**"]
    for mode in ["v2", "v2_no_mem", "vanilla_nc", "vanilla_fc"]:
        v = results["modes"][mode]["__overall__"]["mean_nll"]
        overall_row.append(f"**{v:.3f}**")
    md.append("| " + " | ".join(overall_row) + " |")
    md.append("")

    # Training metrics
    if results.get("training"):
        t = results["training"]
        md.append("### Training-time metrics (MA last 100 steps)\n")
        ma = t["train_ma_last100"]
        md.append(f"- Final step: {t['final_step']}, best val loss: {t['best_val_loss']:.4f}, final val_loss/acc: {t['final_val_loss']:.4f}/{t['final_val_acc']:.4f}")
        md.append(f"- Loss: {ma.get('loss', 0):.3f}, answer_loss: {ma.get('answer_loss', 0):.3f}, answer_acc: {ma.get('answer_acc', 0):.3f}")
        md.append(f"- Aux: load_balance={ma.get('aux_lb', 0):.1f}, z_loss={ma.get('aux_z', 0):.1f}")
        md.append(f"- Contrastive: entry={ma.get('l_contrast_entry', 0):.3f}, per_step={ma.get('l_contrast_per_step', 0):.3f}")
        md.append(f"- Routing diversity: w_unique={ma.get('w_unique_per_window', 0):.2f}, r_unique={ma.get('r_unique_per_window', 0):.2f}")
        md.append(f"- R↔W overlap: entry={ma.get('rw_overlap_entry', 0):.3f}, hop={ma.get('rw_overlap_hop', 0):.3f}, all={ma.get('rw_overlap_all', 0):.3f}")
        md.append(f"- Grad spikes >50 / 1000 steps: {ma.get('spike_count_per_1000', 0):.0f}")
        md.append(f"- Step time: {ma.get('step_s', 0):.3f}s")
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
    model = IntegratedLMV2(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.train(mode=False)
    trainer = Phase1RetrievalTrainerV2(model=model, optimizer=None, pad_token_id=EOS)
    print(f"Loaded {args.ckpt.name} (step={ckpt.get('step', '?')})", flush=True)

    # Sample once, all modes see the same chunks
    print(f"Sampling {args.num_chunks} val chunks (paired)...", flush=True)
    all_chunks = val_sampler.sample_batch(args.num_chunks)

    modes_results = {}
    mem_log = []
    read_visited_list = []

    for mode in ["v2", "v2_no_mem", "vanilla_nc", "vanilla_fc"]:
        print(f"\n=== Mode: {mode} ===", flush=True)
        t_start = time.time()
        samples = []
        for b in range(0, len(all_chunks), args.batch_size):
            batch = all_chunks[b : b + args.batch_size]
            if not batch: break
            if mode == "v2":
                res, rv = eval_v2(trainer, batch, reset_memory=False,
                                   mem_log=(mem_log if b < args.batch_size * 30 else None))
                if rv is not None and b < args.batch_size * 30:
                    read_visited_list.append(rv.cpu())
                samples.extend(res)
            elif mode == "v2_no_mem":
                res, _ = eval_v2(trainer, batch, reset_memory=True)
                samples.extend(res)
            else:
                full, mask, a_starts, a_lens, fams = build_vanilla_batch(batch, mode, args.device)
                nlls = answer_nll_llama(model.llama, full, mask, a_starts, a_lens)
                for (nll, fnll), fam in zip(nlls, fams):
                    samples.append((nll, fnll, fam))
            done = b + len(batch)
            if done % 200 == 0:
                print(f"  {done}/{len(all_chunks)}  ({time.time()-t_start:.1f}s)", flush=True)
        agg = aggregate(samples)
        modes_results[mode] = agg
        overall = agg["__overall__"]
        print(f"  {mode} overall NLL/tok: {overall['mean_nll']:.4f}  "
              f"first_tok NLL: {overall['mean_first_nll']:.4f}  "
              f"({overall['n_samples']} samples, {time.time()-t_start:.1f}s)",
              flush=True)
    # Memory readout stats
    memory_readout = None
    if mem_log:
        norms = [m["norm"] for m in mem_log]
        memory_readout = {
            "mean_norm": sum(norms) / len(norms),
            "std_norm": (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms)) ** 0.5,
            "mean_abs": sum(m["mean_abs"] for m in mem_log) / len(mem_log),
            "n_calls": len(mem_log),
        }
        print(f"\nMemory readout: mean_norm={memory_readout['mean_norm']:.3f}, "
              f"std={memory_readout['std_norm']:.3f}, n_calls={memory_readout['n_calls']}",
              flush=True)

    cross_q = cross_question_divergence(read_visited_list)
    if cross_q:
        print(f"Cross-question Jaccard: {cross_q['mean_jaccard']:.3f}", flush=True)

    manifold_state = manifold_state_snapshot(model)
    print(f"Manifold: {manifold_state['n_active_edges']:,d} active edges "
          f"({100*manifold_state['edge_active_fraction']:.1f}%), "
          f"norm={manifold_state['mean_edge_state_norm']:.2f}",
          flush=True)

    training_metrics = extract_training_metrics(args.train_jsonl)

    payload = {
        "ckpt": str(args.ckpt),
        "ckpt_step": ckpt.get("step"),
        "val_dir": str(args.val_dir),
        "num_chunks": args.num_chunks,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "modes": modes_results,
        "memory_readout": memory_readout,
        "cross_q_divergence": cross_q,
        "manifold_state": manifold_state,
        "training": training_metrics,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {args.output}", flush=True)

    if args.markdown:
        md = emit_markdown(payload, args.version_label, args.ckpt, args)
        args.markdown.write_text(md)
        print(f"Markdown → {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
