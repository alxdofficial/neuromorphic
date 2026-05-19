#!/usr/bin/env python3
"""Per-task read↔write overlap diagnostic for V2 vocabulary-trajectory architecture.

Mirrors scripts/eval/rw_overlap_v1.py but uses V2's IntegratedLM + Phase1RetrievalTrainerV2.

For each val chunk:
  1. Run 8 passage writes (write_mode='passage'), collect write_visited_ids
  2. Run QA read (write_mode='qa'), collect read_visited_ids
  3. Compute per-chunk overlap fractions (rw_target_all, rw_target_entry, rw_target_hop, rw_distractor_mean, rw_target_lift)
  4. Aggregate per task family

Usage:
  python scripts/eval/rw_overlap_v2.py \\
      --ckpt outputs/wave1_v2/ckpt.10000.pt \\
      --val-dir data/wave1/composite_v1/val \\
      --num-chunks 400 --batch-size 8 \\
      --output outputs/wave1_v2/rw_overlap.json \\
      --markdown outputs/wave1_v2/rw_overlap.md \\
      --version-label V2
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.wave1.common.sampler import CompositeRetrievalAdapter
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2.trainer import Phase1RetrievalTrainerV2

EOS = 128001


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--num-chunks", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--version-label", type=str, default="V2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _set_overlap(read_row: torch.Tensor, write_row: torch.Tensor) -> float:
    rs = set(read_row.reshape(-1).tolist())
    ws = set(write_row.reshape(-1).tolist())
    if not rs:
        return float("nan")
    return len(rs & ws) / len(rs)


@torch.no_grad()
def chunk_overlaps(trainer, batch: list[dict]) -> list[dict]:
    cfg = trainer.model.cfg
    M = len(batch)
    device = next(trainer.model.parameters()).device

    tensors = trainer._build_tensors(batch, device)
    passages = tensors["passages"]
    qa = tensors["qa"]
    question_ids = tensors["question_ids"]
    question_lens = tensors["question_lens"]
    target_idxs = tensors["target_idxs"]

    # 8 passage writes.
    write_visited_per_fact: list[torch.Tensor] = []
    n_facts = cfg.n_facts_per_chunk
    for i in range(n_facts):
        passage_ids_i = passages[:, i, :]
        passage_mask_i = (passage_ids_i != trainer.pad_token_id)
        out = trainer.model.forward_window(
            lm_input_ids=passage_ids_i,
            prev_window_hiddens=None,
            attention_mask=passage_mask_i,
            prev_attention_mask=None,
            hard_routing=True,
            write_mode="passage",
        )
        write_visited_per_fact.append(out["write_visited_ids"].detach().cpu())

    # QA read.
    q_hiddens = trainer._compute_question_hiddens(question_ids, question_lens)
    q_mask = (question_ids != trainer.pad_token_id)
    qa_mask = (qa != trainer.pad_token_id)
    out_qa = trainer.model.forward_window(
        lm_input_ids=qa,
        prev_window_hiddens=None,
        attention_mask=qa_mask,
        prev_attention_mask=None,
        read_conditioning_hiddens=q_hiddens,
        read_conditioning_mask=q_mask,
        hard_routing=True,
        write_mode="qa",
    )
    read_visited = out_qa["read_visited_ids"].detach().cpu()
    target_idxs_cpu = target_idxs.cpu()
    wv_stack = torch.stack(write_visited_per_fact, dim=1)  # [M, 8, J, K_w]
    K_w = wv_stack.shape[-1]

    # Reachability probe: like v1 but using V2's edge_dst adjacency, masked
    # by edge_active (V2 has dynamic edges; inactive slots have dst=-1).
    edge_dst_cpu = trainer.model.manifold.edge_dst.detach().cpu()        # [N, K_max]
    edge_active_cpu = trainer.model.manifold.edge_active.detach().cpu()  # [N, K_max]
    K_r = read_visited.shape[-1]

    out_rows: list[dict] = []
    for m in range(M):
        target_idx = int(target_idxs_cpu[m].item())
        fam = batch[m]["metadata"]["target_entity_class"]
        r = read_visited[m]
        w_target = wv_stack[m, target_idx]
        w_all = wv_stack[m].reshape(-1)
        w_distractors = [
            wv_stack[m, i].reshape(-1) for i in range(n_facts) if i != target_idx
        ]

        # Reachability per hop k.
        reachability_per_hop: list[float] = []
        for k in range(1, min(K_r, w_target.shape[-1])):
            target_set_k = set(w_target[:, k].tolist())  # union across write J
            reachable_count = 0
            for j_read in range(r.shape[0]):
                prev_cell = int(r[j_read, k - 1].item())
                # Only count active edges (V2-specific)
                active_mask = edge_active_cpu[prev_cell]
                nbrs = set(edge_dst_cpu[prev_cell][active_mask].tolist())
                if target_set_k & nbrs:
                    reachable_count += 1
            reachability_per_hop.append(reachable_count / r.shape[0])
        reachability_mean = (
            sum(reachability_per_hop) / len(reachability_per_hop)
            if reachability_per_hop else float("nan")
        )
        r_entry = r[:, :1]
        r_hop = r[:, 1:] if r.shape[-1] > 1 else None
        w_target_entry = w_target[:, :1]
        w_target_hop = w_target[:, 1:] if K_w > 1 else None

        rw_target_all = _set_overlap(r, w_target)
        rw_target_entry = _set_overlap(r_entry, w_target_entry)
        rw_target_hop = (
            _set_overlap(r_hop, w_target_hop)
            if r_hop is not None and w_target_hop is not None
            else float("nan")
        )
        rw_all_overlap = _set_overlap(r, w_all)
        distractor_overlaps = [_set_overlap(r, wd) for wd in w_distractors]
        valid_dist = [d for d in distractor_overlaps if d == d]
        rw_distractor_mean = (
            sum(valid_dist) / len(valid_dist) if valid_dist else float("nan")
        )
        rw_target_lift = (
            rw_target_all - rw_distractor_mean
            if (rw_target_all == rw_target_all and rw_distractor_mean == rw_distractor_mean)
            else float("nan")
        )

        out_rows.append({
            "task": fam,
            "rw_target_all": rw_target_all,
            "rw_target_entry": rw_target_entry,
            "rw_target_hop": rw_target_hop,
            "rw_all_overlap": rw_all_overlap,
            "rw_distractor_mean": rw_distractor_mean,
            "rw_target_lift": rw_target_lift,
            "reachability_mean": reachability_mean,
            "reachability_per_hop": reachability_per_hop,
        })
    return out_rows


def aggregate(rows: list[dict]) -> dict:
    by_task = defaultdict(lambda: defaultdict(list))
    by_task_per_hop = defaultdict(lambda: defaultdict(list))
    for r in rows:
        task = r["task"]
        for k in ("rw_target_all", "rw_target_entry", "rw_target_hop",
                  "rw_all_overlap", "rw_distractor_mean", "rw_target_lift",
                  "reachability_mean"):
            v = r.get(k, float("nan"))
            if v == v:
                by_task[task][k].append(v)
                by_task["__overall__"][k].append(v)
        ph = r.get("reachability_per_hop", [])
        for hop_idx, val in enumerate(ph):
            by_task_per_hop[task][hop_idx].append(val)
            by_task_per_hop["__overall__"][hop_idx].append(val)
    out = {}
    for task, d in by_task.items():
        out[task] = {
            k: (sum(vs) / len(vs) if vs else float("nan"))
            for k, vs in d.items()
        }
        out[task]["n_samples"] = len(d.get("rw_target_all", []))
        per_hop = by_task_per_hop[task]
        if per_hop:
            out[task]["reachability_per_hop"] = {
                f"k={hop_idx+1}": (sum(v) / len(v) if v else float("nan"))
                for hop_idx, v in sorted(per_hop.items())
            }
    return out


def emit_markdown(payload: dict, args) -> str:
    md = [f"## {payload['version_label']} — per-task R↔W overlap "
          f"({Path(payload['ckpt']).name})\n"]
    md.append(f"_{args.num_chunks} val chunks, BS={args.batch_size}. "
              f"Higher = read trajectory visits cells the write trajectory deposited._\n")
    md.append("Each value = mean over chunks of `|R ∩ W| / |R|`.\n")
    md.append("- **rw_target_all** = reads vs target passage's writes")
    md.append("- **rw_target_entry** = read entry cells vs target passage write entry cells")
    md.append("- **rw_target_hop** = read non-entry cells vs target passage write non-entry cells")
    md.append("- **rw_all_overlap** = reads vs all 8 passages' writes (high floor)")
    md.append("- **rw_distractor_mean** = mean over 7 non-target passages")
    md.append("- **rw_target_lift** = rw_target_all − rw_distractor_mean (positive = read is task-specific)")
    md.append("")
    agg = payload["per_task"]
    tasks = sorted(t for t in agg if t != "__overall__") + ["__overall__"]
    md.append("| Task | rw_target_all | rw_target_entry | rw_target_hop | rw_distractor_mean | **rw_target_lift** | rw_all_overlap | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in tasks:
        d = agg[t]
        row = [t if t != "__overall__" else "**__overall__**"]
        for k in ("rw_target_all", "rw_target_entry", "rw_target_hop",
                  "rw_distractor_mean", "rw_target_lift", "rw_all_overlap"):
            v = d.get(k, float("nan"))
            cell = f"{v:.3f}" if v == v else "—"
            if k == "rw_target_lift" and v == v:
                cell = f"**{v:+.3f}**"
            row.append(cell)
        row.append(str(d.get("n_samples", 0)))
        md.append("| " + " | ".join(row) + " |")
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
        cfg = TrajMemV2Config(**cfg)
    model = IntegratedLMV2(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.train(mode=False)
    trainer = Phase1RetrievalTrainerV2(model=model, optimizer=None, pad_token_id=EOS)
    print(f"Loaded {args.ckpt.name} (step={ckpt.get('step', '?')}; "
          f"N={cfg.N}, J={cfg.J}, K_read={cfg.K_read}, K_write={cfg.K_write})", flush=True)

    print(f"Sampling {args.num_chunks} val chunks...", flush=True)
    all_chunks = val_sampler.sample_batch(args.num_chunks)

    rows: list[dict] = []
    t0 = time.time()
    for b in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[b : b + args.batch_size]
        if not batch:
            break
        rows.extend(chunk_overlaps(trainer, batch))
        done = b + len(batch)
        if done % 80 == 0:
            print(f"  {done}/{len(all_chunks)}  ({time.time()-t0:.1f}s)", flush=True)

    agg = aggregate(rows)
    payload = {
        "ckpt": str(args.ckpt),
        "version_label": args.version_label,
        "num_chunks": args.num_chunks,
        "batch_size": args.batch_size,
        "n_total_rows": len(rows),
        "config": {"N": cfg.N, "J": cfg.J, "K_read": cfg.K_read, "K_write": cfg.K_write},
        "per_task": agg,
        "rows": rows,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {args.output}", flush=True)
    print(f"\nOverall:", flush=True)
    o = agg["__overall__"]
    print(f"  rw_target_all     = {o.get('rw_target_all', float('nan')):.4f}", flush=True)
    print(f"  rw_target_entry   = {o.get('rw_target_entry', float('nan')):.4f}", flush=True)
    print(f"  rw_target_hop     = {o.get('rw_target_hop', float('nan')):.4f}", flush=True)
    print(f"  rw_distractor_mean= {o.get('rw_distractor_mean', float('nan')):.4f}", flush=True)
    print(f"  rw_target_lift    = {o.get('rw_target_lift', float('nan')):+.4f}", flush=True)
    print(f"  rw_all_overlap    = {o.get('rw_all_overlap', float('nan')):.4f}", flush=True)

    if args.markdown:
        args.markdown.write_text(emit_markdown(payload, args))
        print(f"Markdown → {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
