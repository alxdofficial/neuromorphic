#!/usr/bin/env python3
"""Cross-variant evaluation suite for repr_learning.

Runs all Tier 1 + per-source metrics on a trained checkpoint:
  1. Effective rank of memory tokens (SVD-based)
  2. Mask-ratio sweep: val recon CE at 50, 70, 90, 95, 99% masking
  3. Cross-sample similarity matrix (encoder discriminates inputs?)
  4. Linear probe accuracy: predict source / length-bucket / task-family
  5. Codebook utilization curve + Gini (V2.1, A only)
  6. Decode probe (qualitative)
  7. Per-source val recon (FineWeb vs each composite task)

Usage:
    python scripts/eval/eval_repr.py \\
        --variant flat_baseline \\
        --ckpt outputs/repr_learning/v0_a/ckpts/flat_baseline.last.pt \\
        --out outputs/repr_learning/eval/flat_baseline.json
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel

REPO = Path(__file__).resolve().parents[2]
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"
COMPOSITE_VAL = REPO / "data/wave1/composite_v1/val/passages.jsonl"


# ─── data loading with source/task labels ─────────────────────────────────

@dataclass
class LabeledSample:
    input_ids: list[int]
    source: str          # "fineweb" or "composite"
    task_family: str     # "fineweb" or task name (passphrase, boxes, ...)
    n_tokens: int        # length before truncation
    length_bucket: str   # "short" / "medium" / "long"


def load_labeled_val(cfg: ReprConfig, max_fineweb: int = 500,
                     max_composite_per_task: int = 200) -> list[LabeledSample]:
    """Build a labeled val set with diverse sources."""
    samples: list[LabeledSample] = []
    W = cfg.fixed_window_size

    # FineWeb: slice each doc into W-token windows, take a few per doc
    print(f"[data] loading FineWeb val from {FINEWEB_VAL.name}")
    table = pq.read_table(FINEWEB_VAL, columns=["input_ids"])
    fineweb_count = 0
    for ids in table["input_ids"].to_pylist():
        if fineweb_count >= max_fineweb:
            break
        n_chunks = len(ids) // W
        for i in range(min(n_chunks, 3)):  # cap per-doc
            chunk = ids[i * W : (i + 1) * W]
            samples.append(LabeledSample(
                input_ids=chunk, source="fineweb", task_family="fineweb",
                n_tokens=W, length_bucket=_bucket(W),
            ))
            fineweb_count += 1
            if fineweb_count >= max_fineweb:
                break

    # Composite: group by task_family, take up to N per task. Pack to W tokens.
    print(f"[data] loading composite val from {COMPOSITE_VAL.name}")
    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(COMPOSITE_VAL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_task[row.get("task_family", "unknown")].append(row)

    for task, rows in by_task.items():
        random.Random(7).shuffle(rows)
        per_task = 0
        # Pack short passages together to hit W tokens
        buf: list[int] = []
        for row in rows:
            ids = row.get("passage_token_ids") or []
            buf.extend(ids)
            while len(buf) >= W:
                samples.append(LabeledSample(
                    input_ids=buf[:W], source="composite", task_family=task,
                    n_tokens=W, length_bucket=_bucket(W),
                ))
                buf = buf[W:]
                per_task += 1
                if per_task >= max_composite_per_task:
                    break
            if per_task >= max_composite_per_task:
                break

    print(f"[data] {len(samples)} labeled samples loaded "
          f"({sum(1 for s in samples if s.source == 'fineweb')} fineweb, "
          f"{sum(1 for s in samples if s.source == 'composite')} composite)")
    return samples


def _bucket(n: int) -> str:
    if n < 128:
        return "short"
    if n < 320:
        return "medium"
    return "long"


def sample_span_mask_at_ratio(seq_len: int, target_ratio: float,
                              span_range: tuple[int, int] = (5, 15),
                              rng: random.Random | None = None) -> torch.Tensor:
    """Deterministic-ish span mask at a specified ratio (no overshoot)."""
    if rng is None:
        rng = random.Random()
    target_n = int(seq_len * target_ratio)
    masked: set[int] = set()
    attempts = 0
    span_min, span_max = span_range
    done = False
    while not done and attempts < seq_len * 8:
        span_len = rng.randint(span_min, span_max)
        start = rng.randint(0, max(seq_len - span_len, 0))
        for i in range(start, min(start + span_len, seq_len)):
            if i not in masked:
                masked.add(i)
                if len(masked) >= target_n:
                    done = True
                    break
        attempts += 1
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for p in masked:
        mask[p] = True
    return mask


def batch_from_samples(samples: list[LabeledSample], mask_ratio: float,
                       device: str, seed: int = 0):
    rng = random.Random(seed)
    input_ids = torch.tensor([s.input_ids for s in samples], dtype=torch.long, device=device)
    B, T = input_ids.shape
    attention_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask_positions = torch.stack([
        sample_span_mask_at_ratio(T, mask_ratio, rng=random.Random(seed + i))
        for i in range(B)
    ]).to(device)
    return input_ids, attention_mask, mask_positions


# ─── individual metrics ──────────────────────────────────────────────────

@torch.no_grad()
def metric_effective_rank(model, samples, device, n: int = 64, batch_size: int = 4) -> dict:
    """SVD effective rank of memory matrices.

    For each sample, take [96, d_llama] memory matrix → compute its singular
    values → effective rank = exp(entropy of normalized singular values).
    Lower = more redundant slots, higher = more independent slots.
    Range: 1 (rank-1 collapse) to 96 (all slots independent).
    """
    eff_ranks = []
    sub = samples[:n]
    for i in range(0, len(sub), batch_size):
        batch = sub[i:i + batch_size]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, 0.7, device, seed=42 + i)
        out = model(input_ids, attn_mask, mask_pos)
        mem = out["memory"].float()  # [B, M, d_llama]
        for b in range(mem.shape[0]):
            s = torch.linalg.svdvals(mem[b])  # [M]
            p = s / (s.sum() + 1e-12)
            ent = -(p * (p + 1e-12).log()).sum()
            eff_ranks.append(ent.exp().item())
    return {
        "n_samples": len(eff_ranks),
        "mean": sum(eff_ranks) / len(eff_ranks),
        "min": min(eff_ranks),
        "max": max(eff_ranks),
        "memory_token_count": mem.shape[1] if eff_ranks else 0,
    }


@torch.no_grad()
def metric_mask_ratio_sweep(model, samples, device,
                            ratios: list[float] = (0.5, 0.7, 0.9, 0.95, 0.99),
                            n: int = 256, batch_size: int = 2) -> dict:
    """Val recon CE at each mask ratio."""
    results = {}
    sub = samples[:n]
    for ratio in ratios:
        losses = []
        for i in range(0, len(sub), batch_size):
            batch = sub[i:i + batch_size]
            input_ids, attn_mask, mask_pos = batch_from_samples(batch, ratio, device, seed=100 + i)
            out = model(input_ids, attn_mask, mask_pos)
            losses.append(float(out["loss_recon"].item() if isinstance(out["loss_recon"], torch.Tensor) else out["loss_recon"]))
        results[f"mask_{int(ratio*100)}"] = sum(losses) / max(len(losses), 1)
    return results


@torch.no_grad()
def metric_cross_sample_similarity(model, samples, device, n: int = 64,
                                   batch_size: int = 2) -> dict:
    """Encode N samples; compute pairwise cosine of flattened memory.

    Healthy: diagonal high (self-similarity = 1), off-diagonal moderate-to-low
    (different inputs → different memory). If off-diagonal is also high,
    encoder is collapsing across samples (encoding ≈ constant).
    """
    mems = []
    sub = samples[:n]
    for i in range(0, len(sub), batch_size):
        batch = sub[i:i + batch_size]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, 0.7, device, seed=200 + i)
        out = model(input_ids, attn_mask, mask_pos)
        mem = out["memory"].float()  # [B, M, d_llama]
        # Flatten memory across (slots, dims) → one vector per sample
        flat = mem.reshape(mem.shape[0], -1)
        flat = F.normalize(flat, dim=-1)
        mems.append(flat.cpu())
    M = torch.cat(mems, dim=0)  # [N, M*d_llama]
    sim = M @ M.T               # [N, N]
    diag_mask = torch.eye(sim.shape[0], dtype=torch.bool)
    off_diag = sim[~diag_mask]
    return {
        "n_samples": sim.shape[0],
        "mean_off_diag_cos": off_diag.mean().item(),
        "max_off_diag_cos": off_diag.max().item(),
        "min_off_diag_cos": off_diag.min().item(),
        "median_off_diag_cos": off_diag.median().item(),
    }


@torch.no_grad()
def _encode_all(model, samples, device, batch_size: int = 2) -> torch.Tensor:
    """Encode all samples, return [N, M*d_llama] flat memory features."""
    feats = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, 0.7, device, seed=300 + i)
        out = model(input_ids, attn_mask, mask_pos)
        mem = out["memory"].float().reshape(out["memory"].shape[0], -1)
        feats.append(mem.cpu())
    return torch.cat(feats, dim=0)


def metric_linear_probes(model, samples, device, batch_size: int = 2) -> dict:
    """Train linear classifiers on memory tokens to predict simple properties."""
    print("  [probe] encoding all samples...")
    feats = _encode_all(model, samples, device, batch_size)
    feats = feats.float()  # [N, F]
    N, F_dim = feats.shape

    # Build label vectors
    src_labels = [0 if s.source == "fineweb" else 1 for s in samples]
    task_label_to_id = {t: i for i, t in enumerate(sorted({s.task_family for s in samples}))}
    task_labels = [task_label_to_id[s.task_family] for s in samples]
    length_labels = [{"short": 0, "medium": 1, "long": 2}[s.length_bucket] for s in samples]

    results = {}
    rng = random.Random(0)
    perm = list(range(N))
    rng.shuffle(perm)
    split = int(0.8 * N)
    train_idx = perm[:split]
    test_idx = perm[split:]

    for label_name, labels, n_classes in [
        ("source", src_labels, 2),
        ("task_family", task_labels, len(task_label_to_id)),
        ("length_bucket", length_labels, 3),
    ]:
        labels_t = torch.tensor(labels, dtype=torch.long)
        if n_classes < 2 or labels_t[train_idx].unique().numel() < 2:
            results[label_name] = {
                "skipped": "fewer than 2 classes in training set",
            }
            continue
        # Fit a linear classifier via closed-form pseudo-inverse on one-hot targets
        X_train = feats[train_idx]
        y_train = labels_t[train_idx]
        X_test = feats[test_idx]
        y_test = labels_t[test_idx]

        # Center features (helps closed-form)
        mu = X_train.mean(dim=0, keepdim=True)
        X_train = X_train - mu
        X_test = X_test - mu

        # One-hot target
        Y_train = F.one_hot(y_train, num_classes=n_classes).float()
        # Ridge regression. Primal form (W = (X^T X + λI)^-1 X^T Y) needs
        # an F×F solve — prohibitive when F ≫ N (e.g., 96·2048 = 196,608).
        # Use the dual form (Woodbury): predictions are
        #     X_test W = X_test X^T (X X^T + λI)^-1 Y
        # which only ever materializes an N×N matrix. Mathematically
        # equivalent for ridge regression.
        lam = 1.0
        N_train = X_train.shape[0]
        K_train = X_train @ X_train.T                          # [N, N]
        K_train += lam * torch.eye(N_train)
        K_test = X_test @ X_train.T                            # [N_test, N]
        alpha = torch.linalg.solve(K_train, Y_train)           # [N, C]
        scores = K_test @ alpha                                # [N_test, C]
        preds = scores.argmax(dim=-1)
        acc = (preds == y_test).float().mean().item()
        results[label_name] = {
            "accuracy": acc,
            "n_classes": n_classes,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "chance_acc": 1.0 / n_classes,
        }
    return results


@torch.no_grad()
def metric_codebook_utilization(model, samples, device, n: int = 256,
                                batch_size: int = 2) -> dict:
    """Codebook usage histogram + Gini coefficient (V2.1 + A only)."""
    if not hasattr(model.encoder, "concept_id"):
        return {"skipped": "no codebook in this variant"}
    n_nodes = model.encoder.concept_id.shape[0]
    counts = torch.zeros(n_nodes, dtype=torch.long)
    sub = samples[:n]
    for i in range(0, len(sub), batch_size):
        batch = sub[i:i + batch_size]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, 0.7, device, seed=400 + i)
        out = model(input_ids, attn_mask, mask_pos)
        picked = out["aux"].get("picked_ids")
        if picked is None:
            return {"skipped": "encoder does not expose picked_ids"}
        counts += torch.bincount(picked.reshape(-1).cpu(), minlength=n_nodes).long()

    total = counts.sum().item()
    # Gini coefficient: 0 = uniform, 1 = single-bin
    sorted_counts, _ = counts.sort()
    cumsum = sorted_counts.float().cumsum(dim=0)
    if cumsum[-1] > 0:
        gini = 1.0 - 2.0 * (cumsum.sum() / (cumsum.shape[0] * cumsum[-1])).item() + 1.0 / cumsum.shape[0]
    else:
        gini = float("nan")

    # Concentration: fraction of picks in top 10/100/1000 codes
    counts_desc, _ = counts.sort(descending=True)
    cumfrac = counts_desc.float().cumsum(dim=0) / max(total, 1)
    top10 = cumfrac[min(9, n_nodes - 1)].item()
    top100 = cumfrac[min(99, n_nodes - 1)].item()
    top1000 = cumfrac[min(999, n_nodes - 1)].item()

    return {
        "n_codes_total": n_nodes,
        "n_codes_ever_picked": int((counts > 0).sum().item()),
        "coverage_frac": (counts > 0).float().mean().item(),
        "gini": gini,
        "top10_concentration": top10,
        "top100_concentration": top100,
        "top1000_concentration": top1000,
        "total_picks": total,
    }


@torch.no_grad()
def metric_decode_probe(model, samples, device, n: int = 4,
                        prompt: str = "Summary of the passage: ",
                        gen_tokens: int = 40, top_k: int = 1) -> dict:
    """Feed memory tokens + a prompt to frozen Llama, AR-decode, see what comes."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model.cfg.llama_model)

    samples = samples[:n]
    decoded_samples = []
    for i, sample in enumerate(samples):
        input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask_pos = torch.zeros_like(input_ids, dtype=torch.bool)  # no masking — pure encoding
        out = model(input_ids, attn_mask, mask_pos)
        memory = out["memory"]  # [1, M, d_llama]
        prompt_ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)],
                                  dtype=torch.long, device=device)
        prompt_embeds = model.decoder.llama.get_input_embeddings()(prompt_ids)
        # Concat: [memory_tokens, prompt_embeds]
        all_embeds = torch.cat([memory.to(prompt_embeds.dtype), prompt_embeds], dim=1)

        generated = []
        for _ in range(gen_tokens):
            llm_out = model.decoder.llama(inputs_embeds=all_embeds)
            next_logits = llm_out.logits[:, -1, :]  # [1, V]
            next_id = next_logits.argmax(dim=-1, keepdim=True)  # greedy
            generated.append(next_id.item())
            next_embed = model.decoder.llama.get_input_embeddings()(next_id)
            all_embeds = torch.cat([all_embeds, next_embed.to(all_embeds.dtype)], dim=1)

        original_text = tok.decode(sample.input_ids[:80], skip_special_tokens=True)
        decoded_text = tok.decode(generated, skip_special_tokens=True)
        decoded_samples.append({
            "original_first_80_tokens": original_text[:200],
            "source": sample.source,
            "task_family": sample.task_family,
            "decoded_after_summary_prompt": decoded_text,
        })
    return {"samples": decoded_samples, "prompt": prompt, "gen_tokens": gen_tokens}


@torch.no_grad()
def metric_per_source_recon(model, samples, device, batch_size: int = 2,
                            mask_ratio: float = 0.7) -> dict:
    """Val recon CE broken out by source/task."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, mask_ratio, device, seed=500 + i)
        out = model(input_ids, attn_mask, mask_pos)
        loss = out["loss_recon"].item() if isinstance(out["loss_recon"], torch.Tensor) else out["loss_recon"]
        # Loss is mean across batch positions; attribute to all samples in batch
        for s in batch:
            by_task[s.task_family].append(float(loss))
    return {
        task: {
            "mean_recon_ce": sum(losses) / max(len(losses), 1),
            "n_samples": len(losses),
        }
        for task, losses in by_task.items()
    }


# ─── orchestration ────────────────────────────────────────────────────────

def load_model(variant: str, ckpt_path: Path, llama, cfg: ReprConfig, device: str):
    """Load model from checkpoint."""
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] step {ckpt.get('step', '?')}, "
              f"{len(missing)} missing + {len(unexpected)} unexpected keys")
    else:
        print(f"[load] WARN: no checkpoint at {ckpt_path}, using random init")
    model.train(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-fineweb", type=int, default=500)
    ap.add_argument("--max-composite-per-task", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", nargs="+", default=[
        "effective_rank", "mask_ratio_sweep", "cross_sample_similarity",
        "linear_probes", "codebook_utilization", "decode_probe",
        "per_source_recon",
    ])
    args = ap.parse_args()

    cfg = ReprConfig(batch_size=8, fixed_window_size=256)
    device = args.device

    print(f"Variant: {args.variant}")
    print(f"Loading Llama (shared)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)
    model = load_model(args.variant, args.ckpt, llama, cfg, device)
    print(f"  trainable params: {model.n_trainable_params():,}")

    print(f"Loading labeled val data...")
    samples = load_labeled_val(cfg, max_fineweb=args.max_fineweb,
                                max_composite_per_task=args.max_composite_per_task)

    results: dict = {
        "variant": args.variant,
        "ckpt": str(args.ckpt),
        "n_samples": len(samples),
    }

    runners = {
        "effective_rank": lambda: metric_effective_rank(model, samples, device),
        "mask_ratio_sweep": lambda: metric_mask_ratio_sweep(model, samples, device),
        "cross_sample_similarity": lambda: metric_cross_sample_similarity(model, samples, device),
        "linear_probes": lambda: metric_linear_probes(model, samples, device),
        "codebook_utilization": lambda: metric_codebook_utilization(model, samples, device),
        "decode_probe": lambda: metric_decode_probe(model, samples, device),
        "per_source_recon": lambda: metric_per_source_recon(model, samples, device),
    }
    for name in args.metrics:
        print(f"\n[metric] {name} ...")
        try:
            results[name] = runners[name]()
            print(f"  done")
        except Exception as e:
            results[name] = {"error": repr(e)}
            print(f"  ERROR: {e}")
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    # Brief stdout summary
    print(f"\n{'='*60}\nSummary: {args.variant}\n{'='*60}")
    if "mask_ratio_sweep" in results and isinstance(results["mask_ratio_sweep"], dict):
        for k, v in results["mask_ratio_sweep"].items():
            if isinstance(v, (int, float)):
                print(f"  recon @ {k}: {v:.3f}")
    if "effective_rank" in results and "mean" in results["effective_rank"]:
        print(f"  effective rank: {results['effective_rank']['mean']:.2f} / "
              f"{results['effective_rank']['memory_token_count']}")
    if "cross_sample_similarity" in results and "mean_off_diag_cos" in results["cross_sample_similarity"]:
        print(f"  cross-sample off-diag cos: {results['cross_sample_similarity']['mean_off_diag_cos']:.3f}")
    if "linear_probes" in results:
        for label, r in results["linear_probes"].items():
            if "accuracy" in r:
                print(f"  probe[{label}]: {r['accuracy']:.3f} (chance: {r['chance_acc']:.3f})")
    if "codebook_utilization" in results and "coverage_frac" in results["codebook_utilization"]:
        cu = results["codebook_utilization"]
        print(f"  codebook coverage: {cu['coverage_frac']*100:.1f}% "
              f"({cu['n_codes_ever_picked']}/{cu['n_codes_total']}), "
              f"Gini={cu['gini']:.3f}")


if __name__ == "__main__":
    main()
