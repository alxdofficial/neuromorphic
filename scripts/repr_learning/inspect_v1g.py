#!/usr/bin/env python3
"""v1g sanity-check: do trained models actually make sensible predictions?

For each variant, loads the 500-step checkpoint and:
  (1) Visual decode: prints GT vs predicted tokens at still-masked positions
  (2) Top-1 / top-5 accuracy on the still-masked positions of a val batch
  (3) Predicted-token entropy + concentration in the top-K most common tokens
  (4) Zero-memory ablation: re-run with memory = 0, compare loss + accuracy
      → if loss/accuracy barely move, the encoder isn't contributing
  (5) Memory diagnostics: norms, pairwise cosine of memory tokens

The point is: loss going from 11 → 6 doesn't mean the model is sane. We need
to look at predictions and check the encoder is actually being used.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_sentence import make_sentence_chunk_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"


# (variant_name, label, checkpoint_dir, b_div_scale, mt_div_scale)
VARIANTS = [
    ("flat_baseline",       "A (flat)",      "v1g_flat_baseline",            1000.0, 1000.0),
    ("continuous_baseline", "B (slots@50)",  "v1g_bd50_continuous_baseline",   50.0, 1000.0),
    ("memorizing_baseline", "MT (K=36)",     "v1g_memorizing_baseline",      1000.0,   50.0),
    ("recurrent_baseline",  "Mamba",         "v1g_recurrent_baseline",       1000.0, 1000.0),
    ("vanilla_llama",       "Vanilla",       "v1g_vanilla_llama",            1000.0, 1000.0),
]


def make_cfg(b_diversity_scale: float = 50.0, mt_diversity_scale: float = 50.0) -> ReprConfig:
    return ReprConfig(
        batch_size=2,
        fixed_window_size=1024,
        max_window_size=4096,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=36,
        edge_token_packing="fused",
        b_diversity_scale=b_diversity_scale,
        mt_diversity_scale=mt_diversity_scale,
    )


def to_device(batch, device):
    for f in ("input_ids", "attention_mask", "query_input_ids",
              "mask_positions", "reveal_positions", "query_lengths",
              "query_starts"):
        setattr(batch, f, getattr(batch, f).to(device))
    return batch


def load_ckpt_into(model, ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msd = sd.get("model_state_dict", sd)
    # Strict=False because we deliberately drop frozen Llama base weights.
    result = model.load_state_dict(msd, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("decoder.llama.")]
    unexpected = result.unexpected_keys
    if missing or unexpected:
        print(f"  [load] missing(non-llama)={len(missing)} unexpected={len(unexpected)}")
    return True


def decode_token(tokenizer, tid):
    s = tokenizer.decode([int(tid)])
    return s if s else "_"


def visualize_predictions(tokenizer, batch, out, b_show=0, k_show=0):
    """Print one queried sentence: GT vs predicted tokens at still-masked pos.

    sent_logits: [BK, L_max, vocab]
    still_masked: [BK, L_max] bool
    query_ids_flat: [BK, L_max] long
    """
    B = batch.input_ids.shape[0]
    K = batch.query_input_ids.shape[1]
    L_max = batch.query_input_ids.shape[2]
    idx = b_show * K + k_show
    L = int(batch.query_lengths[b_show, k_show].item())
    gt_ids = out["query_ids_flat"][idx, :L].cpu().tolist()
    still_masked = out["still_masked"][idx, :L].cpu().tolist()
    mask_pos = batch.mask_positions[b_show, k_show, :L].cpu().tolist()
    reveal_pos = batch.reveal_positions[b_show, k_show, :L].cpu().tolist()
    logits = out["sent_logits"][idx, :L, :].float()       # [L, vocab]
    top1_ids = logits.argmax(dim=-1).cpu().tolist()
    top5_ids = logits.topk(5, dim=-1).indices.cpu().tolist()

    parts_gt, parts_pred = [], []
    for t in range(L):
        gt_tok = decode_token(tokenizer, gt_ids[t]).strip() or "_"
        if still_masked[t]:
            # Predict here
            pred_tok = decode_token(tokenizer, top1_ids[t]).strip() or "_"
            mark = "✓" if top1_ids[t] == gt_ids[t] else "·"
            parts_gt.append(f"[{gt_tok}]")
            parts_pred.append(f"[{pred_tok}]{mark}")
        elif mask_pos[t] and reveal_pos[t]:
            # Revealed: model sees GT (treated as "previously predicted")
            parts_gt.append(f"<{gt_tok}>")
            parts_pred.append(f"<{gt_tok}>")
        else:
            # Unmasked visible
            parts_gt.append(gt_tok)
            parts_pred.append(gt_tok)
    print(f"  GT     : {' '.join(parts_gt)}")
    print(f"  PRED   : {' '.join(parts_pred)}")
    # Show top-5 alternatives for the first still-masked position
    first_sm = next((t for t in range(L) if still_masked[t]), None)
    if first_sm is not None:
        gt_tok = decode_token(tokenizer, gt_ids[first_sm]).strip()
        alts = [decode_token(tokenizer, tid).strip() or "_" for tid in top5_ids[first_sm]]
        print(f"  top-5 at first still-masked (gt={gt_tok!r}): {alts}")


def compute_accuracy(out, k_top: int = 5) -> dict:
    """Top-1 and top-k accuracy on still-masked positions."""
    still_masked = out["still_masked"]                    # [BK, L_max]
    if not still_masked.any():
        return {"top1": float("nan"), f"top{k_top}": float("nan"), "n": 0}
    logits = out["sent_logits"]                           # [BK, L_max, vocab]
    targets = out["query_ids_flat"]                       # [BK, L_max]
    sel_logits = logits[still_masked]                     # [N, vocab]
    sel_targets = targets[still_masked]                   # [N]
    top1 = (sel_logits.argmax(dim=-1) == sel_targets).float().mean().item()
    topk = sel_logits.topk(k_top, dim=-1).indices         # [N, k_top]
    topk_hit = (topk == sel_targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1": top1, f"top{k_top}": topk_hit, "n": int(still_masked.sum().item())}


def predicted_token_distribution(out, tokenizer, top_n: int = 5) -> str:
    """Are predictions concentrated on a few common tokens, or spread out?"""
    still_masked = out["still_masked"]
    if not still_masked.any():
        return "n/a"
    preds = out["sent_logits"][still_masked].argmax(dim=-1).cpu().tolist()
    counter = Counter(preds)
    total = len(preds)
    top = counter.most_common(top_n)
    s = ", ".join(f"{decode_token(tokenizer, tid).strip() or '_'!r}={cnt}/{total}"
                  for tid, cnt in top)
    return s


def memory_diagnostics(memory: torch.Tensor) -> dict:
    """Norms, pairwise cosines for memory tokens [B, M, d]."""
    if memory.shape[1] == 0:
        return {"M": 0}
    m = memory.float()
    norms = m.norm(dim=-1)                                # [B, M]
    m_n = F.normalize(m, dim=-1)
    cos = m_n @ m_n.transpose(1, 2)                       # [B, M, M]
    M = cos.shape[1]
    eye = torch.eye(M, dtype=torch.bool, device=cos.device)
    off_diag = cos[:, ~eye].view(cos.shape[0], -1)
    return {
        "M": M,
        "mean_norm": float(norms.mean()),
        "mean_pairwise_cos": float(off_diag.mean()),
        "max_pairwise_cos": float(off_diag.max()),
    }


@torch.no_grad()
def run_one_variant(variant, label, ckpt_dir, b_div, mt_div, llama, tokenizer, batch, device):
    print(f"\n{'='*78}\n  {label}   (variant={variant})\n{'='*78}")
    cfg = make_cfg(b_diversity_scale=b_div, mt_diversity_scale=mt_div)
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    ckpt_path = REPO / "outputs/repr_learning" / ckpt_dir / "ckpts" / f"{variant}.last.pt"
    if not load_ckpt_into(model, ckpt_path):
        print(f"  [warn] no checkpoint at {ckpt_path} — using random init")
    model.train(False)

    # Normal forward
    out = model.compute_sentence_recon_loss(batch)
    acc = compute_accuracy(out)
    # For MT use the per-sentence retrieved memory for diagnostics
    mem_for_diag = out.get("mt_memory_bk")
    if mem_for_diag is None:
        mem_for_diag = out["memory"]
    mem_d = memory_diagnostics(mem_for_diag)
    tok_dist = predicted_token_distribution(out, tokenizer)

    print(f"  loss_recon   = {float(out['loss_recon']):.4f}")
    print(f"  top-1 acc    = {acc['top1']*100:5.1f}%   "
          f"top-5 acc    = {acc['top5']*100:5.1f}%   "
          f"(N={acc['n']} still-masked positions)")
    print(f"  memory       : M={mem_d['M']}  "
          f"mean_norm={mem_d.get('mean_norm', 0):.2f}  "
          f"mean_pairwise_cos={mem_d.get('mean_pairwise_cos', 0):.3f}  "
          f"max={mem_d.get('max_pairwise_cos', 0):.3f}")
    print(f"  top-5 most-predicted tokens: {tok_dist}")

    print(f"\n  --- Sample queried sentence (b=0, k=0) ---")
    visualize_predictions(tokenizer, batch, out, b_show=0, k_show=0)
    print(f"\n  --- Sample queried sentence (b=1, k=2) ---")
    visualize_predictions(tokenizer, batch, out, b_show=1, k_show=2)

    # Zero-memory ablation: monkey-patch to zero out memory.
    # MT uses retrieve_per_sentence (separate code path); other variants use finalize_memory.
    print(f"\n  --- Zero-memory ablation ---")
    if variant == "memorizing_baseline":
        orig_fn = model.encoder.retrieve_per_sentence
        def zero_retrieve(bank, *args, **kwargs):
            mem, aux = orig_fn(bank, *args, **kwargs)
            return torch.zeros_like(mem), aux
        model.encoder.retrieve_per_sentence = zero_retrieve
        out_zm = model.compute_sentence_recon_loss(batch)
        acc_zm = compute_accuracy(out_zm)
        model.encoder.retrieve_per_sentence = orig_fn
    else:
        orig_finalize = model.encoder.finalize_memory
        def zero_finalize(state):
            mem, aux = orig_finalize(state)
            return torch.zeros_like(mem), aux
        model.encoder.finalize_memory = zero_finalize
        out_zm = model.compute_sentence_recon_loss(batch)
        acc_zm = compute_accuracy(out_zm)
        model.encoder.finalize_memory = orig_finalize
    delta_loss = float(out_zm["loss_recon"]) - float(out["loss_recon"])
    delta_top1 = (acc_zm["top1"] - acc["top1"]) * 100
    print(f"  loss_recon w/ zeroed memory = {float(out_zm['loss_recon']):.4f} "
          f"(Δ={delta_loss:+.4f})")
    print(f"  top-1 w/ zeroed memory      = {acc_zm['top1']*100:5.1f}% "
          f"(Δ={delta_top1:+.1f}pp)")
    if delta_loss < 0.05:
        print(f"  ⚠ WARN: memory contributes ≤ 0.05 nat — encoder may not be used")
    else:
        print(f"  ✓ memory contributes {delta_loss:.3f} nat to loss reduction")

    del model
    torch.cuda.empty_cache()
    return {
        "variant": variant,
        "loss_recon": float(out["loss_recon"]),
        "loss_recon_zeromem": float(out_zm["loss_recon"]),
        "delta_memory_value": delta_loss,
        "top1": acc["top1"],
        "top5": acc["top5"],
        "top1_zeromem": acc_zm["top1"],
        "mem_diag": mem_d,
    }


def main():
    device = "cuda"
    print("Loading tokenizer + Llama...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    llama, _ = load_frozen_llama("meta-llama/Llama-3.2-1B", dtype=torch.bfloat16)

    cfg = make_cfg()
    dl = make_sentence_chunk_dataloader(
        cfg, fineweb_path=FINEWEB_VAL, tokenizer=tokenizer,
        chunk_size=4096, n_queries=3, mask_ratio=0.8,
        reveal_lo=0.0, reveal_hi=0.9,
        sentence_min_len=8, sentence_max_len=80,
        num_workers=0, seed=999,
    )
    batch = to_device(next(iter(dl)), device)

    results = []
    for variant, label, ckpt_dir, b_div, mt_div in VARIANTS:
        r = run_one_variant(variant, label, ckpt_dir, b_div, mt_div,
                            llama, tokenizer, batch, device)
        results.append(r)

    print(f"\n{'='*78}\n  SUMMARY\n{'='*78}")
    header = f"  {'variant':<22}  {'loss':>8}  {'zeromem':>8}  {'Δ_mem':>8}  {'top1':>6}  {'top5':>6}  {'top1_zm':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['variant']:<22}  "
              f"{r['loss_recon']:>8.3f}  "
              f"{r['loss_recon_zeromem']:>8.3f}  "
              f"{r['delta_memory_value']:>+8.3f}  "
              f"{r['top1']*100:>5.1f}%  "
              f"{r['top5']*100:>5.1f}%  "
              f"{r['top1_zeromem']*100:>7.1f}%")


if __name__ == "__main__":
    main()
