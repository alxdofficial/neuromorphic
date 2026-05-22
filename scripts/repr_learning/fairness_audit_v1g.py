#!/usr/bin/env python3
"""v1g fairness audit — check that no variant has an architectural advantage
or handicap that makes the comparison meaningless.

For each variant, reports:
 - Total trainable params + per-component breakdown
 - Encoder compute pattern:
     * Number of bi_transformer forwards per chunk
     * Tokens per bi_transformer forward (attention span)
     * Number of slot/cross-attention iterations
 - Memory budget:
     * Pre-projection bottleneck floats per chunk
     * Memory tokens exposed to decoder per query
     * Per-query memory floats exposed
     * Per-chunk total memory floats exposed (memory_tokens × n_queries)
 - Aux loss contribution at typical init

The point is to see, side-by-side, whether the comparison is fair.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_sentence import make_sentence_chunk_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
FINEWEB_VAL = REPO / "data/wave1/fineweb_edu.val.parquet"

VARIANTS = ["flat_baseline", "continuous_baseline",
            "memorizing_baseline", "recurrent_baseline", "vanilla_llama"]


def make_cfg():
    return ReprConfig(
        batch_size=2,
        fixed_window_size=1024,
        max_window_size=4096,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=36,
        edge_token_packing="fused",
        b_diversity_scale=50.0,
        mt_diversity_scale=50.0,
        d_mamba=768,                   # mirror v1g training script
    )


def param_breakdown(model) -> dict:
    """Sum trainable params per top-level component."""
    by_component = {}
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0:2]
        key = ".".join(top)
        by_component[key] = by_component.get(key, 0) + p.numel()
        total += p.numel()
    return {"total": total, "by_component": by_component}


def encoder_compute_summary(variant: str, cfg) -> dict:
    """Describe encoder compute pattern per chunk."""
    # 4096-token chunk, window=1024, n_queries=3
    n_chunk_tok = 4096
    n_win_tok = 1024
    n_windows = n_chunk_tok // n_win_tok
    n_queries = 3
    if variant in ("flat_baseline",):
        return {
            "bi_transformer_forwards": n_windows,
            "tokens_per_forward": n_win_tok,
            "attention_span_max": n_win_tok,
            "slot_attn_iters_total": n_windows,  # 1 per write
            "finalize_pass": "code_head → codebook routing → proj_code",
        }
    if variant in ("continuous_baseline",):
        return {
            "bi_transformer_forwards": n_windows,
            "tokens_per_forward": n_win_tok,
            "attention_span_max": n_win_tok,
            "slot_attn_iters_total": n_windows * 2,  # 2 per write
            "finalize_pass": "cont_head → proj_cont",
        }
    if variant in ("memorizing_baseline",):
        # v1g fix: MT now does streaming bi_transformer per 1024-token window,
        # same as A/B. Bank is built incrementally.
        return {
            "bi_transformer_forwards": n_windows,
            "tokens_per_forward": n_win_tok,
            "attention_span_max": n_win_tok,
            "slot_attn_iters_total": 0,
            "finalize_pass": (
                f"concat per-window keys/values into bank; "
                f"per query × {n_queries}: pool from in_proj(raw_embeds) "
                f"+ top-K=36 retrieve + proj_value"
            ),
        }
    if variant in ("recurrent_baseline",):
        return {
            "bi_transformer_forwards": 0,
            "tokens_per_forward": 0,
            "attention_span_max": n_chunk_tok,
            "slot_attn_iters_total": 0,
            "finalize_pass": (
                f"in_proj → Mamba(d_mamba) × {cfg.mamba_n_layers} layers "
                f"over {n_chunk_tok} tokens → bottleneck → adaptive_pool to 36"
            ),
        }
    if variant in ("vanilla_llama",):
        return {
            "bi_transformer_forwards": 0,
            "tokens_per_forward": 0,
            "attention_span_max": 0,
            "slot_attn_iters_total": 0,
            "finalize_pass": "(none — no encoder)",
        }
    return {}


def memory_budget(variant: str, cfg) -> dict:
    """Memory exposed to decoder per query, per chunk."""
    n_queries = 3
    if variant == "flat_baseline":
        M = cfg.n_flat_codes  # 36
        per_slot_floats = cfg.d_concept_baseline  # 725 (pre-projection)
        same_across_queries = True
    elif variant == "continuous_baseline":
        M = cfg.n_flat_codes
        per_slot_floats = cfg.d_continuous
        same_across_queries = True
    elif variant == "memorizing_baseline":
        M = cfg.n_flat_codes  # K=36 retrieved per query
        per_slot_floats = cfg.d_mt_value
        same_across_queries = False  # per-query different retrievals
    elif variant == "recurrent_baseline":
        M = cfg.n_flat_codes
        per_slot_floats = cfg.d_recurrent
        same_across_queries = True
    elif variant == "vanilla_llama":
        M = 0
        per_slot_floats = 0
        same_across_queries = True
    else:
        return {}

    bottleneck_floats = M * per_slot_floats
    per_query_floats = bottleneck_floats  # post-retrieval/projection (same dim)
    if same_across_queries:
        unique_chunk_floats = bottleneck_floats     # one memory used 3 times
    else:
        unique_chunk_floats = n_queries * bottleneck_floats  # different per query

    # MT's bank capacity (pre-retrieval) is much larger
    bank_floats = 0
    if variant == "memorizing_baseline":
        bank_floats = 4096 * cfg.d_mt_value

    return {
        "M_per_query": M,
        "per_slot_floats": per_slot_floats,
        "bottleneck_floats": bottleneck_floats,
        "per_query_memory_floats": per_query_floats,
        "unique_chunk_memory_floats": unique_chunk_floats,
        "same_memory_across_queries": same_across_queries,
        "raw_bank_floats_if_any": bank_floats,
    }


def aux_loss_at_init(model, batch) -> dict:
    """Aux loss raw value, scaled by cfg coefs, at init / current weights."""
    out = model.compute_sentence_recon_loss(batch)
    recon = float(out["loss_recon"])
    aux_raw = float(out["loss_aux"])
    cfg = model.cfg
    return {
        "recon": recon,
        "aux_raw": aux_raw,
        "aux_contribution": cfg.load_balance_coef * aux_raw,
        "aux_pct_of_total": 100 * cfg.load_balance_coef * aux_raw / max(recon + cfg.load_balance_coef * aux_raw, 1e-6),
    }


@torch.no_grad()
def audit_variant(variant, llama, cfg, batch, device):
    print(f"\n{'='*78}\n  {variant}\n{'='*78}")
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    model.train(False)

    # Params
    pb = param_breakdown(model)
    print(f"  TOTAL trainable params : {pb['total']:>12,}")
    print(f"  By top-level component :")
    for k, v in sorted(pb["by_component"].items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"      {k:<30} {v:>12,}")

    # Encoder compute
    ec = encoder_compute_summary(variant, cfg)
    print(f"  ENCODER compute / chunk :")
    print(f"      bi_transformer forwards   : {ec.get('bi_transformer_forwards', 0)}")
    print(f"      tokens per forward        : {ec.get('tokens_per_forward', 0)}")
    print(f"      max attention span        : {ec.get('attention_span_max', 0)}")
    print(f"      slot_attn iters total     : {ec.get('slot_attn_iters_total', 0)}")
    print(f"      finalize                  : {ec.get('finalize_pass', '')}")

    # Memory budget
    mb = memory_budget(variant, cfg)
    print(f"  MEMORY exposed to decoder :")
    print(f"      memory tokens per query     : {mb.get('M_per_query', 0)}")
    print(f"      per-slot dim                : {mb.get('per_slot_floats', 0)}")
    print(f"      per-query memory floats     : {mb.get('per_query_memory_floats', 0):,}")
    print(f"      same memory across queries  : {mb.get('same_memory_across_queries', True)}")
    print(f"      total UNIQUE floats / chunk : {mb.get('unique_chunk_memory_floats', 0):,}")
    if mb.get('raw_bank_floats_if_any', 0) > 0:
        print(f"      raw bank floats (pre-cap)   : {mb['raw_bank_floats_if_any']:,}  ⚠")

    # Aux loss balance
    al = aux_loss_at_init(model, batch)
    print(f"  LOSS at init (no training yet) :")
    print(f"      recon                  : {al['recon']:.3f}")
    print(f"      aux_raw                : {al['aux_raw']:.3f}")
    print(f"      aux contribution       : {al['aux_contribution']:.4f}")
    print(f"      aux as % of total      : {al['aux_pct_of_total']:.1f}%")

    del model
    torch.cuda.empty_cache()
    return {
        "variant": variant,
        "params": pb["total"],
        "M": mb.get("M_per_query", 0),
        "per_query_floats": mb.get("per_query_memory_floats", 0),
        "unique_chunk_floats": mb.get("unique_chunk_memory_floats", 0),
        "bi_xform_forwards": ec.get("bi_transformer_forwards", 0),
        "attn_span": ec.get("attention_span_max", 0),
        "slot_iters": ec.get("slot_attn_iters_total", 0),
        "aux_pct": al["aux_pct_of_total"],
    }


def main():
    device = "cuda"
    cfg = make_cfg()
    print(f"Loading tokenizer + Llama...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    dl = make_sentence_chunk_dataloader(
        cfg, fineweb_path=FINEWEB_VAL, tokenizer=tokenizer,
        chunk_size=4096, n_queries=3, mask_ratio=0.8,
        reveal_lo=0.0, reveal_hi=0.9,
        sentence_min_len=8, sentence_max_len=80,
        num_workers=0, seed=999,
    )
    batch = next(iter(dl))
    for f in ("input_ids", "attention_mask", "query_input_ids",
              "mask_positions", "reveal_positions", "query_lengths",
              "query_starts"):
        setattr(batch, f, getattr(batch, f).to(device))

    rows = []
    for v in VARIANTS:
        rows.append(audit_variant(v, llama, cfg, batch, device))

    print(f"\n{'='*98}")
    print(f"  FAIRNESS SUMMARY TABLE")
    print(f"{'='*98}")
    print(f"  {'variant':<22} {'params':>11} {'M':>4} {'pq_floats':>11} {'ch_floats':>11} "
          f"{'biXfwd':>7} {'attn_span':>10} {'slot_it':>8} {'aux%':>5}")
    print(f"  {'-'*22} {'-'*11} {'-'*4} {'-'*11} {'-'*11} {'-'*7} {'-'*10} {'-'*8} {'-'*5}")
    for r in rows:
        print(f"  {r['variant']:<22} {r['params']:>11,} {r['M']:>4} "
              f"{r['per_query_floats']:>11,} {r['unique_chunk_floats']:>11,} "
              f"{r['bi_xform_forwards']:>7} {r['attn_span']:>10} "
              f"{r['slot_iters']:>8} {r['aux_pct']:>4.1f}%")


if __name__ == "__main__":
    main()
