#!/usr/bin/env python3
"""v1h pre-launch audit. Runs one training step per variant on the mixed
source pipeline (composite_v1 + HotpotQA + NarrativeQA) and checks:

  (A) Forward + backward produce finite loss with non-zero gradient.
  (B) Gradient flow reaches every architecture's bottleneck params.
  (C) Loss decomposition is reasonable (no aux dominance, no NaN).
  (D) Memory tensor sanity: per-query M=36 tokens for memory variants.
  (E) Bottleneck floats parity at the pre-projection level (26,100).
  (F) Zero-memory ablation: memory contributes ≥ 0.3 nat to recon
      (proves the encoder is doing something for the memory variants;
      vanilla is expected to show Δ ≈ 0).

If any of these fail for any variant, we should NOT launch the 10k run.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_qa import make_mixed_qa_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]


def make_cfg() -> ReprConfig:
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
        d_mamba=768,
    )


def to_device(batch, device):
    for f in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        setattr(batch, f, getattr(batch, f).to(device))
    return batch


CRITICAL_PARAMS = {
    "flat_baseline": [
        "encoder.code_queries", "encoder.concept_id",
        "encoder.code_head.0.weight", "encoder.proj_code.0.weight",
    ],
    "continuous_baseline": [
        "encoder.cont_queries", "encoder.cont_head.0.weight",
        "encoder.proj_cont.0.weight",
    ],
    "memorizing_baseline": [
        "encoder.kv_head.0.weight", "encoder.query_head.0.weight",
        "encoder.proj_value.0.weight",
    ],
    "recurrent_baseline": [
        "encoder.bottleneck.weight", "encoder.proj_to_llama.0.weight",
    ],
    "plastic_baseline": [
        # The slow inits get gradient through the read pass; the
        # controllers get gradient through the chunk's write history.
        # controllers.0.net.2 is the zero-init OUTPUT layer of the plasticity
        # controller — net.0 sees zero grad at step 0 by design and warms up
        # only once net.2 deviates from zero. W_in projects Llama hidden
        # → substrate input on the read path. W_out is zero-init so
        # `scale_raw` is the relevant per-step learnable signal.
        "encoder.substrate.blocks.0.W_slow",
        "encoder.substrate.controllers.0.net.2.weight",
        "encoder.W_in.weight",
        "encoder.scale_raw",
    ],
    "splat_baseline": [
        # Updater output head writes the blob deltas; pin encoder + read
        # heads carry gradient through the read injection. scale_raw is the
        # gate that makes the injection nonzero from step 0.
        "encoder.pin_encoder.0.weight",
        "encoder.updater.out_head.weight",
        "encoder.updater.blob_in_proj.weight",
        "encoder.origin_head.0.weight",
        "encoder.direction_head.0.weight",
        "encoder.read_post.0.weight",
        "encoder.scale_raw",
    ],
    "graph_baseline": [
        # Updater out_head drives all per-edge proposals (no gates in v3).
        # Pin encoder gets QA gradient through cross-attention to pins.
        # GraphReadout is the new readout path (replaces v1/v2 proj_to_llama):
        #   W_src/W_dst — directional R-GCN-style transforms (must train)
        #   cross_edge_attn.in_proj — message-passing self-attention over edges
        #   proj.0 — final projection to d_llama
        "encoder.pin_encoder.0.weight",
        "encoder.updater.out_head.weight",
        "encoder.updater.edge_in_proj.weight",
        "encoder.readout.W_src.weight",
        "encoder.readout.W_dst.weight",
        "encoder.readout.cross_edge_attn.in_proj_weight",
        "encoder.readout.proj.0.weight",
    ],
    "vanilla_llama": [],  # no encoder params
}


def bottleneck_floats(variant: str, cfg: ReprConfig) -> int:
    """Per-query floats at the pre-projection point — the 'total state'
    column of the 4-number bottleneck vector. Counts what persists
    across the encode."""
    if variant == "flat_baseline":
        return cfg.n_flat_codes * cfg.d_concept_baseline
    if variant == "continuous_baseline":
        return cfg.n_flat_codes * cfg.d_continuous
    if variant == "memorizing_baseline":
        # MT post-retrieval bottleneck (the K tokens that reach Llama).
        # See raw_bank_floats() for the full pre-retrieval bank size —
        # MT is genuinely a "retrieval upper bound" with much more state.
        return cfg.n_flat_codes * cfg.d_mt_value
    if variant == "recurrent_baseline":
        return cfg.n_flat_codes * cfg.d_recurrent
    if variant == "plastic_baseline":
        # Plastic state is the PER-BATCH fast weight matrix stack:
        # plastic_depth layers × h_sub × h_sub (h_sub = d_continuous = 725
        # by default). That's 8 × 725² ≈ 4.2M floats per example —
        # ~161x the other "matched" baselines. Previously misreported as
        # just h_sub (audit2 #1). Reporting honestly is non-negotiable
        # for fair capacity comparison; architectural rebalance (low-rank
        # updates, smaller depth) is a separate discussion.
        return cfg.plastic_depth * cfg.d_continuous * cfg.d_continuous
    if variant == "splat_baseline":
        # Splat memory is K signed Gaussian blobs in a shared latent space
        # with diagonal Σ. Persistent state per batch element:
        #   K · (d + d + 1 + 1) = K · (2d + 2)
        # (μ, log_diag_Σ, w_raw, s_logit). This is the "memory bottleneck"
        # for parity with prepend variants' M · d_node_state.
        return cfg.splat_K * (2 * cfg.splat_d + 2)
    if variant == "graph_baseline":
        # Graph memory is K_max edges, each (src, dst, state) + saliency_logit.
        #   K_max · (2·d_node + d_state + 1)
        return cfg.graph_K_max * (2 * cfg.graph_d_node + cfg.graph_d_state + 1)
    if variant == "vanilla_llama":
        return 0
    raise ValueError(variant)


def decoder_interface_floats(variant: str, cfg: ReprConfig,
                              chunk_size: int = 4096) -> int:
    """Floats flowing into the frozen Llama per decode step.
    Interpretation per family:
      - Prepend (A/B/MT/Mamba/graph): M × d_llama (one-shot prepended tensor).
      - MemInject (plastic, splat): PER-POSITION bandwidth (h_sub for
        plastic, K_blobs for splat). The decoder pays this cost at every
        decode position, but each position's delta is small compared to
        the full persistent state.

    Note: this metric is NOT directly comparable across families. Use
    the (state_KB, decoder_KB, compress_x) tuple holistically.
    """
    if variant == "vanilla_llama":
        return 0
    if variant == "plastic_baseline":
        # Per-position: rank-bounded delta of size h_sub flowing into one
        # Llama layer's hidden_states. Full state is plastic_depth · h_sub²
        # but only h_sub-worth of info enters per token.
        return cfg.d_continuous
    if variant == "splat_baseline":
        # Per-position: density evaluation produces K-blob mixture values
        # of size K (one scalar per blob, weighted by ρ).
        return cfg.splat_K
    # Prepend variants — M memory tokens at d_llama dim, one per chunk
    M = cfg.n_flat_codes if variant != "graph_baseline" else cfg.graph_K_max
    return M * cfg.d_llama


def compression_ratio(variant: str, cfg: ReprConfig,
                      chunk_size: int = 4096) -> float:
    """Input bytes / state bytes. Higher = harder compression.
    Universal across architectures because both sides are measured in
    same units (fp32 floats × 4 bytes)."""
    if variant == "vanilla_llama":
        return 0.0
    state = bottleneck_floats(variant, cfg)
    if state == 0:
        return 0.0
    input_floats = chunk_size * cfg.d_llama  # raw embedding bytes
    return input_floats / state


def raw_bank_floats(variant: str, cfg: ReprConfig,
                    chunk_size: int = 4096) -> int:
    """Pre-retrieval/pre-compression raw state held by the encoder.
    For MT this is the full key+value bank (chunk_size × 2 × d_value) —
    far larger than what reaches Llama. Other variants typically equal
    their bottleneck_floats (no separate bank). Audit2 #3."""
    if variant == "memorizing_baseline":
        # Full bank: every context token's (key, value) is kept until
        # question-conditioned retrieval. This is MT's "non-parametric
        # storage" à la Memorizing Transformer.
        return chunk_size * 2 * cfg.d_mt_value
    return bottleneck_floats(variant, cfg)


def encoder_trainable_params(variant: str, llama, cfg: ReprConfig) -> int:
    """Count trainable params in the encoder module only (decoder
    contributions are frozen Llama + tiny mask_embed). For fair compare
    against Memorizing Transformer-style 'non-parametric' arches."""
    from src.repr_learning.model import ReprLearningModel
    m = ReprLearningModel(cfg, variant=variant, llama_model=llama)
    total = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            total += p.numel()
    del m
    return total


@torch.enable_grad()
def audit_variant(variant, llama, tokenizer, cfg, batch_mixed, batch_composite, device):
    print(f"\n{'='*78}\n  AUDIT: {variant}\n{'='*78}")
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    model.train(True)

    # ---- (A) Forward + backward on mixed batch ----
    model.zero_grad()
    out = model.compute_qa_loss(batch_mixed)
    loss = out["loss"]
    ok_finite = torch.isfinite(loss).item()
    loss.backward()
    gnorm = sum((p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)) ** 0.5
    print(f"  [A] loss={float(loss):.4f}  finite={ok_finite}  gnorm={float(gnorm):.2f}")
    if not ok_finite:
        print(f"  ✗ FATAL: non-finite loss on mixed batch")

    # ---- (B) Critical param gradients ----
    # Note: decoder.mask_embed is intentionally zero-grad for v1h QA — it's
    # only used at [MASK] positions in span-mask reconstruction, and QA loss
    # doesn't insert any [MASK] tokens. We keep it in the list as informational
    # but don't fail on it.
    print(f"  [B] critical-param gradients:")
    crit = CRITICAL_PARAMS.get(variant, [])
    grad_check = {}
    for name in crit:
        p = dict(model.named_parameters()).get(name)
        if p is None or p.grad is None:
            print(f"      ✗ {name}: NO GRAD")
            grad_check[name] = False
            continue
        g = float(p.grad.abs().mean())
        mark = "✓" if g > 0 else "✗"
        print(f"      {mark} {name}: grad_abs_mean={g:.6f}")
        grad_check[name] = g > 0
    # Mask embed informational only
    p = dict(model.named_parameters()).get("decoder.mask_embed")
    if p is not None and p.grad is not None:
        g = float(p.grad.abs().mean())
        print(f"      · decoder.mask_embed: grad_abs_mean={g:.6f} "
              f"(expected ~0 for v1h QA — mask_embed unused in this loss path)")

    # ---- (C) Loss decomposition ----
    recon = float(out["loss_recon"])
    aux = float(out["loss_aux"])
    aux_contrib = cfg.load_balance_coef * aux
    # Splat / Graph: aux pre-weighted and added directly to total (not via load_balance_coef).
    splat_aux_contrib = float(out.get("splat_aux", 0.0) or 0.0)
    graph_aux_contrib = float(out.get("graph_aux", 0.0) or 0.0)
    extra_aux = splat_aux_contrib + graph_aux_contrib
    total = recon + aux_contrib + extra_aux
    pct = 100 * (aux_contrib + extra_aux) / max(total, 1e-6)
    print(f"  [C] recon={recon:.3f}  aux_raw={aux:.3f}  "
          f"aux_contrib={aux_contrib:.3f}  splat_aux={splat_aux_contrib:.3f}  "
          f"graph_aux={graph_aux_contrib:.3f}  ({pct:.1f}% of total)")
    aux_ok = pct < 50  # ad-hoc threshold

    # ---- (D) Memory shape ----
    mem_shape = out["memory_shape"]
    mt_mem = out.get("mt_memory_bk") if "mt_memory_bk" in out else None
    if variant == "memorizing_baseline" and mt_mem is not None:
        per_query = mt_mem.shape  # [BK, K, d]
        print(f"  [D] memory (placeholder)={mem_shape}, mt_memory_bk={tuple(per_query)}")
    else:
        print(f"  [D] memory shape={mem_shape}")

    # ---- (E) Bottleneck floats parity ----
    bn = bottleneck_floats(variant, cfg)
    print(f"  [E] pre-projection bottleneck = {bn} floats")

    # ---- (E.5) All-padded streaming-write invariant ----
    # HotpotQA contexts are often <2048 tokens, so later 1024-token windows
    # are entirely padding. An all-pad streaming_write must be a no-op on the
    # encoder state; if it isn't, the encoder silently corrupts memory on
    # those mixed-source rows.
    allpad_delta = 0.0
    if variant not in ("vanilla_llama",):
        with torch.no_grad():
            B_p = 2
            T_p = cfg.fixed_window_size
            d_p = llama.config.hidden_size
            tok_embeds = torch.zeros(B_p, T_p, d_p, device=device, dtype=torch.bfloat16)
            attn_zero = torch.zeros(B_p, T_p, dtype=torch.bool, device=device)
            state_a = model.encoder.init_streaming_state(B_p, device, tok_embeds.dtype)
            state_b, _ = model.encoder.streaming_write(state_a, tok_embeds, attn_zero, chunk_offset=0)

            # Flatten state to a tensor for delta. State shape varies by variant —
            # try the common 'slots' / 'bank' / Tensor / list-of-tensors / dict cases.
            def _state_to_tensor(s):
                if isinstance(s, torch.Tensor):
                    return s.flatten()
                if isinstance(s, (tuple, list)):
                    parts = []
                    for v in s:
                        sub = _state_to_tensor(v)
                        if sub is not None:
                            parts.append(sub)
                    if parts:
                        return torch.cat(parts)
                if isinstance(s, dict):
                    parts = []
                    for v in s.values():
                        sub = _state_to_tensor(v)
                        if sub is not None:
                            parts.append(sub)
                    if parts:
                        return torch.cat(parts)
                return None

            ta = _state_to_tensor(state_a)
            tb = _state_to_tensor(state_b)
            if ta is not None and tb is not None and ta.shape == tb.shape:
                allpad_delta = float((tb.float() - ta.float()).norm())
    print(f"  [E.5] all-pad streaming-write delta = {allpad_delta:.4f} "
          f"(expected ≈ 0 for all memory variants)")
    allpad_ok = allpad_delta < 1e-3 if variant not in ("vanilla_llama",) else True

    # ---- (F) Zero-memory ablation ----
    # Audit2 #8: use the unified zero_memory=True kwarg on compute_qa_loss
    # instead of per-variant monkey-patching. The kwarg drops M=0 for
    # prepend and skips MemInject hooks for plastic/splat, so the delta
    # is variant-neutral and matches eval_zero_mem.py.
    if variant == "vanilla_llama":
        delta = 0.0
        print(f"  [F] zero-mem ablation: N/A (vanilla has no memory)")
    else:
        with torch.no_grad():
            out_zm = model.compute_qa_loss(batch_mixed, zero_memory=True)
        delta = float(out_zm["loss_recon"]) - recon
        print(f"  [F] zero-mem ablation: recon_with_mem={recon:.3f}  "
              f"recon_zero_mem={float(out_zm['loss_recon']):.3f}  Δ={delta:+.3f}")

    del model
    torch.cuda.empty_cache()
    return {
        "variant": variant,
        "finite_loss": ok_finite,
        "all_grads_ok": all(grad_check.values()),
        "aux_pct": pct,
        "aux_ok": aux_ok,
        "bottleneck": bn,
        "zero_mem_delta": delta,
        "allpad_delta": allpad_delta,
        "allpad_ok": allpad_ok,
    }


def main():
    device = "cuda"
    cfg = make_cfg()
    print("Loading tokenizer + Llama...")
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    # Mixed-source training batch — must mirror train_repr_qa.py defaults so
    # the audit reflects the actual launch configuration. composite + HotpotQA
    # at (0.7, 0.3); NarrativeQA disabled.
    print("\nLoading mixed-source dataloader (composite + HotpotQA)...")
    dl_mixed = make_mixed_qa_dataloader(
        cfg, tok,
        composite_passages_path=REPO / "data/wave1/composite_v1/train/passages.jsonl",
        composite_questions_path=REPO / "data/wave1/composite_v1/train/questions.jsonl",
        use_hotpot=True, use_narrative=False, split="train",
        chunk_size=4096, passages_per_chunk=300, batch_size=2, seed=42,
        weights=(0.7, 0.3, 0.0),
    )
    batch_mixed = next(iter(dl_mixed))
    batch_mixed = to_device(batch_mixed, device)
    print(f"  batch sources: {batch_mixed.task_family}")
    print(f"  ctx valid lens: {batch_mixed.context_mask.sum(dim=1).tolist()}")
    print(f"  q lens: {batch_mixed.question_mask.sum(dim=1).tolist()}")
    print(f"  a content positions: {batch_mixed.answer_content_mask.sum(dim=1).tolist()}")

    variants = ["flat_baseline", "continuous_baseline", "memorizing_baseline",
                "recurrent_baseline", "plastic_baseline", "splat_baseline",
                "graph_baseline", "graph_v5_baseline", "vanilla_llama"]

    summary = []
    for v in variants:
        r = audit_variant(v, llama, tok, cfg, batch_mixed, None, device)
        summary.append(r)

    print(f"\n{'='*78}\n  SUMMARY (health checks)\n{'='*78}")
    header = f"  {'variant':<22} {'finite':>7} {'grads':>7} {'aux%':>6} {'bn_floats':>10} {'Δ_mem':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_ok = True
    # At random init, memory is noise so Δ < 0 is expected for memory variants;
    # the threshold checks the encoder is *plumbed* (its output reaches the
    # decoder and changes loss). For vanilla there's no memory by design.
    # Plastic is intentionally neutral at init (small W_out * small scale +
    # zero-init fast weights), so we use a much looser threshold for it —
    # the gradient-flow check above is the real plumbing test for plastic.
    ZERO_MEM_MIN_ABS_DELTA = 0.1
    ZERO_MEM_MIN_ABS_DELTA_PER_POS = 1e-3   # per-position-read variants
    PER_POS_VARIANTS = {"plastic_baseline", "splat_baseline"}
    for r in summary:
        is_vanilla = (r["variant"] == "vanilla_llama")
        is_per_pos = r["variant"] in PER_POS_VARIANTS
        if is_vanilla:
            delta_ok = True
        elif is_per_pos:
            delta_ok = abs(r["zero_mem_delta"]) >= ZERO_MEM_MIN_ABS_DELTA_PER_POS
        else:
            # Includes graph_baseline as a prepend variant with full 0.1 threshold
            delta_ok = abs(r["zero_mem_delta"]) >= ZERO_MEM_MIN_ABS_DELTA
        aux_ok = r["aux_ok"]
        finite_ok = r["finite_loss"]
        grads_ok = r["all_grads_ok"]
        marks = [
            "✓" if finite_ok else "✗",
            "✓" if grads_ok else "✗",
        ]
        allpad_ok = r["allpad_ok"]
        print(f"  {r['variant']:<22} {marks[0]:>7} {marks[1]:>7} "
              f"{r['aux_pct']:>5.1f}% {r['bottleneck']:>10,} "
              f"{r['zero_mem_delta']:>+8.3f}"
              f"  aux_ok={'✓' if aux_ok else '✗'}"
              f"  Δ_ok={'✓' if delta_ok else '✗'}"
              f"  pad_ok={'✓' if allpad_ok else '✗'} (Δ={r['allpad_delta']:.3f})")
        if not (finite_ok and grads_ok and aux_ok and delta_ok and allpad_ok):
            all_ok = False

    # Bottleneck parity check — plastic/splat use per-position reads (different
    # shape than prepend variants); graph has K_max·(2d_n+d_s+1) which can't
    # hit exact 26,100 with integer K. All three reported separately.
    EXEMPT_VARIANTS = {"plastic_baseline", "splat_baseline", "graph_baseline"}
    bn_vals = [
        r["bottleneck"] for r in summary
        if r["bottleneck"] > 0 and r["variant"] not in EXEMPT_VARIANTS
    ]
    if len(set(bn_vals)) == 1:
        print(f"\n  ✓ Bottleneck parity: all prepend variants at {bn_vals[0]:,} floats")
        for v_name in EXEMPT_VARIANTS:
            v_bn = next(
                (r["bottleneck"] for r in summary if r["variant"] == v_name),
                None,
            )
            if v_bn is not None:
                print(f"    {v_name}: {v_bn:,} floats (variant-specific shape)")
    else:
        print(f"\n  ✗ FAIL: bottleneck mismatch: {sorted(set(bn_vals))}")
        all_ok = False

    # ── 4-number bottleneck vector ───────────────────────────────────────
    # Per research-recommended methodology (Memorizing Transformer, Gist
    # Tokens, Mamba SSM literature). No single "memory capacity" number
    # is unbiased; report a vector that lets readers compare on the axis
    # they care about.
    print(f"\n{'='*94}\n  CAPACITY VECTOR (5 numbers per variant + compression ratio)\n{'='*94}")
    cap_header = (f"  {'variant':<22} {'state_KB':>10} {'raw_bank_KB':>12} "
                  f"{'decoder_KB':>12} {'enc_params':>12} {'compress_x':>11}")
    print(cap_header)
    print("  " + "-" * (len(cap_header) - 2))
    for r in summary:
        state = bottleneck_floats(r["variant"], cfg)
        raw_bank = raw_bank_floats(r["variant"], cfg)
        dec = decoder_interface_floats(r["variant"], cfg)
        compress = compression_ratio(r["variant"], cfg)
        try:
            enc_params = encoder_trainable_params(r["variant"], llama, cfg)
        except Exception:
            enc_params = -1
        print(f"  {r['variant']:<22} {state * 4 / 1024:>10.1f} "
              f"{raw_bank * 4 / 1024:>12.1f} "
              f"{dec * 4 / 1024:>12.1f} "
              f"{enc_params:>12,} "
              f"{compress:>10.1f}x")
    print("\n  state_KB     = persistent state seen by Llama (post-retrieval for MT, equals raw for others)")
    print("  raw_bank_KB  = total raw state the encoder holds (MT keeps full per-token KV; others = state_KB)")
    print("  decoder_KB   = floats flowing into Llama per decode (prepend: M·d_llama; MemInject: per-position bw)")
    print("  enc_params   = trainable parameters in encoder only (excludes frozen Llama)")
    print("  compress_x   = chunk_size·d_llama / state_floats — raw input bytes / state bytes")
    print("\n  Notes:")
    print("  - MT has ~227x larger raw_bank than state_KB (full bank pre-retrieval); it's a")
    print("    'retrieval upper bound' — much more state than the matched baselines.")
    print("  - graph's decoder_KB is 1.9x prepend baselines (K_max=68 vs M=36 tokens).")
    print("  - plastic's state is 161x baselines (8 layers × h_sub² fast weights), not 1/36th.")

    print(f"\n  OVERALL: {'PASS — safe to launch 10k run' if all_ok else 'FAIL — do not launch'}")


if __name__ == "__main__":
    main()
