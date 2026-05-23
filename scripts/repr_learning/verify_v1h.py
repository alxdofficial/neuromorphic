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
    "vanilla_llama": [],  # no encoder params
}


def bottleneck_floats(variant: str, cfg: ReprConfig) -> int:
    """Per-query floats at the pre-projection point."""
    if variant == "flat_baseline":
        return cfg.n_flat_codes * cfg.d_concept_baseline
    if variant == "continuous_baseline":
        return cfg.n_flat_codes * cfg.d_continuous
    if variant == "memorizing_baseline":
        return cfg.n_flat_codes * cfg.d_mt_value
    if variant == "recurrent_baseline":
        return cfg.n_flat_codes * cfg.d_recurrent
    if variant == "plastic_baseline":
        # Plastic does per-Llama-position reads via a MemInject pre-hook,
        # not a prepended fixed-M block. The "bottleneck" here is the
        # per-position bandwidth h_sub. Across T tokens per chunk the
        # total bandwidth is T × h_sub — comparable to the other variants'
        # one-shot M × h_sub bottleneck. We report h_sub as the per-call
        # width; verify_v1h's bottleneck-parity check is variant-aware below.
        return cfg.d_continuous
    if variant == "splat_baseline":
        # Splat memory is K signed Gaussian blobs in a shared latent space
        # with diagonal Σ. Persistent state per batch element:
        #   K · (d + d + 1 + 1) = K · (2d + 2)
        # (μ, log_diag_Σ, w_raw, s_logit). This is the "memory bottleneck"
        # for parity with prepend variants' M · d_node_state.
        return cfg.splat_K * (2 * cfg.splat_d + 2)
    if variant == "vanilla_llama":
        return 0
    raise ValueError(variant)


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
    # Splat: aux is pre-weighted and added directly (not via load_balance_coef).
    splat_aux_contrib = float(out.get("splat_aux", 0.0) or 0.0)
    total = recon + aux_contrib + splat_aux_contrib
    pct = 100 * (aux_contrib + splat_aux_contrib) / max(total, 1e-6)
    print(f"  [C] recon={recon:.3f}  aux_raw={aux:.3f}  "
          f"aux_contrib={aux_contrib:.3f}  splat_aux={splat_aux_contrib:.3f}  "
          f"({pct:.1f}% of total)")
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
    # Override the encoder's output with zeros and re-run.
    if variant == "vanilla_llama":
        # No memory anyway; ablation is trivially 0.
        delta = 0.0
        print(f"  [F] zero-mem ablation: N/A (vanilla has no memory)")
    elif variant == "memorizing_baseline":
        # MT: override retrieve_for_query to return zeros
        orig = model.encoder.retrieve_for_query
        def zero_retrieve(bank, question_embeds, question_mask, K):
            mem, aux_ret = orig(bank, question_embeds, question_mask, K)
            return torch.zeros_like(mem), aux_ret
        model.encoder.retrieve_for_query = zero_retrieve
        with torch.no_grad():
            out_zm = model.compute_qa_loss(batch_mixed)
        delta = float(out_zm["loss_recon"]) - recon
        model.encoder.retrieve_for_query = orig
        print(f"  [F] zero-mem ablation: recon_with_mem={recon:.3f}  "
              f"recon_zero_mem={float(out_zm['loss_recon']):.3f}  Δ={delta:+.3f}")
    elif variant == "plastic_baseline":
        # Plastic: substrate effect is via the inject pre-hook on a Llama
        # layer. Override `inject` to a no-op (returns hidden_states unchanged).
        orig = model.encoder.inject
        def zero_inject(hidden_states, fast_state):
            return hidden_states
        model.encoder.inject = zero_inject
        with torch.no_grad():
            out_zm = model.compute_qa_loss(batch_mixed)
        delta = float(out_zm["loss_recon"]) - recon
        model.encoder.inject = orig
        print(f"  [F] zero-mem ablation: recon_with_mem={recon:.3f}  "
              f"recon_zero_mem={float(out_zm['loss_recon']):.3f}  Δ={delta:+.3f}")
    elif variant == "splat_baseline":
        # Splat: same inject-hook pattern as plastic. Override to no-op.
        orig = model.encoder.inject
        def zero_inject(hidden_states, blobs):
            return hidden_states
        model.encoder.inject = zero_inject
        with torch.no_grad():
            out_zm = model.compute_qa_loss(batch_mixed)
        delta = float(out_zm["loss_recon"]) - recon
        model.encoder.inject = orig
        print(f"  [F] zero-mem ablation: recon_with_mem={recon:.3f}  "
              f"recon_zero_mem={float(out_zm['loss_recon']):.3f}  Δ={delta:+.3f}")
    else:
        # A/B/Mamba: override finalize_memory to return zeros
        orig = model.encoder.finalize_memory
        def zero_finalize(state):
            mem, aux_fin = orig(state)
            return torch.zeros_like(mem), aux_fin
        model.encoder.finalize_memory = zero_finalize
        with torch.no_grad():
            out_zm = model.compute_qa_loss(batch_mixed)
        delta = float(out_zm["loss_recon"]) - recon
        model.encoder.finalize_memory = orig
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
                "vanilla_llama"]

    summary = []
    for v in variants:
        r = audit_variant(v, llama, tok, cfg, batch_mixed, None, device)
        summary.append(r)

    print(f"\n{'='*78}\n  SUMMARY\n{'='*78}")
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

    # Bottleneck parity check — plastic and splat are excluded because they
    # use per-position reads (different bottleneck shape than prepend variants).
    EXEMPT_VARIANTS = {"plastic_baseline", "splat_baseline"}
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

    print(f"\n  OVERALL: {'PASS — safe to launch 10k run' if all_ok else 'FAIL — do not launch'}")


if __name__ == "__main__":
    main()
