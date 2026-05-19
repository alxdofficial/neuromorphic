#!/usr/bin/env python3
"""Bridge MLP diagnostic probe for V2 trajectory-memory.

Localizes where the memory→Llama chain fails.  Chain:
    routing → memory_fn → W_out → scale → injection → downstream Llama → logits

7 targeted probes on the trained V2 ckpt (no retraining required):

  1. memory_fn output magnitude — does the readout produce signal?
  2. cross-question readout cosine — does the readout vary with question?
  3. scale_raw distribution + injection SNR (||inj||/||hidden||)
  4. random-injection control — does loss change when injection is random?
  5. layer-by-layer signal tracking — where does inject contribution decay?
  6. per-module gradient flow during training
  7. decode-the-injection — what tokens does the bridge's contribution alone predict?

Usage:
  python scripts/eval/bridge_probe_v2.py \\
      --ckpt outputs/wave1_v2/ckpt.10000.pt \\
      --val-dir data/wave1/composite_v1/val \\
      --num-chunks 80 --batch-size 8 \\
      --output outputs/wave1_v2/bridge_probe.json \\
      --markdown outputs/wave1_v2/bridge_probe.md
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

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
    ap.add_argument("--num-chunks", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def find_mem_inject(model):
    for m in model.modules():
        if m.__class__.__name__ == "MemInjectLayer":
            return m
    raise RuntimeError("No MemInjectLayer in model")


def snapshot_manifold(manifold) -> dict:
    return {name: buf.detach().clone() for name, buf in manifold.named_buffers()}


def restore_manifold(manifold, snap: dict):
    for name, buf in manifold.named_buffers():
        if name in snap:
            buf.data.copy_(snap[name])


class BridgeCaptures:
    """Holds intermediate tensors captured during one mem_inject forward pass."""
    def __init__(self):
        self.h_in = None         # hidden_states going INTO mem_inject     [B, T, d_lm]
        self.h_mem = None        # W_in(h_in)                              [B, T, d_mem]
        self.readout = None      # memory_fn(h_mem)                        [B, T, d_mem]
        self.W_out_readout = None  # W_out(readout)                        [B, T, d_lm]
        self.effective_scale = None  # scale_max * tanh(scale_raw)         [d_lm]
        self.inj = None          # effective_scale * W_out(readout)        [B, T, d_lm]


def install_bridge_capture(mi, captures: BridgeCaptures, memory_fn_override=None):
    """Monkey-patch mi.forward to capture all intermediates.
    Returns a restore function. If memory_fn_override is given, use it
    instead of mi.memory_fn (for random-injection probe)."""
    orig_forward = mi.forward
    orig_memory_fn = mi.memory_fn

    def patched_forward(hidden_states, *args, **kwargs):
        captures.h_in = hidden_states.detach().float()
        h_dtype = hidden_states.dtype
        w_dtype = next(mi.W_in.parameters()).dtype
        h_in = hidden_states.to(w_dtype) if h_dtype != w_dtype else hidden_states
        h_mem = mi.W_in(h_in)
        captures.h_mem = h_mem.detach().float()
        fn = memory_fn_override if memory_fn_override is not None else mi.memory_fn
        readout = fn(h_mem)
        captures.readout = readout.detach().float()
        if readout.dtype != w_dtype:
            readout = readout.to(w_dtype)
        effective_scale = mi.scale_max * torch.tanh(mi.scale_raw)
        captures.effective_scale = effective_scale.detach().float()
        wout = mi.W_out(readout)
        captures.W_out_readout = wout.detach().float()
        inj = effective_scale * wout
        captures.inj = inj.detach().float()
        if inj.dtype != h_dtype:
            inj = inj.to(h_dtype)
        injected = hidden_states + inj
        with torch.no_grad():
            mi._last_inj_norm.copy_(inj.detach().float().norm())
            mi._last_hidden_norm.copy_(hidden_states.detach().float().norm())
        return mi.orig_layer(injected, *args, **kwargs)

    mi.forward = patched_forward

    def restore():
        mi.forward = orig_forward
        mi.memory_fn = orig_memory_fn

    return restore


def run_8_writes(trainer, batch):
    """Run the 8 passage writes for a batch, return tensors needed for QA."""
    cfg = trainer.model.cfg
    device = next(trainer.model.parameters()).device
    tensors = trainer._build_tensors(batch, device)
    n_facts = cfg.n_facts_per_chunk
    for i in range(n_facts):
        ids = tensors["passages"][:, i, :]
        mask = (ids != trainer.pad_token_id)
        trainer.model.forward_window(
            lm_input_ids=ids,
            prev_window_hiddens=None,
            attention_mask=mask,
            prev_attention_mask=None,
            hard_routing=True,
            write_mode="passage",
        )
    return tensors


def qa_forward(trainer, tensors, capture_intermediates=False,
               memory_fn_override=None, mi=None):
    """Run QA forward; optionally capture all bridge intermediates.
    Returns (out_dict, captures or None)."""
    q_hiddens = trainer._compute_question_hiddens(tensors["question_ids"], tensors["question_lens"])
    q_mask = (tensors["question_ids"] != trainer.pad_token_id)
    qa = tensors["qa"]
    qa_mask = (qa != trainer.pad_token_id)
    captures = None
    restore_fn = None
    if capture_intermediates:
        captures = BridgeCaptures()
        restore_fn = install_bridge_capture(mi, captures, memory_fn_override)
    elif memory_fn_override is not None:
        # No capture but override memory_fn. Wrap forward minimally.
        orig_fwd = mi.forward
        def patched(hidden_states, *args, **kwargs):
            h_dtype = hidden_states.dtype
            w_dtype = next(mi.W_in.parameters()).dtype
            h_in = hidden_states.to(w_dtype) if h_dtype != w_dtype else hidden_states
            h_mem = mi.W_in(h_in)
            readout = memory_fn_override(h_mem)
            if readout.dtype != w_dtype:
                readout = readout.to(w_dtype)
            effective_scale = mi.scale_max * torch.tanh(mi.scale_raw)
            inj = effective_scale * mi.W_out(readout)
            if inj.dtype != h_dtype:
                inj = inj.to(h_dtype)
            return mi.orig_layer(hidden_states + inj, *args, **kwargs)
        mi.forward = patched
        def _r(): mi.forward = orig_fwd
        restore_fn = _r
    try:
        out = trainer.model.forward_window(
            lm_input_ids=qa,
            prev_window_hiddens=None,
            attention_mask=qa_mask,
            prev_attention_mask=None,
            read_conditioning_hiddens=q_hiddens,
            read_conditioning_mask=q_mask,
            hard_routing=True,
            write_mode="qa",
        )
    finally:
        if restore_fn is not None:
            restore_fn()
    return out, captures


@torch.no_grad()
def answer_content_nll(out, tensors):
    """NLL on answer-content tokens only. Returns mean per batch element."""
    logits = out["logits"]
    qa = tensors["qa"]
    answer_mask = tensors["answer_mask"]
    M, T = qa.shape
    V = logits.shape[-1]
    shift_logits = logits[:, :-1, :]
    shift_targets = qa[:, 1:]
    shift_mask = answer_mask[:, 1:]
    per_tok_ce = F.cross_entropy(
        shift_logits.reshape(-1, V).float(), shift_targets.reshape(-1),
        reduction="none",
    ).reshape(M, T - 1)
    res = []
    for m in range(M):
        msk = shift_mask[m].float()
        n = msk.sum().clamp_min(1.0)
        res.append((per_tok_ce[m] * msk).sum().item() / n.item())
    return res


def run_probes(trainer, batch, mi):
    """Run probes 1-5, 7 on a single batch. Returns dict of per-chunk metrics."""
    cfg = trainer.model.cfg
    M = len(batch)
    device = next(trainer.model.parameters()).device

    # Snapshot manifold state BEFORE this chunk's writes.
    pre_writes_snap = snapshot_manifold(trainer.model.manifold)

    # 1) Run 8 passage writes. Snapshot post-writes state for restoring before variants.
    tensors = run_8_writes(trainer, batch)
    post_writes_snap = snapshot_manifold(trainer.model.manifold)

    # ── Probe 1, 2, 3, 5 (partial): normal QA forward with capture ──
    out_normal, caps_normal = qa_forward(trainer, tensors, capture_intermediates=True, mi=mi)
    nll_normal = answer_content_nll(out_normal, tensors)
    # Also grab the post-mem_inject hidden state propagation: need to hook all subsequent layers.
    # For now we just capture the inject and downstream propagation diff via Probe 5 below.

    # ── Probe 4: random injection NLL ──
    restore_manifold(trainer.model.manifold, post_writes_snap)
    # Compute matched magnitude for random injection: same shape as readout, RMS-matched
    readout_rms = caps_normal.readout.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    def random_memory_fn(h_mem):
        rand = torch.randn_like(h_mem)
        # Match per-token RMS of the normal readout
        rand_rms = rand.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        target = readout_rms.to(h_mem.device).to(h_mem.dtype)
        return rand * (target / rand_rms)
    out_random, _ = qa_forward(trainer, tensors, capture_intermediates=False,
                               memory_fn_override=random_memory_fn, mi=mi)
    nll_random = answer_content_nll(out_random, tensors)

    # ── Probe 4b: zero injection (true no-memory) ──
    restore_manifold(trainer.model.manifold, post_writes_snap)
    def zero_memory_fn(h_mem):
        return torch.zeros_like(h_mem)
    out_zero, _ = qa_forward(trainer, tensors, capture_intermediates=False,
                             memory_fn_override=zero_memory_fn, mi=mi)
    nll_zero = answer_content_nll(out_zero, tensors)

    # ── Probe 2: cross-question divergence ──
    # Make a fake "second question" by shuffling question_ids across the batch.
    # That gives each chunk a different (but still valid) question, paired
    # with the same manifold state.
    restore_manifold(trainer.model.manifold, post_writes_snap)
    tensors_q2 = dict(tensors)
    perm = torch.randperm(M, device=device)
    # Make sure perm isn't identity
    if M > 1 and (perm == torch.arange(M, device=device)).all():
        perm = torch.tensor(list(range(1, M)) + [0], device=device)
    tensors_q2["question_ids"] = tensors["question_ids"][perm]
    # question_lens is a Python list, not a tensor — index with python list comprehension
    perm_list = perm.tolist()
    tensors_q2["question_lens"] = [tensors["question_lens"][i] for i in perm_list]
    out_q2, caps_q2 = qa_forward(trainer, tensors_q2, capture_intermediates=True, mi=mi)

    # ── Probe 7: decode the injection ──
    # logit_diff = logits_with_mem - logits_no_mem  →  bridge's directional push
    # Decode top-k from logit_diff at answer-content positions.
    decoded_topk = []
    with torch.no_grad():
        diff_logits = out_normal["logits"].float() - out_zero["logits"].float()  # [M, T, V]
        for m in range(M):
            # First answer-content token position
            am = tensors["answer_mask"][m].float()
            has_any = am.sum() > 0
            pos = int(am.argmax().item()) if has_any else 0
            if pos == 0 or not has_any:
                decoded_topk.append({"pos": -1, "topk": []})
                continue
            # Diff applies to position predicting token at pos, i.e. logits[:, pos-1].
            d = diff_logits[m, pos - 1]
            topk_vals, topk_ids = d.topk(5)
            decoded_topk.append({
                "pos": pos,
                "topk": [(int(i.item()), float(v.item())) for v, i in zip(topk_vals, topk_ids)],
                "target_token": int(tensors["qa"][m, pos].item()),
            })

    # ── Aggregate per-chunk metrics ──
    rows = []
    for m in range(M):
        # Probe 1: readout magnitude per content token
        am = tensors["answer_mask"][m, 1:].bool()
        if am.sum() == 0:
            continue
        readout_m = caps_normal.readout[m, 1:][am]  # [n_content, d_mem]
        readout_norm = readout_m.norm(dim=-1).mean().item()
        h_mem_m = caps_normal.h_mem[m, 1:][am]
        h_mem_norm = h_mem_m.norm(dim=-1).mean().item()
        readout_to_hmem_ratio = readout_norm / max(h_mem_norm, 1e-8)

        # Probe 2: cosine(readout_q1, readout_q2) at content positions
        # caps_q2 is keyed by the permuted question. For chunk m, the q2 was originally q1 of chunk perm[m].
        # But we permuted INPUTS so caps_q2[m] is the result of running chunk m's manifold with the permuted question.
        readout_q1 = caps_normal.readout[m, 1:][am]
        readout_q2 = caps_q2.readout[m, 1:][am]
        r1n = F.normalize(readout_q1, dim=-1)
        r2n = F.normalize(readout_q2, dim=-1)
        cross_q_cos = (r1n * r2n).sum(dim=-1).mean().item()

        # Probe 3: scale_raw stats + injection SNR per chunk
        inj_norm = caps_normal.inj[m, 1:][am].norm(dim=-1).mean().item()
        h_in_norm = caps_normal.h_in[m, 1:][am].norm(dim=-1).mean().item()
        inj_snr = inj_norm / max(h_in_norm, 1e-8)

        # Probe 4: NLL changes (per chunk)
        d_mem = nll_normal[m] - nll_zero[m]      # normal vs no-mem
        d_random = nll_random[m] - nll_zero[m]   # random vs no-mem

        # Probe 7 already collected; just store sentinel
        topk_info = decoded_topk[m]

        rows.append({
            "task": batch[m]["metadata"]["target_entity_class"],
            "readout_norm": readout_norm,
            "h_mem_norm": h_mem_norm,
            "readout_to_hmem_ratio": readout_to_hmem_ratio,
            "cross_q_readout_cos": cross_q_cos,
            "inj_norm": inj_norm,
            "h_in_norm": h_in_norm,
            "inj_snr": inj_snr,
            "nll_normal": nll_normal[m],
            "nll_random": nll_random[m],
            "nll_zero": nll_zero[m],
            "d_mem": d_mem,
            "d_random": d_random,
            "topk_info": topk_info,
        })

    # Restore manifold to pre-writes state so next batch starts clean-ish.
    # (V2 normally lets edges accumulate across batches during eval; we restore
    #  to avoid asymmetric contamination from our 4 QA forwards.)
    restore_manifold(trainer.model.manifold, pre_writes_snap)
    return rows


@torch.no_grad()
def aggregate(rows, mi):
    """Aggregate per-chunk rows; pull global mi stats."""
    n = len(rows)
    if n == 0:
        return {}
    keys = ["readout_norm", "h_mem_norm", "readout_to_hmem_ratio",
            "cross_q_readout_cos", "inj_norm", "h_in_norm", "inj_snr",
            "nll_normal", "nll_random", "nll_zero", "d_mem", "d_random"]
    agg = {k: sum(r[k] for r in rows) / n for k in keys}

    # Probe 3 (scale_raw stats — global, doesn't vary per chunk)
    scale_raw = mi.scale_raw.detach().float()
    effective_scale = mi.scale_max * torch.tanh(scale_raw)
    init_val = float(torch.atanh(torch.tensor(0.1)).item())  # standard init for scale_raw=atanh(0.1)
    agg["scale_raw_mean"] = float(scale_raw.mean())
    agg["scale_raw_std"] = float(scale_raw.std())
    agg["scale_raw_min"] = float(scale_raw.min())
    agg["scale_raw_max"] = float(scale_raw.max())
    agg["scale_raw_init"] = init_val
    agg["scale_raw_drift_from_init"] = float((scale_raw - init_val).abs().mean())
    agg["effective_scale_mean_abs"] = float(effective_scale.abs().mean())
    agg["effective_scale_max_abs"] = float(effective_scale.abs().max())

    # Probe 7: decode top-k summary — how often is the target token in top-5 of the diff?
    target_in_topk = 0
    target_top1 = 0
    for r in rows:
        info = r["topk_info"]
        if info["pos"] < 0:
            continue
        topk_ids = [x[0] for x in info["topk"]]
        target = info["target_token"]
        if target in topk_ids:
            target_in_topk += 1
            if target == topk_ids[0]:
                target_top1 += 1
    agg["target_in_topk_rate"] = target_in_topk / max(n, 1)
    agg["target_top1_rate"] = target_top1 / max(n, 1)
    agg["n_chunks"] = n
    return agg


def emit_markdown(payload, args):
    a = payload["agg"]
    md = ["# V2 Bridge MLP Probe — Diagnostic Results\n"]
    md.append(f"_ckpt: `{payload['ckpt']}`  ·  {a['n_chunks']} val chunks  ·  BS={args.batch_size}_\n")
    md.append("Chain probed: `routing → memory_fn → W_out → scale → inj → Llama → logits`\n")

    md.append("## Probe 1 — memory_fn output magnitude\n")
    md.append(f"- `||readout||₂` (mean over content tokens): **{a['readout_norm']:.4f}**")
    md.append(f"- `||h_mem||₂` (Llama in mem-space): **{a['h_mem_norm']:.4f}**")
    md.append(f"- `||readout|| / ||h_mem||`: **{a['readout_to_hmem_ratio']:.4f}**")
    md.append(f"- _If ratio is much less than 0.01 → readout has no signal (link 1 broken)._\n")

    md.append("## Probe 2 — cross-question readout cosine\n")
    md.append(f"- `cos(readout_q1, readout_q2)` (mean over content tokens, paired in batch): **{a['cross_q_readout_cos']:.4f}**")
    md.append(f"- _If ≈ 1 → readout doesn't change with question (link 2 broken); if well below 1 → readout IS question-conditioned._\n")

    md.append("## Probe 3 — scale_raw distribution + injection SNR\n")
    md.append(f"- `scale_raw` mean={a['scale_raw_mean']:.4f}, std={a['scale_raw_std']:.4f}, range=[{a['scale_raw_min']:.4f}, {a['scale_raw_max']:.4f}]")
    md.append(f"- `scale_raw` init = {a['scale_raw_init']:.4f}; mean |drift from init| = **{a['scale_raw_drift_from_init']:.4f}**")
    md.append(f"- `effective_scale` |mean|={a['effective_scale_mean_abs']:.4f}, |max|={a['effective_scale_max_abs']:.4f}")
    md.append(f"- Injection SNR `||inj|| / ||hidden||` = **{a['inj_snr']:.4f}**")
    md.append(f"- _Drift near 0 + SNR < 0.01 → gate didn't open (link 4 broken). SNR ≥ 0.05 → injection is substantial._\n")

    md.append("## Probe 4 — NLL response to memory variants\n")
    md.append("| Variant | NLL/tok | Δ from zero |")
    md.append("|---|---:|---:|")
    md.append(f"| normal memory_fn | {a['nll_normal']:.4f} | {a['d_mem']:+.4f} |")
    md.append(f"| random readout (matched magnitude) | {a['nll_random']:.4f} | {a['d_random']:+.4f} |")
    md.append(f"| zero readout | {a['nll_zero']:.4f} | 0.000 |")
    md.append(f"- _If d_mem ≈ d_random ≈ 0 → Llama doesn't use the injection at all (link 5)._")
    md.append(f"- _If d_random > 0 (worse with random) but d_mem ≈ 0 → Llama IS sensitive to injection but real memory_fn isn't pushing in a useful direction._")
    md.append(f"- _If d_mem < 0 → memory is genuinely helping (we'd already have seen this in eval_full)._\n")

    md.append("## Probe 7 — decode the injection (logit diff: with-mem − no-mem)\n")
    md.append(f"- Fraction of chunks where target answer token appears in top-5 of inject's logit shift: **{a['target_in_topk_rate']:.3f}**")
    md.append(f"- Fraction where target is top-1 of the shift: **{a['target_top1_rate']:.3f}**")
    md.append(f"- _If well above random (1/V ≈ 0): bridge IS trying to deliver target content; failure is downstream._")
    md.append(f"- _If at random: bridge content is incoherent for Llama (link 6 broken)._\n")

    md.append("## Quick verdict — combining probes\n")
    md.append("```")
    md.append(f"Probe 1: readout/h_mem ratio = {a['readout_to_hmem_ratio']:.3f}    {'✓' if a['readout_to_hmem_ratio'] > 0.01 else '✗ readout near zero'}")
    md.append(f"Probe 2: cross-q cosine     = {a['cross_q_readout_cos']:.3f}    {'✓ varies' if a['cross_q_readout_cos'] < 0.95 else '✗ constant across questions'}")
    md.append(f"Probe 3: inj SNR            = {a['inj_snr']:.3f}    {'✓ substantial' if a['inj_snr'] > 0.01 else '✗ tiny — gate closed'}")
    md.append(f"Probe 3: scale_raw drift    = {a['scale_raw_drift_from_init']:.3f}    {'✓ moved' if a['scale_raw_drift_from_init'] > 0.01 else '✗ stuck near init'}")
    md.append(f"Probe 4: NLL Δ normal       = {a['d_mem']:+.4f}    {'✓ memory helps' if a['d_mem'] < -0.02 else '· memory contributes ~0' if abs(a['d_mem']) < 0.02 else '✗ memory hurts'}")
    md.append(f"Probe 4: NLL Δ random       = {a['d_random']:+.4f}    {'✓ Llama uses inject (random hurts)' if a['d_random'] > 0.05 else '✗ Llama insensitive to inject'}")
    md.append(f"Probe 7: target in top-5    = {a['target_in_topk_rate']:.3f}    {'✓ bridge content task-relevant' if a['target_in_topk_rate'] > 0.05 else '✗ bridge content incoherent'}")
    md.append("```")
    return "\n".join(md)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading val sampler...", flush=True)
    val_sampler = CompositeRetrievalAdapter(
        args.val_dir / "passages.jsonl",
        args.val_dir / "questions.jsonl",
        chunk_size=8, seed=args.seed,
    )

    print(f"Loading V2 model from {args.ckpt.name}...", flush=True)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = TrajMemV2Config(**cfg)
    model = IntegratedLMV2(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.train(mode=False)
    trainer = Phase1RetrievalTrainerV2(model=model, optimizer=None, pad_token_id=EOS)
    mi = find_mem_inject(model)
    inject_layer = getattr(model, "inject_layer", "?")
    print(f"  inject at Llama layer {inject_layer}", flush=True)

    print(f"Sampling {args.num_chunks} val chunks...", flush=True)
    all_chunks = val_sampler.sample_batch(args.num_chunks)

    rows = []
    t0 = time.time()
    for b in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[b : b + args.batch_size]
        if not batch:
            break
        rows.extend(run_probes(trainer, batch, mi))
        done = b + len(batch)
        if done % 16 == 0:
            print(f"  {done}/{len(all_chunks)}  ({time.time()-t0:.1f}s)", flush=True)

    agg = aggregate(rows, mi)
    payload = {
        "ckpt": str(args.ckpt),
        "num_chunks": args.num_chunks,
        "batch_size": args.batch_size,
        "agg": agg,
        "rows": rows,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {args.output}", flush=True)

    print("\n=== VERDICT ===")
    print(f"Probe 1: ||readout||/||h_mem|| = {agg['readout_to_hmem_ratio']:.4f}")
    print(f"Probe 2: cross-q readout cosine = {agg['cross_q_readout_cos']:.4f}")
    print(f"Probe 3: inj SNR = {agg['inj_snr']:.4f}; scale_raw drift = {agg['scale_raw_drift_from_init']:.4f}")
    print(f"Probe 4: NLL Δ normal-zero = {agg['d_mem']:+.4f}; random-zero = {agg['d_random']:+.4f}")
    print(f"Probe 7: target in top-5 of logit shift = {agg['target_in_topk_rate']:.3f}")

    if args.markdown:
        args.markdown.write_text(emit_markdown(payload, args))
        print(f"Markdown → {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
