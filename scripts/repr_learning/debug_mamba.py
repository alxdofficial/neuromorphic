#!/usr/bin/env python3
"""Debug Mamba's anomalously low recon CE.

Hypothesis: Mamba's adaptive_avg_pool(256 → 96) creates positionally-aligned
memory tokens where memory[p] ≈ summary of input tokens near position
p * (256/96). Llama can then read off masked tokens via positional alignment.

Tests:
  1. Normal val recon (baseline)
  2. Shuffled memory tokens — permute the 96 memory slots randomly
     If positional alignment matters, this should catastrophically hurt.
  3. Zero memory tokens — no info from encoder at all
     Should give the "no memory" floor.
  4. Memory-token vs embed-of-original cosine — for each memory token at
     position p, compute cosine similarity with the average of Llama's
     embeddings of original tokens in the input range [p*2.67, (p+1)*2.67].
     High similarity → Mamba is effectively pass-through.

Run on a single variant's checkpoint at a time. Compare across variants.

Usage:
    python scripts/repr_learning/debug_mamba.py \\
        --variant recurrent_baseline \\
        --ckpt outputs/repr_learning/v0_mamba/ckpts/recurrent_baseline.last.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel
from scripts.eval.eval_repr import load_labeled_val, batch_from_samples


@torch.no_grad()
def normal_recon(model, samples, device, n_batches: int = 25, mask_ratio: float = 0.7) -> float:
    """Baseline val recon CE."""
    losses = []
    for i in range(0, min(len(samples), n_batches * 2), 2):
        batch = samples[i:i + 2]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, mask_ratio, device, seed=42 + i)
        out = model(input_ids, attn_mask, mask_pos)
        losses.append(float(out["loss_recon"].item() if isinstance(out["loss_recon"], torch.Tensor) else out["loss_recon"]))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def shuffled_memory_recon(model, samples, device, n_batches: int = 25,
                          mask_ratio: float = 0.7) -> float:
    """Recon CE with memory tokens randomly permuted (within each sample).

    If memory is positionally-aligned, this should hurt a LOT. If memory
    is positionally-agnostic (slot-attention-like), this should be ~no-op.
    """
    losses = []
    for i in range(0, min(len(samples), n_batches * 2), 2):
        batch = samples[i:i + 2]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, mask_ratio, device, seed=42 + i)
        # First, run encoder to get memory
        with torch.no_grad():
            embed = model.decoder.llama.get_input_embeddings()
            token_embeds = embed(input_ids)
        memory, aux = model.encoder(token_embeds, attn_mask, mask_positions=mask_pos)
        # Permute memory slots within each sample
        B, M, D = memory.shape
        perms = torch.stack([torch.randperm(M, device=memory.device) for _ in range(B)])
        memory_shuffled = torch.gather(
            memory, 1, perms.unsqueeze(-1).expand(-1, -1, D),
        )
        # Run decoder directly with the shuffled memory
        _, loss = model.decoder(
            input_ids, mask_pos, memory_shuffled,
            attention_mask=attn_mask, token_embeds=token_embeds,
        )
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def zero_memory_recon(model, samples, device, n_batches: int = 25,
                      mask_ratio: float = 0.7) -> float:
    """Recon CE with all-zero memory tokens. Equivalent to vanilla Llama."""
    losses = []
    for i in range(0, min(len(samples), n_batches * 2), 2):
        batch = samples[i:i + 2]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, mask_ratio, device, seed=42 + i)
        with torch.no_grad():
            embed = model.decoder.llama.get_input_embeddings()
            token_embeds = embed(input_ids)
        memory, _ = model.encoder(token_embeds, attn_mask, mask_positions=mask_pos)
        memory_zero = torch.zeros_like(memory)
        _, loss = model.decoder(
            input_ids, mask_pos, memory_zero,
            attention_mask=attn_mask, token_embeds=token_embeds,
        )
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def memory_embed_similarity(model, samples, device, n_samples: int = 16) -> dict:
    """For each memory token at position p, compute cosine similarity with
    the average of Llama's embeddings of original tokens in the corresponding
    input range [p*256/96, (p+1)*256/96].

    Returns: {mean_cos: float, p25, p50, p75 over (sample, position) pairs}.
    """
    sub = samples[:n_samples]
    all_cos = []
    embed = model.decoder.llama.get_input_embeddings()
    for i in range(0, len(sub), 2):
        batch = sub[i:i + 2]
        input_ids, attn_mask, mask_pos = batch_from_samples(batch, 0.7, device, seed=99 + i)
        with torch.no_grad():
            token_embeds = embed(input_ids)
        memory, _ = model.encoder(token_embeds, attn_mask, mask_positions=mask_pos)
        # Compute target: avg of original token embeddings in each window
        B, T, D = token_embeds.shape
        M = memory.shape[1]
        # Adaptive avg pool the embeddings the same way
        emb_t = token_embeds.transpose(1, 2)                # [B, D, T]
        pooled_emb = F.adaptive_avg_pool1d(emb_t.float(), M) # [B, D, M]
        pooled_emb = pooled_emb.transpose(1, 2)              # [B, M, D]

        m_norm = F.normalize(memory.float(), dim=-1)
        e_norm = F.normalize(pooled_emb.float(), dim=-1)
        cos = (m_norm * e_norm).sum(dim=-1)                  # [B, M]
        all_cos.append(cos.flatten().cpu())
    flat = torch.cat(all_cos)
    return {
        "mean_cos": flat.mean().item(),
        "p25": flat.quantile(0.25).item(),
        "p50": flat.median().item(),
        "p75": flat.quantile(0.75).item(),
        "n_points": flat.numel(),
    }


def load_model(variant: str, ckpt_path: Path, llama, cfg: ReprConfig, device: str):
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"[load] step {ckpt.get('step', '?')}")
    model.train(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-batches", type=int, default=25)
    ap.add_argument("--mask-ratio", type=float, default=0.7)
    args = ap.parse_args()

    cfg = ReprConfig(batch_size=2, fixed_window_size=256)
    device = "cuda"

    print(f"Variant: {args.variant}")
    print(f"Loading Llama...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)
    model = load_model(args.variant, args.ckpt, llama, cfg, device)

    print(f"Loading val samples...")
    samples = load_labeled_val(cfg, max_fineweb=100, max_composite_per_task=20)
    print(f"  {len(samples)} samples")

    print(f"\nRunning tests (mask_ratio={args.mask_ratio}, n_batches={args.n_batches})...")
    print(f"  [1/4] normal recon ...")
    normal = normal_recon(model, samples, device, args.n_batches, args.mask_ratio)
    torch.cuda.empty_cache()
    print(f"        normal recon CE: {normal:.4f}")

    print(f"  [2/4] shuffled memory recon ...")
    shuffled = shuffled_memory_recon(model, samples, device, args.n_batches, args.mask_ratio)
    torch.cuda.empty_cache()
    print(f"        shuffled recon CE: {shuffled:.4f}")

    print(f"  [3/4] zero memory recon ...")
    zeros = zero_memory_recon(model, samples, device, args.n_batches, args.mask_ratio)
    torch.cuda.empty_cache()
    print(f"        zero recon CE: {zeros:.4f}")

    print(f"  [4/4] memory ↔ pooled-embed cosine similarity ...")
    sim = memory_embed_similarity(model, samples, device, n_samples=16)
    torch.cuda.empty_cache()
    print(f"        mean cos: {sim['mean_cos']:.4f}  median: {sim['p50']:.4f}  "
          f"p25/p75: {sim['p25']:.4f}/{sim['p75']:.4f}")

    print(f"\n{'='*70}")
    print(f"Diagnosis:")
    print(f"  normal recon CE   : {normal:.4f}")
    print(f"  shuffled recon CE : {shuffled:.4f}   (Δ {shuffled-normal:+.3f})")
    print(f"  zero recon CE     : {zeros:.4f}   (Δ {zeros-normal:+.3f})")
    print(f"  memory↔embed cos  : {sim['mean_cos']:.4f}")
    print(f"{'='*70}")

    interp = []
    if shuffled - normal > 2.0:
        interp.append(
            "Shuffle hurts a LOT — memory uses positional alignment. "
            "Variant relies on memory[p] ≈ info about text near position p*(256/96)."
        )
    elif shuffled - normal > 0.3:
        interp.append("Shuffle hurts moderately — some positional bias but not the whole story.")
    else:
        interp.append("Shuffle is ~no-op — memory is positionally-agnostic (slot-attention style).")

    if zeros - normal > 3.0:
        interp.append("Zero hurts a LOT — memory channel is critical for recon. Encoder is doing real work.")
    elif zeros - normal > 1.0:
        interp.append("Zero hurts moderately — memory contributes meaningfully but Llama can fall back to local context.")
    else:
        interp.append("Zero barely hurts — memory contribution is minimal; Llama mostly relies on local context.")

    if sim["mean_cos"] > 0.5:
        interp.append(
            f"High memory↔embed cos ({sim['mean_cos']:.2f}) — memory tokens are basically "
            "averages of original token embeddings. Encoder is local-pass-through, not abstract compression."
        )
    elif sim["mean_cos"] > 0.2:
        interp.append(
            f"Moderate memory↔embed cos ({sim['mean_cos']:.2f}) — some local correspondence but "
            "memory has been transformed."
        )
    else:
        interp.append(
            f"Low memory↔embed cos ({sim['mean_cos']:.2f}) — memory is in a different space than "
            "raw embeddings; encoder learned an abstract representation."
        )

    print("\nInterpretation:")
    for line in interp:
        print(f"  • {line}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "variant": args.variant,
                "ckpt": str(args.ckpt),
                "normal_recon_ce": normal,
                "shuffled_recon_ce": shuffled,
                "zero_recon_ce": zeros,
                "memory_embed_similarity": sim,
                "interpretation": interp,
            }, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
