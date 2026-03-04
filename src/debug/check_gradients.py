"""
Standalone gradient diagnostic: K=2 segments for realistic gradient flow.

Running multiple segments ensures:
  - Segment 1 commit populates EM state + uses raw_beta → state updated
  - Segment 2 reads from populated EM → non-zero EM param gradients
  - Cross-segment gradient path: loss_2 → read(state_1) → commit(raw_beta) → grad

Reports per-segment activation norms, per-parameter grad norms,
flags zero/vanishing/exploding, and groups by subsystem.

Usage:
    python -m src.debug.check_gradients [checkpoint_path]

If no checkpoint, creates a fresh model with tiny config for a quick check.
"""

import sys
import torch
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM


# Subsystem classification by parameter name prefix
GROUPS = [
    ("stage1", "stage1."),
    ("stage3", "stage3."),
    ("pm", "pm."),
    ("em_neuromod", "em_neuromod."),
    ("em", "em."),
    ("pcm", "pcm."),
    ("W_seed_w", "W_seed_w."),
    ("W_nov", "W_nov."),
    ("embedding", "embedding."),
    ("lm_head", "lm_head."),
    ("proj_up", "proj_up."),
    ("proj_down", "proj_down."),
    ("other", ""),
]

K_SEGMENTS = 2


def classify_param(name: str) -> str:
    for group, prefix in GROUPS:
        if prefix and name.startswith(prefix):
            return group
    return "other"


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        config.validate()
        model = NeuromorphicLM(config).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        print("No checkpoint — using fresh tier_tiny model")
        config = ModelConfig.tier_tiny()
        config.validate()
        model = NeuromorphicLM(config).to(device)

    # Match dtype to state dtype (bf16 on CUDA, fp32 on CPU)
    from ..model.utils import runtime_state_dtype
    dtype = runtime_state_dtype(device)
    model = model.to(dtype=dtype)

    model.train()
    BS = 2
    N = config.N
    model.initialize_states(BS, device)

    # Synthetic data for K segments
    all_inputs = [
        torch.randint(0, config.vocab_size, (BS, N), device=device)
        for _ in range(K_SEGMENTS)
    ]
    all_targets = [
        torch.randint(0, config.vocab_size, (BS, N), device=device)
        for _ in range(K_SEGMENTS)
    ]

    # --- Forward K segments, accumulate loss ---
    model.zero_grad()
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_ce = 0.0
    total_aux = 0.0
    seg_act_norms = []

    for seg_idx in range(K_SEGMENTS):
        logits, aux = model.forward_segment(all_inputs[seg_idx])
        ce = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            all_targets[seg_idx].view(-1),
        )
        total_loss = total_loss + ce + aux
        total_ce += ce.item()
        total_aux += aux.item()

        # Capture per-segment activation norms
        act_norms = getattr(model, "_dbg_act_norms", None)
        seg_act_norms.append(dict(act_norms) if act_norms else None)

    # --- Backward ---
    total_loss.backward()

    # --- Activation norms per segment ---
    print("\n" + "=" * 70)
    print(f"ACTIVATION NORMS at integration point (K={K_SEGMENTS} segments)")
    print("=" * 70)
    for seg_idx, act_norms in enumerate(seg_act_norms):
        print(f"\n  Segment {seg_idx + 1}:")
        if act_norms:
            h_norm = act_norms.get("H", 0)
            for key in ["H", "pm", "em", "cum_em"]:
                val = act_norms.get(key, 0)
                ratio = val / h_norm if h_norm > 0 else float("nan")
                flag = ""
                if key != "H" and ratio > 10:
                    flag = "  << DOMINATES H"
                elif key != "H" and ratio < 0.01:
                    flag = "  << NEGLIGIBLE vs H"
                elif key != "H" and ratio < 0.1:
                    flag = "  < weak vs H"
                print(f"    {key:>8s}: {val:12.4f}  (ratio to H: {ratio:.4f}){flag}")
        else:
            print("    (not available — _dbg_act_norms not set)")

    # --- Per-parameter report ---
    print("\n" + "=" * 70)
    print("PER-PARAMETER GRADIENT REPORT")
    print("=" * 70)

    total_params = 0
    zero_grads = 0
    vanishing = 0
    exploding = 0
    no_grad = 0

    group_norms = {}  # group -> sum of squared norms
    group_counts = {}  # group -> param count

    rows = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total_params += 1
        group = classify_param(name)
        group_counts[group] = group_counts.get(group, 0) + 1

        if p.grad is None:
            no_grad += 1
            rows.append((name, tuple(p.shape), None, None, "NO_GRAD"))
            continue

        gnorm = p.grad.detach().norm().item()
        pnorm = p.detach().norm().item()
        ratio = gnorm / max(pnorm, 1e-12)

        group_norms[group] = group_norms.get(group, 0) + gnorm ** 2

        flag = ""
        if gnorm == 0:
            zero_grads += 1
            flag = "ZERO"
        elif gnorm < 1e-8:
            vanishing += 1
            flag = "VANISHING"
        elif gnorm > 1e3:
            exploding += 1
            flag = "EXPLODING"

        rows.append((name, tuple(p.shape), gnorm, ratio, flag))

    # Print flagged parameters first, then all
    flagged = [r for r in rows if r[4]]
    if flagged:
        print(f"\nFLAGGED PARAMETERS ({len(flagged)}):")
        print(f"  {'Name':<55s} {'Shape':<20s} {'GradNorm':>12s} {'G/P Ratio':>12s} {'Flag'}")
        print("  " + "-" * 105)
        for name, shape, gnorm, ratio, flag in flagged:
            gn_str = f"{gnorm:.2e}" if gnorm is not None else "N/A"
            r_str = f"{ratio:.2e}" if ratio is not None else "N/A"
            print(f"  {name:<55s} {str(shape):<20s} {gn_str:>12s} {r_str:>12s} {flag}")

    print(f"\nALL PARAMETERS ({total_params}):")
    print(f"  {'Name':<55s} {'Shape':<20s} {'GradNorm':>12s} {'G/P Ratio':>12s} {'Flag'}")
    print("  " + "-" * 105)
    for name, shape, gnorm, ratio, flag in rows:
        gn_str = f"{gnorm:.2e}" if gnorm is not None else "N/A"
        r_str = f"{ratio:.2e}" if ratio is not None else "N/A"
        print(f"  {name:<55s} {str(shape):<20s} {gn_str:>12s} {r_str:>12s} {flag}")

    # --- Group summary ---
    print("\n" + "=" * 70)
    print("GROUP SUMMARY")
    print("=" * 70)
    print(f"  {'Group':<20s} {'Params':>8s} {'GradNorm':>12s}")
    print("  " + "-" * 42)
    for group, prefix in GROUPS:
        if group not in group_counts:
            continue
        norm = group_norms.get(group, 0) ** 0.5
        print(f"  {group:<20s} {group_counts[group]:>8d} {norm:>12.4e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Segments:              {K_SEGMENTS}")
    print(f"  Total trainable params: {total_params}")
    print(f"  Zero gradients:    {zero_grads}/{total_params}")
    print(f"  Vanishing (<1e-8): {vanishing}/{total_params}")
    print(f"  Exploding (>1e3):  {exploding}/{total_params}")
    print(f"  No grad tensor:    {no_grad}/{total_params}")
    print(f"  Loss: {total_loss.item():.4f} (CE: {total_ce:.4f}, aux: {total_aux:.4f})")


if __name__ == "__main__":
    main()
