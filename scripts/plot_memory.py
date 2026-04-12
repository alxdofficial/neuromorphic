"""Memory graph plasticity diagnostics.

Loads a phase-1 checkpoint, warms up memory on real Pile tokens so runtime
state is in a realistic regime (not cold-start), then runs an eager per-token
forward pass with telemetry capture, and emits four plots that reveal the
memory graph's plasticity dynamics.

    memory_timescales.png         — learned per-cell plasticity half-lives
    memory_plasticity_stream.png  — per-cell ||ΔW|| heatmap over tokens +
                                    surprise overlay
    memory_hebbian_alignment.png  — cosine(W, Hebbian) per cell over tokens
    memory_cell_roles.png         — write-rate vs read-rate scatter, color =
                                    Pearson(||ΔW||, s_mem_live)

Usage:
    python -m scripts.plot_memory \
        --checkpoint outputs/v12/bootstrap_v3/ckpt_040690.pt \
        --tokens 1024 --bs 4 --warmup 256

    # Plots are written to <checkpoint_parent>/plots/memory/ by default,
    # so they sit next to the phase-1 training plots written by train.py.
    # Override with --out-dir if you want them somewhere else.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import create_dataloader, get_special_token_ids, get_tokenizer
from src.model.config import Config
from src.model.model import Model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Phase-1 checkpoint .pt")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for PNGs. Defaults to "
                        "<checkpoint_parent>/plots/memory/ so diagnostics "
                        "live next to the training plots from train.py.")
    p.add_argument("--tokens", type=int, default=1024,
                   help="Tokens to run in the diagnostic capture pass")
    p.add_argument("--bs", type=int, default=4,
                   help="Batch size (parallel streams)")
    p.add_argument("--warmup", type=int, default=256,
                   help="Tokens of warm-up forward before capture")
    p.add_argument("--tokenizer", default="tinyllama")
    p.add_argument("--phase", default="A")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str, device: torch.device):
    print(f"Loading checkpoint {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config: Config = ckpt["config"]
    model = Model(config).to(device)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (will use init defaults)")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")
    model.train(False)
    return model, config, ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Diagnostic forward pass (eager, per-token, captures telemetry)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostic(model: Model, input_ids: torch.Tensor, device: torch.device):
    """Eager per-token memory forward over all of input_ids.

    The loop is sliced into segments of config.T so the LM lower scan's
    pos_embed[:T] slice stays in range. Memory state is carried across
    segments (it's lifelong by design).

    Captures per modulator call:
        W_before, W_after  (for ΔW)
        hebbian            (for Hebbian-W alignment)
    Captures per token:
        readout norm per cell
        s_mem_live
    """
    mg = model.memory
    lm = model.lm
    assert mg.is_initialized, "memory must be initialized before diagnostic"

    BS, T_total = input_ids.shape
    segment_T = mg.config.T
    dt = torch.bfloat16

    # Cast weight copies once (matches what forward_segment does internally)
    st_w1_full = mg.state_w1.to(dt)
    st_w1_recv = st_w1_full[:, :mg.D_n].contiguous()
    st_w1_h = st_w1_full[:, mg.D_n:].contiguous()
    st_b1 = mg.state_b1.to(dt)
    st_w2 = mg.state_w2.to(dt)
    st_b2 = mg.state_b2.to(dt)
    mg_w1 = mg.msg_w1.to(dt)
    mg_b1 = mg.msg_b1.to(dt)
    mg_w2 = mg.msg_w2.to(dt)
    mg_b2 = mg.msg_b2.to(dt)
    inject_w = mg.inject_w.to(dt)
    inject_b = mg.inject_b.to(dt)
    mod_w1 = mg.mod_w1.to(dt)
    mod_b1 = mg.mod_b1.to(dt)
    mod_w2 = mg.mod_w2.to(dt)
    mod_b2 = mg.mod_b2.to(dt)
    identity = mg._identity(BS, dt, device)

    W_gamma = torch.sigmoid(mg.W_decay_logit).to(dt)
    decay_gamma = torch.sigmoid(mg.decay_gamma_logit).to(dt)
    hebbian_gamma = torch.sigmoid(mg.hebbian_decay_logit).to(dt)

    # LM head weights for in-loop surprise
    lm_head_w = lm.lm_head.weight.to(dt)
    proj_down_w = lm.proj_down.weight.to(dt) if lm.proj_down is not None else None
    proj_down_b = lm.proj_down.bias.to(dt) if lm.proj_down is not None else None
    ln_final_w = lm.ln_final.weight.to(dt)
    ln_final_b = lm.ln_final.bias.to(dt)

    # Working runtime state — clone so we don't mutate the model's
    h = mg.h.clone()
    msg = mg.msg.clone()
    W = mg.W.clone()
    decay = mg.decay.clone()
    hebbian = mg.hebbian.clone()
    prev_readout = mg.prev_readout.clone()
    readout_drift = mg.readout_drift.clone()
    s_mem_live = mg.s_mem_live.clone()
    s_mem_ema_fast = mg.s_mem_ema_fast.clone()

    gate = decay.unsqueeze(-1)
    one_minus_gate = 1.0 - gate

    M = mg.config.modulation_interval
    gain_fast = mg.config.gain_ema_fast
    NC, N, D_n = mg.N_cells, mg.C_n, mg.D_n

    # Captures
    W_before_snaps: list[torch.Tensor] = []
    W_after_snaps: list[torch.Tensor] = []
    hebbian_snaps: list[torch.Tensor] = []
    mod_positions: list[int] = []
    s_mem_at_mod: list[torch.Tensor] = []            # each [BS], at captured mod calls
    readout_norms_per_tok: list[torch.Tensor] = []   # each [BS, NC]
    s_mem_per_tok: list[torch.Tensor] = []           # each [BS]

    # Subsample captures to keep memory bounded: target ~1000 modulator
    # snapshots and ~4000 per-token samples regardless of T_total.
    total_mod_calls = T_total // M
    mod_subsample = max(1, total_mod_calls // 1000)
    tok_subsample = max(1, T_total // 4000)
    mod_call_idx = 0

    amp = torch.autocast(device_type=device.type, dtype=dt)
    # Thread prev_token across segments so cross-segment EOT resets match
    # training (Model.forward_chunk line 67).
    prev_seg_last_tok = None

    for seg_start in range(0, T_total, segment_T):
        seg_end = min(seg_start + segment_T, T_total)
        seg_ids = input_ids[:, seg_start:seg_end]
        seg_len = seg_ids.shape[1]

        # Per-segment EOT reset mask for the LM scan, including the
        # cross-segment boundary via prev_seg_last_tok.
        eos_positions = (seg_ids == mg.config.eot_id)
        reset_mask = torch.zeros_like(eos_positions)
        reset_mask[:, 1:] = eos_positions[:, :-1]
        if prev_seg_last_tok is not None:
            reset_mask[:, 0] = (prev_seg_last_tok == mg.config.eot_id)
        if not reset_mask.any():
            reset_mask = None
        prev_seg_last_tok = seg_ids[:, -1]

        with amp:
            H_mid = lm.forward_scan_lower(seg_ids, reset_mask=reset_mask)
        H_mid = H_mid.to(dt)

        for t in range(seg_len):
            abs_t = seg_start + t
            H_mid_t = H_mid[:, t]
            tok_t = seg_ids[:, t]

            # --- Live memory-head surprise (same as forward_segment):
            # proper CE = logsumexp(logits) - target_logit.
            x = prev_readout.to(dt)
            if proj_down_w is not None:
                x = F.linear(x, proj_down_w.to(dt), proj_down_b.to(dt) if proj_down_b is not None else None)
            x = F.layer_norm(x, (x.shape[-1],), ln_final_w.to(dt), ln_final_b.to(dt))
            logits_full = F.linear(x, lm_head_w.to(dt))
            lse = torch.logsumexp(logits_full.float(), dim=-1)
            target_logit = logits_full.gather(
                1, tok_t.unsqueeze(1)).squeeze(1).float()
            s_mem_live = (lse - target_logit).to(dt)
            s_mem_ema_fast = (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live
            if abs_t % tok_subsample == 0:
                s_mem_per_tok.append(s_mem_live.float().cpu())

            # --- Modulate every M tokens ---
            if abs_t % M == 0:
                W_before_snap = W.clone()
                W, decay = mg._modulate_cells(
                    h, msg, W, decay, hebbian,
                    readout_drift, s_mem_live, s_mem_ema_fast,
                    mod_w1, mod_b1, mod_w2, mod_b2,
                    W_gamma, decay_gamma)
                gate = decay.unsqueeze(-1)
                one_minus_gate = 1.0 - gate
                if mod_call_idx % mod_subsample == 0:
                    W_before_snaps.append(W_before_snap.float().cpu())
                    W_after_snaps.append(W.float().cpu())
                    hebbian_snaps.append(hebbian.float().cpu())
                    mod_positions.append(abs_t)
                    s_mem_at_mod.append(s_mem_live.float().cpu())
                mod_call_idx += 1

            # --- Memory step ---
            h, msg, readout = mg._step(
                h, msg, W, gate, one_minus_gate,
                H_mid_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2)

            if abs_t % tok_subsample == 0:
                readout_cell = readout.reshape(BS, NC, D_n)
                readout_norms_per_tok.append(
                    readout_cell.float().norm(dim=-1).cpu())

            hebbian = mg._hebbian_update(hebbian, msg, hebbian_gamma)

            # prev_readout is still readout[t-1] here — view as per-cell
            # to compute drift, then advance it below.
            new_cell = readout.reshape(BS, NC, D_n)
            prev_cell = prev_readout.view(BS, NC, D_n)
            readout_drift = (new_cell - prev_cell).abs().mean(
                dim=-1, keepdim=True).to(dt)
            prev_readout = readout

    return {
        "W_before": torch.stack(W_before_snaps),             # [n_calls, BS, NC, N, N]
        "W_after":  torch.stack(W_after_snaps),
        "hebbian":  torch.stack(hebbian_snaps),
        "mod_positions": mod_positions,
        "s_mem_at_mod": torch.stack(s_mem_at_mod),           # [n_calls, BS]
        "readout_norms": torch.stack(readout_norms_per_tok), # [n_samples, BS, NC]
        "s_mem":        torch.stack(s_mem_per_tok),          # [n_samples, BS]
        "BS": BS, "T": T_total, "NC": NC, "N": N, "M": M,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _half_life_updates(gamma_np: np.ndarray) -> np.ndarray:
    """Half-life of an EMA with rate γ, in # of updates.

    Old state decays as (1-γ)^n. Half-life n* satisfies (1-γ)^n* = 1/2
      → n* = ln(0.5) / ln(1-γ).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(0.5) / np.log(np.clip(1.0 - gamma_np, 1e-12, 1 - 1e-12))
    return hl


def plot_timescales(mg, out_path: Path, ckpt_step: int):
    """Per-cell learned plasticity half-lives (tokens) for W, decay, Hebbian."""
    M = mg.config.modulation_interval
    NC = mg.N_cells

    W_gamma = torch.sigmoid(mg.W_decay_logit.detach().float()).cpu().numpy()
    decay_gamma = torch.sigmoid(mg.decay_gamma_logit.detach().float()).cpu().numpy()
    hebb_gamma = torch.sigmoid(mg.hebbian_decay_logit.detach().float()).cpu().numpy()

    # W and decay updates happen every M tokens, Hebbian every token.
    W_hl = _half_life_updates(W_gamma) * M
    decay_hl = _half_life_updates(decay_gamma) * M
    hebb_hl = _half_life_updates(hebb_gamma)

    cells = np.arange(NC)
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(cells - width, W_hl, width, label="W plasticity", color="#E74C3C")
    ax.bar(cells,         decay_hl, width, label="decay plasticity", color="#3498DB")
    ax.bar(cells + width, hebb_hl, width, label="Hebbian trace",   color="#27AE60")

    ax.set_yscale("log")
    ax.set_xlabel("cell")
    ax.set_ylabel("half-life (tokens)")
    ax.set_title(f"Learned plasticity timescales per cell  (step {ckpt_step:,})")
    ax.set_xticks(cells)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_plasticity_stream(captures: dict, out_path: Path):
    """Heatmap of per-cell ||ΔW|| over modulator calls, with surprise overlay."""
    W_before = captures["W_before"]           # [n_calls, BS, NC, N, N]
    W_after = captures["W_after"]
    mod_positions = captures["mod_positions"]
    NC = captures["NC"]
    T = captures["T"]
    M = captures["M"]

    # Delta magnitude per cell per modulator call, averaged over batch
    diff = (W_after - W_before).reshape(
        W_before.shape[0], W_before.shape[1], NC, -1)
    # [n_calls, BS, NC] → avg over BS → [n_calls, NC] → transpose to [NC, n_calls]
    delta_norm = diff.norm(dim=-1).mean(dim=1).T.numpy()

    # Surprise signal, averaged over batch
    s_mem = captures["s_mem"].mean(dim=1).numpy()  # [T]

    fig, (ax_heat, ax_surp) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    im = ax_heat.imshow(
        delta_norm, aspect="auto", cmap="magma",
        extent=[mod_positions[0] - M / 2, mod_positions[-1] + M / 2,
                NC - 0.5, -0.5],
        interpolation="nearest")
    ax_heat.set_ylabel("cell")
    ax_heat.set_yticks(range(NC))
    cb = fig.colorbar(im, ax=ax_heat, label="‖ΔW‖")
    ax_heat.set_title(
        "Per-cell plasticity stream  (‖W_after − W_before‖ per modulator call)")

    n_surp = len(s_mem)
    surp_x = np.linspace(0, T - 1, n_surp)
    ax_surp.plot(surp_x, s_mem, color="#E74C3C", lw=1.0)
    ax_surp.axhline(0, color="gray", lw=0.4, ls="--")
    ax_surp.set_xlabel("token position")
    ax_surp.set_ylabel("s_mem_live")
    ax_surp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hebbian_alignment(captures: dict, out_path: Path):
    """Cosine similarity between the Hebbian trace and W per cell, over tokens.

    Plots both full-matrix and off-diagonal-only variants. The full matrix
    is dominated by diagonal/self-structure; the off-diagonal is the actual
    pairwise co-activation signal.
    """
    W_after = captures["W_after"]      # [n_calls, BS, NC, N, N]
    hebbian = captures["hebbian"]      # [n_calls, BS, NC, N, N]
    mod_positions = captures["mod_positions"]
    NC = captures["NC"]
    N = W_after.shape[-1]

    # Full-matrix cosine
    W_flat = W_after.reshape(*W_after.shape[:3], -1)
    H_flat = hebbian.reshape(*hebbian.shape[:3], -1)
    cos_full = F.cosine_similarity(W_flat, H_flat, dim=-1)     # [n_calls, BS, NC]
    align_full = cos_full.mean(dim=1).T.numpy()                # [NC, n_calls]

    # Off-diagonal-only cosine
    diag_mask = torch.eye(N, dtype=torch.bool).view(1, 1, 1, N, N)
    W_off = W_after.masked_fill(diag_mask, 0.0)
    H_off = hebbian.masked_fill(diag_mask, 0.0)
    W_off_flat = W_off.reshape(*W_off.shape[:3], -1)
    H_off_flat = H_off.reshape(*H_off.shape[:3], -1)
    cos_off = F.cosine_similarity(W_off_flat, H_off_flat, dim=-1)
    align_off = cos_off.mean(dim=1).T.numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, NC))

    for c in range(NC):
        axes[0].plot(mod_positions, align_full[c], color=cmap[c],
                     label=f"cell {c}", lw=1.3, alpha=0.9)
    axes[0].axhline(0, color="gray", lw=0.5, ls="--")
    axes[0].set_ylabel("cos (full matrix)")
    axes[0].set_title("Hebbian ↔ W alignment per cell  "
                      "(full vs off-diagonal)")
    axes[0].legend(ncol=NC, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, 1.32))
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.05, 1.05)

    for c in range(NC):
        axes[1].plot(mod_positions, align_off[c], color=cmap[c], lw=1.3,
                     alpha=0.9)
    axes[1].axhline(0, color="gray", lw=0.5, ls="--")
    axes[1].set_xlabel("token position")
    axes[1].set_ylabel("cos (off-diagonal only)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cell_roles(captures: dict, out_path: Path):
    """Per-cell scatter: write rate vs read rate, colored by surprise corr."""
    NC = captures["NC"]
    W_before = captures["W_before"]
    W_after = captures["W_after"]
    mod_positions = captures["mod_positions"]

    diff = (W_after - W_before).reshape(
        W_before.shape[0], W_before.shape[1], NC, -1)
    delta_per_call = diff.norm(dim=-1).mean(dim=1).numpy()   # [n_calls, NC]
    write_rate = delta_per_call.mean(axis=0)                  # [NC]

    # Read rate: mean per-cell readout norm over tokens and batch
    read_rate = captures["readout_norms"].mean(dim=(0, 1)).numpy()  # [NC]

    # Surprise correlation per cell: Pearson between per-call ΔW norm and
    # the surprise at that modulator-call token
    s_mem_at_mod = captures["s_mem_at_mod"].mean(dim=1).numpy()  # [n_calls]
    corr = np.zeros(NC)
    for c in range(NC):
        x = delta_per_call[:, c]
        if x.std() > 1e-8 and s_mem_at_mod.std() > 1e-8:
            corr[c] = float(np.corrcoef(x, s_mem_at_mod)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        write_rate, read_rate,
        c=corr, cmap="RdBu_r", vmin=-1, vmax=1,
        s=220, edgecolors="black", linewidths=0.9)
    for c in range(NC):
        ax.annotate(f"{c}", (write_rate[c], read_rate[c]),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("write rate  (mean ‖ΔW‖ per modulator call)")
    ax.set_ylabel("read rate  (mean ‖readout_cell‖ per token)")
    ax.set_title("Cell roles: write vs read  (color = Pearson(‖ΔW‖, s_mem_live))")
    fig.colorbar(sc, ax=ax, label="surprise correlation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Default: drop plots next to the training plots for the same run.
    if args.out_dir is None:
        out_dir = Path(args.checkpoint).resolve().parent / "plots" / "memory"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Tokenizer + vocab updates
    tokenizer = get_tokenizer(args.tokenizer)
    special_ids = get_special_token_ids(tokenizer)

    # Model
    model, config, ckpt_step = load_model(args.checkpoint, device)
    config.vocab_size = len(tokenizer)
    config.eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
    print(f"  step {ckpt_step:,}  params "
          f"LM={model.lm_param_count()/1e6:.1f}M mem={model.memory_param_count()/1e6:.1f}M")

    # ---- 1. Timescales plot first — no data needed, always works ----
    plot_timescales(model.memory, out_dir / "memory_timescales.png", ckpt_step)

    # ---- Fetch a batch of tokens (warmup + capture in one sequence) ----
    total_tokens = args.warmup + args.tokens
    dataloader = create_dataloader(
        phase=args.phase, tokenizer=tokenizer, batch_size=args.bs,
        seq_length=total_tokens, seed=args.seed, max_steps=1)
    batch = next(iter(dataloader))
    input_ids = batch.input_ids.to(device, non_blocking=True)
    print(f"Data: BS={args.bs}, tokens={total_tokens} "
          f"(warmup={args.warmup}, capture={args.tokens})")

    # ---- Warm up memory with forward_chunk, in segments of config.T ----
    model.memory.initialize_states(args.bs, device)
    warm_ids = input_ids[:, :args.warmup]
    segment_T = config.T
    print("Warming up memory on real tokens...")
    # Thread prev_token across chunk boundaries so cross-chunk EOT resets
    # match training-time semantics (see Model.forward_chunk EOT handling).
    prev_token = None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16), \
         torch.no_grad():
        for start in range(0, args.warmup, segment_T):
            end = min(start + segment_T, args.warmup)
            sub = warm_ids[:, start:end]
            if sub.shape[1] < 2:
                continue
            model.forward_chunk(sub, target_ids=sub, use_memory=True,
                                prev_token=prev_token)
            prev_token = sub[:, -1]
            model.detach_states()

    # ---- Diagnostic capture pass ----
    print(f"Running diagnostic capture ({args.tokens} tokens)...")
    capture_ids = input_ids[:, args.warmup:args.warmup + args.tokens]
    captures = run_diagnostic(model, capture_ids, device)
    n_calls = len(captures["mod_positions"])
    print(f"  captured {n_calls} modulator calls over {captures['T']} tokens")

    # ---- Remaining plots ----
    plot_plasticity_stream(captures, out_dir / "memory_plasticity_stream.png")
    plot_hebbian_alignment(captures, out_dir / "memory_hebbian_alignment.png")
    plot_cell_roles(captures, out_dir / "memory_cell_roles.png")

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
