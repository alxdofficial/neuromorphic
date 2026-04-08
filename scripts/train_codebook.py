"""Standalone trainer for the action RVQ-VAE codebook.

Usage:
    python -m scripts.train_codebook \
        --actions outputs/v12/action_database.pt \
        --out outputs/v12/codebook_v1.pt \
        --epochs 20

Loads the action database saved by `Trainer.flush_action_database`, trains an
ActionVQVAE on it, validates, and saves the fitted model.
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.codebook import ActionVQVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--actions", required=True, help="Path to action_database.pt")
    p.add_argument("--out", required=True, help="Path to save codebook_v1.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-levels", type=int, default=4)
    p.add_argument("--codes-per-level", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--subsample", type=int, default=100_000,
                   help="Randomly keep at most this many samples")
    p.add_argument("--resample-interval", type=int, default=100,
                   help="Steps between dead-code resampling checks")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.actions}")
    actions = torch.load(args.actions, map_location="cpu", weights_only=True)
    assert actions.dim() == 2, f"expected [N, D], got {actions.shape}"
    N, action_dim = actions.shape
    print(f"  raw: {N} samples x {action_dim} dims")

    if N > args.subsample:
        idx = torch.randperm(N)[:args.subsample]
        actions = actions[idx]
        N = actions.shape[0]
        print(f"  subsampled to {N}")

    mean = actions.mean(dim=0, keepdim=True)
    std = actions.std(dim=0, keepdim=True).clamp(min=1e-6)
    actions_norm = (actions - mean) / std

    dataset = TensorDataset(actions_norm)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"))

    vqvae = ActionVQVAE(
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        num_levels=args.num_levels,
        codes_per_level=args.codes_per_level,
        beta=args.beta,
    ).to(device)
    vqvae.set_normalization(mean.to(device), std.to(device))

    effective_vocab = args.codes_per_level ** args.num_levels
    print(f"RVQ-VAE: {args.num_levels} levels x {args.codes_per_level} codes "
          f"= {effective_vocab} effective combinations")
    print(f"  encoder: {action_dim} -> {args.hidden} -> {args.latent_dim}")
    print(f"  decoder: {args.latent_dim} -> {args.hidden} -> {action_dim}")
    n_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(vqvae.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        vqvae.train(True)
        epoch_recon = 0.0
        epoch_commit = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            out = vqvae(batch)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                vqvae.rvq.ema_update(out["z"], out["codes"])
                if step % args.resample_interval == 0 and step > 0:
                    vqvae.rvq.resample_dead_codes(out["z"])

            epoch_recon += out["recon_loss"].item()
            epoch_commit += out["commit_loss"].item()
            n_batches += 1
            step += 1

        avg_recon = epoch_recon / n_batches
        avg_commit = epoch_commit / n_batches

        usage = vqvae.rvq.usage_histogram()
        per_level_active = (usage > 0.001).float().sum(dim=1)
        per_level_entropy = -(usage.clamp(min=1e-9) * usage.clamp(min=1e-9).log()).sum(dim=1)
        print(f"[epoch {epoch+1:3d}] recon={avg_recon:.4f} commit={avg_commit:.4f} "
              f"active/level={per_level_active.tolist()} "
              f"entropy/level={[f'{e:.2f}' for e in per_level_entropy.tolist()]}")

    vqvae.train(False)
    with torch.no_grad():
        all_recon_err = []
        all_codes = []
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            out = vqvae(batch)
            all_recon_err.append(F.mse_loss(out["recon"], batch, reduction="none").mean(dim=1))
            all_codes.append(out["codes"])
        recon_err_per_sample = torch.cat(all_recon_err)
        codes_stacked = torch.cat(all_codes, dim=0)

    print()
    print("=== Final validation ===")
    print(f"  mean per-sample recon MSE: {recon_err_per_sample.mean().item():.6f}")
    print(f"  median per-sample recon MSE: {recon_err_per_sample.median().item():.6f}")
    print(f"  max per-sample recon MSE: {recon_err_per_sample.max().item():.6f}")
    unique_codes = torch.unique(codes_stacked, dim=0)
    print(f"  unique code tuples used: {unique_codes.shape[0]} "
          f"/ {effective_vocab} possible")

    checkpoint = {
        "state_dict": vqvae.state_dict(),   # includes action_mean/std buffers
        "config": {
            "action_dim": action_dim,
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "num_levels": args.num_levels,
            "codes_per_level": args.codes_per_level,
            "beta": args.beta,
        },
    }
    torch.save(checkpoint, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
