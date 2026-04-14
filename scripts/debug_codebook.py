"""Check if codebook collapse is causing the K-homogeneity."""
import torch
from src.codebook import ActionVQVAE

device = torch.device("cuda")

# Load trained codebook (if it still exists)
import os
codebook_path = "outputs/v12/cycle_00/codebook.pt"
if not os.path.exists(codebook_path):
    print(f"No trained codebook at {codebook_path}")
    # Use fresh one
    vqvae = ActionVQVAE(action_dim=1056, num_levels=1, codes_per_level=256).to(device)
else:
    cb_ckpt = torch.load(codebook_path, map_location=device, weights_only=False)
    cb_config = cb_ckpt["config"]
    vqvae = ActionVQVAE(
        action_dim=cb_config["action_dim"],
        latent_dim=cb_config["latent_dim"],
        hidden=cb_config["hidden"],
        num_levels=cb_config["num_levels"],
        codes_per_level=cb_config["codes_per_level"],
    ).to(device)
    vqvae.load_state_dict(cb_ckpt["state_dict"])
    print(f"Loaded trained codebook")

vqvae.train(False)
for p in vqvae.parameters(): p.requires_grad = False

# Sample some random actions (pretend they're modulator outputs)
torch.manual_seed(0)
N_actions = 100
raw_actions = torch.randn(N_actions, 1056, device=device)

# Normalize
normed = vqvae.normalize(raw_actions)
# Encode to latent
z = vqvae.encoder(normed)  # [N, latent_dim]
print(f"z stats: mean={z.mean().item():.3f}, std={z.std().item():.3f}, "
      f"abs_mean={z.abs().mean().item():.3f}")

# Sample 256 codes from SAME z (simulating K trajectories with identical modulator output)
print()
print("Simulating K=256 samples from same modulator output:")
torch.manual_seed(42)
z_single = z[:1]  # [1, latent_dim]
K = 256
# Each sample gets different multinomial draw
codes_all = torch.zeros(K, device=device, dtype=torch.long)
for k in range(K):
    codes_k = vqvae.rvq.sample_codes(z_single, tau=1.0, sample=True)
    codes_all[k] = codes_k[0, 0]

print(f"Unique codes drawn across K=256: {torch.unique(codes_all).numel()}/256")

# Decode each sampled code and check diversity
decoded_all = torch.zeros(K, 1056, device=device)
for k in range(K):
    codes_k = codes_all[k:k+1].unsqueeze(-1)  # [1, 1]
    lvl_idx = torch.arange(vqvae.rvq.num_levels, device=device)
    z_q = vqvae.rvq.codebooks[lvl_idx.unsqueeze(0), codes_k].sum(dim=1)  # [1, latent]
    decoded = vqvae.decoder(z_q)  # [1, 1056]
    decoded_all[k] = decoded[0]

print(f"Decoded action std across K samples: {decoded_all.std(dim=0).mean().item():.6f}")
print(f"Decoded action abs mean: {decoded_all.abs().mean().item():.6f}")
print(f"Relative std: {(decoded_all.std(dim=0).mean() / decoded_all.abs().mean()).item() * 100:.2f}%")

# Also check what the codes DECODE TO
decoded_unique, inverse = torch.unique(decoded_all, dim=0, return_inverse=True)
print(f"Unique DECODED actions across K samples: {decoded_unique.shape[0]}/256")

# Check codebook entry diversity directly
cb = vqvae.rvq.codebooks[0]  # [codes_per_level, latent_dim]
print(f"\nCodebook stats: {cb.shape}")
print(f"Codebook entry pairwise distances (min, mean, max):")
dists = (cb.unsqueeze(0) - cb.unsqueeze(1)).pow(2).sum(-1).sqrt()
mask = torch.eye(cb.shape[0], device=device, dtype=torch.bool)
dists_off = dists[~mask]
print(f"  {dists_off.min().item():.4f}, {dists_off.mean().item():.4f}, {dists_off.max().item():.4f}")

# Check which codes get picked with different z's
print(f"\nPer-z code selection (argmax):")
codes_argmax = []
for i in range(min(20, N_actions)):
    z_i = z[i:i+1]
    codes_i = vqvae.rvq.sample_codes(z_i, tau=1.0, sample=False)  # argmax
    codes_argmax.append(codes_i[0, 0].item())
print(f"  First 20 argmax codes: {codes_argmax}")
print(f"  Unique: {len(set(codes_argmax))}")
