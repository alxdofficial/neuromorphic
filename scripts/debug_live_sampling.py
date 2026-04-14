"""Check sampling diversity during actual rollout (with real modulator outputs)."""
import torch, gc
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

from src.model.config import Config
from src.model.model import Model
from src.data import create_dataloader, get_tokenizer, get_special_token_ids
from src.codebook import ActionVQVAE

ckpt = torch.load("outputs/v12/bootstrap.pt", map_location=device, weights_only=False)
config = ckpt["config"]
tokenizer = get_tokenizer("tinyllama")
config.vocab_size = len(tokenizer)
config.eot_id = get_special_token_ids(tokenizer).get(
    "eos_token_id", tokenizer.eos_token_id)
model = Model(config).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.load_runtime_state(ckpt.get("runtime_state", {}))
del ckpt; gc.collect(); torch.cuda.empty_cache()

for p in model.parameters(): p.requires_grad = False

cb_ckpt = torch.load("outputs/v12/cycle_00/codebook.pt", map_location=device, weights_only=False)
cb_config = cb_ckpt["config"]
vqvae = ActionVQVAE(
    action_dim=cb_config["action_dim"], latent_dim=cb_config["latent_dim"],
    hidden=cb_config["hidden"], num_levels=cb_config["num_levels"],
    codes_per_level=cb_config["codes_per_level"],
).to(device)
vqvae.load_state_dict(cb_ckpt["state_dict"])
vqvae.train(False)
for p in vqvae.parameters(): p.requires_grad = False

BS = 8
model.memory.initialize_states(BS, device)
model._initialized = True

# Run a fresh forward to get realistic modulator inputs
dl = create_dataloader("A", tokenizer, batch_size=BS, seq_length=512, seed=42, max_steps=1)
batch = next(iter(dl))
input_ids = batch.input_ids.to(device)

with torch.no_grad():
    model.forward_chunk(input_ids, use_memory=True)

# Now capture the current modulator input + output
mem = model.memory
mod_input_f32 = mem._build_mod_input().float()  # [BS, NC, mod_in]
mod_output = mem._modulator_forward(mod_input_f32)  # [BS, NC, mod_out]
print(f"mod_input stats: mean={mod_input_f32.mean().item():.3f}, "
      f"std={mod_input_f32.std().item():.3f}, "
      f"abs_mean={mod_input_f32.abs().mean().item():.3f}")
print(f"mod_output stats: mean={mod_output.mean().item():.3f}, "
      f"std={mod_output.std().item():.3f}, "
      f"abs_mean={mod_output.abs().mean().item():.3f}")

# Flatten to [BS*NC, mod_out] for VQ
NC = mod_output.shape[1]
action_flat = mod_output.reshape(BS * NC, -1)
action_norm = vqvae.normalize(action_flat)
print(f"action_norm stats: mean={action_norm.mean().item():.3f}, "
      f"std={action_norm.std().item():.3f}, "
      f"abs_mean={action_norm.abs().mean().item():.3f}")

z = vqvae.encoder(action_norm)  # [BS*NC, latent]
print(f"z stats: mean={z.mean().item():.3f}, "
      f"std={z.std().item():.3f}, "
      f"abs_mean={z.abs().mean().item():.3f}")

# Check argmax codes for these real z
codes_argmax = vqvae.rvq.sample_codes(z, tau=1.0, sample=False)
print(f"Argmax codes (real rollout): {codes_argmax.flatten().tolist()}")
print(f"  Unique argmax: {torch.unique(codes_argmax).numel()}/{BS*NC}")

# Check multinomial sampling diversity — simulate K=8 draws from each z
K = 8
codes_k = []
for k in range(K):
    codes_k.append(vqvae.rvq.sample_codes(z, tau=1.0, sample=True))
codes_stack = torch.stack(codes_k, dim=0).squeeze(-1)  # [K, BS*NC]
print(f"\nMultinomial sampling K={K} times from same z:")
print(f"  codes_stack shape: {codes_stack.shape}")
# Per-slot unique codes across K
unique_per_slot = []
for i in range(BS*NC):
    u = torch.unique(codes_stack[:, i]).numel()
    unique_per_slot.append(u)
print(f"  Unique codes per slot (mean, max): {sum(unique_per_slot)/len(unique_per_slot):.2f}, {max(unique_per_slot)}")

# The key test: decoded actions across K
# For each slot, decode all K samples and check their spread
print(f"\nDecoded action spread across K (per slot):")
cb = vqvae.rvq.codebooks[0]  # [num_codes, latent]
all_decoded = torch.zeros(K, BS*NC, 1056, device=device)
for k in range(K):
    z_q = cb[codes_stack[k]]  # [BS*NC, latent]
    decoded = vqvae.decoder(z_q)  # [BS*NC, 1056]
    denorm = vqvae.denormalize(decoded)
    all_decoded[k] = denorm
# std across K per slot
per_slot_std = all_decoded.std(dim=0)  # [BS*NC, 1056]
print(f"  K-std per slot (mean, max): {per_slot_std.mean().item():.4f}, {per_slot_std.max().item():.4f}")
print(f"  Decoded action abs mean: {all_decoded.abs().mean().item():.4f}")
print(f"  Relative K-std: {per_slot_std.mean().item() / all_decoded.abs().mean().item() * 100:.2f}%")

# Compare: what would happen if we sampled with tau=0.1 (more peaked) or 10.0 (more uniform)?
print()
for test_tau in [0.1, 1.0, 5.0, 10.0]:
    codes_k_t = []
    for k in range(K):
        codes_k_t.append(vqvae.rvq.sample_codes(z, tau=test_tau, sample=True))
    codes_stack_t = torch.stack(codes_k_t, dim=0).squeeze(-1)
    unique_per_slot_t = [torch.unique(codes_stack_t[:, i]).numel() for i in range(BS*NC)]
    # Decoded spread
    all_dec_t = torch.zeros(K, BS*NC, 1056, device=device)
    for k in range(K):
        z_q = cb[codes_stack_t[k]]
        all_dec_t[k] = vqvae.denormalize(vqvae.decoder(z_q))
    k_std = all_dec_t.std(dim=0).mean().item()
    print(f"  tau={test_tau}: unique/slot={sum(unique_per_slot_t)/len(unique_per_slot_t):.2f}, "
          f"decoded K-std={k_std:.4f}")
