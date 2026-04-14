"""Replicate real training rollout exactly, measure K-spread at each layer."""
import torch, gc
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

from src.model.config import Config
from src.model.model import Model
from src.data import create_dataloader, get_tokenizer, get_special_token_ids
from src.codebook import ActionVQVAE
from src.phase2.trainer import Phase2Trainer

# Load phase1_end (modulator trained, ready for phase 2)
ckpt = torch.load("outputs/v12/cycle_00/phase1_end.pt", map_location=device, weights_only=False)
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

# Load trained codebook
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

# Resize memory to phase 2 stage 1 BS (BS=24 for W=512)
BS = 24
K = 8
W = 512
T = 2 * W

model.memory.resize_to_bs(BS)
model._initialized = model.memory._initialized
model.lm._carries = [None] * config.L_total

dl = create_dataloader("A", tokenizer, batch_size=BS, seq_length=T, seed=42, max_steps=2)
p2 = Phase2Trainer(model=model, vqvae=vqvae, dataloader=dl, config=config,
                   device=device, group_size=K)
p2.reward_window = W

# Warm up: one rollout
batch = next(iter(dl))
_ = p2.rollout(batch)
del _
gc.collect(); torch.cuda.empty_cache()

# Run a real rollout with full tracing
batch = next(iter(dl))
input_ids = batch.input_ids.to(device)
prev_token = getattr(batch, "prev_token", None)

# Simulate the rollout manually to instrument
model.lm.detach_carries()
H_mid = p2._run_lower_scan(input_ids, prev_token=prev_token)

init_snap = p2._save_mem_state()
mem = model.memory
for key, val in init_snap.items():
    expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
    setattr(mem, key, expanded.clone())

ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])
bmap = torch.arange(K * BS, device=device) % BS

# Per-trajectory perturbation (variant B)
import os
SIGMA = float(os.environ.get("TRAJ_SIGMA", "0.5"))
latent_dim = vqvae.rvq.code_dim
NC = mem.N_cells
if SIGMA > 0:
    traj_noise_kb = torch.randn(K, BS, 1, latent_dim, device=device, dtype=torch.float32) * SIGMA
    traj_noise = traj_noise_kb.expand(K, BS, NC, latent_dim).reshape(K * BS, NC, latent_dim).contiguous()
    print(f"Using traj_noise with σ={SIGMA}")
else:
    traj_noise = None
    print("No traj_noise (baseline)")

result = mem.forward_segment_phase2(
    H_mid, ids_exp, model.lm, vqvae, tau=1.0, sample=True,
    h_mid_batch_map=bmap, traj_noise=traj_noise)

# Check divergence
readouts = result["readouts"].view(K, BS, T, -1)
print(f"Real rollout (phase1_end + trained codebook):")
print(f"  readouts std across K: {readouts.std(dim=0).mean().item():.6f}")
print(f"  readouts abs mean: {readouts.abs().mean().item():.3f}")
print(f"  Relative: {readouts.std(dim=0).mean().item() / readouts.abs().mean().item() * 100:.2f}%")

# Now run the REAL reward computation used during training
prev_token_exp = None
if prev_token is not None:
    prev_token_exp = prev_token.unsqueeze(0).expand(K, *prev_token.shape).reshape(K * BS)

per_token_reward = p2._compute_per_token_reward(
    result["readouts"], ids_exp, H_mid,
    prev_token=prev_token_exp, h_mid_batch_map=bmap)
print(f"\nper_token_reward shape: {per_token_reward.shape}")  # [K*BS, T]
per_token_reward_k = per_token_reward.view(K, BS, T)
print(f"Per-token reward stats:")
print(f"  abs mean: {per_token_reward_k.abs().mean().item():.4f}")
print(f"  K-std per (b, t) mean: {per_token_reward_k.std(dim=0).mean().item():.6f}")
print(f"  K-std per (b, t) max:  {per_token_reward_k.std(dim=0).max().item():.6f}")
# Check a specific slot
print(f"  [k, b=0, t=100]: {per_token_reward_k[:, 0, 100].tolist()}")
print(f"  [k, b=0, t=500]: {per_token_reward_k[:, 0, 500].tolist()}")

# Now windowed reward
call_positions = result["call_positions"]
windowed = p2._windowed_reward(per_token_reward_k, call_positions, W)
print(f"\nWindowed reward shape: {windowed.shape}")  # [K, n_calls, BS]
print(f"Windowed reward stats:")
print(f"  abs mean: {windowed.abs().mean().item():.4f}")
print(f"  K-std per (call, sample) mean: {windowed.std(dim=0).mean().item():.6e}")
print(f"  K-std per (call, sample) max:  {windowed.std(dim=0).max().item():.6e}")
print(f"  First call [k, b=0]: {windowed[:, 0, 0].tolist()}")
