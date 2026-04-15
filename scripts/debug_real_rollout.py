"""Replicate real training rollout exactly, measure K-spread at each layer.

Points the model at a phase1_end.pt (cycle phase 1 output) and runs the same
rollout + reward path used by Phase2Trainer. The integrated DiscreteActionPolicy
lives inside model.memory — there's no separate VQ-VAE to load anymore.
"""
import gc
import os

import torch

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

from src.model.model import Model
from src.data import create_dataloader, get_tokenizer, get_special_token_ids
from src.phase2.trainer import Phase2Trainer

# Load phase1_end (modulator trained, ready for phase 2). Overridable via env.
ckpt_path = os.environ.get("DEBUG_CKPT", "outputs/v12/cycle_00/phase1_end.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

BS = 24
K = 8
W = 512
T = 2 * W

model.memory.resize_to_bs(BS)
model._initialized = model.memory._initialized
model.lm._carries = [None] * config.L_total

dl = create_dataloader("A", tokenizer, batch_size=BS, seq_length=T, seed=42, max_steps=2)
p2 = Phase2Trainer(model=model, dataloader=dl, config=config,
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
if prev_token is not None:
    prev_token = prev_token.to(device)

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

result = mem.forward_segment_phase2(
    H_mid, ids_exp, model.lm, tau=1.0, sample=True,
    h_mid_batch_map=bmap)

# Check divergence
readouts = result["readouts"].view(K, BS, T, -1)
print("Real rollout (phase1_end + integrated codebook):")
print(f"  readouts std across K: {readouts.std(dim=0).mean().item():.6f}")
print(f"  readouts abs mean: {readouts.abs().mean().item():.3f}")
print(f"  Relative: {readouts.std(dim=0).mean().item() / readouts.abs().mean().item() * 100:.2f}%")

# Now run the REAL reward computation used during training
eot = config.eot_id
prev_token_exp = None
if prev_token is not None:
    prev_token_exp = prev_token.unsqueeze(0).expand(K, *prev_token.shape).reshape(K * BS)

valid_mask = torch.ones(K * BS, T, device=device, dtype=torch.float32)
if prev_token_exp is not None:
    valid_mask[:, 0] = (prev_token_exp != eot).float()
eos_positions = (ids_exp == eot)
if eos_positions.any():
    valid_mask[:, 1:] = valid_mask[:, 1:] * (1.0 - eos_positions[:, :-1].float())

per_token_reward = p2._compute_per_token_reward_lm_ce(
    result["readouts"], H_mid, ids_exp, valid_mask,
    h_mid_batch_map=bmap)
print(f"\nper_token_reward shape: {per_token_reward.shape}")  # [K*BS, T-1]
per_token_reward_k = per_token_reward.view(K, BS, -1)
print("Per-token reward stats:")
print(f"  abs mean: {per_token_reward_k.abs().mean().item():.4f}")
print(f"  K-std per (b, t) mean: {per_token_reward_k.std(dim=0).mean().item():.6f}")
print(f"  K-std per (b, t) max:  {per_token_reward_k.std(dim=0).max().item():.6f}")
print(f"  [k, b=0, t=100]: {per_token_reward_k[:, 0, 100].tolist()}")
print(f"  [k, b=0, t=500]: {per_token_reward_k[:, 0, 500].tolist()}")

# Now windowed reward
call_positions = result["call_positions"]
windowed, _complete = p2._windowed_reward(per_token_reward_k, call_positions, W)
print(f"\nWindowed reward shape: {windowed.shape}")  # [K, n_calls, BS]
print("Windowed reward stats:")
print(f"  abs mean: {windowed.abs().mean().item():.4f}")
print(f"  K-std per (call, sample) mean: {windowed.std(dim=0).mean().item():.6e}")
print(f"  K-std per (call, sample) max:  {windowed.std(dim=0).max().item():.6e}")
print(f"  First call [k, b=0]: {windowed[:, 0, 0].tolist()}")
