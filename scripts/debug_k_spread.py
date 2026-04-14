"""Debug why K trajectories don't diverge in phase 2 rollouts.

Instrument the rollout to check divergence at each level:
- Do K trajectories get DIFFERENT VQ codes?
- After applying quantized actions, do K W's diverge?
- Do K memory states (h, msg) diverge?
- Do K readouts diverge?
- Do K logits (from upper scan + head) diverge?
"""
import sys, gc
import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

from src.model.config import Config
from src.model.model import Model
from src.data import create_dataloader, get_tokenizer, get_special_token_ids
from src.codebook import ActionVQVAE

# Load bootstrap
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
vqvae = ActionVQVAE(action_dim=config.mod_out, num_levels=1,
                    codes_per_level=256).to(device)
# Fit vqvae briefly on random actions so it has nontrivial codebook
torch.manual_seed(0)
act_mean = torch.randn(1, config.mod_out, device=device) * 0.1
act_std = torch.ones(1, config.mod_out, device=device) * 0.5
vqvae.set_normalization(act_mean, act_std)
vqvae.train(False)
for p in vqvae.parameters(): p.requires_grad = False

# Setup: BS=4, K=8 → K*BS=32 (small for clarity)
BS = 4
K = 8
T = 256  # short rollout for tracing

model.memory.initialize_states(BS, device)
model._initialized = True
model.lm._carries = [None] * config.L_total

# Warm up memory a bit
dl_warm = create_dataloader("A", tokenizer, batch_size=BS, seq_length=T, seed=42, max_steps=2)
with torch.no_grad():
    for i, batch in enumerate(dl_warm):
        if i >= 1: break
        input_ids = batch.input_ids.to(device)
        model.forward_chunk(input_ids, use_memory=True)

# Now do a rollout manually and check divergence
dl = create_dataloader("A", tokenizer, batch_size=BS, seq_length=T, seed=42, max_steps=2)
batch = next(iter(dl))
input_ids = batch.input_ids.to(device)

# Expand memory to K*BS
mem = model.memory
init_state = {
    "h": mem.h.clone(), "msg": mem.msg.clone(), "W": mem.W.clone(),
    "decay": mem.decay.clone(), "hebbian": mem.hebbian.clone(),
    "s_mem_live": mem.s_mem_live.clone(),
    "s_mem_ema_fast": mem.s_mem_ema_fast.clone(),
    "prev_readout": mem.prev_readout.clone(),
    "readout_drift": mem.readout_drift.clone(),
}
for key, val in init_state.items():
    expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
    setattr(mem, key, expanded.clone())

# Verify: at rollout start, all K trajectories should be IDENTICAL
print("=" * 60)
print("At rollout start — all K copies should be identical:")
h = mem.h.view(K, BS, *mem.h.shape[1:])
W = mem.W.view(K, BS, *mem.W.shape[1:])
print(f"  h across K: std = {h.std(dim=0).mean().item():.2e}")
print(f"  W across K: std = {W.std(dim=0).mean().item():.2e}")

# Run lower scan (BS-sized, same for all K)
from src.phase2.trainer import Phase2Trainer
p2 = Phase2Trainer(model=model, vqvae=vqvae, dataloader=dl, config=config,
                   device=device, group_size=K)
H_mid = p2._run_lower_scan(input_ids)
print(f"  H_mid shape: {H_mid.shape}")

# Run the phase 2 rollout
ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])
bmap = torch.arange(K * BS, device=device) % BS

# Manually step through to check divergence at each stage
# We can't easily instrument the inner loop without modifying memory.py,
# so let's just run the full rollout and examine the results
result = mem.forward_segment_phase2(
    H_mid, ids_exp, model.lm, vqvae, tau=1.0, sample=True,
    h_mid_batch_map=bmap)

print()
print("=" * 60)
print("After rollout — checking divergence across K trajectories:")
readouts = result["readouts"].view(K, BS, T, -1)
print(f"  readouts shape: {readouts.shape}")
print(f"  readouts std across K: {readouts.std(dim=0).mean().item():.6f}")
print(f"  readouts abs mean: {readouts.abs().mean().item():.6f}")
print(f"  readouts[k=0, b=0, t=0, :8]: {readouts[0, 0, 0, :8].tolist()}")
print(f"  readouts[k=1, b=0, t=0, :8]: {readouts[1, 0, 0, :8].tolist()}")
print(f"  Diff k=0 vs k=1 at t=0: {(readouts[0] - readouts[1]).abs().mean().item():.6f}")
print(f"  Diff k=0 vs k=1 at t=T-1: {(readouts[0, :, -1] - readouts[1, :, -1]).abs().mean().item():.6f}")

# Check codes diversity
codes = result["codes"]  # [n_calls, K*BS, NC, num_levels]
codes = codes.view(codes.shape[0], K, BS, *codes.shape[2:])
print()
print(f"  codes shape: {codes.shape}")
# Are codes across K different? (they should be)
print(f"  codes[call=0, k=0, b=0]: {codes[0, 0, 0].flatten().tolist()[:10]}")
print(f"  codes[call=0, k=1, b=0]: {codes[0, 1, 0].flatten().tolist()[:10]}")
print(f"  codes[call=0, k=7, b=0]: {codes[0, 7, 0].flatten().tolist()[:10]}")
# Count unique codes per (call, b) across K
unique_per_slot = []
for c in range(codes.shape[0]):
    for b in range(BS):
        for nc in range(codes.shape[3]):
            u = torch.unique(codes[c, :, b, nc, 0]).numel()
            unique_per_slot.append(u)
import statistics
print(f"  Unique codes per (call, sample, cell) across K: mean={statistics.mean(unique_per_slot):.2f}, max={max(unique_per_slot)}")

# Check mem state divergence at end
print()
print("At end of rollout — K state divergence:")
h_end = mem.h.view(K, BS, *mem.h.shape[1:])
W_end = mem.W.view(K, BS, *mem.W.shape[1:])
prev_readout_end = mem.prev_readout.view(K, BS, *mem.prev_readout.shape[1:])
print(f"  h_end std across K: {h_end.std(dim=0).mean().item():.6f} (abs mean: {h_end.abs().mean().item():.3f})")
print(f"  W_end std across K: {W_end.std(dim=0).mean().item():.6f} (abs mean: {W_end.abs().mean().item():.3f})")
print(f"  prev_readout std across K: {prev_readout_end.std(dim=0).mean().item():.6f} (abs mean: {prev_readout_end.abs().mean().item():.3f})")
