"""Trace K trajectory divergence over time — does W's K-spread grow or stabilize?

If the hypothesis is correct (convex EMA causes mean reversion), W_k(t)
trajectories will show bounded variance that converges to a constant
regardless of t, not growing variance.
"""
import torch, gc
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

from src.model.config import Config
from src.model.model import Model
from src.data import create_dataloader, get_tokenizer, get_special_token_ids
from src.codebook import ActionVQVAE

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

# BS=2, K=8, long rollout. We'll manually step through and record W over time.
BS = 2
K = 8
T = 2048  # long enough to see convergence

model.memory.initialize_states(BS, device)
model._initialized = True
model.lm._carries = [None] * config.L_total

# Warm up
dl_warm = create_dataloader("A", tokenizer, batch_size=BS, seq_length=256, seed=42, max_steps=2)
with torch.no_grad():
    for i, batch in enumerate(dl_warm):
        if i >= 1: break
        model.forward_chunk(batch.input_ids.to(device), use_memory=True)

# Expand to K*BS
mem = model.memory
init_state = {k: getattr(mem, k).clone() for k in
              ["h", "msg", "W", "decay", "hebbian", "s_mem_live",
               "s_mem_ema_fast", "prev_readout", "readout_drift"]}
for key, val in init_state.items():
    expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
    setattr(mem, key, expanded.clone())

# Now do a rollout and record W at intervals
dl = create_dataloader("A", tokenizer, batch_size=BS, seq_length=T, seed=43, max_steps=1)
batch = next(iter(dl))
input_ids = batch.input_ids.to(device)

from src.phase2.trainer import Phase2Trainer
p2 = Phase2Trainer(model=model, vqvae=vqvae, dataloader=dl, config=config,
                   device=device, group_size=K)
H_mid = p2._run_lower_scan(input_ids)
ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])
bmap = torch.arange(K * BS, device=device) % BS

# Monkey-patch to snapshot W periodically. Easier: just run full rollout and
# observe the final W and readouts K-spread pattern.
result = mem.forward_segment_phase2(
    H_mid, ids_exp, model.lm, vqvae, tau=1.0, sample=True,
    h_mid_batch_map=bmap)

readouts = result["readouts"].view(K, BS, T, -1)  # [K, BS, T, D]

# K-std per (b, t) across K, then analyze how it evolves with t
# Average over D and BS
k_std_per_t = readouts.std(dim=0).abs().mean(dim=(0, 2))  # [T]

print("Readout K-std over time (averaged over BS, D):")
print(f"  t=0:       {k_std_per_t[0].item():.6f}")
print(f"  t=100:     {k_std_per_t[100].item():.6f}")
print(f"  t=500:     {k_std_per_t[500].item():.6f}")
print(f"  t=1000:    {k_std_per_t[1000].item():.6f}")
print(f"  t=1500:    {k_std_per_t[1500].item():.6f}")
print(f"  t=T-1:     {k_std_per_t[-1].item():.6f}")
print(f"  Is it growing (t=T vs t=500)? {k_std_per_t[-1].item() / k_std_per_t[500].item():.2f}x")

# Also check per-window K-std in equal-size chunks
n_windows = 8
window_T = T // n_windows
per_window_std = []
for w in range(n_windows):
    start = w * window_T
    end = start + window_T
    # Get readouts in this window, compute K-std per (b, t, D), avg within window
    chunk = readouts[:, :, start:end, :]  # [K, BS, window_T, D]
    chunk_k_std = chunk.std(dim=0).mean().item()
    per_window_std.append(chunk_k_std)
print(f"\nK-std per window (should stabilize if mean-reverting):")
for w, v in enumerate(per_window_std):
    print(f"  window {w} (t={w*window_T}..{(w+1)*window_T}): K-std = {v:.6f}")

# W K-std at end
W_end = mem.W.view(K, BS, *mem.W.shape[1:])
W_k_std = W_end.std(dim=0).abs().mean().item()
W_magnitude = W_end.abs().mean().item()
print(f"\nFinal W K-spread: {W_k_std:.6f} (|W|_mean = {W_magnitude:.4f})")
print(f"  Relative: {W_k_std / W_magnitude * 100:.2f}%")

# The hypothesis: a pure mean-reverting process with γ=0.02 has equilibrium
# variance = γ / (2-γ) * Var(driving_noise).
# At γ=0.02, equilibrium variance / noise variance = 0.01.
# So W K-std should be about 0.1 * |delta_W_std|.
# Let's check if the W K-spread is at equilibrium or still growing.

# Per-token reward K-divergence over time (if LM CE was diverging sustainably,
# later tokens would have higher spread).
# Compute CE per position.
from src.phase2.trainer import Phase2Trainer as _P2T
target_ids = torch.roll(input_ids, -1, dims=1)
target_ids_exp = target_ids.unsqueeze(0).expand(K, *target_ids.shape).reshape(K*BS, T)

mem_scale = model.lm.mem_scale
H_mid_gathered = H_mid[bmap]
H_enriched = H_mid_gathered + mem_scale * result["readouts"]

sub_bs = 8
all_ce = torch.zeros(K*BS, T-1, device=device, dtype=torch.float32)
for b_start in range(0, K*BS, sub_bs):
    b_end = min(b_start + sub_bs, K*BS)
    for i in range(config.scan_split_at, config.L_total):
        model.lm._carries[i] = None
    H_up = model.lm.forward_scan_upper(H_enriched[b_start:b_end])
    logits = model.lm.forward_output(H_up)
    shifted_logits = logits[:, :-1]
    shifted_targets = target_ids_exp[b_start:b_end, 1:]
    ce = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        shifted_targets.reshape(-1),
        reduction="none").reshape(b_end-b_start, T-1)
    all_ce[b_start:b_end] = ce

ce_k = all_ce.view(K, BS, T-1)
ce_std_per_t = ce_k.std(dim=0).abs().mean(dim=0)  # [T-1]
print(f"\nPer-token CE K-std over time (should grow if trajectories compound):")
n_ts = T // 200
for w in range(n_ts):
    start = w * 200
    end = min(start + 200, T-1)
    chunk_std = ce_std_per_t[start:end].mean().item()
    print(f"  t={start}..{end}: CE K-std = {chunk_std:.6f}")
