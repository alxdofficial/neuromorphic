"""Check why reward doesn't diverge even though readouts do."""
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
# Use TRAINED codebook from cycle 00 to match real conditions
import os
if os.path.exists("outputs/v12/cycle_00/codebook.pt"):
    cb_ckpt = torch.load("outputs/v12/cycle_00/codebook.pt", map_location=device, weights_only=False)
    cb_config = cb_ckpt["config"]
    vqvae = ActionVQVAE(
        action_dim=cb_config["action_dim"], latent_dim=cb_config["latent_dim"],
        hidden=cb_config["hidden"], num_levels=cb_config["num_levels"],
        codes_per_level=cb_config["codes_per_level"],
    ).to(device)
    vqvae.load_state_dict(cb_ckpt["state_dict"])
    print("Using TRAINED codebook from cycle_00")
else:
    vqvae = ActionVQVAE(action_dim=config.mod_out, num_levels=1,
                        codes_per_level=256).to(device)
    vqvae.set_normalization(torch.zeros(1, config.mod_out, device=device),
                            torch.ones(1, config.mod_out, device=device) * 0.5)
    print("Using RANDOM codebook")
vqvae.train(False)
for p in vqvae.parameters(): p.requires_grad = False

BS, K, T = 4, 8, 256

model.memory.initialize_states(BS, device)
model._initialized = True

dl = create_dataloader("A", tokenizer, batch_size=BS, seq_length=T, seed=42, max_steps=1)
batch = next(iter(dl))
input_ids = batch.input_ids.to(device)

# Expand memory
mem = model.memory
init_state = {k: getattr(mem, k).clone() for k in
              ["h", "msg", "W", "decay", "hebbian", "s_mem_live",
               "s_mem_ema_fast", "prev_readout", "readout_drift"]}
for key, val in init_state.items():
    expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
    setattr(mem, key, expanded.clone())

from src.phase2.trainer import Phase2Trainer
p2 = Phase2Trainer(model=model, vqvae=vqvae, dataloader=dl, config=config,
                   device=device, group_size=K)
H_mid = p2._run_lower_scan(input_ids)
ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])
bmap = torch.arange(K * BS, device=device) % BS

result = mem.forward_segment_phase2(
    H_mid, ids_exp, model.lm, vqvae, tau=1.0, sample=True,
    h_mid_batch_map=bmap)

readouts = result["readouts"]   # [K*BS, T, D]

# Now check H_enriched and logits divergence
lm = model.lm
mem_scale = lm.mem_scale  # [D]
print(f"mem_scale stats: mean={mem_scale.mean().item():.3f}, "
      f"abs_mean={mem_scale.abs().mean().item():.3f}, "
      f"max={mem_scale.abs().max().item():.3f}")

# H_enriched at original BS-size H_mid, gathered
H_mid_exp = H_mid[bmap]  # [K*BS, T, D]
H_enriched = H_mid_exp + mem_scale * readouts
print(f"H_mid abs mean: {H_mid_exp.abs().mean().item():.3f}")
print(f"mem_scale * readouts abs mean: {(mem_scale * readouts).abs().mean().item():.3f}")
print(f"H_enriched abs mean: {H_enriched.abs().mean().item():.3f}")

# Divergence across K
H_mid_k = H_mid_exp.view(K, BS, T, -1)
readouts_k = readouts.view(K, BS, T, -1)
H_enriched_k = H_enriched.view(K, BS, T, -1)
print(f"H_mid std across K: {H_mid_k.std(dim=0).mean().item():.6f} (should be ~0)")
print(f"readouts std across K: {readouts_k.std(dim=0).mean().item():.6f}")
print(f"H_enriched std across K: {H_enriched_k.std(dim=0).mean().item():.6f}")
print(f"  (relative to H_enriched mag: {H_enriched_k.std(dim=0).mean().item() / H_enriched.abs().mean().item() * 100:.2f}%)")

# Now run upper scan chunked and get logits. Compare CE across K.
print()
print("Running upper scan + head on K*BS samples...")
# Fresh upper carries
lm._carries[config.scan_split_at:] = [None] * (config.L_total - config.scan_split_at)

# Process in sub_bs=8 chunks to match trainer pattern
sub_bs = 8
target_ids = torch.roll(input_ids, -1, dims=1)  # shift left by 1
target_ids_exp = target_ids.unsqueeze(0).expand(K, *target_ids.shape).reshape(K*BS, T)

all_ce = torch.zeros(K*BS, T-1, device=device, dtype=torch.float32)
for b_start in range(0, K*BS, sub_bs):
    b_end = min(b_start + sub_bs, K*BS)
    sub_H = H_enriched[b_start:b_end]
    # reset upper carries per sub-batch
    for i in range(config.scan_split_at, config.L_total):
        lm._carries[i] = None
    H_up = lm.forward_scan_upper(sub_H)
    logits = lm.forward_output(H_up)  # [sub_bs, T, V]
    # CE shifted: logits[t-1] predicts target[t]
    shifted_logits = logits[:, :-1]
    shifted_targets = target_ids_exp[b_start:b_end, 1:]
    ce = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        shifted_targets.reshape(-1),
        reduction="none").reshape(b_end-b_start, T-1)
    all_ce[b_start:b_end] = ce

# Now look at CE divergence across K
ce_k = all_ce.view(K, BS, T-1)
print(f"CE shape: {ce_k.shape}")
print(f"CE abs mean: {ce_k.abs().mean().item():.4f}")
print(f"CE std across K (per position): {ce_k.std(dim=0).mean().item():.6f}")
print(f"CE std across all: {ce_k.std().item():.4f}")
print(f"CE std relative to magnitude: {ce_k.std(dim=0).mean().item() / ce_k.abs().mean().item() * 100:.3f}%")

# Check per-K mean CE
per_k_ce = ce_k.mean(dim=(1, 2))
print(f"Per-K mean CE (should differ across K):")
for k in range(K):
    print(f"  k={k}: {per_k_ce[k].item():.6f}")
print(f"  Range: {per_k_ce.max().item() - per_k_ce.min().item():.6e}")
print(f"  Std: {per_k_ce.std().item():.6e}")
