"""Benchmark phase 2 rollout + grpo step at each curriculum window.

Runs a small fixed workload and reports tok/s and peak VRAM. Designed
to be run before/after each optimization stage to measure impact.
"""
import sys, time, gc, json
import torch

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

from src.model.config import Config
from src.model.model import Model
from src.data import create_dataloader, get_tokenizer, get_special_token_ids
from src.codebook import ActionVQVAE
from src.phase2.trainer import Phase2Trainer

print("Loading bootstrap ckpt...")
ckpt = torch.load("outputs/v12/bootstrap.pt", map_location=device, weights_only=False)
config = ckpt["config"]
tokenizer = get_tokenizer("tinyllama")
config.vocab_size = len(tokenizer)
config.eot_id = get_special_token_ids(tokenizer).get(
    "eos_token_id", tokenizer.eos_token_id)

model = Model(config).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.load_runtime_state(ckpt.get("runtime_state", {}))
del ckpt
gc.collect(); torch.cuda.empty_cache()

for p in model.parameters():
    p.requires_grad = False
for p in (model.memory.mod_w1, model.memory.mod_b1,
          model.memory.mod_w2, model.memory.mod_b2):
    p.requires_grad = True

vqvae = ActionVQVAE(action_dim=config.mod_out, num_levels=1,
                    codes_per_level=256).to(device)
vqvae.train(False)
for p in vqvae.parameters():
    p.requires_grad = False

BS_LADDER = {512: 24, 1024: 16, 2048: 12, 4096: 8}
K = 8

results = {}
for W, BS in BS_LADDER.items():
    T = 2 * W

    if model.memory.h.shape[0] != BS:
        model.memory.resize_to_bs(BS)
        model._initialized = model.memory._initialized
    model.lm._carries = [None] * config.L_total

    dl = create_dataloader("A", tokenizer, batch_size=BS,
                           seq_length=T, seed=42, max_steps=5)
    p2 = Phase2Trainer(model=model, vqvae=vqvae, dataloader=dl, config=config,
                       device=device, group_size=K)
    p2.reward_window = W

    batch = next(iter(dl))
    result = p2.rollout(batch)
    _ = p2.grpo_step(result)
    del result
    gc.collect(); torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    t_rolls = []
    t_grpos = []
    for step, batch in enumerate(dl):
        if step >= 3:
            break
        torch.cuda.synchronize()
        t0 = time.time()
        result = p2.rollout(batch)
        torch.cuda.synchronize()
        t_rolls.append(time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        _ = p2.grpo_step(result)
        torch.cuda.synchronize()
        t_grpos.append(time.time() - t0)
        del result

    peak = torch.cuda.max_memory_allocated() / 1e9
    avg_roll = sum(t_rolls) / len(t_rolls)
    avg_grpo = sum(t_grpos) / len(t_grpos)
    tok_per_step = BS * T
    tok_s = tok_per_step / (avg_roll + avg_grpo)
    results[W] = {
        "BS": BS, "T": T, "tok_per_step": tok_per_step,
        "rollout_s": round(avg_roll, 3), "grpo_s": round(avg_grpo, 3),
        "peak_gb": round(peak, 2), "tok_s": round(tok_s, 0),
    }
    print(f"W={W:>4d} T={T:>5d} BS={BS:>2d}: "
          f"roll={avg_roll:.2f}s grpo={avg_grpo:.2f}s "
          f"peak={peak:.1f}GB tok/s={tok_s:.0f}")

print("\nJSON:", json.dumps(results, indent=2))
