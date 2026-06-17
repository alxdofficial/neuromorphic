"""STAGE 1 smoke — build biomem_baseline + run compute_masked_reconstruction_loss
on a REAL MAE val batch. Confirms:
  1. constructs on SmolLM2-135M with task_mode=masked_reconstruction;
  2. compute_masked_reconstruction_loss runs fwd+bwd: FINITE loss + FINITE grads;
  3. gradients reach the regulator, cond vectors, readout, AND the query seeds;
  4. memory_shape is [B, <=16, 576];
  5. a few optimizer steps reduce the loss (the path actually learns).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.data_masked_reconstruction import make_sentence_dataloader
from scripts.train.train import to_device

device = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"
SRC = "meta-llama/Llama-3.2-1B"


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "masked_reconstruction"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.n_flat_codes = 16
    return cfg


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dl = make_sentence_dataloader(tok, batch_size=8, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
it = iter(dl)
batch = to_device(next(it), device)
print(f"batch: context {tuple(batch.context_ids.shape)}, k_slots={batch.k_slots}")

fails = []
cfg = matched(ReprConfig())
model = ReprLearningModel(cfg, variant="biomem_baseline").to(device)
model.task_mode = "masked_reconstruction"
model.train()

trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
n_enc = sum(p.numel() for n, p in trainable if n.startswith("encoder."))
n_tot = sum(p.numel() for _, p in trainable)
print(f"trainable: encoder {n_enc/1e6:.3f}M + shared(decoder LoRA/mask) "
      f"{(n_tot-n_enc)/1e6:.3f}M = {n_tot/1e6:.3f}M total")

opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)

# ---- fwd+bwd: finite loss, shapes ----
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_masked_reconstruction_loss(batch)
M = out["mae_M"]; nmask = out["mae_n_masked"]
B, T = batch.context_ids.shape
expect_M = float(batch.k_slots)
ok_M = abs(M - expect_M) < 0.5 and M <= 16.0
print(f"  memory_shape={out['memory_shape']}  M={M:.0f} (expect k_slots={expect_M:.0f}, "
      f"<=16: {'OK' if ok_M else 'BAD'})  n_masked={nmask:.0f}  "
      f"loss={out['loss'].item():.3f}  top1={out['top1_acc'].item():.3f}")
if not ok_M:
    fails.append(f"memory M={M} (expected {expect_M}, <=16)")
if not torch.isfinite(out["loss"]):
    fails.append("non-finite loss")
out["loss"].backward()
if not torch.isfinite(out["loss"]):
    fails.append("non-finite loss")
# all-finite check on the step-0 backward
nonfinite0 = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
if nonfinite0:
    fails.append(f"non-finite grads at step 0: {nonfinite0[:3]}")

# ---- REAL vs OFF executes ----
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    real = model.compute_masked_reconstruction_loss(batch)["loss_recon"].item()
    off = model.compute_masked_reconstruction_loss(batch, zero_memory=True)["loss_recon"].item()
print(f"  REAL={real:.3f}  OFF={off:.3f}  (untrained; OFF≈REAL expected pre-training)")

# ---- a few optimizer steps reduce the loss + accumulate grad-aliveness ----
# COLD-START NOTE: the regulator's last layer is zero-init (gate g=0 at step 0 -> a
# clean-slate grid that writes NOTHING), exactly like a LoRA B=0 adapter. So at the
# FIRST backward the edges W stay 0 and everything UPSTREAM of W (cond, query_seeds,
# seed_enc, in_proj, leak, regulator.fc1/fc2) gets zero grad. We therefore check
# grad-aliveness AFTER a few optimizer steps (once the gate has moved off zero),
# accumulating across steps + a few distinct batches — the mae_smoke.py methodology.
extra = [to_device(next(it), device) for _ in range(3)]
alive = set()
losses = []
for _ in range(8):
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        o = model.compute_masked_reconstruction_loss(batch)
    o["loss"].backward()
    for n, p in trainable:
        if p.grad is not None and p.grad.abs().sum() > 0:
            alive.add(n)
    torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
    opt.step(); losses.append(o["loss"].item())
for b in extra:
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.compute_masked_reconstruction_loss(b)["loss"].backward()
    for n, p in trainable:
        if p.grad is not None and p.grad.abs().sum() > 0:
            alive.add(n)
drop = losses[0] - min(losses)
print(f"  8-step loss: {losses[0]:.3f} → {losses[-1]:.3f} (min {min(losses):.3f}, Δ {drop:+.3f})")
if drop <= 0:
    fails.append("loss never improved in 8 steps")

# ---- grads reach the load-bearing learned objects (post-steps) ----
targets = {
    "regulator":   [n for n, _ in trainable if "encoder.regulator" in n],
    "cond":        [n for n, _ in trainable if n == "encoder.cond"],
    "readout":     [n for n, _ in trainable if "encoder.readout" in n],
    "query_seeds": [n for n, _ in trainable if "encoder.query_seeds" in n],
    "seed_enc":    [n for n, _ in trainable if "encoder.seed_enc" in n],
    "in_proj":     [n for n, _ in trainable if "encoder.in_proj" in n],
    "leak_raw":    [n for n, _ in trainable if "encoder.leak_raw" in n],
}
for grp, names in targets.items():
    got = [n for n in names if n in alive]
    ok = len(got) == len(names) and len(names) > 0
    print(f"  grad reaches {grp:12s}: {len(got)}/{len(names)} {'OK' if ok else 'MISSING'}")
    if not ok:
        fails.append(f"grad missing for {grp}: {[n for n in names if n not in alive]}")

print("\n" + ("BIOMEM SMOKE PASS — clear to launch the MAE training run"
              if not fails else "BIOMEM SMOKE FAIL:\n  " + "\n  ".join(fails)))
sys.exit(0 if not fails else 1)
