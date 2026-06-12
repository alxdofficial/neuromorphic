"""ICAE launch-readiness probe (assigned arm). B=2 emat_bio, bf16 autocast."""
import sys, time, torch
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

torch.manual_seed(0)
device = "cuda"
cfg = ReprConfig()
# Mirror launch config knobs that touch ICAE.
M = 144
cfg.icae_n_slots = M
cfg.n_flat_codes = M
print(f"[cfg] icae_n_slots={cfg.icae_n_slots} d_llama={cfg.d_llama} "
      f"icae_lora_rank={cfg.icae_lora_rank} alpha={cfg.icae_lora_alpha} "
      f"targets={cfg.llama_lora_target_names}")

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
# emat_bio loader at the launch shape: n_pairs=12, n_facts=3, chunk/window=640.
dl = make_emat_bio_dataloader(tok, context_len=640, batch_size=2, n_pairs=12,
                              n_query=1, n_facts=3, split="train", world_seed=0,
                              stream_seed=1, pad_token_id=128_001, num_workers=0)
batch = next(iter(dl))

model = ReprLearningModel(cfg, variant="icae_baseline", llama_model=None).to(device)
model.train(True)

# Param accounting.
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
enc_trainable = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("encoder."))
slot_n = model.encoder.slots.numel()
norm_n = sum(p.numel() for p in model.encoder.norm.parameters())
enc_lora = sum(p.numel() for n, p in model.encoder.named_parameters()
               if p.requires_grad and "lora_" in n)
dec_trainable = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("decoder."))
print(f"[params] total={total:,} trainable={trainable:,}")
print(f"[params] enc_trainable={enc_trainable:,} (lora={enc_lora:,} slots={slot_n:,} norm={norm_n:,})")
print(f"[params] dec_trainable={dec_trainable:,}")

from scripts.repr_learning.train_repr_qa import to_device
batch = to_device(batch, device)

# ---- REAL forward+backward ----
t0 = time.time()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_qa_loss(batch, window_size=640)
loss = out["loss_recon"]
print(f"[REAL] loss={float(loss):.4f} top1={float(out['top1_acc']):.4f} "
      f"memory_shape={out['memory_shape']}")
loss.backward()
fwd_bwd_s = time.time() - t0

# grad check on the three trainable groups
g_lora = [p.grad for n, p in model.encoder.named_parameters()
          if p.requires_grad and "lora_B" in n and p.grad is not None]
slot_grad = model.encoder.slots.grad
norm_scale_grad = model.encoder.norm.scale.grad
lora_grad_norm = (sum((g.float()**2).sum() for g in g_lora).sqrt().item()
                  if g_lora else 0.0)
print(f"[grad] enc lora_B groups w/grad={len(g_lora)} lora_grad_norm={lora_grad_norm:.4e}")
print(f"[grad] slots.grad_norm={(slot_grad.float().norm().item() if slot_grad is not None else 'NONE')}")
print(f"[grad] norm.scale.grad={(norm_scale_grad.item() if norm_scale_grad is not None else 'NONE')}")
# decoder LoRA (read-side)
dec_lora_g = [p.grad for n, p in model.decoder.named_parameters()
              if p.requires_grad and "lora_B" in n and p.grad is not None]
print(f"[grad] dec lora_B groups w/grad={len(dec_lora_g)}")
model.zero_grad(set_to_none=True)

# ---- OFF (zero memory) ----
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    off = model.compute_qa_loss(batch, window_size=640, zero_memory=True)
# ---- SHUF (roll memory) ----
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    shuf = model.compute_qa_loss(batch, window_size=640, shuffle_memory=True)
print(f"[gate] REAL={float(loss):.4f}  OFF={float(off['loss_recon']):.4f}  "
      f"SHUF={float(shuf['loss_recon']):.4f}")
print(f"[gate] OFF-REAL={float(off['loss_recon'])-float(loss):+.4f}  "
      f"SHUF-REAL={float(shuf['loss_recon'])-float(loss):+.4f}")
print(f"[time] fwd+bwd B=2 = {fwd_bwd_s:.2f}s  peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"[mem] M emitted = {out['memory_shape'][1]} (expect {M}); per-slot dim = {out['memory_shape'][2]} (expect {cfg.d_llama})")
