"""Independent ICAE readiness probe (icae_baseline). Canonical launch shape.

Verifies: build, B=2 emat_bio fwd+bwd bf16 w/ grads, M=144 @ d_llama budget,
REAL/SHUF/OFF actually mutate memory, faithfulness (passage perturbation moves
memory; slots are not a constant), read path has no silent no-op.
"""
import sys, time, torch
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

torch.manual_seed(0)
device = "cuda"
cfg = ReprConfig()
M = 144
cfg.icae_n_slots = M
cfg.n_flat_codes = M
CTX = 1024  # canonical launch chunk/window
print(f"[cfg] icae_n_slots={cfg.icae_n_slots} d_llama={cfg.d_llama} "
      f"lora_rank={cfg.icae_lora_rank} alpha={cfg.icae_lora_alpha} "
      f"targets={cfg.llama_lora_target_names}")
print(f"[budget] M*d_llama = {M*cfg.d_llama:,} floats (expect 294,912)")

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
dl = make_emat_bio_dataloader(tok, context_len=CTX, batch_size=2, n_pairs=22,
                              n_query=1, n_facts=3, split="train", world_seed=0,
                              stream_seed=1, pad_token_id=128_001, num_workers=0)
batch = next(iter(dl))

model = ReprLearningModel(cfg, variant="icae_baseline", llama_model=None).to(device)
model.train(True)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
enc_trainable = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("encoder."))
dec_trainable = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("decoder."))
slot_n = model.encoder.slots.numel()
norm_n = sum(p.numel() for p in model.encoder.norm.parameters())
enc_lora = sum(p.numel() for n, p in model.encoder.named_parameters()
               if p.requires_grad and "lora_" in n)
dec_lora = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and n.startswith("decoder.") and "lora_" in n)
print(f"[params] total={total:,} trainable={trainable:,}")
print(f"[params] enc_trainable={enc_trainable:,} (lora={enc_lora:,} slots={slot_n:,} norm={norm_n:,})")
print(f"[params] dec_trainable={dec_trainable:,} (lora={dec_lora:,})")

from scripts.repr_learning.train_repr_qa import to_device
batch = to_device(batch, device)
print(f"[data] ctx_tokens(row0)={int(batch.context_mask[0].sum())} "
      f"B={batch.context_ids.shape[0]} T_ctx={batch.context_ids.shape[1]}")

# ---- Faithfulness: does memory depend on the passage? (run encoder twice) ----
def encode_memory(ctx_ids, ctx_mask):
    embed = model.decoder.llama.get_input_embeddings()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce = embed(ctx_ids)
        st = model.encoder.init_streaming_state(ctx_ids.shape[0], device, ce.dtype)
        st, _ = model.encoder.streaming_write(st, ce, ctx_mask, chunk_offset=0)
        mem, _ = model.encoder.finalize_memory(st)
    return mem.float()

mem_real = encode_memory(batch.context_ids, batch.context_mask)
# Perturb: shuffle the token order of row0's valid passage region heavily.
ctx2 = batch.context_ids.clone()
valid0 = batch.context_mask[0].bool()
idx = valid0.nonzero(as_tuple=True)[0]
perm = idx[torch.randperm(len(idx), device=device)]
ctx2[0, idx] = batch.context_ids[0, perm]
mem_pert = encode_memory(ctx2, batch.context_mask)
delta_row0 = (mem_real[0] - mem_pert[0]).norm().item()
delta_row1 = (mem_real[1] - mem_pert[1]).norm().item()  # unchanged passage -> ~0
# Row-distinctness (so SHUF is meaningful): cosine between the two rows' mem.
r0 = mem_real[0].flatten(); r1 = mem_real[1].flatten()
row_cos = torch.nn.functional.cosine_similarity(r0, r1, dim=0).item()
# Per-slot collapse: mean pairwise cosine among the M slots of row0.
slots0 = torch.nn.functional.normalize(mem_real[0], dim=-1)
slot_cos = (slots0 @ slots0.T)
off = slot_cos[~torch.eye(M, dtype=torch.bool, device=slot_cos.device)].mean().item()
print(f"[faith] mem norm(real,row0)={mem_real[0].norm().item():.3f} "
      f"per-slot L2 mean={mem_real[0].norm(dim=-1).mean().item():.3f} (NormMatch target 0.9)")
print(f"[faith] ||mem(real)-mem(perturbed-passage)|| row0={delta_row0:.4f} (perturbed) "
      f"row1={delta_row1:.4f} (untouched, expect ~0)")
print(f"[faith] row0-vs-row1 mem cosine={row_cos:.4f} (distinct rows => SHUF nontrivial)")
print(f"[faith] mean off-diag slot-slot cosine row0={off:.4f} (1.0 => all slots collapsed)")

# ---- REAL forward+backward ----
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_qa_loss(batch, window_size=CTX)
loss = out["loss"]            # differentiable; loss_recon is detached
loss_report = out["loss_recon"]
print(f"[grad] loss.requires_grad={loss.requires_grad} grad_fn={loss.grad_fn is not None}")
loss.backward()
torch.cuda.synchronize()
fwd_bwd_s = time.time() - t0
print(f"[REAL] loss={float(loss):.4f} top1={float(out['top1_acc']):.4f} "
      f"memory_shape={out['memory_shape']}")

g_loraB = [(n, p.grad) for n, p in model.encoder.named_parameters()
           if p.requires_grad and "lora_B" in n and p.grad is not None]
slot_grad = model.encoder.slots.grad
norm_scale_grad = model.encoder.norm.scale.grad
lora_gn = (sum((g.float()**2).sum() for _, g in g_loraB).sqrt().item() if g_loraB else 0.0)
dec_loraB = [p.grad for n, p in model.decoder.named_parameters()
             if p.requires_grad and "lora_B" in n and p.grad is not None]
dec_gn = (sum((g.float()**2).sum() for g in dec_loraB).sqrt().item() if dec_loraB else 0.0)
print(f"[grad] enc lora_B groups w/grad={len(g_loraB)} lora_grad_norm={lora_gn:.4e}")
print(f"[grad] slots.grad_norm={(slot_grad.float().norm().item() if slot_grad is not None else 'NONE'):.4e}")
print(f"[grad] norm.scale.grad={(norm_scale_grad.item() if norm_scale_grad is not None else 'NONE')}")
print(f"[grad] dec lora_B groups w/grad={len(dec_loraB)} dec_grad_norm={dec_gn:.4e}")
model.zero_grad(set_to_none=True)

# ---- OFF / SHUF (memory mutation controls) ----
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    off = model.compute_qa_loss(batch, window_size=CTX, zero_memory=True)
    shuf = model.compute_qa_loss(batch, window_size=CTX, shuffle_memory=True)
print(f"[gate] REAL={float(loss):.4f}  OFF={float(off['loss_recon']):.4f}  "
      f"SHUF={float(shuf['loss_recon']):.4f}")
print(f"[gate] OFF-REAL={float(off['loss_recon'])-float(loss):+.4f}  "
      f"SHUF-REAL={float(shuf['loss_recon'])-float(loss):+.4f}  "
      f"(untrained; sign not expected, just must DIFFER => memory mutates)")
print(f"[time] fwd+bwd B=2 = {fwd_bwd_s:.2f}s  peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"[mem] M emitted={out['memory_shape'][1]} (expect {M}); "
      f"per-slot dim={out['memory_shape'][2]} (expect {cfg.d_llama}); "
      f"floats={out['memory_shape'][1]*out['memory_shape'][2]:,}")
