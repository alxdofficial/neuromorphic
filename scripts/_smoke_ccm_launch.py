"""Launch-readiness smoke for ccm_baseline ONLY. Builds CCM, runs B=2 emat_bio
fwd+bwd through compute_qa_loss (REAL/SHUF/OFF), checks grad flow into CCM
trainable params, counts params, times s/step. Constructs ONLY ccm (no OOM)."""
import sys, time, torch
sys.path.insert(0, "/home/alex/code/neuromorphic")
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

torch.manual_seed(0)
dev = "cuda"
M = 144
cfg = ReprConfig(batch_size=2, n_flat_codes=M)
cfg.ccm_n_comp = M
cfg.icae_n_slots = M
cfg.autocompressor_n_slots = M
cfg.use_llama_lora = True

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
print("loading shared frozen llama (decoder side)...")
llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

print("building ccm_baseline...")
model = ReprLearningModel(cfg, variant="ccm_baseline", llama_model=llama).to(dev)

# Param accounting
enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
dec_train = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
tot_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[PARAMS] encoder_trainable={enc_train:,}  decoder_trainable={dec_train:,}  total_trainable={tot_train:,}")
# breakdown of encoder trainables
for n, p in model.encoder.named_parameters():
    if p.requires_grad:
        print(f"   enc.{n}: {tuple(p.shape)} = {p.numel():,}")

# n_comp emitted check
print(f"[CCM] n_comp={model.encoder.n_comp}  fold={model.encoder.fold}")

# build emat_bio batch B=2
dl = make_emat_bio_dataloader(tok, context_len=640, batch_size=2, n_pairs=12,
                              n_query=1, n_facts=3, world_seed=0, stream_seed=42,
                              split="train", num_workers=0)
batch = next(iter(dl))
for k in ("context_ids", "context_mask", "question_ids", "question_mask",
          "answer_ids", "answer_content_mask"):
    v = getattr(batch, k, None)
    if v is not None:
        batch = batch._replace(**{k: v.to(dev)}) if hasattr(batch, "_replace") else batch
# move all tensors
import dataclasses
if dataclasses.is_dataclass(batch):
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            setattr(batch, f.name, v.to(dev))
print(f"[BATCH] context={tuple(batch.context_ids.shape)} "
      f"q={tuple(batch.question_ids.shape)} a={tuple(batch.answer_ids.shape)} "
      f"ctx_tokens(row0)={int(batch.context_mask[0].sum())}")

model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

def run(label, **kw):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.compute_qa_loss(batch, window_size=640, **kw)
    return out["loss"]

# REAL/SHUF/OFF forward losses (no_grad first to confirm gate distinctness)
model.eval()
with torch.no_grad():
    real = run("REAL").item()
    shuf = run("SHUF", shuffle_memory=True).item()
    off = run("OFF", zero_memory=True).item()
print(f"[GATE eval] REAL={real:.4f}  SHUF={shuf:.4f}  OFF={off:.4f}  "
      f"SHUF-REAL={shuf-real:+.4f}  OFF-REAL={off-real:+.4f}")

# backward + grad flow check (train mode = real path with ckpt_stream)
model.train()
opt.zero_grad()
t0 = time.time()
loss = run("REAL_train")
loss.backward()
t_fb = time.time() - t0
# grad norms by group
g_enc = [(n, p.grad) for n, p in model.encoder.named_parameters() if p.requires_grad]
n_with_grad = sum(1 for _, g in g_enc if g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)
n_total = len(g_enc)
gnorm_lora = sum((p.grad.norm().item()**2) for n, p in model.encoder.named_parameters()
                 if p.requires_grad and p.grad is not None and "lora" in n)**0.5
gnorm_comp = sum((p.grad.norm().item()**2) for n, p in model.encoder.named_parameters()
                 if p.requires_grad and p.grad is not None and "comp_embed" in n)**0.5
gnorm_norm = sum((p.grad.norm().item()**2) for n, p in model.encoder.named_parameters()
                 if p.requires_grad and p.grad is not None and n.startswith("norm"))**0.5
any_nan = any(p.grad is not None and not torch.isfinite(p.grad).all()
              for p in model.parameters() if p.requires_grad)
print(f"[BWD] loss={loss.item():.4f}  fwd+bwd={t_fb:.2f}s  enc_params_with_live_grad={n_with_grad}/{n_total}")
print(f"[GRAD] gnorm_lora={gnorm_lora:.4e}  gnorm_comp_embed={gnorm_comp:.4e}  gnorm_norm={gnorm_norm:.4e}  any_nan={any_nan}")

# timing a couple optimizer steps (steady-state s/step at B=2)
opt.step()
torch.cuda.synchronize()
times = []
for i in range(3):
    opt.zero_grad()
    t0 = time.time()
    loss = run(f"step{i}")
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    times.append(time.time() - t0)
print(f"[STEP] B=2 s/step (3 iters): {[f'{x:.2f}' for x in times]}  mean={sum(times)/len(times):.2f}s")
print(f"[MEM] peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
      f"reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB")
print("DONE")
