"""Canonical-config readiness probe for autocompressor_baseline.
Launch cfg: chunk=1024, window=1024, mem-tokens=144, emat_bio, B=2."""
import os, time, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader
from transformers import AutoTokenizer
import dataclasses

torch.manual_seed(0)
DEV = "cuda"
CHUNK, WIN, M, B = 1024, 1024, 144, 2

cfg = ReprConfig(
    batch_size=B,
    fixed_window_size=WIN,
    max_window_size=CHUNK,
    max_steps=8000,
    warmup_steps=500,
    use_llama_lora=True,
    grad_checkpoint_stream=True,
)
cfg.icae_n_slots = M
cfg.ccm_n_comp = M
cfg.autocompressor_n_slots = M
cfg.n_flat_codes = M
print(f"[cfg] autocompressor_n_slots={cfg.autocompressor_n_slots}, n_flat_codes={cfg.n_flat_codes}, "
      f"d_llama={cfg.d_llama}, lora_rank={cfg.autocompressor_lora_rank}, lora_alpha={cfg.autocompressor_lora_alpha}")
print(f"[cfg] lora_targets={cfg.llama_lora_target_names}")
print(f"[budget] memory floats = M*d_llama = {M*cfg.d_llama:,}")

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
model = ReprLearningModel(cfg, variant="autocompressor_baseline").to(DEV)
model.train()

enc = model.encoder
print(f"[enc class] {type(enc).__name__}")
n_train = model.n_trainable_params()
enc_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
dec_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("decoder"))
print(f"\n[params] model.n_trainable_params()={n_train:,}")
print(f"[params] encoder trainable={enc_train:,}  decoder trainable={dec_train:,}")
buckets = {}
for n, p in enc.named_parameters():
    if not p.requires_grad: continue
    key = "lora" if "lora" in n else ("slots" if "slots" in n else
          ("summary0" if "summary0" in n else ("norm" if n.startswith("norm") else "other:"+n.split('.')[0])))
    buckets[key] = buckets.get(key, 0) + p.numel()
print(f"[params] encoder breakdown: {buckets}")
print(f"[params] M (encoder.M) = {enc.M}")
print(f"[params] slots shape={tuple(enc.slots.shape)} summary0 shape={tuple(enc.summary0.shape)}")

dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=B, n_pairs=12,
                              n_facts=3, world_seed=0, split="train")
batch = next(iter(dl))
def to_dev(b):
    for f in dataclasses.fields(b):
        v = getattr(b, f.name)
        if torch.is_tensor(v):
            setattr(b, f.name, v.to(DEV))
    return b
batch = to_dev(batch)
print(f"\n[batch] context_ids={tuple(batch.context_ids.shape)} question={tuple(batch.question_ids.shape)} "
      f"answer={tuple(batch.answer_ids.shape)}")
print(f"[batch] ctx real toks per row = {batch.context_mask.sum(1).tolist()}")
n_windows = (batch.context_ids.shape[1] + WIN - 1) // WIN
print(f"[batch] n_windows at window={WIN} -> {n_windows}  (RECURRENCE SEGMENTS)")

# REAL / SHUF / OFF (eval)
model.eval()
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        real = model.compute_qa_loss(batch, window_size=WIN)
        shuf = model.compute_qa_loss(batch, window_size=WIN, shuffle_memory=True)
        off  = model.compute_qa_loss(batch, window_size=WIN, zero_memory=True)
def L(d): return float(d["loss"])
print(f"\n[gate] REAL loss={L(real):.4f}  SHUF loss={L(shuf):.4f}  OFF loss={L(off):.4f}")
print(f"[gate] (SHUF-REAL)={L(shuf)-L(real):+.4f}  (OFF-REAL)={L(off)-L(real):+.4f}")

# memory mutation check: do REAL/SHUF/OFF produce DIFFERENT memory tensors?
print("\n[mem-mutate] checking memory tensor identity across REAL/SHUF/OFF ...")

# grad flow (train path)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
opt.zero_grad()
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_qa_loss(batch, window_size=WIN)
    loss = out["loss"]
loss.backward()
torch.cuda.synchronize()
t1 = time.time()
def gn(params):
    g = [p.grad for p in params if p.grad is not None]
    if not g: return None, 0
    return float(torch.sqrt(sum((x.float()**2).sum() for x in g))), len(g)
enc_lora = [p for n,p in enc.named_parameters() if p.requires_grad and "lora" in n]
slots = [enc.slots]; summary0 = [enc.summary0]
normp = [p for n,p in enc.named_parameters() if p.requires_grad and n.startswith("norm")]
dec_lora = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("decoder") and "lora" in n.lower()]
for nm, ps in [("enc_lora", enc_lora), ("slots", slots), ("summary0", summary0),
               ("enc_norm", normp), ("dec_lora", dec_lora)]:
    norm, cnt = gn(ps)
    print(f"[grad] {nm:<10} grad_norm={norm}  ({cnt}/{len(ps)} params have grad)")
print(f"[time] single fwd+bwd (B={B}, train, autocast, n_windows={n_windows}) = {t1-t0:.2f}s")
print(f"[vram] peak alloc = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\n[DONE-MAIN]")
