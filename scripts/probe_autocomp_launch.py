"""Launch-readiness probe for autocompressor_baseline (recurrent summary).
Mirrors the launch cfg (chunk=640, window=640, mem-tokens=144, emat_bio)."""
import os, time, torch, torch.nn as nn
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = "cuda"
CHUNK, WIN, M, B = 640, 640, 144, 2

# ---- cfg mirroring train_repr_qa.py main() for the launch flags ----
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

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
model = ReprLearningModel(cfg, variant="autocompressor_baseline").to(DEV)
model.train()

# ---- param accounting ----
n_train = model.n_trainable_params()
enc = model.encoder
enc_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
dec_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("decoder"))
print(f"\n[params] model.n_trainable_params()={n_train:,}")
print(f"[params] encoder trainable={enc_train:,}  decoder trainable={dec_train:,}")
# break down encoder trainable
buckets = {}
for n, p in enc.named_parameters():
    if not p.requires_grad: continue
    key = "lora" if "lora" in n else ("slots" if "slots" in n else
          ("summary0" if "summary0" in n else ("norm" if n.startswith("norm") else "other:"+n.split('.')[0])))
    buckets[key] = buckets.get(key, 0) + p.numel()
print(f"[params] encoder breakdown: {buckets}")
print(f"[params] M (encoder.M) = {enc.M}")

# ---- build a real emat_bio batch ----
dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=B, n_pairs=12,
                              n_facts=3, world_seed=0, split="train")
batch = next(iter(dl))
import dataclasses
flds = [f.name for f in dataclasses.fields(batch)] if dataclasses.is_dataclass(batch) else None
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
print(f"[batch] n_windows at window={WIN} -> {n_windows}")

# ---- REAL / SHUF / OFF forward (eval, no_grad) ----
model.eval()
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        real = model.compute_qa_loss(batch, window_size=WIN)
        shuf = model.compute_qa_loss(batch, window_size=WIN, shuffle_memory=True)
        off  = model.compute_qa_loss(batch, window_size=WIN, zero_memory=True)
def L(d): return float(d["loss"])
print(f"\n[gate] REAL loss={L(real):.4f}  SHUF loss={L(shuf):.4f}  OFF loss={L(off):.4f}")
print(f"[gate] (SHUF-REAL)={L(shuf)-L(real):+.4f}  (OFF-REAL)={L(off)-L(real):+.4f}")

# ---- B=2 fwd+bwd grad flow (training path, autocast bf16) ----
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
opt.zero_grad()
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_qa_loss(batch, window_size=WIN)
    loss = out["loss"]
loss.backward()
t1 = time.time()
# grad norms by group
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
print(f"[time] single fwd+bwd (B={B}, train, autocast) = {t1-t0:.2f}s")

print("\n[DONE-MAIN]")
