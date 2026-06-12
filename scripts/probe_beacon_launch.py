"""Launch-readiness probe for beacon_baseline at the EMAT-bio launch config.
Constructs ONLY beacon. B=2 fwd+bwd, grad flow, REAL/SHUF/OFF, effective M, params, s/step.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

device = "cuda"
# Mirror the trainer's cfg construction for the launch config.
CHUNK = 640; WINDOW = 640; M = 144
cfg = ReprConfig()
cfg.batch_size = 2
# Trainer sets these from --mem-tokens (M) and chunk/window:
cfg.beacon_ratio = max(1, CHUNK // M)        # exactly the trainer's derivation
print(f"[cfg] beacon_ratio(alpha) = {cfg.beacon_ratio}  (trainer: max(1, {CHUNK}//{M}))")
_ceil = lambda a, b: -(-a // b)
beacon_M = _ceil(CHUNK, WINDOW) * _ceil(WINDOW, cfg.beacon_ratio)
print(f"[cfg] effective beacon M = {beacon_M}  vs prepend M = {M}  "
      f"({beacon_M/M:.3f}x; budget guard trips? {abs(beacon_M-M) > max(1, M//10)})")

model = ReprLearningModel(cfg, variant="beacon_baseline").to(device)
model.train()

# Param accounting
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
enc_train = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("encoder."))
print(f"[params] total={total/1e6:.2f}M  trainable={trainable/1e6:.3f}M  "
      f"encoder-trainable={enc_train/1e6:.3f}M")
# Break down trainable encoder params
named_tr = [(n, p.numel()) for n, p in model.named_parameters()
            if p.requires_grad and p.numel() > 0]
print(f"[params] #trainable tensors = {len(named_tr)}")
# beacon proj + embed + norm breakdown
beac = sum(c for n, c in named_tr if ".beacon." in n)
emb = sum(c for n, c in named_tr if "beacon_embed" in n)
norm = sum(c for n, c in named_tr if "encoder.norm" in n)
lora = sum(c for n, c in named_tr if "lora" in n.lower())
print(f"[params]   beacon projections = {beac/1e6:.3f}M  beacon_embed={emb}  "
      f"enc.norm={norm}  decoder-lora={lora/1e6:.3f}M")

# Real emat_bio batch
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=2, n_pairs=12,
                              n_query=1, n_facts=3, split="train", world_seed=0,
                              stream_seed=1, pad_token_id=cfg.pad_token_id, num_workers=0)
batch = next(iter(dl))
batch = batch.to(device) if hasattr(batch, "to") else batch
# Move batch tensors
for attr in dir(batch):
    v = getattr(batch, attr, None)
    if torch.is_tensor(v):
        setattr(batch, attr, v.to(device))
print(f"[data] context_ids shape = {batch.context_ids.shape}  "
      f"question_ids = {batch.question_ids.shape}  answer_ids = {batch.answer_ids.shape}")

def run(zero=False, shuf=False):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.compute_qa_loss(batch, window_size=WINDOW,
                                    zero_memory=zero, shuffle_memory=shuf)
    return out

# REAL forward+backward + grad flow
model.zero_grad(set_to_none=True)
t0 = time.time()
out = run()
loss = out["loss_recon"]
loss.backward()
torch.cuda.synchronize()
t_step = time.time() - t0
print(f"[fwd+bwd] REAL loss_recon = {float(loss):.4f}  top1={float(out['top1_acc']):.4f}  "
      f"s/step(1st, incl compile/cache) = {t_step:.2f}")

# Grad norms by group
def gnorm(pred):
    ns = [p.grad.detach().norm() for n, p in model.named_parameters()
          if p.grad is not None and pred(n)]
    return float(torch.norm(torch.stack(ns))) if ns else 0.0
gn_beacon = gnorm(lambda n: ".beacon." in n)
gn_embed = gnorm(lambda n: "beacon_embed" in n)
gn_norm = gnorm(lambda n: "encoder.norm" in n)
gn_lora = gnorm(lambda n: "lora" in n.lower())
n_with_grad = sum(1 for _, p in model.named_parameters() if p.requires_grad and p.grad is not None)
n_req = sum(1 for _, p in model.named_parameters() if p.requires_grad)
print(f"[grad] beacon_proj={gn_beacon:.4e}  beacon_embed={gn_embed:.4e}  "
      f"enc.norm={gn_norm:.4e}  decoder_lora={gn_lora:.4e}")
print(f"[grad] {n_with_grad}/{n_req} trainable tensors have grad")

# REAL/SHUF/OFF gate
model.train(False)
with torch.no_grad():
    rl = float(run()["loss_recon"])
    ol = float(run(zero=True)["loss_recon"])
    sl = float(run(shuf=True)["loss_recon"])
print(f"[gate] REAL={rl:.4f}  SHUF={sl:.4f}  OFF={ol:.4f}  "
      f"OFF-REAL={ol-rl:+.4f}  SHUF-REAL={sl-rl:+.4f}")

# Confirm the memory actually has beacon_M tokens at the prepend
model.train(True)
B = batch.context_ids.shape[0]
emb_layer = model.decoder.llama.get_input_embeddings()
state = model.encoder.init_streaming_state(B, device, torch.bfloat16)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ce = emb_layer(batch.context_ids[:, :WINDOW])
    cm = (batch.context_ids[:, :WINDOW] != cfg.pad_token_id)
    state, _ = model.encoder.streaming_write(state, ce, cm)
    mem, _ = model.encoder.finalize_memory(state)
print(f"[mem] finalized memory shape = {tuple(mem.shape)}  (expect M≈{beacon_M})")

# Timing: a clean 2nd step (no first-call overhead)
model.zero_grad(set_to_none=True)
torch.cuda.synchronize(); t0 = time.time()
out = run(); out["loss_recon"].backward(); torch.cuda.synchronize()
print(f"[timing] 2nd fwd+bwd s/step = {time.time()-t0:.3f}")
print(f"[vram] peak = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
