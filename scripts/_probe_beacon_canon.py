"""Canonical-config beacon readiness probe (chunk=1024, window=1024, M=144, BS=2)."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

device = "cuda"
CHUNK = 1024; WINDOW = 1024; M = 144
cfg = ReprConfig()
cfg.batch_size = 2
cfg.beacon_ratio = max(1, CHUNK // M)
_ceil = lambda a, b: -(-a // b)
beacon_M = _ceil(CHUNK, WINDOW) * _ceil(WINDOW, cfg.beacon_ratio)
guard_trips = abs(beacon_M - M) > max(1, M // 10)
print(f"[cfg] alpha=beacon_ratio={cfg.beacon_ratio}  beacon_M={beacon_M}  "
      f"prepend_M={M}  ratio={beacon_M/M:.4f}  floats={beacon_M*cfg.d_llama}  "
      f"prepend_floats={M*cfg.d_llama}  guard_trips={guard_trips}  "
      f"thresh_tokens={max(1, M//10)}")

t_build = time.time()
model = ReprLearningModel(cfg, variant="beacon_baseline").to(device)
print(f"[build] ok in {time.time()-t_build:.1f}s")
model.train()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
enc_train = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("encoder."))
named_tr = [(n, p.numel()) for n, p in model.named_parameters()
            if p.requires_grad and p.numel() > 0]
beac = sum(c for n, c in named_tr if ".beacon." in n)
emb = sum(c for n, c in named_tr if "beacon_embed" in n)
norm = sum(c for n, c in named_tr if "encoder.norm" in n)
lora = sum(c for n, c in named_tr if "lora" in n.lower())
print(f"[params] total={total/1e6:.2f}M trainable={trainable/1e6:.3f}M "
      f"enc_trainable={enc_train/1e6:.3f}M n_tensors={len(named_tr)}")
print(f"[params] beacon_proj={beac/1e6:.3f}M beacon_embed={emb} enc.norm={norm} "
      f"dec_lora={lora/1e6:.3f}M")

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=2, n_pairs=12,
                              n_query=1, n_facts=3, split="train", world_seed=0,
                              stream_seed=1, pad_token_id=cfg.pad_token_id, num_workers=0)
batch = next(iter(dl))
for attr in dir(batch):
    v = getattr(batch, attr, None)
    if torch.is_tensor(v):
        setattr(batch, attr, v.to(device))
print(f"[data] ctx={tuple(batch.context_ids.shape)} q={tuple(batch.question_ids.shape)} "
      f"a={tuple(batch.answer_ids.shape)}")

def run(zero=False, shuf=False):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.compute_qa_loss(batch, window_size=WINDOW,
                                     zero_memory=zero, shuffle_memory=shuf)

model.zero_grad(set_to_none=True)
torch.cuda.synchronize(); t0 = time.time()
out = run(); loss = out["loss"]; loss.backward(); torch.cuda.synchronize()
print(f"[fwd+bwd 1st] loss={float(loss):.4f} loss_recon={float(out['loss_recon']):.4f} "
      f"top1={float(out['top1_acc']):.4f} "
      f"finite={torch.isfinite(loss).item()} grad_fn={loss.grad_fn is not None} s={time.time()-t0:.2f}")

def gnorm(pred):
    ns = [p.grad.detach().norm() for n, p in model.named_parameters()
          if p.grad is not None and pred(n)]
    return float(torch.norm(torch.stack(ns))) if ns else 0.0
print(f"[grad] beacon_proj={gnorm(lambda n: '.beacon.' in n):.4e} "
      f"beacon_embed={gnorm(lambda n: 'beacon_embed' in n):.4e} "
      f"enc.norm={gnorm(lambda n: 'encoder.norm' in n):.4e} "
      f"dec_lora={gnorm(lambda n: 'lora' in n.lower()):.4e}")
n_grad = sum(1 for _, p in model.named_parameters() if p.requires_grad and p.grad is not None)
n_req = sum(1 for _, p in model.named_parameters() if p.requires_grad)
print(f"[grad] {n_grad}/{n_req} trainable tensors have grad")

# finalized memory shape (confirm beacon_M tokens)
model.train(True)
B = batch.context_ids.shape[0]
emb_layer = model.decoder.llama.get_input_embeddings()
state = model.encoder.init_streaming_state(B, device, torch.bfloat16)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ce = emb_layer(batch.context_ids[:, :WINDOW])
    cm = (batch.context_ids[:, :WINDOW] != cfg.pad_token_id)
    state, _ = model.encoder.streaming_write(state, ce, cm)
    mem, _ = model.encoder.finalize_memory(state)
print(f"[mem] finalized shape={tuple(mem.shape)} (expect M={beacon_M})  "
      f"floats/example={mem.shape[1]*mem.shape[2]}")

# gate
model.train(False)
with torch.no_grad():
    rl = float(run()["loss_recon"]); ol = float(run(zero=True)["loss_recon"])
    sl = float(run(shuf=True)["loss_recon"])
print(f"[gate] REAL={rl:.4f} SHUF={sl:.4f} OFF={ol:.4f} "
      f"OFF-REAL={ol-rl:+.4f} SHUF-REAL={sl-rl:+.4f}")

# clean 2nd step timing
model.train(True)
model.zero_grad(set_to_none=True)
torch.cuda.synchronize(); t0 = time.time()
out = run(); out["loss"].backward(); torch.cuda.synchronize()
print(f"[timing] 2nd fwd+bwd s/step={time.time()-t0:.3f} (BS=2; BS=8 ~3-4x)")
print(f"[vram] peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
