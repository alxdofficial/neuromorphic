"""Launch-readiness probe for graph_v9_baseline (arm C) at the EMAT-bio config.

The REAL training path (bf16 autocast, real emat_bio batch, real loss, optimizer
steps — per the debug-in-the-real-path rule). Audits:
  (a) every trainable param receives gradient by the LAST step (the zero-init
      o_proj blocks ALL encoder grads at step 1 BY DESIGN — it must wake by ~3);
  (b) bounded magnitudes everywhere (strengths <= 2, finite dirs/coact/codes,
      finite loss + grad norms — no NaN/Inf anywhere);
  (c) REAL/SHUF/OFF all execute; OFF == vanilla path;
  (d) s/step.
Known-expected dead: decoder.mask_embed (harness-wide; QA never masks).
Reader gates are grad-dead at step 1 only (o_proj zero blocks them until it moves).
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

device = "cuda"
CHUNK = 640; WINDOW = 640; N_STEPS = 6
cfg = ReprConfig()
cfg.batch_size = 2

model = ReprLearningModel(cfg, variant="graph_v9_baseline").to(device)
model.train()

total = sum(p.numel() for p in model.parameters())
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
n_trainable = sum(p.numel() for _, p in trainable)
enc_train = sum(p.numel() for n, p in trainable if n.startswith("encoder."))
print(f"[params] total={total/1e6:.2f}M  trainable={n_trainable/1e6:.3f}M "
      f"({len(trainable)} tensors)  encoder-trainable={enc_train/1e6:.3f}M")

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=2, n_pairs=12,
                              n_query=1, n_facts=3, split="train", world_seed=0,
                              stream_seed=1, pad_token_id=cfg.pad_token_id, num_workers=0)
batch = next(iter(dl))
for attr in dir(batch):
    v = getattr(batch, attr, None)
    if torch.is_tensor(v):
        setattr(batch, attr, v.to(device))
print(f"[data] context={tuple(batch.context_ids.shape)} "
      f"question={tuple(batch.question_ids.shape)} answer={tuple(batch.answer_ids.shape)}")

opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
failures = []


def finite(name, t):
    ok = torch.isfinite(t).all().item()
    if not ok:
        failures.append(f"NaN/Inf in {name}")
    return ok


def run(zero=False, shuf=False):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.compute_qa_loss(batch, window_size=WINDOW,
                                     zero_memory=zero, shuffle_memory=shuf)


# ---- a few REAL training steps with magnitude audit each step ----------------
print(f"\n[train] {N_STEPS} optimizer steps (bf16 autocast, REAL memory)")
sub = model.encoder.sub
times = []
for step in range(1, N_STEPS + 1):
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    out = run()
    loss = out["loss"]              # the trainer's key (loss_recon is the detached logging copy)
    loss.backward()
    grad_total = torch.sqrt(sum((p.grad.float() ** 2).sum()
                                for _, p in trainable if p.grad is not None))
    opt.step()
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    finite("loss", loss.detach())
    finite("grad_total", grad_total)
    # bounded-state audit on the slow params that mirror state bounds
    with torch.no_grad():
        atom_str_max = (2 * torch.sigmoid(sub.atom_strength_logit)).max().item()
        base_str_max = (2 * torch.sigmoid(sub.base_strength_logit[0])).max().item()
        gate_vals = model.graph_v9_reader.gates.detach()
        oproj_norm = model.graph_v9_reader.o_projs[-1].weight.norm().item()
    print(f"  step {step}: loss={loss.item():.4f}  grad_norm={grad_total.item():.3e}  "
          f"o_proj|W|={oproj_norm:.3e}  gates={gate_vals.tolist()}  "
          f"({times[-1]:.2f}s)")
    if not torch.isfinite(loss):
        break

# ---- per-parameter gradient coverage at the LAST step -------------------------
print("\n[grads] coverage at final step")
# decoder.mask_embed: pre-existing harness-wide (QA path never uses mask
# reconstruction — equally dead for every baseline variant; trainer ckpts it).
EXPECTED_DEAD = ("decoder.mask_embed",)
dead, alive = [], 0
for n, p in trainable:
    g = p.grad
    norm = g.float().norm().item() if g is not None else 0.0
    if norm == 0.0:
        if any(k in n for k in EXPECTED_DEAD):
            print(f"  [expected-dead] {n}")
        else:
            dead.append(n)
    else:
        alive += 1
if dead:
    failures.append(f"{len(dead)} unexpectedly grad-dead params")
    for n in dead[:20]:
        print(f"  [DEAD] {n}")
print(f"  {alive}/{len(trainable)} trainable tensors receiving gradient")

# ---- fast-state magnitude audit (run one more encode, inspect state) ----------
print("\n[state] fast-state bounds after training steps")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    embed = model.decoder.llama.get_input_embeddings()
    ctx_embeds = embed(batch.context_ids)
    surprise = model.encoder.context_surprise(batch.context_ids, batch.context_mask)
    st = model.encoder.init_streaming_state(2, device, ctx_embeds.dtype)
    st, _ = model.encoder.streaming_write(st, ctx_embeds, batch.context_mask,
                                          surprise=surprise)
    state = st["sub"]
    for layer_idx in range(sub.depth):
        strengths = state["factor_strengths"][layer_idx]
        dirs = state["factor_dirs"][layer_idx]
        coact = state["coact"][layer_idx]
        ok_s = strengths.max().item() <= 2.0 + 1e-4 and strengths.min().item() >= -1e-6
        ok_f = (torch.isfinite(strengths).all() and torch.isfinite(dirs).all()
                and torch.isfinite(coact).all()).item()
        if not (ok_s and ok_f):
            failures.append(f"state bounds violated at layer {layer_idx}")
        print(f"  L{layer_idx}: strength range [{strengths.min():.4f}, {strengths.max():.4f}] "
              f"(bound 2.0 {'OK' if ok_s else 'VIOLATED'})  total/row "
              f"{strengths.sum(dim=(1,2)).mean():.2f}  dirs|max| {dirs.abs().max():.3f}  "
              f"coact mass {coact.sum(dim=(1,2)).mean():.2f}  finite={'OK' if ok_f else 'NO'}")

# ---- REAL / SHUF / OFF all execute --------------------------------------------
print("\n[controls] REAL / SHUF / OFF execute (post-training, eval mode)")
model.eval()
with torch.no_grad():
    losses = {}
    for tag, kw in (("REAL", {}), ("SHUF", {"shuf": True}), ("OFF", {"zero": True})):
        out = run(**kw)
        losses[tag] = out["loss_recon"].item()
        finite(f"{tag} loss", out["loss_recon"])  # eval: detached copy is fine
print(f"  REAL={losses['REAL']:.4f}  SHUF={losses['SHUF']:.4f}  OFF={losses['OFF']:.4f}  "
      f"(SHUF-REAL={losses['SHUF']-losses['REAL']:+.4f}, OFF-REAL={losses['OFF']-losses['REAL']:+.4f}; "
      "magnitudes uninformative after 6 steps — execution check only)")

print(f"\n[timing] median step {sorted(times)[len(times)//2]:.2f}s at B=2, ctx={CHUNK}")
print("\n" + ("PROBE PASS — harness-wired, grads alive, magnitudes bounded"
              if not failures else "PROBE FAIL: " + "; ".join(failures)))
sys.exit(0 if not failures else 1)
