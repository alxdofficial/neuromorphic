"""DEEP independent readiness probe for ccm_baseline ONLY (EMAT-bio gate).
Constructs ONLY CCM. Verifies at the CANONICAL config (chunk=window=1024,
n_facts=3, BS=2): builds; B=2 fwd+bwd grads flow into CCM trainables;
emits exactly M=144 (ccm_n_comp); REAL/SHUF/OFF mutate memory; CCM
faithfulness (recurrence conditions on prior memory; merge=1/t mean fixed M;
COMP-LoRA gates to <COMP> positions only so text tokens pass through the
frozen base unchanged); budget floats = 144*2048 = 294,912; param count; s/step.
"""
import sys, time, dataclasses, torch
sys.path.insert(0, "/home/alex/code/neuromorphic")
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

torch.manual_seed(0)
dev = "cuda"
M = 144
CHUNK = 1024; WINDOW = 1024  # canonical emat_bio
cfg = ReprConfig(batch_size=2, n_flat_codes=M)
cfg.ccm_n_comp = M           # trainer sets this from --mem-tokens
cfg.use_llama_lora = True

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
print("loading shared frozen llama (decoder side)...", flush=True)
llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)
print("building ccm_baseline (ONLY ccm)...", flush=True)
model = ReprLearningModel(cfg, variant="ccm_baseline", llama_model=llama).to(dev)

# ---- param accounting ----
enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
dec_train = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
tot_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[PARAMS] encoder_trainable={enc_train:,} decoder_trainable={dec_train:,} total_trainable={tot_train:,}")
grp = {}
for n, p in model.encoder.named_parameters():
    if p.requires_grad:
        key = "lora" if "lora" in n else ("comp_embeds" if "comp_embed" in n else ("norm" if n.startswith("norm") else "other"))
        grp[key] = grp.get(key, 0) + p.numel()
print(f"[PARAMS enc breakdown] {grp}")
print(f"[CCM] n_comp={model.encoder.n_comp} fold={model.encoder.fold} "
      f"lora_rank={cfg.ccm_lora_rank} targets={cfg.ccm_lora_targets} scale={cfg.ccm_lora_alpha/cfg.ccm_lora_rank}")
n_lora_layers = len(model.encoder._lora_layers)
print(f"[CCM] wrapped {n_lora_layers} linears as COMP-gated LoRA")

# ---- budget ----
floats = M * cfg.d_llama
print(f"[BUDGET] M*d_llama = {M}*{cfg.d_llama} = {floats:,} floats (claim 294,912 => {'MATCH' if floats==294912 else 'MISMATCH'})")

# ---- data at canonical config ----
dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=2, n_pairs=16,
                              n_query=1, n_facts=3, world_seed=0, stream_seed=42,
                              split="train", num_workers=0)
batch = next(iter(dl))
if dataclasses.is_dataclass(batch):
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            setattr(batch, f.name, v.to(dev))
print(f"[BATCH] context={tuple(batch.context_ids.shape)} q={tuple(batch.question_ids.shape)} "
      f"a={tuple(batch.answer_ids.shape)} ctx_tok(row0)={int(batch.context_mask[0].sum())} "
      f"ctx_tok(row1)={int(batch.context_mask[1].sum())}", flush=True)

# ============ CCM FAITHFULNESS PROBES (encoder-direct) ============
enc = model.encoder
embed = model.decoder.llama.get_input_embeddings()
with torch.no_grad():
    ctx_emb = embed(batch.context_ids)

# (1) emitted M check via finalize
with torch.no_grad():
    st = enc.init_streaming_state(2, dev, ctx_emb.dtype)
    st, _ = enc.streaming_write(st, ctx_emb, batch.context_mask)
    mem, _ = enc.finalize_memory(st)
print(f"[EMIT] finalize_memory shape={tuple(mem.shape)} (expect [2,{M},{cfg.d_llama}]) "
      f"M_emitted={mem.shape[1]} norm_mean={mem.norm(dim=-1).mean().item():.3f} "
      f"(NormMatch target ~0.9)")

# (2) RECURRENCE: window2 must condition on window1's memory (prefix passed in).
#     Run two windows; check mem after t=2 differs from a fresh single-window t=1
#     on the SAME second window (proving prefix conditioning is live).
with torch.no_grad():
    half = ctx_emb.shape[1] // 2
    w1, m1 = ctx_emb[:, :half], batch.context_mask[:, :half]
    w2, m2 = ctx_emb[:, half:], batch.context_mask[:, half:]
    # recurrent: t1 then t2 (t2 sees prefix=mem_t1)
    s = enc.init_streaming_state(2, dev, ctx_emb.dtype)
    s, _ = enc.streaming_write(s, w1, m1)
    mem_t1 = s["mem"].clone()
    s, _ = enc.streaming_write(s, w2, m2)
    mem_t2_recurrent = s["mem"].clone()
    # control: t2 alone (prefix=None) — what merge would give if recurrence were dead
    s0 = enc.init_streaming_state(2, dev, ctx_emb.dtype)
    s0, _ = enc.streaming_write(s0, w2, m2)
    mem_t2_alone = s0["mem"].clone()
    # merge fold: mem_t2 = 0.5*mem_t1 + 0.5*h_comp(w2|prefix=mem_t1)
    delta_recur_vs_alone = (mem_t2_recurrent - mem_t2_alone).norm().item()
    # also verify the merge arithmetic: shape stays M (fixed), not growing
print(f"[RECUR] mem_t1 shape={tuple(mem_t1.shape)} mem_t2 shape={tuple(mem_t2_recurrent.shape)} "
      f"(merge=> fixed M, NOT growing). ||mem_t2_recurrent - mem_t2_alone||={delta_recur_vs_alone:.4f} "
      f"(>0 => window2 conditioned on prior memory => recurrence LIVE)")

# (3) COMP-LoRA GATING: text-token hidden states must be IDENTICAL whether the
#     COMP-LoRA is engaged or not (the gate zeros LoRA at non-COMP positions).
#     Probe _comp_forward: run base over [window ++ COMP], capture full hiddens,
#     compare text positions with LoRA's A/B temporarily zeroed.
with torch.no_grad():
    # capture text-region hidden by monkey-reading the base over the same input
    # with mask engaged vs lora-zeroed.
    B_, _, d = w1.shape
    comp = enc.comp_embeds.to(w1.dtype).unsqueeze(0).expand(B_, enc.n_comp, d)
    inp = torch.cat([w1, comp], dim=1)
    attn = torch.ones(B_, inp.shape[1], device=dev, dtype=torch.long)
    cm = torch.zeros(B_, inp.shape[1], 1, device=dev, dtype=inp.dtype)
    cm[:, -enc.n_comp:, :] = 1.0
    enc._mask[0] = cm
    h_gated = enc.base.model(inputs_embeds=inp, attention_mask=attn).last_hidden_state
    enc._mask[0] = None  # mask None => LoRA no-op everywhere
    h_nolora = enc.base.model(inputs_embeds=inp, attention_mask=attn).last_hidden_state
    # text positions = all but last n_comp
    txt = slice(0, inp.shape[1] - enc.n_comp)
    text_diff = (h_gated[:, txt] - h_nolora[:, txt]).abs().max().item()
    comp_diff = (h_gated[:, -enc.n_comp:] - h_nolora[:, -enc.n_comp:]).abs().max().item()
print(f"[GATE-LoRA] text-position max|Δ(gated vs lora-off)|={text_diff:.3e} "
      f"(expect ~0 => text passes through frozen base; CCM signature). "
      f"COMP-position max|Δ|={comp_diff:.3e} (LoRA B init 0 => may be ~0 at init)")

# ============ GATE eval: REAL/SHUF/OFF distinctness ============
model.eval()
def run(**kw):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.compute_qa_loss(batch, window_size=WINDOW, **kw)["loss"].item()
real = run(); shuf = run(shuffle_memory=True); off = run(zero_memory=True)
print(f"[GATE eval] REAL={real:.4f} SHUF={shuf:.4f} OFF={off:.4f} "
      f"SHUF-REAL={shuf-real:+.4f} OFF-REAL={off-real:+.4f} "
      f"(distinct => mutation works; init gate ~0 expected pre-train)")

# ============ BWD grad flow (REAL training path) ============
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
opt.zero_grad()
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model.compute_qa_loss(batch, window_size=WINDOW)["loss"]
loss.backward()
torch.cuda.synchronize()
t_fb = time.time() - t0
g_enc = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
n_live = sum(1 for _, p in g_enc if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
gn_lora = sum(p.grad.norm().item()**2 for n, p in g_enc if p.grad is not None and "lora" in n)**0.5
gn_comp = sum(p.grad.norm().item()**2 for n, p in g_enc if p.grad is not None and "comp_embed" in n)**0.5
gn_norm = sum(p.grad.norm().item()**2 for n, p in g_enc if p.grad is not None and n.startswith("norm"))**0.5
any_nan = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
print(f"[BWD] loss={loss.item():.4f} fwd+bwd={t_fb:.2f}s enc_live_grad={n_live}/{len(g_enc)} any_nan={any_nan}")
print(f"[GRAD] gnorm_lora={gn_lora:.4e} gnorm_comp_embed={gn_comp:.4e} gnorm_norm={gn_norm:.4e}")
opt.step()

# ---- s/step steady state at B=2 (canonical window=1024) ----
torch.cuda.synchronize()
times = []
for i in range(3):
    opt.zero_grad()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = model.compute_qa_loss(batch, window_size=WINDOW)["loss"]
    l.backward(); opt.step(); torch.cuda.synchronize()
    times.append(time.time() - t0)
print(f"[STEP] B=2 s/step: {[f'{x:.2f}' for x in times]} mean={sum(times)/len(times):.2f}s")
print(f"[MEM] peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB")
print("DONE")
