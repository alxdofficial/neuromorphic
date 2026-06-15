"""Pre-training magnitude + gradient-flow audit for the hlvocab emit modes
(edge_query vs slotattn), FRESH init. Answers, before any training:
  - FORWARD magnitudes: is any signal too loud relative to others? Specifically the
    emitted-token additive components value(=1) : role : tag : ctx, the memory norm,
    and internal module activation norms.
  - GRADIENT magnitudes: does grad flow to EVERY module at a reasonable scale
    (no dead modules, no exploding/vanishing, balanced grad/param)?
  - EMIT health: slot competition (attn entropy/max, slot uniqueness) and the
    emitted-memory effective-rank at init (does competition lift it off rank-1?).

Run: python scripts/diagnostics/hlvocab_emit_diag.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.data_masked_reconstruction import make_sentence_dataloader

dev = "cuda"; BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"


def matched(cfg, emit):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "masked_reconstruction"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.hlvocab_d_code = 256; cfg.hlvocab_nodes = (512, 256, 128)
    cfg.hlvocab_top_k = 4; cfg.hlvocab_m_max = 16; cfg.hlvocab_tap_layer = 6
    cfg.hlvocab_emit = emit; cfg.hlvocab_slot_iters = 3
    return cfg


def eff_rank(X):
    X = F.normalize(X.float(), dim=-1)
    s = torch.linalg.svdvals(X - X.mean(0, keepdim=True)); s2 = s * s
    return (s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item()


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
dl = make_sentence_dataloader(tok, batch_size=16, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
batch = next(iter(dl))
for a in ("context_ids", "context_mask", "question_ids", "question_mask",
          "answer_ids", "answer_mask", "answer_content_mask"):
    v = getattr(batch, a, None)
    if torch.is_tensor(v): setattr(batch, a, v.to(dev))


def run(emit):
    print("\n" + "=" * 78); print(f"EMIT = {emit}"); print("=" * 78)
    model = ReprLearningModel(matched(ReprConfig(), emit), variant="hlvocab_baseline").to(dev)
    model.task_mode = "masked_reconstruction"; model.train()
    sub = model.encoder.sub

    # ── forward activation magnitudes via hooks ──────────────────────────────
    fmag = {}
    def hook(name):
        def f(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t): fmag[name] = t.detach().float().norm(dim=-1).mean().item()
        return f
    hs = []
    targets = {"edge_in": sub.edge_in, "sel_to_tok": sub.sel_to_tok,
               "read_in": sub.read_in, "read_out": sub.read_out}
    if hasattr(sub, "slot_emit"): targets["slot_emit"] = sub.slot_emit
    for nm, mod in targets.items(): hs.append(mod.register_forward_hook(hook(nm)))

    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.compute_masked_reconstruction_loss(batch)
    out["loss"].backward()
    for h in hs: h.remove()
    # substrate telemetry isn't forwarded into the loss output → get it from finalize
    with torch.no_grad():
        st0 = model.encoder.init_streaming_state(batch.context_ids.shape[0], dev, torch.float32)
        emb0 = model.decoder.llama.get_input_embeddings()(batch.context_ids)
        st0, _ = model.encoder.streaming_write(st0, emb0, batch.context_mask)
        _, aux = model.encoder.finalize_memory(st0)

    # ── emitted-token component balance: token = unit(value) + role + tag + ctx ──
    print("\n(A) EMITTED-TOKEN component magnitudes (token = value + role + tag + ctx)")
    print(f"  value(unit)   ‖·‖ = 1.000  (reference)")
    print(f"  role_emb      ‖·‖ = {sub.role_emb.detach().float().norm(dim=-1).mean().item():.3f}")
    print(f"  tag_emb       ‖·‖ = {sub.tag_emb.detach().float().norm(dim=-1).mean().item():.3f}")
    print(f"  ctx (sel_to_tok) ‖·‖ = {fmag.get('sel_to_tok', float('nan')):.3f}  <- if >> 1 it drowns the value")
    print(f"  → memory norm (post norm-match) = {float(aux.get('hlvocab_memory_norm', float('nan'))):.3f} (target ~3.18)")

    print("\n(B) INTERNAL forward activation norms (mean L2 over last dim)")
    for k in ("edge_in", "slot_emit", "read_in", "read_out"):
        if k in fmag: print(f"  {k:12s} {fmag[k]:.3f}")

    # ── gradient flow per module group ───────────────────────────────────────
    print("\n(C) GRADIENT FLOW per sub-module (grad_norm, param_norm, grad/param)")
    groups = {}
    for n, p in sub.named_parameters():
        if not p.requires_grad: continue
        key = n.split(".")[0]
        g = 0.0 if p.grad is None else p.grad.detach().float().norm().item()
        pn = p.detach().float().norm().item()
        d = groups.setdefault(key, [0.0, 0.0]); d[0] += g * g; d[1] += pn * pn
    print(f"  {'module':<18}{'grad_norm':>12}{'param_norm':>12}{'grad/param':>12}")
    for k in sorted(groups):
        gn = math.sqrt(groups[k][0]); pn = math.sqrt(groups[k][1])
        flag = "  <-- DEAD" if gn < 1e-9 else ("  <-- tiny" if gn / max(pn, 1e-12) < 1e-4 else "")
        print(f"  {k:<18}{gn:>12.2e}{pn:>12.2e}{gn/max(pn,1e-12):>12.2e}{flag}")

    # ── emit health + fresh memory eff_rank ──────────────────────────────────
    print("\n(D) EMIT health + memory effective-rank (fresh init)")
    for k in ("hlvocab_sel_attn_entropy", "hlvocab_sel_attn_max", "hlvocab_slot_uniq_edges",
              "hlvocab_edge_inter_frac"):
        if k in aux: print(f"  {k} = {float(aux[k]):.3f}")
    model.eval()
    with torch.no_grad():
        st = model.encoder.init_streaming_state(batch.context_ids.shape[0], dev, torch.float32)
        emb = model.decoder.llama.get_input_embeddings()(batch.context_ids)
        st, _ = model.encoder.streaming_write(st, emb, batch.context_mask)
        mem, _ = model.encoder.finalize_memory(st)
        ranks = [eff_rank(mem[bi]) for bi in range(mem.shape[0])]
    print(f"  memory eff_rank (fresh) = {sum(ranks)/len(ranks):.1f}/{mem.shape[1]}  "
          f"(edge_query fresh was ~5.0; trained collapsed to 1.2)")
    del model; torch.cuda.empty_cache()


run("edge_query")
run("slotattn")
print("\nDONE.")
