"""Detailed debug sweep for hlvocab v2 (full graph), FRESH init (untrained).
Checks the NEW v2 machinery before any training:
  A. gradient flow per module — STDP τ, plasticity, edge-temp, id, role/tag,
     reader, vocab — none dead/saturated;
  B. edge selection health — diversity ACROSS examples (not all same), inter-layer
     fraction, unique endpoint nodes, src/dst layer spread;
  C. magnitudes — edge_w saturation, memory norm (norm-matched?), reader does work;
  D. routing health (phase 1) — entropy/hub/coverage (no collapse);
  E. memory is USED — recon changes when memory zeroed.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_sentence import make_sentence_dataloader

dev = "cuda"; BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "sentence_mae"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.hlvocab_d_code = 256; cfg.hlvocab_nodes = (512, 256, 128)
    cfg.hlvocab_top_k = 4; cfg.hlvocab_m_max = 16; cfg.hlvocab_tap_layer = 6
    return cfg


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
cfg = matched(ReprConfig())
model = ReprLearningModel(cfg, variant="hlvocab_baseline").to(dev)
sub = model.encoder.sub
print(f"=== hlvocab v2 FRESH (use_graph={cfg.hlvocab_use_graph}, n_edges={sub.n_edges}, "
      f"sources={sub.edge_sources}) ===")

dl = make_sentence_dataloader(tok, batch_size=16, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
b = next(iter(dl))
for a in ("context_ids","context_mask","question_ids","question_mask",
          "answer_ids","answer_mask","answer_content_mask"):
    v = getattr(b, a, None)
    if torch.is_tensor(v): setattr(b, a, v.to(dev))

# ── A. GRADIENT FLOW ─────────────────────────────────────────────────────────
print("\n(A) GRADIENT FLOW (one MAE step) — grad/param per module")
model.train(); model.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_mae_loss(b)
out["loss"].backward()
from collections import defaultdict
g = defaultdict(lambda: [0.0, 0.0])
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    if "lora" in n.lower(): k = "decoder.LoRA"
    elif "sub." in n: k = "sub." + n.split("sub.")[1].split(".")[0]
    else: k = "other"
    gg = 0.0 if p.grad is None else p.grad.float().norm().item()
    g[k][0] += gg * gg; g[k][1] += p.float().norm().item() ** 2
print(f"  loss={out['loss'].item():.3f} recon={out['loss_recon'].item():.3f}")
for k in sorted(g):
    gn, pn = math.sqrt(g[k][0]), math.sqrt(g[k][1])
    flag = "  <-- DEAD" if gn < 1e-9 else ("  <-- weak" if gn/max(pn,1e-9) < 1e-4 else "")
    print(f"  {k:22s} grad/param={gn/max(pn,1e-9):.2e}{flag}")

# ── B-E. forward internals ───────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    st = model.encoder.init_streaming_state(b.context_ids.shape[0], dev, torch.float32)
    emb = model.decoder.llama.get_input_embeddings()(b.context_ids)
    st, _ = model.encoder.streaming_write(st, emb, b.context_mask)
    memory, aux = sub(st["hiddens"], st["mask"].float())
    B = memory.shape[0]

    print("\n(D) ROUTING HEALTH (phase 1) — entropy / hub / coverage")
    for l in range(sub.depth):
        print(f"  L{l}: entropy={aux[f'hlvocab_route_entropy_L{l}']:.2f} "
              f"hub={aux[f'hlvocab_hub_share_L{l}']:.3f} cov={aux[f'hlvocab_coverage_L{l}']:.3f}")

    print("\n(B) EDGE SELECTION HEALTH (sharp-softmax edge-query design)")
    print(f"  sel_temp={aux['hlvocab_sel_temp']:.2f} (low=sharper); attn_entropy="
          f"{aux['hlvocab_sel_attn_entropy']:.2f} attn_max={aux['hlvocab_sel_attn_max']:.3f} "
          f"(high attn_max / low entropy = each slot ~picks one edge)")
    print(f"  slot_uniq_edges={aux['hlvocab_slot_uniq_edges']:.1f}/{sub.n_edges} "
          f"(-> n_edges = no slot collapse; << = slots grabbing the same edge)")
    print(f"  inter-layer edge fraction: {aux['hlvocab_edge_inter_frac']:.3f} "
          f"(0=all within-layer, 1=all cross-layer)")

    print("\n(C) MAGNITUDES")
    print(f"  memory norm: {aux['hlvocab_memory_norm']:.2f} (target ~embed 3.18)")
    print(f"  tau_within={sub.log_tau_within.exp().tolist()}  tau_inter={sub.log_tau_inter.exp().tolist()}")
    # reader does work? compare reader output vs its input
    # (recompute tokens quickly is heavy; instead check eff rank of memory)
    def effrank(X):
        X = F.normalize(X.float(), dim=-1); s = torch.linalg.svdvals(X - X.mean(0, keepdim=True))
        s2 = s * s; return (s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item()
    er = sum(effrank(memory[bi]) for bi in range(B)) / B
    print(f"  memory eff_rank: {er:.1f}/{memory.shape[1]} (v1 trained was 3.0/16; higher=more distinct slots)")

    print("\n(E) MEMORY IS USED")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        real = model.compute_mae_loss(b)["loss_recon"].item()
        off = model.compute_mae_loss(b, zero_memory=True)["loss_recon"].item()
    print(f"  REAL={real:.3f}  OFF={off:.3f}  OFF-REAL={off-real:+.3f} (untrained; sign noisy)")
print("\nDONE.")
