"""Runtime debug sweep for graph_v9 arm C: dtype leakage, routing calibration on
REAL data, absorption gate magnitudes, state separability, conservation drift.
Encoder-only (one Llama) for speed. Run under the trainer's bf16 autocast."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphV9PyramidEncoder
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

device = "cuda"
cfg = ReprConfig()
enc = GraphV9PyramidEncoder(cfg).to(device)
enc.train()
sub = enc.sub
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
dl = make_emat_bio_dataloader(tok, context_len=640, batch_size=4, n_pairs=12,
                              n_query=1, n_facts=3, split="train", world_seed=0,
                              stream_seed=1, pad_token_id=cfg.pad_token_id, num_workers=0)
batch = next(iter(dl))
ctx_ids = batch.context_ids.to(device); ctx_mask = batch.context_mask.to(device)

# ---- (1) dtype audit inside the substrate under trainer autocast --------------
print("=== (1) dtype inside _chunk_forward under bf16 autocast ===")
dtypes = {}
orig_layer_chunk = sub._layer_chunk
def spy_layer(layer_idx, routing_input, apply_input, *a, **kw):
    out = orig_layer_chunk(layer_idx, routing_input, apply_input, *a, **kw)
    dtypes[f"L{layer_idx}_apply_input"] = str(apply_input.dtype)
    dtypes[f"L{layer_idx}_operated"] = str(out[0].dtype)
    dtypes[f"L{layer_idx}_strengths_out"] = str(out[2].dtype)
    return out
sub._layer_chunk = spy_layer
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    embeds = enc.base.get_input_embeddings()(ctx_ids)
    surprise = enc.context_surprise(ctx_ids, ctx_mask)
    st = enc.init_streaming_state(4, device, embeds.dtype)
    st, _ = enc.streaming_write(st, embeds, ctx_mask, surprise=surprise)
sub._layer_chunk = orig_layer_chunk
for k, v in dtypes.items():
    flag = "" if "float32" in v else "   <-- BF16 LEAK (substrate claims fp32)"
    print(f"  {k}: {v}{flag}")

# ---- (2) routing calibration on REAL data (init was random-query-derived) -----
print("=== (2) routing entropy on real data vs effective_k target ===")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = enc.base.model(inputs_embeds=embeds, attention_mask=ctx_mask.long(),
                         output_hidden_states=True, use_cache=False)
    h = out.hidden_states[cfg.graph_v9_tap_layer + 1].float()
    scores0 = sub.route(0, h)                                   # [B,T,N0]
    real = ctx_mask.bool()
    ent0 = -(scores0.clamp_min(1e-12) * scores0.clamp_min(1e-12).log()).sum(-1)[real]
    # layer-1 routing input: run layer-0 apply on the projected seed
    from src.repr_learning.graph_substrate_v9 import _unit_rms
    codes = _unit_rms(sub.seed_proj(h)) * ctx_mask.unsqueeze(-1).float()
    st0 = sub.init_state(4, device)
    operated = sub.apply_chain(codes, st0["factor_dirs"][0],
                               st0["factor_strengths"][0], scores0 * ctx_mask.unsqueeze(-1).float())
    codes1 = _unit_rms(operated) * ctx_mask.unsqueeze(-1).float()
    scores1 = sub.route(1, codes1)
    ent1 = -(scores1.clamp_min(1e-12) * scores1.clamp_min(1e-12).log()).sum(-1)[real]
import math
print(f"  target effective_k = {cfg.graph_v9_effective_k} (entropy {math.log(cfg.graph_v9_effective_k):.2f})")
print(f"  L0 real-data: entropy {ent0.mean():.2f} -> effective k {ent0.mean().exp():.1f} "
      f"(uniform would be {math.log(sub.config.nodes[0]):.2f} -> {sub.config.nodes[0]})")
print(f"  L1 real-data: entropy {ent1.mean():.2f} -> effective k {ent1.mean().exp():.1f} "
      f"(uniform would be {math.log(sub.config.nodes[1]):.2f} -> {sub.config.nodes[1]})")

# ---- (3) absorption gate magnitudes + relocation per boundary ------------------
print("=== (3) absorption gate + relocation magnitudes (real doc) ===")
stats = []
orig_absorb = sub._absorb
def spy_absorb(layer_idx, dirs, strengths, coact, srp):
    d1, s1, flux = orig_absorb(layer_idx, dirs, strengths, coact, srp)
    with torch.no_grad():
        cn = coact / coact.sum(-1, keepdim=True).clamp_min(1e-6)
        n = coact.shape[1]
        eye = torch.eye(n, device=coact.device, dtype=torch.bool)
        cn_offdiag = cn.masked_fill(eye, 0.0)
        moved = (s1 - strengths).abs().sum(dim=(1, 2)).mean()
        stats.append((layer_idx, cn_offdiag.max().item(), srp.mean().item(),
                      flux.mean().item(), moved.item()))
    return d1, s1, flux
sub._absorb = spy_absorb
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    st = enc.init_streaming_state(4, device, embeds.dtype)
    st, _ = enc.streaming_write(st, embeds, ctx_mask, surprise=surprise)
sub._absorb = orig_absorb
for (l, cmax, srp, flux, moved) in stats:
    print(f"  boundary L{l}: Cn_max={cmax:.3f}  surprise={srp:.3f}  "
          f"flux/row={flux:.3f}  |Δstrength|/row={moved:.3f}")
state = st["sub"]
base_total = (2 * torch.sigmoid(sub.base_strength_logit[0])).sum().item()
s_end = state["factor_strengths"][1]
print(f"  end-of-doc L1: strength range [{s_end.min():.3f}, {s_end.max():.3f}]  "
      f"total/row {s_end.sum(dim=(1,2)).mean():.4f} vs base {base_total:.1f} "
      f"(drift = bf16 leak if nonzero)")

# ---- (4) state separability: does different content -> different state? -------
print("=== (4) per-document state separability (the SHUF question, state-level) ===")
packed = sub.pack_state(state)                                   # [4, L, N, w]
flat = packed[:, 1].reshape(4, -1)                               # writable layer only
flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
cos = (flat @ flat.t())
off = cos[~torch.eye(4, dtype=torch.bool, device=device)]
base_flat = torch.cat([sub.base_dirs[0].reshape(1, -1),
                       ], dim=-1)
print(f"  pairwise cos of L1 packed state across 4 different docs: "
      f"mean {off.mean():.4f}  min {off.min():.4f}  (1.0 = identical = SHUF would equal REAL)")
diffs = (packed[:, 1].unsqueeze(0) - packed[:, 1].unsqueeze(1)).norm(dim=(-2, -1))
print(f"  pairwise L2 of L1 state: mean {diffs[~torch.eye(4, dtype=torch.bool, device=device)].mean():.3f} "
      f"vs state norm {packed[:, 1].norm(dim=(-2, -1)).mean():.3f}")
print("done")
