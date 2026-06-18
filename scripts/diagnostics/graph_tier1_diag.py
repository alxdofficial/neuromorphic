"""Tier-1 graph diagnostics on the trained PREPEND ckpt (no re-train). Answers:
  1. Pointer sharpness  — is the discrete "snap" real, or a diffuse blend of many nodes?
  2. Binding gate       — REAL vs OFF vs SHUF: is the memory's CONTENT used (vs membership)?
  3. Channel ablations  — zero edge_state / zero endpoints: which channel carries signal?
  4. Bank coverage      — over the full val set, how many of N nodes are ever selected?

Run:  python scripts/diagnostics/graph_tier1_diag.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoConfig

from src.memory.config import ReprConfig
from src.memory.common import resolve_special_ids
from src.memory.model import ReprLearningModel

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
SRC_TOK = "meta-llama/Llama-3.2-1B"
DEV = "cuda"
REPO = Path(__file__).resolve().parents[2]
TOK = AutoTokenizer.from_pretrained(BACKBONE)
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token


def mae_cfg():
    c = ReprConfig()
    c.llama_model = BACKBONE
    c.d_llama = AutoConfig.from_pretrained(BACKBONE).hidden_size
    c.pad_token_id, c.sep_token_id = resolve_special_ids(TOK)
    c.task_mode = "masked_reconstruction"
    c.use_llama_lora = True; c.llama_lora_rank = 16; c.llama_lora_alpha = 32
    c.n_flat_codes = 16
    c.icae_n_slots = 16; c.icae_lora_rank = 104; c.icae_lora_alpha = 208
    c.ccm_n_comp = 16; c.ccm_lora_rank = 52; c.ccm_lora_alpha = 104
    c.autocompressor_n_slots = 16; c.autocompressor_lora_rank = 52; c.autocompressor_lora_alpha = 104
    c.beacon_ratio = 8
    from src.memory.common import beacon_wrap_layers
    c.beacon_wrap_layers = beacon_wrap_layers(AutoConfig.from_pretrained(BACKBONE).num_hidden_layers, 11)
    return c


def load_variant(variant, tag):
    m = ReprLearningModel(mae_cfg(), variant=variant).to(DEV)
    m.task_mode = "masked_reconstruction"
    ckpt = REPO / f"outputs/memory/{tag}_{variant}/ckpts/{variant}.best.pt"
    sd = torch.load(ckpt, map_location=DEV, weights_only=False)["model_state_dict"]
    m.load_state_dict(sd, strict=False)
    return m.eval()


def val_batches(n, bs=16):
    from src.memory.data_masked_reconstruction import make_sentence_dataloader
    from scripts.train.train import to_device
    dl = make_sentence_dataloader(TOK, batch_size=bs, src_tokenizer_name=SRC_TOK, split="val",
                                  seed=7, pad_token_id=resolve_special_ids(TOK)[0] or 0, num_workers=0)
    it = iter(dl)
    return [to_device(next(it), DEV) for _ in range(n)]


@torch.no_grad()
def graph_dict(model, batch):
    enc = model.encoder
    ctx = model.decoder.llama.get_input_embeddings()(batch.context_ids)
    st = enc.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
    st, _ = enc.streaming_write(st, ctx, batch.context_mask)
    _, aux = enc.finalize_memory(st)
    return aux["graph"]


@torch.no_grad()
def seeded_loss(model, batch, *, zero_memory=False, shuffle_memory=False, seed=0):
    torch.manual_seed(seed)                          # identical MAE mask across conditions
    out = model.compute_masked_reconstruction_loss(
        batch, zero_memory=zero_memory, shuffle_memory=shuffle_memory, mask_ratio=0.85)
    return float(out["loss_recon"]), float(out["top1_acc"])


class ablate_parser:
    """Context manager: zero chosen keys in the parser's output graph dict (instance patch)."""
    def __init__(self, model, keys):
        self.parser = model.encoder.parser; self.keys = keys
    def __enter__(self):
        self._orig = self.parser.forward
        keys, orig = self.keys, self._orig
        def patched(obs, obs_mask, state=None):
            g = orig(obs, obs_mask, state=state)
            return {k: (torch.zeros_like(v) if k in keys else v) for k, v in g.items()}
        self.parser.forward = patched
        return self
    def __exit__(self, *a):
        self.parser.forward = self._orig


print("=" * 74)
print("GRAPH TIER-1 DIAGNOSTICS — trained prepend ckpt (mae4k_prepend)")
print("=" * 74)
graph = load_variant("graph_baseline", "mae4k_prepend")
batches = val_batches(20, bs=16)
N = graph.encoder.gcfg.n_nodes
E = graph.encoder.gcfg.n_edges
print(f"N(bank)={N}  E(edges)={E}  val batches={len(batches)}×{batches[0].context_ids.shape[0]}\n")

# ── 1. Pointer sharpness ──────────────────────────────────────────────────────
print("1. POINTER SHARPNESS  (is the discrete snap real?)")
top1, top5, ppl, committed, n = 0., 0., 0., 0., 0
with torch.no_grad():
    for b in batches[:6]:
        g = graph_dict(graph, b)
        ptr = torch.cat([g["src_ptr"], g["dst_ptr"]], dim=1).reshape(-1, N).float()  # [.,N]
        top1 += ptr.max(-1).values.sum().item()
        top5 += ptr.topk(5, -1).values.sum(-1).sum().item()
        ent = -(ptr * ptr.clamp_min(1e-12).log()).sum(-1)
        ppl += ent.exp().sum().item()
        committed += (ptr.max(-1).values > 0.5).sum().item()
        n += ptr.shape[0]
print(f"   mean top-1 mass     = {top1/n:.3f}   (1.0 = perfect one-hot snap)")
print(f"   mean top-5 mass     = {top5/n:.3f}")
print(f"   mean perplexity     = {ppl/n:.1f} nodes  (1 = one-hot; {N} = uniform)")
print(f"   frac one-hot (>0.5) = {committed/n:.3f}")
print(f"   >> the gather averages ~{ppl/n:.0f} bank vectors per pointer → blend, not a snap\n"
      if ppl/n > 5 else "   >> sharp snap\n")

# ── 2. Binding gate REAL / OFF / SHUF ─────────────────────────────────────────
print("2. BINDING GATE  (REAL << SHUF ≲ OFF ⇒ memory CONTENT is used)")
for name, v, tag in [("graph", "graph_baseline", "mae4k_prepend"),
                     ("autocomp(ref)", "autocompressor_baseline", "mae4k")]:
    mdl = graph if v == "graph_baseline" else load_variant(v, tag)
    real = sum(seeded_loss(mdl, b, seed=i)[0] for i, b in enumerate(batches[:6])) / 6
    off = sum(seeded_loss(mdl, b, zero_memory=True, seed=i)[0] for i, b in enumerate(batches[:6])) / 6
    shuf = sum(seeded_loss(mdl, b, shuffle_memory=True, seed=i)[0] for i, b in enumerate(batches[:6])) / 6
    gap = off - real
    print(f"   {name:14} REAL={real:.3f}  SHUF={shuf:.3f}  OFF={off:.3f}   (OFF-REAL={gap:+.3f})")
    if v != "graph_baseline":
        del mdl; import gc; gc.collect(); torch.cuda.empty_cache()
print()

# ── 3. Channel ablations ──────────────────────────────────────────────────────
print("3. CHANNEL ABLATIONS  (Δ recon loss when a channel is zeroed)")
real = sum(seeded_loss(graph, b, seed=i)[0] for i, b in enumerate(batches[:6])) / 6
for label, keys in [("zero edge_state (relation)", ["edge_state"]),
                    ("zero endpoints (identity)", ["src_value", "dst_value"])]:
    with ablate_parser(graph, keys):
        abl = sum(seeded_loss(graph, b, seed=i)[0] for i, b in enumerate(batches[:6])) / 6
    print(f"   {label:30} loss {real:.3f} → {abl:.3f}   (Δ={abl-real:+.3f})")
print("   (Δ≈0 ⇒ that channel carries ~no signal)\n")

# ── 4. Bank coverage over the full val set ────────────────────────────────────
print("4. BANK COVERAGE  (how many of N nodes are EVER selected)")
used = torch.zeros(N, dtype=torch.long)
with torch.no_grad():
    for b in batches:
        g = graph_dict(graph, b)
        idx = torch.cat([g["src_ptr"].argmax(-1).reshape(-1),
                         g["dst_ptr"].argmax(-1).reshape(-1)]).cpu()
        used.scatter_add_(0, idx, torch.ones_like(idx))
n_used = int((used > 0).sum())
picks = int(used.sum())
top10 = used.sort(descending=True).values[:10].sum().item()
print(f"   distinct nodes used = {n_used}/{N}  ({100*n_used/N:.1f}%)   dead = {N-n_used}")
print(f"   top-10 nodes take {100*top10/picks:.1f}% of all {picks} picks  (hub concentration)")
print("\n" + "=" * 74)
