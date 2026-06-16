"""Why does the graph collapse on MAE but bind on conditioned_reconstruction?

Hypothesis: baselines PREPEND M memory vectors → decoder reads them per-position via
native attention (read rank up to M). The graph INJECTS one additive gated vector per
position at a single mid-late layer; if that signal is low-rank across positions, every
masked position gets ≈the same nudge → can't reconstruct distinct tokens (MAE/continuation
need ~85% of positions, each distinct) but is fine for ONE addressed value (cond_recon).

Measures, on a real val MAE batch, with the TRAINED checkpoints:
  graph : read_effrank (PR of the injected vec across masked positions) + ptr/bank/edge canaries
  baseline(s): participation-ratio of the M prepended memory slots (the content rank the
               decoder can read). Compared against M (the ceiling) and the graph's read rank.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoConfig

from src.memory.config import ReprConfig
from src.memory.common import resolve_special_ids
from src.memory.model import ReprLearningModel, _participation_ratio

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
SRC_TOK = "meta-llama/Llama-3.2-1B"
DEV = "cuda"
REPO = Path(__file__).resolve().parents[2]
TOK = AutoTokenizer.from_pretrained(BACKBONE)
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token


def mae_cfg():
    """Replicate train.py's masked_reconstruction override (the bits that matter)."""
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
    cfg = mae_cfg()
    m = ReprLearningModel(cfg, variant=variant).to(DEV)
    m.task_mode = "masked_reconstruction"
    ckpt = REPO / f"outputs/memory/{tag}_{variant}/ckpts/{variant}.best.pt"
    sd = torch.load(ckpt, map_location=DEV, weights_only=False)["model_state_dict"]
    res = m.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if "base" not in k and "llama" not in k or "lora" in k.lower()]
    print(f"  loaded {variant}: {len(sd)} tensors; unexpected={len(res.unexpected_keys)} "
          f"(non-base missing≈{len(miss)})")
    return m.eval()


def val_batch(bs=16):
    from src.memory.data_masked_reconstruction import make_sentence_dataloader
    from scripts.train.train import to_device
    dl = make_sentence_dataloader(TOK, batch_size=bs, src_tokenizer_name=SRC_TOK,
                                  split="val", seed=7, pad_token_id=resolve_special_ids(TOK)[0] or 0,
                                  num_workers=0)
    return to_device(next(iter(dl)), DEV)


@torch.no_grad()
def baseline_memory_rank(model, batch):
    """PR of the M prepended memory vectors (content rank the decoder can read)."""
    embed = model.decoder.llama.get_input_embeddings()
    ctx = embed(batch.context_ids)
    st = model.encoder.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
    st, _ = model.encoder.streaming_write(st, ctx, batch.context_mask)
    mem, _ = model.encoder.finalize_memory(st)              # [B, M, d]
    B, M, d = mem.shape
    pr_all = _participation_ratio(mem.reshape(B * M, d))    # rank across ALL B*M vectors
    pr_perex = sum(_participation_ratio(mem[i]) for i in range(B)) / B  # avg per-example (≤M)
    return M, pr_all, pr_perex


print("=" * 74)
print("WHY THE GRAPH COLLAPSES ON MAE — read-rank diagnostic (trained ckpts, val batch)")
print("=" * 74)
batch = val_batch(bs=16)
print(f"val batch: B={batch.context_ids.shape[0]}, T={batch.context_ids.shape[1]}, "
      f"valid_tok≈{int(batch.context_mask.float().mean()*batch.context_ids.shape[1])}/ex\n")

# ---- graph ----
print("GRAPH (inject read):")
gm = load_variant("graph_baseline", "mae4k")
with torch.no_grad():
    out = gm.compute_masked_reconstruction_loss(batch, mask_ratio=0.85)
gk = ["graph_read_effrank", "graph_read_gate", "graph_ptr_entropy", "graph_nodes_used",
      "graph_bank_effrank", "graph_edge_effrank"]
print(f"  val_loss_recon = {float(out['loss_recon']):.3f}   top1 = {float(out['top1_acc']):.3f}")
for k in gk:
    if k in out:
        print(f"  {k:<22} = {out[k]:.3f}" if isinstance(out[k], float) else f"  {k:<22} = {out[k]}")
print(f"  >> the injected signal carries ~{out.get('graph_read_effrank', float('nan')):.1f} "
      f"effective dims across masked positions (E={gm.encoder.gcfg.n_edges} edges available)\n")
del gm; import gc; gc.collect(); torch.cuda.empty_cache()

# ---- baselines (prepend read) ----
for v, tag in [("autocompressor_baseline", "mae4k"), ("icae_baseline", "mae4k")]:
    print(f"{v.upper().replace('_BASELINE','')} (prepend read):")
    bm = load_variant(v, tag)
    M, pr_all, pr_perex = baseline_memory_rank(bm, batch)
    with torch.no_grad():
        out = bm.compute_masked_reconstruction_loss(batch, mask_ratio=0.85)
    print(f"  val_loss_recon = {float(out['loss_recon']):.3f}   top1 = {float(out['top1_acc']):.3f}")
    print(f"  M = {M} prepended slots; memory effrank (all B*M) = {pr_all:.1f}; "
          f"per-example (≤M) = {pr_perex:.1f}")
    print(f"  >> decoder can read ~{pr_perex:.1f} distinct dims/example via attention over {M} slots\n")
    del bm; gc.collect(); torch.cuda.empty_cache()

print("=" * 74)
print("READ: graph delivers a near-rank-1 ADDITIVE nudge (same for all positions) → can't")
print("reconstruct distinct masked tokens. Baselines expose M content-addressable slots the")
print("decoder reads per-position. cond_recon needs only ONE value → rank-1 read is enough.")
