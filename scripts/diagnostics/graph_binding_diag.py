"""graph model — the binding-gate diagnostic on a trained checkpoint.

The wall every prior model (v6/v8/v9) failed: the read collapses to ~rank-1
(membership, not binding). This measures, on the trained `graph` ckpt:
  1. REAL / OFF / SHUF recon — the binding gate (want REAL << SHUF, SHUF ~>= OFF).
  2. nodes_used — how many distinct bank nodes the edge endpoints actually point to
     (the vocabulary-collapse canary).
  3. eff_rank of the edge-states across the E edges (is the continuous relation
     channel carrying distinct per-edge content, or pooled to one direction?).
  4. eff_rank of the INJECTED read signal across decode positions (the binding-
     relevant rank — prior models collapsed this to ~1).

Run after a graph_baseline run:  CKPT=outputs/memory/<tag>_graph_baseline/ckpts/graph_baseline.best.pt python scripts/diagnostics/graph_binding_diag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.data_masked_reconstruction import make_sentence_dataloader

dev = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"
CKPT = os.environ.get("CKPT",
    "outputs/memory/graph_v1_4k_graph_baseline/ckpts/graph_baseline.best.pt")
N_BATCHES = int(os.environ.get("N_BATCHES", "8"))


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "masked_reconstruction"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.n_flat_codes = 16
    return cfg                       # graph_* defaults come from config.py


def eff_rank(X):                     # participation-ratio effective rank of rows
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    s2 = s * s
    return (s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item()


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = ReprLearningModel(matched(ReprConfig()), variant="graph_baseline").to(dev)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
res = model.load_state_dict(sd["model_state_dict"], strict=False)
missing_mem = [k for k in res.missing_keys
               if not k.startswith("decoder.llama.") and not k.startswith("encoder.base.")]
print(f"loaded {CKPT}\n  step={sd.get('step')}  missing(mem)={missing_mem[:4]}  unexpected={res.unexpected_keys[:4]}")
model.eval()
model.task_mode = "masked_reconstruction"

dl = make_sentence_dataloader(tok, batch_size=64, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
it = iter(dl)

enc = model.encoder
reals, offs, shufs = [], [], []
codes_used, edge_ranks, inj_ranks = [], [], []
for bi in range(N_BATCHES):
    batch = next(it)
    for a in ("context_ids", "context_mask"):
        setattr(batch, a, getattr(batch, a).to(dev))
    torch.manual_seed(1234 + bi)     # SAME mask across REAL/OFF/SHUF for this batch
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(1234 + bi); reals.append(model.compute_masked_reconstruction_loss(batch)["loss_recon"].item())
        torch.manual_seed(1234 + bi); offs.append(model.compute_masked_reconstruction_loss(batch, zero_memory=True)["loss_recon"].item())
        torch.manual_seed(1234 + bi); shufs.append(model.compute_masked_reconstruction_loss(batch, shuffle_memory=True)["loss_recon"].item())

    # graph geometry + injected-read rank (REAL path), captured via the hook
    with torch.no_grad():
        embed = model.decoder.llama.get_input_embeddings()
        ctx = embed(batch.context_ids)
        st = enc.init_streaming_state(ctx.shape[0], dev, ctx.dtype)
        st, _ = enc.streaming_write(st, ctx, batch.context_mask)
        _, aux = enc.finalize_memory(st)
        g = aux["graph"]
        sel = torch.cat([g["src_ptr"].argmax(-1).reshape(-1), g["dst_ptr"].argmax(-1).reshape(-1)])
        codes_used.append(sel.unique().numel())                  # distinct nodes pointed to
        # edge-state diversity across the E edges (flatten batch×E → eff rank)
        es = g["edge_state"].reshape(-1, g["edge_state"].shape[-1])
        edge_ranks.append(eff_rank(es))
        # injected read signal across decode positions (capture inj from the hook)
        cap = {}
        layer = model.decoder.llama.model.layers[enc.inject_layer]
        def _cap(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            cap["inj"] = enc.reader(h, g)
            return o
        hh = layer.register_forward_hook(_cap)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.decoder.llama.model(inputs_embeds=ctx, attention_mask=batch.context_mask.long(), use_cache=False)
        hh.remove()
        inj = cap["inj"]                                  # [B,T,d]
        # rank across positions, pooled over the batch (mask to real tokens)
        m = batch.context_mask.bool()
        inj_flat = inj[m]                                 # [N,d]
        inj_ranks.append(eff_rank(inj_flat))

R = sum(reals) / len(reals); O = sum(offs) / len(offs); S = sum(shufs) / len(shufs)
print(f"\n=== binding gate (mean over {N_BATCHES}×64) ===")
print(f"  REAL recon = {R:.4f}")
print(f"  OFF  recon = {O:.4f}   (OFF-REAL = {O-R:+.4f}  → memory helps)")
print(f"  SHUF recon = {S:.4f}   (SHUF-REAL = {S-R:+.4f} → content-specific binding)")
print(f"  SHUF-OFF   = {S-O:+.4f}   (>0 ⇒ wrong memory worse than none = strong binding)")
print(f"\n=== graph geometry ===")
print(f"  nodes_used (distinct nodes pointed to / batch): {sum(codes_used)/len(codes_used):.1f}  "
      f"(of N={enc.gcfg.n_nodes} bank, E={enc.gcfg.n_edges} edges → {2*enc.gcfg.n_edges} endpoints)")
print(f"  edge_state eff_rank (across edges):       {sum(edge_ranks)/len(edge_ranks):.2f}  (of d_graph={enc.gcfg.d_graph})")
print(f"  INJECTED read eff_rank (across positions): {sum(inj_ranks)/len(inj_ranks):.2f}  "
      f"(of d_llama={enc.cfg.d_llama}; prior models collapsed this to ~1)")
