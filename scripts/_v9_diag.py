"""hlvocab debug sweep — load the trained 4k checkpoint and measure the things
static review can't: gradient flow per module, routing collapse, node/centroid
collapse, presence saturation, multi-resolution selection, memory-norm OOD."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_sentence import make_sentence_dataloader
from src.repr_learning.models.hierarchical_learned_vocab.substrate import _unit_rms, _unit

dev = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"
CKPT = "outputs/repr_learning/mae_135m_4k_v9_graph_v9_baseline/ckpts/graph_v9_baseline.best.pt"


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
if os.environ.get("FRESH"):
    print("=== FRESH (untrained, new architecture) — mechanics check ===")
else:
    ck = torch.load(CKPT, map_location="cpu")
    print(f"=== CKPT: step={ck.get('step')} best_val_recon={ck.get('best_val_recon'):.4f} "
          f"@ step {ck.get('best_val_step')} ===")
    miss, unexp = model.load_state_dict(ck["model_state_dict"], strict=False)
    miss = [m for m in miss if "base." not in m and ".llama." not in m or "lora" in m]
    print(f"load: {len(miss)} missing (non-frozen), {len(unexp)} unexpected")
    if miss[:8]: print("  missing sample:", miss[:8])

dl = make_sentence_dataloader(tok, batch_size=16, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
batch = next(iter(dl))
for a in ("context_ids","context_mask","question_ids","question_mask",
          "answer_ids","answer_mask","answer_content_mask"):
    v = getattr(batch, a, None)
    if torch.is_tensor(v): setattr(batch, a, v.to(dev))
print(f"batch: ctx {tuple(batch.context_ids.shape)} k_slots={batch.k_slots}\n")

sub = model.encoder.sub

# ─────────────────────────────────────────────────────────────────────────────
# (A) GRADIENT FLOW — one real fwd/bwd, grad norm per module group
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 78); print("(A) GRADIENT FLOW (one real MAE step)"); print("=" * 78)
model.train()
model.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model.compute_mae_loss(batch)
out["loss"].backward()
groups = {}
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    key = ("decoder.LoRA" if "lora" in n.lower() else
           "sub." + n.split("sub.")[1].split(".")[0] if "sub." in n else "other:" + n.split(".")[0])
    g = 0.0 if p.grad is None else p.grad.detach().float().norm().item()
    pn = p.detach().float().norm().item()
    d = groups.setdefault(key, [0.0, 0.0, 0])
    d[0] += g * g; d[1] += pn * pn; d[2] += 1
print(f"  loss={out['loss'].item():.3f}  recon={out['loss_recon'].item():.3f}  "
      f"top1={out['top1_acc'].item():.3f}")
print(f"  {'group':<22}{'#tens':>6}{'grad_norm':>12}{'param_norm':>12}{'grad/param':>12}")
for k in sorted(groups):
    gn = math.sqrt(groups[k][0]); pn = math.sqrt(groups[k][1])
    print(f"  {k:<22}{groups[k][2]:>6}{gn:>12.2e}{pn:>12.2e}{(gn/max(pn,1e-12)):>12.2e}"
          + ("   <-- DEAD" if gn < 1e-9 else ""))

# ─────────────────────────────────────────────────────────────────────────────
# walk the substrate on the real tapped hiddens (eval, no grad)
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    st = model.encoder.init_streaming_state(batch.context_ids.shape[0], dev, torch.float32)
    emb = model.decoder.llama.get_input_embeddings()(batch.context_ids)
    st, _ = model.encoder.streaming_write(st, emb, batch.context_mask)
    hiddens, mask = st["hiddens"], st["mask"].float()
    B, L = hiddens.shape[:2]
    m = mask.unsqueeze(-1)
    x = _unit_rms(sub.seed_proj(hiddens.float())) * m

    print("\n" + "=" * 78); print("(B) ROUTING HEALTH per layer (target entropy = log(eff_k)=%.2f)" % math.log(8))
    print("=" * 78)
    print(f"  {'L':<3}{'nodes':>6}{'temp':>7}{'entropy':>9}{'hub_share':>11}{'active_frac':>12}{'coverage':>10}")
    sel_layer_of = []
    cand_tokens, cand_pres = [], []
    for l in range(sub.depth):
        route_in = hiddens.float() if l == 0 else x
        scores = sub.route(l, route_in) * m                       # [B,L,N]
        N = scores.shape[-1]
        # entropy per token (over nodes), averaged over real tokens
        p = scores.clamp_min(1e-12)
        ent = (-(p * p.log()).sum(-1) * mask).sum() / mask.sum()
        act_mass = scores.sum(dim=1)                              # [B,N]
        total = act_mass.sum(-1, keepdim=True).clamp_min(1e-6)
        hub_share = (act_mass / total).max(-1).values.mean()     # top node's share
        marg = getattr(sub, f"act_marginal_L{l}").clamp_min(1e-6)
        active_frac = (act_mass > marg).float().mean()
        argmax_nodes = (scores.argmax(-1) * mask.long() + (mask.long() - 1) * -1)  # winners
        coverage = scores.argmax(-1)[mask.bool()].unique().numel() / N
        denom = scores.sum(dim=1).clamp_min(1e-6)
        centroid = torch.einsum("bln,bld->bnd", scores, x) / denom.unsqueeze(-1)
        npmi = act_mass / marg
        presence = torch.sigmoid(sub.presence_a[l] * npmi + sub.presence_b[l])
        token = sub.emit_projs[l](centroid + sub.node_values[l].unsqueeze(0))
        cand_tokens.append(token); cand_pres.append(presence)
        sel_layer_of += [l] * N
        if l < sub.depth - 1:
            x = sub._perturb(l, x, scores) * m
        print(f"  {l:<3}{N:>6}{sub._route_temp(l).item():>7.3f}{ent.item():>9.3f}"
              f"{hub_share.item():>11.3f}{active_frac.item():>12.3f}{coverage:>10.3f}")

    print("\n" + "=" * 78); print("(C) NODE COLLAPSE (mean |off-diag cosine|; ->1 = collapsed)")
    print("=" * 78)
    for l in range(sub.depth):
        for name, P in (("keys", sub.node_keys[l]), ("values", sub.node_values[l])):
            u = F.normalize(P.float(), dim=-1)
            c = (u @ u.t()).abs()
            N = c.shape[0]; off = (c.sum() - N) / (N * (N - 1))
            print(f"  L{l} node_{name:<7} mean|cos|={off.item():.4f}  max|cos|(off)="
                  f"{(c - torch.eye(N, device=c.device)).max().item():.3f}")

    print("\n" + "=" * 78); print("(D) PRESENCE + MULTI-RESOLUTION SELECTION")
    print("=" * 78)
    pres = torch.cat(cand_pres, dim=1)                            # [B, sumN]
    tokens = torch.cat(cand_tokens, dim=1)
    print(f"  presence: mean={pres.mean():.3f} std={pres.std():.3f} "
          f"frac>0.99={ (pres>0.99).float().mean():.3f} frac<0.01={(pres<0.01).float().mean():.3f}")
    topp, topi = pres.topk(min(sub.config.m_max, pres.shape[1]), dim=1)
    layer_of = torch.tensor(sel_layer_of, device=dev)
    sel_layers = layer_of[topi]                                  # [B,16]
    for l in range(sub.depth):
        frac = (sel_layers == l).float().mean()
        print(f"  selected-from-L{l}: {frac.item()*100:5.1f}%  (layer has "
              f"{sub.config.nodes[l]}/{sum(sub.config.nodes)} nodes = "
              f"{100*sub.config.nodes[l]/sum(sub.config.nodes):.0f}% of pool)")
    print(f"  selected-presence: mean={topp.mean():.3f} std={topp.std():.3f}")
    # diversity of the 16 selected memory tokens (pre-gate)
    sel = tokens.gather(1, topi.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    u = F.normalize(sel.float(), dim=-1)
    cos = torch.einsum("bid,bjd->bij", u, u).abs()
    M = cos.shape[1]; offc = (cos.sum((1, 2)) - M) / (M * (M - 1))
    print(f"  selected-token mean|cos| (blurriness; ->1 = all-same)={offc.mean().item():.4f}")

    print("\n" + "=" * 78); print("(E) MEMORY-NORM OOD (vs real SmolLM2 token embeddings ~0.9)")
    print("=" * 78)
    memory = sel * topp.unsqueeze(-1)
    mn = memory.float().norm(dim=-1)
    real = emb[batch.context_mask.bool()].float().norm(dim=-1)
    print(f"  memory token L2 norm: mean={mn.mean():.3f} std={mn.std():.3f} "
          f"min={mn.min():.3f} max={mn.max():.3f}")
    print(f"  REAL SmolLM2 embed L2 norm: mean={real.mean():.3f} std={real.std():.3f}")
    print(f"  OOD factor (memory/real) = {(mn.mean()/real.mean()).item():.1f}x")
print("\nDONE.")
