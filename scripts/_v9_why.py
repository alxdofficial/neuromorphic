"""Why is hlvocab v1 a WEAK compressor? Load the trained fix2 model and measure:
  1. node collapse — did the LEARNED vocabulary collapse to few directions?
  2. node usage — how many of the N nodes are actually used (effective vocab)?
  3. memory diversity — are the 16 emitted slots distinct or redundant (eff. rank)?
  4. per-slot value — recon vs #slots used (does each added slot add info, or do
     they saturate fast = each slot carries the same gist = pooling-blur)?
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_sentence import make_sentence_dataloader

dev = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"
CKPT = "outputs/repr_learning/mae_135m_4k_v9fix2_graph_v9_baseline/ckpts/graph_v9_baseline.best.pt"


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "sentence_mae"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.hlvocab_d_code = 256; cfg.hlvocab_nodes = (512, 256, 128)
    cfg.hlvocab_top_k = 4; cfg.hlvocab_m_max = 16; cfg.hlvocab_tap_layer = 6
    return cfg


def eff_rank(X):  # participation-ratio effective rank of rows of X
    X = F.normalize(X.float(), dim=-1)
    s = torch.linalg.svdvals(X - X.mean(0, keepdim=True))
    s2 = (s * s); return (s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item()


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
cfg = matched(ReprConfig())
model = ReprLearningModel(cfg, variant="hlvocab_baseline").to(dev)
ck = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ck["model_state_dict"], strict=False)
model.eval()
sub = model.encoder.sub
print(f"=== trained fix2 (step {ck.get('best_val_step')}, val_recon {ck.get('best_val_recon'):.3f}) ===")

dl = make_sentence_dataloader(tok, batch_size=16, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
batches = []
it = iter(dl)
for _ in range(4):
    b = next(it)
    for a in ("context_ids","context_mask","question_ids","question_mask",
              "answer_ids","answer_mask","answer_content_mask"):
        v = getattr(b, a, None)
        if torch.is_tensor(v): setattr(b, a, v.to(dev))
    batches.append(b)

# ── 1. NODE COLLAPSE (trained keys/values) ───────────────────────────────────
print("\n(1) NODE COLLAPSE — trained vocabulary (eff_rank of N nodes; mean|cos|)")
for l in range(sub.depth):
    for nm, P in (("keys", sub.node_keys[l]), ("vals", sub.node_values[l])):
        N = P.shape[0]; u = F.normalize(P.float(), dim=-1); c = (u @ u.t()).abs()
        off = (c.sum() - N) / (N * (N - 1))
        print(f"  L{l} {nm}: N={N:4d}  eff_rank={eff_rank(P):6.1f}  mean|cos|={off:.3f}")

# ── 2/3. usage + memory diversity over real forward ──────────────────────────
print("\n(2) NODE USAGE + (3) MEMORY DIVERSITY (real forward, 4 batches)")
usage_cnt = [torch.zeros(n, device=dev) for n in sub.config.nodes]
mem_ranks, mem_cos, sel_layer_hist = [], [], torch.zeros(sub.depth, device=dev)
with torch.no_grad():
    for b in batches:
        st = model.encoder.init_streaming_state(b.context_ids.shape[0], dev, torch.float32)
        emb = model.decoder.llama.get_input_embeddings()(b.context_ids)
        st, _ = model.encoder.streaming_write(st, emb, b.context_mask)
        hiddens, mask = st["hiddens"], st["mask"].float()
        memory, aux = sub(hiddens, mask)                      # real [B,16,d]
        for bi in range(memory.shape[0]):
            mem_ranks.append(eff_rank(memory[bi]))
            u = F.normalize(memory[bi].float(), dim=-1); c = (u @ u.t()).abs()
            M = c.shape[0]; mem_cos.append(((c.sum() - M) / (M * (M - 1))).item())
        # node usage = argmax routing per layer
        m = mask.unsqueeze(-1); x = sub._unit_rms_in(emb) if hasattr(sub, "_unit_rms_in") else None
        from src.repr_learning.hierarchical_learned_vocab import _unit_rms
        x = _unit_rms(sub.seed_proj(hiddens.float())) * m
        for l in range(sub.depth):
            sc = sub.route(l, (hiddens.float() if l == 0 else x)) * m
            am = sc.argmax(-1)[mask.bool()]
            usage_cnt[l].scatter_add_(0, am, torch.ones_like(am, dtype=torch.float))
            if l < sub.depth - 1: x = sub._perturb(l, x, sc) * m
for l in range(sub.depth):
    used = (usage_cnt[l] > 0).sum().item(); N = sub.config.nodes[l]
    p = usage_cnt[l] / usage_cnt[l].sum().clamp_min(1)
    ent = -(p * p.clamp_min(1e-12).log()).sum().item()
    print(f"  L{l}: used {used:4d}/{N} ({100*used/N:4.1f}%)  usage_entropy={ent:.2f} "
          f"(max={math.log(N):.2f})  perplexity={math.exp(ent):.0f}")
import statistics as st_
print(f"  MEMORY (16 slots): eff_rank mean={st_.mean(mem_ranks):.1f}/16  "
      f"mean|cos|={st_.mean(mem_cos):.3f}  (->1 = redundant/blurry)")

# ── 4. PER-SLOT VALUE: recon vs #slots used (the money shot) ─────────────────
print("\n(4) PER-SLOT VALUE — recon vs #memory slots (same 85% mask; k=0 is floor)")
mask_ratio = 0.85
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for kk in [0, 1, 2, 3, 4, 6, 8, 12, 16]:
        tot, n = 0.0, 0
        for bi, b in enumerate(batches):
            B, T = b.context_ids.shape
            emb = model.decoder.llama.get_input_embeddings()(b.context_ids)
            st = model.encoder.init_streaming_state(B, dev, torch.float32)
            st, _ = model.encoder.streaming_write(st, emb, b.context_mask)
            memory, _ = model.encoder.finalize_memory(st)
            memory = memory[:, :kk].to(emb.dtype)
            torch.manual_seed(1234 + bi)                       # SAME mask across kk
            rnd = torch.rand(B, T, device=dev)
            masked = (rnd < mask_ratio) & b.context_mask
            mvec = model.decoder.mask_embed.to(emb.dtype)
            dec = torch.where(masked.unsqueeze(-1), mvec.view(1, 1, -1), emb)
            full = torch.cat([memory, dec], dim=1)
            attn = torch.cat([torch.ones(B, kk, device=dev), b.context_mask.float()], dim=1).long()
            h = model.decoder.llama.model(inputs_embeds=full, attention_mask=attn,
                                          use_cache=False).last_hidden_state[:, kk:]
            lm = (masked[:, 1:] & b.context_mask[:, 1:])
            sh = h[:, :-1][lm]; tg = b.context_ids[:, 1:][lm]
            ce = F.cross_entropy(model.decoder.llama.lm_head(sh).float(), tg, reduction="sum")
            tot += ce.item(); n += lm.sum().item()
        print(f"  k={kk:2d} slots -> recon={tot/n:.3f}" + ("   (floor/OFF)" if kk == 0 else ""))
print("\nDONE.")
