"""MAE objective + backbone validation: the floor/ceiling band scan.

Validates three things at once before any compressor is wired:
  1. true-MAE mechanic: causal masked-infill — replace ~85% of span tokens with a
     [MASK] embedding, predict the true token at masked positions in ONE forward
     (position p predicted from hidden p-1, whose causal prefix is masks/anchors +
     memory — NOT the true tokens, so the teacher-forcing local-prior cheat is gone).
  2. SmolLM2 backbone works end-to-end with the sentence-pair data.
  3. the BAND: CE with no memory (FLOOR) vs full-context-as-memory (CEILING). A wide
     band = room for a compressor to show signal; a narrow band = the objective
     can't resolve compression quality (the EMAT problem).

Memory here is a stand-in (the true span embeddings as a "perfect uncompressed
code") for the ceiling, and absent for the floor — no learned compressor yet.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.repr_learning.data_sentence import make_sentence_dataloader

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--src-tok", default="meta-llama/Llama-3.2-1B")
ap.add_argument("--mask-ratio", type=float, default=0.85)
ap.add_argument("--batches", type=int, default=20)
args = ap.parse_args()
device = "cuda"

tok = AutoTokenizer.from_pretrained(args.backbone)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=torch.float32).to(device).eval()
d = model.config.hidden_size
embed = model.get_input_embeddings()
# learned [MASK] embedding stand-in: use the mean token embedding (deterministic,
# reasonable; a trained run would learn it)
with torch.no_grad():
    mask_vec = embed.weight.mean(0).detach()

dl = make_sentence_dataloader(tok, batch_size=8, src_tokenizer_name=args.src_tok,
                              split="val", num_workers=0, pad_token_id=tok.pad_token_id)


def mae_ce(ids, ctx_mask, memory_embeds, mask_ratio, gen):
    """Causal masked-infill CE on the span. memory_embeds [B,M,d] or None (floor).
    Returns mean CE over masked, predicted positions."""
    B, T = ids.shape
    real = ctx_mask
    span_emb = embed(ids)                                                   # [B,T,d]
    # mask ~mask_ratio of REAL positions (anchors = the rest); never mask pads
    rnd = torch.rand(B, T, generator=gen, device=ids.device)
    masked = (rnd < mask_ratio) & real
    dec_in = torch.where(masked.unsqueeze(-1), mask_vec.view(1, 1, d), span_emb)
    if memory_embeds is not None:
        M = memory_embeds.shape[1]
        full = torch.cat([memory_embeds, dec_in], dim=1)
        attn = torch.cat([torch.ones(B, M, device=ids.device), real.float()], dim=1)
        offset = M
    else:
        full = dec_in
        attn = real.float()
        offset = 0
    out = model.model(inputs_embeds=full, attention_mask=attn.long(), use_cache=False)
    hid = out.last_hidden_state                                             # [B, M+T, d]
    logits = model.lm_head(hid[:, offset:])                                 # [B,T,V] span logits
    # predict token p from hidden p-1; target masked positions
    pred = logits[:, :-1]                                                   # predicts t_1..t_{T-1}
    tgt = ids[:, 1:]
    loss_mask = masked[:, 1:] & real[:, 1:]
    if loss_mask.sum() == 0:
        return None
    ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1])[loss_mask.reshape(-1)],
                         tgt.reshape(-1)[loss_mask.reshape(-1)])
    return ce.item()


gen = torch.Generator(device=device).manual_seed(0)
floors, ceils, ks = [], [], []
it = iter(dl)
with torch.no_grad():
    for _ in range(args.batches):
        b = next(it)
        ids = b.context_ids.to(device); cm = b.context_mask.to(device)
        # CEILING: memory = the true (uncompressed) span embeddings (perfect code)
        ceil_mem = embed(ids)
        f = mae_ce(ids, cm, None, args.mask_ratio, gen)
        c = mae_ce(ids, cm, ceil_mem, args.mask_ratio, gen)
        if f and c:
            floors.append(f); ceils.append(c); ks.append(b.k_slots)

import numpy as np
fl, cl = np.mean(floors), np.mean(ceils)
print(f"\n=== MAE band scan: {args.backbone} (mask {args.mask_ratio}) ===")
print(f"  FLOOR  (no memory)            CE = {fl:.4f}  ppl {np.exp(fl):.1f}")
print(f"  CEILING (full uncompressed)   CE = {cl:.4f}  ppl {np.exp(cl):.1f}")
print(f"  BAND (floor - ceiling)        = {fl - cl:.4f}  ({100*(fl-cl)/fl:.0f}% of floor)")
print(f"  (a compressor's CE must land between these; wide band = measurable signal)")
print(f"  n={len(floors)} batches, code size k≈{int(np.median(ks))}")
