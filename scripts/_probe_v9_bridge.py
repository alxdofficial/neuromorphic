"""Bridge-hypothesis probe (overnight run-2 follow-up).

Claim under test: the question routes (via the verbatim KEY phrase) into the
key's nodes, while the fact's VALUE content was deposited into the nodes the
VALUE tokens routed to — so the query never visits the stored content unless a
key->value bridge moves it. Quantified here on a trained arm-B checkpoint:

  key-ish context positions  = tokens whose id appears in the question (the key
                               phrase is verbatim in emat_bio)
  value-ish positions        = all other real context tokens
  measure: cos(question mean routing vector, per-group mean routing vector),
           plus the score-mass overlap of question routing vs each group's nodes.

If cos(Q, key) >> cos(Q, value): the query lands on key nodes; binding requires
content to MOVE there (the absorption bridge). Run on GPU between gate runs.
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphV9PyramidEncoder
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/repr_learning/emat_bio_v9b1_graph_v9_baseline/ckpts/graph_v9_baseline.last.pt"
ARM = sys.argv[2] if len(sys.argv) > 2 else "B"

device = "cuda"
cfg = ReprConfig()
cfg.graph_v9_arm = ARM
enc = GraphV9PyramidEncoder(cfg).to(device).eval()
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)  # our own ckpt; tensors only
sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
own_missing = [m for m in missing if not m.startswith("base.")]
print(f"[ckpt] step {ckpt.get('step')}: loaded {len(enc_sd)} tensors; "
      f"non-base missing={len(own_missing)} ({own_missing[:4]}) unexpected={len(unexpected)}")

tok = AutoTokenizer.from_pretrained(cfg.llama_model)
dl = make_emat_bio_dataloader(tok, context_len=640, batch_size=8, n_pairs=12,
                              n_query=1, n_facts=3, split="validation", world_seed=0,
                              stream_seed=7, pad_token_id=cfg.pad_token_id, num_workers=0)
batch = next(iter(dl))
ctx_ids = batch.context_ids.to(device); ctx_mask = batch.context_mask.to(device)
q_ids = batch.question_ids.to(device); q_mask = batch.question_mask.to(device)

with torch.no_grad():
    embeds = enc.base.get_input_embeddings()(ctx_ids)
    out = enc.base.model(inputs_embeds=embeds, attention_mask=ctx_mask.long(),
                         output_hidden_states=True, use_cache=False)
    h_ctx = out.hidden_states[cfg.graph_v9_tap_layer + 1].float()
    q_emb = enc.base.get_input_embeddings()(q_ids)
    q_out = enc.base.model(inputs_embeds=q_emb, attention_mask=q_mask.long(),
                           output_hidden_states=True, use_cache=False)
    h_q = q_out.hidden_states[cfg.graph_v9_tap_layer + 1].float()

    for layer_idx in range(enc.sub.depth):
        # routing input: layer 0 = raw hiddens; layer 1 = operated codes — for the
        # bridge question, layer 0 routing is the addressing primitive; report both
        # using raw hiddens as a first-order proxy for L1 (codes need full flow).
        if layer_idx == 0:
            sc_ctx = enc.sub.route(0, h_ctx) * ctx_mask.unsqueeze(-1).float()
            sc_q = enc.sub.route(0, h_q) * q_mask.unsqueeze(-1).float()
        else:
            from src.repr_learning.graph_substrate_v9 import _unit_rms
            st0 = enc.sub.init_state(ctx_ids.shape[0], device)
            codes_c = _unit_rms(enc.sub.seed_proj(h_ctx)) * ctx_mask.unsqueeze(-1).float()
            codes_c = enc.sub.apply_chain(codes_c, st0["factor_dirs"][0],
                                          st0["factor_strengths"][0],
                                          enc.sub.route(0, h_ctx) * ctx_mask.unsqueeze(-1).float())
            codes_c = _unit_rms(codes_c)
            sc_ctx = enc.sub.route(1, codes_c) * ctx_mask.unsqueeze(-1).float()
            codes_q = _unit_rms(enc.sub.seed_proj(h_q)) * q_mask.unsqueeze(-1).float()
            codes_q = enc.sub.apply_chain(codes_q, st0["factor_dirs"][0],
                                          st0["factor_strengths"][0],
                                          enc.sub.route(0, h_q) * q_mask.unsqueeze(-1).float())
            codes_q = _unit_rms(codes_q)
            sc_q = enc.sub.route(1, codes_q) * q_mask.unsqueeze(-1).float()

        results = {"key_cos": [], "val_cos": [], "key_mass": [], "val_mass": []}
        for b in range(ctx_ids.shape[0]):
            q_set = set(q_ids[b][q_mask[b].bool()].tolist())
            real = ctx_mask[b].bool()
            is_key = torch.tensor([int(t.item()) in q_set for t in ctx_ids[b]],
                                  device=device) & real
            is_val = (~is_key) & real
            if is_key.sum() < 2 or is_val.sum() < 2:
                continue
            q_vec = sc_q[b][q_mask[b].bool()].mean(0)
            key_vec = sc_ctx[b][is_key].mean(0)
            val_vec = sc_ctx[b][is_val].mean(0)
            results["key_cos"].append(F.cosine_similarity(q_vec, key_vec, dim=0).item())
            results["val_cos"].append(F.cosine_similarity(q_vec, val_vec, dim=0).item())
            # mass overlap: how much of the question's routing mass lands on the
            # top-32 nodes of each group
            top_key = key_vec.topk(32).indices
            top_val = val_vec.topk(32).indices
            results["key_mass"].append(q_vec[top_key].sum().item())
            results["val_mass"].append(q_vec[top_val].sum().item())
        mean = lambda xs: sum(xs) / max(len(xs), 1)
        print(f"L{layer_idx}: cos(Q,key)={mean(results['key_cos']):.4f}  "
              f"cos(Q,value)={mean(results['val_cos']):.4f}  "
              f"Q-mass on key-nodes={mean(results['key_mass']):.4f}  "
              f"on value-nodes={mean(results['val_mass']):.4f}")
print("interpretation: key>>value supports the bridge hypothesis "
      "(query lands on key nodes; stored value content lives elsewhere)")
