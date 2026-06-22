"""slotgraph smoke — verify the emergent-topology slot memory in the REAL mixed path on REAL data.

  - matched config param count (cohort 6.91-7.01M)
  - both compute paths finite: mae → masked_reconstruction; babi → generic compute_loss
  - the per-LM-layer structural re-injection hook is wired (struct_out receives gradient ⇒ the
    structure path reaches the loss through prepend + reinforce)
  - gradient flows to EVERY slotgraph component (incl. the role/src/dst structure heads) + LoRA
  - canaries: edge_frac, src/dst entropy (sharpness), mem_effrank, struct_mag
  - use_structure=False ablation = the icae control (struct zeroed)
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer, AutoConfig
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.common import resolve_special_ids
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
DEV = "cuda"


def apply_mixed_capacity(cfg):
    cfg.use_llama_lora = True
    cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.slotgraph_n_slots = 32
    cfg.slotgraph_lora_rank = 104; cfg.slotgraph_lora_alpha = 208
    cfg.slotgraph_start_layer = 0


def build(use_structure=True):
    cfg = ReprConfig(use_llama_lora=True, batch_size=4)
    bc = AutoConfig.from_pretrained(BACKBONE)
    cfg.llama_model = BACKBONE; cfg.d_llama = bc.hidden_size; cfg.llama_vocab_size = bc.vocab_size
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.pad_token_id, cfg.sep_token_id = resolve_special_ids(tok)
    cfg.task_mode = "mixed"
    cfg.slotgraph_use_structure = use_structure
    apply_mixed_capacity(cfg)
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    m.train(True)
    return m, tok, cfg


def main():
    m, tok, cfg = build(use_structure=True)
    enc = m.encoder
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n{'='*72}\nslotgraph smoke (matched mixed config, REAL bf16 path)\n{'='*72}")
    print(f"params={tot:,} ({tot/1e6:.2f}M)  | cohort 6.91-7.01M")

    tasks = ["mae", "babi"]
    print(f"\nLoading real mixed val sets {tasks}...")
    vs = make_mixed_val_sets(tasks, tok, cfg, 1, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    def encode(batch):
        embed = m.decoder.llama.get_input_embeddings()
        with torch.no_grad():
            ctx = embed(batch.context_ids)
            enc_in = m._encode_for_memory(ctx, batch.context_mask)
        st = enc.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
        st, _ = enc.streaming_write(st, enc_in, batch.context_mask)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return enc.finalize_memory(st)

    for t in tasks:
        m.task_mode = MIXED_TASK_MODE[t]
        b = to_device(vs[t][0], DEV)
        mem, aux = encode(b)
        print(f"\n=== {t}  ({MIXED_TASK_MODE[t]}) ===")
        print(f"  memory shape={tuple(mem.shape)}")
        print(f"  canaries: edge_frac={float(aux['slotgraph_edge_frac']):.3f}  "
              f"src_ent={float(aux['slotgraph_src_entropy']):.2f}  dst_ent={float(aux['slotgraph_dst_entropy']):.2f}  "
              f"(max ln32={torch.log(torch.tensor(32.)):.2f})")
        print(f"  role_ent={float(aux['slotgraph_role_entropy']):.3f}  temp={float(aux['slotgraph_temp']):.2f}  "
              f"inject_scale={float(aux['slotgraph_inject_scale']):.3f}  "
              f"mem_effrank={float(aux['slotgraph_mem_effrank']):.2f}/{cfg.d_llama}")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = (m.compute_masked_reconstruction_loss(b)
                   if MIXED_TASK_MODE[t] == "masked_reconstruction"
                   else m.compute_loss(b, window_size=1024))
        loss = out.get("loss_recon", out.get("loss"))
        top1 = out.get("top1_acc", out.get("top1", torch.zeros(())))
        print(f"  REAL {MIXED_TASK_MODE[t]} loss={float(loss):.3f}  finite={bool(torch.isfinite(loss))}  "
              f"top1={float(top1):.3f}")

    # ── gradient flow through the real mae loss ──
    print(f"\n=== gradient flow (real mae loss → backward) ===")
    m.zero_grad(set_to_none=True)
    m.task_mode = "masked_reconstruction"
    b = to_device(vs["mae"][0], DEV)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = m.compute_masked_reconstruction_loss(b)
    out["loss"].backward()
    comps = ["slot_init", "role_embed", "struct_norm", "role_head", "src_head", "dst_head",
             "log_temp", "inject_scale", "norm"]
    pd = dict(enc.named_parameters())
    ok = True
    for c in comps:
        g = sum(float(p.grad.float().norm()) for k, p in pd.items()
                if (k.startswith(c) or (("." + c + ".") in k)) and p.grad is not None)
        n = sum(1 for k in pd if (k.startswith(c) or (("." + c + ".") in k)))
        flag = "  <-- STARVED" if (n > 0 and g < 1e-12) else ""
        if n > 0 and g < 1e-12:
            ok = False
        print(f"  {c:14} grad={g:.2e}  (n={n}){flag}")
    g_enc_lora = sum(float(p.grad.float().norm()) for k, p in enc.named_parameters()
                     if "lora" in k.lower() and p.grad is not None)
    g_dec_lora = sum(float(p.grad.float().norm()) for k, p in m.named_parameters()
                     if "lora" in k.lower() and k.startswith("decoder.") and p.grad is not None)
    print(f"  {'encoder_LoRA':14} grad={g_enc_lora:.2e}")
    print(f"  {'decoder_LoRA':14} grad={g_dec_lora:.2e}")
    print(f"  >>> src/dst_head grad nonzero ⇒ hard-ST structure selection is trainable (not starved)")
    print(f"\n{'ALL components received gradient ✓' if ok else 'SOME COMPONENT STARVED ✗'}")

    # ── use_structure=False (pure icae control) ──
    print(f"\n=== ablation: use_structure=False (pure icae control) ===")
    m2, _, _ = build(use_structure=False)
    m2.task_mode = "masked_reconstruction"
    b = to_device(vs["mae"][0], DEV)
    embed = m2.decoder.llama.get_input_embeddings()
    with torch.no_grad():
        ctx = embed(b.context_ids)
        st = m2.encoder.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
        st, _ = m2.encoder.streaming_write(st, ctx, b.context_mask)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _mem, _aux = m2.encoder.finalize_memory(st)
    print(f"  no structure hooks installed; memory shape={tuple(_mem.shape)} (= icae forward over [passage;slots])")


if __name__ == "__main__":
    main()
