"""slotgraph v2 (graph-as-language) smoke — verify the redesign in the REAL mixed bf16 path.

Checklist (docs/slotgraph_redesign.md §6):
  1. both compute paths finite (mae → masked_reconstruction; babi → generic compute_loss); no NaN.
  2. cross-attn gates init 0 ⇒ step-0 read is a NO-OP ⇒ REAL == OFF (clean cold-start).
  3. gradient reaches every component: encoder GT layers, structure heads, read cross-attn, LoRA.
  4. node-dropout drops NODES, keeps EDGES; a forward at p=p_max is finite.
  5. endpoints valid (edges→nodes, ~0 self-loops); canaries sane.
  6. params ~ param-matched to icae (~6.9M).
STOP after this for review before any training.
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
    cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32          # decoder LoRA
    cfg.slotgraph_d_node = 64
    cfg.slotgraph_n_nodes = 144; cfg.slotgraph_n_edges = 144     # (N+E)*64 = 18432 floats
    cfg.slotgraph_enc_layers = 4
    cfg.slotgraph_xattn_heads = 4
    cfg.slotgraph_lora_rank = 56; cfg.slotgraph_lora_alpha = 112  # encoder LoRA (matches train.py capacity)


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


def grp(component, params):
    n = sum(p.numel() for p in params)
    g = sum(1 for p in params if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    t = sum(1 for p in params if p.requires_grad)
    return f"{component:18} grad {g}/{t}  ({n/1e6:.2f}M)"


def main():
    m, tok, cfg = build(use_structure=True); enc = m.encoder
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n{'='*74}\nslotgraph v2 smoke (matched mixed config, REAL bf16 path)\n{'='*74}")
    print(f"params={tot:,} ({tot/1e6:.2f}M)  | target ~6.9M (icae)")
    print(f"gates init: read_xattn[0].gate={float(enc.read_xattn[0].gate.detach()):.4f} (small +ve to "
          f"bootstrap the encoder), id_scale={float(enc.id_scale.detach()):.3f}")

    vs = make_mixed_val_sets(["mae", "babi"], tok, cfg, 3, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    for task in ("mae", "babi"):
        m.task_mode = MIXED_TASK_MODE[task]
        b = to_device(vs[task][0], DEV)
        # (2) cold-start: REAL == OFF at init (gates 0 → read no-op)
        m.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            real0 = m.compute_loss(b, window_size=1024)["loss_recon"].item()
            off0 = m.compute_loss(b, window_size=1024, zero_memory=True)["loss_recon"].item()
            shuf0 = m.compute_loss(b, window_size=1024, shuffle_memory=True)["loss_recon"].item()
        m.train(True)
        print(f"\n[{task}] init REAL={real0:.4f} OFF={off0:.4f} SHUF={shuf0:.4f}  "
              f"(read active at init by gate; REAL−OFF={real0-off0:+.4f})")
        # (1,3) forward+backward, grad coverage
        m.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m.compute_loss(b, window_size=1024)
        finite = torch.isfinite(out["loss"]).item()
        out["loss"].backward()
        print(f"[{task}] train loss={out['loss'].item():.4f} finite={finite}")
        print("   " + grp("GT encoder", list(enc.gt_layers.parameters())))
        print("   " + grp("read cross-attn", list(enc.read_xattn.parameters())))
        sh = [p for L in enc.gt_layers for p in (*L.q_src.parameters(), *L.q_dst.parameters(),
              *L.k_node.parameters(), *L.edge_combine.parameters())]
        print("   " + grp("structure heads", sh))
        print("   " + grp("ids+role+seeds", [enc.node_id, enc.id_scale, enc.role_embed, enc.slot_init]))

    # (4) node-dropout: drops nodes, keeps edges; forward at p_max finite
    enc.node_drop_p = cfg.slotgraph_node_drop_max
    keep = enc.node_keep_mask(4, DEV, training=True)
    dropped_nodes = int((~keep[:, :enc.N]).any(0).sum()); edges_all_kept = bool(keep[:, enc.N:].all())
    m.task_mode = "masked_reconstruction"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ld = m.compute_loss(to_device(vs["mae"][0], DEV), window_size=1024)["loss"]
    print(f"\n[node-drop p={enc.node_drop_p}] dropped {dropped_nodes}/{enc.N} nodes, edges all kept: "
          f"{edges_all_kept}, forward finite: {torch.isfinite(ld).item()}")
    enc.node_drop_p = 0.0

    # (5) canaries
    m.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        m.task_mode = "babi"
        out = m.compute_loss(to_device(vs["babi"][0], DEV), window_size=1024)
    can = {k: round(float(v), 3) for k, v in out.items()
           if k.startswith("slotgraph_") and torch.is_tensor(v) and v.ndim == 0}
    print(f"\ncanaries: {can}")
    print(f"\n{'='*74}\nSMOKE DONE — review before training\n{'='*74}")


if __name__ == "__main__":
    main()
