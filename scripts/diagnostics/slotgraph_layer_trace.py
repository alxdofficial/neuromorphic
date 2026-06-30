"""slotgraph DEPTH trace — per-write-layer metrics in one forward, to locate WHERE (which of the 4
write layers) binding / pooling / routing-collapse sets in. Run on a trained ckpt for the end-of-training
depth profile; the TIME axis (depth × training-step) comes from a normal training run (per-layer canaries
are auto-logged at each val step when the encoder's collect_layer_metrics flag is on — train.py sets it).

  mem_effrank  — rank of the READ tokens if materialized from THIS layer (rank-1 = pooling). THE key row:
                 does the read collapse early (immediate) or late (progressive across depth)?
  node/edge_effrank — vocabulary / edge-state distinctness per layer
  nodes_used / avg_degree / n_components — selection spread vs hub, per layer
  src_entropy / routing_diversity — selection decisiveness / input-dependence, per layer

Usage: .venv/bin/python scripts/diagnostics/slotgraph_layer_trace.py --ckpt outputs/memory/sgv3sink_slotgraph_baseline/ckpts/slotgraph_baseline.last.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer, AutoConfig
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.common import resolve_special_ids
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

BACKBONE = "HuggingFaceTB/SmolLM2-135M"; DEV = "cuda"
TASKS = ("mae", "babi", "continuation", "condrecon_bio")
ROWS = ("mem_effrank", "node_effrank", "edge_effrank", "nodes_used", "avg_degree",
        "n_components", "src_entropy", "routing_diversity")


def build(ckpt):
    cfg = ReprConfig(use_llama_lora=True, batch_size=8)
    bc = AutoConfig.from_pretrained(BACKBONE)
    cfg.llama_model = BACKBONE; cfg.d_llama = bc.hidden_size; cfg.llama_vocab_size = bc.vocab_size
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.pad_token_id, cfg.sep_token_id = resolve_special_ids(tok)
    cfg.task_mode = "mixed"; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.slotgraph_d_node = 64; cfg.slotgraph_n_nodes = 144; cfg.slotgraph_n_edges = 144
    cfg.slotgraph_enc_layers = 4; cfg.slotgraph_xattn_heads = 4
    cfg.slotgraph_lora_rank = 56; cfg.slotgraph_lora_alpha = 112
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    if ckpt:
        sd = torch.load(ckpt, map_location=DEV)
        state = sd
        for k in ("model_state_dict", "model", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                state = sd[k]; break
        miss, unexp = m.load_state_dict(state, strict=False)
        print(f"loaded {ckpt}: {len(miss)} missing, {len(unexp)} unexpected")
    return m, tok, cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=None); args = ap.parse_args()
    m, tok, cfg = build(args.ckpt); enc = m.encoder
    enc.collect_layer_metrics = True
    n_layers = len(enc.gt_layers)
    state = os.path.basename(args.ckpt) if args.ckpt else "UNTRAINED"
    print(f"\n{'='*78}\nslotgraph DEPTH trace — {state}  ({n_layers} write layers)\n{'='*78}")
    vs = make_mixed_val_sets(list(TASKS), tok, cfg, 8, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)
    m.eval()
    for task in TASKS:
        b = to_device(vs[task][0], DEV)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            m.task_mode = MIXED_TASK_MODE[task]
            out = m.compute_loss(b, window_size=1024)
        print(f"\n-- {task} " + "-" * (72 - len(task)))
        print(f"  {'metric':18}" + "".join(f"{'L'+str(i):>9}" for i in range(n_layers)) + f"{'→read':>9}")
        for r in ROWS:
            cells = []
            for i in range(n_layers):
                v = out.get(f"slotgraph_L{i}_{r}")
                cells.append(f"{float(v):9.2f}" if v is not None else "    --   ")
            fin = out.get(f"slotgraph_{r}")
            finc = f"{float(fin):9.2f}" if fin is not None else "    --   "
            print(f"  {r:18}" + "".join(cells) + finc)
    print(f"\n{'='*78}\n  mem_effrank is THE row: where across depth does the read collapse to ~rank-1 (pooling)?\n{'='*78}")


if __name__ == "__main__":
    main()
