"""slotgraph gradient-flow probe — is the relational (edge) machinery actually getting trained?

Loads the TRAINED slotgraph, runs several real (mae+babi) forward/backwards, and reports the mean
gradient norm reaching each component group. THE slotgraph splits the edge state into a semantic
relation vector R and a scalar confidence C, so the groups are:
  · RELATION  — op_W / phi / R-gates / rd_k / edge_norm  (the semantic-direction write)
  · CONFIDENCE— conf_W / conf_attn_W / conf_gate_*        (the scalar existence/strength write)
  · INJECT    — U / U_ln / resid_scale                    (value-path edge residual)
  · READ      — read_q/k/kx / edge_up / edge_sal          (message-passing read)
  · NODE      — node_init / node_gate / tok_proj          (node content path)
  · encoder / decoder LoRA
If the RELATION+CONFIDENCE gradient is ≪ the NODE/LoRA gradient (and tiny in absolute terms), the
edge heads get no effective training signal → they stay near init → the graph carries no bits.
(Pre-R/C this probed q_src_head/msg/slot_init, all since removed — updated 2026-07.)
"""
from __future__ import annotations
import sys, dataclasses, argparse
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.training import make_mixed_val_sets, to_device
from src.memory.data.mixes import TASK_MODE

DEV = "cuda"

# name → startswith prefix (a "LORA:" prefix additionally requires "lora" in the param name).
GROUPS = {
    "REL op_W":        "encoder.op_W",
    "REL phi":         "encoder.phi",          # phi_i / phi_j / phi_ln
    "REL gates":       "encoder.gate_",        # gate_a_i/j, gate_b_i/j (+ biases)
    "REL rd_k":        "encoder.rd_k",
    "REL rel_q/k":     "encoder.rel_",         # sparse-relation routing projections (rel_q/rel_k/rel_ln)
    "REL edge_norm":   "encoder.edge_norm",
    "CONF conf_W":     "encoder.conf_W",
    "CONF conf_attn":  "encoder.conf_attn_W",
    "CONF conf_gates": "encoder.conf_gate",    # conf_gate_a/b_i/j (+ biases/obs)
    "INJECT U":        "encoder.U",            # U + U_ln
    "INJECT resid":    "encoder.resid_scale",
    "READ q/k/kx":     "encoder.read_",
    "READ edge_up":    "encoder.edge_up",
    "READ edge_sal":   "encoder.edge_sal",
    "NODE init/gate":  "encoder.node_",        # node_init / node_gate / node_logsig
    "NODE tok_proj":   "encoder.tok_proj",
    "NODE type_embed": "encoder.type_embed",   # node/pointer type codes
    "READ out_norm":   "encoder.norm",         # final _NormMatch on emitted memory
    "encoder_LoRA":    "LORA:encoder.base",
    "decoder_LoRA":    "LORA:decoder.llama",
}
EDGE = [g for g in GROUPS if g.startswith(("REL", "CONF", "INJECT"))]   # the relational machinery
NODE = [g for g in GROUPS if g.startswith(("NODE", "READ"))]            # content / readout path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "outputs/memory/podrun_slotgraph_baseline/ckpts/slotgraph_baseline.best.pt"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--m-slots", type=int, default=96)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]; valid = {f.name for f in dataclasses.fields(ReprConfig)}
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    m.load_state_dict(sd["model_state_dict"], strict=False)
    m.train(True)
    vs = make_mixed_val_sets(["mae", "babi"], tok, cfg, 4, ctx_len=args.ctx, m_slots=args.m_slots,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    def grad_for(prefix):
        lora = prefix.startswith("LORA:"); base = prefix[5:] if lora else prefix
        tot = 0.0
        for n, p in m.named_parameters():
            if p.grad is None:
                continue
            if n.startswith(base) and (not lora or "lora" in n.lower()):
                tot += float(p.grad.float().norm()) ** 2
        return tot ** 0.5

    acc = {g: 0.0 for g in GROUPS}; nb = 0
    for t in ("mae", "babi"):
        m.task_mode = TASK_MODE[t]
        for b in vs[t]:
            b = to_device(b, DEV)
            m.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = (m.compute_masked_reconstruction_loss(b) if t == "mae"
                       else m.compute_loss(b, window_size=args.ctx))
            out["loss"].backward()
            for g, pref in GROUPS.items():
                acc[g] += grad_for(pref)
            nb += 1

    print(f"\n{'='*60}\nslotgraph gradient flow on the TRAINED model ({nb} batches)\n{'='*60}")
    for g in GROUPS:
        print(f"  {g:20} mean grad = {acc[g]/nb:.3e}")
    edge_sum = sum(acc[g] for g in EDGE) / nb
    node_sum = sum(acc[g] for g in NODE) / nb
    print(f"\n  EDGE (relation+conf+inject) total = {edge_sum:.3e}   NODE/READ total = {node_sum:.3e}   "
          f"ratio edge/node = {edge_sum/max(node_sum,1e-12):.4f}")
    print(f"  {'→ edge machinery gets ≪ node gradient (no effective pressure)' if edge_sum < 0.2*node_sum else '→ edge machinery gets comparable gradient'}")


if __name__ == "__main__":
    main()
