"""slotgraph gradient-flow probe — are the topology components actually getting trained?

Loads the TRAINED slotgraph, runs several real (mae+babi) forward/backwards, and reports the
mean gradient norm reaching each component group: the TOPOLOGY heads (role/src/dst + inject_raw +
log_temp + role_embed) vs the CONTENT path (slot_init, out, struct_norm) vs the encoder/decoder
LoRA. If the topology gradient is ≪ the content/LoRA gradient (and tiny in absolute terms), the
structure heads get no effective training signal → they stay at init → random edges (no-pressure).
"""
from __future__ import annotations
import sys, dataclasses
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

DEV = "cuda"
CKPT = REPO / "outputs/memory/mixed4k_bio_slotgraph_baseline/ckpts/slotgraph_baseline.best.pt"

GROUPS = {
    "TOPO q_src_head": ("encoder.q_src_head",),
    "TOPO q_dst_head": ("encoder.q_dst_head",),
    "TOPO k_head":     ("encoder.k_head",),
    "TOPO log_temp":  ("encoder.log_temp",),
    "MP   msg":       ("encoder.msg",),
    "MP   update":    ("encoder.update",),
    "CONTENT slot_init": ("encoder.slot_init",),
    "CONTENT norm":   ("encoder.norm",),
    "encoder_LoRA":   ("encoder.base", "lora"),
    "decoder_LoRA":   ("decoder.llama", "lora"),
}


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]; valid = {f.name for f in dataclasses.fields(ReprConfig)}
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    m.load_state_dict(sd["model_state_dict"], strict=False)
    m.train(True)
    vs = make_mixed_val_sets(["mae", "babi"], tok, cfg, 4, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    def grad_for(prefixes):
        tot = 0.0
        for n, p in m.named_parameters():
            if p.grad is None:
                continue
            if all(k in n.lower() if k == "lora" else n.startswith(k) for k in prefixes):
                tot += float(p.grad.float().norm()) ** 2
        return tot ** 0.5

    acc = {g: 0.0 for g in GROUPS}; nb = 0
    for t in ("mae", "babi"):
        m.task_mode = MIXED_TASK_MODE[t]
        for b in vs[t]:
            b = to_device(b, DEV)
            m.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = (m.compute_masked_reconstruction_loss(b) if t == "mae"
                       else m.compute_loss(b, window_size=1024))
            out["loss"].backward()
            for g, pref in GROUPS.items():
                acc[g] += grad_for(pref)
            nb += 1

    print(f"\n{'='*60}\nslotgraph gradient flow on the TRAINED model ({nb} batches)\n{'='*60}")
    topo = [g for g in GROUPS if g.startswith("TOPO")]
    cont = [g for g in GROUPS if g.startswith("CONTENT")]
    for g in GROUPS:
        print(f"  {g:20} mean grad = {acc[g]/nb:.3e}")
    topo_sum = sum(acc[g] for g in topo) / nb
    cont_sum = sum(acc[g] for g in cont) / nb
    print(f"\n  TOPOLOGY total = {topo_sum:.3e}   CONTENT total = {cont_sum:.3e}   "
          f"ratio topo/content = {topo_sum/max(cont_sum,1e-12):.4f}")
    print(f"  {'→ topology gets ≪ content gradient (no effective pressure)' if topo_sum < 0.2*cont_sum else '→ topology gets comparable gradient'}")


if __name__ == "__main__":
    main()
