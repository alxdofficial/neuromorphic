"""slotgraph diagnostic: what's working / what's not.

On the trained slotgraph checkpoints (3 seeds):
  (A) Eval-time ablation: full vs id-only (struct off) vs struct-only (id off) vs
      neither (~icae), measuring babi_em + mae/continuation REAL loss + SHUF-REAL.
      CAVEAT: weights were trained WITH struct+id, so turning a component OFF at
      eval is partly distribution shift, not pure marginal contribution. Read the
      "OFF barely changes" direction as 'decorative'; "OFF tanks" is confounded.
  (B) Topology canaries (unconfounded) per task: mp_delta (does the MP read move
      the output?), node_entropy/max (hub-collapse), src/dst entropy (edge sharpness),
      mem_effrank (rank), hops.

Usage:  .venv/bin/python scripts/diagnostics/slotgraph_ablation_probe.py
"""
import sys, os, glob, math, statistics, dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

DEV = "cuda"
SEED_DIRS = {
    42: "valrun_slotgraph_slotgraph_baseline",
    1:  "valrun_slotgraph_s1_slotgraph_baseline",
    2:  "valrun_slotgraph_s2_slotgraph_baseline",
}
ROOT = "outputs/memory"
CANARIES = ["slotgraph_mp_delta", "slotgraph_node_entropy", "slotgraph_src_entropy",
            "slotgraph_dst_entropy", "slotgraph_mem_effrank", "slotgraph_mp_hops",
            "slotgraph_endpoint_entropy_max"]


def load_model(d):
    cps = sorted(glob.glob(f"{ROOT}/{d}/ckpts/*.pt"))
    sd = torch.load(cps[-1], map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]
    valid = {f.name for f in dataclasses.fields(ReprConfig)}
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    m.load_state_dict(sd["model_state_dict"], strict=False)
    m.eval()
    return m, cfg


def run(m, batches, fam, shuf=False):
    losses, emh, emn, aux = [], 0, 0, {}
    m.task_mode = MIXED_TASK_MODE[fam]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b in batches:
            b = to_device(b, DEV)
            out = m.compute_loss(b, window_size=1024, shuffle_memory=shuf)
            losses.append(out["loss_recon"].item())
            if fam == "babi":
                for f, em in zip(b.task_family, out["per_example_em"].tolist()):
                    if f == "babi":
                        emh += em; emn += 1
            for k in CANARIES:
                if k in out:
                    aux.setdefault(k, []).append(float(out[k]))
    em = emh / emn if emn else None
    return statistics.mean(losses), em, {k: statistics.mean(v) for k, v in aux.items()}


def ms(vals, p=3, scale=1.0, pct=False):
    xs = [v * scale for v in vals if v is not None]
    if not xs:
        return "—"
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    suf = "%" if pct else ""
    return f"{mean:.{p}f}±{std:.{p}f}{suf}"


SETTINGS = [
    (True, True,  "full (struct+id)"),
    (False, True, "id-only (struct OFF)"),
    (True, False, "struct-only (id OFF)"),
    (False, False, "neither (~icae)"),
]


def main():
    seeds = [s for s, d in SEED_DIRS.items() if glob.glob(f"{ROOT}/{d}/ckpts/*.pt")]
    print(f"seeds present: {seeds}")
    abl = {lab: {"babi_em": [], "mae": [], "mae_shuf": [], "cont": [], "cont_shuf": []} for *_, lab in SETTINGS}
    topo = {t: {k: [] for k in CANARIES} for t in ("babi", "mae")}
    tok = vs = None

    for s in seeds:
        m, cfg = load_model(SEED_DIRS[s])
        if tok is None:
            tok = AutoTokenizer.from_pretrained(cfg.llama_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            vs = make_mixed_val_sets(["babi", "mae", "continuation"], tok, cfg, 5, ctx_len=1024,
                                     m_slots=32, mae_src_tok="meta-llama/Llama-3.2-1B",
                                     babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)
        for st, idf, lab in SETTINGS:
            m.encoder.use_structure = st; m.encoder.use_id = idf
            _, babi_em, _ = run(m, vs["babi"], "babi")
            mae_r, _, _ = run(m, vs["mae"], "mae")
            mae_s, _, _ = run(m, vs["mae"], "mae", shuf=True)
            con_r, _, _ = run(m, vs["continuation"], "continuation")
            con_s, _, _ = run(m, vs["continuation"], "continuation", shuf=True)
            abl[lab]["babi_em"].append(babi_em)
            abl[lab]["mae"].append(mae_r); abl[lab]["mae_shuf"].append(mae_s - mae_r)
            abl[lab]["cont"].append(con_r); abl[lab]["cont_shuf"].append(con_s - con_r)
        # topology (full setting) per task
        m.encoder.use_structure = True; m.encoder.use_id = True
        for t in ("babi", "mae"):
            _, _, aux = run(m, vs[t], t)
            for k in CANARIES:
                if k in aux:
                    topo[t][k].append(aux[k])
        del m; torch.cuda.empty_cache()

    print("\n## (A) Eval-time ablation (mean±std over seeds) — babi_em ↑, loss ↓, SHUF−REAL ↑=binds")
    print(f"{'setting':22} {'babi_em':>12} {'mae':>12} {'mae SHUF−R':>12} {'cont':>12} {'cont SHUF−R':>12}")
    for *_, lab in SETTINGS:
        a = abl[lab]
        print(f"{lab:22} {ms(a['babi_em'],0,100,True):>12} {ms(a['mae']):>12} "
              f"{ms(a['mae_shuf']):>12} {ms(a['cont']):>12} {ms(a['cont_shuf']):>12}")

    print("\n## (B) Topology canaries (full model, mean±std over seeds), per task")
    print(f"{'canary':30} {'babi':>16} {'mae':>16}")
    for k in CANARIES:
        print(f"{k.replace('slotgraph_',''):30} {ms(topo['babi'][k]):>16} {ms(topo['mae'][k]):>16}")


if __name__ == "__main__":
    main()
