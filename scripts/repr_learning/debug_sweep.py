"""Dynamic implementation-correctness sweep for all 5 memory encoders + the read path.

NOT training — tiny forward/backward passes at random init, to catch wiring / grad-flow /
shape / collapse / contract bugs BEFORE we build the K/V split. Run: python scripts/repr_learning/debug_sweep.py
"""
import sys, traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a
from scripts.repr_learning.train_stage_a import stage_a_cfg
from scripts.repr_learning.train_stage_a_qa import to_qabatch

DEV = "cuda"
VARIANTS = ["graph_v6_baseline", "vqvae_baseline", "slot_attention_baseline",
            "memorizing_transformer_baseline", "mamba_baseline"]
D_ENC = {"vqvae_baseline": 1600, "memorizing_transformer_baseline": 1536}
results = {}


def cfg_for(v):
    return replace(stage_a_cfg("nc8"),
                   graph_v6_d_updater=384, graph_v6_updater_layers=3, graph_v6_read_ffn_mult=1,
                   d_enc=D_ENC.get(v, 1408), d_mamba=1408)


def rec(v, name, ok, detail=""):
    results.setdefault(v, []).append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}", flush=True)


def soft(v, name, detail):                     # informational, not pass/fail
    print(f"  [info] {name:38s} {detail}", flush=True)


def _to(d):
    return {k: (x.to(DEV) if torch.is_tensor(x) else x) for k, x in d.items()}


def main():
    tok = AutoTokenizer.from_pretrained(stage_a_cfg("nc8").llama_model)
    it = iter(StageAKVDataset(tok, n_pairs=8, seed=1))
    b1 = _to(collate_stage_a([next(it) for _ in range(4)]))
    b2 = _to(collate_stage_a([next(it) for _ in range(4)]))   # distinct passages

    for v in VARIANTS:
        print(f"\n########## {v} ##########", flush=True)
        try:
            cfg = cfg_for(v)
            model = ReprLearningModel(cfg, variant=v, llama_model=None).to(DEV)
            model.train()
            emb = model.decoder.llama.get_input_embeddings()
            enc = model.encoder
            d_llama = cfg.d_llama

            def run_enc(batch):
                ids, mask = batch["passage"], batch["passage_mask"]
                te = emb(ids).float()
                st = enc.init_streaming_state(ids.size(0), ids.device, te.dtype)
                st, _ = enc.streaming_write(st, te, mask, chunk_offset=0)
                return enc.finalize_memory(st)

            mem1, aux1 = run_enc(b1)
            mem2, aux2 = run_enc(b2)
            is_mt = aux1.get("mt_bank") is not None

            # ---- finalize contract ----
            rec(v, "finalize_finite", torch.isfinite(mem1).all().item(), f"mem {tuple(mem1.shape)}")
            if is_mt:
                bk = aux1["mt_bank"]
                rec(v, "mt_bank_nonempty", bk["values"].shape[1] > 0, f"bank {tuple(bk['values'].shape)}")
                rec(v, "mt_bank_finite", torch.isfinite(bk["values"]).all().item())
                pooled1 = aux1["mt_bank"]["values"].mean(1)
                pooled2 = aux2["mt_bank"]["values"].mean(1)
            else:
                rec(v, "finalize_shape", mem1.dim() == 3 and mem1.shape[-1] == d_llama,
                    f"{tuple(mem1.shape)} expect [B,M,{d_llama}]")
                pooled1, pooled2 = mem1.mean(1), mem2.mean(1)

            # ---- input-dependence (collapse-from-init catcher) ----
            if is_mt:
                # pooled-mean is washed out by per-token anisotropy for MT; instead test that the BANK
                # content (the passage) changes what a FIXED query retrieves: retrieve(bank1) vs retrieve(bank2).
                qf = emb(b1["keys"][:, 0]).float()
                m = b1["keys_mask"][:, 0]
                r_b1, _ = enc.retrieve_for_query(aux1["mt_bank"], qf, m, cfg.n_flat_codes)
                r_b2, _ = enc.retrieve_for_query(aux2["mt_bank"], qf, m, cfg.n_flat_codes)
                cos = F.cosine_similarity(r_b1.mean(1), r_b2.mean(1), dim=-1).mean().item()
                rec(v, "input_dependence", cos < 0.999, f"fixed-query retrieve(bank1)vs(bank2) cosine {cos:.4f}")
            else:
                cos = F.cosine_similarity(pooled1, pooled2, dim=-1).mean().item()
                rec(v, "input_dependence", cos < 0.999, f"cross-passage pooled cosine {cos:.4f} (<0.999)")

            # ---- within-row slot distinctness (not collapsed to one vector) ----
            if not is_mt and mem1.shape[1] > 1:
                Mn = F.normalize(mem1[0].float(), dim=-1)
                off = (Mn @ Mn.t())[~torch.eye(Mn.shape[0], dtype=bool, device=Mn.device)]
                soft(v, "slot_distinctness(@init)", f"within-row mean slot cosine {off.mean().item():.3f}")

            # ---- grad flow via the REAL read path ----
            qb = to_qabatch(b1)
            model.zero_grad(set_to_none=True)
            out = model.compute_qa_loss(qb)
            rec(v, "qa_loss_finite", torch.isfinite(out["loss"]).item(), f"loss {float(out['loss']):.3f}")
            out["loss"].backward()
            ep = [(n, p) for n, p in enc.named_parameters() if p.requires_grad]
            dead = [n for n, p in ep if p.grad is None or float(p.grad.abs().sum()) == 0.0]
            nonfin = [n for n, p in ep if p.grad is not None and not torch.isfinite(p.grad).all()]
            rec(v, "no_dead_enc_params", len(dead) == 0, f"{len(dead)}/{len(ep)} dead: {dead[:6]}")
            rec(v, "grad_finite", len(nonfin) == 0, f"{len(nonfin)} nonfinite: {nonfin[:3]}")

            # ---- read-path wiring: memory must reach the decoder, shuffle must change it ----
            with torch.no_grad():
                real = float(model.compute_qa_loss(qb)["loss_recon"])
                offv = float(model.compute_qa_loss(qb, zero_memory=True)["loss_recon"])
                shuf = float(model.compute_qa_loss(qb, shuffle_memory=True)["loss_recon"])
            rec(v, "memory_reaches_decoder", abs(real - offv) > 1e-4, f"REAL {real:.4f} vs OFF {offv:.4f}")
            rec(v, "shuffle_changes_forward", abs(shuf - real) > 1e-4, f"REAL {real:.4f} vs SHUF {shuf:.4f}")

            # ---- MT-specific: retrieve_for_query must depend on the query ----
            if is_mt:
                bk = aux1["mt_bank"]
                q1 = emb(b1["keys"][:, 0]).float()
                q2 = emb(b1["keys"][:, 1]).float()
                r1, _ = enc.retrieve_for_query(bk, q1, b1["keys_mask"][:, 0], cfg.n_flat_codes)
                r2, _ = enc.retrieve_for_query(bk, q2, b1["keys_mask"][:, 1], cfg.n_flat_codes)
                rec(v, "mt_retrieve_query_dependent", (r1 - r2).abs().max().item() > 1e-4,
                    f"max|r(q1)-r(q2)| {(r1 - r2).abs().max().item():.4f}; r shape {tuple(r1.shape)}")
                rec(v, "mt_retrieve_finite", torch.isfinite(r1).all().item())

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            rec(v, "EXCEPTION", False, repr(e)[:240])
            traceback.print_exc()
            torch.cuda.empty_cache()

    print("\n\n================ SUMMARY ================")
    for v in VARIANTS:
        ch = results.get(v, [])
        fails = [c[0] for c in ch if not c[1]]
        print(f"{v:34s} {len(ch) - len(fails)}/{len(ch)} pass" + (f"   FAILS: {fails}" if fails else "   ✓ all pass"))


if __name__ == "__main__":
    main()
