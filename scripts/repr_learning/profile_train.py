"""Training-efficiency profiler for the Stage-A encoders (through FROZEN Llama).

Per encoder: step time / peak memory / throughput vs batch size (find max-fitting BS + where throughput
plateaus), plus a coarse component breakdown (encoder write+finalize vs total forward vs backward) — to see
whether BS=8 underutilizes the GPU and where the time actually goes. Run:
  .venv/bin/python scripts/repr_learning/profile_train.py
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer

from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a
from scripts.repr_learning.train_stage_a import stage_a_cfg
from scripts.repr_learning.train_stage_a_qa import to_qabatch

DEV = "cuda"
VARIANTS = ["graph_v6_baseline", "vqvae_baseline", "slot_attention_baseline",
            "memorizing_transformer_baseline", "mamba_baseline"]
D_ENC = {"vqvae_baseline": 1600, "memorizing_transformer_baseline": 1536}
BSS = [8, 16, 32, 64, 96]


def cfg_for(v):
    # graph_v6_prepend_read=False = PRODUCTION inject-at-13 (short backward, ~42ms). prepend was a
    # stopgap that wrongly inflated graph_v6 stage-2 cost ~2.4x. NOTE: the OTHER 4 baselines are
    # prepend-style by design (memory at the INPUT) -> they keep the full 16-layer backward (~100ms).
    return replace(stage_a_cfg("nc8"), graph_v6_d_updater=384, graph_v6_updater_layers=3,
                   graph_v6_read_ffn_mult=1, d_enc=D_ENC.get(v, 1408), d_mamba=1408, graph_v6_prepend_read=False)


def _to(d):
    return {k: (x.to(DEV) if torch.is_tensor(x) else x) for k, x in d.items()}


def make_batches(tok, bs, k=4):
    it = iter(StageAKVDataset(tok, n_pairs=8, seed=1))
    return [_to(collate_stage_a([next(it) for _ in range(bs)])) for _ in range(k)]


def time_avg(fn, n=8, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    tok = AutoTokenizer.from_pretrained(stage_a_cfg("nc8").llama_model)
    for v in VARIANTS:
        print(f"\n#### {v}", flush=True)
        cfg = cfg_for(v)
        model = ReprLearningModel(cfg, variant=v, llama_model=None).to(DEV)
        model.train()

        # ---- BS sweep: step time / peak mem / throughput ----
        best = (0, 0.0)
        for bs in BSS:
            try:
                batches = make_batches(tok, bs)
                torch.cuda.reset_peak_memory_stats()
                i = {"n": 0}

                def step():
                    b = batches[i["n"] % len(batches)]
                    i["n"] += 1
                    model.zero_grad(set_to_none=True)
                    model.compute_qa_loss(to_qabatch(b))["loss"].backward()

                dt = time_avg(step)
                mem = torch.cuda.max_memory_allocated() / 1e9
                thr = bs / dt
                flag = "  <- max throughput" if thr > best[1] else ""
                if thr > best[1]:
                    best = (bs, thr)
                print(f"  BS={bs:3d}  step {dt * 1000:7.1f}ms  peak {mem:5.1f}GB  throughput {thr:6.1f} samp/s{flag}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  BS={bs:3d}  OOM (cap reached)", flush=True)
                torch.cuda.empty_cache()
                break

        # ---- component breakdown at BS=8 (encoder vs total-forward vs backward) ----
        try:
            b = make_batches(tok, 8)[0]
            emb = model.decoder.llama.get_input_embeddings()

            def enc_only():
                ids, mask = b["passage"], b["passage_mask"]
                te = emb(ids).float()
                st = model.encoder.init_streaming_state(ids.size(0), ids.device, te.dtype)
                st, _ = model.encoder.streaming_write(st, te, mask, chunk_offset=0)
                model.encoder.finalize_memory(st)

            def fwd_only():
                model.compute_qa_loss(to_qabatch(b))["loss"]

            holder = {}

            def fwd_bwd():
                model.zero_grad(set_to_none=True)
                loss = model.compute_qa_loss(to_qabatch(b))["loss"]
                holder["loss"] = loss

            t_enc = time_avg(enc_only)
            t_fwd = time_avg(fwd_only)

            def bwd():
                fwd_bwd()
                holder["loss"].backward()

            t_full = time_avg(bwd)
            print(f"  [BS=8 breakdown] encoder-fwd {t_enc*1000:6.1f}ms | total-fwd {t_fwd*1000:6.1f}ms "
                  f"(Llama≈{(t_fwd-t_enc)*1000:6.1f}ms) | fwd+bwd {t_full*1000:6.1f}ms", flush=True)
        except Exception as e:
            print(f"  [breakdown skipped] {repr(e)[:120]}", flush=True)

        print(f"  BEST throughput: BS={best[0]} @ {best[1]:.1f} samp/s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
