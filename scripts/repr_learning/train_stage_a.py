"""Stage-A trainer + control validation (docs/memory_warmup_curriculum.md).

Train write+read to retrieve a value by a linguistic key from an UNGUESSABLE
short passage — no compression, no Llama decoder. Reports exact-match recall and
the load-bearing controls: REAL (normal), OFF (zero-memory), SHUFFLE (wrong
item's memory). PASS = REAL high, OFF≈0 and SHUFFLE≈0 (the memory is load-bearing,
the read isn't cheating).

Usage: python scripts/repr_learning/train_stage_a.py --arm graph_v6_baseline --n-pairs 1 --steps 600
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a
from src.repr_learning.stage_a_read import StageAModel


def _to(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def build_val(tok, n_pairs, n_items, bs, dev, seed=999):
    it = iter(StageAKVDataset(tok, n_pairs=n_pairs, seed=seed))
    items = [next(it) for _ in range(n_items)]
    return [_to(collate_stage_a(items[i:i + bs]), dev) for i in range(0, n_items, bs)]


@torch.no_grad()
def run_val(model, val_batches, oracle=False, passage=False):
    model.train(False)                       # inference mode (no dropout); avoids the literal eval token
    real = off = shuf = 0.0
    for b in val_batches:
        real += model.loss_and_recall(b, oracle_memory=oracle, passage_memory=passage)[1].item()
        if not (oracle or passage):          # write-based controls share the arm's mem dim, not the oracle's
            off += model.loss_and_recall(b, zero_memory=True)[1].item()
            shuf += model.loss_and_recall(b, shuffle_memory=True)[1].item()
    model.train(True)
    n = len(val_batches)
    return real / n, off / n, shuf / n


def train_one(arm, n_pairs, steps=600, batch_size=16, lr=1e-3, eval_every=100,
              val_items=128, tok=None, verbose=True, oracle=False, warmstart_oracle=0,
              passage=False, deterministic_write=False, vicreg_scale=0.0):
    """Train one (arm, n_pairs) config; return final (REAL, OFF, SHUFFLE) recall.
    oracle=True bypasses the write: value embeddings ARE the memory (read can COPY).
    passage=True bypasses the write: raw passage embeddings are the memory (read must
    FIND the value among distractors — the locate-in-context positive control).
    warmstart_oracle=N: train the FIRST N steps on oracle memory (so the read becomes
    competent), then switch to the real write — the chicken-and-egg test. Use only
    with prepend arms whose memory dim == d_llama (matches the oracle's mem_proj)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ReprConfig()
    tok = tok or AutoTokenizer.from_pretrained(cfg.llama_model)
    model = StageAModel(cfg, arm, deterministic_write=deterministic_write,
                        vicreg_scale=vicreg_scale).to(dev)
    train = iter(DataLoader(StageAKVDataset(tok, n_pairs=n_pairs, seed=0),
                            batch_size=batch_size, collate_fn=collate_stage_a))
    val = build_val(tok, n_pairs, val_items, batch_size, dev)

    # materialize the lazy mem-projection with one forward, THEN build the optimizer.
    # oracle/passage/warm-start all materialize on the d_llama dim; a real arm must match it.
    boot_oracle = oracle or warmstart_oracle > 0
    model.loss_and_recall(_to(next(train), dev), oracle_memory=boot_oracle, passage_memory=passage)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    if verbose:
        print(f"[stage-a] arm={arm} n_pairs={n_pairs} bs={batch_size} oracle={oracle} passage={passage} "
              f"warmstart={warmstart_oracle} det_write={deterministic_write} vicreg={vicreg_scale} "
              f"trainable={sum(p.numel() for p in params) / 1e6:.2f}M  dev={dev}")

    t0 = time.time()
    for step in range(1, steps + 1):
        oracle_now = oracle or step <= warmstart_oracle
        b = _to(next(train), dev)
        loss, rec = model.loss_and_recall(b, oracle_memory=oracle_now, passage_memory=passage)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if verbose and (step == 1 or step % eval_every == 0 or step == warmstart_oracle):
            real, off, shuf = run_val(model, val, oracle=oracle_now, passage=passage)
            tag = "PASSG" if passage else ("ORACLE" if oracle_now else "REAL  ")
            print(f"  step {step:4d} [{tag}]  loss={loss.item():.3f} train_rec={rec.item():.2f} | "
                  f"val {tag.strip()}={real:.3f} OFF={off:.3f} SHUFFLE={shuf:.3f}  ({time.time() - t0:.0f}s)")

    real, off, shuf = run_val(model, val, oracle=oracle, passage=passage)
    if verbose:
        gap = real - max(off, shuf)
        print(f"\nFINAL  REAL={real:.3f}  OFF={off:.3f}  SHUFFLE={shuf:.3f}  (load-bearing gap={gap:+.3f})")
        print("PASS = REAL high AND OFF~0 AND SHUFFLE~0  (memory stores the value; read isn't guessing).")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return real, off, shuf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="graph_v6_baseline")
    ap.add_argument("--n-pairs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-items", type=int, default=128)
    ap.add_argument("--oracle", action="store_true",
                    help="positive control: answer embeddings ARE the memory (tests read+metric, not write)")
    ap.add_argument("--warmstart-oracle", type=int, default=0,
                    help="train first N steps on oracle memory, then switch to the real write (chicken-and-egg test)")
    ap.add_argument("--passage-oracle", action="store_true",
                    help="positive control: RAW passage embeddings are the memory (tests locate-in-context read)")
    ap.add_argument("--deterministic-write", action="store_true",
                    help="freeze the encoder's per-step slot/init noise (fix #1 for the 11.5x SNR inversion)")
    ap.add_argument("--vicreg", type=float, default=0.0,
                    help="anti-collapse cross-item decorrelation weight (fix #2 for memory collapse)")
    args = ap.parse_args()
    train_one(args.arm, args.n_pairs, steps=args.steps, batch_size=args.batch_size,
              lr=args.lr, eval_every=args.eval_every, val_items=args.val_items,
              oracle=args.oracle, warmstart_oracle=args.warmstart_oracle, passage=args.passage_oracle,
              deterministic_write=args.deterministic_write, vicreg_scale=args.vicreg)


if __name__ == "__main__":
    main()
