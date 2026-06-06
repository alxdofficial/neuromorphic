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


# ── Stage-A MATCHED size presets ──────────────────────────────────────────────
# Anchored on the graph; every arm shares one bottleneck B = K_edge × d_read.
# Invariants (locked):
#   • d_read = 2·d_node + d_state → each fact token carries its FULL edge, so the read
#     interface EQUALS the per-item persistent edge state ("persistent == interface").
#   • node bank K_node × d_node is a SHARED learned basis (like the flat codebook),
#     reported separately, NOT part of the matched per-item B. K_node tracks ENTITIES.
#   • baselines: n_flat_codes=K_edge memory tokens at per-slot dim = d_read → identical B.
#   • graph write width d_updater = d_read = baseline encoder width d_enc → no arm is
#     bottlenecked at the write stage (fair + honestly no-compression).
#
# nc8 = the LOCKED working config (2026-06-03): NO-COMPRESSION for 8 facts.
#   B = 108 × 2048 = 221,184 = (median 8-fact passage = 108 tok) × d_llama(2048). The graph
#   holds the passage token-granularly (108 edges × 2048 ≈ one token/edge), matching the
#   baselines token-for-token (108 slots × 2048). K_node=48 tracks entities (~3 owners + value
#   nodes), decoupled from the token-granular edge count. tiny..ultra = the fact-granular ladder.
#   (median re-pinned 90→108 after the 2026-06-03 data enrichment grew passages.)
#            (K_edge, d_node, d_state, K_node)  → d_read=2dn+ds, B=K_edge·d_read
STAGE_A_PRESETS = {
    "nc8":    (108, 512, 1024, 48),    # B=221,184  ← LOCKED: no-compression, 8 facts, = baselines 108×2048
    "tiny":   (8,   256, 256,  16),    # B=6,144    (legacy fact-granular, fast iteration)
    "small":  (32,  128, 256,  64),    # B=16,384
    "medium": (64,  192, 384,  128),   # B=49,152
    "large":  (128, 256, 512,  256),   # B=131,072
    "xl":     (196, 384, 384,  392),   # B=225,792
    "ultra":  (384, 512, 512,  768),   # B=589,824
}


def stage_a_cfg(preset: str = "nc8") -> ReprConfig:
    """Build a MATCHED Stage-A config from the preset table (see STAGE_A_PRESETS).

    Anchored on the graph: graph fact tokens and baseline memory tokens both equal
    B = K_edge × d_read floats handed to the unified read. d_updater is raised to d_read
    so the graph's write isn't a narrower bottleneck than the baselines' encoder (d_enc).
    """
    if preset not in STAGE_A_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(STAGE_A_PRESETS)}")
    k_edge, d_node, d_state, k_node = STAGE_A_PRESETS[preset]
    d_read = 2 * d_node + d_state            # fact token == full edge → no compression
    return ReprConfig(
        graph_v6_K_node=k_node, graph_v6_K_edge=k_edge,
        graph_v6_d_node=d_node, graph_v6_d_state=d_state, graph_v6_d_read=d_read,
        graph_v6_d_updater=d_read,           # write width = edge width = baseline encoder width
        n_flat_codes=k_edge, d_enc=d_read,
        d_continuous=d_read, d_concept_baseline=d_read, d_mt_value=d_read, d_recurrent=d_read,
    )


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


@torch.no_grad()
def mem_rank(model, batch):
    """Fact-token effective rank: within-item (are the M slots diverse?) + cross-item
    (do different passages give different memory? → the collapse monitor)."""
    model.train(False)
    mem = model._write_memory(batch["passage"], batch["passage_mask"])   # [B,M,d]
    model.train(True)
    B, M = mem.shape[0], mem.shape[1]
    def er(X):
        s = torch.linalg.svdvals(X.float()); s = s[s > 1e-9]; p = s / s.sum()
        return float(torch.exp(-(p * p.log()).sum()))
    within = sum(er(mem[i]) for i in range(min(B, 8))) / min(B, 8)
    return within, er(mem.reshape(B, -1)), M


def train_one(arm, n_pairs, steps=600, batch_size=16, lr=1e-3, eval_every=100,
              val_items=128, tok=None, verbose=True, oracle=False, warmstart_oracle=0,
              passage=False, deterministic_write=False, recon_weight=0.0, preset="nc8"):
    """Train one (arm, n_pairs) config; return final (REAL, OFF, SHUFFLE) recall.
    oracle=True bypasses the write: value embeddings ARE the memory (read can COPY).
    passage=True bypasses the write: raw passage embeddings are the memory (read must
    FIND the value among distractors — the locate-in-context positive control).
    warmstart_oracle=N: train the FIRST N steps on oracle memory (so the read becomes
    competent), then switch to the real write — the chicken-and-egg test. Use only
    with prepend arms whose memory dim == d_llama (matches the oracle's mem_proj)."""
    # warm-start materializes the read's LazyLinear mem_proj on oracle memory (d_llama=2048);
    # graph fact-tokens are d_read=512, so a graph warm-start would crash at the switch.
    assert not (warmstart_oracle > 0 and arm == "graph_v6_baseline"), (
        "warmstart-oracle is incompatible with graph_v6 (mem_proj fixes at d_llama=2048, "
        "real graph memory is d_read=512); use a prepend arm or drop --warmstart-oracle")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = stage_a_cfg(preset)               # MATCHED config; n_pairs=K_edge is at-capacity
    B = cfg.graph_v6_K_edge * cfg.graph_v6_d_read
    tok = tok or AutoTokenizer.from_pretrained(cfg.llama_model)
    model = StageAModel(cfg, arm, deterministic_write=deterministic_write,
                        recon_weight=recon_weight).to(dev)
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
        print(f"[stage-a] arm={arm} preset={preset} (K_edge={cfg.graph_v6_K_edge} d_read={cfg.graph_v6_d_read} "
              f"B={B:,}) n_pairs={n_pairs} bs={batch_size} oracle={oracle} passage={passage} "
              f"warmstart={warmstart_oracle} det_write={deterministic_write} "
              f"recon={recon_weight} trainable={sum(p.numel() for p in params) / 1e6:.2f}M  dev={dev}")

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
            rk = ""
            if not (oracle_now or passage):   # fact-token rank only meaningful for the real write
                wi, cr, M = mem_rank(model, val[0])
                rk = f"  rank[within {wi:.1f}/{M} cross {cr:.1f}]"
            print(f"  step {step:4d} [{tag}]  loss={loss.item():.3f} train_rec={rec.item():.2f} | "
                  f"val {tag.strip()}={real:.3f} OFF={off:.3f} SHUFFLE={shuf:.3f}{rk}  ({time.time() - t0:.0f}s)")

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
    ap.add_argument("--recon-weight", type=float, default=0.0,
                    help="passage-reconstruction objective weight (force the write to encode its input)")
    ap.add_argument("--preset", default="nc8", choices=list(STAGE_A_PRESETS),
                    help="MATCHED size preset. nc8=LOCKED no-compression for 8 facts "
                         "(B=221,184=108x2048). tiny..ultra = legacy fact-granular ladder.")
    args = ap.parse_args()
    train_one(args.arm, args.n_pairs, steps=args.steps, batch_size=args.batch_size,
              lr=args.lr, eval_every=args.eval_every, val_items=args.val_items,
              oracle=args.oracle, warmstart_oracle=args.warmstart_oracle, passage=args.passage_oracle,
              deterministic_write=args.deterministic_write, recon_weight=args.recon_weight,
              preset=args.preset)


if __name__ == "__main__":
    main()
