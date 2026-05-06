"""Wave 3 GRPO production-shape throughput bench.

Like ``bench_grpo.py`` but with the ACTUAL Wave 3 production stack:
- Real Wave 3 chat-injected loader for prefixes (UltraChat filler +
  passphrase fact + question, chat-templated).
- Real BertCosineReward (sentence-transformers/all-mpnet-base-v2)
  instead of the constant placeholder.
- Real Wave 3 reference text (paraphrased fact answers, variable length).

Used to answer:
- "How fast does Wave 3 GRPO actually run end-to-end?"
- "What's the BERT reward overhead per step?"
- "What K to use?" (K-sweep)
- "What B = BS_outer to use?" (B-sweep at fixed K)

Usage:
  # K-sweep at B=1 (legacy):
  PYTHONPATH=. .venv/bin/python scripts/bench_grpo_wave3.py \\
      --bs-outer-list 1 --k-list 4 8 16 \\
      --t-pre 256 --gen-length 128 --warmup 2 --iter 5

  # B-sweep at K=8 (post-BS_outer):
  PYTHONPATH=. .venv/bin/python scripts/bench_grpo_wave3.py \\
      --bs-outer-list 1 2 4 8 --k-list 8 \\
      --t-pre 256 --gen-length 128 --warmup 2 --iter 5

  # 2-D sweep (B × K):
  PYTHONPATH=. .venv/bin/python scripts/bench_grpo_wave3.py \\
      --bs-outer-list 1 2 4 8 --k-list 4 8 16

Throughput metrics (reported per cell):
- steps/s         — full GRPO steps per second
- session/s       — distinct (prefix, ref) pairs processed per second.
                    Equals B × steps/s. THIS is the wall-clock-relevant
                    metric for "how long does a Wave 3 cycle take."
- gpu-fwd-tok/s   — total tokens forwarded through the LM per second
                    (B × K × (T_pre + gen_length)). Useful for sanity
                    checks against GPU compute ceiling.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data.passphrase_chat_loader import passphrase_chat_grpo_iter
from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rewards import (
    BertCosineReward, load_default_bert,
)
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch, phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import grpo_step


def _walker_cfg_for(d_mem: int, T: int) -> GraphWalkerConfig:
    return GraphWalkerConfig(
        D_s=d_mem, D_model=d_mem,
        vocab_size=128_256,         # Llama-3.2-1B
        segment_T=T, mod_period=T, tbptt_block=T,
        compile_on_train=False,
    )


def _collate_batches(batches):
    """Stack B single-prefix ChatGRPOBatch into a B-prefix
    (prefix_ids[B, T_pre], list[ref tensors]) pair.

    The loader yields ChatGRPOBatch with prefix_ids[1, T_pre] +
    reference_ids[L_b]. For BS_outer = B, we fetch B and concat the
    prefixes (they all share T_pre after pad-to-256), but keep refs
    as a list because ref lengths vary.
    """
    prefixes = torch.cat([b.prefix_ids for b in batches], dim=0)
    refs = [b.reference_ids for b in batches]
    return prefixes, refs


def _bench_one(
    args, B: int, K: int,
) -> tuple[float, float, float, float, float] | None:
    """Returns (steps_per_sec, sessions_per_sec, gpu_fwd_tok_per_sec,
    peak_gb, ms_per_iter), or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        walker_cfg = _walker_cfg_for(d_mem=args.d_mem, T=args.t_pre)
        cfg = PretrainedGWConfig(
            model_name=args.model,
            inject_layer=args.inject_layer,
            d_mem=args.d_mem,
            memory=walker_cfg,
            T=args.t_pre, bs=1,
            llama_dtype="bf16",
            grpo_K=K,
            grpo_rollout_len=args.gen_length,
        )
        wrapper = GraphWalkerPretrainedLM(cfg).cuda()
        wrapper.train(True)
        vocab = wrapper.llama.config.vocab_size

        # Phase-1 prime: build _prev_snapshot_* so first GRPO step has grad.
        prime_opt = torch.optim.AdamW(
            [p for _, p in wrapper.trainable_parameters()],
            lr=1e-7, fused=True,
        )
        prime_in = torch.randint(0, vocab, (1, args.t_pre), device="cuda")
        phase1_pretrained_step(
            wrapper, prime_opt,
            Phase1Batch(input_ids=prime_in, target_ids=prime_in),
            amp_dtype=torch.bfloat16,
        )
        del prime_opt, prime_in
        torch.cuda.empty_cache()

        # Phase-2 freeze
        wrapper.freeze_all_but_E_bias_and_neuromod()
        opt = torch.optim.AdamW(
            [p for _, p in wrapper.trainable_parameters()],
            lr=1e-5, fused=True,
        )
        n_trainable = sum(
            p.numel() for _, p in wrapper.trainable_parameters()
        )

        # BERT reward + Wave 3 loader
        bert = load_default_bert(device="cuda")
        chat_tok = AutoTokenizer.from_pretrained(args.chat_tokenizer)
        reward_fn = BertCosineReward(
            bert_model=bert, tokenizer=chat_tok, device="cuda",
            reference_cache={},  # Wave 3 reuses references; cache them
        )
        data_iter = passphrase_chat_grpo_iter(
            expanded_path=args.passphrase_expanded,
            tokenizer=chat_tok,
            T_pre=args.t_pre,
            L_ref=args.gen_length,
            filler_mid_min=args.filler_min,
            filler_mid_max=args.filler_max,
            n_heldout=20,
            device="cuda",
            ultrachat_bin=args.ultrachat_bin,
            seed=42,
        )

        # Pre-fetch warmup + timed batches: each "outer batch" needs B
        # individual loader yields, collated together.
        outer_batches = []
        n_needed = args.warmup + args.iter
        for _ in range(n_needed):
            singles = [next(data_iter) for _ in range(B)]
            outer_batches.append(_collate_batches(singles))

        def step(prefix_ids, refs):
            return grpo_step(
                wrapper, opt,
                prefix_ids=prefix_ids,
                reference_cont=refs,
                reward_fn=reward_fn,
                num_rollouts=K,
                gen_length=args.gen_length,
            )

        # Warmup
        for i in range(args.warmup):
            step(*outer_batches[i])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed
        t0 = time.perf_counter()
        for i in range(args.warmup, n_needed):
            step(*outer_batches[i])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        n_iter = args.iter
        ms_per_iter = elapsed / n_iter * 1000
        steps_per_sec = n_iter / elapsed
        # Sessions/s = B sessions per step × steps/s
        sessions_per_sec = B * steps_per_sec
        # GPU forward tokens = B*K rollouts each of (T_pre + gen_length).
        gpu_fwd_tok_per_sec = (
            B * K * (args.t_pre + args.gen_length) / (elapsed / n_iter)
        )
        print(
            f"  B={B:>2} K={K:>3}  trainable={n_trainable / 1e6:>5.1f}M  "
            f"{steps_per_sec:>5.2f} steps/s  "
            f"sess {sessions_per_sec:>5.2f}/s  "
            f"gpu {gpu_fwd_tok_per_sec / 1000:>5.1f}k tok/s  "
            f"peak {peak_gb:>5.2f} GB  "
            f"{ms_per_iter:>7.1f} ms/iter",
            flush=True,
        )
        return (steps_per_sec, sessions_per_sec, gpu_fwd_tok_per_sec,
                peak_gb, ms_per_iter)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or \
                "out of memory" in str(e).lower():
            print(f"  B={B:>2} K={K:>3}    OOM", flush=True)
            return None
        raise
    finally:
        try:
            del wrapper, opt, bert, reward_fn, data_iter, outer_batches
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--chat-tokenizer", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--d-mem", type=int, default=256)
    ap.add_argument("--t-pre", type=int, default=256)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--bs-outer-list", type=int, nargs="+", default=[1],
                    help="BS_outer values to sweep (independent prefixes per step).")
    ap.add_argument("--k-list", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iter", type=int, default=5)
    ap.add_argument("--filler-min", type=int, default=100)
    ap.add_argument("--filler-max", type=int, default=200)  # short for production T_pre=256
    ap.add_argument("--passphrase-expanded",
                    default="data/passphrase/expanded.json")
    ap.add_argument("--ultrachat-bin",
                    default="data/phase_B/ultrachat_llama32.bin")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"=== Wave 3 GRPO bench (real chat loader + BERT reward) ===",
          flush=True)
    print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  T_pre={args.t_pre}, gen_length={args.gen_length}", flush=True)
    print(f"  filler_min={args.filler_min}, filler_max={args.filler_max}",
          flush=True)
    print(f"  warmup={args.warmup}, iter={args.iter}", flush=True)
    print(f"  sweeping B={args.bs_outer_list}, K={args.k_list}", flush=True)
    print()

    rows = []
    for B in args.bs_outer_list:
        b_oom = False
        for K in args.k_list:
            r = _bench_one(args, B=B, K=K)
            if r is None:
                # OOM at this (B, K). Larger K at the same B will also
                # OOM, so skip. Try next B (might OOM too — gracefully stop).
                b_oom = True
                break
            rows.append((B, K, *r))
        if b_oom and B == args.bs_outer_list[0]:
            # Even the smallest K at this B OOM'd. Bigger B would too.
            print(f"  Stopping outer sweep at B={B} (smallest K OOM'd)",
                  flush=True)
            break

    print()
    print("=" * 88, flush=True)
    print("  Wave 3 GRPO summary (real BERT reward, real chat data):",
          flush=True)
    print("   B   K  steps/s   sess/s   gpu-tok/s   peak VRAM   ms/iter",
          flush=True)
    print("  " + "-" * 60, flush=True)
    for B, K, sps, sess, gts, gb, ms in rows:
        print(f"  {B:>2}  {K:>3}    {sps:>5.2f}    {sess:>5.2f}    "
              f"{gts / 1000:>5.1f}k     {gb:>5.2f} GB    {ms:>6.1f}",
              flush=True)
    if len(rows) > 1:
        # Session-throughput speedup vs the smallest (B=min, K=min) cell.
        # NOTE: report sess/s speedup, NOT steps/s. steps/s decreases with B
        # (each step does more work), but sess/s — which is what governs
        # wall-clock for a given dataset size — scales near-linearly until
        # memory pressure hits.
        base = rows[0]
        base_sess = base[3]
        print()
        print(f"  session-throughput speedup vs B={base[0]}, K={base[1]}:",
              flush=True)
        for B, K, _sps, sess, *_ in rows:
            sx = sess / base_sess
            print(f"    B={B:>2} K={K:>3}  {sx:>4.2f}× sess/s", flush=True)


if __name__ == "__main__":
    main()
