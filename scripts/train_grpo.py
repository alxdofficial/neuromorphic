"""Phase-2 GRPO trainer for Waves 3 and 4.

Drives ``grpo_session_step`` (DeepSeek-style sample/replay, multi-session
batched) over a stream of ``MultiTurnSession`` objects. Both Wave 3 and
Wave 4 route through the same code path; the only difference is the
loader (``passphrase_chat_grpo_session_iter`` vs
``wildchat_turn_pair_grpo_batch_iter``).

Per outer step:
  Phase 1 (separate trainer): forward [BS, T] -> CE -> backward
                              -> opt.step -> update_plasticity
  Phase 2 (this script):      sample K rollouts -> score (BERT-cosine)
                              -> replay with grad -> REINFORCE backward
                              -> opt.step -> detach_memory

Trainable surface is restricted to ``memory.neuromod.*`` (the production
Phase-2 minimum policy surface). Phase-1 cold-start checkpoint is
required: REINFORCE on a fresh wrapper has no signal.

Wave dispatch:
  --data passphrase-chat-grpo   Wave 3 (chat-injected passphrase, 2-turn
                                sessions, uniform-batched fast path)
  --data wildchat-grpo          Wave 4 (WildChat real chat, turn-batched
                                Verlog-style — flat pool of TurnPairs
                                with sort-and-sample, same uniform-batched
                                fast path as Wave 3)

Usage (Wave 3):
  PYTHONPATH=. .venv/bin/python scripts/train_grpo.py \\
      --data passphrase-chat-grpo \\
      --resume outputs/wave2_ultrachat/ckpt_final.pt \\
      --max-steps 75000 --grpo-K 8 --bs-outer 8 \\
      --T-pre 2048 --gen-length 128 \\
      --filler-min 1000 --filler-max 1800 \\
      --lm-context-window 256 \\
      --work-dir outputs/wave3_passphrase_chat \\
      --lr 3e-5 --warmup 200

Usage (Wave 4):
  PYTHONPATH=. .venv/bin/python scripts/train_grpo.py \\
      --data wildchat-grpo \\
      --resume outputs/wave3_passphrase_chat/ckpt_final.pt \\
      --max-steps 100000 --grpo-K 8 --bs-outer 8 \\
      --turn-pair-pool-size 2048 \\
      --gen-length 128 \\
      --lm-context-window 1024 \\
      --work-dir outputs/wave4_wildchat \\
      --lr 1e-5 --warmup 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rewards import BertCosineReward, load_default_bert
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch, phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import grpo_session_step
from src.graph_walker.telemetry import StatsCollector


def _build_wave3_iter(args, tokenizer, device):
    """Wave 3: passphrase chat-injected GRPO iterator (session format).

    Yields MultiTurnSession with 2 turns (user prefix + assistant ref).
    Same protocol as Wave 4 — both go through grpo_session_step.
    """
    from src.data.passphrase_chat_loader import passphrase_chat_grpo_session_iter
    return passphrase_chat_grpo_session_iter(
        expanded_path=args.passphrase_expanded,
        tokenizer=tokenizer,
        T_pre=args.T_pre,
        L_ref=args.gen_length,
        filler_mid_min=args.filler_min,
        filler_mid_max=args.filler_max,
        n_heldout=args.n_heldout,
        device=device,
        ultrachat_bin=args.ultrachat_bin,
        seed=args.seed,
    )


def _build_wave4_iter(args, tokenizer, device):
    """Wave 4: WildChat-1M turn-batched iterator (Verlog-style).

    Yields list[MultiTurnSession] of size B per next() call. Each
    yielded "session" is a 2-turn wrapper around one TurnPair (a
    cumulative-prior + assistant-reference pair extracted from some
    real WildChat session at some assistant-turn index). Pairs within
    a batch have near-uniform prior length via sort-and-sample.

    Slots directly into grpo_session_step's uniform-batched fast path
    (same as Wave 3) — true B*K parallel rollouts per outer step.
    """
    from src.data.wildchat_loader import wildchat_turn_pair_grpo_batch_iter
    return wildchat_turn_pair_grpo_batch_iter(
        bin_path=args.wildchat_bin,
        turns_path=args.wildchat_turns,
        sessions_path=args.wildchat_sessions,
        batch_size=args.bs_outer,
        pool_size=args.turn_pair_pool_size,
        device=device,
        seed=args.seed,
        min_assistant_turns=args.min_assistant_turns,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        choices=("passphrase-chat-grpo", "wildchat-grpo"),
        required=True,
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--chat-tokenizer", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Tokenizer with chat_template (Wave 3 needs this)")
    ap.add_argument(
        "--resume", required=True,
        help="Phase-1 cold-start checkpoint (REINFORCE has no signal "
             "from fresh init; Wave 2 UltraChat final ckpt is the standard).",
    )
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=75000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42,
                    help="Train/heldout passphrase split seed (Wave 3 only). "
                         "Pin to a value you'll reuse at eval time.")

    # Phase-2 specifics
    ap.add_argument("--T-pre", type=int, default=256)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--grpo-K", type=int, default=8,
                    help="Rollouts per step. Production = 8 (DeepSeek default).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)

    # Optimizer
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Wave-specific data paths
    ap.add_argument("--passphrase-expanded",
                    default="data/passphrase/expanded.json")
    ap.add_argument("--ultrachat-bin",
                    default="data/phase_B/ultrachat_llama32.bin")
    ap.add_argument("--wildchat-bin",
                    default="data/phase_B/wildchat_llama32.bin")
    ap.add_argument("--wildchat-turns",
                    default="data/phase_B/wildchat_llama32_turns.npy",
                    help="v2-schema turn-boundary index (Wave 4 multi-turn).")
    ap.add_argument("--wildchat-sessions",
                    default="data/phase_B/wildchat_llama32_sessions.npy",
                    help="v2-schema session-boundary index (Wave 4 multi-turn).")
    ap.add_argument("--min-assistant-turns", type=int, default=1,
                    help="Skip Wave 4 sessions with fewer than N assistant turns.")
    ap.add_argument("--max-prior-tokens", type=int, default=4096,
                    help="Wave 4 only: cap cumulative prior tokens passed to "
                         "each assistant turn's prefix pass. Limits VRAM in "
                         "long sessions; trades long-context fidelity for "
                         "memory headroom.")
    ap.add_argument("--bs-outer", type=int, default=1,
                    help="B sessions (or turn-pairs) per outer step. For Wave 3, "
                         "B sessions stack via uniform-batched fast path. For "
                         "Wave 4, B turn-pairs from sort-and-sample also stack "
                         "via the same fast path. Both: ~5x sess/s speedup at "
                         "B=8 vs B=1.")
    ap.add_argument("--turn-pair-pool-size", type=int, default=2048,
                    help="Wave 4 only: # turn-pairs maintained in the sort-and-"
                         "sample pool. Larger = better neighbor quality (less "
                         "padding waste from prior-length spread), more memory. "
                         "M=2048 ≈ 65 MB for 4K-avg priors.")
    ap.add_argument("--lm-context-window", type=int, default=None,
                    help="If set, two-phase forward: walker absorbs the full "
                         "prefix but LM only attends to the last N tokens. "
                         "Forces walker to carry information beyond the LM's "
                         "attention reach. None = LM sees everything walker "
                         "sees (default). For Wave 3 with long filler, try "
                         "256-512. For Wave 4 long chats, try 512-1024.")
    ap.add_argument("--filler-min", type=int, default=100)
    ap.add_argument("--filler-max", type=int, default=1500)
    ap.add_argument("--n-heldout", type=int, default=20)
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[setup] device: {device}")

    # ---- wrapper from cold-start checkpoint ----
    # Accept BOTH checkpoint schemas:
    # - train_pretrained_gw.py format: {"wrapper": state_dict,
    #     "config": vars(args), "step": ..., "opt": ..., "sched": ...}
    # - train_grpo.py format (own previous checkpoints): {"model": state_dict,
    #     "config": PretrainedGWConfig, "args": vars(args), ...}
    print(f"[setup] resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    # Resolve state_dict key (`wrapper` from phase-1 trainer, `model` from
    # phase-2 trainer's own checkpoints).
    if "wrapper" in ckpt:
        state = ckpt["wrapper"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        raise KeyError(
            f"checkpoint at {args.resume} has neither 'wrapper' nor 'model' "
            f"key (got: {list(ckpt.keys())})"
        )

    # Resolve config. Phase-1 trainer saves vars(args) (a dict) under
    # 'config'; phase-2 trainer saves a PretrainedGWConfig under 'config'
    # and vars(args) under 'args'. We need a PretrainedGWConfig either
    # way to build the wrapper.
    raw_cfg = ckpt.get("config")
    if isinstance(raw_cfg, PretrainedGWConfig):
        cfg = raw_cfg
        ckpt_args = ckpt.get("args", {})
    elif isinstance(raw_cfg, dict):
        # Phase-1 schema: rebuild PretrainedGWConfig from saved CLI args.
        ckpt_args = raw_cfg
        cfg = PretrainedGWConfig.llama_1b(
            model_name=ckpt_args.get("model", args.model),
            inject_layer=ckpt_args.get("inject_layer", 8),
            d_mem=ckpt_args.get("d_mem", 256),
            T=ckpt_args.get("T", 256),
            bs=ckpt_args.get("bs", 8),
        )
    else:
        print(f"[warn] checkpoint has no recognizable config; "
              f"falling back to PretrainedGWConfig.llama_1b()")
        cfg = PretrainedGWConfig.llama_1b()
        ckpt_args = {}

    wrapper = GraphWalkerPretrainedLM(cfg).to(device)
    # strict=False to tolerate small architectural drift, but log
    # missing/unexpected keys so we don't silently load a wrong
    # checkpoint into a renamed schema.
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    if missing:
        # Filter out tied / non-persistent buffers that legitimately
        # don't appear in saved state_dicts.
        real_missing = [k for k in missing if not (
            k.endswith("_cache") or k.startswith("memory._captured_routes")
        )]
        if real_missing:
            print(f"[warn] checkpoint missing {len(real_missing)} keys "
                  f"(first 5: {real_missing[:5]})")
    if unexpected:
        print(f"[warn] checkpoint had {len(unexpected)} unexpected keys "
              f"(first 5: {unexpected[:5]})")
    print(f"[setup] wrapper loaded; total params="
          f"{sum(p.numel() for p in wrapper.parameters()) / 1e6:.1f}M; "
          f"saved step={ckpt.get('step', '?')}")

    # ---- Phase-1 priming step ----
    # Runtime state buffers (`_prev_snapshot_ids/feats/co_visit_flat`) are
    # not in the model state_dict, so a fresh-loaded wrapper has them all
    # set to None. The first GRPO step would then have `_active_delta_nm
    # = None` and routing scores would have no gradient to neuromod
    # → `log_pi_mean.requires_grad = False` → backward fails.
    #
    # Fix: run ONE phase-1 step BEFORE applying the phase-2 freeze, with
    # the full trainable surface. This populates `_prev_snapshot_*` so
    # the first GRPO step's `_begin_plastic_window` rebuilds a non-None
    # `_active_delta_nm` and routing carries gradient.
    print("[setup] phase-1 priming pass (populates _prev_snapshot_*)")
    prime_opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()],
        lr=1e-7,  # tiny — we don't want to actually update params
        fused=torch.cuda.is_available(),
    )
    prime_vocab = wrapper.llama.config.vocab_size
    prime_in = torch.randint(
        0, prime_vocab, (1, args.T_pre), dtype=torch.long, device=device,
    )
    phase1_pretrained_step(
        wrapper, prime_opt,
        Phase1Batch(input_ids=prime_in, target_ids=prime_in),
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    del prime_opt, prime_in
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Verify priming actually populated _prev_snapshot_* — without these,
    # the next GRPO step's _begin_plastic_window can't rebuild
    # _active_delta_nm and routing has no grad. Fail loud at setup
    # rather than silently producing zero-signal training.
    mem = wrapper.memory
    if mem._prev_snapshot_ids is None or mem._prev_snapshot_feats is None:
        raise RuntimeError(
            "phase-1 priming pass did not populate _prev_snapshot_*. "
            "Plasticity may have been suppressed by the loaded checkpoint "
            "config. Cannot start GRPO without a valid neuromod snapshot."
        )
    if (mem.cfg.plasticity_mode == "neuromod_only"
            and mem._prev_snapshot_co_visit_flat is None):
        raise RuntimeError(
            "neuromod_only mode requires _prev_snapshot_co_visit_flat; "
            "priming did not produce it (plasticity didn't fire?)."
        )
    print(f"[setup] priming done; _prev_snapshot_ids has "
          f"{int(mem._prev_snapshot_ids.numel())} touched cols")

    # ---- Phase-2 minimum policy surface ----
    wrapper.freeze_all_but_E_bias_and_neuromod()
    trainable = [p for _, p in wrapper.trainable_parameters()]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[setup] phase-2 trainable surface: {n_trainable / 1e6:.2f}M params "
          f"(only memory.neuromod.*)")

    # ---- optimizer + LR schedule ----
    opt = torch.optim.AdamW(
        trainable, lr=args.lr,
        fused=torch.cuda.is_available(),
    )
    # Linear warmup + cosine decay to 10% of peak.
    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        progress = min(progress, 1.0)
        # Cosine to 10% of peak
        import math
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    # ---- BERT-cosine reward ----
    print(f"[setup] loading BERT scorer (all-mpnet-base-v2) on {device}")
    bert_model = load_default_bert(device=device)
    base_tokenizer = AutoTokenizer.from_pretrained(args.chat_tokenizer)
    reward_fn = BertCosineReward(
        bert_model=bert_model,
        tokenizer=base_tokenizer,
        device=device,
        reference_cache={} if args.data == "passphrase-chat-grpo" else None,
    )

    # ---- data iterator ----
    if args.data == "passphrase-chat-grpo":
        data_iter = _build_wave3_iter(args, base_tokenizer, device)
    elif args.data == "wildchat-grpo":
        data_iter = _build_wave4_iter(args, base_tokenizer, device)
    else:
        raise ValueError(f"unknown --data: {args.data}")

    # ---- training loop ----
    stats_path = work_dir / "stats.jsonl"
    stats_f = stats_path.open("a")
    print(f"[setup] writing stats to {stats_path}")
    print(f"[setup] starting training, max_steps={args.max_steps}, "
          f"K={args.grpo_K}, T_pre={args.T_pre}, gen_length={args.gen_length}")

    wrapper.train(False)  # walker train mode is set by sample/replay internally
    t0 = time.perf_counter()
    for step in range(args.max_steps):
        # LR schedule
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)

        # Fetch B sessions per outer step.
        # - Wave 3 iter yields one MultiTurnSession at a time → collect B.
        # - Wave 4 (turn-pair) iter yields a pre-batched list[MultiTurnSession]
        #   of size B already → use directly.
        try:
            first = next(data_iter)
        except StopIteration:
            print(f"[train] data iterator exhausted at step {step}", flush=True)
            break

        if isinstance(first, list):
            sessions_batch = first
            if len(sessions_batch) != args.bs_outer:
                raise ValueError(
                    f"Wave 4 turn-pair iter yielded a batch of "
                    f"{len(sessions_batch)} sessions; expected "
                    f"--bs-outer={args.bs_outer}"
                )
        else:
            # Single-yielding iter: collect remaining B-1 sessions. If the
            # iter exhausts mid-collection, drop the partial batch (we
            # already have `first` plus some others, but the uniform-
            # batched fast path requires exactly B sessions; falling back
            # to a smaller batch would break the [B, T_pre] assumption).
            sessions_batch = [first]
            try:
                for _ in range(args.bs_outer - 1):
                    sessions_batch.append(next(data_iter))
            except StopIteration:
                print(
                    f"[train] data iter exhausted mid-collection at step "
                    f"{step} ({len(sessions_batch)}/{args.bs_outer} sessions); "
                    f"dropping partial batch", flush=True,
                )
                break

        # Unified session-format dispatch: both Wave 3 (passphrase chat,
        # 2 turns: user prefix + assistant ref) and Wave 4 (WildChat,
        # variable N turns) route through grpo_session_step.
        if args.data in ("wildchat-grpo", "passphrase-chat-grpo"):
            sstats = grpo_session_step(
                wrapper, opt,
                sessions=sessions_batch,
                reward_fn=reward_fn,
                num_rollouts=args.grpo_K,
                max_response_len=args.gen_length,
                temperature=args.temperature,
                top_p=args.top_p,
                grad_clip=args.grad_clip,
                eos_id=base_tokenizer.eos_token_id,
                max_prior_tokens=args.max_prior_tokens,
                lm_context_window=args.lm_context_window,
            )
            # Aggregate per-turn stats for logging.
            n_turns = sstats.n_assistant_turns
            mean_reward = (
                sum(sstats.per_turn_reward_mean) / n_turns if n_turns else 0.0
            )
            mean_grad = (
                sum(sstats.per_turn_grad_norm) / n_turns if n_turns else 0.0
            )
            mean_loss = (
                sum(sstats.per_turn_loss) / n_turns if n_turns else 0.0
            )
            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                steps_per_sec = (step + 1) / max(elapsed, 1e-9)
                row = {
                    "step": step,
                    "phase": "phase2_mt",
                    "n_assistant_turns": n_turns,
                    "session_tokens": sstats.total_session_tokens,
                    "mean_reward": mean_reward,
                    "mean_grad_norm": mean_grad,
                    "mean_loss": mean_loss,
                    "eos_fraction": sstats.eos_fraction,
                    "lr": opt.param_groups[0]["lr"],
                    "steps_per_sec": steps_per_sec,
                    "time_s": elapsed,
                }
                stats_f.write(json.dumps(row) + "\n")
                stats_f.flush()
                print(
                    f"[train] step={step} turns={n_turns} "
                    f"mean_r={mean_reward:.3f} mean_grad={mean_grad:.3f} "
                    f"eos_frac={sstats.eos_fraction:.2f} "
                    f"steps/s={steps_per_sec:.2f}",
                    flush=True,
                )
        else:
            raise ValueError(f"unsupported --data: {args.data}")

        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = work_dir / f"ckpt_step{step + 1}.pt"
            torch.save({
                "step": step + 1,
                "model": wrapper.state_dict(),
                "opt": opt.state_dict(),
                "config": cfg,
                "args": vars(args),
            }, ckpt_path)
            print(f"[ckpt] wrote {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = work_dir / "ckpt_final.pt"
    torch.save({
        "step": args.max_steps,
        "model": wrapper.state_dict(),
        "opt": opt.state_dict(),
        "config": cfg,
        "args": vars(args),
    }, final_path)
    stats_f.close()
    print(f"[done] wrote {final_path}", flush=True)


if __name__ == "__main__":
    main()
