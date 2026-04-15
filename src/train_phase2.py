"""Phase 2 training entry point.

Loads a phase-1 checkpoint (which has a trained DiscreteActionPolicy
inside model.memory), runs the GRPO curriculum over the factored
categorical policy, saves the resulting checkpoint.

Usage:
    python -m src.train_phase2 \
        --checkpoint outputs/v13/bootstrap.pt \
        --out outputs/v13/phase2_cycle0.pt \
        --bs 8
"""

import argparse
import os

import torch

from .model.config import Config
from .model.model import Model
from .phase2.trainer import Phase2Trainer, CurriculumStage
from .data import create_dataloader, get_tokenizer, get_special_token_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Phase 1 / bootstrap checkpoint .pt")
    p.add_argument("--out", required=True, help="Where to save post-phase-2 checkpoint")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=0,
                   help="Fixed seed for the eval dataloader. Independent of "
                        "--seed so replication experiments share the same held-"
                        "out eval set → eval_ce is apples-to-apples across seeds.")
    p.add_argument("--tokenizer", type=str, default="tinyllama")
    p.add_argument("--stage1-tokens", type=int, default=10_000_000,
                   help="Tokens at reward window 512")
    p.add_argument("--stage2-tokens", type=int, default=10_000_000,
                   help="Tokens at reward window 1024")
    p.add_argument("--stage3-tokens", type=int, default=10_000_000,
                   help="Tokens at reward window 2048")
    p.add_argument("--stage4-tokens", type=int, default=10_000_000,
                   help="Tokens at reward window 4096")
    p.add_argument("--eval-interval", type=int, default=50,
                   help="Eval every N GRPO steps (0 = disable)")
    p.add_argument("--eval-batches", type=int, default=8,
                   help="Scored batches (excl. warmup). Should match phase 1.")
    p.add_argument("--eval-warmup-batches", type=int, default=4,
                   help="Warmup batches before scoring (warm memory state)")
    p.add_argument("--eval-bs", type=int, default=None,
                   help="Eval BS; defaults to --bs. Set to match phase 1 for parity.")
    p.add_argument("--warmup-batches", type=int, default=8,
                   help="Forward-only batches to warm the phase-2 memory state "
                        "before GRPO starts (0 = no warmup).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Load phase 1 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: Config = ckpt["config"]
    # Preserve the phase-1 step counter so the next cycle's phase 1 resume
    # continues the LR scheduler and cumulative step tracking correctly.
    phase1_step = ckpt.get("step", 0)

    # Pass-through state for the next cycle's phase 1: phase 2 has its own
    # AdamW for the modulator and doesn't touch the phase-1 optimizer or
    # cosine scheduler. We preserve those + the streaming-dataloader offset
    # so phase 1 resumes its LR schedule, Adam momentum, and stream cursor.
    # Memory runtime state is NOT passed through — we save phase 2's own
    # collapsed end-state below since that reflects what the post-GRPO
    # modulator built.
    phase1_optimizer_state = ckpt.get("optimizer_state_dict")
    phase1_scheduler_state = ckpt.get("scheduler_state_dict")
    phase1_runtime_state = ckpt.get("runtime_state")  # for phase 2 init only
    phase1_dataloader_state = ckpt.get("dataloader_state")

    tokenizer = get_tokenizer(args.tokenizer)
    special_ids = get_special_token_ids(tokenizer)
    config.vocab_size = len(tokenizer)
    config.eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)

    model = Model(config).to(device)
    # Load params only — phase-1 runtime state was at phase-1 BS and is
    # shape-incompatible with phase-2 BS, so we re-initialize at phase-2 BS
    # and then warm up on real data before starting GRPO (see below).
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys in checkpoint "
              f"(will stay at fresh init): {missing[:5]}"
              f"{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys in checkpoint "
              f"(ignored): {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}")
    # Eval mode disables dropout — phase-2 rollouts must be deterministic
    # given the same input + sampled codes.
    model.train(False)
    print(f"Loaded model params (step={ckpt.get('step', '?')})")

    # Initialize at phase-2 BS, then load phase-1's runtime state and
    # resize to phase-2 BS. Phase 1 trained at BS=80 with lane-local
    # memory (no merging), so resize_to_bs samples lanes when shrinking.
    # Each lane's W/decay/hebbian is a valid state produced by the shared
    # modulator on that lane's content stream.
    # Transient state (h, msg, etc.) is reset to zero — those are
    # input-dependent and don't transfer meaningfully across BS changes.
    model.memory.initialize_states(args.bs, device)
    # CRITICAL: sync model._initialized with the memory's initialization state.
    # Without this, the first forward_chunk call (e.g. in warmup) sees
    # `not self._initialized` as True and calls initialize_states again,
    # zeroing out whatever W/decay/hebbian we just loaded below.
    model._initialized = model.memory._initialized
    if phase1_runtime_state is not None:
        try:
            mem_state = phase1_runtime_state.get("memory", {})
            if mem_state.get("initialized", False):
                # Load at whatever BS the phase-1 state was saved at
                model.memory.load_runtime_state(mem_state)
                # Resize to phase 2's BS by tiling/trimming lanes.
                # Each lane's W/decay/hebbian is a valid memory state
                # produced by the shared modulator on that lane's content.
                model.memory.resize_to_bs(args.bs)
                model._initialized = model.memory._initialized
                print(f"  Loaded phase-1 memory state, resized to "
                      f"phase-2 BS={args.bs}")
        except Exception as e:
            print(f"  WARN: could not transfer phase-1 memory state: {e}")
            model.memory.initialize_states(args.bs, device)
            model._initialized = model.memory._initialized

    # (No separate codebook file — codebook + decoder live inside
    # model.memory.discrete_policy, trained during bootstrap + phase 1.)

    # Stages first so we know the max reward window → segment length.
    stages = [
        CurriculumStage(reward_window=512,  token_budget=args.stage1_tokens),
        CurriculumStage(reward_window=1024, token_budget=args.stage2_tokens),
        CurriculumStage(reward_window=2048, token_budget=args.stage3_tokens),
        CurriculumStage(reward_window=4096, token_budget=args.stage4_tokens),
    ]
    max_window = max(s.reward_window for s in stages)
    print(f"Phase 2 segment length T = {max_window} (max curriculum window)")

    # Per-stage dataloader factory: each curriculum stage's rollout
    # sequence length is 2x the reward window. This gives half the
    # modulation events complete (non-truncated) reward windows — the
    # other half are at positions too close to the sequence end and
    # contribute zero via _windowed_reward's completeness mask.
    # Without this ratio, rewards would be heavily biased toward early
    # actions since only they get full windows.
    #
    # Per-stage seed offset: without this, every stage's fresh dataloader
    # starts reading from the same shard position (jitter is seed-derived,
    # so seed=42 always gives the same starting offsets). That hammers the
    # shard prefix across all 4 stages of all cycles. Offsetting by stage
    # index gives each stage a different starting position. Combined with
    # train_loop.py passing a cycle-specific `--seed` to train_phase2, we
    # also de-duplicate across cycles. See audit #3.
    # Per-stage BS scaling: phase 2's dominant tensor is readouts
    # [K*BS, T, D] bf16 which scales with K*BS*T. With K=8 fixed and
    # D=2048, we can increase BS at shorter windows to maximize GPU
    # utilization while keeping peak VRAM ~12-14 GB across all stages.
    # The BS ladder below gives ~3-6x more samples per GRPO step at
    # short windows without changing peak VRAM meaningfully, but yields
    # fewer optimizer updates per stage — acceptable since the modulator
    # only has 4 trainable tensors.
    BS_PER_WINDOW = {512: 24, 1024: 16, 2048: 12, 4096: 8}

    def bs_for_window(reward_window: int) -> int:
        # Find the largest window <= reward_window and use its BS.
        # Falls back to args.bs for windows outside the ladder.
        for w in sorted(BS_PER_WINDOW.keys(), reverse=True):
            if reward_window >= w:
                return BS_PER_WINDOW[w]
        return args.bs

    def train_loader_factory(reward_window: int, stage_idx: int = 0):
        seq_length = 2 * reward_window
        stage_seed = args.seed + stage_idx * 10_000
        stage_bs = bs_for_window(reward_window)
        return create_dataloader(
            phase="A", tokenizer=tokenizer, batch_size=stage_bs,
            seq_length=seq_length, seed=stage_seed, max_steps=10**9)
    # Initial placeholder dataloader (unused; replaced per-stage by factory).
    dataloader = train_loader_factory(max_window, stage_idx=0)

    # Eval on held-out shard. Uses long sequences (= max curriculum window)
    # so we can actually measure whether the phase-2 long-horizon reward
    # shaping is helping at its intended horizon. The lower scan handles
    # arbitrary length (chunks internally into config.T pieces).
    eval_bs = args.eval_bs if args.eval_bs is not None else args.bs
    total_eval_batches = args.eval_batches + args.eval_warmup_batches
    eval_seq_length = max_window
    eval_loader_factory = None
    if args.eval_interval > 0:
        def _make_eval_loader():
            return create_dataloader(
                phase="A-val", tokenizer=tokenizer, batch_size=eval_bs,
                seq_length=eval_seq_length, seed=args.eval_seed,
                max_steps=total_eval_batches)
        eval_loader_factory = _make_eval_loader

    # Metrics alongside the output checkpoint
    out_dir = os.path.dirname(args.out) or "."
    metrics_path = os.path.join(out_dir, "phase2_metrics.jsonl")

    trainer = Phase2Trainer(
        model=model, dataloader=dataloader,
        train_loader_factory=train_loader_factory,
        config=config, device=device,
        group_size=args.group_size, lr=args.lr, tau=args.tau,
        entropy_coeff=args.entropy_coeff,
        metrics_path=metrics_path,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        eval_warmup_batches=args.eval_warmup_batches,
        eval_loader_factory=eval_loader_factory,
    )

    # Warm up the (cold) phase-2 memory state on real data before GRPO.
    # Phase 1 ran at BS=80 and its memory state is incompatible with
    # phase 2's BS=12, so we have to re-initialize. Without warmup, GRPO
    # starts on zero-state memory which doesn't match what the deployed
    # model sees. Run N batches of pure forward (no grad, no codes) so
    # memory accumulates context.
    if args.warmup_batches > 0:
        # Use the first stage's (BS, T) for warmup so memory state is at
        # the right size before stage 1 starts and we don't materialize
        # a [BS, max_window, vocab_size] logits tensor that OOMs at large
        # BS. Each stage runs at T = 2 * reward_window per
        # train_loader_factory, so warmup matches that convention.
        warmup_bs = bs_for_window(stages[0].reward_window)
        warmup_T = 2 * stages[0].reward_window
        print(f"\nWarming up phase-2 memory state "
              f"({args.warmup_batches} forward batches, BS={warmup_bs}, T={warmup_T})...")
        if (model.memory._initialized
                and model.memory.h.shape[0] != warmup_bs):
            model.memory.resize_to_bs(warmup_bs)
            model._initialized = model.memory._initialized
        warmup_loader = create_dataloader(
            phase="A", tokenizer=tokenizer, batch_size=warmup_bs,
            seq_length=warmup_T, seed=args.seed - 1,  # distinct from train
            max_steps=args.warmup_batches)
        model.train(False)
        with torch.no_grad():
            prev_tok = None
            for wi, batch in enumerate(warmup_loader):
                if wi >= args.warmup_batches:
                    break
                input_ids = batch.input_ids.to(device)
                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(device)
                model.forward_chunk(
                    input_ids, use_memory=True,
                    prev_token=batch_prev if batch_prev is not None else prev_tok,
                )
                prev_tok = input_ids[:, -1]
        print(f"  memory warmed up ({args.warmup_batches} batches of "
              f"{warmup_T} tokens)")

    trainer.run_curriculum(stages)

    # Save — phase 1 optimizer/scheduler pass-through (modulator GRPO has its
    # own AdamW that we discard; the next cycle's phase 1 needs the phase-1
    # AdamW & cosine schedule preserved). For runtime state we save phase 2's
    # CURRENT (post-GRPO) memory state as-is. Each lane's W/decay/hebbian
    # reflects the modulator's response to that lane's content — valid state
    # that the next cycle's phase 1 will resize (tile/trim) to its BS.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lane_stats = model.memory.compute_lane_divergence()
    if lane_stats:
        print(f"  Lane divergence at save: "
              f"W_div_rel={lane_stats.get('lane_W_relative_div', 0):.4f} "
              f"heb_div_rel={lane_stats.get('lane_hebbian_relative_div', 0):.4f}")
    phase2_runtime_state = model.runtime_state_dict()
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": phase1_step,
        "phase2_step": trainer.global_step,
        "phase": "phase2",
        "runtime_state": phase2_runtime_state,
    }
    if phase1_optimizer_state is not None:
        save_dict["optimizer_state_dict"] = phase1_optimizer_state
    if phase1_scheduler_state is not None:
        save_dict["scheduler_state_dict"] = phase1_scheduler_state
    if phase1_dataloader_state is not None:
        save_dict["dataloader_state"] = phase1_dataloader_state
    # Atomic write: temp + rename so a mid-save crash doesn't leave a
    # corrupt file at the canonical path for the next cycle to pick up.
    tmp_out = args.out + ".tmp"
    torch.save(save_dict, tmp_out)
    os.replace(tmp_out, args.out)
    print(f"Saved phase-2 checkpoint to {args.out} "
          f"(phase1_step={phase1_step}, phase2_step={trainer.global_step}, "
          f"opt={'Y' if phase1_optimizer_state else 'N'} "
          f"sched={'Y' if phase1_scheduler_state else 'N'} "
          f"runtime=Y(phase2 collapsed BS={args.bs}) "
          f"data={'Y' if phase1_dataloader_state else 'N'})")


if __name__ == "__main__":
    main()
