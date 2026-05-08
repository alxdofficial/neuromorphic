"""Wave 4 entry point — long-session GRPO with chat data (plan §4.5).

Reads TurnPair parquet from preprocess_chat.py (same format as Wave 2),
samples J responses per TurnPair via the model, scores against the
ground-truth response (exact match + BERT cosine, per project policy),
runs group-relative policy gradient.

Differs from Wave 3 in that the data is multi-turn chat TurnPairs
(prior includes prior conversation context) and the reward function is
exact-match-or-BERT-cosine on the response. The architectural mechanics
are otherwise the same as Wave 3.

Usage:
    python scripts/train_wave4.py \\
        --data-paths data/wave4/wildchat_long.parquet \\
        --num-samples 4 --num-steps 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.loaders import TurnPairDataset
from src.trajectory_memory.training.phase2 import grpo_step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--lr-memory", type=float, default=1e-4)
    ap.add_argument("--lr-adapter", type=float, default=5e-5)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None)
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    cfg = getattr(TrajMemConfig, args.config_tier)()
    tokenizer = get_tokenizer()

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    if args.checkpoint_in:
        ckpt = torch.load(args.checkpoint_in, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])

    memory_params = (
        list(model.manifold.parameters())
        + list(model.read_module.parameters())
        + list(model.write_module.parameters())
        + list(model.read_attn.parameters())
    )
    mem_inject = model.host.layer_list()[cfg.inject_layer]
    adapter_params = [
        p for n, p in mem_inject.named_parameters()
        if p.requires_grad and not n.startswith("orig_layer")
    ]
    optimizer = torch.optim.AdamW([
        {"params": memory_params, "lr": args.lr_memory},
        {"params": adapter_params, "lr": args.lr_adapter},
    ])

    # Use TurnPairDataset (same as Wave 2) but treat each pair as a Wave 3-style
    # GRPO example: prompt = prior_ids; gold = response_ids decoded.
    dataset = TurnPairDataset(
        args.data_paths, batch_size=1, pad_id=tokenizer.pad_token_id,
    )
    print(f"Wave 4 dataset: {len(dataset._rows)} TurnPairs")

    step = 0
    rewards_history = []
    t_start = time.time()
    for batch in dataset:
        if step >= args.num_steps:
            break
        # batch is BS=1 TurnPair; unwrap.
        prompt_ids = batch.prior_ids[0]
        if (mask := batch.prior_mask[0]).all():
            prompt_ids = prompt_ids
        else:
            # Strip padding.
            prompt_ids = prompt_ids[mask]
        gold_resp = batch.response_ids[0][batch.response_mask[0]]
        gold_text = tokenizer.decode(gold_resp.tolist(), skip_special_tokens=True)
        prompt_ids = prompt_ids.to(args.device)

        out = grpo_step(
            model, prompt_ids,
            optimizer=optimizer,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            reward_kind="exact_match_or_bert_cosine",
            gold=gold_text,
            meta={"all_answers": [gold_text]},
            tokenizer=tokenizer,
        )
        rewards_history.append(sum(out["rewards"]) / len(out["rewards"]))
        if step % args.log_every == 0:
            avg = sum(rewards_history[-args.log_every:]) / max(
                len(rewards_history[-args.log_every:]), 1
            )
            elapsed = time.time() - t_start
            print(f"  step {step:>4}  loss={out['policy_loss']:.4f}  "
                  f"r={out['rewards']}  avg_r10={avg:.3f}  "
                  f"({elapsed/(step+1):.2f}s/step)")
        step += 1

    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "rewards_history": rewards_history,
        }, args.checkpoint_out)
        print(f"Saved {args.checkpoint_out}")


if __name__ == "__main__":
    main()
