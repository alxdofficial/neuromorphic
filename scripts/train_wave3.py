"""Wave 3 entry point — verifiable-reward GRPO (plan §4.5).

Reads prompt-response parquet from preprocess_grpo.py, samples J
responses per prompt, scores against gold, runs group-relative policy
gradient.

Usage:
    python scripts/train_wave3.py \\
        --data-paths data/wave3/gsm8k.parquet data/wave3/narrativeqa.parquet \\
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
from src.trajectory_memory.training.loaders import PromptResponseDataset
from src.trajectory_memory.training.phase2 import grpo_step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=4,
                    help="J responses per prompt for group-relative advantage")
    ap.add_argument("--max-new-tokens", type=int, default=256)
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
        print(f"Resumed from {args.checkpoint_in}")

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

    dataset = PromptResponseDataset(args.data_paths)
    print(f"Wave 3 dataset: {len(dataset)} prompts")

    step = 0
    rewards_history = []
    t_start = time.time()
    for example in dataset:
        if step >= args.num_steps:
            break
        prompt_ids = torch.tensor(example["prompt_ids"], dtype=torch.int64).to(args.device)
        gold_text = tokenizer.decode(example["gold_ids"], skip_special_tokens=True)

        out = grpo_step(
            model, prompt_ids,
            optimizer=optimizer,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            reward_kind=example["reward_kind"],
            gold=gold_text,
            meta=example["meta"],
            tokenizer=tokenizer,
        )
        rewards_history.append(sum(out["rewards"]) / len(out["rewards"]))
        if step % args.log_every == 0:
            avg = sum(rewards_history[-args.log_every:]) / max(
                len(rewards_history[-args.log_every:]), 1
            )
            elapsed = time.time() - t_start
            print(f"  step {step:>4}  loss={out['policy_loss']:.4f}  "
                  f"mean_r={out['rewards']}  avg_r10={avg:.3f}  "
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
