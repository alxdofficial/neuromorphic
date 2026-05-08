"""Wave 2 entry point — long-chat TF NTP (plan §4.5).

Reads TurnPair parquet from preprocess_chat.py, length-buckets, runs
Phase 1 / Wave 2 step.

Usage:
    python scripts/train_wave2.py \\
        --data-paths data/wave2/wildchat_long.parquet \\
        --batch-size 2 --num-steps 500
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
from src.trajectory_memory.training.phase1 import phase1_wave2_step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--lr-memory", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="resume from a Wave 1 checkpoint")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    cfg = getattr(TrajMemConfig, args.config_tier)()
    tokenizer = get_tokenizer()

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    if args.checkpoint_in:
        ckpt = torch.load(args.checkpoint_in, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.checkpoint_in} step {ckpt.get('step', '?')}")

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

    dataset = TurnPairDataset(
        args.data_paths,
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
    )
    print(f"Wave 2 dataset: {len(dataset._rows)} TurnPairs, "
          f"{len(dataset)} batches per epoch")

    step = 0
    losses = []
    t_start = time.time()
    while step < args.num_steps:
        for batch in dataset:
            if step >= args.num_steps:
                break
            batch.prior_ids = batch.prior_ids.to(args.device)
            batch.response_ids = batch.response_ids.to(args.device)
            batch.prior_mask = batch.prior_mask.to(args.device)
            batch.response_mask = batch.response_mask.to(args.device)

            out = phase1_wave2_step(model, batch, optimizer=optimizer)
            losses.append(out["loss"])
            if step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1)
                elapsed = time.time() - t_start
                print(f"  step {step:>5}  loss={out['loss']:.4f}  avg10={avg:.4f}  "
                      f"({elapsed/(step+1):.2f}s/step)")
            step += 1

    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "losses": losses,
        }, args.checkpoint_out)
        print(f"Saved {args.checkpoint_out}")


if __name__ == "__main__":
    main()
