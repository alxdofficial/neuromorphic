"""Wave 1 entry point — long-doc TF NTP pretraining (plan §4.5).

Reads pre-tokenized parquet from preprocess_longdoc.py (and optionally
synthesize_needle.py), packs into D*T_window chunks, runs cross-window
TBPTT with Phase 1 / Wave 1 step.

Usage:
    python scripts/train_wave1.py \\
        --data-paths data/wave1/fineweb_edu.parquet data/wave1/needle.parquet \\
        --batch-size 2 --num-steps 1000 \\
        --checkpoint-out outputs/wave1/ckpt.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.loaders import LongDocDataset
from src.trajectory_memory.training.phase1 import phase1_wave1_step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True,
                    type=Path, help="parquet files from preprocess_longdoc.py")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--lr-memory", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    cfg = getattr(TrajMemConfig, args.config_tier)()
    print(f"Config tier: {args.config_tier}")
    print(f"  N={cfg.N}, J={cfg.J}, K_read={cfg.K_read}, D={cfg.D}, "
          f"T_window={cfg.T_window}")

    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Two param groups: memory side (concept_ids/states/state_init/read+write
    # modules + read_attn) at lr_memory; Llama-side adapter (W_in/W_out/scale)
    # at lr_adapter. All other Llama params are frozen (no group needed).
    memory_params = (
        list(model.manifold.parameters())
        + list(model.read_module.parameters())
        + list(model.write_module.parameters())
        + list(model.read_attn.parameters())
    )
    mem_inject = model.host.layer_list()[cfg.inject_layer] if model.host else None
    adapter_params = []
    if mem_inject is not None:
        # MemInjectLayer's W_in / W_out / scale + cross-attn weights are trainable.
        adapter_params = [
            p for n, p in mem_inject.named_parameters()
            if p.requires_grad and not n.startswith("orig_layer")
        ]
    optimizer = torch.optim.AdamW([
        {"params": memory_params, "lr": args.lr_memory},
        {"params": adapter_params, "lr": args.lr_adapter},
    ])

    dataset = LongDocDataset(
        args.data_paths,
        chunk_tokens=cfg.D * cfg.T_window,
        pad_id=pad_id,
        drop_short=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    # State carries. Reset per training sequence (which is one chunk in
    # this minimal trainer; for longer sequences, set --carry-state and
    # don't reset on each step).
    print(f"Starting Wave 1 training: {args.num_steps} steps")
    step = 0
    losses = []
    t_start = time.time()
    for chunk_batch in loader:
        if step >= args.num_steps:
            break
        chunk = chunk_batch.to(args.device)
        # `chunk` is [BS, D*T_window] — DataLoader stacks individual chunks.
        out = phase1_wave1_step(
            model, chunk, optimizer=optimizer,
            prev_states=None, prev_window_hiddens=None, prev_lm_context=None,
        )
        losses.append(out["loss"])
        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1)
            elapsed = time.time() - t_start
            print(f"  step {step:>5}  loss={out['loss']:.4f}  avg10={avg:.4f}  "
                  f"({elapsed/(step+1):.2f}s/step)")
        step += 1

    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "losses": losses,
        }, args.checkpoint_out)
        print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
