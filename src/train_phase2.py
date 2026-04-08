"""Phase 2 training entry point.

Loads a phase-1 (bootstrap or cycle) checkpoint and a fitted RVQ codebook,
runs the GRPO curriculum over discrete codes, saves the resulting modulator.

Usage:
    python -m src.train_phase2 \
        --checkpoint outputs/v12/bootstrap.pt \
        --codebook outputs/v12/codebook_v1.pt \
        --out outputs/v12/phase2_cycle0.pt \
        --bs 8
"""

import argparse
import os

import torch

from .model.config import Config
from .model.model import Model
from .codebook import ActionVQVAE
from .phase2.trainer import Phase2Trainer, CurriculumStage
from .data import create_dataloader, get_tokenizer, get_special_token_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Phase 1 / bootstrap checkpoint .pt")
    p.add_argument("--codebook", required=True, help="Codebook .pt")
    p.add_argument("--out", required=True, help="Where to save post-phase-2 checkpoint")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", type=str, default="tinyllama")
    p.add_argument("--stage1-tokens", type=int, default=25_000_000)
    p.add_argument("--stage2-tokens", type=int, default=15_000_000)
    p.add_argument("--stage3-tokens", type=int, default=10_000_000)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # Load phase 1 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: Config = ckpt["config"]

    tokenizer = get_tokenizer(args.tokenizer)
    special_ids = get_special_token_ids(tokenizer)
    config.vocab_size = len(tokenizer)
    config.eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)

    model = Model(config).to(device)
    # Load params only — runtime memory state is at phase-1 BS and shape-incompatible
    # with phase-2 BS. We re-initialize memory fresh at phase-2 BS below.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Eval mode disables dropout in scan layers — phase-2 rollouts must be
    # deterministic given the same input + sampled codes.
    model.train(False)
    print(f"Loaded model params (step={ckpt.get('step', '?')})")

    # Fresh memory runtime state at phase-2 BS. Lifelong memory starts from
    # zeros for phase 2 — this is a deliberate discontinuity, not a bug.
    model.memory.initialize_states(args.bs, device)

    # Load codebook
    print(f"Loading codebook: {args.codebook}")
    cb_ckpt = torch.load(args.codebook, map_location=device, weights_only=False)
    cb_config = cb_ckpt["config"]
    vqvae = ActionVQVAE(
        action_dim=cb_config["action_dim"],
        latent_dim=cb_config["latent_dim"],
        hidden=cb_config["hidden"],
        num_levels=cb_config["num_levels"],
        codes_per_level=cb_config["codes_per_level"],
        beta=cb_config["beta"],
    ).to(device)
    vqvae.load_state_dict(cb_ckpt["state_dict"])
    vqvae.train(False)
    print(f"  action_dim={cb_config['action_dim']} latent={cb_config['latent_dim']} "
          f"levels={cb_config['num_levels']}x{cb_config['codes_per_level']}")

    # Stages first so we know the max reward window → segment length.
    stages = [
        CurriculumStage(reward_window=512, token_budget=args.stage1_tokens),
        CurriculumStage(reward_window=2048, token_budget=args.stage2_tokens),
        CurriculumStage(reward_window=4096, token_budget=args.stage3_tokens),
    ]
    max_window = max(s.reward_window for s in stages)
    print(f"Phase 2 segment length T = {max_window} (max curriculum window)")

    dataloader = create_dataloader(
        phase="A", tokenizer=tokenizer, batch_size=args.bs,
        seq_length=max_window, seed=args.seed, max_steps=10**9)

    trainer = Phase2Trainer(
        model=model, vqvae=vqvae, dataloader=dataloader,
        config=config, device=device,
        group_size=args.group_size, lr=args.lr, tau=args.tau,
    )

    trainer.run_curriculum(stages)

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "runtime_state": model.runtime_state_dict(),
        "config": config,
        "step": trainer.global_step,
        "phase": "phase2",
    }, args.out)
    print(f"Saved phase-2 checkpoint to {args.out}")


if __name__ == "__main__":
    main()
