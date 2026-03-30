"""Validate model(s) on held-out validation data.

Usage:
    python -m scripts.eval_val --checkpoint outputs/v9/<run>/v9_step122000.pt
    python -m scripts.eval_val --checkpoint outputs/v9/<run>/v9_step122000.pt --no-memory
    python -m scripts.eval_val --checkpoint none --no-memory --d-inner 2100
"""
import sys
sys.path.insert(0, ".")

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from src.v8.config import V8Config
from src.v8.model import V8Model
from src.data.tokenizer import get_tokenizer, get_special_token_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint, or 'none' for untrained model")
    p.add_argument("--val-shard", type=str, default="data/pile/pile_val.bin")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--d-inner", type=int, default=None)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Max val tokens to use (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer("tinyllama")
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)

    config = V8Config.tier_a()
    config.vocab_size = vocab_size
    config.eot_id = eot_id
    if args.d_inner is not None:
        config.d_inner = args.d_inner
    config.validate()

    T = config.T
    BS = args.bs

    model = V8Model(config).to(device).to(torch.bfloat16)

    if args.checkpoint and args.checkpoint != "none":
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.lm.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "memory_params" in ckpt and not args.no_memory:
            model.memory.load_state_dict(ckpt["memory_params"], strict=False)
        print(f"  Loaded (step {ckpt.get('step', '?')})")
    else:
        print("Using untrained model (sanity check)")

    model.train(False)
    use_memory = not args.no_memory

    val_tokens = np.memmap(args.val_shard, dtype=np.uint16, mode='r')
    n_val = len(val_tokens)
    if args.max_tokens:
        n_val = min(n_val, args.max_tokens)
    print(f"Val shard: {n_val:,} tokens from {args.val_shard}")

    chunk_size = T + 1
    # Distribute tokens across BS streams
    stream_len = n_val // BS
    n_chunks = stream_len // chunk_size
    print(f"Chunks: {n_chunks} (BS={BS}, T={T})")
    print(f"  use_memory={use_memory}, d_inner={config.d_inner}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    model.lm.initialize_carries()
    if use_memory:
        model.memory.to(device)
        model.memory.initialize_states(BS)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            batch_tokens = []
            for b in range(BS):
                start = b * stream_len + chunk_idx * chunk_size
                end = start + chunk_size
                if end > b * stream_len + stream_len:
                    break
                chunk = val_tokens[start:end]
                batch_tokens.append(torch.tensor(chunk.astype(np.int64),
                                                 dtype=torch.long))

            if len(batch_tokens) < BS:
                break

            batch = torch.stack(batch_tokens).to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            result = model.forward_chunk(
                input_ids, target_ids=target_ids,
                use_memory=use_memory)

            logits = result["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                target_ids.reshape(-1))

            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()

            model.lm.detach_carries()
            if use_memory:
                model.memory.detach_states()

            if (chunk_idx + 1) % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  chunk {chunk_idx+1}/{n_chunks}: "
                      f"val_loss={avg:.4f} ppl={math.exp(min(avg, 20)):.1f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULT")
    print(f"  val_loss = {avg_loss:.4f}")
    print(f"  val_ppl  = {ppl:.1f}")
    print(f"  tokens   = {total_tokens:,}")
    print(f"  memory   = {'ON' if use_memory else 'OFF'}")
    print(f"  d_inner  = {config.d_inner}")
    print(f"  params   = {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
