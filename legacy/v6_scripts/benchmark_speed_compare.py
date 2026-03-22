"""Benchmark training throughput: neuromorphic LM vs Mamba baseline.

Runs both models on random token batches to isolate model/optimizer speed
from dataloader/tokenizer overhead. Intended for quick apples-to-apples
throughput checks on a single GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
from transformers import MambaConfig, MambaForCausalLM

from src.data.streaming import StreamBatch
from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.training.trainer import TBPTTTrainer


@dataclass
class BenchResult:
    name: str
    avg_step_s: float
    tok_per_s: float
    warmup_steps: int
    timed_steps: int
    batch_size: int
    seq_len: int


def _rand_batch(bs: int, t: int, vocab: int, device: torch.device) -> StreamBatch:
    x = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
    y = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
    prev = torch.zeros(bs, dtype=torch.long, device=device)
    return StreamBatch(input_ids=x, target_ids=y, prev_token=prev)


def bench_neuromorphic(
    bs: int,
    t: int,
    vocab: int,
    warmup: int,
    steps: int,
    device: torch.device,
    use_compile: bool,
) -> BenchResult:
    cfg = ModelConfig.tier_a(N=t)
    cfg.set_phase("B")
    cfg.vocab_size = vocab
    cfg.eot_id = 2
    cfg.use_compile = use_compile

    model = NeuromorphicLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    trainer = TBPTTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=iter(()),
        config=cfg,
        device=device,
        collector=None,
        log_interval=10_000,
    )

    for _ in range(warmup):
        trainer.train_chunk(_rand_batch(bs, t, vocab, device))
        trainer.global_step += 1

    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        batch = _rand_batch(bs, t, vocab, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer.train_chunk(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        trainer.global_step += 1
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    tok_per_step = bs * t
    return BenchResult(
        name="neuromorphic_tier_a_phase_b",
        avg_step_s=avg,
        tok_per_s=tok_per_step / avg,
        warmup_steps=warmup,
        timed_steps=steps,
        batch_size=bs,
        seq_len=t,
    )


def bench_mamba(
    bs: int,
    t: int,
    vocab: int,
    warmup: int,
    steps: int,
    device: torch.device,
) -> BenchResult:
    cfg = MambaConfig(
        vocab_size=vocab,
        hidden_size=768,
        num_hidden_layers=24,
        state_size=16,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
    )
    model = MambaForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)

    for _ in range(warmup):
        x = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
        y = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        x = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
        y = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    tok_per_step = bs * t
    return BenchResult(
        name="mamba_130m_cfg",
        avg_step_s=avg,
        tok_per_s=tok_per_step / avg,
        warmup_steps=warmup,
        timed_steps=steps,
        batch_size=bs,
        seq_len=t,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    neu = bench_neuromorphic(
        bs=args.bs,
        t=args.seq_len,
        vocab=args.vocab,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
        use_compile=not args.no_compile,
    )
    mamba = bench_mamba(
        bs=args.bs,
        t=args.seq_len,
        vocab=args.vocab,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
    )

    result = {
        "env": {
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "neuromorphic": asdict(neu),
        "mamba": asdict(mamba),
        "ratio_mamba_over_neuromorphic": mamba.tok_per_s / max(neu.tok_per_s, 1e-9),
    }

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
