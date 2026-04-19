#!/usr/bin/env python3
"""Measure peak VRAM for the phase-1 autoregressive unroll.

Runs ONE `run_phase1_ar` step at each (BS, T_prefix, gen_length) cell,
optionally with HF gradient checkpointing on Llama. Reports
`torch.cuda.max_memory_allocated` and a PASS/OOM verdict.

Output: CSV to stdout + jsonl to the path given by --out. Cells that OOM
are reported as `peak_gb=OOM` and the sweep continues on the next cell
after clearing CUDA state.

Decision rule we're testing:
  - If Llama-3B + our target (BS=4, T=512, gen=32) fits within 22 GB
    without checkpointing → skip checkpointing for simplicity.
  - Else enable `model.gradient_checkpointing_enable()` and re-measure.
  - If still OOM, fall back to shorter T or TBPTT inside the unroll.

Usage:
    python scripts/bench_phase1_ar_vram.py --model 3B --bs 1,2,4 \\
        --t-prefix 256,512,1024 --gen-length 16,32 \\
        --checkpointing off,on --out bench/phase1_ar_vram.jsonl
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.pretrained.config import PretrainedConfig
from src.pretrained.llm_wrapper import PretrainedLMWithMemory
from src.pretrained.train_phase1_ar import Phase1ARBatch, run_phase1_ar


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_bool_list(s: str) -> list[bool]:
    return [x.strip().lower() in ("on", "true", "1", "yes") for x in s.split(",")]


_HOST_FACTORY = {
    "llama-1b": PretrainedConfig.llama_1b,
    "llama-3b": PretrainedConfig.llama_3b,
    "tinyllama": PretrainedConfig.tinyllama_1b1,
    "smollm2-360m": PretrainedConfig.smollm2_360m,
    "smollm2-135m": PretrainedConfig.smollm2_135m,
}


def _build_wrapper(host_name: str, checkpointing: bool) -> PretrainedLMWithMemory:
    """Load the HF host + memory; toggle grad checkpointing on the backbone."""
    if host_name not in _HOST_FACTORY:
        known = ", ".join(sorted(_HOST_FACTORY))
        raise ValueError(f"unknown host {host_name!r}; known: {known}")
    cfg = _HOST_FACTORY[host_name]()

    wrapper = PretrainedLMWithMemory(cfg)
    wrapper = wrapper.to("cuda")

    if checkpointing:
        # HF standard: disable KV cache during checkpointing (the two
        # can't coexist in a grad path) and switch to non-reentrant
        # checkpointing, which supports graph re-entry through the mem
        # inject layer without re-running modulator fires.
        wrapper.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return wrapper


def _measure_one_cell(
    host_name: str,
    bs: int,
    t_prefix: int,
    gen_length: int,
    checkpointing: bool,
) -> dict:
    """One training step. Returns dict with peak_gb, status, ms_step."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    result = {
        "host": host_name,
        "bs": bs,
        "t_prefix": t_prefix,
        "gen_length": gen_length,
        "checkpointing": checkpointing,
        "status": "pending",
    }

    try:
        wrapper = _build_wrapper(host_name, checkpointing)
        vocab_size = wrapper.host.vocab_size
        trainable = [p for _, p in wrapper.trainable_parameters()]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

        # Random int64 token streams — shape is what matters, not content.
        gen = torch.Generator(device="cuda").manual_seed(0)
        prefix = torch.randint(0, vocab_size, (bs, t_prefix),
                                device="cuda", generator=gen, dtype=torch.long)
        cont = torch.randint(0, vocab_size, (bs, gen_length),
                              device="cuda", generator=gen, dtype=torch.long)

        def data_iter():
            while True:
                yield Phase1ARBatch(prefix_ids=prefix, continuation_ids=cont)

        torch.cuda.synchronize()
        t0 = time.time()
        run_phase1_ar(
            wrapper, optimizer, data_iter(),
            steps=1, log_interval=1_000_000,  # suppress telemetry logging
        )
        torch.cuda.synchronize()
        ms_step = (time.time() - t0) * 1000

        peak_bytes = torch.cuda.max_memory_allocated()
        result["peak_gb"] = round(peak_bytes / (1024 ** 3), 2)
        result["ms_step"] = round(ms_step, 1)
        result["status"] = "ok"

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        result["peak_gb"] = None
        result["ms_step"] = None
    except Exception as e:  # noqa: BLE001 — we want to continue the sweep
        result["status"] = f"err:{type(e).__name__}"
        result["peak_gb"] = None
        result["ms_step"] = None
        result["error"] = str(e)[:200]

    # Scrub everything before the next cell.
    for name in ("wrapper", "optimizer", "prefix", "cont"):
        if name in locals():
            del locals()[name]
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="llama-1b,llama-3b",
                        help=f"Comma-separated from: {', '.join(sorted(_HOST_FACTORY))}")
    parser.add_argument("--bs", type=str, default="1,2,4")
    parser.add_argument("--t-prefix", type=str, default="256,512,1024")
    parser.add_argument("--gen-length", type=str, default="16,32,64")
    parser.add_argument("--checkpointing", type=str, default="off,on",
                        help="Comma-separated from {off,on}")
    parser.add_argument("--out", type=Path, default=Path("bench/phase1_ar_vram.jsonl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; this benchmark requires a GPU.")

    hosts = [m.strip() for m in args.host.split(",") if m.strip()]
    bss = _parse_int_list(args.bs)
    t_pres = _parse_int_list(args.t_prefix)
    gen_lens = _parse_int_list(args.gen_length)
    ckpts = _parse_bool_list(args.checkpointing)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells = list(itertools.product(hosts, bss, t_pres, gen_lens, ckpts))
    print(f"Sweeping {len(cells)} cells. GPU: {torch.cuda.get_device_name(0)}, "
          f"total VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB\n")
    print(f"{'host':>14}  {'bs':>3}  {'T_pre':>6}  {'gen':>4}  {'ckpt':>5}  "
          f"{'peak_gb':>8}  {'ms/step':>8}  status")
    print("-" * 77)

    with args.out.open("w") as f:
        for (host_name, bs, t_pre, gen_len, ckpt) in cells:
            r = _measure_one_cell(host_name, bs, t_pre, gen_len, ckpt)
            f.write(json.dumps(r) + "\n")
            f.flush()
            peak = f"{r['peak_gb']:.2f}" if r["peak_gb"] is not None else "---"
            ms = f"{r['ms_step']:.1f}" if r["ms_step"] is not None else "---"
            ckpt_s = "on" if ckpt else "off"
            print(f"{host_name:>14}  {bs:>3}  {t_pre:>6}  {gen_len:>4}  {ckpt_s:>5}  "
                  f"{peak:>8}  {ms:>8}  {r['status']}")

    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
