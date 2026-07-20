#!/usr/bin/env python3
"""No-download H2O smoke test using a tiny randomly initialized GQA Llama."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--head-mode", choices=["query_head", "kv_head"], default="query_head")
    args = ap.parse_args()

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from src.memory.eval.h2o_llama import H2OLlamaEngine

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    engine = H2OLlamaEngine(
        model,
        heavy_size=16,
        recent_size=16,
        prefill_chunk_size=8,
        head_mode=args.head_mode,
    )
    result = engine.generate(
        torch.arange(768).remainder(config.vocab_size).view(1, -1),
        max_new_tokens=4,
        eos_token_ids=[],
    )
    assert result.token_ids.shape == (1, 4)
    assert result.diagnostics["max_unpruned_length"] <= 40
    assert result.diagnostics["final_max_layer_length"] <= 32
    assert result.diagnostics["position_mode"] == "rolling"
    print(json.dumps({
        "status": "ok",
        "device": device,
        "dtype": str(dtype),
        "generated": result.token_ids[0].tolist(),
        **result.diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
