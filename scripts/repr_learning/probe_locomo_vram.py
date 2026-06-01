"""Peak-VRAM probe for LoCoMo-length (≈24.5k token) eval, all model types.

LoCoMo conversations are 12.8k–24.4k tokens. The streaming/memory variants
compress to an O(1) footprint regardless of length; vanilla_full_context
re-forwards the whole context, so it is the binding VRAM constraint. We run
the (no-grad) QA forward — a conservative upper bound for the KV-cached
AR-decode used at eval — at the max LoCoMo length for each variant × batch.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer  # noqa: E402
from src.repr_learning.config import ReprConfig  # noqa: E402
from src.repr_learning.model import ReprLearningModel  # noqa: E402
from src.repr_learning.data_qa import collate_qa  # noqa: E402
from src.repr_learning.decoder import load_frozen_llama  # noqa: E402

T_CTX = 24576          # max LoCoMo conversation length
DEVICE = "cuda"


def make_cfg():
    return ReprConfig(
        batch_size=1,
        fixed_window_size=1024,
        max_window_size=T_CTX,
        n_flat_codes=192,
        d_continuous=1432, d_concept_baseline=1432,
        d_mt_value=1432, d_recurrent=1432,
        d_enc=816, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3264,
        d_mamba=1256,
        edge_token_packing="fused",
        use_llama_lora=True,
    )


def make_batch(B, cfg, tok):
    g = torch.Generator().manual_seed(0)
    # plausible token ids (avoid specials); short Q/A
    ctx = torch.randint(1000, 120000, (B, T_CTX), generator=g)
    cmask = torch.ones(B, T_CTX, dtype=torch.bool)
    q = tok("What is the special thing mentioned?", add_special_tokens=False,
            return_attention_mask=False)["input_ids"]
    a = tok("the answer", add_special_tokens=False,
            return_attention_mask=False)["input_ids"]
    samples = [{
        "context_ids": ctx[i], "context_mask": cmask[i],
        "question_ids": torch.tensor(q), "answer_ids": torch.tensor(a),
        "answer_content_mask_list": [True] * len(a),
        "task_family": "locomo", "question_type": "probe",
        "answer_refs": ["the answer"],
    } for i in range(B)]
    return collate_qa(samples, pad_token_id=cfg.pad_token_id)


def probe(variant, B, tok):
    cfg = make_cfg()
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(DEVICE)
    model.train(False)
    batch = make_batch(B, cfg, tok)
    for f in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        setattr(batch, f, getattr(batch, f).to(DEVICE))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            _ = model.compute_qa_loss(batch, window_size=1024)
        peak = torch.cuda.max_memory_allocated() / 1e9
        res = f"{peak:5.1f} GB"
    except torch.cuda.OutOfMemoryError:
        res = "OOM"
    del model, llama, batch
    torch.cuda.empty_cache()
    return res


def main():
    print(f"Context length probed: {T_CTX} tokens (max LoCoMo conv)\n")
    cfg = make_cfg()
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    variants = ["vanilla_full_context", "graph_v6_baseline",
                "memorizing_baseline", "flat_baseline"]
    print(f"  {'variant':<24}{'B=1':>10}{'B=2':>10}")
    for v in variants:
        row = {}
        for B in (1, 2):
            row[B] = probe(v, B, tok)
        print(f"  {v:<24}{row[1]:>10}{row[2]:>10}")


if __name__ == "__main__":
    main()
