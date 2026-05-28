"""Decode v5.4 memory tokens to text.

Two modes, both useful but answering different questions:

``--mode lmhead``: project each memory token against Llama's lm_head
(tied to embed_tokens) to find the top-K vocab tokens it's closest to.
Tells us whether memory tokens live in vocab-embedding geometry. v5.4
result: they don't — output is gibberish subwords across all chunks,
because the encoder learned to put info in directions Llama's *attention
layers* extract, not in directions that lm_head sees.

``--mode generative``: prepend memory tokens as input embeddings to Llama
(matching how the encoder is actually USED in compute_qa_loss for prepend
variants), give Llama a tiny starter prompt, and let it generate N tokens
freely. This is the architecturally correct "what content does memory
carry" probe: it runs the full Llama stack with memory as the input, so
attention can do its job. Output is constrained to the model's training
distribution (QA), so completions look answer-shaped rather than summary-
shaped — but topical leakage (right entities, right domain) is the signal.

Run:
    python scripts/repr_learning/decode_memory_v5.py --mode lmhead --chunks 4 --top 8
    python scripts/repr_learning/decode_memory_v5.py --mode generative --chunks 4 --max-new-tokens 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/alex/code/neuromorphic")
sys.path.insert(0, str(ROOT))

from src.repr_learning.config import ReprConfig                        # noqa: E402
from src.repr_learning.encoder import GraphV5BaselineEncoder           # noqa: E402
from src.repr_learning.data_qa import (                                # noqa: E402
    HotpotQADataset, MixedQADataset, QADataset, collate_qa,
)

COMPOSITE_VAL_P = ROOT / "data/wave1/composite_v1/val/passages.jsonl"
COMPOSITE_VAL_Q = ROOT / "data/wave1/composite_v1/val/questions.jsonl"

CKPT = ROOT / "outputs/repr_learning/v5_4_first_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt"
ALLOWED_UNEXPECTED = {"soft_pointer.W_v.weight"}


def load_encoder(ckpt_path: Path):
    cfg = ReprConfig()
    enc = GraphV5BaselineEncoder(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                 if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    bad = [k for k in unexpected if k not in ALLOWED_UNEXPECTED]
    if missing or bad:
        raise RuntimeError(f"ckpt drift: missing={missing[:5]} bad_unexpected={bad[:5]}")
    enc.train(False)
    print(f"[ckpt] step={sd.get('step', -1)} loaded clean")
    return enc, cfg


def run_encoder(enc, cfg, llama_embed, batch, device):
    """Stream the chunk and return the K_node=32 memory tokens."""
    enc = enc.to(device)
    input_ids = batch.context_ids.to(device)
    attention_mask = batch.context_mask.to(device)
    with torch.no_grad():
        token_embeds = llama_embed(input_ids)
        T, W = token_embeds.shape[1], 1024
        state = enc.init_streaming_state(token_embeds.shape[0],
                                          device=device, dtype=token_embeds.dtype)
        for s in range(0, T, W):
            e = min(s + W, T)
            state, _ = enc.streaming_write(
                state, token_embeds[:, s:e, :],
                attention_mask=attention_mask[:, s:e],
                chunk_offset=s,
            )
        N = state["N"]
        _, attn_src = enc.soft_pointer(state["q_src"], N)
        _, attn_dst = enc.soft_pointer(state["q_dst"], N)
        memory, _ = enc.readout(N, attn_src, attn_dst, state["state"])
    return memory  # [B, K_node, d_llama]


def decode_token(memory: torch.Tensor, lm_head_w: torch.Tensor,
                 tokenizer, top_k: int) -> list[list[str]]:
    """For each memory token, return the top-K vocab tokens it would
    project to under Llama's LM head. lm_head_w: [V, d_llama]."""
    B, K, D = memory.shape
    flat = memory.reshape(B * K, D)
    # Llama's pre-LM-head RMSNorm scales hidden states; without it the
    # raw dot product is dominated by magnitude. Apply RMS.
    rms = flat.norm(dim=-1, keepdim=True) / (D ** 0.5)
    flat = flat / rms.clamp_min(1e-6)
    logits = flat @ lm_head_w.T                                       # [BK, V]
    topk_ids = logits.topk(top_k, dim=-1).indices                      # [BK, top_k]
    out = []
    for i in range(B * K):
        toks = [tokenizer.decode([int(t)]).strip() or f"\\u{int(t):x}"
                for t in topk_ids[i].tolist()]
        out.append(toks)
    return out  # length B*K_node, each a list of top_k token strings


def generate_from_memory(llama, tokenizer, memory: torch.Tensor,
                          starter_prompts: list[str], max_new_tokens: int,
                          device) -> list[str]:
    """Feed [memory; starter_prompt_embeds] to Llama and greedy-generate.

    memory: [B, K_node, d_llama]. Starter prompts are short text strings,
    one per example, that bias the autoregressive head toward a useful
    generation register (raw BOS often produces nothing structured).
    Returns one decoded string per example (the generated continuation only,
    not the starter prompt).
    """
    B, K, D = memory.shape
    assert len(starter_prompts) == B
    embed_layer = llama.get_input_embeddings()
    pieces = []                                                       # one [1, M+T, D]
    starter_lens = []
    for i in range(B):
        st_ids = tokenizer(starter_prompts[i], add_special_tokens=False,
                           return_tensors="pt")["input_ids"].to(device)
        st_emb = embed_layer(st_ids)                                  # [1, T, D]
        full = torch.cat([memory[i:i+1].to(st_emb.dtype), st_emb], dim=1)
        pieces.append(full)
        starter_lens.append(int(st_ids.shape[1]))
    # Pad to a common length so we can batch (left-pad to keep generation
    # aligned at the right end where the next token is appended).
    max_len = max(p.shape[1] for p in pieces)
    inputs_embeds = torch.zeros(B, max_len, D, dtype=pieces[0].dtype, device=device)
    attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, p in enumerate(pieces):
        n = p.shape[1]
        inputs_embeds[i, max_len - n:] = p[0]
        attn[i, max_len - n:] = 1

    with torch.no_grad():
        gen_out = llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # When inputs_embeds is used, HuggingFace generate returns ONLY the
    # newly-generated token ids (not the input prefix), unlike when input_ids
    # is used. So gen_out[i] is exactly the continuation.
    return [tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            for seq in gen_out]


def generate_qa_answers(llama, tokenizer, memory: torch.Tensor,
                         question_ids_list: list[list[int]], max_new_tokens: int,
                         device) -> list[str]:
    """Feed [memory; question_embeds] to Llama and greedy-generate the
    answer autoregressively. This mirrors the training input format
    (compute_qa_loss prepends memory then puts the question, then computes
    teacher-forced CE on the answer). At inference, we generate the answer
    instead — the honest test of "can the model actually produce the
    answer it's being scored on."
    """
    B, K, D = memory.shape
    assert len(question_ids_list) == B
    embed_layer = llama.get_input_embeddings()
    pieces = []
    for i in range(B):
        q_tensor = torch.tensor(question_ids_list[i], dtype=torch.long,
                                 device=device).unsqueeze(0)         # [1, T_q]
        q_emb = embed_layer(q_tensor)                                  # [1, T_q, D]
        full = torch.cat([memory[i:i+1].to(q_emb.dtype), q_emb], dim=1)
        pieces.append(full)
    max_len = max(p.shape[1] for p in pieces)
    inputs_embeds = torch.zeros(B, max_len, D, dtype=pieces[0].dtype, device=device)
    attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, p in enumerate(pieces):
        n = p.shape[1]
        inputs_embeds[i, max_len - n:] = p[0]                          # left-pad
        attn[i, max_len - n:] = 1
    with torch.no_grad():
        gen = llama.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in gen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lmhead", "generative", "qa"], default="lmhead")
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--top", type=int, default=8, help="(lmhead) top-K vocab tokens per memory slot")
    ap.add_argument("--max-new-tokens", type=int, default=80,
                    help="(generative/qa) tokens to generate after the prompt")
    ap.add_argument("--source", choices=["hotpot", "mixed"], default="hotpot",
                    help="(qa) data source. mixed = composite_v1 + hotpot mix "
                         "matching training weights [0.7, 0.3].")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path; defaults depend on --mode")
    args = ap.parse_args()
    if args.out is None:
        suffix = {"lmhead": "decode_probe", "generative": "decode_generative",
                  "qa": "decode_qa"}[args.mode]
        args.out = ROOT / "docs/plots" / f"v5_4_{suffix}.txt"

    enc, cfg = load_encoder(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[data] loading {args.chunks} chunks from source={args.source} @ {args.chunk_size} tokens")
    if args.source == "mixed":
        # composite_v1 + hotpot mix matching training weights [0.7, 0.3].
        # Direct sub-dataset construction (mirroring make_mixed_qa_dataloader).
        comp = QADataset(
            COMPOSITE_VAL_P, COMPOSITE_VAL_Q,
            chunk_size=args.chunk_size, passages_per_chunk=300,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=0,
        )
        hp = HotpotQADataset(
            split="validation", tokenizer=tokenizer, chunk_size=args.chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=1,
        )
        ds = MixedQADataset(sources=[comp, hp], weights=[0.7, 0.3], seed=0)
    else:
        ds = HotpotQADataset(tokenizer=tokenizer, split="validation",
                             chunk_size=args.chunk_size,
                             pad_token_id=tokenizer.pad_token_id or 128_001)
    it = iter(ds)
    samples = [next(it) for _ in range(args.chunks)]
    batch = collate_qa(samples, pad_token_id=tokenizer.pad_token_id or 128_001)

    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, dtype=torch.float32)
    llama.train(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_embed = llama.get_input_embeddings().to(device)
    # Llama-3.2-1B uses tied embeddings: lm_head.weight is embed_tokens.weight.
    lm_head_w = llama.get_output_embeddings().weight.detach().to(device)

    memory = run_encoder(enc, cfg, llama_embed, batch, device).detach()
    print(f"[encoder] memory tokens: {tuple(memory.shape)}  (B, K_node, d_llama)")

    B, K, _ = memory.shape
    # Pull original chunk text for each example so the user can eyeball
    # whether memory tokens correlate with chunk content.
    ctx_text_per_chunk = []
    for b in range(B):
        ids = batch.context_ids[b]
        mask = batch.context_mask[b]
        kept = ids[mask.bool()][: 600]
        ctx_text_per_chunk.append(tokenizer.decode(kept.tolist()))

    lines: list[str] = []
    if args.mode == "lmhead":
        decoded = decode_token(memory, lm_head_w.float(), tokenizer, args.top)
        for b in range(B):
            lines.append(f"\n{'=' * 78}")
            lines.append(f"CHUNK {b}  —  context (first ~600 tokens):")
            lines.append(ctx_text_per_chunk[b][:1500].replace("\n", " "))
            lines.append("")
            q_ids = samples[b]["question_ids"].tolist()
            a_ids = samples[b]["answer_ids"].tolist()
            lines.append(f"Question: {tokenizer.decode(q_ids).strip()}")
            lines.append(f"Answer:   {tokenizer.decode(a_ids).strip()}")
            lines.append(f"\nMemory slot decode (top-{args.top} vocab tokens per slot, "
                         f"from memory @ lm_head.T after RMS-norm):")
            for k in range(K):
                toks = decoded[b * K + k]
                display = " | ".join(f"{t!r}" if t.strip() else f"[ws:{t!r}]" for t in toks)
                lines.append(f"  slot {k:2d}: {display}")
    elif args.mode == "qa":
        # Real-question generative QA: prepend memory + the chunk's actual
        # question, greedy-generate up to max_new_tokens, compare to gold.
        # Mirrors compute_qa_loss input format but generates the answer
        # autoregressively instead of teacher-forcing on the gold.
        # Pre-collation samples store question_ids/answer_ids as already-
        # trimmed tensors (no padding mask yet). Use them directly.
        q_id_lists = [samples[b]["question_ids"].tolist() for b in range(B)]
        gens = generate_qa_answers(
            llama.to(device), tokenizer, memory,
            question_ids_list=q_id_lists,
            max_new_tokens=args.max_new_tokens, device=device,
        )
        n_exact = 0
        n_contains = 0
        for b in range(B):
            lines.append(f"\n{'=' * 78}")
            lines.append(f"CHUNK {b}  ({samples[b].get('task_family', '?')})  "
                         f"—  context (first ~600 tokens):")
            lines.append(ctx_text_per_chunk[b][:1500].replace("\n", " "))
            lines.append("")
            q_ids = q_id_lists[b]
            a_ids = samples[b]["answer_ids"].tolist()
            gold = tokenizer.decode(a_ids).strip()
            pred_full = gens[b].strip()
            # Truncate prediction at first newline or sentence end for cleaner display
            pred_short = pred_full.split("\n")[0].split(".")[0].strip()
            ok_exact = gold.lower() == pred_short.lower()
            ok_contains = gold.lower() in pred_full.lower()
            n_exact += int(ok_exact)
            n_contains += int(ok_contains)
            mark = "[EXACT]" if ok_exact else ("[CONTAINS]" if ok_contains else "[MISS]")
            lines.append(f"Q:    {tokenizer.decode(q_ids).strip()}")
            lines.append(f"Gold: {gold}")
            lines.append(f"Pred: {pred_full}")
            lines.append(f"      {mark}")
        lines.append(f"\n{'=' * 78}")
        lines.append(f"SUMMARY over {B} samples:")
        lines.append(f"  exact match (first sentence): {n_exact}/{B}")
        lines.append(f"  gold substring in prediction: {n_contains}/{B}")
    else:
        # generative: prepend memory + a few starter prompts, greedy-generate.
        # Multiple prompts per chunk so we can see if memory steers the
        # generation consistently regardless of starter.
        starter_set = [
            "The document",
            "Based on the passage,",
            "Question: What is this about?\nAnswer:",
        ]
        gen_outputs: dict[str, list[str]] = {}
        for sp in starter_set:
            gens = generate_from_memory(
                llama.to(device), tokenizer, memory,
                starter_prompts=[sp] * B,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            gen_outputs[sp] = gens
        for b in range(B):
            lines.append(f"\n{'=' * 78}")
            lines.append(f"CHUNK {b}  —  context (first ~600 tokens):")
            lines.append(ctx_text_per_chunk[b][:1500].replace("\n", " "))
            lines.append("")
            q_ids = samples[b]["question_ids"].tolist()
            a_ids = samples[b]["answer_ids"].tolist()
            lines.append(f"Question (trained on): {tokenizer.decode(q_ids).strip()}")
            lines.append(f"Gold answer:           {tokenizer.decode(a_ids).strip()}")
            lines.append("")
            lines.append("Generative decode — Llama.generate() with memory prepended:")
            for sp in starter_set:
                gen = gen_outputs[sp][b].replace("\n", " ")
                lines.append(f"  starter={sp!r}")
                lines.append(f"    → {gen}")

    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"\n[output] wrote {args.out}  ({len(lines)} lines)")
    print("\n--- first chunk preview ---")
    cutoff = next((i for i, ln in enumerate(lines[1:], 1)
                   if ln.startswith("=" * 78)), len(lines))
    print("\n".join(lines[1:cutoff]))


if __name__ == "__main__":
    main()
