"""Decode probe: what does Llama decode if we feed it ONLY the read trajectory?

Trained model is loaded. For a sampled val chunk:
1. Run the 8 writes normally - manifold state at QA window.
2. Run the read window - get the read trajectory concept states.
3. Project the 32 trajectory states (J*K_read = 4*8) via mem_inject.W_out to d_lm.
4. Feed those 32 "memory token embeddings" to Llama as inputs_embeds (skipping
   the token embedding lookup). Optionally prepend BOS.
5. AR-decode N tokens. Print decoded text alongside the ground-truth target fact.

The point: tell us WHAT the memory module's read trajectory carries.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, ".")

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.phase1_retrieval import RetrievalSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wave1_v4/ckpt.pt")
    ap.add_argument("--val-jsonl", default="data/wave1_retrieval/facts_val.jsonl")
    ap.add_argument("--num-chunks", type=int, default=5)
    ap.add_argument("--gen-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    print("Loading tokenizer + config...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    cfg = TrajMemConfig()
    if cfg.D < 9:
        cfg.D = 9

    print(f"Loading model + ckpt from {args.ckpt}...")
    model = IntegratedLM(cfg).to(DEVICE, dtype=DTYPE)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    print(f"  loaded checkpoint at step {ck.get('trainer_state', {}).get('step_count', '?')}")

    sampler = RetrievalSampler(args.val_jsonl, seed=42, chunk_size=8)

    mem_inject = model._mem_inject_layer()
    if mem_inject is None:
        print("ERROR: no mem_inject layer found.")
        return

    W_out = mem_inject.W_out
    print(f"mem_inject.W_out: {W_out.__class__.__name__}")
    print(f"  cfg.D_concept={cfg.D_concept}, cfg.d_lm={cfg.d_lm}")

    pad_id = tok.pad_token_id or tok.eos_token_id

    for chunk_idx in range(args.num_chunks):
        chunk = sampler.sample_chunk()
        target_idx = chunk["target_idx"]
        target_meta = chunk["metadata"]
        target_fact_id = chunk["target_fact_id"]
        question_ids = chunk["question_token_ids"]
        answer_ids = chunk["answer_token_ids"]

        T = cfg.T_window
        passages = torch.empty(1, 8, T, dtype=torch.int64, device=DEVICE)
        for i in range(8):
            ids = chunk["fact_passages_token_ids"][i]
            ids = ids[:T] + [pad_id] * max(0, T - len(ids))
            passages[0, i] = torch.tensor(ids[:T], device=DEVICE)

        qa_ids_list = list(question_ids) + list(answer_ids)
        if len(qa_ids_list) > T:
            qa_ids_list = qa_ids_list[:T]
        qa_ids_padded = qa_ids_list + [pad_id] * (T - len(qa_ids_list))
        qa = torch.tensor([qa_ids_padded], device=DEVICE)

        with torch.no_grad():
            prev_state = model.manifold.reset_states(batch_size=1)
            prev_hiddens = None
            for i in range(8):
                out = model.forward_window(
                    lm_input_ids=passages[:, i, :],
                    prev_window_hiddens=prev_hiddens,
                    prev_states=prev_state,
                    target_mask=None,
                    hard_routing=False,
                    use_kv_cache=False,
                    write_only_grad=False,
                )
                prev_state = out["new_states"]
                prev_hiddens = out["current_hiddens"]

            out_qa = model.forward_window(
                lm_input_ids=qa,
                prev_window_hiddens=prev_hiddens,
                prev_states=prev_state,
                target_mask=None,
                hard_routing=False,
                use_kv_cache=False,
            )
            # forward_window's output dict stores IDs, not states. Recover
            # states by indexing the post-write manifold state at the
            # read-visited IDs. (Read doesn't mutate state, so this matches
            # what mem_inject's cross-attn sees.)
            read_visited_ids = out_qa["read_visited"]  # [1, J, K_read] int
            J, K_read = read_visited_ids.shape[1], read_visited_ids.shape[2]
            flat_ids = read_visited_ids.reshape(1, -1)  # [1, J*K_read]
            # prev_state is [1, N, D_concept]; gather along dim=1.
            mem_tokens = torch.gather(
                prev_state, dim=1,
                index=flat_ids.unsqueeze(-1).expand(-1, -1, cfg.D_concept),
            )  # [1, J*K_read, D_concept]

            w_dtype = next(W_out.parameters()).dtype
            mem_lm = W_out(mem_tokens.to(w_dtype)).to(DTYPE)

            bos_id = tok.bos_token_id or tok.eos_token_id
            bos_emb = model.llama.get_input_embeddings()(
                torch.tensor([[bos_id]], device=DEVICE),
            )
            full_emb = torch.cat([bos_emb, mem_lm], dim=1)

            # Install a no-op memory_fn so mem_inject doesn't fail (we don't
            # want a memory cross-attn during this decode — the memory is
            # IN the input embeddings, not pulled via mem_inject).
            mem_inject.memory_fn = lambda h: torch.zeros_like(h)

            gen_ids = []
            past_embs = full_emb
            for _ in range(args.gen_tokens):
                out = model.llama(inputs_embeds=past_embs, use_cache=False)
                logits = out.logits[:, -1, :]
                if args.temperature > 0:
                    probs = F.softmax(logits / args.temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_id = int(logits.argmax(dim=-1).item())
                if next_id == tok.eos_token_id:
                    break
                gen_ids.append(next_id)
                next_emb = model.llama.get_input_embeddings()(
                    torch.tensor([[next_id]], device=DEVICE),
                )
                past_embs = torch.cat([past_embs, next_emb], dim=1)

            decoded = tok.decode(gen_ids, skip_special_tokens=False)
            mem_inject.memory_fn = None

        print()
        print("=" * 80)
        print(f"CHUNK {chunk_idx+1}/{args.num_chunks}")
        print(f"  target fact id:    {target_fact_id}")
        print(f"  target class.attr: {target_meta['target_entity_class']}.{target_meta['target_attribute']}")
        print(f"  target_idx:        {target_idx}")
        print()
        gt_q = tok.decode(question_ids, skip_special_tokens=True)
        gt_a = tok.decode(answer_ids, skip_special_tokens=True)
        print(f"  Q (gold): {gt_q}")
        print(f"  A (gold): {gt_a}")
        print()
        print(f"  Decoded from read trajectory (32 memory tokens + BOS):")
        print(f"  >>> {decoded!r}")


if __name__ == "__main__":
    main()
