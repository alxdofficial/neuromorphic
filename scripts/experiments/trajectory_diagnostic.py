"""Run diagnostic probes on the trained trajectory model checkpoint.

What we're investigating: WHY does the trajectory model underperform the
flat-bank baseline? Hypotheses to test:

  H1. Trajectories oscillate / revisit same cells. K_read=8 hops produce
      fewer than 8 unique cells per trajectory.
  H2. All 8 facts in a chunk route to similar entry concepts (entry_proj
      not discriminating facts).
  H3. Pairwise overlap of write regions across the 8 facts in a chunk is
      high — facts blur together via scatter_mean.
  H4. The fraction of cells "ever touched" across full training is small
      — model is using a tiny subset of the 4096 cells.
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
from transformers import AutoTokenizer

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.phase1_retrieval import RetrievalSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wave1_v4/ckpt.pt")
    ap.add_argument("--val-jsonl", default="data/wave1_retrieval/facts_val.jsonl")
    ap.add_argument("--num-chunks", type=int, default=32)
    ap.add_argument("--flat-bank", action="store_true")
    args = ap.parse_args()

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    cfg = TrajMemConfig()
    cfg.flat_bank = args.flat_bank
    if cfg.D < 9:
        cfg.D = 9
    model = IntegratedLM(cfg).to(DEVICE, dtype=DTYPE)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    print(f"loaded ckpt at step {ck.get('trainer_state', {}).get('step_count', '?')}")

    sampler = RetrievalSampler(args.val_jsonl, seed=42, chunk_size=8)
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Aggregators
    unique_per_traj_list = []           # H1: unique cells per K_read=8 hop trajectory
    entry_concept_variance = []         # H2: variance of entries across 8 facts in a chunk
    pairwise_write_overlap = []         # H3: pairwise overlap of write regions across facts
    global_touched = set()              # H4: cells ever touched by writes
    global_read_touched = set()         # cells ever touched by reads

    for chunk_idx in range(args.num_chunks):
        chunk = sampler.sample_chunk()
        T = cfg.T_window
        passages = torch.empty(1, 8, T, dtype=torch.int64, device=DEVICE)
        for i in range(8):
            ids = chunk["fact_passages_token_ids"][i]
            ids = (ids[:T] + [pad_id] * max(0, T - len(ids)))[:T]
            passages[0, i] = torch.tensor(ids, device=DEVICE)
        qa = list(chunk["question_token_ids"]) + list(chunk["answer_token_ids"])
        qa = (qa + [pad_id] * max(0, T - len(qa)))[:T]
        qa = torch.tensor([qa], device=DEVICE)

        with torch.no_grad():
            prev_state = model.manifold.reset_states(batch_size=1)
            prev_hiddens = None
            entry_concepts_per_fact = []         # one entry-concept per fact (J entries actually, use [0])
            write_visits_per_fact = []           # set of cells per fact

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
                wv = out["write_visited"]        # [1, J, K_write] int
                # First-hop entry concept per trajectory.
                entry_per_j = wv[0, :, 0].tolist()  # J entry concepts
                entry_concepts_per_fact.append(entry_per_j[0])  # take j=0 as representative
                flat = wv[0].reshape(-1).tolist()
                write_visits_per_fact.append(set(flat))
                global_touched.update(flat)

                # H1: unique cells per write trajectory.
                for j in range(wv.shape[1]):
                    traj_visits = wv[0, j].tolist()
                    unique_per_traj_list.append(len(set(traj_visits)))

            # Read at QA window.
            out_qa = model.forward_window(
                lm_input_ids=qa,
                prev_window_hiddens=prev_hiddens,
                prev_states=prev_state,
                target_mask=None,
                hard_routing=False,
                use_kv_cache=False,
            )
            rv = out_qa["read_visited"]
            for j in range(rv.shape[1]):
                traj_visits = rv[0, j].tolist()
                unique_per_traj_list.append(len(set(traj_visits)))
            global_read_touched.update(rv[0].reshape(-1).tolist())

            # H2: entry variance across the 8 facts in this chunk.
            entry_concepts = set(entry_concepts_per_fact)
            entry_concept_variance.append(len(entry_concepts))

            # H3: pairwise overlap of write regions.
            n_facts = 8
            overlaps = []
            for i in range(n_facts):
                for j in range(i + 1, n_facts):
                    a, b = write_visits_per_fact[i], write_visits_per_fact[j]
                    if a and b:
                        ov = len(a & b) / min(len(a), len(b))
                        overlaps.append(ov)
            pairwise_write_overlap.extend(overlaps)

    print()
    print("=" * 75)
    print(f"DIAGNOSTIC: trained trajectory model, {args.num_chunks} val chunks")
    print("=" * 75)

    # H1
    n_traj = len(unique_per_traj_list)
    avg_unique = sum(unique_per_traj_list) / n_traj
    print(f"\nH1 - Oscillation / coverage per trajectory (K=8 hops):")
    print(f"  avg unique cells per trajectory: {avg_unique:.2f} / 8 hops")
    print(f"  distribution: min={min(unique_per_traj_list)}, max={max(unique_per_traj_list)}")
    if avg_unique < 5:
        print(f"  ⚠ WARNING: avg < 5 means trajectories oscillate")
    elif avg_unique < 7:
        print(f"  ⚠ avg < 7: some oscillation")
    else:
        print(f"  ✓ good coverage")

    # H2
    avg_entry_var = sum(entry_concept_variance) / len(entry_concept_variance)
    print(f"\nH2 - Entry-concept variance across 8 facts/chunk:")
    print(f"  avg distinct entry concepts per chunk: {avg_entry_var:.2f} / 8 facts")
    if avg_entry_var < 3:
        print(f"  ⚠ WARNING: <3 distinct entries -> facts collapse to same region")
    elif avg_entry_var < 6:
        print(f"  ⚠ <6 distinct: partial collapse")
    else:
        print(f"  ✓ facts route to diverse entries")

    # H3
    avg_overlap = sum(pairwise_write_overlap) / len(pairwise_write_overlap)
    print(f"\nH3 - Pairwise write-region overlap (across 8 facts in a chunk):")
    print(f"  avg overlap: {avg_overlap:.3f}")
    print(f"  (0.0 = facts disjoint; 1.0 = facts overlap completely)")
    if avg_overlap > 0.5:
        print(f"  ⚠ WARNING: high overlap -> facts blur together via scatter_mean")
    elif avg_overlap > 0.2:
        print(f"  ⚠ moderate overlap: some blur")
    else:
        print(f"  ✓ facts well separated")

    # H4
    total_writes_touched = len(global_touched)
    total_reads_touched = len(global_read_touched)
    write_frac = total_writes_touched / cfg.N
    read_frac = total_reads_touched / cfg.N
    print(f"\nH4 - Global cell coverage across {args.num_chunks} chunks:")
    print(f"  cells ever WRITTEN: {total_writes_touched} / {cfg.N} ({write_frac:.1%})")
    print(f"  cells ever READ:    {total_reads_touched} / {cfg.N} ({read_frac:.1%})")
    if write_frac < 0.10:
        print(f"  ⚠ WARNING: <10% write coverage -> model uses tiny manifold subset")
    elif write_frac < 0.30:
        print(f"  ⚠ partial coverage")
    else:
        print(f"  ✓ good coverage")


if __name__ == "__main__":
    main()
