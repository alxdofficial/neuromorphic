"""CLI for the memory training harness: argparse builder + args→ReprConfig mapping.

Extracted verbatim from ``scripts/train/train.py::main`` (harness reorg phase 3). ``build_parser``
holds all 91 flags; ``args_to_config`` holds the post-parse validation, budget matching, and every
cfg override. The only changes from the original ``main`` body: the ``--mixed-tasks`` default/choices
now read ``DEFAULT_TRAIN_MIX`` / ``TASK_SPEC`` from ``src.memory.data.mixes`` (same values), and the
post-parse block is a function taking ``(args, ap)`` returning ``(cfg, composite_task_weights)``.
"""
from __future__ import annotations

import argparse

import torch

from src.memory.config import ReprConfig
from src.memory.data.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS
from src.memory.data.mixes import DEFAULT_TRAIN_MIX, TASK_SPEC, DEFAULT_MIXED_M


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: stop `--out <path>` from prefix-matching `--out-tag`,
    # which silently baked a full relative path into the tag and re-nested every
    # run under outputs/memory/outputs/memory/.
    ap = argparse.ArgumentParser(allow_abbrev=False)
    # Active suite: latest graph + published closed-book compressor baselines.
    # Retired graph/plastic/splat and older flat/continuous/MT/Mamba variants
    # remain selectable via explicit --variants if needed.
    # hlvocab_baseline + soft_pointer_graph_baseline are ABANDONED (2026-06-15) —
    # still selectable via explicit --variants for reproduction, out of the default.
    ap.add_argument("--variants", nargs="+", default=[
        "slotgraph_baseline",         # emergent-topology slot memory (supersedes graph_baseline)
        "biomem_baseline",            # chunk-parallel gated-delta synaptic-grid (fast-weights arm)
        "icae_baseline",              # ICAE (ICLR'24)
        "ccm_baseline",               # CCM (ICLR'24)
        "autocompressor_baseline",    # AutoCompressor/RMT-style recurrent summary
        "beacon_baseline",            # Activation Beacon
        "vanilla_llama",              # MAE loss FLOOR (band lower bound)
        "vanilla_full_context",       # MAE loss CEILING (band upper bound)
    ])
    ap.add_argument("--steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="Override cfg.learning_rate (default 1e-4). Scale with "
                         "BS — e.g. sqrt rule: 1e-4×sqrt(BS/2) → BS=16 ≈ 2.5e-4.")
    ap.add_argument("--warmup", type=int, default=500,
                    help="LR warmup steps (default 500). Recurrent ports "
                         "(autocompressor) need a longer warmup for stability.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=32,
                    help="Number of batches in the fixed val set. With the "
                         "composite mix sampling 9 families + 3 external sources, "
                         "10 batches ≈ 1 example per family (high per-family "
                         "noise). 32 batches × BS=2 = 64 examples ≈ ~5 per family. "
                         "Bumped from old default of 10.")
    # Default chunk_size 4096→8192 (2026-05-28 tranche-3 protocol; hard datasets
    # need the larger window to fit evidence + distractors).
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--mem-tokens", type=int, default=144,
                    help="Matched MEMORY budget: M memory tokens × d_llama, "
                         "matched across ICAE/CCM/AutoCompressor/Beacon "
                         "(and soft_pointer_graph if selected). Derives icae_n_slots, "
                         "ccm_n_comp, autocompressor_n_slots, and Beacon's α.")
    ap.add_argument("--passages-per-chunk", type=int, default=0,
                    help="composite_v1 passages sampled per chunk. 0 = auto: "
                         "scales with chunk_size (~75 per 1024 tokens). "
                         "Manual override accepted as positive int.")
    ap.add_argument("--task", type=str, default="mixed",
                    choices=["mixed", "qa"],
                    help="mixed = ONE model trained on an equal round-robin of --mixed-tasks "
                         "(mae babi continuation condrecon_bio), evaluated per-task (default, active); "
                         "qa = composite multi-hop QA mix. (Standalone single-task entry points were "
                         "removed; the per-task loaders live on as components of --task mixed.)")
    ap.add_argument("--mixed-tasks", nargs="+", default=list(DEFAULT_TRAIN_MIX),
                    choices=list(TASK_SPEC),
                    help="mixed: tasks in the equal round-robin (default: mae babi continuation "
                         "condrecon_bio). mae = long-passage contiguous MAE compression; babi = "
                         "relational; continuation = next-token prediction; condrecon_bio = "
                         "biographical key→value closed-book recall. Used only when --task mixed.")
    ap.add_argument("--mixed-gate-batches", type=int, default=0,
                    help="mixed val: run the REAL/SHUF/OFF binding gate (example-specificity diagnostic) on "
                         "the first N val batches per task (0=off; ~triples that task's eval cost). Use e.g. 8.")
    ap.add_argument("--mixed-ctx", type=int, default=1024,
                    help="mixed: uniform context_len/chunk for ALL tasks (default 1024).")
    ap.add_argument("--mixed-M", type=int, default=DEFAULT_MIXED_M,
                    help="mixed: uniform memory budget M (slots/edges) for ALL tasks "
                         "(default = mixes.DEFAULT_MIXED_M = 64 → 16:1 compression at ctx=1024). Raised from 32 for the "
                         "streaming-write regime: gives binding headroom so a retention failure "
                         "is a binding failure, not slot-starvation (forgetting pressure comes "
                         "from distractor load > M, not a tiny M).")
    ap.add_argument("--mae-mask-ratio", type=float, default=0.85,
                    help="mae: fraction of answer tokens replaced by <mask> in the forward.")
    ap.add_argument("--hlvocab-emit", choices=["edge_query", "slotattn"], default="edge_query",
                    help="hlvocab emit read-out: edge_query (independent sharp-softmax) "
                         "| slotattn (Slot-Attention competition — slots partition candidates).")
    ap.add_argument("--cond-recon-n-pairs", type=int, default=64,
                    help="conditioned_reconstruction: number of key→value pairs packed into the context (capacity).")
    ap.add_argument("--cond-recon-n-query", type=int, default=1,
                    help="conditioned_reconstruction: keys recalled per example. 1 = single; >1 = multi.")
    ap.add_argument("--cond-recon-value-len", type=int, default=1,
                    help="conditioned_reconstruction: words per value (1 = single-token value).")
    ap.add_argument("--cond-recon-bio-n-facts", type=int, default=3,
                    help="conditioned_reconstruction_bio: random facts packed per value sentence (2-4).")
    ap.add_argument("--bio-query-window", type=int, default=None,
                    help="STREAMING-WRITE retention probe (condrecon_bio): pin the queried key→value "
                         "pair into this encoder window (0 = first = max retention lag, distractors "
                         "after; -1 = last = recency baseline; unset = any window). Ties to "
                         "--window-size; use with --window-size < --mixed-ctx to make the streaming "
                         "windows real (e.g. --window-size 256 --mixed-ctx 1024 --bio-query-window 0).")
    ap.add_argument("--backbone", type=str, default=None,
                    help="override cfg.llama_model (e.g. HuggingFaceTB/SmolLM2-135M for "
                         "the compression line). Auto-sets d_llama from the config.")
    ap.add_argument("--src-tokenizer", type=str, default="meta-llama/Llama-3.2-1B",
                    help="tokenizer that produced the FineWeb-EDU parquet ids (for "
                         "decode→retokenize in the sentence loader).")
    ap.add_argument("--contrastive-shuf-coef", type=float, default=0.0,
                    help="add coef*softplus(L_real - L_shuf) to the loss: makes the "
                         "binding gate ITSELF a training objective (2x step cost; "
                         "the sanctioned aux-loss fallback after the architectural "
                         "ladder, 2026-06-12). Needs batch>1 for the roll.")
    ap.add_argument("--cond-recon-bio-world-seed", type=int, default=0,
                    help="conditioned_reconstruction_bio: world-build seed (train uses this; val uses +10000 → disjoint).")
    ap.add_argument("--compress-len", type=int, default=1024,
                    help="continuation/ae/mae: # natural-text tokens compressed into the 128-token "
                         "memory (then dropped). 1024 = 8x compression (the aligned default).")
    ap.add_argument("--predict-len", type=int, default=64,
                    help="continuation: # next tokens to predict from memory only (closed-book). "
                         "Default 64 isolates the memory signal (less local-autoregression dilution).")
    ap.add_argument("--babi-tasks", type=int, nargs="+", default=list(BABI_DEFAULT_TASKS),
                    help="babi: which bAbI task ids to pool (default = memory-focused subset "
                         "1/2/3/7/8/11/12/13/14: supporting facts, counting, lists, coreference, time).")
    ap.add_argument("--graph-d-graph", type=int, default=0,
                    help="graph: override the graph/vocabulary width d_graph (0 = task default). "
                         "576 matches d_llama → full-rank read tokens (removes the rank handicap).")
    ap.add_argument("--graph-n-nodes", type=int, default=0,
                    help="graph: override the node-bank size N (0 = task default 1024). "
                         "Smaller (384/512) = a tighter vocabulary (the '1024 is oversized' lever); "
                         "barely affects params (bank is N×d_graph).")
    ap.add_argument("--graph-entmax-alpha", type=float, default=1.0,
                    help="graph: node-selection sparsity. 1.0 = softmax (dense blend, default); "
                         "1.5 = entmax (sparse, commits to a few nodes); 2.0 = sparsemax.")
    ap.add_argument("--anomaly-from", type=int, default=-1,
                    help="debug: from this step on, run loss.backward() under torch.autograd.detect_anomaly "
                         "so the first non-finite GRADIENT halts with a traceback to the exact forward op.")
    ap.add_argument("--slotgraph-no-structure", action="store_true",
                    help="slotgraph: disable the MP-read structure = plain prepend of the id-tagged slots "
                         "('id-tagged ICAE'; true pure-ICAE = the icae_baseline variant). "
                         "Ablation: does the message-passing read add anything over id-tagged slots?")
    ap.add_argument("--biomem-no-membrane", action="store_true",
                    help="biomem: disable the LIF membrane (fire on the instantaneous readout, not the "
                         "leaky-integrated potential). Ablation: does the per-neuron membrane help?")
    ap.add_argument("--slotgraph-no-id", action="store_true",
                    help="slotgraph: drop the FIXED orthonormal id_embed from the slots (and routing-head "
                         "input) = pure-ICAE-via-same-code. Pair with --slotgraph-no-structure to isolate "
                         "the id-tag contribution: id-on vs id-off, both flat. Does the free id tagging beat ICAE?")
    ap.add_argument("--slotgraph3-no-write-expand", action="store_true",
                    help="slotgraph3: do NOT materialize expanded edge tokens in the WRITE context — write "
                         "over [window; slots] only, expand edges for the READ prepend ONLY. Ablation: is the "
                         "graph purely a read-time decode, or does the write need to see the structure? "
                         "(Strips the write-forward pooling attractor on the routing matrix A.)")
    ap.add_argument("--slotgraph3-write", choices=["lm", "custom"], default=None,
                    help="slotgraph3 write mixer: 'lm' = frozen SmolLM2 attention + enc-LoRA (pretrained prior, "
                         "~7M matched); 'custom' = frozen-LM window encode (no grad) → from-scratch graph-mixer "
                         "blocks over [hiddens; graph] (position-free; text comprehension held constant). Probe: "
                         "does a purpose-built graph mixer beat LM attention? (default: cfg = lm)")
    ap.add_argument("--slotgraph3-custom-layers", type=int, default=None,
                    help="slotgraph3 custom write: number of from-scratch transformer blocks (default cfg=4, "
                         "d_ff=2×d → ~13M UNMATCHED capacity probe).")
    ap.add_argument("--slotgraph3-gate-ids", action="store_true",
                    help="slotgraph3 soft-id: endpoint labels ride INSIDE the routing weight "
                         "(E = topv·(φ+ids)+role, Switch gate-multiplication) → router gets gradient through "
                         "the dominant id channel; weak edges stop emitting full-loudness labels.")
    ap.add_argument("--slotgraph3-read-topk", type=int, default=None,
                    help="slotgraph3: edges materialized per node (default: mixed-capacity 8). Set 15 (=K-1) "
                         "for DENSE-FORWARD training: every in-support sparsemax edge is materialized (real "
                         "gradient to all of them; exact-zero edges auto-masked from attention) — the frozen "
                         "top-8 boundary disappears and selection self-anneals as sparsemax sharpens.")
    ap.add_argument("--slotgraph3-st-leak", action="store_true",
                    help="slotgraph3 STRAIGHT-THROUGH expansion: forward = exact hard top-k edge tokens "
                         "(context stays K·topk — no K² token growth); backward = soft mixture whose leak = "
                         "the router's out-of-top-k sparsemax mass (self-annealing 95/5 → 100/0). Dense "
                         "gradient to unselected destinations at zero context cost (the scalable "
                         "alternative to --slotgraph3-read-topk 15).")
    ap.add_argument("--slotgraph3-n-nodes", type=int, default=None,
                    help="slotgraph3: override node count K (default: mixed-capacity M/2=16). K=128 = the "
                         "capacity probe (state 2K=256 vectors, UNMATCHED — does binding appear once "
                         "nodes ≥ entities-per-context?). Pair with --slotgraph3-edge-budget.")
    ap.add_argument("--slotgraph3-edge-budget", type=int, default=None,
                    help="slotgraph3: GLOBAL edge budget E — materialize the strongest E edges of the whole "
                         "graph instead of top-k per node (read stays E tokens for ANY K; hubs allowed). "
                         "Requires --slotgraph3-st-leak. E=128 keeps today's 128-token read at K=128.")
    ap.add_argument("--slotgraph3-route-key", choices=["edge", "node"], default=None,
                    help="slotgraph3: which latent provides routing q/k. 'node' = K/V split (route by node "
                         "CONTENT, edge_lat freed for pure relation semantics — kills the routing-stability↔"
                         "relation-content gradient fight on edge_lat). Default cfg: 'edge' (v1).")
    ap.add_argument("--slotgraph3-write-layers", type=int, default=None,
                    help="slotgraph3 LM arm depth: 0 = full ride (graph tokens through ALL layers); N>0 = "
                         "text runs the frozen prefix no-grad, graph tokens splice in for the last N layers "
                         "(+LoRA). N=4 is depth-matched to the custom arm and ~2× faster than full ride.")
    ap.add_argument("--slotgraph3-read", choices=["edges", "raw"], default=None,
                    help="slotgraph3 read: 'raw' = prepend node LATENTS as content tokens (pretrained-space, "
                         "direct decoder→latent gradient, no φ) + edge POINTER tokens (concat [id_src;id_dst] "
                         "+ relation tag). The literature-verdict read (kills the φ/Q-Former bottleneck, "
                         "id-sum direction-blindness, and the missing node-token incidence channel). "
                         "'edges' (default) = v1 φ-synthesized edge tokens.")
    ap.add_argument("--rect-prepend-mask", action="store_true",
                    help="KBLaM-style rectangular decoder mask: prepended memory tokens attend only to "
                         "themselves (no memory↔memory mixing through decoder layers); text attends into "
                         "memory normally.")
    ap.add_argument("--bidir-mem-attn", action="store_true",
                    help="Set-LLM read geometry: the prepended memory block attends to ITSELF "
                         "bidirectionally (edge tokens compose with both endpoint node tokens regardless "
                         "of emission order); text stays causal. Mutually exclusive with "
                         "--rect-prepend-mask.")
    ap.add_argument("--slotgraph3-no-boundary", action="store_true",
                    help="slotgraph3: DISABLE the learned <mem_start>/<mem_end> boundary tokens "
                         "(default ON — explicit span markers, the frozen-LM-injection consensus).")
    ap.add_argument("--slotgraph3-route-act", choices=["softmax", "sparsemax"], default=None,
                    help="routing activation over destinations: softmax (+train-time Gumbel noise; default) "
                         "or sparsemax (LEGACY — exact-zero support = dead-gradient ratchet).")
    ap.add_argument("--slotgraph3-no-init-noise", action="store_true",
                    help="DISABLE per-forward Gaussian sampling of the initial latents (slot-symmetry "
                         "breaking, Slot-Attention style; default ON).")
    ap.add_argument("--slotgraph3-no-write-norm-match", action="store_true",
                    help="DISABLE scaling graph tokens to the text-hidden RMS at the write splice "
                         "(default ON; raw tokens are ~400x quieter than layer-26 hiddens).")
    ap.add_argument("--slotgraph3-no-write-boundary-bidir", action="store_true",
                    help="DISABLE the write-side <mem_start> + bidirectional graph block "
                         "(default ON — matches the read geometry).")
    ap.add_argument("--slotgraph3-write-update", choices=["delta", "additive"], default=None,
                    help="slot update rule: delta (per-slot content gate + competitive read + gated "
                         "interpolation; default) or additive (LEGACY scalar-gated residual).")
    ap.add_argument("--slotgraph3-edge-write", choices=["assoc", "slot"], default=None,
                    help="edge-latent update: assoc (keyed delta-rule write into the persistent 24x24 "
                         "map — files relations under partner-key addresses; default) or slot (LEGACY "
                         "T2 interpolation). assoc needs edge_state=matrix + write_update=delta.")
    ap.add_argument("--slotgraph3-layer-anchor", action="store_true",
                    help="GCNII-style per-layer id/role RE-INJECTION (the 'simple version'): re-add the "
                         "node/edge SLOT identity after each write-suffix layer (WRITE) and the edge-token "
                         "identity before each decoder layer (READ) so the frozen LM's mixing can't smooth "
                         "the graph tokens together through depth. Requires --slotgraph3-read edges + "
                         "--slotgraph3-write lm + --slotgraph3-write-layers N>0.")
    ap.add_argument("--objective-mode", choices=["plain", "contrastive", "trajectory", "behavioral_kl"], default="plain",
                    help="training objective (mixed trainer): 'plain' = CE only; 'contrastive' = "
                         "+ objective_coef × in-batch InfoNCE (each example's memory must explain its "
                         "own target best vs all B-1 other memories; 1 encoder run + GradCache rolled "
                         "reads); 'trajectory' = contrastive + GRPO on the sampled discrete read "
                         "expansion (router-only REINFORCE, reward = binding advantage); 'behavioral_kl' "
                         "= kl_ce_coef·CE + kl_coef·KL(teacher=full-context ‖ student=memory) on answer "
                         "spans (context distillation — the loss-neutrality fix; teacher stop-grad, "
                         "differentiable, no RL).")
    ap.add_argument("--objective-coef", type=float, default=0.5,
                    help="weight of the InfoNCE term (contrastive/trajectory modes).")
    ap.add_argument("--kl-coef", type=float, default=2.0,
                    help="behavioral_kl: weight of the KL(teacher‖student) term (survey default α≈2).")
    ap.add_argument("--kl-ce-coef", type=float, default=1.0,
                    help="behavioral_kl: weight of the CE-to-ground-truth term (grounds the distillation).")
    ap.add_argument("--kl-temp", type=float, default=2.0,
                    help="behavioral_kl: softmax temperature on teacher/student logits (Hinton-style T≈2).")
    ap.add_argument("--grpo-samples", type=int, default=4,
                    help="trajectory mode: number of Gumbel-top-k read-expansion rollouts per step.")
    ap.add_argument("--grpo-coef", type=float, default=1.0,
                    help="trajectory mode: weight of the REINFORCE policy term (per-decision-scaled).")
    ap.add_argument("--grpo-entropy-coef", type=float, default=0.01,
                    help="trajectory mode: entropy bonus on the router distribution (fights policy "
                         "collapse; A3C-standard 0.01).")
    ap.add_argument("--objective-inv-temp", type=float, default=1.0,
                    help="inverse temperature on the row-STANDARDIZED InfoNCE logits (1.0 = unit-sigma "
                         "spread; raise to sharpen).")
    ap.add_argument("--rank-reward-coef", type=float, default=0.0,
                    help="MCR² coding-rate reward on the within-example memory (plain mode): charges the "
                         "objective for rank so memory can't collapse to a low-rank blur. Also the "
                         "(objective vs write-capacity) discriminator. ~0.01-0.1; 0=off.")
    ap.add_argument("--uniform-mem-pos", action="store_true",
                    help="decoder read: give ALL prepended memory tokens the same RoPE position (0) so they "
                         "form an unordered SET equidistant from text (removes intra-memory ordering + RoPE "
                         "distance bias); text keeps normal positions 1..T.")
    ap.add_argument("--slotgraph3-edge-state", choices=["flat", "matrix"], default=None,
                    help="slotgraph3: 'matrix' = view the SAME per-node edge_lat floats as a 24×24 associative "
                         "map; rel(i→j) = M_i·rel_key(node_j) → per-PAIR relation codes (structural per-edge "
                         "specificity; TPR/fast-weights bind-in-write). 'flat' (default) = one shared vector "
                         "per source (per-dst relation = implicit unbinding φ must learn).")
    ap.add_argument("--graph-encoder-lora-rank", type=int, default=0,
                    help="graph: LoRA-adapt the encoder forward like the baselines (0=frozen tap). "
                         "Evens the encoder footing (the graph historically read a frozen tap).")
    ap.add_argument("--graph-read-final", action="store_true",
                    help="graph: read the FINAL hidden (full forward) instead of the mid tap.")
    ap.add_argument("--graph-free-endpoints", action="store_true",
                    help="graph: regress FREE src/dst vectors (drop the bank/selection/topology).")
    ap.add_argument("--beacon-param", nargs="+", default=None,
                    help="Beacon capacity knob: which projections get a trainable copy "
                         "(default q k v ≈ 102M). e.g. --beacon-param v shrinks toward ~17M.")
    ap.add_argument("--beacon-wrap-layers", nargs="+", type=int, default=None,
                    help="Beacon capacity knob: which Llama layer indices to wrap "
                         "(default all 16). e.g. --beacon-wrap-layers 0 1 2 3 → ~4 layers.")
    ap.add_argument("--port-lora-rank", type=int, default=None,
                    help="Capacity knob for ICAE/CCM/AutoCompressor: override their LoRA rank "
                         "(defaults 32/8/32 ≈ 4–6M). e.g. 256 pushes them to ~27–55M (above "
                         "Beacon's ~10M binding floor) to test capacity-vs-mechanism on conditioned-reconstruction.")
    ap.add_argument("--probe-bs", action="store_true",
                    help="Per-arm max-batch-size VRAM probe (no training). For each --variants "
                         "arm: push BS up until OOM, report max-fitting BS + peak VRAM + samp/s, "
                         "then exit. Uses the production cfg + the conditioned-reconstruction data path.")
    ap.add_argument("--probe-bs-list", nargs="+", type=int,
                    default=[8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                    help="BS values to try in --probe-bs (ascending; stops at first OOM).")
    ap.add_argument("--out-tag", type=str, default="v1h")
    ap.add_argument("--resume", action="store_true")
    # Per-window activation checkpointing on the encoder streaming write. With the
    # FlashAttention encoder path (packed windows drop the mask) most variants fit
    # without it, so default ON is a safety net you can disable for full speed.
    ap.add_argument("--grad-ckpt-stream", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--no-hotpot", action="store_true",
                    help="Disable HotpotQA source (default: enabled)")
    # 2026-05-28: hard-only protocol enables narrative + musique by default;
    # use --no-narrative/--no-musique to disable (action='store_false').
    ap.add_argument("--narrative", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable NarrativeQA source (default: ENABLED for "
                         "tranche-3 hard-only protocol). Uses random window "
                         "(oracle-centering removed in post-audit fix).")
    ap.add_argument("--musique", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable MuSiQue-Ans source (default: ENABLED for "
                         "tranche-3 hard-only protocol). Contamination-controlled "
                         "2-4 hop QA — complements HotpotQA by eliminating "
                         "shortcut reasoning.")
    ap.add_argument("--babilong", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable BABILong source (default: ENABLED for the "
                         "v2.1 joint sweep — train + held-out eval, fine-tuned "
                         "small-model track only). Synthetic state-tracking, "
                         "pre-formatted at the config length (4k/8k/16k). "
                         "Use --no-babilong to disable.")
    ap.add_argument("--babilong-config", type=str, default="auto",
                    help="BABILong length config. 'auto' picks the closest "
                         "config below chunk_size (e.g. 4k for chunk=4096, "
                         "8k for chunk=8192). Manual: 0k, 1k, 2k, 4k, 8k, "
                         "16k, 32k, 64k, 128k.")
    ap.add_argument("--mix-weights", nargs="+", type=float,
                    default=[0.2, 0.2, 0.2, 0.2, 0.2],
                    metavar="W",
                    help="Sampling weights for (composite, hotpot, narrative, "
                         "musique, babilong). v2.1 joint-sweep default: equal "
                         "0.2 each across the 5 sources (composite restricted "
                         "to biographical via --composite-task-weights). Equal-"
                         "by-source is the least-gameable fair-head-to-head mix. "
                         "Older 3-tuple callers still work; missing entries "
                         "default to 0.")
    ap.add_argument("--composite-task-weights", nargs="+",
                    default=["biographical:1.0"],
                    metavar="FAMILY:W",
                    help="Per-family weights inside composite_v1. v2.1 joint-"
                         "sweep default: 'biographical:1.0' — composite is "
                         "restricted to the biographical family only (the "
                         "hardest/most-relational family; atomic+relational+"
                         "temporal+aggregation question types over a controlled "
                         "entity-relation world). Unlisted families get weight 0 "
                         "(filtered out). Pass e.g. '' or list families to "
                         "override; 'biographical:2.0 calendar:1.0' for ratios.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Stop training when best.pt hasn't updated for this "
                         "many consecutive val evals past --min-step-for-stop. "
                         "Best-staleness criterion (was previously smoothed "
                         "rolling mean — that one triggered on volatility "
                         "and could fire on the same step a new best landed). "
                         "0 disables. Default 5 (≈ 2500-step plateau at "
                         "val_every=500).")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.01,
                    help="Min val_recon drop to count as a real improvement "
                         "(resets the patience counter). Was hardcoded 1e-4 — "
                         "~200x below val noise (~0.02), so sub-noise drift kept "
                         "resetting patience and runs ground to the step cap. "
                         "0.01 is a meaningful-improvement threshold above noise.")
    ap.add_argument("--min-step-for-stop", type=int, default=3000,
                    help="Don't trigger early-stop before this step. Skips "
                         "warmup-noise era where val is bouncy. Bumped 2000→"
                         "3000 after tranche 1 v2: flat_baseline was still "
                         "improving past step 5000 when patience fired at 5k. "
                         "Slow learners need more runway before plateau check.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global RNG seed (torch/numpy/random). Wired for reproducibility.")
    ap.add_argument("--allow-unmatched-backbone", action="store_true",
                    help="Permit masked_reconstruction on a non-d=576 backbone "
                         "(param-matched ranks are calibrated for SmolLM2-135M).")
    return ap


def args_to_config(args, ap):
    """Apply post-parse validation + build the ReprConfig from parsed args.

    Mutates ``args`` in place (chunk_size / mix_weights / babilong_config / passages_per_chunk
    auto-adjust) and returns ``(cfg, composite_task_weights)``. ``ap`` is the parser (for
    ``ap.error``)."""
    # ── reproducibility: wire the seed (was an unused cfg field) ─────────────
    import random as _random
    import numpy as _np
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ── fail-fast guards (cheap, before any model/data construction) ─────────
    if "hlvocab_baseline" in args.variants and args.task != "masked_reconstruction" \
            and args.chunk_size > 1024:
        raise SystemExit(
            f"hlvocab_baseline builds an [L,L] STDP kernel guarded at L<=1024, but "
            f"--task {args.task} --chunk-size {args.chunk_size} yields L>1024. Use "
            f"--task masked_reconstruction, or --chunk-size <=1024 (single window).")
    if args.contrastive_shuf_coef > 0 and args.batch_size < 2:
        raise SystemExit(
            f"--contrastive-shuf-coef {args.contrastive_shuf_coef} needs batch_size>=2 "
            f"(SHUF rolls memory along the batch dim; B==1 would leave REAL memory).")

    if args.task == "mixed":
        # mixed: the uniform interface — ALL tasks share context_len = mixed_ctx and
        # M = mixed_M. Drive chunk_size/window/compress_len/predict_len from it so the
        # downstream capacity block + per-task loaders all see one consistent length.
        args.chunk_size = args.mixed_ctx
        args.window_size = min(args.window_size, args.chunk_size)
        args.compress_len = args.mixed_ctx          # continuation compress span
        print(f"[auto] mixed: tasks={args.mixed_tasks}  ctx={args.mixed_ctx}  "
              f"M={args.mixed_M}  window_size={args.window_size}  predict_len={args.predict_len}")

    if "/" in args.out_tag:
        ap.error(
            f"--out-tag must be a bare tag, not a path (got {args.out_tag!r}). "
            f"Outputs go to outputs/memory/<out_tag>_<variant>/ automatically; "
            f"pass e.g. --out-tag tranche5_mamba_canonical"
        )

    if "v21" in args.variants:
        raise SystemExit("v21 is not supported in v1h yet.")

    # Flag/weight consistency. The flags toggle source *availability*; the
    # weights control *sampling*. A source with weight=0 is never sampled,
    # so enabling the flag without bumping the weight is a no-op that
    # silently loads ~570MB of data (HotpotQA) or downloads NarrativeQA
    # while contributing nothing. Surface this immediately.
    # Pad mix_weights to length 5 for the 5-source schema.
    padded_weights = list(args.mix_weights) + [0.0] * (5 - len(args.mix_weights))
    args.mix_weights = padded_weights

    # Parse --composite-task-weights "family:weight" pairs into a dict (used by the QA path).
    composite_task_weights = None
    if args.composite_task_weights:
        composite_task_weights = {}
        for item in args.composite_task_weights:
            if ":" not in item:
                raise SystemExit(
                    f"--composite-task-weights expects 'family:weight', got {item!r}"
                )
            fam, w = item.split(":", 1)
            composite_task_weights[fam.strip()] = float(w)
        print(f"[composite] per-family weights: {composite_task_weights}")

    # QA composite mix-weight ⇔ source-flag consistency. ONLY meaningful for --task qa (the
    # composite path). --task mixed uses its own mae/babi/continuation/condrecon_bio loaders and
    # ignores these QA-source flags, so gate the checks — otherwise a mixed run that passes
    # --mix-weights zeroing a QA source (whose flag defaults on) SystemExits for no reason.
    if args.task == "qa":
        if args.narrative and args.mix_weights[2] <= 0:
            raise SystemExit(
                "--narrative is set but mix_weights[2] (NarrativeQA) is 0. "
                "Either drop --narrative or pass --mix-weights with a positive "
                "third value (e.g. --mix-weights 0.5 0.3 0.2)."
            )
        if (not args.no_hotpot) and args.mix_weights[1] <= 0:
            # HotpotQA is on by default; if user explicitly zeros the weight
            # they likely meant to disable the source entirely.
            raise SystemExit(
                "HotpotQA is enabled but mix_weights[1] is 0. Either pass "
                "--no-hotpot or set --mix-weights with a positive second value."
            )
        if args.mix_weights[0] <= 0:
            raise SystemExit(
                "Composite (mix_weights[0]) is 0. composite_v1 is the primary "
                "source and cannot be disabled."
            )
        if args.musique and args.mix_weights[3] <= 0:
            raise SystemExit(
                "--musique is set but mix_weights[3] (MuSiQue) is 0. Either drop "
                "--musique or pass --mix-weights with a positive fourth value "
                "(e.g. --mix-weights 0.4 0.2 0.2 0.2)."
            )
        if args.babilong and args.mix_weights[4] <= 0:
            raise SystemExit(
                "--babilong is set but mix_weights[4] (BABILong) is 0. Either drop "
                "--babilong or pass --mix-weights with a positive fifth value "
                "(e.g. --mix-weights 0.35 0.15 0.15 0.15 0.2)."
            )

    # Base config. Memory-token count + per-variant LoRA ranks/slots are set
    # below (matched-budget block + masked_reconstruction override). LoRA-all:
    # every arm gets the SAME decoder LoRA on the frozen backbone, so the decoder
    # budget is identical and only the memory mechanism differs.
    cfg = ReprConfig(
        batch_size=args.batch_size,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        use_llama_lora=True,
        grad_checkpoint_stream=args.grad_ckpt_stream,
        **({"learning_rate": args.lr} if args.lr is not None else {}),
    )

    # ── backbone resolution (MUST precede budget reporting) ──────────────────
    # Resolve --backbone d_llama/vocab/pad BEFORE the budget block so the printed
    # decoder-read float budget uses the real d_llama (was printing the 2048
    # default even on SmolLM2 d=576) [fix I].
    if args.backbone is not None:
        cfg.llama_model = args.backbone
        from transformers import AutoConfig as _AC, AutoTokenizer as _AT
        _bc = _AC.from_pretrained(args.backbone)
        cfg.d_llama = _bc.hidden_size
        cfg.llama_vocab_size = _bc.vocab_size
        _bt = _AT.from_pretrained(args.backbone)
        # LLM-AGNOSTIC pad + sep derived from the active backbone tokenizer (the old
        # 128001/198 defaults are Llama-only → out of range on SmolLM2 etc.).
        from src.memory.common import resolve_special_ids as _rsi
        cfg.pad_token_id, cfg.sep_token_id = _rsi(_bt)
        print(f"[backbone] {args.backbone}  d_llama={cfg.d_llama}  "
              f"vocab={cfg.llama_vocab_size}  pad={cfg.pad_token_id}  sep={cfg.sep_token_id}")

    # ── Matched MEMORY budget (decoder-read M × d_llama) ────────────────
    # mem_tokens is the single knob; the prepend conditioned-reconstruction arms all emit ~M tokens at
    # d_llama, so the decoder reads the SAME float budget from each — only the
    # memory MECHANISM differs. Beacon (concat) derives α = chunk//M so its total
    # ≈ M. Trainable params are NOT matched (LoRA ports vs the graph substrate
    # differ by design, ~2.5M–48M–100M) — they are reported, not equated.
    M = args.mem_tokens
    cfg.icae_n_slots = M
    cfg.ccm_n_comp = M
    cfg.autocompressor_n_slots = M
    cfg.n_flat_codes = M             # flat/continuous/MT prepend M too (was 192 -> mismatch)
    cfg.beacon_ratio = max(1, args.chunk_size // M)
    if args.beacon_param is not None:
        cfg.beacon_param = tuple(args.beacon_param)
    if args.beacon_wrap_layers is not None:
        cfg.beacon_wrap_layers = tuple(args.beacon_wrap_layers)
    if args.port_lora_rank is not None:
        cfg.icae_lora_rank = args.port_lora_rank
        cfg.ccm_lora_rank = args.port_lora_rank
        cfg.autocompressor_lora_rank = args.port_lora_rank
        print(f"[capacity] ICAE/CCM/AutoCompressor LoRA rank → {args.port_lora_rank}")
    cfg.mae_mask_ratio = args.mae_mask_ratio
    cfg.cond_recon_bio_query_window = args.bio_query_window   # streaming retention placement (mixed path)
    _ceil = lambda a, b: -(-a // b)
    _beacon_M = (_ceil(args.chunk_size, args.window_size)
                 * _ceil(args.window_size, cfg.beacon_ratio))
    print(f"[memory budget] mem_tokens={M} × d_llama={cfg.d_llama} = "
          f"{M * cfg.d_llama:,} prepend decoder-read floats/arm")
    for _a, _m in (("icae", M), ("ccm", M), ("autocompressor", M), ("beacon", _beacon_M)):
        print(f"   {_a:<18} M={_m:<4} → {_m * cfg.d_llama:,} floats")
    if args.task == "mixed":
        # mixed ignores --mem-tokens: the override below sets a FIXED M (mixed_M). The
        # QA-shaped budget figures above (M, _beacon_M) do NOT describe what it emits —
        # skip the multi-window beacon assertion.
        print(f"   [{args.task}] the above --mem-tokens budget is IGNORED; the "
              f"compression-line override below sets the fixed memory budget.")
    elif abs(_beacon_M - M) > max(1, M // 10):
        raise SystemExit(
            f"[memory budget] beacon M={_beacon_M} is >10% off mem_tokens={M} "
            f"(α={cfg.beacon_ratio}); adjust --mem-tokens / chunk / window so the "
            f"matched memory budget holds before launching.")

    # ── compression line: param-matched baselines (backbone resolved above) ───
    if args.task == "mixed":
        # MIXED multi-task: ONE model per arch on a round-robin of mae+babi+
        # continuation, per-task eval. UNIFORM interface: context_len = mixed_ctx
        # and a FIXED M = mixed_M for EVERY task (override the per-task ceil(ctx/30)
        # / k-bucket logic). compute_loss is dispatched per batch (MAE → infill;
        # babi/continuation → generic), so we set BOTH the MAE compressor slots AND
        # the generic prepend budget to the same fixed M. cfg.task_mode = "mixed"
        # is metadata only — the trainer sets model.task_mode per batch.
        cfg.task_mode = "mixed"
        cfg.use_llama_lora = True
        cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
        _M = args.mixed_M                                # FIXED budget (no ceil(chunk/30))
        # MAE-calibrated, param-matched ranks (verified to hold within ~0.1M:
        # icae 6.01M / ccm 6.01M / ac 6.03M / beacon 6.08M memory; graph ~6.9M (d_graph=256)).
        # M barely affects params — the M×d slot embeddings (~37K at M=64) are negligible vs the
        # ~6M LoRA — so these ranks stay matched across the M=32↔64 change. See param_count.py.
        cfg.n_flat_codes = _M
        cfg.icae_n_slots = _M; cfg.icae_lora_rank = 104; cfg.icae_lora_alpha = 208
        cfg.ccm_n_comp = _M; cfg.ccm_lora_rank = 52; cfg.ccm_lora_alpha = 104
        cfg.autocompressor_n_slots = _M
        cfg.autocompressor_lora_rank = 52; cfg.autocompressor_lora_alpha = 104
        cfg.beacon_ratio = max(1, args.mixed_ctx // _M)  # ratio honors the uniform ctx:M (16:1 at M=64)
        from transformers import AutoConfig as _ACLM
        from src.memory.common import beacon_wrap_layers as _bwlm
        _nlayersm = _ACLM.from_pretrained(cfg.llama_model).num_hidden_layers
        cfg.beacon_wrap_layers = _bwlm(_nlayersm, 11)
        # slotgraph (icae-write + fixed partition + RMSNorm-bounded MP read): own frozen base +
        # encoder-LoRA + endpoint heads + the MP read modules (msg/update ≈1.0M). Encoder-LoRA rank
        # TRIMMED to r85 (from icae's r104) to offset the MP params → total ≈ icae's ~6.9M.
        cfg.slotgraph_n_slots = _M
        cfg.slotgraph_n_nodes = _M // 2          # FIXED partition: half nodes, half edges
        cfg.slotgraph_d_key = 64                 # content-addressed routing query/key dim
        cfg.slotgraph_lora_rank = 82; cfg.slotgraph_lora_alpha = 164   # +MP +query/key heads → params ≈ icae
        # slotgraph2 (per-layer graph transformer; soft-dst paintbrush; PREPEND read). Run as a BINDING
        # PROBE, NOT a param-matched cohort entry: recurrent=False → 4 distinct d=576 layers ≈ 24M (~3.4×
        # the ~7M cohort). Justified because the probe question — does routing_diversity lift / SHUF−REAL
        # go positive — is param-ORTHOGONAL; do NOT draw a "structure beats flat at fixed params" claim from
        # its babi-EM. For the matched comparison set cfg.slotgraph2_recurrent=True (ONE shared ~6M layer ×L).
        cfg.slotgraph2_n_slots = _M
        cfg.slotgraph2_n_nodes = _M // 2          # fixed partition: half nodes, half edges
        cfg.slotgraph2_d_key = 64
        # slotgraph3 (compressed-implicit graph, LM-attention write, expanded edge read). MATCHED on
        # STATE: 2K = 32 latents (= baselines' M=32 memory bottleneck) and ~7.0M trainable (enc-LoRA r56).
        # The DECODER INTERFACE is larger than the state: raw read = 16 node tokens + K×read_topk = 128
        # pointer tokens + 2 boundary = 146 prepend positions (legacy edges read: 128) — a deterministic
        # re-representation of the 32 stored latents, NOT extra stored memory. Report BOTH numbers when
        # comparing budgets. A:=I control at eval only (set enc.force_identity_A).
        cfg.slotgraph3_n_nodes = _M // 2          # K = 16 nodes → 2K = 32 stored latents (matched to M)
        cfg.slotgraph3_read_topk = 8              # expanded read view = 128 edge tokens (re-representation of the 32 latents)
        cfg.slotgraph3_d_key = 128                # richer routing keys; enc-LoRA r56 (config) → ~7.0M matched trainable
        # vqicae (icae + VQ-discretized slots): encoder-LoRA r96 + projns + EMA codebook (a buffer,
        # not gradient-trained) → ~7.0M trainable, matched to icae. Large codebook K=8192.
        cfg.vqicae_n_slots = _M
        cfg.vqicae_lora_rank = 100; cfg.vqicae_lora_alpha = 200
        cfg.vqicae_codebook_size = 8192; cfg.vqicae_d_code = 256
        # biomem (gated fast-Hebbian grid): M seeds → prepend = the same M×d read budget as the
        # cohort; readout_h tuned to ~6.9M trainable (W is per-example fast state, NOT counted).
        cfg.biomem_n_slots = _M
        print(f"[capacity] mixed: FIXED M={_M}, beacon_ratio={cfg.beacon_ratio} (ctx "
              f"{args.mixed_ctx}:M = {args.mixed_ctx // _M}:1); baselines icae r104 / "
              f"ccm r52 / ac r52 / beacon 11L / slotgraph r85+MP-read / "
              f"vqicae r100+K{cfg.vqicae_codebook_size} (param-matched ~6.9-7.0M). "
              f"slotgraph2 = {'~6M recurrent (matched)' if cfg.slotgraph2_recurrent else '~24M/4-distinct-layer BINDING PROBE (UNMATCHED — no fixed-param claim)'}.")
        if cfg.d_llama != 576 and not args.allow_unmatched_backbone:
            raise SystemExit(
                f"mixed param-matched ranks are calibrated for SmolLM2-135M (d=576); "
                f"got d_llama={cfg.d_llama} (backbone={cfg.llama_model}). Pass "
                f"--backbone HuggingFaceTB/SmolLM2-135M, or --allow-unmatched-backbone "
                f"to override (capacity match will be off).")
    elif args.task == "qa":
        # composite multi-hop QA: 30:1 COMPRESSION + ~13M memory params. Memory budget
        # M = ceil(context / 30); params matched to the graph anchor (d_graph=384,
        # write=3/read=2, N=1024 ≈ 13.0M) on SmolLM2-135M (d=576). Ranks calibrated
        # for d=576 — capacity match is approximate on other backbones.
        # (See scripts/diagnostics/param_count.py.)
        cfg.use_llama_lora = True
        cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
        _ctx = args.compress_len if args.task == "continuation" else args.chunk_size
        _M = max(1, -(-_ctx // 30))                      # ceil(ctx/30) — the 30:1 budget
        cfg.n_flat_codes = _M
        cfg.icae_n_slots = _M; cfg.icae_lora_rank = 223; cfg.icae_lora_alpha = 446
        cfg.ccm_n_comp = _M; cfg.ccm_lora_rank = 111; cfg.ccm_lora_alpha = 222
        cfg.autocompressor_n_slots = _M
        cfg.autocompressor_lora_rank = 110; cfg.autocompressor_lora_alpha = 220
        cfg.beacon_ratio = 30                            # condensing ratio = the 30:1 compression
        from transformers import AutoConfig as _ACL2
        from src.memory.common import beacon_wrap_layers as _bwl2
        _nlayers2 = _ACL2.from_pretrained(cfg.llama_model).num_hidden_layers
        cfg.beacon_wrap_layers = _bwl2(_nlayers2, 23)
        # graph: ~13M via WIDER node/edge vectors (d_graph 256→384); E = M (same 30:1 budget)
        cfg.graph_d_graph = 384; cfg.graph_write_layers = 3; cfg.graph_read_layers = 2
        cfg.graph_n_nodes = 1024; cfg.graph_n_edges = _M
        print(f"[capacity] {args.task}: 30:1 compression → M={_M} (ctx {_ctx}); "
              f"~13M memory params (graph d_graph=384, E={_M}, baselines icae r223 / "
              f"ccm r111 / ac r110 / beacon 23L).")
    cfg.contrastive_shuf_coef = args.contrastive_shuf_coef
    # graph experiment overrides (win over the task defaults above): wider node/edge
    # vectors (removes the read-token rank handicap) + sparse node selection (entmax).
    if args.graph_d_graph > 0:
        cfg.graph_d_graph = args.graph_d_graph
        print(f"[graph override] d_graph = {cfg.graph_d_graph}")
    if args.graph_n_nodes > 0:
        cfg.graph_n_nodes = args.graph_n_nodes
        print(f"[graph override] n_nodes = {cfg.graph_n_nodes}")
    if args.graph_entmax_alpha > 1.0:
        cfg.graph_entmax_alpha = args.graph_entmax_alpha
        print(f"[graph override] node selection = entmax α={cfg.graph_entmax_alpha}")
    if args.graph_encoder_lora_rank > 0:
        cfg.graph_encoder_lora_rank = args.graph_encoder_lora_rank
        print(f"[graph override] encoder-LoRA rank = {cfg.graph_encoder_lora_rank}")
    if args.graph_read_final:
        cfg.graph_read_final = True
        print("[graph override] reading FINAL hidden (full forward)")
    if args.graph_free_endpoints:
        cfg.graph_free_endpoints = True
        print("[graph override] FREE endpoints (no bank/selection)")
    if args.slotgraph_no_structure:
        cfg.slotgraph_use_structure = False
        print("[slotgraph override] structure OFF = plain prepend of id-tagged slots (id-tagged ICAE; "
              "true pure-ICAE is the icae_baseline variant)")
    if args.slotgraph_no_id:
        cfg.slotgraph_use_id = False
    if args.biomem_no_membrane:
        cfg.biomem_membrane = False
        print("[biomem override] membrane OFF = fire on the instantaneous readout, not the leaky-integrated potential")
    if args.slotgraph3_no_write_expand:
        cfg.slotgraph3_write_expand = False
        print("[slotgraph3 override] write-expand OFF = write over [window; slots] only; graph expanded for the READ prepend only")
    if args.slotgraph3_write is not None:
        cfg.slotgraph3_write = args.slotgraph3_write
        print(f"[slotgraph3 override] write mixer = {cfg.slotgraph3_write}"
              + (" (from-scratch blocks, NO frozen prior — UNMATCHED capacity probe)" if cfg.slotgraph3_write == "custom" else ""))
    if args.slotgraph3_custom_layers is not None:
        cfg.slotgraph3_custom_layers = int(args.slotgraph3_custom_layers)
    if args.slotgraph3_gate_ids:
        cfg.slotgraph3_gate_ids = True
        print("[slotgraph3 override] gate-ids ON = E = topv·(φ+ids)+role (router gradient through the id channel)")
    if args.slotgraph3_read_topk is not None:
        cfg.slotgraph3_read_topk = int(args.slotgraph3_read_topk)
        print(f"[slotgraph3 override] read_topk = {cfg.slotgraph3_read_topk}"
              + (" (dense-forward: all in-support edges materialized)" if cfg.slotgraph3_read_topk >= 15 else ""))
    if args.slotgraph3_st_leak:
        cfg.slotgraph3_st_leak = True
        print("[slotgraph3 override] straight-through leak ON = hard top-k forward, A-mass soft backward "
              "(dense gradient, zero context growth)")
    if args.slotgraph3_n_nodes is not None:
        cfg.slotgraph3_n_nodes = int(args.slotgraph3_n_nodes)
        print(f"[slotgraph3 override] K = {cfg.slotgraph3_n_nodes} nodes (state 2K={2*cfg.slotgraph3_n_nodes} "
              f"vectors — CAPACITY PROBE, unmatched)")
    if args.slotgraph3_edge_budget is not None:
        cfg.slotgraph3_edge_budget = int(args.slotgraph3_edge_budget)
        print(f"[slotgraph3 override] GLOBAL edge budget = {cfg.slotgraph3_edge_budget} strongest edges "
              f"(read stays {cfg.slotgraph3_edge_budget} tokens at any K)")
    if args.slotgraph3_route_key is not None:
        cfg.slotgraph3_route_key = args.slotgraph3_route_key
        print(f"[slotgraph3 override] route-by-{cfg.slotgraph3_route_key}"
              + (" (K/V split: node content addresses, edge_lat = pure relation)" if cfg.slotgraph3_route_key == "node" else ""))
    if args.slotgraph3_edge_state is not None:
        cfg.slotgraph3_edge_state = args.slotgraph3_edge_state
        print(f"[slotgraph3 override] edge_state = {cfg.slotgraph3_edge_state}"
              + (" (per-PAIR relation codes: rel(i→j) = M_i·rel_key(n_j))" if cfg.slotgraph3_edge_state == "matrix" else ""))
    if args.slotgraph3_write_layers is not None:
        cfg.slotgraph3_write_layers = int(args.slotgraph3_write_layers)
        print(f"[slotgraph3 override] LM write depth = last-{cfg.slotgraph3_write_layers} layers "
              f"(frozen no-grad text prefix; LoRA trains only in the suffix)")
    if args.slotgraph3_read is not None:
        cfg.slotgraph3_read = args.slotgraph3_read
        print(f"[slotgraph3 override] read = {cfg.slotgraph3_read}"
              + (" (RAW node-content tokens + [id_src;id_dst] pointer edges — no φ)" if cfg.slotgraph3_read == "raw" else ""))
    if args.rect_prepend_mask:
        cfg.rect_prepend_mask = True
        print("[override] rectangular prepend mask ON (memory tokens attend to self only — KBLaM-style)")
    if args.bidir_mem_attn:
        if args.rect_prepend_mask:
            raise SystemExit("--bidir-mem-attn and --rect-prepend-mask are mutually exclusive")
        cfg.bidir_mem_attn = True
        print("[override] bidirectional memory-block attention ON (Set-LLM: memory composes, text causal)")
    if args.slotgraph3_no_boundary:
        cfg.slotgraph3_boundary_tokens = False
        print("[slotgraph3 override] boundary tokens OFF (<mem_start>/<mem_end> disabled)")
    if args.slotgraph3_route_act is not None:
        cfg.slotgraph3_route_act = args.slotgraph3_route_act
        print(f"[slotgraph3 override] route activation = {cfg.slotgraph3_route_act}")
    if args.slotgraph3_no_init_noise:
        cfg.slotgraph3_init_noise = False
        print("[slotgraph3 override] init noise OFF (deterministic shared init — LEGACY)")
    if args.slotgraph3_no_write_norm_match:
        cfg.slotgraph3_write_norm_match = False
        print("[slotgraph3 override] write splice norm-match OFF (LEGACY 400x quiet tokens)")
    if args.slotgraph3_no_write_boundary_bidir:
        cfg.slotgraph3_write_boundary_bidir = False
        print("[slotgraph3 override] write boundary+bidir OFF (LEGACY causal-over-graph)")
    if args.slotgraph3_write_update is not None:
        cfg.slotgraph3_write_update = args.slotgraph3_write_update
        print(f"[slotgraph3 override] write update = {cfg.slotgraph3_write_update}")
    if args.slotgraph3_edge_write is not None:
        cfg.slotgraph3_edge_write = args.slotgraph3_edge_write
        print(f"[slotgraph3 override] edge write = {cfg.slotgraph3_edge_write}"
              + (" (keyed delta-rule assoc write into the persistent 24x24 map)"
                 if cfg.slotgraph3_edge_write == "assoc" else " (LEGACY slot interpolation)"))
    if args.slotgraph3_layer_anchor:
        cfg.slotgraph3_layer_anchor = True
        print("[slotgraph3 override] per-layer id/role anchor ON = GCNII re-injection each layer "
              "(WRITE: slot identity in the suffix loop; READ: edge identity before each decoder layer)")
    cfg.objective_mode = args.objective_mode
    cfg.objective_coef = float(args.objective_coef)
    cfg.objective_inv_temp = float(args.objective_inv_temp)
    cfg.rank_reward_coef = float(args.rank_reward_coef)
    if args.rank_reward_coef > 0:
        print(f"[objective] MCR² rank-reward ON, coef={args.rank_reward_coef} (plain mode; charges for "
              f"within-example memory rank — the a-vs-d diagnosis discriminator)")
    cfg.grpo_samples = int(args.grpo_samples)
    cfg.grpo_coef = float(args.grpo_coef)
    cfg.grpo_entropy_coef = float(args.grpo_entropy_coef)
    cfg.kl_coef = float(args.kl_coef)
    cfg.kl_ce_coef = float(args.kl_ce_coef)
    cfg.kl_temp = float(args.kl_temp)
    if args.objective_mode != "plain":
        if args.task != "mixed":
            raise SystemExit(f"--objective-mode {args.objective_mode} is implemented in the MIXED trainer "
                             f"only (got --task {args.task}). The 2026-07-02 lesson: fail loudly rather "
                             f"than record an inert flag.")
        if args.contrastive_shuf_coef > 0:
            raise SystemExit("--objective-mode and --contrastive-shuf-coef are mutually exclusive "
                             "(the legacy softplus is its own mode; pick one).")
        # behavioral_kl needs no in-batch negatives (teacher/student, not memory rolls) → any B ok.
        if args.batch_size < 2 and args.objective_mode != "behavioral_kl":
            raise SystemExit(f"--objective-mode {args.objective_mode} needs batch_size >= 2 "
                             f"(in-batch negatives; got {args.batch_size}).")
        # contrastive is objective-level (GradCache over ANY prepend memory) so it runs for the
        # AUX-LOSS-FREE prepend baselines too — valuable as a watermark CONTROL (if icae-contrastive
        # also Goodharts SHUF−REAL with flat EM, the watermark is objective-driven, not sg3-specific).
        # EXCLUDED: vqicae (emits vq_loss → silently inert under the per-example-CE surrogate); hlvocab
        # (load_balance). trajectory stays sg3-only (needs sample_read_expansion).
        _OBJ_OK = {"slotgraph3_baseline", "icae_baseline", "ccm_baseline",
                   "autocompressor_baseline", "beacon_baseline"}
        if args.objective_mode == "trajectory" and set(args.variants) != {"slotgraph3_baseline"}:
            raise SystemExit(f"--objective-mode trajectory supports --variants slotgraph3_baseline only "
                             f"(needs sample_read_expansion); got {args.variants}.")
        if not set(args.variants).issubset(_OBJ_OK):
            raise SystemExit(f"--objective-mode supports {sorted(_OBJ_OK)} (aux-loss-free prepend arms); "
                             f"got {args.variants}. vqicae/hlvocab emit aux losses inert under the "
                             f"GradCache surrogate — excluded by design.")
        if args.objective_mode == "trajectory":
            if (args.slotgraph3_read or "raw") != "raw" or (args.slotgraph3_edge_budget or 0) > 0:
                raise SystemExit("--objective-mode trajectory needs the raw read and per-node top-k "
                                 "(edge_budget=0) — sample_read_expansion supports only that config.")
        print(f"[objective override] mode={args.objective_mode} InfoNCE coef={cfg.objective_coef}"
              + (f" GRPO G={cfg.grpo_samples} coef={cfg.grpo_coef}"
                 if args.objective_mode == "trajectory" else ""))
    if args.uniform_mem_pos:
        cfg.uniform_mem_pos = True
        print("[override] uniform memory position-ids ON (all memory tokens at RoPE pos 0; text at 1..T)")
    cfg.mixed_gate_batches = int(args.mixed_gate_batches)   # REAL/SHUF/OFF binding gate in mixed val (0=off)
    cfg.task_mode = args.task        # accurate ckpt metadata (dispatch still keys on this)
    # actual PER-TASK context length — encoder time-constants (assoc-decay half-life) derive from it.
    # mixed: compress_len := mixed_ctx (set above); qa: the context is CHUNK_SIZE (compress_len would
    # under-count 8x at the defaults → 8x-too-fast decay); other single tasks: compress_len.
    cfg.ctx_len = int(args.chunk_size) if args.task == "qa" else int(args.compress_len)
    cfg.seed = args.seed             # record the actual seed in ckpt metadata
    cfg.anomaly_from = args.anomaly_from   # debug: backward anomaly detection from this step (-1 = off)

    # Auto-pick BABILong config to match chunk_size (audit fix #10).
    if args.babilong_config == "auto":
        # Map chunk_size to nearest BABILong config at or below it.
        cs = args.chunk_size
        if cs >= 16384:
            args.babilong_config = "16k"
        elif cs >= 8192:
            args.babilong_config = "8k"
        elif cs >= 4096:
            args.babilong_config = "4k"
        elif cs >= 2048:
            args.babilong_config = "2k"
        elif cs >= 1024:
            args.babilong_config = "1k"
        else:
            args.babilong_config = "0k"
        if args.babilong:
            print(f"[auto] babilong_config = {args.babilong_config} "
                  f"(scaled for chunk_size={args.chunk_size})")

    # Auto-scale composite passages_per_chunk with chunk_size if user passed 0.
    # composite_v1 passages average ~13 tokens; we target ~75 passages per
    # 1024 chunk tokens so the chunk fills to ~95% even after rejecting
    # over-long candidates.
    if args.passages_per_chunk <= 0:
        args.passages_per_chunk = max(75, (args.chunk_size // 1024) * 75)
        print(f"[auto] composite passages_per_chunk = {args.passages_per_chunk} "
              f"(scaled for chunk_size={args.chunk_size})")

    print(f"config: chunk={args.chunk_size}, window={args.window_size}, "
          f"passages_per_chunk={args.passages_per_chunk}")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")
    return cfg, composite_task_weights
