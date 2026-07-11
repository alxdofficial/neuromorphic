"""CLI for the memory training harness: argparse builder + args→ReprConfig mapping.

Extracted from ``scripts/train/train.py::main`` (harness reorg phase 3). ``build_parser``
holds the flags; ``args_to_config`` holds the post-parse validation, budget matching, and every
cfg override. ``--mixed-tasks`` default/choices read ``DEFAULT_TRAIN_MIX`` / ``TASK_SPEC`` from
``src.memory.data.mixes``; the post-parse block is a function taking ``(args, ap)`` returning
the ``ReprConfig``. The retired composite-QA (``--task qa``) machinery has been removed.
"""
from __future__ import annotations

import argparse

import torch

from src.memory.config import ReprConfig
from src.memory.data.sources.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS
from src.memory.data.mixes import DEFAULT_TRAIN_MIX, TASK_SPEC, TASK_ALIASES, DEFAULT_MIXED_M


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
        # ACTIVE trainable cohort (2026-07-10): published closed-book compressors + our graph arm,
        # matched ~7M params / M=96. THE slotgraph is slotgraph_baseline.
        # (beacon/ccm/vqicae/biomem were retired and removed 2026-07-11.)
        "icae_baseline",              # ICAE (ICLR'24) — prepend
        "autocompressor_baseline",    # AutoCompressor/RMT-style recurrent summary — prepend
        "titans_baseline",            # Titans-inspired (deep-MLP test-time-autograd memory) — prepend read (not MAC); needs --no-grad-ckpt-stream
        "gisting_baseline",           # Gisting (per-layer gist-KV) — native per-layer-KV read
        "memoryllm_baseline",         # MemoryLLM (per-layer pool + random-drop) — native per-layer-KV read
        "slotgraph_baseline",         # OUR arm — THE slotgraph (prepend+bidir)
        "vanilla_llama",              # MAE loss FLOOR (band lower bound; eval-only)
        "vanilla_full_context",       # MAE loss CEILING (band upper bound; eval-only)
        "h2o_baseline",               # training-free KV eviction reference (eval-only)
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
                    help="Number of batches in the fixed per-task val set "
                         "(mixed trainer materializes this many per task).")
    # Default chunk_size 4096→8192 (2026-05-28 tranche-3 protocol; hard datasets
    # need the larger window to fit evidence + distractors).
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--window-size", type=int, default=256,
                    help="streaming-write granularity (~paragraph). With --mixed-ctx 2048 ⇒ 8 windows: "
                         "streaming continuation predicts at each boundary; condrecon retention-lag varies.")
    ap.add_argument("--mem-tokens", type=int, default=144,
                    help="Matched MEMORY budget: M memory tokens × d_llama, "
                         "matched across ICAE/AutoCompressor "
                         "(and soft_pointer_graph if selected). Derives icae_n_slots "
                         "and autocompressor_n_slots.")
    ap.add_argument("--task", type=str, default="mixed",
                    choices=["mixed"],
                    help="mixed = ONE model trained on an equal round-robin of --mixed-tasks "
                         "(reconstruct babi doc_qa continuation fact_recall), evaluated per-task (the "
                         "only training path). The retired composite multi-hop QA mix (--task qa) and "
                         "its standalone single-task entry points were removed.")
    ap.add_argument("--mixed-tasks", nargs="+", default=list(DEFAULT_TRAIN_MIX),
                    choices=list(TASK_SPEC) + list(TASK_ALIASES),   # accepts old aliases (mae/qa_rc/condrecon_bio)
                    help="mixed: tasks in the equal round-robin (default: reconstruct babi doc_qa "
                         "continuation fact_recall). reconstruct = contiguous-passage MAE compression "
                         "(fidelity); babi = relational binding (multi-segment renamed); doc_qa = real "
                         "multi-source RC QA (squad/triviaqa/hotpot/musique/multiwoz); continuation = "
                         "multi-horizon next-token prediction; fact_recall = biographical key→value "
                         "closed-book recall. Old names (mae/qa_rc/condrecon_bio) still accepted.")
    ap.add_argument("--mixed-gate-batches", type=int, default=0,
                    help="mixed val: run the REAL/SHUF/OFF binding gate (example-specificity diagnostic) on "
                         "the first N val batches per task (0=off; ~triples that task's eval cost). Use e.g. 8.")
    ap.add_argument("--mixed-ctx", type=int, default=2048,
                    help="mixed: uniform context_len/chunk for ALL tasks (default 2048 → 8×256 windows; "
                         "tiles big-context QA better + fills bio at 40 pairs; 21:1 compression at M=96).")
    ap.add_argument("--mixed-M", type=int, default=DEFAULT_MIXED_M,
                    help="mixed: uniform memory budget M (slots/edges) for ALL tasks "
                         "(default = mixes.DEFAULT_MIXED_M = 96 → 21:1 compression at ctx=2048). 32→64→96: 96 "
                         "keeps CAPACITY off the table (96 slots ≫ ~30 packed bindings) so the sweep measures "
                         "ADDRESSING/structure, not slot-starvation; forgetting pressure comes from "
                         "distractor load > M, not a tiny M.")
    ap.add_argument("--mae-mask-ratio", type=float, default=0.85,
                    help="mae: fraction of answer tokens replaced by <mask> in the forward.")
    ap.add_argument("--no-continuation-multi-horizon", action="store_true",
                    help="continuation: predict once at the full cutoff (single-shot) instead of at every "
                         "streaming-window boundary. Ablation switch for the multi-horizon objective.")
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
    ap.add_argument("--backbone", type=str, default="HuggingFaceTB/SmolLM2-135M",
                    help="cfg.llama_model backbone. DEFAULT = SmolLM2-135M (d=576) — the backbone the "
                         "mixed cohort's param-matched ranks are calibrated for (the mixed path hard-errors "
                         "on d≠576 unless --allow-unmatched-backbone). Auto-sets d_llama from the config.")
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
    ap.add_argument("--rect-prepend-mask", action="store_true",
                    help="KBLaM-style rectangular decoder mask: prepended memory tokens attend only to "
                         "themselves (no memory↔memory mixing through decoder layers); text attends into "
                         "memory normally.")
    ap.add_argument("--bidir-mem-attn", action="store_true",
                    help="Set-LLM read geometry: the prepended memory block attends to ITSELF "
                         "bidirectionally (edge tokens compose with both endpoint node tokens regardless "
                         "of emission order); text stays causal. Mutually exclusive with "
                         "--rect-prepend-mask.")
    ap.add_argument("--objective-mode", choices=["plain", "contrastive", "behavioral_kl"], default="behavioral_kl",
                    help="training objective (mixed trainer). DEFAULT = behavioral_kl (the loss-neutrality "
                         "fix — the project's active objective). 'plain' = CE only; 'contrastive' = "
                         "+ objective_coef × in-batch InfoNCE (each example's memory must explain its "
                         "own target best vs all B-1 other memories; 1 encoder run + GradCache rolled "
                         "reads); 'behavioral_kl' "
                         "= kl_ce_coef·CE + kl_coef·KL(teacher=full-context ‖ student=memory) on answer "
                         "spans (context distillation — the loss-neutrality fix; teacher stop-grad, "
                         "differentiable, no RL).")
    ap.add_argument("--objective-coef", type=float, default=0.5,
                    help="weight of the InfoNCE term (contrastive mode).")
    ap.add_argument("--kl-coef", type=float, default=2.0,
                    help="behavioral_kl: weight of the KL(teacher‖student) term (survey default α≈2).")
    ap.add_argument("--kl-ce-coef", type=float, default=1.0,
                    help="behavioral_kl: weight of the CE-to-ground-truth term (grounds the distillation).")
    ap.add_argument("--kl-temp", type=float, default=2.0,
                    help="behavioral_kl: softmax temperature on teacher/student logits (Hinton-style T≈2).")
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
    ap.add_argument("--graph-encoder-lora-rank", type=int, default=0,
                    help="graph: LoRA-adapt the encoder forward like the baselines (0=frozen tap). "
                         "Evens the encoder footing (the graph historically read a frozen tap).")
    ap.add_argument("--graph-read-final", action="store_true",
                    help="graph: read the FINAL hidden (full forward) instead of the mid tap.")
    ap.add_argument("--graph-free-endpoints", action="store_true",
                    help="graph: regress FREE src/dst vectors (drop the bank/selection/topology).")
    ap.add_argument("--port-lora-rank", type=int, default=None,
                    help="Capacity knob for ICAE/AutoCompressor: override their LoRA rank "
                         "(defaults 32/32 ≈ 4–6M). e.g. 256 pushes them to ~27–55M "
                         "to test capacity-vs-mechanism on conditioned-reconstruction.")
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
    ap.add_argument("--compile-decoder", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="torch.compile(dynamic=True) the shared frozen decoder transformer — "
                         "speeds student+teacher decode for hook-free arms (icae/ac/gisting/"
                         "memoryllm). 4-D rect/bidir masks (slotgraph) "
                         "graph-break; leave off for those.")
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

    Mutates ``args`` in place (chunk_size / window / compress_len auto-adjust from mixed_ctx)
    and returns the ``ReprConfig``. ``ap`` is the parser (for ``ap.error``)."""
    # ── reproducibility: wire the seed (was an unused cfg field) ─────────────
    import random as _random
    import numpy as _np
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ── fail-fast guards (cheap, before any model/data construction) ─────────
    if "hlvocab_baseline" in args.variants and args.chunk_size > 1024:
        raise SystemExit(
            f"hlvocab_baseline builds an [L,L] STDP kernel guarded at L<=1024, but "
            f"--chunk-size {args.chunk_size} yields L>1024. Use --chunk-size <=1024 "
            f"(single window).")
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
    # memory MECHANISM differs. Trainable params are NOT matched (LoRA ports vs the graph substrate
    # differ by design, ~2.5M–48M–100M) — they are reported, not equated.
    M = args.mem_tokens
    cfg.icae_n_slots = M
    cfg.autocompressor_n_slots = M
    cfg.n_flat_codes = M             # flat/continuous/MT prepend M too (was 192 -> mismatch)
    if args.port_lora_rank is not None:
        cfg.icae_lora_rank = args.port_lora_rank
        cfg.autocompressor_lora_rank = args.port_lora_rank
        print(f"[capacity] ICAE/AutoCompressor LoRA rank → {args.port_lora_rank}")
    cfg.mae_mask_ratio = args.mae_mask_ratio
    cfg.continuation_multi_horizon = not args.no_continuation_multi_horizon
    cfg.cond_recon_bio_world_seed = args.cond_recon_bio_world_seed   # threaded into _build_source (bio world)
    cfg.cond_recon_bio_query_window = args.bio_query_window   # streaming retention placement (mixed path)
    print(f"[memory budget] mem_tokens={M} × d_llama={cfg.d_llama} = "
          f"{M * cfg.d_llama:,} prepend decoder-read floats/arm")
    for _a, _m in (("icae", M), ("autocompressor", M)):
        print(f"   {_a:<18} M={_m:<4} → {_m * cfg.d_llama:,} floats")
    if args.task == "mixed":
        # mixed ignores --mem-tokens: the override below sets a FIXED M (mixed_M). The
        # QA-shaped budget figures above (M) do NOT describe what it emits.
        print(f"   [{args.task}] the above --mem-tokens budget is IGNORED; the "
              f"compression-line override below sets the fixed memory budget.")

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
        # icae 6.01M / ac 6.03M memory; graph ~6.9M (d_graph=256)).
        # M barely affects params — the M×d slot embeddings (~37K at M=64) are negligible vs the
        # ~6M LoRA — so these ranks stay matched across the M=32↔64 change. See param_count.py.
        cfg.n_flat_codes = _M
        cfg.icae_n_slots = _M; cfg.icae_lora_rank = 104; cfg.icae_lora_alpha = 208
        cfg.autocompressor_n_slots = _M
        cfg.autocompressor_lora_rank = 52; cfg.autocompressor_lora_alpha = 104
        # Faithful AutoCompressor accumulates κ summaries per seg_len segment → κ·n_segments ≈ M.
        # seg_len = window_size so the single-shot MAE path (chunks 2048→8 internally) and the
        # multi-window path both accumulate to the same M budget.
        cfg.autocompressor_segment_len = args.window_size
        _ac_nwin = max(1, args.mixed_ctx // args.window_size)
        cfg.autocompressor_summary_per_window = max(1, _M // _ac_nwin)   # 96//8 = 12 at ctx2048/win256
        # MemoryLLM: N=M slots/layer pool (~1.66M) + q/k/v/o LoRA all layers. r46 measured 7.88M
        # (LoRA denser than estimated); r39 ≈ 7.0M cohort match. TODO precise-calibrate w/ param_count.
        cfg.memoryllm_n_mem = _M
        cfg.memoryllm_lora_rank = 39; cfg.memoryllm_lora_alpha = 78
        cfg.memoryllm_k_new = max(1, _M // 12)           # random-drop K/window (~half-life over 8 windows)
        # Gisting: κ=M/n_seg gist tokens per seg_len segment (accumulate to M), q/v LoRA r104 = ICAE-matched.
        cfg.gisting_n_gist = _M; cfg.gisting_segment_len = args.window_size
        cfg.gisting_gist_per_seg = max(1, _M // _ac_nwin)
        cfg.gisting_lora_rank = 104; cfg.gisting_lora_alpha = 208
        # Titans: read = N_p persistent + M_q readout = M; deep-MLP memory h sized to ~7M (2·d·h dominates).
        cfg.titans_d_mem = cfg.d_llama
        cfg.titans_n_persistent = max(1, _M // 6); cfg.titans_n_read_seeds = _M - cfg.titans_n_persistent
        cfg.titans_mem_hidden = 4650                      # 6.03M ENCODER → 6.95M TOTAL (+0.92M shared decoder
                                                          # read-LoRA), matched to the cohort. The earlier 5448
                                                          # sized the ENCODER to 7M, ignoring the +0.92M LoRA
                                                          # every arm carries → 7.93M total, out of band (audit).
        # THE slotgraph (docs/slotgraph_design.md): 96 nodes = M read budget, dense N×N relation d_e=32
        # plus one dynamic confidence scalar per ordered pair,
        # separate write/read adapters on separate frozen LM copies. Read geometry is forced in model.py.
        cfg.slotgraph_n_nodes = _M
        cfg.slotgraph_window = args.window_size
        # write-LoRA rank 84 → 6.95M TOTAL trainable (encoder 6.03M + shared decoder read-LoRA 0.92M),
        # parameter- and read-length-MATCHED to the cohort (icae 6.97M / autocompressor 6.92M) — NOT
        # capacity-matched (persistent STATE floats vary ~194× across arms; see fairness axis). Was rank 16
        # = 3.03M total,
        # which under-parameterized the arm ~2.3× (audit fairness finding #4). The rank is the encoder-
        # capacity knob (adapts the frozen LM's attention for the graph harvest), analogous to icae r104.
        cfg.slotgraph_lora_rank = 84; cfg.slotgraph_lora_alpha = 168
        # h2o (training-free KV eviction): M = same budget; no encoder LoRA (eval-only arm).
        cfg.h2o_n_budget = _M
        print(f"[param-match] mixed: FIXED M={_M} (ctx {args.mixed_ctx}:M = {args.mixed_ctx // _M}:1); "
              f"ACTIVE cohort icae r104 / ac r52 / titans h4650 / gisting r104 / memoryllm r39 / "
              f"slotgraph r84-LoRA×2 (~6.95M trainable each). MATCHED: params + read-length; NOT persistent "
              f"state (floats vary ~194×: titans ~10.7M vs icae 55K). h2o eval-only.")
        if cfg.d_llama != 576 and not args.allow_unmatched_backbone:
            raise SystemExit(
                f"mixed param-matched ranks are calibrated for SmolLM2-135M (d=576); "
                f"got d_llama={cfg.d_llama} (backbone={cfg.llama_model}). Pass "
                f"--backbone HuggingFaceTB/SmolLM2-135M, or --allow-unmatched-backbone "
                f"to override (param/read-length match will be off).")
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
    if args.rect_prepend_mask:
        cfg.rect_prepend_mask = True
        print("[override] rectangular prepend mask ON (memory tokens attend to self only — KBLaM-style)")
    if args.bidir_mem_attn:
        if args.rect_prepend_mask:
            raise SystemExit("--bidir-mem-attn and --rect-prepend-mask are mutually exclusive")
        cfg.bidir_mem_attn = True
        print("[override] bidirectional memory-block attention ON (Set-LLM: memory composes, text causal)")
    cfg.objective_mode = args.objective_mode
    cfg.objective_coef = float(args.objective_coef)
    cfg.objective_inv_temp = float(args.objective_inv_temp)
    # early-stop knobs (were parsed but never consumed — now wired into the mixed trainer's val loop).
    cfg.patience = int(args.patience)
    cfg.early_stop_min_delta = float(args.early_stop_min_delta)
    cfg.min_step_for_stop = int(args.min_step_for_stop)
    cfg.rank_reward_coef = float(args.rank_reward_coef)
    if args.rank_reward_coef > 0:
        print(f"[objective] MCR² rank-reward ON, coef={args.rank_reward_coef} (plain mode; charges for "
              f"within-example memory rank — the a-vs-d diagnosis discriminator)")
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
        # EXCLUDED: hlvocab (load_balance).
        _OBJ_OK = {"slotgraph_baseline", "icae_baseline",
                   "autocompressor_baseline",
                   # aux-loss-free arms added 2026-07-09 (gisting/memoryllm/titans emit no vq/load-balance
                   # aux loss, so they are valid under behavioral_kl; per-layer-KV students dispatch fine
                   # via _prefix_kv_forward). Still gated by the pre-training verification (#158).
                   # slotgraph is aux-loss-free (prepend arm) → valid under behavioral_kl.
                   "gisting_baseline", "memoryllm_baseline", "titans_baseline"}
        # eval-only arms (vanilla floor/ceiling, h2o) don't train, so the training objective doesn't apply
        # to them — exempt them from the whitelist (they're skipped in train.py's mixed_variants loop).
        _EVAL_ONLY = {"vanilla_llama", "vanilla_full_context", "h2o_baseline"}
        _train_variants = set(args.variants) - _EVAL_ONLY
        if not _train_variants.issubset(_OBJ_OK):
            raise SystemExit(f"--objective-mode supports {sorted(_OBJ_OK)} (aux-loss-free prepend arms) "
                             f"+ eval-only {sorted(_EVAL_ONLY)}; got trainable {sorted(_train_variants)}. "
                             f"hlvocab emits aux losses inert under the GradCache surrogate — excluded.")
        # GradCache (contrastive) detaches ONLY the prepend memory into mem_leaf and re-enters
        # the encoder graph exactly once (objectives.py mem.backward). The per-layer-KV arms carry their
        # DIFFERENTIABLE content in aux["past_kv"] (empty prepend), which GradCache passes verbatim to
        # every roll — so the 2nd roll's backward hits the freed encoder graph ("backward a 2nd time").
        # behavioral_kl is safe (its own CE+KL backward, no GradCache), so keep these arms whitelisted
        # for it and block them ONLY for the GradCache mode.
        if args.objective_mode == "contrastive":
            _KV_ARMS = {"gisting_baseline", "memoryllm_baseline"}
            _bad_kv = _KV_ARMS.intersection(args.variants)
            if _bad_kv:
                raise SystemExit(
                    f"--objective-mode {args.objective_mode} (GradCache) does NOT support the per-layer-KV "
                    f"arms {sorted(_bad_kv)}: their differentiable memory lives in aux['past_kv'], which "
                    f"GradCache reuses across rolls → 2nd-backward-through-freed-graph crash. These arms "
                    f"are valid under --objective-mode behavioral_kl only.")
        print(f"[objective override] mode={args.objective_mode} InfoNCE coef={cfg.objective_coef}")
    if args.uniform_mem_pos:
        cfg.uniform_mem_pos = True
        print("[override] uniform memory position-ids ON (all memory tokens at RoPE pos 0; text at 1..T)")
    cfg.mixed_gate_batches = int(args.mixed_gate_batches)   # REAL/SHUF/OFF binding gate in mixed val (0=off)
    cfg.task_mode = args.task        # accurate ckpt metadata (dispatch still keys on this)
    # actual PER-TASK context length — encoder time-constants (assoc-decay half-life) derive from it.
    # mixed: compress_len := mixed_ctx (set above).
    cfg.ctx_len = int(args.compress_len)
    cfg.seed = args.seed             # record the actual seed in ckpt metadata
    cfg.anomaly_from = args.anomaly_from   # debug: backward anomaly detection from this step (-1 = off)

    print(f"config: chunk={args.chunk_size}, window={args.window_size}")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")
    return cfg
