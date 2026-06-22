"""Config dataclass for the memory / compression module.

Only fields consumed by the LIVE code path are kept here. The active line is
masked_reconstruction (MAE) on a frozen backbone; the models that read this
config are: the `graph` relational parser (the current model; docs/graph_model.md),
the abandoned hlvocab / soft_pointer_graph, and the four faithful baseline ports
(icae / ccm / autocompressor / beacon) plus the vanilla floor/ceiling.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ReprConfig:
    """All hyperparameters for the memory module (compression + binding)."""

    # ── Backbone (frozen LM the encoder injects memory into) ───────────────
    # "llama_model" is historical — any HF chat/base model works. The loader
    # detects whether the tokenizer has a chat template and routes accordingly
    # (see src/memory/chat_template.py). The active compression runs use
    # HuggingFaceTB/SmolLM2-135M (d=576); the default below is for ad-hoc builds.
    llama_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    d_llama: int = 2048             # backbone hidden size
    llama_vocab_size: int = 128_256
    # System prompt placed BEFORE the memory slot in the chat scaffold; only
    # used when the backbone tokenizer has a chat template.
    system_intro_for_memory: str = (
        "You are a helpful assistant. The following text contains memories "
        "from a long document. Use only those memories to answer the user's question."
    )
    # Append the backbone's chat-end token to the answer tokens so teacher-forced
    # training learns a clean end-of-turn marker (chat-template prompting).
    append_answer_eot: bool = True

    # ── Memory token count (prepend budget) ────────────────────────────────
    # Number of memory tokens the baselines prepend to the decoder. The trainer
    # sets this to --mem-tokens (and to 16 for the masked_reconstruction line).
    n_flat_codes: int = 96

    # ── Routing aux losses (inert extension hooks) ─────────────────────────
    # load_balance: anti-collapse on classification routing. z_loss: Mixtral-style
    # penalty on logsumexp(scores)² to bound logit magnitude. codebook_orth: soft
    # orthogonality. NOTE: no ACTIVE encoder currently emits any of these losses
    # (the design avoids aux losses — anti-collapse is architectural), so all three
    # multiply zero today. Kept as wired hooks: an encoder that emits the matching
    # aux key (load_balance_loss / z_loss / codebook_orth_loss) gets it weighted.
    load_balance_coef: float = 0.01
    z_loss_coef: float = 1e-3
    codebook_orth_coef: float = 0.01

    # ── Objective / dispatch ───────────────────────────────────────────────
    # task_mode routes compute_loss (set by the trainer). contrastive_shuf_coef
    # adds a SHUF-contrastive binding term. Real fields (not dynamic attrs) so
    # dataclasses.asdict(cfg) captures them in checkpoint metadata for drift checks.
    task_mode: str = "qa"
    contrastive_shuf_coef: float = 0.0

    # ── Decoder LoRA (shared by every variant so the MAE protocol is learnable)
    # Manual LoRA (no `peft`): W + (B @ A)·(alpha/rank); A small-init, B zero-init
    # so step-0 output equals the frozen base. Standard Hu et al. (2021) q/v rank-16.
    use_llama_lora: bool = False
    llama_lora_rank: int = 16
    llama_lora_alpha: int = 16
    llama_lora_target_names: tuple = ("q_proj", "v_proj")

    # ── ICAE baseline (In-Context Autoencoder, Ge et al. ICLR 2024) ─────────
    # SEPARATE frozen base copy + its OWN encoder-LoRA + M slots (distinct from
    # the decoder LoRA so the two adapters can't collide on a shared base).
    icae_lora_rank: int = 32        # r32 converges in the matched budget; r64 needs ≥1500 steps
    icae_lora_alpha: int = 64
    icae_n_slots: int = 0           # 0 ⇒ fall back to n_flat_codes (matched budget)

    # ── AutoCompressor / RMT recurrent-summary port ────────────────────────
    # Same frozen-backbone + encoder-LoRA setup as ICAE, but the M summary tokens
    # recur across windows. Official Llama-2 recipe wraps q,k,v,o.
    autocompressor_lora_rank: int = 32
    autocompressor_lora_alpha: int = 64
    autocompressor_lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    autocompressor_n_slots: int = 0  # 0 ⇒ fall back to n_flat_codes

    # ── CCM baseline (Compressed Context Memory, Kim et al. ICLR 2024) ──────
    # Recurrent compression via a conditional LoRA gated to fire ONLY on <COMP>
    # token positions (CCM's signature), rank 8 / q,k,v,o. Port reads the COMP
    # tokens' last-layer hiddens as the M memory vectors.
    ccm_lora_rank: int = 8
    ccm_lora_alpha: int = 16        # scale = alpha / rank = 2.0
    ccm_lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    ccm_n_comp: int = 0             # <COMP> tokens per step; 0 ⇒ n_flat_codes
    ccm_fold: str = "merge"         # "merge" (1/t mean, fixed M) | "concat" (grows)

    # ── Beacon baseline (Activation Beacon, Zhang et al., arXiv:2401.03462) ──
    # SEPARATE full beacon q/k/v projections per layer (cloned-init from base,
    # NOT LoRA — Beacon's distinguishing axis vs CCM) routed to beacon token
    # positions; interleaved beacons (one per α-unit); streaming concat. Port
    # reads beacon last-layer hiddens as the M memory vectors.
    beacon_param: tuple = ("q", "k", "v")   # which projections get a separate beacon copy
    beacon_ratio: int = 0                   # condensing ratio α; 0 ⇒ auto from n_flat_codes
    beacon_wrap_layers: tuple = ()          # layer indices to wrap; () ⇒ ALL layers

    # ── masked_reconstruction (MAE) objective ──────────────────────────────
    mae_mask_ratio: float = 0.85            # fraction of tokens masked in the forward

    # ── Activation efficiency ──────────────────────────────────────────────
    # Activation-checkpoint each streaming-write window (peak ≈ one window of
    # encoder activations instead of all n_windows). Exact gradients; recompute
    # in backward. grad_checkpoint_llama: checkpoint the decoder forward (needed
    # for the full-context ceiling arm).
    grad_checkpoint_stream: bool = True
    grad_checkpoint_llama: bool = False

    # ── Tokenizer specials ─────────────────────────────────────────────────
    # SmolLM2 / Llama-3.2 have no dedicated pad id; we reuse a special token for
    # padding and mask those positions out of attention/loss. sep = newline.
    pad_token_id: int = 128_001
    sep_token_id: int = 198

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 5000

    # ── graph (relational-parser graph memory over a learnable node bank) ────
    # models/graph/; design: docs/graph_model.md (SOURCE OF TRUTH). A learnable
    # NODE BANK is the vocabulary (replaces the VQ-VAE — no encode-snap/EMA/commit
    # collapse). The WRITE is a TokenGT parser: E edge-query slots self-attend +
    # cross-attend the observation, and each POINTS into the bank to select src/dst
    # (sharp learnable-temp softmax, never regresses) + regresses an edge state.
    # The READ binds each edge op(src,dst,edge)→one vector and cross-attends those
    # into the frozen LLM (RMS-matched, gated) at a mid-late layer.
    graph_d_graph: int = 256            # graph/vocabulary space width (decoupled from d_llama)
    graph_n_nodes: int = 1024           # N — node bank size (the learnable vocabulary)
    graph_n_edges: int = 16             # E — edge budget
    # M = number of PREPENDED memory tokens / Perceiver latent read-queries. 0 ⇒ fall
    # back to M = n_edges (the legacy concat-per-edge read). >0 DECOUPLES the prepend
    # budget from E: the reader's latent queries cross-attend the 3E graph tokens and
    # emit exactly M tokens (Perceiver-IO / DETR style). The mixed branch sets 32.
    graph_n_read_queries: int = 0
    graph_window: int = 256             # obs window for the PERSISTENT carry-forward: the
                                        # parser ingests the prior graph + each window → updates
                                        # it. Inputs ≤ one window (every MAE sentence) = one parse.
    graph_write_layers: int = 3         # parser depth (self → cross-nodes → cross-obs); ≥3
    graph_read_layers: int = 2          # reader depth (cross-attend edges + causal self)
    graph_heads: int = 4
    graph_ffn_mult: int = 2             # FFN expansion (capacity-matches ~4.6M to the baselines)
    # pointer-softmax sharpness at init (log-temp; 0 ⇒ temp=1, consistent with every
    # attention block). Over QK-RMSNorm'd cosine-scale logits this starts SOFT/near-
    # uniform (the gradient-rich cold-start) and the learnable temp sharpens it (watch
    # graph_ptr_entropy). A negative init (e.g. -1 ⇒ temp≈0.37) biases toward selection
    # from step 0 — the prime sweep knob if the pointer fails to sharpen.
    graph_ptr_logit_temp_init: float = 0.0
    graph_entmax_alpha: float = 1.0     # node-selection sparsity: 1.0=softmax, 1.5/2.0=sparse (entmax)
    graph_obs_tap_layer: int = 6        # frozen-backbone layer tapped for the observation
    graph_encoder_lora_rank: int = 0    # >0: LoRA-adapt the encoder forward like the baselines (0=frozen tap)
    graph_encoder_lora_alpha: int = 0   # 0 → 2×rank
    graph_read_final: bool = False      # read the FINAL hidden (full forward) instead of the mid tap
    graph_free_endpoints: bool = False  # regress FREE src/dst vectors (no bank/selection/topology)
    # ── read mechanism (docs/graph_neighborhood_read.md) ───────────────────────
    # "perceiver"            = the bag-of-edges Perceiver-IO read (control; pools 3E
    #                          edge tokens into M latents — no inter-edge structure).
    # "neighborhood_tokengt" = the node-centric read: each of M slots SELECTS one bank
    #                          node, gathers its incident-edge neighbourhood, and
    #                          TokenGT-aggregates that local subgraph into one token
    #                          (the center node read out after L self-attn layers). The
    #                          slot↔edge incidence w[m,e]=S[m]·(p_src[e]+p_dst[e]) is
    #                          differentiable, so the read pushes gradient into BOTH the
    #                          read-selection AND the write-pointers (topology repair).
    graph_read_mode: str = "perceiver"
    graph_read_degree_cap: int = 0      # k = max incident edges per node window; 0 ⇒ max(1, E//8)
    graph_read_gumbel: bool = True      # Gumbel-softmax node selection (sharp-but-explorable) vs plain softmax
    graph_read_gumbel_temp: float = 1.0 # Gumbel temperature (>0; lower ⇒ sharper/harder selection)

    # ── biomem (gated fast-Hebbian cortical-column grid; models/biomem/) ─────
    biomem_n_cols: int = 36            # C — #columns; C*K must == d_llama (36*16=576)
    biomem_k: int = 16                 # K — column width
    biomem_depth_h: int = 3            # H — layers per column (H-1=2 fast-edge layer-pairs).
                                       # C=36,K=16,H=3 → W = 2*36*16^2 = 18,432 floats = EXACT
                                       # raw-float match to the baselines' 32-token memory-state
                                       # (n_pairs*K=32; trades column width/depth to hit parity).
    biomem_n_slots: int = 16           # M — query seeds = prepend memory tokens (k-sliced)
    biomem_d_cond: int = 48            # per-(col,layer-pair) conditioning vector width
    biomem_reg_hidden: int = 64        # plasticity-regulator MLP hidden width (per-edge; keep narrow)
    biomem_seed_hidden: int = 1280     # query-seed encoder MLP hidden width
    biomem_readout_hidden: int = 4352  # readout MLP hidden width (param-matched ~6.9M to the cohort)
    biomem_leak_init: float = 0.1      # initial leak lambda (learned, sigmoid-parameterized)
    biomem_base_write_rate_init: float = 0.1  # eta0 — learnable base write-rate (softplus); >0 so the
                                       # write-side gets gradient even when the gate g≈0 (cold-start fix)
    biomem_theta_scale: float = 0.1    # std of the random-fixed per-neuron threshold theta
    biomem_read_tap_layer: int = 15    # decoder layer whose hidden state seeds the query-conditioned read
    biomem_grad_checkpoint: bool = True  # checkpoint the per-token sweep (essential for BPTT memory)

    # ── slotgraph (emergent-topology slot memory; models/slotgraph/) ─────────
    # ICAE write (own frozen base + encoder-LoRA, M slots appended to the passage, run through the
    # LM's OWN layers) + a per-LM-layer head that predicts HARD (straight-through) whether each slot
    # is a NODE or an EDGE and (for edges) which two slot-POSITIONS it links; concretized as a TokenGT
    # role+identity embedding (id = fixed orthonormal per-position code; transparent SUM for edges)
    # and re-injected before the next layer. READ = prepend the slots' final hiddens. use_structure
    # =False ⇒ pure ICAE control. The LM does the mixing; only the tiny structure heads are new.
    slotgraph_n_slots: int = 32          # M — slots = PREPEND tokens (matches baselines' M=32)
    slotgraph_lora_rank: int = 32        # encoder-LoRA rank (mixed block sets ~104 to match icae params)
    slotgraph_lora_alpha: int = 64
    slotgraph_start_layer: int = 0       # predict/inject structure from this base layer onward (0 = all)
    slotgraph_temp_init: float = 1.0     # straight-through surrogate temperature (gradient sharpness)
    slotgraph_use_structure: bool = True # False ⇒ pure ICAE (no structure heads/injection) = the ablation

    # ── vqicae (ICAE with VQ-VAE-discretized slots; models/vqicae/) ──────────
    # ICAE write, then each slot is quantized to its nearest code in a large EMA codebook
    # (straight-through + commitment loss + dead-code reinit). Tests discreteness of the memory.
    vqicae_n_slots: int = 0              # M slots; 0 ⇒ n_flat_codes
    vqicae_lora_rank: int = 32           # encoder-LoRA rank (mixed block sets ~96 to match icae params)
    vqicae_lora_alpha: int = 64
    vqicae_codebook_size: int = 8192     # K — codebook entries (EMA buffer, not gradient-trained)
    vqicae_d_code: int = 256             # quantization dim (slot d_llama → d_code → quantize → d_llama)
    vqicae_commit_beta: float = 0.25     # commitment-loss weight (encoder output → its code)
    vqicae_ema_decay: float = 0.99       # codebook EMA decay
    vqicae_reinit_thresh: float = 1e-3   # EMA cluster-size below which a dead code is revived. MUST be
                                         # ≪ the per-code usage (~batch/K); 1.0 reinits ~all codes/step.

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42                  # wired in the trainer (torch/np/random) for reproducibility
