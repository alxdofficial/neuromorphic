"""Config dataclass for the memory / compression module.

Only fields consumed by the LIVE code path are kept here. The active line is the mixed
4-task objective on a frozen backbone; the models that read this config are: the four
published baseline ports (icae / ccm / autocompressor / beacon), biomem (fast-Hebbian),
slotgraph (emergent-topology slots), vqicae (VQ-discretized slots), and the vanilla
floor/ceiling. (The old `graph` relational parser was retired — slotgraph supersedes it.)
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

    # ── biomem (chunk-parallel gated-delta synaptic-grid memory; models/biomem/) ─────
    biomem_n_cols: int = 36            # C — #columns (= scan heads); C*K = d_grid = 576
    biomem_k: int = 16                 # K — column width (per-column fast-weight head_dim)
    biomem_depth_h: int = 5            # H — layers per column → H-1=4 STACKED synaptic fast-weight layers.
                                       # The DEEP all-synaptic column (info through 4 fast-weight layers with a
                                       # learned per-neuron threshold + residual between each) is what makes this
                                       # biomem and not a 1-layer DeltaNet. W = n_pairs*C*K² floats/example is
                                       # per-example transient state (UNCOUNTED, like a KV cache); the read
                                       # budget (M*d=32*576) is what's matched to the baselines.
    biomem_n_slots: int = 32           # M — # learned read seeds = # prepend memory tokens (32×576 read budget)
    biomem_use_surprise: bool = True   # feed the frozen LM's per-token next-token prediction error (−log p, a
                                       # free pretrained surprise signal) into the write-rate gate + decay.
    biomem_per_layer_refresh: bool = True  # at EVERY decoder layer, re-read W with the current (attention-mixed)
                                       # slot hiddens and add a zero-init-gated recall → slots become query-aware.
    biomem_readout_hidden: int = 4500  # readout MLP hidden width (param-matched ~6.9M to the cohort)
    biomem_decay_init: float = 0.99    # init per-(layer,col) retention α (input-dependent decay = sigmoid(decay_proj);
                                       # zero-init weight ⇒ α starts uniform ≈ this, then LEARNS input-dependence)
    biomem_theta_scale: float = 0.1    # std of the per-neuron threshold theta (random INIT; learned)
    biomem_membrane: bool = True       # LIF membrane: each neuron leaky-integrates its input current o over
                                       # tokens (m_t = λ·m_{t-1}+o_t) and fires hardtanh(m−θ) — a per-neuron
                                       # ACTIVITY memory (built-up potential = dynamic threshold), complementing
                                       # the synaptic memory W. Parallel (per-channel exp-kernel causal conv).
    biomem_membrane_window: int = 64   # truncation length of the membrane EMA (λ_max^64≈1e-3 → effectively exact)
    biomem_membrane_max_decay: float = 0.9  # cap on the per-neuron leak λ = max·sigmoid(raw) (keeps the window valid)

    # ── slotgraph (fixed-partition graph slot memory; models/slotgraph/) ──────
    # ICAE write (own frozen base + encoder-LoRA, M slots appended to the passage, run through the
    # LM's OWN layers) with a FIXED node/edge partition (slots 0..K-1 nodes, K..M-1 edges; see
    # slotgraph_n_nodes). Each EDGE slot predicts HARD (straight-through) which two NODE slots it links
    # (masked to the node pool → every edge is node→node). READ = a multi-hop residual message-passing
    # GNN over the predicted graph (node state RMSNorm-bounded each hop, ungated; zero-init update for a
    # plain-read cold-start), so the prepended memory is a FUNCTION of the topology and the endpoint
    # heads get real loss gradient. use_structure=False ⇒ plain prepend (id-tagged ICAE control).
    # Magnitudes: bounded internally by RMSNorm + one measured output rescale to the embedding norm;
    # no scale coefficients/gates, no per-layer injection (see the encoder MAGNITUDE/GRADIENT POLICY).
    slotgraph_n_slots: int = 32          # M — slots = PREPEND tokens (matches baselines' M=32)
    slotgraph_n_nodes: int = 16          # K — slots 0..K-1 are NODES, K..M-1 are EDGES (FIXED partition).
                                         # Role is assigned by position, NOT predicted: "which slot is a
                                         # node" is then 100%-reliable from step 0 (a noisy role head made
                                         # the edges→nodes mask unreliable for most of training). Edges
                                         # always point into the fixed node pool. TokenGT-style: types are
                                         # GIVEN; only the connectivity (endpoints) + content are learned.
    slotgraph_lora_rank: int = 32        # encoder-LoRA rank (mixed block sets ~85 to match icae+MP params)
    slotgraph_lora_alpha: int = 64
    slotgraph_temp_init: float = 1.0     # straight-through surrogate temperature (gradient sharpness)
    slotgraph_use_structure: bool = True # False ⇒ plain prepend of the id-tagged slots = "id-tagged ICAE"
                                         # (slots still carry the fixed id tags; the TRUE pure-ICAE
                                         # control is the separate icae_baseline variant). The ablation.
    slotgraph_use_id: bool = True        # False ⇒ drop the fixed orthonormal id_embed from the slots (and
                                         # routing-head input) ⇒ pure-ICAE-via-same-code. The id-tag ablation:
                                         # does the FREE (0-param buffer) identity tagging beat plain ICAE?
    slotgraph_max_hops: int = 5          # cap on the adaptive MP hop count (over-smoothing / compute guard);
                                         # #hops = the predicted graph's DIAMETER (reachability saturation)
    slotgraph_d_key: int = 64            # query/key dim for content-addressed endpoint routing (edge
                                         # queries · node keys, then single-step Sinkhorn competition)

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
