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
    ctx_len: int = 1024                  # ACTUAL training context length (set by the trainer from
                                         # --mixed-ctx / --compress-len). Encoders that scale time
                                         # constants (e.g. slotgraph3 assoc-write decay half-life =
                                         # f(ctx_len/window)) derive from THIS, never a literal.
    contrastive_shuf_coef: float = 0.0
    # objective_mode — the 2026-07-02 objective ladder (the wall is the OBJECTIVE, not the arch):
    #   "plain":       CE_real only (loss-neutral wrt binding — the historical default).
    #   "contrastive": CE_real + objective_coef · InfoNCE over the IN-BATCH memory matrix — each
    #                  example's target must be explained best by its OWN memory vs all B−1 other
    #                  memories (1 encoder run, B rolled decoder reads via memory_override). Non-
    #                  saturating multi-negative upgrade of the legacy 1-negative softplus
    #                  (contrastive_shuf_coef, kept independent for reproducibility; setting both
    #                  is an error).
    #   "trajectory":  contrastive + GRPO on the discrete read expansion: sample grpo_samples
    #                  Gumbel-top-k edge sets from the routing logits, reward = per-example
    #                  binding advantage (CE_shuf − CE_real, no-grad reads), group-relative
    #                  advantage × log-prob REINFORCE on the ROUTER ONLY (hybrid: continuous
    #                  params keep the pathwise InfoNCE gradient; the discrete choice — which has
    #                  no honest gradient — gets the unbiased estimator).
    # rank_reward (2026-07-03 diagnosis): the emitted per-example memory collapses to ~rank-2 (a blur,
    # << the ~5-8 entity floor). Plain CE never CHARGES for rank (simplicity bias) → add an MCR²
    # coding-rate REWARD on the WITHIN-example memory tokens: R_i = ½·logdet(I + (d/(M·ε²))·ZᵀZ) per
    # example (Yu et al. 2020), maximized. Run on PLAIN CE it is BOTH the fix AND the (a-objective vs
    # d-write) discriminator: EM↑+rank↑ → write CAN bind, plain just never charged (a); rank↑ but EM
    # flat → write emits high-rank non-relational content (d). NOTE: bends avoid-aux-losses — justified
    # as it redefines what the objective charges for (like contrastive), not a regularizer patch.
    rank_reward_coef: float = 0.0        # weight of −R (0 = off); ~0.01-0.1 range
    rank_reward_eps: float = 0.5         # MCR² quantization ε² (scale set by unit-normed tokens)
    objective_mode: str = "plain"
    objective_coef: float = 0.5          # weight of the InfoNCE term (contrastive/trajectory)
    objective_inv_temp: float = 1.0      # inverse temperature on the ROW-STANDARDIZED InfoNCE logits
                                         # (2026-07-03 audit: raw mean-NLL margins at τ=1 give a near-
                                         # uniform softmax = dead gradient; logits are standardized per
                                         # example over rolls first, so 1.0 = unit-σ spread; raise to
                                         # sharpen — the CLIP/SimCLR regime is effectively 1/τ ≈ 5–10
                                         # PRE-standardization).
    grpo_samples: int = 4                # G rollouts per step (trajectory; audit: G≥4 — G=2 is fragile)
    grpo_coef: float = 1.0               # weight of the REINFORCE policy term, applied to the PER-
                                         # DECISION-scaled policy loss (logp/n_decisions; constant
                                         # divisor = pure coefficient, NOT the Dr.GRPO length bias)
    grpo_entropy_coef: float = 0.01      # entropy BONUS on the router distribution (A3C-standard
                                         # 0.01): entropy collapse is the documented failure mode of
                                         # group-relative policy gradients; Gumbel exploration alone
                                         # dies once logits sharpen.

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
    mae_mask_ratio: float = 0.85            # fraction of tokens masked in the forward (spec.mask_ratio overrides per-episode)

    # ── continuation objective ─────────────────────────────────────────────
    # Predict the next predict_len block at EACH streaming-window boundary (memory-through-256,
    # -512, …) so one episode tests memory at growing compression horizons. Auto-active when
    # window_size < total_len; False = single-shot (predict once at the full cutoff, old behavior).
    continuation_multi_horizon: bool = True

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
    # streaming-write retention probe (condrecon_bio): pin the queried key→value pair into this
    # encoder window (0 = first = max retention lag, distractors after; -1 = last = recency
    # baseline; None = any window). Ties to --window-size. See project_streaming_write.
    cond_recon_bio_query_window: Optional[int] = None

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
    mixed_gate_batches: int = 0        # mixed val: REAL/SHUF/OFF binding gate on the first N batches/task (0=off)

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

    # ── slotgraph2 (per-layer graph-transformer over ICAE-encoded input; PREPEND read; models/slotgraph2/) ─
    # Intermediate between simple slotgraph (ICAE one-shot) and the chain-furl idea. Fixed partition
    # (K nodes + M-K edges) at d=d_llama; per streaming window the frozen LM encodes the window, then
    # L graph-transformer layers rewrite the graph state (ADDITIVE-residual node/edge latents + per-layer
    # SOFT destination "paintbrush" vs learnable node-id keys, source fixed to the home node). Persists
    # across windows; prepend the M graph tokens.
    slotgraph2_n_slots: int = 32         # M = prepend budget (matches baselines' M=32)
    slotgraph2_n_nodes: int = 16         # K node slots; M-K edge slots (fixed partition)
    slotgraph2_n_layers: int = 4         # graph-transformer layers per write window
    slotgraph2_window: int = 256         # streaming window size (input tokens per write step)
    slotgraph2_d_key: int = 64           # endpoint query/key dim (edge queries · node-id keys)
    slotgraph2_heads: int = 4            # attention heads in the graph-transformer layer
    slotgraph2_recurrent: bool = False   # True ⇒ ONE shared layer applied L× (param-light; ~matches budget)
    slotgraph2_lora_rank: int = 32       # encoder-LoRA rank (frozen-LM input encoder)
    slotgraph2_lora_alpha: int = 64
    # write mixer: "gt" = 4 custom graph-transformer layers off the LM's final hidden (A; expressive but
    # from-scratch); "lm" = ICAE-style — the graph tokens ride INSIDE the frozen LM's forward per window so
    # the PRETRAINED attention does the message passing (B; steers the mechanism that empirically binds).
    slotgraph2_write: str = "lm"

    # ── slotgraph3 (compressed-implicit graph, EXPANDED edge read; models/slotgraph3/) ──────────
    # State = per node (node_latent, edge_latent) + fixed id, all at d=d_llama. Expand to explicit edge
    # tokens (sparsemax routing from edge_latents → φ(src,dst,edge) + endpoint ids) DURING the write (so
    # layers see the structure) and BEFORE the read (prepend top-k edges/node; NO raw slots). d kept = 576.
    slotgraph3_n_nodes: int = 16         # K nodes; state carries 2K latents
    slotgraph3_n_layers: int = 4         # write layers per streaming window
    slotgraph3_window: int = 256         # streaming window size
    slotgraph3_d_key: int = 128          # routing query/key dim (edge_lat query · edge_lat key)
    slotgraph3_heads: int = 4            # (unused in LM-write path; kept for config compat)
    slotgraph3_read_topk: int = 8        # edges kept per node → prepend = K × read_topk tokens (K=16 → 128)
    slotgraph3_lora_rank: int = 56       # encoder-LoRA rank (write mixer; bumped to reach ~7M matched)
    slotgraph3_lora_alpha: int = 112
    slotgraph3_write_expand: bool = True # True: materialize expanded edges in the WRITE context (mixer sees
                                         # structure). False: write over [window; slots] only, expand edges
                                         # for the READ prepend ONLY — routing gradient comes purely from the
                                         # binding read (strips the write-forward pooling attractor on A).
    slotgraph3_write: str = "lm"         # write mixer: "lm" = frozen SmolLM2 attention + enc-LoRA (pretrained
                                         # prior, ~7M matched). "custom" = frozen LM encodes the window (no grad)
                                         # → from-scratch _SG3Blocks mix graph tokens over [hiddens; graph]
                                         # (position-free) — "does a purpose-built graph mixer beat LM attention?"
    slotgraph3_gate_ids: bool = False    # soft-id: endpoint labels ride INSIDE the routing weight
                                         # (E = topv·(φ+ids)+role) → router gets gradient through the id
                                         # channel; weak edges stop emitting full-loudness labels.
    slotgraph3_st_leak: bool = False     # straight-through expansion: forward = exact hard top-k tokens
                                         # (context stays K·topk); backward = soft mixture with leak ε =
                                         # the router's out-of-top-k sparsemax mass (self-annealing) →
                                         # dense gradient to unselected edges at ZERO context growth.
    slotgraph3_edge_budget: int = 0      # >0: GLOBAL edge budget — materialize the strongest E edges of the
                                         # whole graph (flattened A top-E) instead of top-k per node. Keeps
                                         # the read at E tokens for ANY node count (per-node top-k would grow
                                         # K·k); important nodes may hold many edges, irrelevant ones zero.
                                         # Requires st_leak (backward = global A-mass leak). 0 = per-node.
    slotgraph3_route_key: str = "node"   # which latent provides routing q/k: "edge" (v1: edge_lat serves BOTH
                                         # routing and relation → gradients fight; router-stability pressure
                                         # drags relation content toward genericity) or "node" (K/V split:
                                         # route by node CONTENT — input-dependent by construction — and let
                                         # edge_lat carry pure relation semantics). DEFAULT "node" (2026-07-02
                                         # tokenization-geometry batch: pairs with edge_state="matrix").
    slotgraph3_write_layers: int = 0     # LM arm depth: 0 = graph tokens ride ALL layers (full ride);
                                         # N>0 = text runs the frozen prefix alone (no grad), graph tokens
                                         # splice in for the last N layers (+LoRA there) — depth-matched to
                                         # the custom arm at N=4, ~2× faster than the full ride.
    slotgraph3_read: str = "raw"         # "edges" (v1, LEGACY control): prepend φ-SYNTHESIZED edge tokens only
                                         # (id-SUM + φ — the literature-worst geometry; kept for A/B). "raw"
                                         # (DEFAULT, 2026-07-02): the literature-verdict read — prepend the
                                         # node latents THEMSELVES as content tokens + edge POINTER tokens,
                                         # each formed by CONCAT-then-PROJECT (TokenGT): tok_proj([content ‖
                                         # id_src ‖ id_dst ‖ type]) → d, ONE shared projection for node and
                                         # edge tokens so the id subspace lands in the same output directions
                                         # (attention id-matching by construction). Replaces the 3-way
                                         # additive superposition (content+id+role at mismatched scales —
                                         # √3 norm-inflation → attention-sink; frozen LM can't unmix a sum
                                         # it was never trained to read).
    uniform_mem_pos: bool = False        # decoder read: give ALL prepended memory tokens the SAME RoPE
                                         # position (0) so they're an unordered SET, mutually equidistant
                                         # from text — removes the arbitrary intra-memory ordering + the
                                         # RoPE distance bias (which memory a text token finds "closer").
                                         # Text keeps normal relative positions (1..T after the memory slot).
    rect_prepend_mask: bool = False      # KBLaM-style rectangular decoder mask: memory tokens attend only
                                         # to THEMSELVES (no memory↔memory mixing/blurring across layers);
                                         # text attends into memory normally. Applies to the prepend read.
                                         # (Correct for independent KV facts; WRONG for a graph — see
                                         # bidir_mem_attn. Kept as a control. Mutually exclusive with it.)
    bidir_mem_attn: bool = False         # Set-LLM read geometry: the M prepended memory tokens attend to
                                         # each other BIDIRECTIONALLY (an edge token can see BOTH endpoint
                                         # node tokens regardless of emission order; composition needs it);
                                         # text stays causal. Plain-causal-over-a-prepended-SET imposes a
                                         # spurious order through visibility — the one geometry the set-
                                         # invariance literature uniformly warns against. Default False
                                         # (shared decoder path — baselines keep their trained geometry);
                                         # slotgraph3 runs enable via --bidir-mem-attn.
    slotgraph3_edge_state: str = "matrix"  # "flat": edge_lat_i is ONE shared vector for all of node i's edges
                                         # (per-dst relation = implicit unbinding a read must learn — no bias
                                         # for it). "matrix" (DEFAULT 2026-07-02): the SAME 576 floats viewed
                                         # as a 24×24 associative map M_i; rel(i→j) = M_i·rel_key(node_j) →
                                         # per-PAIR relation code, structural per-edge specificity (TPR/
                                         # fast-weights bind-in-write). Needs square d (576 = 24² ✓).
    slotgraph3_boundary_tokens: bool = True  # wrap the memory prepend in learned on-manifold <mem_start>/
                                         # <mem_end> tokens (init mean_embed + emb_std·randn, norm-matched).
                                         # Every working frozen-LM injection marks the span explicitly
                                         # (Qwen <|vision_start/end|>, ICAE [AE]/[LM], gist mask-boundary);
                                         # a sub-scale added role vector does not. M grows by 2.
    # ── 2026-07-03 write-audit repairs (T1/T2; see research_graph_write_audit) ────────────
    slotgraph3_route_act: str = "softmax"   # routing activation over destinations. "softmax" (DEFAULT) +
                                         # train-time Gumbel noise on the logits (parameter-free explora-
                                         # tion; noise ~ the logit scale at init, signal outgrows it).
                                         # "sparsemax" (LEGACY): exact zeros = out-of-support edges get
                                         # EXACTLY zero gradient → support monotonically shrinks (the
                                         # dead-gradient ratchet behind rdiv pinned-from-step-1).
    slotgraph3_init_noise: bool = True   # per-FORWARD Gaussian sampling of the initial latents around the
                                         # learned init (Slot Attention): breaks slot symmetry — identical
                                         # deterministic slots + shared heads get identical gradients and
                                         # stay identical forever (effrank 2-4, hub collapse). Learned
                                         # per-dim σ (init = measured embed std); train-time only.
    slotgraph3_write_norm_match: bool = True  # scale graph tokens to the TEXT-hidden RMS at the write
                                         # splice (learned scalar × dynamic text-norm match). Measured
                                         # 400× mismatch at the last-4 splice (graph 1.6 vs text 641):
                                         # a slot's residual stream is swamped by attention output after
                                         # one layer → slot identity erased inside the write.
    slotgraph3_write_boundary_bidir: bool = True  # write-side dialect harmonization: <mem_start> before
                                         # the graph block + BIDIRECTIONAL attention among graph tokens
                                         # in the write mask (matches the read geometry; under causal,
                                         # expanded edges could never see the node slots at all).
    slotgraph3_edge_write: str = "assoc"  # EDGE-latent update rule (T3, 2026-07-03). "assoc" (DEFAULT):
                                         # keyed delta-rule associative write into the persistent 24×24
                                         # map — M_i ← (1−α)·M_i + g_i·(v − M_i·k)⊗k, where k = the
                                         # UNIT-normalized routing-weighted partner-key mixture (SAME
                                         # rel_key as the read → write-address = read-address; unit keys
                                         # give the exact-overwrite property M'k = v), v = rel_val(edge-
                                         # slot hidden), α = retention-biased decay (init 1/(2·n_windows)
                                         # = 1/8: half-life ≈ the full context). Files facts under
                                         # partner addresses instead of smearing all 576 dims; M_i
                                         # becomes accumulated WRITE HISTORY — state not derivable from
                                         # node content (the recomputed-edges-carry-zero-bits fix).
                                         # "slot" (LEGACY): the T2 gated interpolation via the edge-slot
                                         # hidden. assoc REQUIRES edge_state="matrix" + write_update=
                                         # "delta" (fails loudly otherwise).
    slotgraph3_write_update: str = "delta"  # slot update rule. "delta" (DEFAULT): per-slot CONTENT gate
                                         # (bias init +1.5 = write-open, Jozefowicz) + slot-attention
                                         # COMPETITIVE read of the window (softmax over SLOTS, zero-init
                                         # proj) + gated INTERPOLATION lat += g·(cand − lat) — error-
                                         # correcting (repeated same-content writes converge instead of
                                         # accumulating; kills the additive averaging fixed point).
                                         # "additive" (LEGACY): lat += sigmoid(β)·head(LN(hidden)).
    slotgraph3_layer_anchor: bool = False  # GCNII-style per-layer id/role RE-INJECTION (the "simple
                                         # version" ingredient, 2026-07-04). The id/role identity is
                                         # re-added at EVERY LM layer (not just at tokenization) on both
                                         # passes: WRITE — the node/edge SLOT identity (id_scale·nid +
                                         # role) is re-injected (norm-matched to the stream) after each
                                         # write-suffix layer, so the frozen LM's mixing can't smooth the
                                         # slots together (the 15→5 depth collapse); READ — the edge-token
                                         # identity (id_scale·(id_src+id_dst) + role) rides in mem_aux
                                         # ['prepend_struct'] and model.py re-injects it before every
                                         # decoder layer (existing reinforce hook). Anchors distinctness
                                         # THROUGH depth so φ(src,edge,dst) reads distinct endpoints.
                                         # REQUIRES read='edges' (additive-separable id) + write='lm' +
                                         # write_layers>0 (the manual suffix loop is where re-injection
                                         # lives). Learned scalar anchor_gate (init 0.1) scales the
                                         # re-injected unit-direction. Fails loudly on incompatible config.
    slotgraph3_custom_layers: int = 4    # custom write: # of from-scratch transformer blocks
    slotgraph3_custom_dff: int = 1152    # custom write: block FFN hidden (2×d=1152 → ~13M UNMATCHED capacity probe)
    slotgraph3_custom_heads: int = 9     # custom write: attention heads (d=576 / 9 = head_dim 64)

    # ── slotgraph4 (fixed-topology sparse edge-state slot graph; models/slotgraph4/) ─────────────
    # N node slots + a FIXED k-regular small-world edge-state tensor E:[N,k,d_e] (Watts-Strogatz, degree-
    # preserved). NO routing head: topology is fixed, only edge STATES + node ASSIGNMENT are learned. Input =
    # frozen-LM window hiddens; write = recurrent _SG4Block stack (fused self+cross, ReZero, competitive
    # assignment) + ONE propose→commit gated write per window (decoupled α/β, EntNet post-norm, NO delta rule);
    # read = PREPEND (node-centric + top-k salience edges). See docs/slotgraph4_design.md.
    slotgraph4_n_nodes: int = 24         # N node slots (small; N ≤ d/2 for orthonormal ids)
    slotgraph4_edges_per_node: int = 12  # k = WS out-degree (high for small N — nearly dense; storage=[N,k])
    slotgraph4_d_edge: int = 64          # d_e — compact edge state dim (keeps the [N,k,d_e] tensor cheap)
    slotgraph4_ws_beta: float = 0.5      # Watts-Strogatz rewiring prob (0=ring lattice, 1=random k-regular)
    slotgraph4_seed: int = 0             # topology RNG seed (the WS graph is a deterministic FIXED buffer)
    slotgraph4_window: int = 256         # streaming window size (input tokens per write step)
    slotgraph4_d_key: int = 128          # competitive-read / node-pool query/key dim
    slotgraph4_heads: int = 9            # write-block attention heads (d=576 / 9 = head_dim 64)
    slotgraph4_d_ff: int = 2304          # write-block FFN hidden (recurrent single block ≈ 7M matched — verify)
    slotgraph4_write_layers: int = 4     # write-stack depth (recurrent ⇒ ONE shared block applied this many times)
    slotgraph4_recurrent: bool = True    # True ⇒ one shared _SG4Block × write_layers (param-light, preferred)
    slotgraph4_read_topk: int = 32       # explicit edge tokens in the read (highest learned salience, soft-gated)
    slotgraph4_boundary_tokens: bool = True  # learned on-manifold <mem_start>/<mem_end> around the prepend
    slotgraph4_init_noise: bool = True   # per-forward Gaussian sampling of the initial latents (symmetry break)

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
