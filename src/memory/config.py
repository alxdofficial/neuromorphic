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

    # ── slotgraph — THE graph memory (models/slotgraph/; docs/slotgraph_design.md) ──────────────
    # 96 node slots, NO edge tokens. ONE shared frozen LM (two rank-16 LoRAs: write-harvest + read-decode).
    # WRITE = harvest the LoRA'd LM's per-layer node-block attention a_ij; inject a persistent per-edge
    # state E[i,j] (d_e-vec) on the VALUE path as an additive residual U*(sum_j a_ij E[i,j]). E is a
    # streaming state (across windows), init ZERO, updated by an error-correcting, per-edge-gated,
    # EntNet-bounded commit of a within-window inter-layer diff+product trace. READ = prepend+bidir,
    # E shapes the node tokens (graph-conv blend + learned-salience pointers), not tokenized.
    slotgraph_n_nodes: int = 96          # N node slots (= M read budget; dense N×N edges, no fixed topology)
    slotgraph_d_edge: int = 32           # d_e — per-edge state width (heads×timescales + content proj; keep small)
    slotgraph_window: int = 256          # streaming window size (input tokens per write/harvest step)
    slotgraph_write_layers: int = 6      # LM depth harvested for the write (last-N; later=more semantic +
                                         # bounds retained attention matrices. 0 = all 30 layers = OOM-prone).
    slotgraph_bptt_detach_every: int = 1  # detach persistent E/X every K committed windows (truncated BPTT).
                                         # 1 = per-window state (design granularity); K>1 keeps some cross-
                                         # window credit; 0 = full 8-deep recurrence (gnorm-compounding).
    slotgraph_lora_rank: int = 16        # write-LoRA + read-LoRA rank (two adapters on the shared base)
    slotgraph_lora_alpha: int = 32
    slotgraph_read_topk: int = 0         # explicit edge-pointer tokens in the read (0 = node-centric only)
    slotgraph_read_hops: int = 1         # graph-conv read hops (1 until rank confirmed; 2 enables E^2 reach)
    slotgraph_d_key: int = 128           # read-side node-pool / salience query dim
    slotgraph_init_noise: bool = True    # per-forward Gaussian slot init (symmetry break, Slot Attention)
    slotgraph_layer_pair_gap: int = 0    # inter-layer operator: 0 = consecutive (l,l+1); k>0 = distant (l,l+k)

    # ── h2o (Heavy-Hitter Oracle KV eviction; models/h2o/) ────────────────────
    # Training-free eval-only baseline. Keeps the M tokens with highest cumulative
    # attention-received score (sum over all layers/heads). Own frozen base copy.
    # No trainable encoder params; only the shared read-LoRA trains.
    h2o_n_budget: int = 0              # M KV slots to keep; 0 ⇒ n_flat_codes
    h2o_recent_ratio: float = 0.1      # fraction of M reserved for most-recent tokens (H2O local window)

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
