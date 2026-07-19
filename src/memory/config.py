"""Config dataclass for the memory / compression module.

Only fields consumed by the LIVE code path are kept here. The active line is the mixed
4-task objective on a frozen backbone; the models that read this config are: the
published baseline ports (icae / autocompressor / gisting / memoryllm / titans),
slotgraph (emergent-topology slots), the training-free h2o KV-eviction reference, and
the vanilla floor/ceiling. (The old `graph` relational parser was retired — slotgraph
supersedes it; beacon/ccm/biomem/vqicae were retired and removed 2026-07-11.)
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
    # grad_checkpoint_decode: activation-checkpoint the per-layer-KV read decode
    # (_prefix_kv_forward). The reconstruct task decodes the whole ~2048-token passage
    # with a 30-layer KV prefix and retains it for backward, so the per-layer-KV arms
    # (gisting/memoryllm) OOM at B=8 on 24GB. That path is hook-free ⇒ exact recompute.
    grad_checkpoint_decode: bool = True

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

    mixed_gate_batches: int = 0        # mixed val: REAL/SHUF/OFF binding gate on the first N batches/task (0=off)

    # ── slotgraph — THE graph memory (models/slotgraph/; docs/design/slotgraph_design.md) ──────────────
    # 96 node slots, NO edge tokens. ONE shared frozen LM (two rank-16 LoRAs: write-harvest + read-decode).
    # WRITE = harvest per-layer node attention. Each ordered pair has a unit semantic relation R[i,j]
    # (d_e-wide) plus scalar confidence C[i,j] in [0,1]; both recur across internal windows and update
    # from the input. C controls semantic writes, value-path feedback, graph messages, and pointer salience.
    # READ = prepend+bidir; pair state shapes node tokens but is never emitted as N² edge tokens.
    slotgraph_n_nodes: int = 96          # N node slots (= M read budget; dense N×N edges, no fixed topology)
    slotgraph_d_edge: int = 32           # semantic relation width; confidence is a separate scalar (not in d_e)
    slotgraph_window: int = 256          # streaming window size (input tokens per write/harvest step)
    slotgraph_write_layers: int = 6      # LM depth harvested for the write (last-N; later=more semantic +
                                         # bounds retained attention matrices. 0 = all 30 layers = OOM-prone).
    slotgraph_bptt_detach_every: int = 0  # detach persistent R/C/X every K committed windows (truncated BPTT).
    slotgraph_inject_harvest_only: bool = True   # MOVE 1 (default ON): run the per-layer a_ij recompute +
                                         # value-path edge injection ONLY on the harvested layers (last
                                         # write_layers), not all L — inject WHERE you harvest. Principled
                                         # (relations are a late-layer phenomenon; injecting into early local-
                                         # feature layers is the wrong representational level) AND the big speed
                                         # win (kills the 30-vs-6 flash attn-recompute asymmetry). Early-layer
                                         # injection is dropped — a semantic change, so keep the flag to A/B.
    slotgraph_live_read: bool = False    # OPTION B (faithful live read): read the memory as PREPENDED node
                                         # tokens whose node↔node attention is edge-MODULATED live inside the
                                         # decoder's own last-K self-attention (the write's injection reused at
                                         # read), with a bidir memory mask. Unlike the KV read (edges frozen into
                                         # KV) or the old prepend (edges baked once via a side message-pass, then
                                         # smeared), the edge state R/C is re-injected FRESH from the store at
                                         # every modulated layer → the relational structure is never smeared, and
                                         # "edges modify inter-node attention" is exercised in the DECODER's real
                                         # attention at read time. ≈ prepend cost (no materialize pass).
                                         # 0 = FULL BPTT (default): the end-of-episode loss traces credit all
                                         # the way back to window 0 — required for delayed-retention learning
                                         # (a window-0 write learning from a window-7 retrieval failure). Kept
                                         # feasible by SMALL batch; the newer stabilizers (a_nn renorm, tanh-
                                         # capped injection, unit R, bounded C) replace detach as gnorm control.
                                         # K>0 = truncate every K windows (the old band-aid; only if OOM).
    slotgraph_lora_rank: int = 16        # write-LoRA rank (encoder copy); read-LoRA is the decoder copy's
                                         # adapter — two frozen copies, NOT one shared base. The mixed-training
                                         # preset overrides write rank to 84 (~7M trainable, capacity-matched).
    slotgraph_lora_alpha: int = 32
    slotgraph_read_topk: int = 0         # explicit edge-pointer tokens in the read (0 = node-centric only)
    slotgraph_read_hops: int = 1         # graph-conv read hops (1 until rank confirmed; 2 enables two-hop reach)
    slotgraph_d_key: int = 128           # read-side node-pool / salience query dim
    slotgraph_init_noise: bool = True    # per-forward Gaussian slot init (symmetry break, Slot Attention)
    slotgraph_diverse_node_init: bool = False  # ANTI-OVERSMOOTHING: init the N node slots to N DISTINCT
                                         # embedding vectors (on-manifold, high-rank, low-cosine) instead of one
                                         # shared mean + small noise (which starts near-collapsed at cos≈0.91).
                                         # Rationale: the nodes are re-attended over themselves 30 layers × ~9
                                         # forwards ≈ 270 self-mixing passes — a low-pass filter (Dong et al.
                                         # 2021 rank collapse). Starting UNCOLLAPSED gives the write distinct
                                         # content to preserve rather than create against the smear.
    slotgraph_id_reinject: bool = False  # IDENTITY RE-INJECTION (ROOT anti-collapse, primary fix): re-stamp a
                                         # fixed orthonormal per-node id into the node hiddens at EVERY LM layer
                                         # — like a positional encoding, re-applied not carried. Two tricks make
                                         # it robust: (1) stamped at a FRACTION of the content norm so it's never
                                         # swamped as content grows (finding-4 fix), (2) the prior id-component is
                                         # projected out first so it does NOT accumulate over 30 layers. Keeps the
                                         # slots DISTINCT → dense softmax addresses them selectively (the baseline
                                         # recipe: memoryllm/gisting bind with dense attn over distinct slots).
                                         # Content updates handle only content; identity is re-added. Makes the
                                         # entmax/mask changes optional. Pairs with slotgraph_diverse_node_init.
    slotgraph_id_strength: float = 1.0   # id-RMS : content-RMS ratio for the re-stamp. 1.0 = true RMS-MATCH
                                         # (id and content equal RMS → 50/50 energy split), the principled
                                         # zero-magic-number default. Frozen q/k can't reweight id-vs-content, so
                                         # this ratio is a real knob: tune DOWN (0.4/0.7) only if content needs
                                         # more room. (matched to each node's OWN content norm per-layer, so it's
                                         # scale-immune to Llama's large layer-varying hidden RMS.)
    slotgraph_sparse_relation: bool = False  # SPARSE ADDRESSING (anti-homogenization): replace the LM's DENSE
                                         # node↔node softmax topology (a_nn, ~uniform 1/N → pools all slots into
                                         # one address) with a DECOUPLED entmax-1.5 operator over the node
                                         # hiddens. entmax-1.5 gives exact zeros (each node relates to only a few
                                         # neighbors = a real sparse graph) WITH live gradients (unlike sparsemax's
                                         # dead-gradient ratchet). Sharpness lives in the learnable rel_q/rel_k
                                         # projections, not a scalar temp. Feeds both the edge feature and the
                                         # value-path injection. Pair with slotgraph_diverse_node_init.
    slotgraph_rel_init_scale: float = 0.5  # rel_q/rel_k init = (1/√d)·scale. Sets INITIAL entmax sparsity.
                                         # NOTE the score std ≈ scale² (dot-product of two ~scale-std vectors,
                                         # ×d_e^-0.5), so support falls FAST: ×0.5≈56 neighbours/96, ×1≈12,
                                         # ×2≈3, ×3≈1.7 (near one-hot). 0.5 = moderate start; projections are
                                         # learnable so training sharpens from here. (Was 3.0 = accidental
                                         # near-hard random graph before the projections learn anything.)
    slotgraph_layer_pair_gap: int = 0    # inter-layer operator: 0 = consecutive (l,l+1); k>0 = distant (l,l+k)
    slotgraph_flash_harvest: bool = True # True: SDPA/flash forward + recompute ONLY the [N,S] node-query
                                         # attention for the harvest (no [S,S] eager matrix → the B=8 memory
                                         # fix). False: eager forward + output_attentions (reference path).
    slotgraph_kv_read: bool = False      # EXPERIMENT (v1): read the graph as a PER-LAYER-KV prefix instead of
                                         # prepend+bidir. finalize runs the final nodes through the LM node-block
                                         # (nodes-only, same per-layer C·R value-path injection as the write) and
                                         # captures each layer's node k/v → passive KV prefix. Edges shape the
                                         # node KV via the injection; they never become tokens. Isolates the
                                         # read-surface axis vs gisting/memoryllm at matched M=N read length.

    # ── h2o (Heavy-Hitter Oracle KV eviction; models/h2o/) ────────────────────
    # Training-free eval-only baseline. Keeps the M tokens with highest cumulative
    # attention-received score (sum over all layers/heads). Own frozen base copy.
    # No trainable encoder params; only the shared read-LoRA trains.
    h2o_n_budget: int = 0              # M KV slots to keep; 0 ⇒ n_flat_codes
    h2o_recent_ratio: float = 0.1      # fraction of M reserved for most-recent tokens (H2O local window)

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42                  # wired in the trainer (torch/np/random) for reproducibility
