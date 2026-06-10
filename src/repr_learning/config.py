"""Config dataclass for V2.1 representation learning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ReprConfig:
    """All hyperparameters for V2.1 representation learning.

    Default values match docs/v2.1_repr_learning.md.
    """

    # ── Backbone (LM the encoder injects memory into) ──────────────────────
    # Field is "llama_model" historically — any HF chat model works. Set this
    # to an Instruct/chat-tuned model to enable chat-template scaffolding
    # (see src/repr_learning/chat_template.py). The model loader detects
    # whether the tokenizer has a chat template and routes accordingly.
    llama_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    d_llama: int = 2048  # Llama-3.2-1B hidden size
    llama_vocab_size: int = 128_256
    # System prompt placed BEFORE the memory slot in the chat scaffold.
    # Only used when the backbone tokenizer has a chat template.
    system_intro_for_memory: str = (
        "You are a helpful assistant. The following text contains memories "
        "from a long document. Use only those memories to answer the user's question."
    )
    # When True, append the backbone's chat-end token (eot/im_end/etc.) to
    # the answer tokens so teacher-forced training learns to emit a clean
    # end-of-turn marker. Required for sensible AR decode behavior under
    # chat-template prompting.
    append_answer_eot: bool = True

    # ── Encoder ────────────────────────────────────────────────────────────
    d_enc: int = 512                # encoder transformer hidden size
    enc_n_layers: int = 2
    enc_n_heads: int = 4
    enc_ffn_dim: int = 1024
    enc_dropout: float = 0.0

    # ── Bottleneck (V2.1) ──────────────────────────────────────────────────
    n_nodes: int = 4096             # codebook size
    d_concept: int = 1024           # concept_id vector dim (slow weights, scoring queries)
    # v1e split: codebook+scoring stays wide (1024) for routing discrimination,
    # but per-edge state carried forward as memory is narrower. When >0, the
    # codebook lookup is projected down to this dim, modifier MLP operates
    # there, and projection-to-d_llama starts from this dim. When 0, fall
    # back to v0/v1d behavior where the carried state equals d_concept.
    d_node_state: int = 0           # 0 = no compression; v1e sets this to 128
    d_edge: int = 128               # edge_vec dim (per-window state)
    n_edges: int = 30               # K — edges per window
    # 30 edges × (128 src + 128 edge + 128 dst) = 11,520 floats per window,
    # matching the 16 × 725 = 11,600 bottleneck of the baselines to within 0.7%.
    edge_token_packing: Literal["triple", "fused"] = "fused"
    # "triple" → 3 tokens per edge (src, edge, dst) interleaved → 90 total
    # "fused"  → 1 token per edge (concat src+edge+dst, project once) → 30 total.
    #            Default in v1e. Cuts memory tokens 3×; Llama sees semantic
    #            units (whole relations) rather than role-tagged atoms.
    d_proj_hidden_v21_fused: int = 256  # hidden dim of the fused projection MLP

    # ── V2.1 modifier MLP ──────────────────────────────────────────────────
    # The modifier MLP takes (concept_id[id], src_q) → 1024-dim "modified
    # concept" via a residual delta. Lets V2.1 encode instance-level info
    # (specific names/numbers/symbols) on top of the categorical codebook
    # routing. Output Linear is zero-initialized so the model starts identical
    # to a no-modifier V2.1 and learns to specialize.
    d_modifier_hidden: int = 256

    # ── V2.1 projection MLP hidden dims ────────────────────────────────────
    # Slim, low-rank projections — V2.1 spends params on the modifier MLPs
    # and codebook instead of huge projection hidden layers. Each role gets
    # its own MLP (separate weights distinguish src/dst/edge roles).
    d_proj_hidden_v21_main: int = 128   # src/dst proj hidden (1024 → 128 → 2048)
    d_proj_hidden_v21_edge: int = 64    # edge proj hidden (128 → 64 → 2048)

    # ── Bottleneck (Baselines) ─────────────────────────────────────────────
    # All four baselines (A, B, MT, Mamba) match V2.1's pre-projection budget
    # of ~69,600 floats per chunk by using 96 slots × 725 dims each.
    n_flat_codes: int = 96          # baseline slot count (= V2.1 memory token count)

    # Baseline A: discrete codebook + flat slots, no edge structure.
    #   pre-projection: 96 × 725 = 69,600
    d_concept_baseline: int = 725   # A's codebook concept_id dim

    # Baseline B: continuous slots, no codebook ("A without quantization").
    #   pre-projection: 96 × 725 = 69,600
    d_continuous: int = 725         # B's continuous slot dim
    slot_iters: int = 3             # B: canonical Slot Attention refinement iterations (Locatello 2020, Alg. 1)

    # Baseline 4 (MT): per-token KV bank, query from unmasked positions,
    # top-96 retrieved as memory tokens. Tests "retrieve don't compress."
    #   pre-projection (after retrieval): 96 × 725 = 69,600
    d_mt_value: int = 725           # MT memory bank entry dim

    # Baseline 5 (Mamba): state-space-model encoder, per-token outputs
    # bottlenecked to d_recurrent then pooled 256 → 96.
    #   pre-projection (after pool): 96 × 725 = 69,600
    # Canonical Mamba (tranche-5): narrower + deeper + pre-norm RMSNorm
    # residual blocks (matches the official mamba_ssm Block structure).
    # NOTE: the QA trainer OVERRIDES d_mamba (train_repr_qa.py v5.5 block) — the
    # operative value lives there: 1280 × 4L ≈ 49.5M trainable, matched to
    # graph_v5's 48.6M (probed 2026-05-29). This dataclass default only applies
    # to scripts that build a bare ReprConfig(). Prior tranche-4 used d1792 × 2L
    # with NO per-block norm — a non-canonical, shallow config that handicapped
    # Mamba; a later override briefly ran d1792 × 4L (90.8M, ~2× graph — too big).
    d_mamba: int = 1280             # Mamba internal d_model (matches trainer override)
    mamba_n_layers: int = 4         # canonical depth (RMSNorm pre-norm); fits BS=8 sans grad-ckpt
    mamba_d_state: int = 16
    mamba_expand: int = 2
    d_recurrent: int = 725          # Mamba per-token bottleneck width

    # ── Routing / selection ────────────────────────────────────────────────
    # We use classification-style picking (NOT VQ-VAE quantization).
    # Forward: argmax over scores; backward: Gumbel-softmax + STE.
    selection_temperature: float = 0.5
    load_balance_coef: float = 0.01

    # ── Routing z-loss (V2.1 only) ─────────────────────────────────────────
    # Mixtral-style: penalize (logsumexp(scores))². Keeps the absolute logit
    # magnitude bounded, which prevents `score_log_scale` from running away
    # and avoids a routing-collapse death spiral where sharp picks reinforce
    # themselves faster than load_balance can resist.
    z_loss_coef: float = 1e-3

    # ── Modifier delta watchdog (V2.1 only) ────────────────────────────────
    # Soft clip: scale delta down so ‖delta‖ ≤ ratio · ‖concept‖. Set to 0
    # to disable. v1c showed clip_ratio=1.0 was 100% active and cut modifier
    # capacity by ~75% (v1b natural delta was 3-4× concept norm). v1c_v2
    # defaults to 0 (disabled); we just watch the metric.
    modifier_delta_clip_ratio: float = 0.0

    # ── Q-Former diversity loss ────────────────────────────────────────────
    # The Q-Former's learned queries collapsed in v1c (mem_dispersion ≈ 0.9
    # vs ≈ 0.3 without). Same failure mode we hit on continuous_baseline.
    # Fix recipe: orthogonal init + scaled squared-cos penalty on outputs.
    qformer_diversity_coef: float = 1.0      # base coef
    qformer_diversity_scale: float = 1000.0  # multiplier (matches B's recipe)

    # ── B (continuous_baseline) diversity scale ───────────────────────────
    # Multiplier on B's squared-off-diag-cos penalty. DEFAULT 0: canonical
    # Slot Attention (Locatello 2020) prevents slot collapse via stochastic
    # shared-Gaussian init + GRU update + iterative competition — it uses NO
    # diversity/orthogonality loss. The earlier non-zero scale was a
    # non-canonical crutch that distorted the baseline (forced orthogonality
    # real slots never impose). Re-enable as a SMALL tie-breaker only if a
    # retrain shows collapse (watch diversity_slots_raw telemetry).
    b_diversity_scale: float = 0.0

    # ── MT (memorizing_baseline) diversity scale ─────────────────────────
    # Default 0: canonical Memorizing Transformers (Wu et al. 2022) uses NO
    # diversity loss. Per-position retrieval naturally diversifies (different
    # decoding positions retrieve different keys), so the earlier penalty was
    # a non-canonical crutch. Re-enable small only if a retrain shows collapse.
    mt_diversity_scale: float = 0.0

    # ── Faithful Memorizing Transformers (mt_faithful) ───────────────────
    # The REAL Wu et al. (2022) mechanism (see src/repr_learning/mt_attention.py),
    # replacing the broken `memorizing_baseline`. A single decoder layer's
    # self-attention is augmented with a kNN read over a per-token (key, value)
    # datastore gathered from the context, blended into the local attention via
    # a learned per-KV-head sigmoid gate (the ONLY trainable param this baseline
    # adds — 8 scalars for Llama-3.2-1B).
    mt_layer: int = 8          # which Llama decoder layer hosts the kNN read
    mt_topk: int = 32          # k for top-k kNN retrieval per query
    mt_gate_init_bias: float = 0.0   # gate = sigmoid(bias); 0 → 0.5 (equal mix)

    # ── Role embeddings on V2.1 memory tokens ──────────────────────────────
    # Llama receives 96 memory tokens but has no built-in way to tell src
    # from edge from dst. A learned per-role bias (added at projection time)
    # gives Llama a positional/typed signal it can pattern-match on.
    use_role_embeddings: bool = True

    # ── Llama LoRA (v1d) ───────────────────────────────────────────────────
    # Manual LoRA implementation (no `peft` dep). Wraps target Linear layers
    # in Llama's attention with W + (B @ A)·scale; A is small-init random,
    # B is zero-init so output equals base at step 0. Only A, B are trained.
    # Standard Hu et al. (2021) recipe: q_proj + v_proj, rank=16.
    use_llama_lora: bool = False
    llama_lora_rank: int = 16
    llama_lora_alpha: int = 16     # scale = alpha / rank
    # Names of Linear submodules to wrap. Default targets attention Q+V.
    llama_lora_target_names: tuple = ("q_proj", "v_proj")

    # ── ICAE baseline (In-Context Autoencoder, Ge et al. ICLR 2024) ─────────
    # Encoder = a SEPARATE frozen base copy + its OWN encoder-LoRA + M slots
    # (option A, docs/emat_baselines_plan.md). Distinct from the decoder LoRA
    # above so the two adapters can't collide on a shared base.
    icae_lora_rank: int = 32
    icae_lora_alpha: int = 64       # scale = alpha / rank = 2.0
    icae_n_slots: int = 0           # 0 ⇒ fall back to n_flat_codes (matched budget)

    # ── CCM baseline (Compressed Context Memory, Kim et al. ICLR 2024) ──────
    # Recurrent compression via a conditional LoRA gated to fire ONLY on <COMP>
    # token positions (CCM's signature), recipe-faithful rank 8 / q,k,v,o.
    # Port reads the COMP tokens' last-layer hiddens as the M memory vectors.
    ccm_lora_rank: int = 8
    ccm_lora_alpha: int = 16        # scale = alpha / rank = 2.0
    ccm_lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    ccm_n_comp: int = 0             # <COMP> tokens per step; 0 ⇒ n_flat_codes
    ccm_fold: str = "merge"         # "merge" (1/t mean, fixed M) | "concat" (grows)

    # ── Beacon baseline (Activation Beacon, Zhang et al. BAAI, arXiv:2401.03462)
    # SEPARATE full beacon q/k/v projections per layer (cloned-init from base,
    # NOT LoRA — this is Beacon's distinguishing axis vs CCM) routed to beacon
    # token positions; interleaved beacons (one per α-unit); streaming concat.
    # Port reads beacon last-layer hiddens as the M memory vectors. Trainable is
    # heavy (~100M on 1B with q,k,v) — that's faithful; report it. Drop to
    # ("k","v") or fewer to shrink.
    beacon_param: tuple = ("q", "k", "v")   # which projections get a separate beacon copy
    beacon_ratio: int = 0                   # condensing ratio α (beacons/window=W/α); 0 ⇒ auto from n_flat_codes
    beacon_window: int = 0                  # 0 ⇒ use trainer window_size
    beacon_wrap_layers: tuple = ()          # layer indices to wrap; () ⇒ ALL layers (capacity knob)
    mae_mask_ratio: float = 0.85            # mae task: fraction of answer tokens masked in the forward

    # ── Activation efficiency ───────────────────────────────────────────────
    # Activation-checkpoint each streaming-write window so we don't retain all
    # n_windows of encoder activations for one backward. This is what makes the
    # windowed encoders (flat/continuous/MT) fit at chunk=8192/BS=8 — without it
    # 8 windows of per-window activations are held at once and OOM. Exact
    # gradients; trades recompute for ~per-window activation peak.
    grad_checkpoint_stream: bool = True
    # Gradient-checkpoint the Llama decoder forward (needed for the full-context
    # arm, which forwards the whole 8192-token context through Llama with grad).
    grad_checkpoint_llama: bool = False

    # ── Q-Former adapter (V2.1 only, optional) ─────────────────────────────
    # BLIP-2-style cross-attention adapter inserted between the V21 encoder's
    # projected memory tokens and Llama's input. Learned "Llama-side" queries
    # cross-attend to encoder memory; the output replaces the raw projected
    # tokens. Designed to bridge the gap between V2.1's structured edge
    # triplets and Llama's preferred input distribution.
    use_qformer_adapter: bool = False
    qformer_d_adapter: int = 320
    qformer_n_heads: int = 8
    qformer_n_layers: int = 1
    qformer_ffn_mult: int = 4         # FFN hidden = ffn_mult × d_adapter

    # ── Codebook regularization (V2.1 only) ────────────────────────────────
    # v0 showed codebook pairwise cos climbing to +0.46 over 30K steps —
    # codes drift toward homogeneity even with high coverage (19%). Two
    # countermeasures:
    #   1) Soft orthogonality penalty on a random subsample of codebook
    #      entries each step (full N² is prohibitive for N=4096).
    #   2) Periodic dead-code revival: codes never picked over a window
    #      get re-seeded from heavily-used codes + small noise.
    codebook_orth_coef: float = 0.01
    codebook_orth_subsample: int = 256   # rows of codebook sampled per step
    dead_code_revival_interval: int = 1000     # check every N steps
    dead_code_revival_warmup: int = 2000       # don't revive before this step
    dead_code_revival_window: int = 1000       # picks tracked over the last N steps
    dead_code_revival_noise_std: float = 0.01  # noise on re-seeded entry

    # ── Training data ──────────────────────────────────────────────────────
    window_size_min: int = 128
    window_size_max: int = 384
    window_size_median: int = 256
    # v0: fixed window. v1b: variable in [min_window_size, max_window_size].
    fixed_window_size: int = 256

    # ── Variable-length packing (v1b+) ─────────────────────────────────────
    # When use_variable_length=True, the trainer uses SentencePackedDataset:
    #   - FineWeb: pick a doc, take random-length window from random offset
    #     within it (no cross-document slicing).
    #   - Composite: pack consecutive complete passages with sep_token_id
    #     until the window is in [min_window_size, max_window_size].
    use_variable_length: bool = False
    min_window_size: int = 128
    max_window_size: int = 1024
    # Llama-3.2: no dedicated pad id. We reuse <|end_of_text|>=128001 for
    # padding in dataset tensors; attention masks mask these positions out.
    # NOTE: Llama-3.2-1B-Instruct uses <|eot_id|>=128009 for eos AND pad
    # (different from base Llama). The 128001 value here is still a valid
    # special token in the Instruct tokenizer — it just isn't its eos —
    # but as long as the attention mask is consistent, both work.
    pad_token_id: int = 128_001
    sep_token_id: int = 198      # newline — natural sentence/paragraph separator

    # ── Masking ────────────────────────────────────────────────────────────
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 0.9
    mask_span_min: int = 5
    mask_span_max: int = 15

    # ── Loss ───────────────────────────────────────────────────────────────
    # Reconstruction loss is per-token CE on masked positions.
    # No VQ commitment loss (we use classification, not VQ-VAE).
    # Just load-balance aux loss to prevent classification collapse.

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 5000
    eval_every: int = 1000

    # ── Plastic substrate (Exp 2) ──────────────────────────────────────────
    # Depth of the substrate stack. D=8 puts plastic at ~13M trainable
    # params (matching the 12.5-14.9M band of the other memory variants).
    plastic_depth: int = 8
    # When False, the plasticity controller receives surprise=0 instead of
    # the per-token Llama NLL. The extra Llama-on-context forward dominates
    # plastic's per-step cost (6× slowdown vs other variants), so keeping
    # it off matches step time. Re-enable for ablation studies.
    plastic_use_surprise: bool = False
    # Llama layer index at which to inject the per-position memory readout.
    plastic_inject_layer: int = 8

    # ── Graph substrate (Exp 1 v3) ────────────────────────────────────────
    # See docs/exp1_graph_baseline.md for design.
    # K_max=68, d_node=d_state=128. Bottleneck convention matches splat:
    # K_max·(2·d_node + d_state + 1) = 68·385 = 26,180 floats — the +1 is now
    # the `u` (pick-affinity EMA) scalar, replacing the v1/v2 learned
    # saliency_logit. Same float-count, different semantics.
    graph_K_max: int = 68          # edge budget
    graph_d_node: int = 128        # endpoint dim
    graph_d_state: int = 128       # edge state dim
    # Updater + projection scaling chosen to land at ~14-15M trainable params
    # (matches A/B/MT/Mamba's 12.5-14.9M band).
    graph_updater_layers: int = 4  # transformer-updater depth
    graph_d_updater: int = 384     # updater token dim
    # graph_d_proj_hidden was the v1/v2 projection MLP hidden dim. v3 uses
    # graph_readout_d_hidden (below) inside GraphReadout instead. Field
    # removed to avoid confusion — if you see it elsewhere, that code is stale.
    # v4 graph_baseline knobs.
    graph_u_decay: float = 0.5               # fast EMA for 4-window chunks (was 0.95; left u init-dominated)
    graph_gate_init_bias: float = 1.0        # gate init: sigmoid(-1) ≈ 0.27 (anchor-leaning, escapes saturation)
    graph_grace_windows: int = 4             # [unused in v4] kept for backward-compat with cfg loaders
    graph_max_overwrites_fraction: float = 0.05  # [unused in v4] recycle was removed
    graph_update_strength_scale: float = 4.0     # [unused in v4] alpha-from-cosine was replaced by GraphGate
    graph_readout_n_heads: int = 4           # cross-edge message-passing attention heads
    graph_readout_d_hidden: int = 512        # d_hidden inside the directional readout

    # ── Graph v5 (Exp 1 v5): shared node bank + soft-pointer edges ─────────
    # See src/repr_learning/graph_substrate_v5.py for the design.
    #
    # HONEST capacity accounting (revised 2026-05-27 post-audit):
    # Both N (bank) AND edges persist across windows. v5.4's MP readout
    # also EMITS K_node tokens directly. Both must count toward bottleneck:
    #     state = K_node·d_node + K_edge·(2·d_node + d_state)
    # At K_node=32, K_edge=57, d_node=d_state=128: 4,096 + 21,888 = 25,984.
    # Matches 26,100 baseline budget within 0.4%. Prior config (K_node=64,
    # K_edge=68) was 34,304 floats = 31% over budget; comparisons against
    # baselines at the old config were unfairly favorable to v5.
    #
    # K_proposal IS still encoder-internal (per-window scratch, discarded
    # after slot routing — does not persist, never reaches decoder).
    graph_v5_K_node: int = 32              # # shared node slots (counts toward bottleneck)
    graph_v5_K_edge: int = 57              # # persistent edges (counts toward bottleneck)
    graph_v5_K_proposal: int = 80          # # per-window candidate proposals (encoder-internal scratch)
    graph_v5_d_node: int = 128             # node vector dim
    graph_v5_d_state: int = 128            # edge state dim
    graph_v5_d_updater: int = 384          # edge updater token dim
    graph_v5_updater_layers: int = 4       # transformer-updater depth
    graph_v5_updater_n_heads: int = 16     # heads in edge updater AttnBlocks (matches v4.2)
    graph_v5_node_gate_init_bias: float = 0.5  # node write gate init → sigmoid(-0.5) ≈ 0.38
    graph_v5_edge_gate_init_bias: float = 1.0  # edge update gate init → sigmoid(-1) ≈ 0.27 (anchor)
    graph_v5_init_log_sigma: float = 0.0   # initial log σ for (μ, σ) of N/state/q init noise (σ=1.0)
    # Init τ for the trained soft pointer. Lowered 1.0 → 0.3 (2026-05-27 post-
    # audit): at τ=1.0, endpoint_cos_mean≈0.99 at init → MP routing degenerate
    # in early training. Starting sharper gives v5.4's MP readout meaningful
    # structure from step 0. Learnable τ can still drift up if model wants.
    graph_v5_read_temperature: float = 0.3
    # v5.3: trained soft pointer with K/V split. K_split projects N through
    # separate W_k (for scoring) and W_v (for aggregation) — Csordás et al.
    # 2019 fix for the DNC pathology where N playing both roles produces
    # flat noisy address distributions. Combined with learnable τ above.
    graph_v5_soft_pointer_kv_split: bool = True
    graph_v5_readout_n_heads: int = 4      # [legacy v5.3 GraphReadoutV5] cross-edge MP heads
    graph_v5_readout_d_hidden: int = 512   # [legacy v5.3 GraphReadoutV5] d_hidden
    # v5.4: message-passing readout (replaces v5.3's cross-edge attention readout).
    # T rounds of bipartite MP, Q/K/V split with K=N (stable address book) and
    # V=msg_buf (evolving content). Outputs K_node memory tokens to Llama.
    # T=4 chosen 2026-05-27: at K_node=64 with sharp pointers, 4-hop reach is
    # plenty for the multi-hop tasks in composite_v1.
    graph_v5_n_message_rounds: int = 4
    graph_v5_mp_d_hidden: int = 256        # msg_mlp hidden dim (2× d_node default)
    # Per-node mean normalization on aggregation (GAT/mean style). Without
    # this, hub nodes touched by N edges get N× larger agg than isolated
    # nodes → variance imbalance + hub-driven oversmoothing. Default ON
    # (added 2026-05-27 post-audit).
    graph_v5_mp_degree_normalize: bool = True
    # GCNII-style anchor strength: at each MP round, blend the updated buf
    # with the seed = W_init(N). (1-α)·updated + α·seed. Prevents msg_buf
    # from drifting away from node identity over T rounds. α=0 disables.
    # Default 0.1 (light touch — empirical sweet spot in GCNII literature).
    graph_v5_mp_anchor_strength: float = 0.1

    # ── Graph v6 (docs/graph_v6.md) ────────────────────────────────────────
    # Soft-pointer graph memory with a no-op-free, per-token read. Persistent
    # state N[K_node,d_node] + per-edge (q_src,q_dst [d_node], state [d_state]) —
    # same float accounting as graph_v5. Defaults match the v5 operative config
    # (274,944 substrate floats); re-match baselines once the arch is frozen.
    graph_v6_K_node: int = 128
    graph_v6_K_edge: int = 196
    graph_v6_d_node: int = 480    # widened 384→480 to MATCH the ports' 262,144-float memory
    graph_v6_d_state: int = 480    # (per-example state = C + content + a_accum + q_src/q_dst/state)
    graph_v6_d_updater: int = 640          # write-transformer token dim
    graph_v6_updater_layers: int = 5
    graph_v6_updater_heads: int = 16
    graph_v6_d_read: int = 512             # fact-token / read dim
    graph_v6_read_heads: int = 8           # multi-head cross-attention read heads
    graph_v6_read_ffn_mult: int = 4
    graph_v6_builder_mlp_hidden: int = 768  # post-FiLM residual MLP hidden
    graph_v6_read_temperature: float = 0.3
    graph_v6_node_gate_init_bias: float = 0.5
    graph_v6_edge_gate_init_bias: float = 1.0
    graph_v6_init_log_sigma: float = 0.0
    graph_v6_film_hidden: int = 512
    graph_v6_inject_layer: int = 13        # v6.1: late-layer inject (was 8 = mid-stack "conform
                                           # zone" where a wrong read flips the answer; 13/16 ≈
                                           # top-third "ignore zone", Ben-Artzy — a bad read is harmless)

    # ── graph_v7 READ mode (cross-attention read vs prepend) ───────────────
    # "prepend" (default, existing behavior): the K_edge memory tokens are
    # prepended to the decode sequence and the frozen Llama self-attention is
    # relied on to use them. Empirically it doesn't — the memory gradient
    # collapses (grad_norm_memory 56→0.15) and the write never learns.
    # "cross_attn": skip the prepend (M=0) and instead install a dedicated,
    # TRAINABLE multi-head cross-attention read at a few decoder layers. Each
    # decode token queries the graph's memory vectors; the result is
    # gated-added into Llama's residual stream so gradient flows back to the
    # graph encoder from step 0 (gate init NONZERO). graph_v7 only.
    graph_read_mode: str = "prepend"
    graph_read_n_layers: int = 4                       # informational (len of indices)
    graph_read_layer_indices: tuple = (3, 7, 11, 15)   # which decoder layers host a read
    graph_read_inner_dim: int = 512                    # per-read q/k/v projection width
    graph_read_n_heads: int = 8                        # cross-attention heads
    graph_read_gate_init: float = 0.1                  # learnable scalar gate init (NONZERO — load-bearing)

    # ── graph_v7 WRITE contextualization (encoder INPUT, not the read) ─────
    # "raw" (default, existing behavior): the graph substrate builds its memory
    # from RAW Llama token embeddings (context-free lookups). EVERY port
    # (ICAE/CCM/AutoComp/Beacon) instead writes from CONTEXTUALIZED hidden
    # states (input run through a frozen Llama → last_hidden_state). This flag
    # gives the graph the same option: when "contextualized", run the FROZEN
    # DECODER Llama over the context (one extra no_grad pass — no second base,
    # avoiding the ports' OOM) and feed its hidden states into route_enc/
    # content_enc IN PLACE of raw embeds (same [B,T,d_llama] shape, so the graph
    # pipeline — routing, co-activation C, TokenGT edge update, ⊙ bind, the
    # cross-attn read — is UNCHANGED). The Llama stays frozen: gradient does NOT
    # flow into the base; the graph's own route_enc/content_enc remain the
    # trainable write. Contextualization is WRITE-side ONLY — the decoder still
    # decodes the ORIGINAL raw embeds. Composes independently with
    # graph_read_mode. graph_v7 only.
    graph_write_context: str = "raw"             # "raw" | "contextualized"
    # 0 = full/all decoder layers → use last_hidden_state ("full contextualize").
    # N>0 = run only the first N layers and read hidden_states[N] ("lower attune"
    # mode — hidden_states[0] = embeds, [k] = after layer k; clamped to n_layers).
    graph_write_context_layers: int = 0

    # ── graph_v7 bind-early / unbind-late associative memory (HRR) ──────────
    # The diagnosis: graph_v7's content accumulator MEAN-POOLS token values into
    # shared type-atoms → per-example memory is near-constant across passages
    # (cross-passage cosine 0.999) → SHUF=REAL (no binding). The fix: tag each
    # token's value with its entity-KEY via an HRR (circular-convolution) bind
    # the instant BEFORE it pools, so each atom holds a recoverable superposition
    # (Plate 1995, Holographic Reduced Representations); UNBIND by the question's
    # key (circular correlation) at read. When True, the WRITE pools BOUND pairs
    # and a dedicated unbind READ replaces the cross-attn read; the structural
    # TokenGT/edge/materialize path is left in place but RECALL goes through the
    # unbind (the Hadamard ⊙ fact_builder is bypassed — not cleanly invertible).
    # REQUIRES graph_write_context=="contextualized" (the entity-key needs
    # Llama's contextualization). graph_v7_baseline only.
    graph_v7_bind: bool = False
    # NONZERO learnable scalar gate init for the unbind read (load-bearing — lets
    # gradient reach key_proj/value_proj/W_recover/route_enc from step 0).
    graph_v7_bind_gate_init: float = 0.1

    # ── graph_v7 substrate hyperparameters (hoisted from getattr-buried defaults) ──
    # Co-activation accumulator decay applied once per WINDOW (encode-once EMAT
    # favors a long-memory accumulator; the within-scope outer-product structure
    # is the load-bearing part regardless of cadence).
    graph_v7_decay: float = 0.98
    # Aux-loss weights (added directly via the graph_aux path; NOT re-scaled by
    # load_balance_coef). Edge competition (claim distinct (src,dst) pairs) and
    # atom decorrelation (keep the vocabulary spread). Aux-loss-as-fallback.
    graph_v7_competition_coef: float = 0.1
    graph_v7_decorr_coef: float = 0.1
    # Split routing/endpoint temperatures (learnable; these set the INIT).
    # Routing over normalized-cosine logits needs a SMALL tau to peak — a sharp
    # per-token assignment is what lets atoms specialize (SHUF=REAL root cause).
    # Endpoint materialization tolerates a larger tau (it pools over the active
    # set, not a single atom). Both are learnable nn.Parameters from these inits.
    graph_v7_route_tau_init: float = 0.1
    graph_v7_endpoint_tau_init: float = 0.3
    # FIXED top-k active set: the |active| atoms (by accumulated activation) that
    # endpoints may address. Replaces the relative-fraction mask (a > frac·amax),
    # which was budget-unstable. Used in BOTH materialize and the edge-update KV
    # mask so the two stay consistent.
    graph_v7_active_topk: int = 32
    # TokenGT updater hidden width (~48M trainable at 800, matches the hand-built
    # cluster) and the per-window scope (# tokens whose outer-product co-activation
    # is accumulated). Hoisted from getattr-buried defaults.
    graph_v7_d_updater: int = 800
    graph_v7_scope_size: int = 16

    # ── graph_v8 (corrected columnar V8; implementation: graph_substrate_v8.py) ──
    # The 2026-06-09 design-correction of v8: same N nodes at EVERY layer (concept
    # columns, positional upward writes — no write-time matching), per-token
    # input-driven routing + co-activation at every layer (no consolidation clock;
    # timescales = learnable per-layer decay ladder), per-NODE fusion proposals
    # (partners' HRR self-binds weighted by the node's coact column), keys AND
    # values delta-written, real NLL surprise, K/V-split cross-attn read
    # (same-layer K_ℓ/V_ℓ reads via shared routers; GraphV8SymReader).
    # High-capacity graph_v8 anchor: final read memory is
    # n_layers · n_nodes · 2 · d_mem = 3 · 1024 · 2 · 2048
    # = 12,582,912 floats. Baseline comparison runs should scale their memory
    # budgets to this number rather than using the old 128-token prepend default.
    graph_v8_d_mem: int = 2048
    graph_v8_n_nodes: int = 1024
    graph_v8_n_layers: int = 3          # persistent layers above L0 (4 incl. L0)
    graph_v8_chunk: int = 256           # chunkwise-parallel token batch (also ckpt unit)
    # LAYER-MATCHED splice points, one per MEMORY LAYER incl. L0 (same-layer K/V
    # read, 2026-06-10): WRITE-side routing for source layer i consumes encoder
    # hiddens at layers[i] (i=0..2: atoms, K1, K2); READ of memory layer ℓ hooks
    # the decoder at layers[ℓ] and routes over K_ℓ via router ℓ (router 3 over K3
    # is read-only — L3 never writes upward). (3, 6, 10, 14): atoms earliest
    # (most concrete), spacing so Llama digests each read, ≥1 layer + final norm
    # after the deepest read.
    graph_v8_reader_layers: tuple = (3, 6, 10, 14)
    graph_v8_reader_inner_dim: int = 224      # 7*32; v/o-only reader (routers are shared)

    # ── JEPA loss coefficients (dormant path; hoisted for hygiene) ──────────
    # VicReg variance/covariance anti-collapse weights on the online memory.
    jepa_var_coef: float = 5.0
    jepa_cov_coef: float = 0.5

    # ── Gaussian Splat substrate (Exp 3) ──────────────────────────────────
    # See docs/exp3_gaussian_splat_baseline.md for full design.
    # v3 sweep: K=100, d=128 → bottleneck K·(2d+2) = 25,800 floats
    # (matches 26,100 band of A/B/MT/Mamba). Roughly 2x more blobs vs v2
    # (was K=51) at half the per-blob width; tests whether more cells help
    # at the cost of narrower per-blob representation.
    splat_K: int = 100             # number of signed Gaussian blobs (fixed)
    splat_d: int = 128             # latent space dimensionality
    splat_K_rays: int = 16         # ray probes per Llama position at read time
    splat_updater_layers: int = 3  # TransformerUpdater cross+self-attn depth
    splat_inject_layer: int = 8    # Llama layer to install the read pre-hook
    # Auxiliary loss coefficients. v3 raises λ_adj 10× (v2 had L_adj=47.9 at
    # convergence — blobs never stabilized) and λ_sat 50× (signs were
    # saturating with old λ=0.001 contributing essentially nothing).
    splat_alpha_pin: float = 0.1
    splat_beta_prop: float = 0.1
    splat_lambda_adj: float = 0.5
    splat_lambda_sat: float = 0.05

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    # ── Validation ─────────────────────────────────────────────────────────
    def __post_init__(self):
        assert 0.0 <= self.mask_ratio_min <= self.mask_ratio_max <= 1.0
        assert self.mask_span_min <= self.mask_span_max
        assert self.window_size_min <= self.window_size_median <= self.window_size_max
        # v0-v1d: V2.1 (3 × n_edges memory tokens) was matched to baselines
        # (n_flat_codes memory tokens). v1e relaxes this — bottleneck width
        # is measured in pre-projection floats per window, not token count.

    @property
    def d_node_carried(self) -> int:
        """Per-edge carried state dimension. d_node_state if set, else d_concept."""
        return self.d_node_state if self.d_node_state > 0 else self.d_concept

    @property
    def bottleneck_floats_v21(self) -> int:
        """V2.1 pre-projection bottleneck width per window."""
        return self.n_edges * (2 * self.d_node_carried + self.d_edge)

    @property
    def bottleneck_floats_baseline(self) -> int:
        """Baselines (A/B/MT/Mamba) pre-projection bottleneck width per window."""
        # All four use n_flat_codes × per-slot dim. Per-slot dim varies by variant
        # but is set to d_concept_baseline = d_continuous = d_mt_value = d_recurrent
        # by convention. Take d_continuous as representative.
        return self.n_flat_codes * self.d_continuous

    @property
    def bottleneck_floats_graph_v6(self) -> int:
        """graph_v6 persistent substrate: node bank + per-edge (q_src, q_dst, state).
        Same accounting as graph_v5 (the two soft-pointer queries are the 2·d_node term)."""
        return (self.graph_v6_K_node * self.graph_v6_d_node
                + self.graph_v6_K_edge * (2 * self.graph_v6_d_node + self.graph_v6_d_state))

    @property
    def n_memory_tokens(self) -> int:
        """Total memory tokens prepended to Llama input. Variant-specific:
        - V2.1 with triple packing: 3 × n_edges (96 default)
        - V2.1 with fused packing:  n_edges     (32 default)
        - Baselines: n_flat_codes (16 default in v1e)
        """
        return self.n_flat_codes  # baseline default; V2.1 overrides via its packing
