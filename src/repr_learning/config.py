"""Config dataclass for V2.1 representation learning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ReprConfig:
    """All hyperparameters for V2.1 representation learning.

    Default values match docs/v2.1_repr_learning.md.
    """

    # ── Llama backbone ─────────────────────────────────────────────────────
    llama_model: str = "meta-llama/Llama-3.2-1B"
    d_llama: int = 2048  # Llama-3.2-1B hidden size
    llama_vocab_size: int = 128_256

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

    # Baseline 4 (MT): per-token KV bank, query from unmasked positions,
    # top-96 retrieved as memory tokens. Tests "retrieve don't compress."
    #   pre-projection (after retrieval): 96 × 725 = 69,600
    d_mt_value: int = 725           # MT memory bank entry dim

    # Baseline 5 (Mamba): state-space-model encoder, per-token outputs
    # bottlenecked to d_recurrent then pooled 256 → 96.
    #   pre-projection (after pool): 96 × 725 = 69,600
    d_mamba: int = 1024             # Mamba internal d_model
    mamba_n_layers: int = 2
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
    # Multiplier on B's squared-off-diag-cos penalty inside finalize_memory
    # and forward. Held at 1000 for v1e/v1f back-compat; v1g lowered it
    # after observing aux stuck at ~2000 (62% of total loss) on 500-step
    # smoke. Lower scale frees gradient budget for recon and lets the
    # competing slot-attn vs diversity balance settle to a new equilibrium.
    b_diversity_scale: float = 1000.0

    # ── MT (memorizing_baseline) diversity scale ─────────────────────────
    # Same penalty applied to MT's retrieved memory tokens to fight slot
    # collapse. Default 1000 for v1e back-compat; v1g lowers to 50 same
    # as B for the same reason (otherwise aux dominates recon).
    mt_diversity_scale: float = 1000.0

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
    # Llama-3.2: pad has no dedicated id; eos=128001 works (attention masked).
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

    # ── Gaussian Splat substrate (Exp 3) ──────────────────────────────────
    # See docs/exp3_gaussian_splat_baseline.md for full design.
    # K=51, d=256 → bottleneck K·(2d+2) = 26,214 floats (matches 26,100
    # band of A/B/MT/Mamba prepend variants).
    splat_K: int = 51              # number of signed Gaussian blobs (fixed)
    splat_d: int = 256             # latent space dimensionality
    splat_K_rays: int = 8          # ray probes per Llama position at read time
    splat_updater_layers: int = 3  # TransformerUpdater cross+self-attn depth
    splat_inject_layer: int = 8    # Llama layer to install the read pre-hook
    # Auxiliary loss coefficients (importance weights — sublosses are
    # internally normalized so these are pure relative-importance scalars).
    splat_alpha_pin: float = 0.1
    splat_beta_prop: float = 0.1
    splat_lambda_adj: float = 0.05
    splat_lambda_sat: float = 0.001

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
    def n_memory_tokens(self) -> int:
        """Total memory tokens prepended to Llama input. Variant-specific:
        - V2.1 with triple packing: 3 × n_edges (96 default)
        - V2.1 with fused packing:  n_edges     (32 default)
        - Baselines: n_flat_codes (16 default in v1e)
        """
        return self.n_flat_codes  # baseline default; V2.1 overrides via its packing
