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

    # ── soft_pointer_graph (models/soft_pointer_graph/; docs/graph_v6.md) ───
    # Soft-pointer graph memory with a no-op-free, per-token read. Persistent
    # state N[K_node,d_node] + per-edge (q_src,q_dst [d_node], state [d_state]).
    spg_K_node: int = 128
    spg_K_edge: int = 196
    spg_d_node: int = 480
    spg_d_state: int = 480
    spg_d_updater: int = 640           # write-transformer token dim
    spg_updater_layers: int = 5
    spg_updater_heads: int = 16
    spg_d_read: int = 512             # fact-token / read dim
    spg_read_ffn_mult: int = 4        # prepend-read projection MLP expansion (d_read → mult·d_read → d_llama)
    spg_builder_mlp_hidden: int = 768  # post-FiLM residual MLP hidden
    spg_read_temperature: float = 0.3
    spg_node_gate_init_bias: float = 0.5
    spg_edge_gate_init_bias: float = 1.0
    spg_init_log_sigma: float = 0.0
    spg_film_hidden: int = 512

    # ── hierarchical_learned_vocab (Compression-by-Vocabulary; hlvocab) ─────
    # models/hierarchical_learned_vocab/; design: docs/compression_model_design.md.
    # v1 (use_graph=False) = nodes-only soft-clustering compressor. v2 (default) adds
    # multi-scale STDP edges (within+inter-layer) + sharp-softmax edge selection +
    # stateless TokenGT node-tokens + a dedicated causal graph reader.
    hlvocab_d_code: int = 256           # shared code space (vocabulary)
    hlvocab_nodes: tuple = (512, 256, 128)  # nodes per layer (low->high)
    hlvocab_top_k: int = 4              # perturbation sparsity (active nodes/token)
    hlvocab_m_max: int = 16             # max emitted node-tokens (>= max k)
    hlvocab_effective_k: float = 8.0    # target #active nodes at init -> route temp
    hlvocab_tap_layer: int = 6          # v1 single-tap (nodes-only ablation)
    # v2: one frozen-backbone hidden-layer tap per vocab scale (low→high). Each vocab
    # layer routes its OWN contextualized Llama tap (mixing+position), replacing the
    # per-token perturbation chain. len must == len(hlvocab_nodes).
    hlvocab_tap_layers: tuple = (6, 15, 24)
    hlvocab_use_graph: bool = True      # True = v2 full graph; False = v1 nodes-only
    hlvocab_edge_topP: int = 32         # node prefilter: edges among top-P active nodes/layer
    hlvocab_edge_cand: int = 48         # candidate edges after the cheap STDP-lift prefilter
    hlvocab_d_sel: int = 192            # context-aware edge-selector transformer width
    hlvocab_sel_layers: int = 2
    hlvocab_sel_heads: int = 4
    hlvocab_d_read: int = 192           # dedicated graph-reader width
    hlvocab_reader_layers: int = 2
    hlvocab_reader_heads: int = 4
    # emit read-out: "edge_query" (independent sharp-softmax slots, collapse-prone)
    # | "slotattn" (Slot-Attention competition — slots PARTITION candidates).
    hlvocab_emit: str = "edge_query"
    hlvocab_slot_iters: int = 3         # Slot-Attention refinement iterations (slotattn)

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

    # ── slotmem factorization experiments (control / Exp1 discreteness / Exp2b graph-write)
    slotmem_n_slots: int = 16           # M memory tokens (= slots / edges); k-sliced like the cohort
    slotmem_d_slot: int = 520           # control slot-attention width (param-matched ~6.1M)
    slotmem_iters: int = 3              # slot-attention refinement iterations
    slotmem_vocab_d_slot: int = 496     # Exp1 slot width (smaller to make room for the bank)
    slotmem_vocab_n: int = 512          # Exp1 node-bank size
    slotmem_vocab_entmax: float = 1.5   # Exp1 selection sparsity (1.0=softmax)
    slotmem_d_graph: int = 336          # Exp2b free-graph width; param-matched
    slotmem_n_edges: int = 16           # Exp2b edges (= M edge tokens)
    slotmem_write_layers: int = 3       # Exp2b parser depth
    slotmem_heads: int = 4
    slotmem_ffn_mult: int = 2

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42                  # wired in the trainer (torch/np/random) for reproducibility
