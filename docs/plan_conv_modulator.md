# Implementation Plan — Conv-Grid Modulator

Companion to `docs/design_conv_modulator.md`. Ordered steps, file-by-file
changes, test targets, and risk/mitigation.

## Goal

Convert the current NC=8 × N=32 cell-grid memory with a per-cell MLP modulator
into a NC=1 × N=256 single-pool design with a conv-over-edge-grid modulator,
while preserving:
- The `Model.forward_chunk` interface to the rest of the codebase
- The LM integration (H_mid → memory → readout, mem_scale, mem_pred_head)
- TBPTT semantics, Gumbel-softmax training, codebook + decoder structure
- GRPO readiness for future work (single categorical per event)

## Branch state

Starting from `main` (commit `daefce9`). Branch `conv-grid-modulator` is the
work branch. No existing commits on this branch yet beyond the design +
plan docs.

## Implementation order

Steps are ordered so each one leaves the code in a runnable state (on the
tier_tiny config at least). **No step should land without its test passing.**

### Step 1 — Config schema

File: `src/model/config.py`

Change dataclass fields:
- Add `N_total: int = 256` (was `N_cells × neurons_per_cell`).
- Add `NC_pools: int = -1` (derived: `D // D_n`; was `N_cells`).
- Add `d_proj: int = 16` (node feature compression dim for modulator input).
- Add `conv_channels: int = 192` (encoder conv hidden width).
- Add `conv_layers: int = 6`.
- Add `conv_kernel: int = 7`.
- Add `conv_groups: int = 32` (GroupNorm groups; shared between encoder
  and decoder for uniformity).
- Add `conv_dropout: float = 0.1` (dropout between conv layers in encoder).
- Add `decoder_seed_spatial: int = 4`.
- Add `decoder_seed_channels: int = 256` (decoupled from D_code).
- Add `gamma_max: float = 0.97` (bf16-safe ceiling on plasticity γ).
- Add `role_dim: int = 4`.
- Add `checkpoint_decoder: bool = False` (activation-ckpt flag for decoder;
  mirrors existing `checkpoint_memory` for the memory loop).
- Update `num_codes: 512 → 4096`, `code_dim: 64 → 384`.
- Remove: `action_rank`, `decoder_hidden`, `decoder_groups` (separate from
  `conv_groups`).
- Deprecate `N_cells`, `neurons_per_cell`, `K` (initial W sparsity — no longer
  applicable with single-pool dense W). Leave for one cycle of back-compat if
  convenient; delete in step 10.
- Update `validate()`: derive `NC_pools = D // D_n`, assert `N_total >=
  2 * alpha * NC_pools + 1` (pool ports + at least one internal).
- Update `mod_in` / `mod_out` computed properties (`mod_in` is no longer used
  in the same way; new conv encoder has a different input contract). Keep
  or delete? Delete after step 4.
- Update `tier_tiny` classmethod to use new fields, test config.

**Test**: `Config.tier_a()` and `tier_tiny()` both validate. Unit test
verifies derived fields.

### Step 2 — Memory runtime state refactor

File: `src/model/memory.py`

Collapse NC dimension in runtime state tensors:
- `h`: `[BS, NC, N, D_n] → [BS, N, D_n]` where `N = N_total`.
- `msg`: same.
- `W`: `[BS, NC, N, N] → [BS, N, N]`.
- `hebbian`: same.
- `decay`: `[BS, NC, N] → [BS, N]`.
- `prev_readout`: stays `[BS, D]`.
- `readout_drift`: `[BS, NC, 1] → [BS, NC_pools, 1]` (drift is still per-pool,
  since readout is per-pool).

Update:
- `initialize_states` to new shapes.
- `detach_states`, `runtime_state_dict`, `load_runtime_state`, `resize_to_bs`
  to match.
- `compute_lane_divergence` — simpler now, no NC axis.
- `_receive = W @ msg`: now a single `[BS, N, N] @ [BS, N, D_n]` matmul.
- `_inject`: iterate over `NC_pools`, distribute H_mid slices to each pool's
  input ports using `input_port_indices[p]` slicing.
- `_readout`: iterate over pools, sum output ports per pool, concat.
- `_hebbian_update`: single matmul, no NC dim.

**Keep (unchanged semantics)**:
- `forward_segment` signature.
- The per-block TBPTT loop (`_run_block`).
- The per-token surprise computation (no_grad).
- `mem_pred_loss` computation.
- `state_mlp`, `msg_mlp` as shared MLPs over flattened [BS·N, D_n].

**Delete (or deprecate then delete)**:
- `compute_modulator_stats` — still useful but needs rewrite for new modulator.
- `_modulate_cells` — replaced by new module.

**Test**: end-to-end forward pass on tier_tiny config runs without shape
errors. Compare output shapes to current values (readouts shape must still
be `[BS, T, D]`).

### Step 3 — Port-pool indexing helpers

File: `src/model/memory.py` (new helpers)

```python
def _compute_port_indices(self) -> dict:
    """Build index tensors for pool-based inject/readout.

    NC_pools pools, alpha input ports + alpha output ports per pool.
    Layout: [pool0 input ports, pool0 output ports, pool1 input, ...].
    Internal neurons follow: [2*alpha*NC_pools, N_total).
    """
    ...
```

Produces:
- `input_port_idx[p]`: tensor of indices for pool p's input port neurons.
- `output_port_idx[p]`: tensor of indices for pool p's output port neurons.
- `role_id[i]`: tensor `[N]` in {0=input, 1=output, 2=internal}.

**Test**: `port_indices` partition [0, N) correctly; no duplicates; roles
match layout.

### Step 4 — Conv-grid modulator module

New file: `src/model/grid_modulator.py`

```python
class ConvGridModulator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.N = config.N_total
        self.D_n = config.D_n
        self.d_proj = config.d_proj
        C_h = config.conv_channels
        k = config.conv_kernel
        pad = k // 2

        # Three node-feature projections (h, msg_emit, msg_recv)
        self.h_proj = nn.Linear(D_n, d_proj, bias=False)
        self.msg_emit_proj = nn.Linear(D_n, d_proj, bias=False)
        self.msg_recv_proj = nn.Linear(D_n, d_proj, bias=False)
        self.role_emb = nn.Embedding(3, config.role_dim)

        # Input channel count — see design doc observation section
        # 3 raw edge + 4·d_proj node + 2·role_dim roles + 1 decay + 2 global
        C_in = 3 + 4*d_proj + 2*role_dim + 1 + 2

        # Stem: project observation channels to conv width
        self.stem = nn.Conv2d(C_in, C_h, kernel_size=k, padding=pad)
        self.stem_norm = nn.GroupNorm(config.conv_groups, C_h)

        # Residual conv blocks (pre-norm). Layers 2..L_conv.
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.GroupNorm(config.conv_groups, C_h),
                'conv': nn.Conv2d(C_h, C_h, kernel_size=k, padding=pad),
            }) for _ in range(config.conv_layers - 1)
        ])
        self.dropout = nn.Dropout2d(config.conv_dropout)

        # Pooling head → code logits
        self.logit_head = nn.Linear(C_h, config.num_codes)

    def build_input(self, h, msg, received, W, hebbian, decay,
                     s_live, s_ema, role_id) -> Tensor:
        """Build [BS, C_in, N, N] edge feature map in bf16."""
        BS, N, _ = h.shape
        dt = h.dtype

        h_p = self.h_proj(h)                          # [BS, N, d_proj]
        me_p = self.msg_emit_proj(msg)                # [BS, N, d_proj]
        mr_p = self.msg_recv_proj(received)           # [BS, N, d_proj]
        role_e = self.role_emb(role_id).to(dt)        # [N, role_dim]

        # Broadcast per-neuron features into grid (receiver down col, sender across row)
        def bcast_row(x):   # [BS, N, F] → [BS, N, N, F] broadcasting along j axis
            return x.unsqueeze(2).expand(BS, N, N, x.shape[-1])
        def bcast_col(x):   # [BS, N, F] → [BS, N, N, F] broadcasting along i axis
            return x.unsqueeze(1).expand(BS, N, N, x.shape[-1])

        W_ij = W.unsqueeze(-1)                         # [BS, N, N, 1]
        heb_ij = hebbian.unsqueeze(-1)                 # [BS, N, N, 1]
        asym = (W - W.transpose(-1, -2)).unsqueeze(-1) # [BS, N, N, 1]

        s_live_b = s_live.view(BS, 1, 1, 1).expand(BS, N, N, 1)
        s_ema_b  = s_ema.view(BS, 1, 1, 1).expand(BS, N, N, 1)

        channels = torch.cat([
            W_ij, heb_ij, asym,                        # 3 raw edge
            bcast_row(h_p), bcast_col(h_p),            # 2·d_proj node
            bcast_row(me_p), bcast_row(mr_p),          # 2·d_proj msg (per receiver)
            bcast_row(role_e), bcast_col(role_e),      # 2·role_dim
            bcast_row(decay.unsqueeze(-1)),            # 1 decay (receiver)
            s_live_b, s_ema_b,                          # 2 global
        ], dim=-1)                                      # [BS, N, N, C_in]

        return channels.permute(0, 3, 1, 2).contiguous()  # [BS, C_in, N, N]

    def encoder_forward(self, x: Tensor) -> Tensor:
        """Pre-norm residual conv stack → pooled cell feature."""
        x = F.gelu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            h = block['norm'](x)
            h = F.gelu(block['conv'](h))
            h = self.dropout(h)
            x = x + h                                    # residual
        pooled = x.mean(dim=(2, 3))                       # [BS, C_h]
        return pooled

    def forward(self, h, msg, received, W, hebbian, decay,
                s_live, s_ema, role_id) -> Tensor:
        E = self.build_input(h, msg, received, W, hebbian, decay,
                              s_live, s_ema, role_id)
        pooled = self.encoder_forward(E)
        logits = self.logit_head(pooled)                 # [BS, K]
        return logits
```

**Test**: forward a random input, assert output shape `[BS, K]`; gradient check
(backward without errors); memory footprint at BS=72 N=256 under 1 GB for the
build_input tensor.

### Step 5 — Discrete policy refactor + conv-transpose decoder

File: `src/model/discrete_policy.py`

Change shape assumptions:
- Input: `logits: [BS, K]` (was `[BS, NC, K]`).
- `codes: [BS]` (was `[BS, NC]`).
- `soft: [BS, K]` (was `[BS, NC, K]`).
- `log_pi: [BS]`.

The logit-computation path moves OUT of this file (now done by ConvGridModulator).
`DiscreteActionPolicy` becomes a thinner module: codebook + decoder + sampling
primitives. Keep:
- `sample_discrete`, `sample_gumbel_soft`, `log_prob`, `entropy`,
  `update_usage`, `reset_dead_codes`.
- `decode` / `decode_soft` (redesigned — they now return `ΔW_raw [BS, N, N]`
  and `Δdecay_raw [BS, N]` instead of a flat action vector).

Remove:
- `compute_logits` (moved to `ConvGridModulator`).
- `logit_w1/b1/w2/b2` parameters.
- Old dense-MLP decoder params (`dec_w1`, `dec_b1`, `dec_w2`, `dec_b2`).

New decoder: **conv-transpose generator** (pre-norm residual, resize+conv).

```python
class ConvTransposeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = config.N_total
        D_code = config.code_dim
        S = config.decoder_seed_spatial         # 4
        C0 = config.decoder_seed_channels        # 256

        self.N = N
        # Start projection: code_emb → spatial seed
        self.init_proj = nn.Linear(D_code, S * S * C0)

        # Upsample stages. Channel ladder: 256 → 128 → 96 → 64 → 48 → 32 → 32.
        # Each stage: upsample → pre-norm residual conv block.
        channel_ladder = [C0, 128, 96, 64, 48, 32, 32]
        self.stages = nn.ModuleList()
        for c_in, c_out in zip(channel_ladder[:-1], channel_ladder[1:]):
            self.stages.append(nn.ModuleDict({
                'norm':    nn.GroupNorm(config.conv_groups, c_in),
                'conv':    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                # 1×1 projection for residual if channels mismatch
                'proj':    (nn.Conv2d(c_in, c_out, kernel_size=1)
                            if c_in != c_out else nn.Identity()),
            }))

        # Final 1×1 head → ΔW_raw (ZERO-INIT — critical)
        self.dW_head = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.zeros_(self.dW_head.weight)
        nn.init.zeros_(self.dW_head.bias)

        # Δdecay head: row-pool the feature map → per-neuron MLP
        self.decay_head = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1),
        )
        # Diagonal mask (cached buffer)
        self.register_buffer('diag_mask', torch.eye(N).unsqueeze(0))

    def forward(self, emb: Tensor) -> tuple[Tensor, Tensor]:
        """emb: [BS, D_code] → (ΔW_normed [BS, N, N], Δdecay_raw [BS, N])"""
        BS = emb.shape[0]
        x = self.init_proj(emb).reshape(BS, -1, 4, 4)     # seed

        for stage in self.stages:
            y = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=False)
            h = F.gelu(stage['conv'](stage['norm'](y)))
            x = h + stage['proj'](y)                       # residual over upsampled input
        # x: [BS, 32, N, N]

        # ΔW head
        dW_raw = self.dW_head(x).squeeze(1)                # [BS, N, N]
        dW_raw = dW_raw * (1.0 - self.diag_mask)           # zero diagonal
        dW_normed = F.rms_norm(dW_raw, normalized_shape=(self.N,))

        # Δdecay head: row-pool then per-neuron MLP
        row_feat = x.mean(dim=-1).transpose(1, 2)           # [BS, N, 32]
        dDecay_raw = self.decay_head(row_feat).squeeze(-1)  # [BS, N]

        return dW_normed, dDecay_raw
```

Activation checkpointing: wrap the decoder `forward` in
`torch.utils.checkpoint.checkpoint` when `config.checkpoint_decoder=True`.
Peak activation is `[BS, 32, N, N]` at the final stage (~600 MB at BS=72,
N=256).

The `DiscreteActionPolicy.decode*` methods wrap this, adding Gumbel/hard
sampling upstream:

```python
def decode(self, codes: Tensor) -> tuple[Tensor, Tensor]:
    emb = self.codebook[codes]                     # [BS, D_code]
    return self.decoder(emb)                        # (ΔW_normed, dDecay_raw)

def decode_soft(self, soft_weights: Tensor) -> tuple[Tensor, Tensor]:
    emb = soft_weights @ self.codebook              # differentiable lookup
    return self.decoder(emb)
```

**Tests**:
- Shape: decoder input `[BS, D_code]`, output `ΔW ∈ [BS, N, N]`,
  `Δdecay ∈ [BS, N]`.
- Zero-init correctness: at init, `ΔW ≈ 0` (within float noise) for any code.
- Diagonal mask: `ΔW[:, i, i] == 0` for all i.
- Gradient flow: loss on decoder output has grad wrt `codebook`, `init_proj`,
  all conv stages, both heads.

### Step 6 — Integrate modulator into memory forward

File: `src/model/memory.py`

Replace `_modulate_cells` with a new `_modulate` method that:
1. Calls `ConvGridModulator` to get `logits: [BS, K]`.
2. Samples code via `DiscreteActionPolicy.sample_*` (Gumbel soft in phase 1,
   hard categorical in phase 2).
3. Gets `(ΔW_normed, Δdecay_raw)` from the conv-transpose decoder directly —
   no rank-factoring step to stitch together.
4. EMA-blend into `W` and `decay` using per-neuron `W_decay_logit`,
   `decay_gamma_logit` (shape `[N]`, not `[NC]`).

Update module construction in `MemoryGraph.__init__`:
- Remove `logit_w1/b1/w2/b2` ownership (now in ConvGridModulator).
- Remove old MLP decoder params from `DiscreteActionPolicy` (replaced by
  `ConvTransposeDecoder` submodule).
- Add `self.modulator = ConvGridModulator(config)`.

Update `compute_modulator_stats` — measure applied plasticity as before,
just with new shapes.

**Test**: full forward pass end-to-end on tier_tiny; scalar loss returned;
backward pass doesn't error; gradients flow to conv encoder params,
codebook, conv-transpose decoder params (all stages), and memory dynamics
MLPs (verify via param.grad is not None).

### Step 7 — Training loop shape updates

File: `src/train.py`

Most of this file is shape-agnostic, but check:
- `compute_mod_grad_norm` in memory.py — update to reference conv params
  instead of logit_w1/w2.
- `compute_param_norms` — update param name list.
- `compute_component_grad_norms` — update param name list.
- Any metric that references `per_cell_logpi_std` or similar — now single
  code per event, so "per-cell" is not a concept; drop these metrics.

File: `src/train_phase2.py` — defer. Phase 2 GRPO redesign (autoregressive
sampling on pretrained LM) is separate future work. For now mark this file
as non-functional with a comment header pointing to the future plan.

File: `src/train_loop.py` — verify it doesn't make assumptions about memory
shapes or per-cell metrics. Likely fine.

File: `src/trainer.py` — verify likewise.

**Test**: run a ~10-step training loop on tier_tiny config and confirm loss
decreases.

### Step 8 — Runtime-state backward-compatibility

File: `src/model/memory.py`

In `load_runtime_state`, detect old checkpoints (that have NC dimensions in h,
msg, W, etc.) and either:
- Reject with a clear error message ("checkpoint is from old 8-cell design,
  not loadable on conv-grid-modulator branch").
- Or adapt: reshape `[BS, NC, N, ...]` by averaging / pooling NC into a new
  single pool (lossy but enables smooth transition).

Recommendation: reject. A warm restart from an old checkpoint is complex and
the architectural change is too large for a principled port. The tradeoff:
we lose the verify_01 bootstrap (~3h compute). Acceptable; we're redesigning.

**Test**: loading an old checkpoint produces a clear error.

### Step 9 — Unit + integration tests

New test files:
- `tests/test_grid_modulator.py` — conv encoder shape + gradient tests.
- `tests/test_discrete_policy_v2.py` — updated discrete policy tests.
- `tests/test_memory_single_cell.py` — memory forward/backward in the single-cell
  regime, tier_tiny config.

Update existing tests:
- `tests/test_memory.py` — shape updates throughout.
- `tests/test_model_e2e.py` — same.
- Any test asserting `h.shape[1] == NC` etc.

**Goal**: tests pass on tier_tiny. Full-scale (tier_a) tested separately via
a short training run (step 11 below).

### Step 10 — Cleanup / back-compat removal

Once the new design runs end-to-end:
- Delete deprecated config fields (`N_cells`, `neurons_per_cell`, `K`).
- Delete dead code paths (`_modulate_cells`, old logit_head params).
- Delete old tests that no longer apply.

### Step 11 — Smoke training run

Run a small training on tier_a config:

```bash
python -u -m src.train \
    --steps 1000 \
    --bs 32 \
    --lr 3e-4 \
    --save-dir outputs/conv_mod_smoke
```

Verify:
- Loss decreases.
- `eval_ce_loss` goes down from random-init (~10.37) toward ~7 within 1000
  steps.
- No NaN, no divergence of h / W / hebbian norms.
- Throughput (tok/s) reasonable — expect ~25-40K at BS=32 on 4090.

If this works, scale up to the real training budget.

## Must-carry-over from current code

These are non-obvious subtleties from the current implementation that must
survive the refactor. Losing any of them is a known way to break training.

### Precision: bf16 EVERYWHERE

Deliberate simplification from current code: all params and runtime state
in bf16, no f32 branches in the memory module.

The one bf16-fragile op (`(1-γ)·X + γ·Y` with γ near 1) is handled by
**clamping γ to `gamma_max = 0.97`** via `gamma_max * sigmoid(logit)`.
At γ=0.97, `(1-γ)=0.03` has ~0.78% bf16 precision, which is enough for EMA
accumulation not to compound errors.

Operations PyTorch auto-promotes to f32 under autocast (no manual handling):
`F.softmax`, `F.log_softmax`, `F.gumbel_softmax`, `F.cross_entropy`,
`F.group_norm`, `F.layer_norm`, `F.rms_norm`, tensor-core matmul accumulation,
AdamW optimizer state.

**Removed from current code** (in Step 1):
- f32 cast on Hebbian update → now bf16 throughout
- f32 cast on W EMA blend → now bf16 throughout
- f32 cast on decay EMA blend → now bf16 throughout
- f32 cast on mod_input construction → now bf16 throughout
- `compute_modulator_stats` precision handling → bf16 reads OK for telemetry

**Kept from current code**:
- Decay stored directly in [0,1] (convex EMA property)
- `clamp(min=1e-6, max=1.0-1e-6)` on γ before log in half-life computation
  (just to avoid inf — not a precision issue)

**Watch in telemetry**: distribution of learned γ values across the 256
neurons. If many peg at `gamma_max=0.97`, the clamp is a real bottleneck
and we should consider localized f32 blends for specific neurons.

### Initialization

| Item | Source | Detail |
|---|---|---|
| Xavier + TANH_GAIN=5/3 for tanh layers | `memory.py:123-129` | state/msg MLP layers |
| Xavier (gain=1.0) for linear-output layers | `memory.py:141` | inject projection |
| Small-std (1e-3) for decoder output layer | `discrete_policy.py:93` | Old MLP decoder. **New design: zero-init the final `dW_head` 1×1 conv**, same purpose (start at no-op). |
| Codebook init `randn * D_code**-0.5` | `discrete_policy.py:80` | Unit-norm-ish per row |
| Plasticity logits at `-3.0` → γ ≈ `0.97·sigmoid(-3)` = 0.046 | see design | Slow initial integration; clamp ceiling at 0.97 |
| Plasticity logit at `+2.0` → γ ≈ `0.97·sigmoid(+2)` = 0.854 | see design | Faster hebbian tracking |

### Gradient management

| Item | Source | New design notes |
|---|---|---|
| Split grad clip pools (LM / dynamics / modulator) | `trainer.py:73-103` | Update pool membership: **modulator pool** = conv encoder + logit head. **dynamics pool** = state/msg MLPs + inject + neuron_id + plasticity logits + codebook + decoder. |
| sqrt(N)-scaled per-pool clip budgets | `trainer.py:101-103` | Auto-adjusts to new pool sizes. |
| `mem_lr_scale` separate LR for memory | `train.py:168` | Keep at 1.0; revisit if mem/LM grad scales diverge. |
| `H_mid.detach()` at memory input | `model.py:82` | **Load-bearing** — memory doesn't get gradient via input path, only via readout path. |
| Cosine LR + warmup schedule | `train.py:185-192` | Unchanged. |
| Fused AdamW on CUDA | `train.py:181` | Unchanged. |
| EOT-aware valid mask on mem_pred_loss | `memory.py:991-1005` | Carry over. Live surprise signal does NOT mask — spike at EOT is legitimate. |

### Training mechanics

| Item | Source | New design notes |
|---|---|---|
| `torch.compile` on `_run_block` | `memory.py:977` | Verify compile handles new conv encoder call. Disable on encoder specifically if needed; keep on step loop. |
| Activation checkpoint option | `memory.py:1057` | Keep for memory loop. Add similar option for conv-transpose decoder (its peak activation `[BS, 32, 256, 256]` ≈ 600 MB). |
| `F.rms_norm` fused kernel (not manual) | `memory.py:677` | Use for ΔW row-normalization in the new decoder. |
| Dead-code reset, bootstrap only | `discrete_policy.py:275`, `train.py:377` | **Bump reset cadence for K=4096**: with 8× more codes, dead-code risk is higher. Reset every 200-300 steps instead of 500 during first few thousand steps. |
| Gumbel τ anneal 1.0 → 0.3 across `lr_target_step` | `train.py` step_callback, `memory.py:83` | Unchanged mechanism. **Consider a floor >0.3 for large K** — if K=4096 makes training unstable, raise floor to 0.5. |
| Compile-safe code usage (one_hot+sum, not bincount) | `discrete_policy.py:266` | Keep — required for torch.compile compatibility. |
| AdamW momentum zeroing on dead-code reset | `discrete_policy.py:312` | Keep — otherwise stale momentum drags reset rows back. |
| `prev_token` passed through for chunk boundaries | `model.py:40`, `memory.py:945` | Keep — EOT mask at position 0 depends on it. |

### Telemetry / debuggability (add to the new design)

- Per-neuron plasticity `γ` distribution (min/max/std across N=256). With 256 independent γ values, watch for pathological tails.
- Conv encoder activation magnitudes per layer (raw scale check at init).
- Decoder `ΔW_raw` distribution pre-rms-norm, per modulation event.
- `mod_grad_norm` — scale of gradient flowing back to the conv encoder.

## Testing strategy

Three levels:

1. **Unit (fast)**: each new module has a shape + gradient test. Run
   continuously during development.
2. **Integration (medium)**: end-to-end forward + backward on tier_tiny with
   BS=2, T=8. Catches wiring bugs.
3. **Smoke (slow)**: 1000-step training on tier_a to verify the full loop
   trains and doesn't diverge. Before merging back to main.

## Risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Conv encoder fails to learn useful features (low mem_leverage) | medium | Compare vs a fallback "flat MLP over pooled grid features" — isolates whether conv vs dense is the issue |
| Conv-transpose decoder doesn't produce useful spatial structure | medium | Zero-init final head means decoder starts at no-op; if it stays at no-op, encoder side is the problem. If it diverges, check RMSNorm + EMA gate. |
| Content-compressed observation too lossy | low | Ablate by feeding statistics-only input (zero out h_proj, msg_proj channels) |
| Port-pool indexing bugs (inject/readout broken) | medium | Property test: round-trip H_mid → inject → memory step → readout; expect identity-ish under zero plasticity |
| Training instability from GroupNorm / new init | low | Gradient clipping, scale check of initial activations |
| VRAM blowup from naive grid materialization | high if not careful | The d_proj=16 compression is load-bearing; unit-test peak VRAM < 2GB for modulator at BS=72 N=256 |
| Decoder activation stack too big at large spatial dims | medium | Activation-checkpoint the decoder if peak VRAM is tight; decoder's off the critical path so extra recompute is cheap |
| Checkerboard artifacts in ΔW from transpose-conv aliasing | low (we use resize+conv, not native ConvTranspose) | If we ever switch to native ConvTranspose2d, validate that downstream RMSNorm absorbs artifacts |
| Torch.compile fails on new module | low | Fall back to eager if needed; not blocking |

## Rollback plan

If the design fails to train competitively:
1. Keep the branch; don't merge.
2. Resurrect `main` for continued baseline work.
3. If the conv modulator is the specific failure point (not the NC=1
   collapse), try attention-with-edge-bias as a drop-in replacement for the
   conv encoder — same observation tensor, same output shape, different
   processing.

## Deferred / non-goals for this branch

- Phase 2 GRPO training. Deferred until the pretrained-LM pivot (separate
  branch `pretrained-lm-memory`) is ready and autoregressive sampling is on
  the table.
- Multi-code policies (H > 1). If phase-1 results are strong, revisit for
  GRPO SNR.
- Direct h-writing by modulator (beyond W and decay edits). Flagged in
  design doc §Open Questions.
- Attention-with-edge-bias fallback. Only implement if conv fails.

## Success criteria

For the branch to be considered "working":

1. `eval_ce_loss` on 1.5B-token training matches or beats current ~4.7
   post-bootstrap value (current best: verify_01 at 4.72).
2. `mem_leverage_ce` ≥ 1.2 nats (current verify_01: 1.19 nats).
3. Training throughput ≥ 25K tok/s at BS=72 (current ~42K; a regression is
   acceptable if leverage goes up).
4. All tests pass.

Merge target: `main`.
