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
- Add `conv_channels: int = 64` (conv hidden width).
- Add `conv_layers: int = 4`.
- Add `conv_kernel: int = 3`.
- Add `conv_groups: int = 8` (for GroupNorm).
- Add `action_rank: int = 32` (rank of factored ΔW).
- Add `role_dim: int = 4` (role embedding dim).
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

        self.h_proj = nn.Linear(D_n, d_proj, bias=False)
        self.msg_proj = nn.Linear(D_n, d_proj, bias=False)
        self.role_emb = nn.Embedding(3, config.role_dim)  # input/output/internal

        # Compute input channel count from design
        C_in = 3 + 3*d_proj + 1 + 2*role_dim + 2
        self.conv_in = nn.Conv2d(C_in, conv_channels, kernel=3, padding=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(conv_channels, conv_channels, kernel=3, padding=1)
            for _ in range(conv_layers - 1)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(conv_groups, conv_channels) for _ in range(conv_layers)
        ])

        # Pooling head → code logits
        self.logit_head = nn.Linear(conv_channels, config.num_codes)

    def build_input(self, h, msg, W, hebbian, decay, s_live, s_ema, role_id) -> Tensor:
        """Build the [BS, N, N, C_in] edge feature map."""
        h_p = self.h_proj(h)        # [BS, N, d_proj]
        msg_p = self.msg_proj(msg)  # [BS, N, d_proj]
        role_e = self.role_emb(role_id)  # [N, role_dim]

        # Broadcast to grid: h_p[i] broadcast over j dim, h_p[j] over i dim, etc.
        ...
        # Concat all channels
        return E  # [BS, N, N, C_in]

    def encoder_forward(self, E: Tensor) -> Tensor:
        """Conv stack → pooled cell feature."""
        x = E.permute(0, 3, 1, 2)  # [BS, C_in, N, N] for Conv2d
        x = self.conv_in(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(norm(F.gelu(x)))
        pooled = x.mean(dim=(2, 3))  # global avg pool → [BS, C_h]
        return pooled

    def forward(self, h, msg, W, hebbian, decay, s_live, s_ema, role_id) -> Tensor:
        E = self.build_input(h, msg, W, hebbian, decay, s_live, s_ema, role_id)
        pooled = self.encoder_forward(E)
        logits = self.logit_head(pooled)  # [BS, K]
        return logits
```

**Test**: forward a random input, assert output shape `[BS, K]`; gradient check
(backward without errors); memory footprint at BS=72 N=256 under 1 GB for the
build_input tensor.

### Step 5 — Discrete policy refactor

File: `src/model/discrete_policy.py`

Change shape assumptions:
- Input: `logits: [BS, K]` (was `[BS, NC, K]`).
- `codes: [BS]` (was `[BS, NC]`).
- `soft: [BS, K]` (was `[BS, NC, K]`).
- `log_pi: [BS]`.

The logit-computation path moves OUT of this file (now done by ConvGridModulator).
`DiscreteActionPolicy` becomes a thinner module: codebook + decoder + sampling
primitives. Keep:
- `sample_discrete`, `sample_gumbel_soft`, `log_prob`, `entropy`, `decode`,
  `decode_soft`, `update_usage`, `reset_dead_codes`.

Remove:
- `compute_logits` (moved to `ConvGridModulator`).
- `logit_w1/b1/w2/b2` parameters.

The `forward` convenience method needs a rewrite:

```python
def forward(self, logits, phase="phase1", tau=1.0) -> dict:
    """logits comes from ConvGridModulator; this module does the rest."""
    ...
```

New decoder shape:
- Input: `[BS, D_code]`.
- Output: `[BS, 2·N·r + N]` — split into U, V, Δdecay.
- Decoder hidden: `H_dec = 256` by default.

```python
self.dec_w1 = nn.Parameter(torch.empty(code_dim, decoder_hidden))
self.dec_b1 = nn.Parameter(torch.zeros(decoder_hidden))
self.dec_w2 = nn.Parameter(torch.empty(decoder_hidden, 2 * N * action_rank + N))
self.dec_b2 = nn.Parameter(torch.zeros(2 * N * action_rank + N))
```

**Test**: sampling and decoding produce correct shapes; codebook lookup
differentiable (soft path); categorical sampling gives valid log_pi.

### Step 6 — Integrate modulator into memory forward

File: `src/model/memory.py`

Replace `_modulate_cells` with a new `_modulate` method that:
1. Calls `ConvGridModulator` to get `logits: [BS, K]`.
2. Passes to `DiscreteActionPolicy.forward(logits, phase=...)` for sampling +
   decoding.
3. Parses decoder output `[BS, 2Nr + N]` into `U, V, Δdecay`.
4. Computes `ΔW = U @ V.T`, rms_norm over last dim.
5. EMA-blend into `W` and `decay` using per-neuron `W_decay_logit`,
   `decay_gamma_logit` (shape `[N]`, not `[NC]`).

Update module construction in `MemoryGraph.__init__`:
- Remove `logit_w1/b1/w2/b2` ownership (now in ConvGridModulator).
- Add `self.modulator = ConvGridModulator(config)`.
- Update `DiscreteActionPolicy` instantiation with new action_dim.

Update `compute_modulator_stats` — measure applied plasticity as before,
just with new shapes.

**Test**: full forward pass end-to-end on tier_tiny; scalar loss returned;
backward pass doesn't error; gradients flow to conv params, codebook,
decoder, and memory dynamics MLPs (verify via param.grad is not None).

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
| Conv encoder fails to learn useful features (low_mem_leverage) | medium | Compare vs a fallback "flat MLP over pooled grid features" — isolates whether conv vs dense is the issue |
| Low-rank ΔW too restrictive | low-medium | Ablate r ∈ {16, 32, 64, 128}; easy to bump |
| Content-compressed observation too lossy | low | Ablate by feeding statistics-only input (zero out h_proj, msg_proj channels) |
| Port-pool indexing bugs (inject/readout broken) | medium | Property test: round-trip H_mid → inject → memory step → readout; expect identity-ish under zero plasticity |
| Training instability from GroupNorm / new init | low | Gradient clipping, scale check of initial activations |
| VRAM blowup from naive grid materialization | high if not careful | The d_proj=16 compression is load-bearing; unit-test peak VRAM < 2GB for modulator at BS=72 N=256 |
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
