# Pretrained-LM + Memory Graph

**Branch:** `pretrained-lm-memory-v2`.
**Status:** smoke-clean on CPU. No GPU run yet.

This doc describes how the memory graph (from the from-scratch attention-neuromod
work — see `design.md` for its internals) is grafted onto a pretrained Llama-3.2
backbone, and the two-phase training protocol (Gumbel bootstrap → autoregressive
GRPO).

---

## 1. High-level picture

```
     input_ids                                               logits
        │                                                       ▲
        ▼                                                       │
  ┌─────────────┐      ┌──────────────┐       ┌─────────────┐   │
  │ embed_tokens│ ───► │ layers 0..L-1│ ─────►│MemInjectLayer│──▶ layers L+1..N-1 ─▶ norm ─▶ lm_head
  └─────────────┘      └──────────────┘   │   │   (wraps L)  │
                                           │   └──────┬──────┘
                                           │          │
                                           │  ┌───────▼───────┐
                                           │  │  memory_fn    │
                                           │  │  (closure)    │
                                           │  └───────┬───────┘
                                           │          │
                                           │  ┌───────▼───────┐
                                           └──│ memory graph  │
                                              │  (cells × N)  │
                                              └───────────────┘
```

- **Host LM**: Llama-3.2-3B (or 1B for dev). All weights frozen.
- **Rolling context window**: the LM processes at most `T = 2048` tokens at once.
  For longer sequences we chunk; memory state carries across chunks.
- **Read/write point**: a single mid-stack layer (`L = 14` of 28 for 3B; `L = 8`
  of 16 for 1B). That layer is wrapped with a `MemInjectLayer`.
- **Side-channel flow**: at layer `L`, hidden states `h ∈ [BS, T, d_lm]` are
  projected into memory space `W_in: d_lm → d_mem = 2048`, fed to the memory
  graph (which reads state, computes a readout, possibly writes new
  plasticity), projected back `W_out: d_mem → d_lm`, and added to `h` via a
  per-dim trainable `scale` gate before the wrapped `LlamaDecoderLayer` runs.
- **Trainable parameters**: memory graph (~4M) + `W_in, W_out` (~12.6M on 3B)
  + `scale` (~3K). Everything else in Llama is frozen.

---

## 2. File map

```
src/pretrained/
├── config.py                    # PretrainedConfig, llama_1b() / llama_3b() factories
├── llm_wrapper.py               # PretrainedLMWithMemory — loads + freezes + wires
├── mem_inject_layer.py          # MemInjectLayer — wraps one LlamaDecoderLayer
├── llama_mem_adapter.py         # duck-types Llama as the `lm` object memory expects
├── rollout.py                   # autoregressive_rollout primitive (phase 2)
├── train_phase1.py              # run_phase1() — Gumbel bootstrap training loop
└── train_phase2.py              # grpo_step() — REINFORCE on rollouts

src/model/                       # core memory graph (shared with from-scratch main)
├── memory.py                    # MemoryGraph, forward_segment, _run_block, _modulate
├── attention_modulator.py       # shared-trunk modulator + DirectDecoder (ΔW/Δdecay)
├── discrete_policy.py           # codebook + Gumbel/Categorical sampling + log_pi
└── triton_memory_step.py        # fused per-token LIF + W@msg + readout kernel
```

---

## 3. MemInjectLayer — the integration primitive

Forward, `h ∈ [BS, T, d_lm]`:

```python
# 1. project LM hidden into memory space
h_mem = W_in(h)                              # [BS, T, d_mem]

# 2. read + write memory via closure
readout = memory_fn(h_mem)                   # [BS, T, d_mem]
# memory_fn is a closure installed by PretrainedLMWithMemory.forward
# that calls MemoryGraph.forward_segment with the correct input_ids and
# the Llama-shaped lm adapter (see §4).

# 3. gated additive inject, then delegate to the original layer
injected = h + scale * W_out(readout)        # [BS, T, d_lm]
return orig_layer(injected, *args, **kwargs)
```

- `W_in` and `W_out` are `nn.Linear(d_lm, d_mem, bias=False)` / vice versa.
  When `d_lm == d_mem` (Llama-3.2-1B case) they initialize to identity so
  the layer starts near a no-op; otherwise Xavier.
- `scale` is a per-dim `nn.Parameter(torch.full((d_lm,), sqrt(alpha)=2.0))`.
- `memory_fn` is optional; when `None`, `scale` must be zero. Any other
  combination raises `RuntimeError` — silent bypass would produce
  incorrect outputs indistinguishable from a working run.

---

## 4. Memory graph at-a-glance

From `docs/design.md` (unchanged by the pivot):

- `NC = 8` cells × `N = 32` neurons × `D_n = 256` per-neuron state.
- Block-diagonal `W ∈ [BS, NC, N, N]`: cells are isolated at the W level.
- Shared-trunk modulator: one `AttentionModulator` batched over `NC` as a
  batch dim; per-cell identity is a learned `cell_emb [NC, d_cell=16]`
  concatenated into the per-neuron tokens.
- **Event clocks** driven by a per-token index `t` inside a segment:
  - Every token: `W @ msg` + inject (first α ports) + LIF (`h = tanh(decay·h + (1-decay)·received)`) + readout pool (α output ports, scaled 1/√α). Fused in a Triton kernel on CUDA.
  - Every 4 tokens: `msg = MLP(h)` + Hebbian EMA update.
  - Every 16 tokens: modulator → code logits → policy sample → decoder → `ΔW, Δdecay` → γ-clamped EMA blend.

### Adapter for Llama

`src/pretrained/llama_mem_adapter.py` provides a small object with the
five fields `MemoryGraph.forward_segment` reads off its `lm` argument:

| Field                     | Custom LM              | Llama adapter               |
|---------------------------|------------------------|-----------------------------|
| `lm.lm_head`              | `nn.Linear(D, vocab)`  | `llama.lm_head`             |
| `lm.proj_down`            | optional down-proj     | `MemInjectLayer.W_out` — reuses the same d_mem→d_lm projection |
| `lm.ln_final.weight`      | `LayerNorm.weight`     | `llama.model.norm.weight`   |
| `lm.ln_final.bias`        | `LayerNorm.bias`       | `None` (RMSNorm has none)   |
| `lm.mem_head_logits(x)`   | method → [BS,T,vocab]  | method: `lm_head(llama.model.norm(W_out(x)))` |

`forward_segment` takes a `use_rmsnorm: bool` kwarg; `_run_block`
branches on it to pick `F.rms_norm` vs `F.layer_norm` for the per-token
surprise signal. Llama path passes `use_rmsnorm=True, rms_eps=1e-5`.

---

## 5. Phase 1 — Gumbel-soft backprop bootstrap

Goal: make the memory write useful code-conditioned ΔW, so that by end
of phase 1 the memory-augmented LM beats vanilla Llama by at least
~0.2 nats of CE on held-out evaluation.

### Protocol

- **Loss**:  `L = CE(logits, target_ids) + 0.1 · mem_pred_loss`
  - CE is standard next-token prediction on `target_ids = input_ids[..., 1:]`.
  - `mem_pred_loss` is a memory-side auxiliary: memory's own readout goes
    through `W_out → llama.norm → lm_head` and is CE'd against the same
    targets. This couples memory's readout space to the LM's prediction
    space and gives a direct gradient from the memory graph to its own
    writes without having to go through the injection gate.

- **Sampling**: Gumbel-softmax with hard straight-through. Gradient flows
  through the soft distribution back to modulator logits; forward uses
  the argmax one-hot so the decoder sees a single code per cell.
- **Temperature schedule**: `τ: 1.0 → 0.3` linearly across
  `anneal_across_steps` (typically < total steps so it pins before decay
  completes).
- **TBPTT**: detach memory state (`h`, `msg`, `W`, `decay`, `hebbian`,
  `prev_readout`, `s_mem_*`) every `tbptt_block = 32` tokens within a
  segment. `forward_segment` also detaches the persistent state across
  segments.
  - **Invariant**: `tbptt_block >= 2 · modulation_interval`. Otherwise
    the modulator's last write lands on the boundary and its gradient
    gets severed before any readout consumes it. Validated in
    `Config.validate()`.

- **Autocast**: on CUDA, `torch.autocast(cuda, bfloat16)` wraps the
  forward. Llama weights stay fp32; memory state runs bf16; autocast
  keeps the matmul math in bf16 and reductions (softmax, CE) in fp32.

- **Gate for advancing to phase 2**: `eval_ce_vanilla - eval_ce_with_memory
  ≥ 0.2` nats on a held-out slice. Not automated yet.

### Default phase-1 hyperparameters

| Knob                       | Default | Notes |
|----------------------------|---------|-------|
| `T` (segment length)       | 512     | rolling window ≤ Llama context |
| `BS`                       | 16      | phase 1 |
| `tbptt_block`              | 16      | within-segment detach |
| Gumbel `τ_start → τ_end`   | 1.0 → 0.3 |  |
| `mem_pred_weight`          | 0.1     |  |
| `max_grad_norm`            | 1.0     |  |

*(Those are the `PretrainedConfig` defaults. The tier_tiny memory config
has a much smaller `T=8` for unit tests only.)*

### Entry point

```python
from src.pretrained.config import PretrainedConfig
from src.pretrained.llm_wrapper import PretrainedLMWithMemory
from src.pretrained.train_phase1 import Phase1Batch, run_phase1

cfg = PretrainedConfig.llama_3b()
wrapper = PretrainedLMWithMemory(cfg)
wrapper.reset_memory(bs=cfg.bs)
opt = torch.optim.AdamW([p for _, p in wrapper.trainable_parameters()], lr=1e-4)

def data_iter():
    # yield Phase1Batch(input_ids=[BS,T], target_ids=[BS,T], prev_token=[BS]|None)
    ...

run_phase1(wrapper, opt, data_iter(), steps=N, mem_pred_weight=0.1,
           gumbel_tau_start=1.0, gumbel_tau_end=0.3,
           anneal_across_steps=int(0.9*N),
           on_step=lambda log: print(log))
```

---

## 6. Phase 2 — Autoregressive GRPO

Goal: fix long-horizon credit assignment that Gumbel backprop can't reach
(memory writes at step `t` whose benefit only materializes at step
`t + k ·T_segment`).

### Why teacher-forced GRPO fails (from verify_01)

If K rollouts all see the same teacher-forced tokens, memory state
differs per rollout but the reward (CE on the same tokens) barely
moves — memory adds ~1 nat to a ~4 nat problem, and tiny code changes
produce tiny CE changes. Measured SNR on verify_01: 2.8e-4 within-K
spread vs 2.4 across-slot std. 611 steps of GRPO with that SNR produced
zero movement.

### Autoregressive fix

- K rollouts share a prefix.
- Prefix is processed **once** with memory in phase-2 mode (hard
  Categorical sampling, independent K samples across the batch).
  Memory states diverge across rollouts at each modulator fire.
- Then tokens are **generated autoregressively** — each rollout samples
  its own next token from the memory-augmented logits. Because memory
  states differ, logits differ, so the K rollouts diverge in token
  space, not just memory space.
- Reward varies across rollouts because generated sequences are
  genuinely different.

### Protocol per step

1. Replicate the prefix K times (`[K, T_prefix]`).
2. `wrapper.current_phase = "phase2"`; `wrapper.train(True)`;
   `wrapper.reset_memory(bs=K)`.
3. **Prefix pass with grad and KV cache**:
   `out = wrapper(prefix_rep, use_cache=True)`.
   - Memory samples K independent hard codes at each modulator fire.
   - `memory._last_log_pi_sum [K]` accumulates
     `Σ_fires log π(code_fire)` — graph-connected back through
     `log_softmax` to the modulator logits.
   - `out.past_key_values` carries Llama's KV cache; `out.logits[:, -1]`
     is the distribution over the first generated position.
4. **Generate**: `wrapper.train(False)` (memory uses argmax; no new codes
   sampled during gen). First token sampled from the prefix's
   last-position logits. Each subsequent step passes only the newly
   sampled token + KV cache, so memory does exactly one LIF update per
   generated token — no re-processing of the prefix, no state drift.
5. **Reward**: `reward_fn(generated [K, L], reference [L]) → rewards
   [K]`. Default in `train_phase2.py`: per-rollout fraction of tokens
   matching the reference. Real runs will want something denser —
   candidates: log-prob of the reference continuation under each
   rollout's memory-augmented LM, BLEU, or a task-specific downstream
   reward.
6. **Advantage**: `A = (r - r.mean()) / max(r.std(), adv_std_floor)`, per
   group. The std floor (default 1e-3) keeps near-uniform early-training
   rewards from producing runaway 1e4-scale advantages. Unlike the
   PPO-standard `std + 1e-8`, a hard floor means small real variance is
   clamped to "no meaningful signal, small advantage" rather than
   "explosive advantage magnified through log π".
7. **REINFORCE loss**:
   `L = -(log_pi_sum · A.detach()).mean()`.
   Backward flows through `log_pi_sum` to every modulator weight that
   produced the logits, and through `log_softmax` to the codebook
   entries that were looked up.
8. Clip, step, detach memory, restore phase to `"phase1"`.

### Per-step hyperparameters (defaults)

| Knob               | Default | Notes                                                          |
|--------------------|---------|----------------------------------------------------------------|
| `num_rollouts`     | 8       | K parallel rollouts                                            |
| `gen_length`       | 256     | tokens generated per rollout                                   |
| `temperature`      | 1.0     | token-sampling temperature                                     |
| `max_grad_norm`    | 1.0     |                                                                |
| `adv_std_floor`    | 1e-3    | `denom = max(rewards.std(), floor)` — stops noise-level reward variance from producing 1e4-scale advantages |

**Deliberately not wired (future work):**

- **KL-to-reference-policy**. Needs a reference modulator snapshot (e.g., the pre-phase-2 weights) plus per-fire logits re-evaluated under both policies. Not a one-liner; tracked in the status table.
- **Entropy bonus**. Needs per-fire logits plumbed out of `_modulate` through `_run_block` into `forward_segment`. The sampled `log_pi` alone is not a proxy for the full-distribution entropy.

### Entry point

```python
from src.pretrained.train_phase2 import grpo_step

opt = torch.optim.AdamW([p for _, p in wrapper.trainable_parameters()], lr=1e-4)

for step in range(N):
    prefix, reference = next_pair()    # prefix [1, T_prefix], reference [L]
    log = grpo_step(
        wrapper, opt,
        prefix_ids=prefix, reference_cont=reference,
        num_rollouts=8, gen_length=256, temperature=1.0)
    print(log.loss, log.reward_mean, log.log_pi_mean)
```

---

## 7. What's built vs what's not

| Component                             | State          |
|---------------------------------------|----------------|
| Wrapper + freeze + scale-zero identity | ✓ smoke-covered |
| MemInjectLayer + W_in/W_out/scale      | ✓ smoke-covered |
| Llama adapter (RMSNorm path)           | ✓ smoke-covered |
| Memory integration end-to-end          | ✓ smoke-covered |
| Phase-1 `run_phase1()` loop            | ✓ smoke-covered (synthetic + real wikitext) |
| Autoregressive rollout primitive       | ✓ smoke-covered (K divergence) |
| Phase-2 `grpo_step()` loop             | ✓ smoke-covered (stub reward, gradient reaches modulator + codebook) |
| Advantage-std floor for stability      | ✓ smoke-covered (near-uniform rewards clamped) |
| `_last_log_pi_sum` phase gating        | ✓ smoke-covered (None after phase-1 / eval) |
| Llama tokenizer presets                | ✓ in `src/data/tokenizer.py` |
| CLI driver (phase 1 + phase 2)         | ✗ not built |
| Real phase-1 data shards (100M tokens) | ✗ not built — smoke streams wikitext-2 in memory |
| Production phase-2 reward              | ✗ token-match is a smoke placeholder |
| KL penalty vs reference policy         | ✗ not wired — needs reference modulator snapshot + per-fire logit re-eval |
| Entropy bonus                          | ✗ not wired — needs per-fire logits from `_modulate` |
| Checkpoint save/load                   | ✗ not built |
| GPU smoke                              | ✗ all current smokes are CPU |

---

## 8. Known invariants / footguns

- `tbptt_block >= 2 · modulation_interval`. Enforced in `Config.validate()`.
- Llama loaded `torch_dtype=torch.float32`. Memory state is bf16 on CUDA.
  Without autocast, the round-trip through `W_in/W_out` truncates small
  gradients. `run_phase1` and `grpo_step` wrap their forward passes in
  `torch.autocast(device_type, bfloat16)` on CUDA; CPU uses a null
  context.
- `MemInjectLayer` raises if `memory_fn is None and scale != 0`. The
  `attach_memory=False` constructor path pins `scale=0` so the
  transparent bypass remains valid.
- Gen-time memory: modulator never fires at T=1 per step (clock resets).
  Memory's slow state (`W`, `decay`, `hebbian`) is frozen at the
  post-prefix value; fast state (`h`, `msg`, `prev_readout`) updates
  per-token. This is intentional — the policy trained in phase 2 is the
  memory's writes during prefix processing.
- `wrapper.current_phase` toggles sampling mode (`phase1` Gumbel-soft vs
  `phase2` hard Categorical + log_pi). The `grpo_step` sets it for the
  prefix pass; `autoregressive_rollout` sets and restores it around
  the primitive.
- `memory._last_log_pi_sum` is **only** written on phase-2 training
  forwards. Phase-1 / eval forwards set it to `None`. Read it
  immediately after the phase-2 prefix pass; don't hold the attribute
  across an intervening phase-1 call.
- `MemInjectLayer` injects **before** the original `LlamaDecoderLayer`
  runs (modifies the layer's input). Standard adapter patterns (Houlsby,
  LongMem) typically inject after a sublayer. Pre-layer gives the memory
  contribution maximum influence (passes through attention + FFN of
  layer L and all subsequent layers) at the cost of more potential
  disruption to the layer's RMSNorm statistics. Reconsider if phase-1
  loss fails to descend from the vanilla-Llama baseline.
