# v8 Neural Memory Graph — Implementation Plan

> **Status**: Implementation plan. Ready to build.
> **Design doc**: `docs/architecture_v8_neural_memory_graph.md`
> **Branch**: `v7-single-scan-stack` (start from here)

## Core Architecture

Full-D scan stack (same as v7) provides the language model. Memory graph
provides per-CC persistent memory that runs every token. Neuromodulator
(PPO-trained) controls memory plasticity.

```
Pass 1: Pre-memory scan layers[0..L_mem-1] over all T tokens  (parallel, full D)
         → H, surprise available at every position

Memory loop: for t in 0..T-1:                                  (sequential, cheap)
  CC→memory: inject (H_slice[t], surprise_slice[t]) per CC
  Memory graph step: all neurons receive → modulate → route
  Memory→CC: read signals for position t

Pass 2: Post-memory scan layers[L_mem..L_total-1] over all T   (parallel, full D)
         → H now integrates memory context
         → logits
```

This gives us:
- **Full-D scan** for language modeling (works from step 1, no bootstrap risk)
- **Per-token memory access** (every position gets its own memory read/write)
- **Per-token surprise** feeds into memory (from Pass 1's PCM)
- **GPU-efficient** scans over full T=2048 (no segments for scans)
- **Cheap memory loop** (no autograd, SIMD across neurons, ~0.1ms per step)

## Scale & Config

```python
@dataclass
class V8Config:
    # Scan Stack (Language Model) — same as v7
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns (= memory blocks)
    D_cc: int = -1               # derived: D // C = 128
    L_total: int = 10            # total scan layers
    L_mem: int = 5               # memory injection point
    d_inner: int = 1024
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # PCM (per-CC, independent weights)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1

    # Memory Graph
    N_neurons: int = 1024        # total neurons (C * M_per_block)
    M_per_block: int = 64        # neurons per block
    D_mem: int = 128             # primitive vector dimension
    inter_block_k: int = 16      # sparse connections to other blocks
    mem_temperature: float = 1.0 # routing softmax temperature

    # Neuromodulator
    neuromod_hidden: int = 256
    neuromod_layers: int = 2
    action_every: int = 8        # act every N tokens
    max_action_magnitude: float = 0.1

    # PPO
    ppo_gamma: float = 0.99      # full-segment credit propagation
    ppo_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 512
    ppo_lr: float = 3e-4
    ppo_ent_coef: float = 0.003
    ppo_vf_coef: float = 0.5

    # Training
    T: int = 2048                # full chunk length (no segments for scans)
    N: int = 128                 # pos_embed size is T, N only for compat
    gradient_checkpointing: bool = False
    use_compile: bool = True
    lifelong_mode: bool = False

    @property
    def actions_per_chunk(self) -> int:
        return self.T // self.action_every  # 256
```

## Detailed Architecture

### Scan Stack (Language Model)

Identical to v7. Full D=2048 scan layers, pos_embed sized to T=2048.
The scan stack provides all cross-column mixing and causal token processing.
Memory does NOT need to provide cross-column mixing — the scan handles it.

```python
class V8LM(nn.Module):
    """Language model with memory injection. Scan stack is v7-compatible."""

    def __init__(self, config):
        # Embedding + pos_embed [T, D]
        self.embedding = nn.Embedding(vocab_size, D_embed)
        self.proj_up = nn.Linear(D_embed, D)
        self.proj_down = nn.Linear(D, D_embed)
        self.pos_embed = nn.Parameter(torch.randn(config.T, D) * 0.02)

        # Full-D scan layers (shared across all positions, all CCs)
        self.layers = nn.ModuleList([
            ScanLayer(D, d_inner, dropout, n_layers=L_total, glu_output=True)
            for _ in range(L_total)
        ])

        # Per-CC PCM (independent weights, operates on D_cc slices)
        self.pcm_modules = nn.ModuleList([
            SingleColumnPCM(D_cc) for _ in range(C)
        ]) if pcm_enabled else None

        # Per-CC memory interface
        self.mem_proj_in = nn.ModuleList([
            nn.Linear(D_cc + D_cc, D_mem)  # (H_slice + surprise_slice) → D_mem
            for _ in range(C)
        ])
        self.mem_proj_out = nn.ModuleList([
            nn.Linear(D_mem, D_cc)  # memory signal → D_cc
            for _ in range(C)
        ])
        # Zero-init mem_proj_out so memory starts silent
        for proj in self.mem_proj_out:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        # Learnable memory gate per CC (starts at sigmoid(0) = 0.5)
        self.mem_gate = nn.Parameter(torch.zeros(C))

        # Output head
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, vocab_size, bias=False)

    def forward(self, input_ids, mem_signals):
        """
        Args:
            input_ids:   [BS, T]
            mem_signals: [BS, T, C, D_mem] — per-position, per-CC memory signals

        Returns:
            logits:       [BS, T, vocab]
            surprise_all: [BS, T, C, D_cc] — per-CC surprise at each position
            aux_loss:     scalar
        """
        BS, T = input_ids.shape

        # Embed
        x = self.embedding(input_ids)           # [BS, T, D_embed]
        x = self.proj_up(x)                     # [BS, T, D]
        x = x + self.pos_embed[:T]              # [BS, T, D]

        # --- Pass 1: Pre-memory scan (full D, parallel over T) ---
        H = x
        for i in range(self.config.L_mem):
            carry = self._carries[i]
            H, h_last = self.layers[i](H, carry)
            self._carries[i] = h_last

        # --- PCM: per-CC surprise computation ---
        H_cols = H.view(BS, T, C, D_cc)        # [BS, T, C, D_cc]
        x_cols = x.view(BS, T, C, D_cc)

        surprise_all = []
        aux_loss = torch.tensor(0.0, device=H.device)
        for c in range(C):
            surp, z_hat, z = self.pcm_modules[c].compute_surprise(
                H_cols[:, :, c], x_cols[:, :, c]   # each [BS, T, D_cc]
            )
            surprise_all.append(surp)
            aux_loss = aux_loss + self.pcm_modules[c].prediction_loss(z_hat, z)

        surprise_tensor = torch.stack(surprise_all, dim=2)  # [BS, T, C, D_cc]
        aux_loss = aux_loss / C * self.config.pcm_pred_weight

        # Apply PCM gain to H (view as columns, apply per-CC, reshape back)
        for c in range(C):
            H_cols[:, :, c] = self.pcm_modules[c].apply_gain(
                H_cols[:, :, c], surprise_all[c]
            )
        H = H_cols.reshape(BS, T, D)

        # --- Memory injection: additive, per-CC ---
        # Project memory signals [BS, T, C, D_mem] → [BS, T, C, D_cc]
        mem_projected = torch.stack([
            self.mem_proj_out[c](mem_signals[:, :, c])
            for c in range(C)
        ], dim=2)                                # [BS, T, C, D_cc]
        gate = torch.sigmoid(self.mem_gate)      # [C]
        mem_contribution = (gate[None, None, :, None] * mem_projected)
        H = H + mem_contribution.reshape(BS, T, D)

        # --- Pass 2: Post-memory scan (full D, parallel over T) ---
        for i in range(self.config.L_mem, self.config.L_total):
            carry = self._carries[i]
            H, h_last = self.layers[i](H, carry)
            self._carries[i] = h_last

        # --- Output ---
        out = self.proj_down(H)
        out = self.ln_final(out)
        logits = self.lm_head(out) * (self.config.D_embed ** -0.5)

        return logits, surprise_tensor, aux_loss
```

### CC→Memory Signal

Each CC sends a signal to its memory block every token. The signal combines
the CC's hidden state slice and surprise:

```python
# For CC c at position t:
cc_signal = self.mem_proj_in[c](
    torch.cat([H_cols[:, t, c], surprise[:, t, c]], dim=-1)
)  # [BS, D_mem]
```

This means surprise is part of the write signal — surprising tokens produce
different memory stimulation than predicted tokens.

### Memory Graph (unchanged from design doc)

```python
class MemoryGraph:
    """Neural memory graph. Runs every token, outside autograd."""

    # State per stream:
    #   primitives:  [BS, N_neurons, D_mem]
    #   thresholds:  [BS, N_neurons, max_conn]
    #   activations: [BS, N_neurons, D_mem]  (current step buffer)
    #   prev_output: [BS, N_neurons, D_mem]  (previous step, read-only)

    def step(self, cc_signals):
        """One timestep. Inject CC signals, all neurons step, return CC reads.

        Args:
            cc_signals: [BS, C, D_mem] — one signal per CC/block
        Returns:
            mem_signals: [BS, C, D_mem] — one signal per CC/block
        """
        with torch.no_grad():
            self._inject_cc_signals(cc_signals)
            inputs = self._gather_inputs()          # gather from prev_output
            outputs = self._modulate(inputs)         # f(input, primitive)
            self._route_outputs(outputs)             # scatter to activations
            mem_signals = self._read_cc_ports()
            self.prev_output = self.activations.clone()
            self.activations.zero_()                 # reset for next step
        return mem_signals
```

### Neuromodulator + PPO (unchanged from design doc)

Shared MLP, acts every 8 tokens, trained by PPO with gamma=0.99.
Reward = negative per-token CE loss (block-level average).

### Top-Level Model

```python
class V8Model(nn.Module):

    def forward_chunk(self, input_ids, reset_mask=None):
        """Process T=2048 tokens with per-token memory access.

        Returns logits, aux_loss, ppo_experience.
        """
        BS, T = input_ids.shape

        # --- Pass 1: Pre-memory scan + PCM (parallel over T) ---
        logits_pass1 = None  # not needed, just H and surprise
        H_pre, surprise, aux_loss = self.lm.forward_pre_memory(input_ids)
        # H_pre: [BS, T, D], surprise: [BS, T, C, D_cc]

        # --- Memory loop: step memory graph for each token ---
        mem_signals = torch.zeros(BS, T, C, D_mem, device=device)
        ppo_experience = []

        H_cols = H_pre.view(BS, T, C, D_cc)
        for t in range(T):
            # Build per-CC signal: H_slice + surprise
            cc_to_mem = torch.stack([
                self.lm.mem_proj_in[c](
                    torch.cat([H_cols[:, t, c], surprise[:, t, c]], dim=-1)
                ) for c in range(C)
            ], dim=1)  # [BS, C, D_mem]

            # Memory step
            mem_out = self.memory.step(cc_to_mem)  # [BS, C, D_mem]
            mem_signals[:, t] = mem_out

            # Neuromodulator (every action_every tokens)
            if t % self.config.action_every == 0:
                obs = self.memory.get_neuron_obs()
                action, logprob, entropy, value = self.neuromod.get_action_and_value(
                    obs.reshape(-1, obs_dim)  # flatten [BS, N_neurons] → [BS*N_neurons]
                )
                self.memory.apply_neuromod_actions(
                    action[:, :D_mem].reshape(BS, N_neurons, D_mem),
                    action[:, D_mem:].reshape(BS, N_neurons, max_conn),
                )
                ppo_experience.append((obs, action, logprob, value))

            # Doc boundary reset (check if current token is EOT)
            if not self.config.lifelong_mode:
                eot_mask = (input_ids[:, t] == self.config.eot_id)
                if eot_mask.any():
                    self.memory.reset_streams(eot_mask)

        # --- Pass 2: Post-memory scan (parallel over T, with memory injected) ---
        logits, aux_loss2 = self.lm.forward_post_memory(H_pre, mem_signals)

        return logits, aux_loss, ppo_experience, mem_signals
```

## Training Loop

```python
class V8Trainer:
    def train_chunk(self, batch):
        """One training step: LM forward + backward + PPO update."""
        input_ids = batch.input_ids   # [BS, T]
        target_ids = batch.target_ids  # [BS, T]

        # Forward
        logits, aux_loss, ppo_exp, _ = self.model.forward_chunk(
            input_ids, reset_mask=(batch.prev_token == self.config.eot_id)
        )

        # LM loss + backward (only through scan stack, not memory)
        ce_loss, valid_count = batched_cross_entropy(logits, target_ids, loss_mask)
        lm_loss = ce_loss / valid_count.clamp(min=1) + aux_loss
        lm_loss.backward()
        clip_grad_norm_(self.model.lm.parameters(), 1.0)
        self.lm_optimizer.step()
        self.lm_optimizer.zero_grad()

        # Per-token reward for PPO (detached from LM graph)
        with torch.no_grad():
            per_token_ce = per_token_cross_entropy(logits.detach(), target_ids)
            # Aggregate to per-block reward at neuromod action frequency
            block_rewards = compute_block_rewards(
                per_token_ce, C, self.config.action_every
            )  # [actions_per_chunk, BS, C] → reshape for PPO

        # PPO update
        ppo_buffer = build_ppo_buffer(ppo_exp, block_rewards)
        ppo_metrics = self.ppo_trainer.update(ppo_buffer)

        # Detach scan carries (memory graph state persists, no detach needed)
        self.model.lm.detach_carries()

        return {**lm_metrics, **ppo_metrics}
```

**No K-segment loop in the trainer.** The LM processes all T=2048 tokens in
one `forward_chunk` call. The memory loop inside `forward_chunk` steps per-token
but is cheap and outside autograd. One backward pass, one optimizer step, one
PPO update per chunk.

## File Structure

```
src/v8/
├── __init__.py
├── config.py              # V8Config dataclass
├── lm.py                  # V8LM: scan stack + PCM + memory interface
├── memory_graph.py        # MemoryGraph (no autograd, SIMD)
├── neuromodulator.py      # Neuromodulator policy + value network
├── ppo.py                 # PPORolloutBuffer, GAE, PPOTrainer
├── model.py               # V8Model: top-level wiring (LM + memory + neuromod)
├── trainer.py             # V8Trainer: LM + PPO training loop
└── pcm.py                 # SingleColumnPCM (non-grouped WithinScanPCM)

tests/v8/
├── test_memory_graph.py   # Graph step, routing, energy conservation
├── test_lm.py             # Scan + PCM + memory injection shapes
├── test_ppo.py            # Buffer, GAE, policy update
├── test_integration.py    # Full forward + backward + PPO
└── test_gradients.py      # All LM params get gradient
```

## Implementation Order

1. **V8Config** — dataclass, validation, tier presets
2. **SingleColumnPCM** — extract from existing WithinScanPCM, plain nn.Linear
3. **MemoryGraph** — the core innovation, test in isolation
4. **V8LM** — reuse ScanLayer, add per-CC PCM + memory interface, test forward
5. **Neuromodulator** — policy + value MLP
6. **PPO** — rollout buffer, GAE, update loop
7. **V8Model** — wire LM + memory + neuromod, test full forward
8. **V8Trainer** — joint LM + PPO training loop
9. **Train on Pile** — compare LM-only vs LM+memory
10. **Benchmark** — throughput, memory usage

## Success Criteria

1. **LM trains normally** — loss ~4.5-5.0 without memory (v7 baseline territory)
2. **Memory graph adds <5% overhead** — wall clock close to v7
3. **PPO converges** — neuromod actions are non-random, KL stays <0.02
4. **Memory helps** — loss with memory < loss without memory
5. **Throughput** — ≥80K tok/s at BS=8-16 on RTX 4090 (v7 was 59K-103K)

## Why This Should Work for Language Modeling

The scan stack is a proven LM backbone (v7 loss ~4.9). Memory provides:

- **Per-CC context enrichment**: each column gets a persistent signal from its
  memory block, carrying information from earlier in the document
- **Cross-segment recall**: memory state persists across the T=2048 chunk
  boundary (memory graph is never detached, only scan carries are)
- **Adaptive storage**: neuromodulator learns WHAT to store based on surprise
  and prediction loss — high-surprise content gets stronger primitives
- **Natural forgetting**: without neuromodulator reinforcement, neuron
  activations decay through signal propagation (energy conservation ensures
  signal diminishes over time without active maintenance)

The scan handles language modeling. Memory handles long-range context and
adaptation. The neuromodulator bridges them through RL. Each component is
independently testable and useful.
