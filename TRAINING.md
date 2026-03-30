# Training Instructions

## v10-gnn: Shared-Weight GNN Memory Graph — Current (branch: v10-gnn)

**Status:** Implementation in progress. See `docs/plan_v10_gnn.md` for full design.

### Architecture

- **Lower scan** (sensory cortex): 2 layers, small and fast
- **PCM**: Dynamic predictive coding (transition prediction), RMSNorm on surprise
- **Memory graph**: N=4096 neurons, D=32, K sparse connections
  - Shared-weight GNN (PyG MessagePassing)
  - Sequential simulation: 1 step per token
  - Shared modulator → w_conn, decay, identity_delta
  - Shared state MLP with structural decay
  - Shared message MLP
  - Rolling window [N, W=16, D] of past activations (detached, persists)
  - Phi-based structural plasticity (2% global prune/grow, 20% exploration)
- **Perceiver**: compress rolling window → 64 memory words
- **Upper decoder** (frontal cortex): causal cross-attention to memory words → logits

### Previous Architecture (v9-backprop)

Archived on branch `v9-backprop`. See `docs/archive/plan_v9_backprop.md`.
Per-neuron weights (58M params) provided negligible benefit over LM-only baseline.

---

## Data Pipeline

The Pile via pre-tokenized .bin shards:

```bash
python scripts/prepare_data.py --tokens 12B --seed 42
```

Validation shard: `data/pile/pile_val.bin` (5M tokens)

## General Notes

- Always use `-u` flag with python to disable output buffering
- The `outputs/` directory is in `.gitignore`
- Always commit code changes before starting long training runs
- All scratchpads and debug notes must use point-form markdown
