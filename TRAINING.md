# Training Instructions

## v11 Cell-Based Memory Graph — Current (branch: v11-cells)

**Status**: Current default path is the scan-batched shared-I/O implementation on
`v11-cells`. It is trainable end-to-end and meets the throughput target on a
4090 with the recommended fused AdamW setup.

### Architecture
- 256 cells × 124 neurons = 31,744 total neurons (D_neuron=8)
- K=16 cell-local connections, `R_rounds=2` by default
- Shared state/msg MLPs, per-neuron modulator
- Shared inject/readout ports (`alpha=4`) by default
- Structural plasticity disabled by default while throughput work continues
- Recommended optimizer: `V11Model.make_optimizer(...)` for CUDA fused AdamW

### Previous versions (archived on their branches)
- v9-backprop: `v9-backprop` branch (N=512, D=256, shared-weight scan, 24K tok/s)
- v8: `v8-rl-neuromod` branch
- v6: `v6-main-backup` branch
- v4: `v4-iterative-backup` branch
