# Training Instructions

## v11 Cell-Based Memory Graph — Current (branch: v11-cells)

**Status**: Implementation complete. Triton forward kernel works at 65K neurons.
Backward requires reduced N or fused Triton backward (in progress).

### Architecture
- 256 cells × 256 neurons = 65,536 total neurons (D_neuron=8)
- K=16 cell-local connections, R=4 message-passing rounds/token
- Shared state/msg MLPs, per-cell modulator
- Dedicated inject (4/cell) and readout (4/cell) port neurons
- Total: ~52M params (LM=51.6M, Memory=1.1M)

### Previous versions (archived on their branches)
- v9-backprop: `v9-backprop` branch (N=512, D=256, shared-weight scan, 24K tok/s)
- v8: `v8-rl-neuromod` branch
- v6: `v6-main-backup` branch
- v4: `v4-iterative-backup` branch
