"""Triton kernels for v10 scalar neuron memory graph.

With scalar neurons (D=1), the per-step operations are simple element-wise
ops that PyTorch handles efficiently. Custom Triton kernels may be added
later for the gather operation if profiling shows it's beneficial.

The v9-backprop fused dendritic gather kernels have been removed as they
were designed for D=32 vector neurons with dendritic tree reduction.
"""

# Placeholder — v10 scalar neurons may not need custom Triton kernels.
# The gather (activation[conn_indices]) on scalar values is efficient
# as a standard PyTorch index operation.
