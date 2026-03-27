"""Triton kernels for v9.1 neuron dynamics — PLACEHOLDER.

Phase 1 (current): Python loop + per-step gradient checkpointing.
Phase 2 (future): Fused Triton forward + backward kernels.

The Python path in memory_graph.py is the reference implementation.
"""

# Triton kernels will be added here when we optimize for throughput.
# For now, memory_graph.py uses per-step gradient checkpointing which
# gives correct gradients with manageable memory (~150MB peak per step).
