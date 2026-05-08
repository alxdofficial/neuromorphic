"""Llama host adapters + MemInjectLayer.

Reused by `src.trajectory_memory.integrated_lm`:
- `src.pretrained.hosts.build_host` for Llama-family attribute paths.
- `src.pretrained.mem_inject_layer.MemInjectLayer` for the cross-attn
  injection at one mid-stack Llama layer.

Nothing else in this package — earlier graph_walker plumbing
(`PretrainedLMWithMemory`, `MemAdapter`, rollout/training loops) is
archived under `abandoned/graph-walker`.
"""
