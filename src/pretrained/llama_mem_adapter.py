"""Back-compat shim. The adapter is now host-agnostic — see `mem_adapter.py`.

This file exists only so lingering imports keep working. Prefer:
    from src.pretrained.mem_adapter import MemAdapter
"""

from src.pretrained.mem_adapter import LlamaMemAdapter, MemAdapter

__all__ = ["LlamaMemAdapter", "MemAdapter"]
