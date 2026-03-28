"""Triton kernel tests — v10.

v10 uses scalar neurons with standard PyTorch operations.
Custom Triton kernels may be added later if profiling shows benefit.
"""

import pytest


class TestTritonPlaceholder:
    def test_placeholder(self):
        """Placeholder — v10 scalar neurons use standard PyTorch ops."""
        pass
