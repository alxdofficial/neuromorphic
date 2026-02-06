"""
Working Memory — sliding window attention with ring-buffer KV cache.

One shared instance across the entire model. Provides transformer-like
precision on short context (copying, binding) via bounded attention
over the last W tokens.

State per stream:
    wm_K: [BS, W, D_wm]   — key cache
    wm_V: [BS, W, D_wm]   — value cache
    wm_valid: [BS, W]      — bool mask for valid entries
    wm_ptr: [BS]           — ring buffer write index
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin


class WorkingMemory(nn.Module, StateMixin):
    _state_tensor_names = ["wm_K", "wm_V", "wm_valid", "wm_ptr"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.W = config.W
        self.D_wm = config.D_wm
        self.n_heads = config.n_heads_wm
        self.head_dim = config.D_wm // config.n_heads_wm
        assert config.D_wm % config.n_heads_wm == 0

        # Projections
        self.W_q = nn.Linear(config.D, config.D_wm)
        self.W_k = nn.Linear(config.D, config.D_wm)
        self.W_v = nn.Linear(config.D, config.D_wm)
        self.W_o = nn.Linear(config.D_wm, config.D)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # State tensors (lazily initialized on first step)
        self.wm_K: Tensor = None
        self.wm_V: Tensor = None
        self.wm_valid: Tensor = None
        self.wm_ptr: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize state tensors on first forward pass."""
        self.wm_K = torch.zeros(BS, self.W, self.D_wm, device=device)
        self.wm_V = torch.zeros(BS, self.W, self.D_wm, device=device)
        self.wm_valid = torch.zeros(BS, self.W, dtype=torch.bool, device=device)
        self.wm_ptr = torch.zeros(BS, dtype=torch.long, device=device)

    def step(self, x: Tensor, reset_mask: Tensor) -> Tensor:
        """Process one token through working memory.

        Args:
            x: [BS, D] — current token embedding
            reset_mask: [BS] bool — True for streams at doc boundary

        Returns:
            y_wm: [BS, D] — working memory output
        """
        BS, D = x.shape
        device = x.device

        # Lazy init
        if self.wm_K is None:
            self._lazy_init(BS, device)

        # Reset validity for masked streams
        if reset_mask.any():
            self.wm_valid = self.wm_valid.clone()
            self.wm_valid[reset_mask] = False
            self.wm_ptr = self.wm_ptr.clone()
            self.wm_ptr[reset_mask] = 0

        # Project q, k, v
        q = self.W_q(x)    # [BS, D_wm]
        k = self.W_k(x)    # [BS, D_wm]
        v = self.W_v(x)    # [BS, D_wm]

        # Write (k, v) into ring buffer at current ptr
        ptr = self.wm_ptr  # [BS]
        batch_idx = torch.arange(BS, device=device)

        self.wm_K = self.wm_K.clone()
        self.wm_V = self.wm_V.clone()
        self.wm_valid = self.wm_valid.clone()

        self.wm_K[batch_idx, ptr] = k.detach()
        self.wm_V[batch_idx, ptr] = v.detach()
        self.wm_valid[batch_idx, ptr] = True

        # Multi-head attention over valid entries
        # Reshape q: [BS, n_heads, head_dim]
        q_h = q.view(BS, self.n_heads, self.head_dim)
        # Reshape K, V cache: [BS, W, n_heads, head_dim] -> [BS, n_heads, W, head_dim]
        K_h = self.wm_K.view(BS, self.W, self.n_heads, self.head_dim).transpose(1, 2)
        V_h = self.wm_V.view(BS, self.W, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [BS, n_heads, W]
        attn = torch.einsum("bnd, bnwd -> bnw", q_h, K_h) * self.scale

        # Mask invalid positions
        valid_mask = self.wm_valid.unsqueeze(1)  # [BS, 1, W]
        attn = attn.masked_fill(~valid_mask, float("-inf"))

        # Softmax (handle all-invalid case)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # all-invalid -> zero attention

        # Store for debug metrics (overwritten each step, ~32K floats)
        self._last_attn = attn.detach()  # [BS, n_heads, W]

        # Weighted sum: [BS, n_heads, head_dim]
        out = torch.einsum("bnw, bnwd -> bnd", attn, V_h)

        # Merge heads and project
        out = out.reshape(BS, self.D_wm)
        y_wm = self.W_o(out)  # [BS, D]

        # Advance pointer (ring buffer wrap)
        self.wm_ptr = (ptr + 1) % self.W

        return y_wm
