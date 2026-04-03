"""V11 Model — Split-scan LM + Cell-based memory graph.

Reuses V8LM (scan layers, PCM, inject_memory) unchanged.
Only the memory graph is replaced with CellMemoryGraph.
"""

import inspect

import torch
import torch.nn as nn
from torch import Tensor

from ..v8.lm import V8LM
from .config import V11Config
from .memory_graph import CellMemoryGraph


class V11Model(nn.Module):
    """Language model with cell-based neuromorphic memory."""

    def __init__(self, config: V11Config):
        super().__init__()
        config.validate()
        self.config = config

        # Reuse V8LM — construct with V8Config defaults, override what we need.
        from ..v8.config import V8Config
        lm_config = V8Config.tier_a(
            D=config.D, D_embed=config.D_embed,
            vocab_size=config.vocab_size, T=config.T,
            action_every=config.T,  # one segment per chunk
            d_inner=config.d_inner, glu_output=config.glu_output,
            L_total=config.L_total, scan_split_at=config.scan_split_at,
            pcm_enabled=config.pcm_enabled,
            pcm_pred_weight=config.pcm_pred_weight,
            pcm_hidden=config.pcm_hidden,
            dropout=config.dropout,
            tie_embeddings=config.tie_embeddings,
        )
        lm_config.validate()
        self.lm = V8LM(lm_config)

        # Override mem_scale init for v11 readout scaling.
        # v11 readout divides by alpha (not sqrt(N_per_slice)), so
        # mem_scale should init to alpha to cancel the 1/alpha in readout.
        with torch.no_grad():
            self.lm.mem_scale.fill_(float(config.alpha))

        # Cell-based memory graph
        device = torch.device('cpu')  # moved to device later via .to()
        self.memory = CellMemoryGraph(config, device)

    def forward_chunk(self, input_ids: Tensor,
                      target_ids: Tensor | None = None,
                      use_memory: bool = True) -> dict:
        """Process one chunk of tokens.

        Args:
            input_ids: [BS, T]
            target_ids: [BS, T] or None
            use_memory: if False, skip memory graph (baseline mode)

        Returns:
            dict with 'logits', 'aux_loss', 'surprise'
        """
        BS, T = input_ids.shape
        config = self.config

        # Detect internal EOS for scan state reset
        eot = config.eot_id
        reset_mask = (input_ids == eot) if eot >= 0 else None

        # Lower scan → H_mid + PCM surprise
        H_mid, surprise, x, aux_loss = self.lm.forward_scan_lower(
            input_ids, reset_mask=reset_mask)

        if use_memory and not self.memory.is_initialized():
            raise RuntimeError(
                "use_memory=True but memory is not initialized. "
                "Call model.initialize_states(BS) first.")
        if use_memory:
            # Memory graph processes detached H_mid
            cc_all = H_mid.detach().to(self.memory.dtype)
            action_every = config.action_every
            n_segments = T // action_every

            if n_segments == 1:
                mem_out = self.memory.forward_segment(cc_all)
            else:
                mem_out_segs = []
                for seg in range(n_segments):
                    t0 = seg * action_every
                    t1 = t0 + action_every
                    seg_cc = cc_all[:, t0:t1]
                    seg_out = self.memory.forward_segment(seg_cc)
                    mem_out_segs.append(seg_out)

                mem_out = torch.cat(mem_out_segs, dim=1)
            H_enriched = self.lm.inject_memory(H_mid, mem_out)
        else:
            H_enriched = H_mid

        # Upper scan
        H = self.lm.forward_scan_upper(
            H_enriched, surprise=surprise, reset_mask=reset_mask)

        # Output head
        logits = self.lm.forward_output(H)

        result = {'logits': logits, 'aux_loss': aux_loss, 'surprise': surprise}

        if target_ids is not None:
            import torch.nn.functional as F_
            # Mask EOS positions
            loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
            if eot >= 0:
                loss_mask = input_ids != eot
            logits_flat = logits[loss_mask].view(-1, logits.shape[-1])
            targets_flat = target_ids[loss_mask].view(-1)
            if logits_flat.numel() > 0:
                result['loss'] = F_.cross_entropy(logits_flat, targets_flat)
            else:
                result['loss'] = torch.tensor(0.0, device=logits.device)

        return result

    def initialize_states(self, BS: int):
        self.memory.initialize_states(BS)
        self.lm.initialize_carries()

    def detach_states(self):
        self.memory.detach_states()
        self.lm.detach_carries()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lm_param_count(self) -> int:
        return sum(p.numel() for p in self.lm.parameters() if p.requires_grad)

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters()
                   if p.requires_grad)

    def optimizer_param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Build AdamW param groups with separate LM/memory learning rates.

        Weight decay follows the common dim-based rule:
        - tensors with ndim >= 2 use decay
        - biases / norm scales / scalar params do not
        """
        mem_lr = lr * self.config.mem_lr_scale
        groups = []

        def split_params(module: nn.Module):
            decay, no_decay = [], []
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                if param.ndim >= 2:
                    decay.append(param)
                else:
                    no_decay.append(param)
            return decay, no_decay

        lm_decay, lm_no_decay = split_params(self.lm)
        mem_decay, mem_no_decay = split_params(self.memory)

        if lm_decay:
            groups.append({
                "params": lm_decay,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if lm_no_decay:
            groups.append({
                "params": lm_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
            })
        if mem_decay:
            groups.append({
                "params": mem_decay,
                "lr": mem_lr,
                "weight_decay": weight_decay,
            })
        if mem_no_decay:
            groups.append({
                "params": mem_no_decay,
                "lr": mem_lr,
                "weight_decay": 0.0,
            })

        return groups

    def make_optimizer(
        self,
        lr: float,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> torch.optim.Optimizer:
        """Create the recommended AdamW optimizer for v11 training.

        Uses fused AdamW on CUDA when the local PyTorch build supports it.
        """
        param_groups = self.optimizer_param_groups(lr, weight_decay)
        opt_kwargs = dict(
            params=param_groups,
            betas=betas,
            eps=eps,
        )
        first_param = next(self.parameters(), None)
        use_fused = (
            first_param is not None and
            first_param.is_cuda and
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        if use_fused:
            opt_kwargs["fused"] = True
        return torch.optim.AdamW(**opt_kwargs)
