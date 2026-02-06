"""
MetricsCollector — collects training metrics and saves to JSONL.

Two-tier collection:
  - Basic (every step): loss, ppl, lr, throughput, grad_norm, reg — ~8 floats
  - Full (every N steps): gate distributions, PM/EM/WM state, per-module grad norms
"""

import json
import math
import os
import time
from pathlib import Path

import torch
from torch import Tensor

from ..model.model import NeuromorphicLM
from ..model.config import ModelConfig


class MetricsCollector:
    def __init__(
        self,
        model: NeuromorphicLM,
        config: ModelConfig,
        output_path: str = "checkpoints/metrics.jsonl",
        collect_every: int = 50,
        basic_every: int = 1,
    ):
        self.model = model
        self.config = config
        self.output_path = output_path
        self.collect_every = collect_every
        self.basic_every = basic_every

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # PM commit rate accumulators: {(block_idx, layer_idx): [count, total]}
        self._pm_commit_accum: dict[tuple, list] = {}
        # EM write rate accumulators: {block_idx: [count, total, novelty_sum]}
        self._em_write_accum: dict[int, list] = {}

        self._file = open(output_path, "a")

    def should_collect_full(self, step: int) -> bool:
        """Whether this step requires full collection."""
        return self.collect_every > 0 and step % self.collect_every == 0

    def log_basic(self, step: int, loss: float, ppl: float, lr: float,
                  tok_s: float, grad_norm: float, reg: float, elapsed: float):
        """Write a basic metrics line (every step)."""
        record = {
            "step": step,
            "loss": loss,
            "ppl": min(ppl, 1e6),
            "lr": lr,
            "tok_s": tok_s,
            "grad_norm": grad_norm,
            "reg": reg,
            "elapsed": elapsed,
        }
        self._write(record)

    def log_full(self, step: int, gate_stats: dict, basic: dict):
        """Write a full metrics line merging basic + gate + memory + grad stats."""
        record = dict(basic)
        record["step"] = step
        record["full"] = True

        # Gate stats from forward pass
        self._merge_gate_stats(record, gate_stats)

        # Memory subsystem stats (read directly from model state)
        self._collect_pm_stats(record)
        self._collect_em_stats(record)
        self._collect_wm_stats(record)

        # Per-module gradient norms
        self._collect_grad_norms(record)

        # Flush accumulated commit/write rates
        self._flush_rates(record)

        self._write(record)

    def record_pm_commit(self, block_idx: int, layer_idx: int,
                         commit_mask: Tensor):
        """Accumulate PM commit rate across spans within a chunk."""
        key = (block_idx, layer_idx)
        if key not in self._pm_commit_accum:
            self._pm_commit_accum[key] = [0.0, 0]
        self._pm_commit_accum[key][0] += commit_mask.float().mean().item()
        self._pm_commit_accum[key][1] += 1

    def record_em_write(self, block_idx: int, write_mask: Tensor,
                        novelty_mean: float):
        """Accumulate EM write rate across spans within a chunk."""
        if block_idx not in self._em_write_accum:
            self._em_write_accum[block_idx] = [0.0, 0, 0.0]
        self._em_write_accum[block_idx][0] += write_mask.float().mean().item()
        self._em_write_accum[block_idx][1] += 1
        self._em_write_accum[block_idx][2] += novelty_mean

    def _flush_rates(self, record: dict):
        """Flush accumulated commit/write rates into the record."""
        for (b, l), (total, count) in self._pm_commit_accum.items():
            if count > 0:
                record[f"pm_commit_rate_b{b}_l{l}"] = total / count
        self._pm_commit_accum.clear()

        for b, (total, count, nov) in self._em_write_accum.items():
            if count > 0:
                record[f"em_write_rate_b{b}"] = total / count
                record[f"em_novelty_mean_b{b}"] = nov / count
        self._em_write_accum.clear()

    def _merge_gate_stats(self, record: dict, gate_stats: dict):
        """Merge per-block, per-layer gate stats into flat record."""
        # gate_stats: {block_idx: {layer_idx: {"gate_a": Tensor, "gate_b": Tensor, "h_norm": float}}}
        for b_idx, layers in gate_stats.items():
            for l_idx, lstats in layers.items():
                prefix = f"b{b_idx}_l{l_idx}"
                ga = lstats["gate_a"]  # [BS, D_h]
                gb = lstats["gate_b"]  # [BS, D_h]
                # Gate a stats (sigmoid output, 0-1)
                record[f"{prefix}_gate_a_mean"] = ga.mean().item()
                record[f"{prefix}_gate_a_std"] = ga.std().item()
                record[f"{prefix}_gate_a_near0"] = (ga < 0.1).float().mean().item()
                record[f"{prefix}_gate_a_near1"] = (ga > 0.9).float().mean().item()
                # Gate b stats (tanh output, -1 to 1)
                record[f"{prefix}_gate_b_mean"] = gb.mean().item()
                record[f"{prefix}_gate_b_std"] = gb.std().item()
                record[f"{prefix}_gate_b_abs_mean"] = gb.abs().mean().item()
                # Hidden state norm
                record[f"{prefix}_h_norm"] = lstats["h_norm"]

    def _collect_pm_stats(self, record: dict):
        """Read PM state tensors and compute summary stats."""
        if not self.config.pm_enabled:
            return
        for b_idx, block in enumerate(self.model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                if pm.pm_a is None:
                    continue
                prefix = f"pm_b{b_idx}_l{l_idx}"
                pm_a = pm.pm_a.detach()  # [BS, r]
                record[f"{prefix}_a_mean"] = pm_a.mean().item()
                record[f"{prefix}_a_max"] = pm_a.max().item()
                record[f"{prefix}_a_sum"] = pm_a.sum(dim=-1).mean().item()
                record[f"{prefix}_nonzero"] = (pm_a > 0.01).float().mean().item()
                if pm.elig_K is not None:
                    elig_norm = pm.elig_K.detach().norm(dim=-1).mean().item()
                    record[f"{prefix}_elig_norm"] = elig_norm

    def _collect_em_stats(self, record: dict):
        """Read EM state tensors and compute summary stats."""
        if not self.config.em_enabled:
            return
        for b_idx, block in enumerate(self.model.blocks):
            em = block.em
            if em.em_S is None:
                continue
            prefix = f"em_b{b_idx}"
            em_S = em.em_S.detach()  # [BS, M]
            record[f"{prefix}_S_mean"] = em_S.mean().item()
            record[f"{prefix}_S_max"] = em_S.max().item()
            record[f"{prefix}_S_sum"] = em_S.sum(dim=-1).mean().item()
            record[f"{prefix}_nonzero"] = (em_S > 0.01).float().mean().item()

    def _collect_wm_stats(self, record: dict):
        """Read WM state and compute attention entropy."""
        wm = self.model.wm
        if not hasattr(wm, "_last_attn") or wm._last_attn is None:
            return
        attn = wm._last_attn  # [BS, n_heads, W]
        # Entropy: -sum(p * log(p))
        log_attn = torch.log(attn + 1e-10)
        entropy = -(attn * log_attn).sum(dim=-1)  # [BS, n_heads]
        record["wm_entropy_mean"] = entropy.mean().item()
        record["wm_entropy_std"] = entropy.std().item()
        record["wm_entropy_min"] = entropy.min().item()
        record["wm_entropy_max"] = entropy.max().item()
        # Buffer utilization
        if wm.wm_valid is not None:
            valid = wm.wm_valid.detach().float()  # [BS, W]
            record["wm_buffer_util"] = valid.mean().item()

    def _collect_grad_norms(self, record: dict):
        """Per-module gradient norms after backward."""
        module_groups = {
            "embedding": [self.model.embedding],
            "lm_head": [self.model.lm_head],
            "wm": [self.model.wm],
            "W_in": [self.model.W_in],
        }
        for b_idx, block in enumerate(self.model.blocks):
            module_groups[f"block_{b_idx}"] = [block]
            for l_idx, layer in enumerate(block.layers):
                module_groups[f"b{b_idx}_l{l_idx}_gates"] = [
                    layer.gate_a, layer.gate_b
                ]
                if self.config.pm_enabled:
                    module_groups[f"b{b_idx}_l{l_idx}_pm"] = [layer.pm]

        for name, modules in module_groups.items():
            total_norm_sq = 0.0
            for mod in modules:
                for p in mod.parameters():
                    if p.grad is not None:
                        total_norm_sq += p.grad.detach().norm().item() ** 2
            record[f"gnorm_{name}"] = math.sqrt(total_norm_sq)

    def _write(self, record: dict):
        """Write a single JSON line."""
        # Convert any remaining non-serializable values
        clean = {}
        for k, v in record.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    clean[k] = None
                else:
                    clean[k] = round(v, 6)
            else:
                clean[k] = v
        self._file.write(json.dumps(clean) + "\n")
        self._file.flush()

    def close(self):
        """Close the output file."""
        self._file.close()
