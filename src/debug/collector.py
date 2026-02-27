"""
MetricsCollector — collects training metrics and saves to JSONL.

Two-tier collection:
  - Basic (every step): loss, ppl, lr, throughput, grad_norm, reg — ~8 floats
  - Full (every N steps): PM/EM state, per-module grad norms, plasticity warnings
"""

import json
import math
import os
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
        phase: str | None = None,
    ):
        self.model = model
        self.config = config
        self.output_path = output_path
        self.collect_every = collect_every
        self.basic_every = basic_every
        self.phase = phase

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # PM commit rate accumulators: {block_idx: [count, total]}
        self._pm_commit_accum: dict[int, list] = {}
        # EM write rate accumulators: {block_idx: [count, novelty_sum, g_sum]}
        self._em_write_accum: dict[int, list] = {}

        self._file = open(output_path, "a")
        self._writes_since_flush = 0
        self._flush_every = 50  # flush to disk every N writes

    def should_collect_full(self, step: int) -> bool:
        """Whether this step requires full collection."""
        return self.collect_every > 0 and step % self.collect_every == 0

    def log_basic(self, step: int, loss: float, ppl: float, lr: float,
                  tok_s: float, grad_norm: float, reg: float, elapsed: float,
                  extras: dict = None, mode: str = "train"):
        """Write a basic metrics line (respects basic_every interval)."""
        if self.basic_every > 1 and step % self.basic_every != 0:
            return
        record = {
            "step": step,
            "mode": mode,
            "loss": loss,
            "ppl": min(ppl, 1e6),
            "lr": lr,
            "tok_s": tok_s,
            "grad_norm": grad_norm,
            "reg": reg,
            "elapsed": elapsed,
        }
        if extras:
            record.update(extras)
        self._write(record)

    def log_full(self, step: int, gate_stats: dict, basic: dict,
                 extras: dict = None, mode: str = "train"):
        """Write a full metrics line merging basic + memory + grad stats."""
        record = dict(basic)
        record["step"] = step
        record["mode"] = mode
        record["full"] = True
        if extras:
            record.update(extras)

        # Memory subsystem stats (read directly from model state)
        self._collect_pm_stats(record)
        self._collect_em_stats(record)

        # Per-module gradient norms
        self._collect_grad_norms(record)

        # Flush accumulated commit/write rates
        self._flush_rates(record)

        # Global summaries and warning signals
        self._collect_plasticity_summary(record)

        # Lifelong persistence stats (Phase B)
        if self.config.lifelong_mode:
            self._collect_lifelong_stats(record)

        self._write(record)

    def record_pm_commit(self, block_idx: int, p_commit: Tensor):
        """Accumulate PM commit strength across passes within a chunk.

        p_commit is a continuous [BS] tensor (0-1), not a binary mask.
        """
        if block_idx not in self._pm_commit_accum:
            self._pm_commit_accum[block_idx] = [torch.zeros((), device=p_commit.device), 0]
        self._pm_commit_accum[block_idx][0] += p_commit.float().mean().detach()
        self._pm_commit_accum[block_idx][1] += 1

    def record_em_write(self, block_idx: int,
                        novelty_mean: Tensor | float, g_em_mean: Tensor | float):
        """Accumulate EM write stats across passes within a chunk."""
        if torch.is_tensor(novelty_mean):
            novelty = novelty_mean.detach()
        else:
            novelty = torch.tensor(float(novelty_mean))
        if torch.is_tensor(g_em_mean):
            g_em = g_em_mean.detach()
        else:
            g_em = torch.tensor(float(g_em_mean))

        if block_idx not in self._em_write_accum:
            self._em_write_accum[block_idx] = [0, novelty, g_em]
        else:
            nov_sum = self._em_write_accum[block_idx][1]
            g_sum = self._em_write_accum[block_idx][2]
            self._em_write_accum[block_idx][1] = nov_sum + novelty.to(nov_sum.device)
            self._em_write_accum[block_idx][2] = g_sum + g_em.to(g_sum.device)
        self._em_write_accum[block_idx][0] += 1

    def _flush_rates(self, record: dict):
        """Flush accumulated commit/write rates into the record."""
        for b, (total, count) in self._pm_commit_accum.items():
            if count > 0:
                val = total / count
                record[f"pm_commit_rate_b{b}"] = float(val.item()) if torch.is_tensor(val) else val
        self._pm_commit_accum.clear()

        for b, accum in self._em_write_accum.items():
            count, nov, g = accum[0], accum[1], accum[2]
            if count > 0:
                nov_val = nov / count
                g_val = g / count
                record[f"em_novelty_mean_b{b}"] = float(nov_val.item()) if torch.is_tensor(nov_val) else nov_val
                record[f"em_g_em_mean_b{b}"] = float(g_val.item()) if torch.is_tensor(g_val) else g_val
        self._em_write_accum.clear()

    def _collect_pm_stats(self, record: dict):
        """Read PM state tensors and compute summary stats.

        v4: PM lives at block.pm (one PM per block, no layers).
        """
        if not self.config.pm_enabled:
            return
        for b_idx, block in enumerate(self.model.blocks):
            pm = block.pm
            if pm.pm_a is None:
                continue
            prefix = f"pm_b{b_idx}"
            pm_a = pm.pm_a.detach()  # [BS, r]
            record[f"{prefix}_a_mean"] = pm_a.mean().item()
            record[f"{prefix}_a_max"] = pm_a.max().item()
            record[f"{prefix}_a_sum"] = pm_a.sum(dim=-1).mean().item()
            record[f"{prefix}_nonzero"] = (pm_a > 0.01).float().mean().item()

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

    def _collect_plasticity_summary(self, record: dict):
        """Global PM/EM summaries with warning flags."""
        pm_commit_vals = [
            v for k, v in record.items()
            if k.startswith("pm_commit_rate_b") and isinstance(v, (float, int))
        ]
        em_write_vals = [
            v for k, v in record.items()
            if k.startswith("em_g_em_mean_b") and isinstance(v, (float, int))
        ]

        if pm_commit_vals:
            record["pm_commit_rate_global"] = sum(pm_commit_vals) / len(pm_commit_vals)
        if em_write_vals:
            record["em_write_rate_global"] = sum(em_write_vals) / len(em_write_vals)

        pm_budget = record.get("pm_budget_util_global")
        em_budget = record.get("em_budget_util_global")
        pm_commit = record.get("pm_commit_rate_global")
        em_write = record.get("em_write_rate_global")

        record["warn_commit_collapse"] = float(
            pm_commit is not None and pm_commit < 1e-3
        )
        record["warn_write_collapse"] = float(
            em_write is not None and em_write < 1e-3
        )
        record["warn_budget_saturation"] = float(
            (pm_budget is not None and pm_budget > 0.98)
            or (em_budget is not None and em_budget > 0.98)
        )

    def _collect_lifelong_stats(self, record: dict):
        """Collect cross-document memory persistence stats (Phase B)."""
        pm_nonzero_total = 0.0
        pm_slots_total = 0
        pm_budget_total = 0.0
        pm_budget_cap = 0.0

        for block in self.model.blocks:
            pm = block.pm
            if pm.pm_a is None:
                continue
            pm_a = pm.pm_a.detach()
            pm_nonzero_total += (pm_a > 0.01).float().sum().item()
            pm_slots_total += pm_a.numel()
            pm_budget_total += pm_a.sum().item()
            pm_budget_cap += pm.budget * pm_a.shape[0]  # budget * BS

        if pm_slots_total > 0:
            record["pm_persistence"] = pm_nonzero_total / pm_slots_total
            record["pm_budget_util"] = pm_budget_total / pm_budget_cap if pm_budget_cap > 0 else 0.0

        em_nonzero_total = 0.0
        em_slots_total = 0
        em_budget_total = 0.0
        em_budget_cap = 0.0

        for block in self.model.blocks:
            em = block.em
            if em.em_S is None:
                continue
            em_S = em.em_S.detach()
            em_nonzero_total += (em_S > 0.01).float().sum().item()
            em_slots_total += em_S.numel()
            em_budget_total += em_S.sum().item()
            em_budget_cap += em.budget * em_S.shape[0]  # budget * BS

        if em_slots_total > 0:
            record["em_persistence"] = em_nonzero_total / em_slots_total
            record["em_budget_util"] = em_budget_total / em_budget_cap if em_budget_cap > 0 else 0.0

    def _collect_grad_norms(self, record: dict):
        """Per-module gradient norms after backward.

        v4 structure: model.embedding, model.lm_head, model.fan_out, model.fan_in,
        model.blocks[b].columns, model.blocks[b].pm, model.blocks[b].em,
        model.blocks[b].pm_neuromod, model.blocks[b].em_neuromod
        """
        module_groups = {
            "embedding": [self.model.embedding],
            "lm_head": [self.model.lm_head],
            "fan_out": [self.model.fan_out],
            "fan_in": [self.model.fan_in],
        }
        for b_idx, block in enumerate(self.model.blocks):
            module_groups[f"block_{b_idx}"] = [block]
            module_groups[f"b{b_idx}_columns"] = [block.columns]
            if self.config.pm_enabled:
                module_groups[f"b{b_idx}_pm"] = [block.pm]
                module_groups[f"b{b_idx}_pm_neuromod"] = [block.pm_neuromod]
            if self.config.em_enabled:
                module_groups[f"b{b_idx}_em"] = [block.em]
                module_groups[f"b{b_idx}_em_neuromod"] = [block.em_neuromod]

        for name, modules in module_groups.items():
            total_norm_sq = 0.0
            for mod in modules:
                for p in mod.parameters():
                    if p.grad is not None:
                        total_norm_sq += p.grad.detach().norm().item() ** 2
            record[f"gnorm_{name}"] = math.sqrt(total_norm_sq)

    def _write(self, record: dict):
        """Write a single JSON line."""
        if self.phase is not None and "phase" not in record:
            record = {**record, "phase": self.phase}
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
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_every:
            self._file.flush()
            self._writes_since_flush = 0

    def log_record(self, record: dict):
        """Write an arbitrary metrics record."""
        self._write(record)

    def close(self):
        """Flush remaining buffered writes and close the output file."""
        self._file.flush()
        self._file.close()
