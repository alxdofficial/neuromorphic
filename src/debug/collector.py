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

        # PM commit rate accumulators
        self._pm_commit_accum: list = []  # [(total, count)]
        # EM write rate accumulators
        self._em_write_accum: list = []  # [(count, novelty_sum, g_sum)]

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

    def record_pm_commit(self, p_commit: Tensor):
        """Accumulate PM commit strength across passes within a chunk.

        p_commit is a continuous [BSB] tensor (0-1), not a binary mask.
        """
        self._pm_commit_accum.append(
            (p_commit.float().mean().detach(), 1)
        )

    def record_em_write(self, novelty_mean: Tensor | float, g_em_mean: Tensor | float):
        """Accumulate EM write stats across passes within a chunk."""
        if torch.is_tensor(novelty_mean):
            novelty = novelty_mean.detach()
        else:
            novelty = torch.tensor(float(novelty_mean))
        if torch.is_tensor(g_em_mean):
            g_em = g_em_mean.detach()
        else:
            g_em = torch.tensor(float(g_em_mean))

        self._em_write_accum.append((1, novelty, g_em))

    def _flush_rates(self, record: dict):
        """Flush accumulated commit/write rates into the record."""
        if self._pm_commit_accum:
            total = sum(t.item() if torch.is_tensor(t) else t for t, _ in self._pm_commit_accum)
            count = sum(c for _, c in self._pm_commit_accum)
            if count > 0:
                record["pm_commit_rate"] = total / count
        self._pm_commit_accum.clear()

        if self._em_write_accum:
            count = sum(c for c, _, _ in self._em_write_accum)
            if count > 0:
                nov_total = sum(
                    n.item() if torch.is_tensor(n) else n
                    for _, n, _ in self._em_write_accum
                )
                g_total = sum(
                    g.item() if torch.is_tensor(g) else g
                    for _, _, g in self._em_write_accum
                )
                record["em_novelty_mean"] = nov_total / count
                record["em_g_em_mean"] = g_total / count
        self._em_write_accum.clear()

    def _collect_pm_stats(self, record: dict):
        """Read PM state tensors and compute summary stats.

        v4: single PM with state [BS*B, r, D_col].
        """
        if not self.config.pm_enabled:
            return
        pm = self.model.pm
        if pm.pm_a is None:
            return
        pm_a = pm.pm_a.detach()  # [BS*B, r]
        record["pm_a_mean"] = pm_a.mean().item()
        record["pm_a_max"] = pm_a.max().item()
        record["pm_a_sum"] = pm_a.sum(dim=-1).mean().item()
        record["pm_nonzero"] = (pm_a > 0.01).float().mean().item()

    def _collect_em_stats(self, record: dict):
        """Read EM state tensors and compute summary stats."""
        if not self.config.em_enabled:
            return
        em = self.model.em
        if em.em_S is None:
            return
        em_S = em.em_S.detach()  # [BS*B, M]
        record["em_S_mean"] = em_S.mean().item()
        record["em_S_max"] = em_S.max().item()
        record["em_S_sum"] = em_S.sum(dim=-1).mean().item()
        record["em_nonzero"] = (em_S > 0.01).float().mean().item()

    def _collect_plasticity_summary(self, record: dict):
        """Global PM/EM summaries with warning flags."""
        pm_commit = record.get("pm_commit_rate")
        em_write = record.get("em_g_em_mean")

        record["warn_commit_collapse"] = float(
            pm_commit is not None and pm_commit < 1e-3
        )
        record["warn_write_collapse"] = float(
            em_write is not None and em_write < 1e-3
        )

        pm_budget = record.get("pm_budget_util_global")
        em_budget = record.get("em_budget_util_global")
        record["warn_budget_saturation"] = float(
            (pm_budget is not None and pm_budget > 0.98)
            or (em_budget is not None and em_budget > 0.98)
        )

    def _collect_lifelong_stats(self, record: dict):
        """Collect cross-document memory persistence stats (Phase B)."""
        pm = self.model.pm
        if pm.pm_a is not None:
            pm_a = pm.pm_a.detach()
            pm_nonzero = (pm_a > 0.01).float().sum().item()
            pm_slots = pm_a.numel()
            pm_budget_total = pm_a.sum().item()
            pm_budget_cap = pm.budget * pm_a.shape[0]

            if pm_slots > 0:
                record["pm_persistence"] = pm_nonzero / pm_slots
            if pm_budget_cap > 0:
                record["pm_budget_util"] = pm_budget_total / pm_budget_cap

        em = self.model.em
        if em.em_S is not None:
            em_S = em.em_S.detach()
            em_nonzero = (em_S > 0.01).float().sum().item()
            em_slots = em_S.numel()
            em_budget_total = em_S.sum().item()
            em_budget_cap = em.budget * em_S.shape[0]

            if em_slots > 0:
                record["em_persistence"] = em_nonzero / em_slots
            if em_budget_cap > 0:
                record["em_budget_util"] = em_budget_total / em_budget_cap

    def _collect_grad_norms(self, record: dict):
        """Per-module gradient norms after backward.

        v4 structure: model.embedding, model.lm_head, model.fan_in,
        model.columns, model.pm, model.em, model.pm_neuromod, model.em_neuromod
        """
        module_groups = {
            "embedding": [self.model.embedding],
            "lm_head": [self.model.lm_head],
            "fan_in": [self.model.fan_in],
            "columns": [self.model.columns],
        }
        if self.model.proj_up is not None:
            module_groups["proj_up"] = [self.model.proj_up]
            module_groups["proj_down"] = [self.model.proj_down]
        if self.config.pm_enabled:
            module_groups["pm"] = [self.model.pm]
            module_groups["pm_neuromod"] = [self.model.pm_neuromod]
        if self.config.em_enabled:
            module_groups["em"] = [self.model.em]
            module_groups["em_neuromod"] = [self.model.em_neuromod]

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
