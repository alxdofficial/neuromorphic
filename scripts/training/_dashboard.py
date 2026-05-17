"""Periodic background plot rendering for the training scripts.

Adds a `--plot-every-secs` knob: when set, the training loop checks
wall-clock and fires `scripts/diagnostics/plot_v2.py` in a background
subprocess on that cadence. Non-blocking — training keeps running while
the plotter reads the (append-only) jsonl.

Each train script calls `maybe_render_dashboard(...)` once per logging
iteration; it no-ops until the cadence has elapsed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


_PLOT_SCRIPT = Path(__file__).resolve().parents[1] / "diagnostics" / "plot_v2.py"


class DashboardRenderer:
    """Track wall time between renders + spawn subprocess plotter calls."""

    def __init__(self, jsonl_path: str | Path, out_path: str | Path, every_secs: float):
        self.jsonl = Path(jsonl_path)
        self.out = Path(out_path)
        self.every_secs = float(every_secs)
        self.last_render = 0.0
        self._proc: subprocess.Popen | None = None

    def maybe_render(self) -> None:
        """Fire a render if enough wall time has passed AND no previous
        render is still running. Background subprocess; non-blocking."""
        if self.every_secs <= 0:
            return
        now = time.time()
        if now - self.last_render < self.every_secs:
            return
        # Don't pile up — skip if the previous render hasn't finished.
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.jsonl.exists():
            return
        self.last_render = now
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [sys.executable, str(_PLOT_SCRIPT),
             "--jsonl", str(self.jsonl), "--out", str(self.out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
