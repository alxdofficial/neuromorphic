"""Efficiency and throughput metrics for model comparison."""

from .efficiency import (
    EfficiencyReport,
    compute_avg_bytes_per_token,
    compute_bpb,
    format_comparison_table,
    measure_flops_per_token,
    measure_inference_throughput,
    measure_training_throughput,
    measure_vram_breakdown,
    save_reports_json,
)

__all__ = [
    "EfficiencyReport",
    "compute_avg_bytes_per_token",
    "compute_bpb",
    "format_comparison_table",
    "measure_flops_per_token",
    "measure_inference_throughput",
    "measure_training_throughput",
    "measure_vram_breakdown",
    "save_reports_json",
]
