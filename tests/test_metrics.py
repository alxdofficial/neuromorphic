"""Tests for src.metrics.efficiency — throughput and efficiency measurement utilities."""

import json
import math
from dataclasses import asdict

import torch
import torch.nn as nn
import pytest

from tests.conftest import make_tiny_config
from src.model.model import NeuromorphicLM
from src.metrics.efficiency import (
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


BS = 2
VOCAB = 64

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_and_fwd():
    """Create tiny neuromorphic model + forward fn for testing."""
    cfg = make_tiny_config(use_compile=False)
    model = NeuromorphicLM(cfg)
    model.initialize_states(BS, torch.device("cpu"))

    def fwd(m, ids):
        logits, _aux = m.forward_segment(ids)
        return logits

    return model, fwd


class _MockTokenizer:
    """Tokenizer that splits on spaces (for testing avg_bytes_per_token)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


# ---------------------------------------------------------------------------
# CPU tests — FLOPs
# ---------------------------------------------------------------------------

class TestFlops:
    def test_flops_counts_positive(self):
        model, fwd = _make_model_and_fwd()
        input_ids = torch.randint(0, VOCAB, (BS, 16))
        result = measure_flops_per_token(model, input_ids, fwd)
        assert result["total_flops"] > 0
        assert result["flops_per_token"] > 0

    def test_flops_simple_linear(self):
        """nn.Linear(D, D) should yield 2*D*D FLOPs per element."""
        D = 64

        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(D, D, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel()
        # Shape: [BS, N, D] — treat BS*N as token count
        x = torch.randn(BS, 16, D)
        result = measure_flops_per_token(model, x, lambda m, inp: m(inp))

        expected_total = 2 * D * D * BS * 16
        assert result["total_flops"] == expected_total, (
            f"got {result['total_flops']}, expected {expected_total}"
        )


# ---------------------------------------------------------------------------
# CPU tests — BPB
# ---------------------------------------------------------------------------

class TestBPB:
    def test_bpb_identity(self):
        """compute_bpb(ln(2), 1.0) should equal 1.0."""
        result = compute_bpb(math.log(2), 1.0)
        assert abs(result - 1.0) < 1e-9

    def test_bpb_zero_loss(self):
        """compute_bpb(0.0, X) should equal 0.0."""
        assert compute_bpb(0.0, 3.5) == 0.0
        assert compute_bpb(0.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# CPU tests — avg bytes per token
# ---------------------------------------------------------------------------

class TestAvgBytesPerToken:
    def test_known_text(self):
        """Known text + mock tokenizer -> expected range."""
        tok = _MockTokenizer()
        text = "hello world foo bar"  # 4 words -> 4 tokens, 19 bytes
        result = compute_avg_bytes_per_token(tok, [text])
        expected = len(text.encode("utf-8")) / 4
        assert abs(result - expected) < 1e-9

    def test_default_sample(self):
        """Default sample should produce a reasonable range with mock tokenizer."""
        tok = _MockTokenizer()
        result = compute_avg_bytes_per_token(tok)
        # English text: typically 4-8 bytes per space-delimited word
        assert 3.0 <= result <= 10.0

    def test_empty_tokens(self):
        """If tokenizer returns no tokens, should return 1.0 (safe fallback)."""

        class EmptyTokenizer:
            def encode(self, text):
                return []

        result = compute_avg_bytes_per_token(EmptyTokenizer(), ["hello"])
        assert result == 1.0


# ---------------------------------------------------------------------------
# CPU tests — EfficiencyReport
# ---------------------------------------------------------------------------

class TestEfficiencyReport:
    def test_fields_roundtrip(self):
        """Dataclass should roundtrip through asdict."""
        report = EfficiencyReport(
            model_name="test",
            param_count=1000,
            train_tok_per_sec=5000.0,
            infer_tok_per_sec=10000.0,
            flops_per_token_fwd=1_000_000,
            peak_vram_train_gb=1.5,
            vram_weights_gb=0.5,
            vram_optimizer_gb=0.5,
            vram_activations_gb=0.5,
            batch_size=8,
            seq_len=128,
            device_name="cpu",
            bpb=1.23,
        )
        d = asdict(report)
        assert d["model_name"] == "test"
        assert d["param_count"] == 1000
        assert d["bpb"] == 1.23

    def test_bpb_optional(self):
        """BPB should default to None."""
        report = EfficiencyReport(
            model_name="test",
            param_count=1000,
            train_tok_per_sec=0.0,
            infer_tok_per_sec=0.0,
            flops_per_token_fwd=0,
            peak_vram_train_gb=0.0,
            vram_weights_gb=0.0,
            vram_optimizer_gb=0.0,
            vram_activations_gb=0.0,
            batch_size=1,
            seq_len=1,
            device_name="cpu",
        )
        assert report.bpb is None


# ---------------------------------------------------------------------------
# CPU tests — format / output
# ---------------------------------------------------------------------------

class TestFormatTable:
    def test_contains_model_names(self):
        reports = [
            EfficiencyReport(
                model_name="ModelA",
                param_count=100_000,
                train_tok_per_sec=1000.0,
                infer_tok_per_sec=2000.0,
                flops_per_token_fwd=500_000,
                peak_vram_train_gb=1.0,
                vram_weights_gb=0.3,
                vram_optimizer_gb=0.3,
                vram_activations_gb=0.4,
                batch_size=8,
                seq_len=128,
                device_name="cpu",
            ),
            EfficiencyReport(
                model_name="ModelB",
                param_count=200_000,
                train_tok_per_sec=800.0,
                infer_tok_per_sec=1600.0,
                flops_per_token_fwd=900_000,
                peak_vram_train_gb=2.0,
                vram_weights_gb=0.6,
                vram_optimizer_gb=0.6,
                vram_activations_gb=0.8,
                batch_size=4,
                seq_len=128,
                device_name="cpu",
                bpb=1.5,
            ),
        ]
        table = format_comparison_table(reports)
        assert "ModelA" in table
        assert "ModelB" in table
        assert "Train tok/s" in table

    def test_empty_reports(self):
        assert format_comparison_table([]) == "(no reports)"


class TestSaveJson:
    def test_roundtrip(self, tmp_path):
        report = EfficiencyReport(
            model_name="test",
            param_count=1000,
            train_tok_per_sec=5000.0,
            infer_tok_per_sec=10000.0,
            flops_per_token_fwd=1_000_000,
            peak_vram_train_gb=1.5,
            vram_weights_gb=0.5,
            vram_optimizer_gb=0.5,
            vram_activations_gb=0.5,
            batch_size=8,
            seq_len=128,
            device_name="cpu",
        )
        path = tmp_path / "reports.json"
        save_reports_json([report], path)
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["model_name"] == "test"


# ---------------------------------------------------------------------------
# CUDA tests (skipped without GPU)
# ---------------------------------------------------------------------------

@requires_cuda
class TestTrainingThroughputCUDA:
    def test_positive_throughput(self):
        cfg = make_tiny_config(use_compile=False)
        model = NeuromorphicLM(cfg).to("cuda").to(torch.bfloat16)
        model.initialize_states(BS, torch.device("cuda"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def fwd(m, ids):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _aux = m.forward_segment(ids)
                return logits

        result = measure_training_throughput(
            model, fwd, optimizer,
            bs=BS, seq_len=16, vocab=VOCAB,
            device=torch.device("cuda"),
            warmup=2, measure=3,
            detach_fn=lambda m: m.detach_states(),
        )
        assert result["tok_per_sec"] > 0
        assert result["ms_per_step"] > 0
        assert result["peak_vram_gb"] > 0


@requires_cuda
class TestInferenceThroughputCUDA:
    def test_positive_throughput(self):
        cfg = make_tiny_config(use_compile=False)
        model = NeuromorphicLM(cfg).to("cuda").to(torch.bfloat16)
        model.initialize_states(BS, torch.device("cuda"))

        def fwd(m, ids):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _aux = m.forward_segment(ids)
                return logits

        result = measure_inference_throughput(
            model, fwd,
            bs=BS, seq_len=16, vocab=VOCAB,
            device=torch.device("cuda"),
            warmup=2, measure=3,
            detach_fn=lambda m: m.detach_states(),
        )
        assert result["tok_per_sec"] > 0
        assert result["ms_per_step"] > 0


@requires_cuda
class TestVRAMBreakdownCUDA:
    def test_sums_approximately(self):
        cfg = make_tiny_config(use_compile=False)
        model = NeuromorphicLM(cfg).to("cuda").to(torch.bfloat16)
        model.initialize_states(BS, torch.device("cuda"))

        def fwd(m, ids):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _aux = m.forward_segment(ids)
                return logits

        result = measure_vram_breakdown(
            model, fwd, torch.optim.AdamW,
            bs=BS, seq_len=16, vocab=VOCAB,
            device=torch.device("cuda"),
            detach_fn=lambda m: m.detach_states(),
        )
        # weights + optimizer + activations should approximately equal peak
        summed = result["weights_gb"] + result["optimizer_gb"] + result["activations_gb"]
        # Allow some tolerance (CUDA allocator granularity)
        assert abs(summed - result["peak_gb"]) < 0.1, (
            f"sum={summed:.4f} vs peak={result['peak_gb']:.4f}"
        )


class TestPredTokPerSecField:
    def test_predicted_tok_per_sec_field(self):
        """EfficiencyReport.predicted_tok_per_sec defaults to 0 and is settable."""
        r = EfficiencyReport(
            model_name="test", param_count=100,
            train_tok_per_sec=1000.0, infer_tok_per_sec=0.0,
            flops_per_token_fwd=0, peak_vram_train_gb=0.0,
            vram_weights_gb=0.0, vram_optimizer_gb=0.0,
            vram_activations_gb=0.0, batch_size=4, seq_len=16,
            device_name="cpu", predicted_tok_per_sec=500.0,
        )
        assert r.predicted_tok_per_sec == 500.0

        r2 = EfficiencyReport(
            model_name="test-ntp", param_count=100,
            train_tok_per_sec=1000.0, infer_tok_per_sec=0.0,
            flops_per_token_fwd=0, peak_vram_train_gb=0.0,
            vram_weights_gb=0.0, vram_optimizer_gb=0.0,
            vram_activations_gb=0.0, batch_size=4, seq_len=16,
            device_name="cpu",
        )
        assert r2.predicted_tok_per_sec == 0.0


class TestFormatTablePredTok:
    def test_pred_tok_column_appears(self):
        """Pred tok/s column appears when any report has predicted_tok_per_sec > 0."""
        reports = [
            EfficiencyReport(
                model_name="Baseline", param_count=100_000,
                train_tok_per_sec=1000.0, infer_tok_per_sec=2000.0,
                flops_per_token_fwd=500_000, peak_vram_train_gb=1.0,
                vram_weights_gb=0.3, vram_optimizer_gb=0.3,
                vram_activations_gb=0.4, batch_size=8, seq_len=128,
                device_name="cpu",
            ),
            EfficiencyReport(
                model_name="Neuro", param_count=200_000,
                train_tok_per_sec=800.0, infer_tok_per_sec=1600.0,
                flops_per_token_fwd=900_000, peak_vram_train_gb=2.0,
                vram_weights_gb=0.6, vram_optimizer_gb=0.6,
                vram_activations_gb=0.8, batch_size=4, seq_len=128,
                device_name="cpu", predicted_tok_per_sec=300.0,
            ),
        ]
        table = format_comparison_table(reports)
        assert "Pred tok/s" in table
        assert "300" in table

    def test_no_pred_tok_column_when_zero(self):
        """Pred tok/s column absent when all reports have predicted_tok_per_sec == 0."""
        reports = [
            EfficiencyReport(
                model_name="Baseline", param_count=100_000,
                train_tok_per_sec=1000.0, infer_tok_per_sec=2000.0,
                flops_per_token_fwd=500_000, peak_vram_train_gb=1.0,
                vram_weights_gb=0.3, vram_optimizer_gb=0.3,
                vram_activations_gb=0.4, batch_size=8, seq_len=128,
                device_name="cpu",
            ),
        ]
        table = format_comparison_table(reports)
        assert "Pred tok/s" not in table
