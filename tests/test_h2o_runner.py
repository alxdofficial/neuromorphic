"""Small no-model tests for H2O runner configuration and tokenizer plumbing."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.memory.eval.baselines import build_messages
from src.memory.eval.tier2_common import git_commit

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_kvcompress_h2o_test",
    REPO / "scripts/baselines/tier2/run_kvcompress.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class _Tokenizer:
    def __init__(self, result):
        self.result = result

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "question"}]
        assert kwargs == {"add_generation_prompt": True, "return_tensors": "pt"}
        return self.result


@pytest.mark.parametrize("wrapped", [False, True])
def test_chat_input_ids_accepts_transformers_4_and_5_return_types(wrapped):
    tensor = torch.tensor([[1, 2, 3]])
    result = {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)} if wrapped else tensor
    actual = RUNNER._chat_input_ids(_Tokenizer(result), [{"role": "user", "content": "question"}])
    assert actual is tensor


def test_default_h2o_budget_is_even_split():
    args = SimpleNamespace(max_capacity_prompt=2048, h2o_heavy_size=None, h2o_recent_size=None)
    assert RUNNER._h2o_budgets(args) == (1024, 1024)


def test_h2o_budget_must_sum_to_capacity():
    args = SimpleNamespace(max_capacity_prompt=2048, h2o_heavy_size=1000, h2o_recent_size=1000)
    with pytest.raises(ValueError, match="sum"):
        RUNNER._h2o_budgets(args)


def test_h2o_artifact_knobs_that_change_generation_are_distinct():
    def knob(head_mode="query_head", heavy=1024, recent=1024, chunk=128, revision="0e9e39f"):
        args = SimpleNamespace(
            max_capacity_prompt=2048,
            h2o_heavy_size=heavy,
            h2o_recent_size=recent,
            prefill_chunk_size=chunk,
            h2o_head_mode=head_mode,
            model_revision=revision,
        )
        return RUNNER._h2o_knob(args)

    signatures = {
        knob(),
        knob(head_mode="kv_head"),
        knob(heavy=512, recent=1536),
        knob(chunk=64),
        knob(revision="different-revision"),
    }
    assert len(signatures) == 5


def test_h2o_artifact_is_explicitly_position_rolling():
    args = SimpleNamespace(
        max_capacity_prompt=2048,
        h2o_heavy_size=1024,
        h2o_recent_size=1024,
        prefill_chunk_size=512,
        h2o_head_mode="query_head",
        model_revision="revision",
    )
    assert RUNNER._h2o_knob(args).startswith("rolling-")


def test_common_prefix_is_chunk_aligned_and_leaves_suffix():
    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    right = torch.tensor([[1, 2, 3, 4, 5, 9, 7, 8]])
    assert RUNNER._common_aligned_prefix(left, right, alignment=4) == 4

    identical = torch.tensor([[1, 2, 3, 4]])
    assert RUNNER._common_aligned_prefix(identical, identical, alignment=2) == 2


def test_mab_prompt_uses_native_template_without_generic_question_wrapper():
    item = {
        "question": "Where is the key?",
        "system": "Remember this context.",
        "question_template": "TASK: {question}\nAnswer:",
        "context_header": "# Context",
    }
    messages = RUNNER._benchmark_messages(build_messages, item, "The key is in Paris.", "memoryagentbench")

    assert messages[0] == {"role": "system", "content": "Remember this context."}
    assert messages[1]["content"] == "# Context\nThe key is in Paris.\n\nTASK: Where is the key?\nAnswer:"
    assert "# Question" not in messages[1]["content"]


def test_longmemeval_prompt_places_question_date_once():
    item = {"question": "What happened?", "question_date": "2025/01/02"}
    messages = RUNNER._benchmark_messages(build_messages, item, "A dated event.", "longmemeval")

    assert messages[1]["content"].count("Current Date: 2025/01/02") == 1
    assert messages[1]["content"].endswith("# Question\nWhat happened?\nAnswer:")


def test_dirty_signature_includes_untracked_file_contents(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)

    untracked = tmp_path / "adapter.py"
    untracked.write_text("version = 1\n")
    first = git_commit(tmp_path)
    untracked.write_text("version = 2\n")
    second = git_commit(tmp_path)
    assert first != second
