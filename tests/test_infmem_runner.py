from types import SimpleNamespace

from scripts.baselines.tier2.run_infmem import _normalize_usage, _set_upstream_env


def test_normalize_usage_labels_legacy_two_stage_finish_reasons():
    usage = _normalize_usage(
        {
            "requests": 4,
            "finish_reasons": ["length", "stop", "stop", "length"],
        },
        thinking_enabled=True,
    )

    assert usage["finish_reasons"] == [
        {"stage": "thinking", "reason": "length"},
        {"stage": "answer", "reason": "stop"},
        {"stage": "thinking", "reason": "stop"},
        {"stage": "answer", "reason": "length"},
    ]
    assert usage["thinking_length_cutoffs"] == 1
    assert usage["answer_length_cutoffs"] == 1
    assert usage["length_cutoffs"] == 2


def test_normalize_usage_preserves_current_stage_labels():
    reasons = [
        {"stage": "thinking", "reason": "length"},
        {"stage": "answer", "reason": "stop"},
    ]
    usage = _normalize_usage({"finish_reasons": reasons}, thinking_enabled=True)

    assert usage["finish_reasons"] == reasons
    assert usage["thinking_length_cutoffs"] == 1
    assert usage["answer_length_cutoffs"] == 0


def test_normalize_usage_without_thinking_labels_every_request_as_answer():
    usage = _normalize_usage(
        {"finish_reasons": ["stop", "length"]},
        thinking_enabled=False,
    )

    assert [entry["stage"] for entry in usage["finish_reasons"]] == ["answer", "answer"]
    assert usage["thinking_length_cutoffs"] == 0
    assert usage["answer_length_cutoffs"] == 1


def test_set_upstream_env_serializes_all_protocol_controls(monkeypatch):
    names = [
        "SERVE_HOST",
        "SERVE_PORT",
        "RECURRENT_MAX_CONTEXT_LEN",
        "RECURRENT_CHUNK_SIZE",
        "RECURRENT_MAX_NEW",
        "ENABLE_THINK",
        "EARLY_STOP",
        "BM25_IMPL",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)

    _set_upstream_env(
        SimpleNamespace(
            serve_host="127.0.0.2",
            serve_port=8123,
            max_context_tokens=2_000_000,
            chunk_tokens=5000,
            max_new_tokens=1024,
            enable_thinking=False,
            early_stop=3,
            bm25_impl="enhanced",
        )
    )

    import os

    assert {name: os.environ[name] for name in names} == {
        "SERVE_HOST": "127.0.0.2",
        "SERVE_PORT": "8123",
        "RECURRENT_MAX_CONTEXT_LEN": "2000000",
        "RECURRENT_CHUNK_SIZE": "5000",
        "RECURRENT_MAX_NEW": "1024",
        "ENABLE_THINK": "false",
        "EARLY_STOP": "3",
        "BM25_IMPL": "enhanced",
    }
