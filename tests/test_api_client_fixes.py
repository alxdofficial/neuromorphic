"""Regression tests for the OpenRouter client fixes (offline, stubbed HTTP — no network).

Covers the pre-run adversarial findings: a 200 error-body scored as a wrong answer (#1), a null `message`
crashing → rebilled retries (#2), and the happy/4xx paths.
"""
import asyncio

from src.memory.eval.api_client import OpenRouterClient


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _FakeHTTP:
    def __init__(self, resp):
        self.resp = resp
        self.posts = 0

    async def post(self, url, json=None, timeout=None):
        self.posts += 1
        return self.resp


def _run(payload=None, status=200, text=""):
    c = OpenRouterClient(api_key="test", concurrency=2, retries=3)
    fake = _FakeHTTP(_FakeResp(status, payload, text))
    c._client = fake
    res = asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], max_tokens=10))
    return res, fake.posts


def test_happy_path():
    res, posts = _run(payload={"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    assert res.text == "Paris" and res.error is None and res.finish_reason == "stop"
    assert res.prompt_tokens == 5 and res.completion_tokens == 1 and posts == 1


def test_200_error_body_is_error_not_wrong_answer():
    # #1: HTTP 200 carrying {"error": ...} with no choices → surfaced as ERROR (excluded from scoring),
    # NOT a scored empty (wrong) answer that silently deflates accuracy.
    res, posts = _run(payload={"error": {"message": "provider failed", "code": 500}})
    assert res.error is not None and res.text == ""
    assert posts == 1                                  # not retried


def test_200_no_choices_is_error():
    res, posts = _run(payload={"choices": [], "usage": {}})
    assert res.error is not None and posts == 1


def test_200_null_message_content_filter_is_error():
    # audit #2: message == JSON null with a TERMINAL finish_reason (content_filter) → surfaced as ERROR
    # (excluded from scoring), not a silent empty answer. Must not crash and must NOT rebill (no retry).
    res, posts = _run(payload={"choices": [{"message": None, "finish_reason": "content_filter"}], "usage": {}})
    assert res.error is not None and res.text == "" and res.finish_reason == "content_filter"
    assert posts == 1                                  # NOT retried


def test_200_null_message_nonterminal_is_graceful_empty():
    # message == null with a NORMAL finish_reason must still not crash → graceful empty, error None, no rebill.
    res, posts = _run(payload={"choices": [{"message": None, "finish_reason": "stop"}], "usage": {}})
    assert res.error is None and res.text == "" and posts == 1


def test_200_null_content_ok():
    res, posts = _run(payload={"choices": [{"message": {"content": None}, "finish_reason": "length"}],
                               "usage": {}})
    assert res.text == "" and res.finish_reason == "length" and posts == 1


def test_4xx_no_retry():
    res, posts = _run(status=401, text="User not found")
    assert res.error and "401" in res.error and posts == 1
