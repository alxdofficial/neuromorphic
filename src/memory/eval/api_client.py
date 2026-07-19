"""Async OpenRouter chat client for the Phase-2 API reference baselines.

Deterministic (temperature 0) generation with bounded concurrency, retry/backoff on transient errors,
and per-call token accounting so the runner can print an exact $ cost. No local GPU. The default model
panel and price table live here so every runner reports cost consistently.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default reference panel: cheap open small models + the cheapest genuinely-large frontier models,
# spread across vendors (Qwen / Meta / Google / DeepSeek). All fit ≥115k context. Override via --models.
DEFAULT_MODELS = [
    # small / cheap open models (all serve ≥128k so full-context holds a ~115k history)
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-flash-lite",
    # cheap large frontier (1M ctx)
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.5-flash-02-23",
    # large open/frontier (paid, no rate cap)
    "openai/gpt-oss-120b",
]

# $/token (prompt, completion) from OpenRouter /models, 2026-07 — for the cost read-out only.
PRICING = {
    "qwen/qwen-2.5-7b-instruct": (0.04e-6, 0.10e-6),
    "meta-llama/llama-3.1-8b-instruct": (0.05e-6, 0.08e-6),
    "meta-llama/llama-3.2-3b-instruct": (0.0509e-6, 0.335e-6),
    "google/gemini-2.5-flash-lite": (0.10e-6, 0.40e-6),
    "google/gemini-3.1-flash-lite": (0.25e-6, 1.00e-6),
    "deepseek/deepseek-v4-flash": (0.098e-6, 0.196e-6),
    "qwen/qwen3-235b-a22b-2507": (0.09e-6, 0.55e-6),
    "qwen/qwen3.7-plus": (0.32e-6, 1.28e-6),
    "qwen/qwen3.5-flash-02-23": (0.065e-6, 0.26e-6),
    "minimax/minimax-m3": (0.30e-6, 1.20e-6),
    "minimax/minimax-m2.5": (0.15e-6, 0.90e-6),
    "z-ai/glm-5.2": (0.343e-6, 1.078e-6),
    "z-ai/glm-4.7-flash": (0.061e-6, 0.40e-6),
    "openai/gpt-5-nano": (0.05e-6, 0.40e-6),
    "openai/gpt-4o-mini": (0.15e-6, 0.60e-6),
    "openai/gpt-5-mini": (0.25e-6, 2.00e-6),
    "openai/gpt-oss-120b": (0.037e-6, 0.17e-6),
    "tencent/hy3": (0.20e-6, 0.80e-6),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return prompt_tokens * pin + completion_tokens * pout


@dataclass
class CallResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    finish_reason: str | None = None      # "stop" | "length" | ... ; "length" ⇒ answer may be cut off


@dataclass
class Usage:
    """Running token/cost/error tally across a batch of calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_errors: int = 0
    _model: str = ""

    def add(self, r: CallResult) -> CallResult:
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        if r.error:
            self.n_errors += 1
        return r

    def cost(self) -> float:
        return cost_usd(self._model, self.prompt_tokens, self.completion_tokens)


class OpenRouterClient:
    """Thin async wrapper. Use as `async with OpenRouterClient() as c: await c.chat(...)`."""

    def __init__(self, api_key: str | None = None, concurrency: int = 8,
                 timeout: float = 180.0, retries: int = 5):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (pass api_key= or set the env var)")
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.retries = retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()

    async def chat(self, model: str, messages: list[dict], max_tokens: int = 256,
                   temperature: float = 0.0) -> CallResult:
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        last = "exhausted retries"
        for attempt in range(self.retries):
            retryable = False
            # hold a concurrency slot ONLY for the POST + parse — NOT during backoff, so a request sleeping
            # off a 429 doesn't stall the others (keeps effective concurrency up under rate-limiting).
            async with self.sem:
                try:
                    r = await self._client.post(OPENROUTER_URL, json=payload, timeout=self.timeout)
                except Exception as e:  # noqa: BLE001 — pre-response (network/timeout): safe to retry
                    last = f"{type(e).__name__}: {e}"
                    retryable = True
                else:
                    if r.status_code in (429, 500, 502, 503, 504):
                        last = f"HTTP {r.status_code}: {r.text[:120]}"
                        retryable = True
                    elif 400 <= r.status_code < 500:
                        # permanent client error (context-length 400, auth 401, not-found 404, 422 …):
                        # retrying can't fix it — surface immediately instead of burning the retry budget.
                        return CallResult("", 0, 0, f"HTTP {r.status_code}: {r.text[:200]}")
                    else:
                        try:
                            d = r.json()
                        except Exception as e:  # noqa: BLE001 — malformed 2xx body; the call ALREADY billed,
                            return CallResult("", 0, 0, f"bad-json: {type(e).__name__}: {e}")  # so do NOT retry
                        # OpenRouter can return HTTP 200 with a top-level {"error": ...} and no choices
                        # (provider/moderation failures). That's an ERROR, not a wrong answer — surface it so
                        # it's EXCLUDED from scoring, not silently counted as an empty (wrong) response.
                        if d.get("error") or not d.get("choices"):
                            return CallResult("", 0, 0, error=str(d.get("error") or "200 with no choices"))
                        choice = d["choices"][0]
                        # reasoning models put chain-of-thought in `reasoning`; the ANSWER is in `content`.
                        # `message` itself can be JSON null (content filter) → guard both levels (a bare
                        # .get on None would raise AttributeError → retry → REBILL the 200).
                        msg = ((choice.get("message") or {}).get("content")) or ""
                        u = d.get("usage") or {}
                        return CallResult(msg, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                                          None, choice.get("finish_reason"))
            if retryable and attempt < self.retries - 1:
                await asyncio.sleep(2 ** attempt)          # backoff OUTSIDE the semaphore (slot released)
        return CallResult("", 0, 0, last)

    async def map(self, model: str, message_lists: list[list[dict]], max_tokens: int = 256):
        """Run many chat calls concurrently (bounded by the semaphore); returns list[CallResult]."""
        return await asyncio.gather(*[self.chat(model, m, max_tokens) for m in message_lists])
