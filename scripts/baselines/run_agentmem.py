#!/usr/bin/env python3
"""Phase-2 2b baseline: agent-memory frameworks (A-MEM / MemoryOS) over LongMemEval / MemoryAgentBench.

The agent-memory orchestration is local, while both LLM roles run over OpenRouter. MiniLM can either run in
each worker on CPU or, for a parallel fleet, once on the local GPU behind the shared embedding service. By
default Mistral Nemo/DekaLLM is the strict-schema A-MEM controller (metadata, evolution, and query rewriting),
and Llama 3.1 8B/DeepInfra is the shared Phase-2 final-answer reader. The 2b "agent-memory paradigm" reference
(docs/baselines/PHASE2_BASELINES.md §2.5): an external memory store + retrieval bolted onto a frozen chat LLM.

  --method a-mem     A-MEM (Xu et al., NeurIPS'25; github.com/WujiangXu/A-mem, MIT). `AgenticMemorySystem`:
                     add_note(text) ingests (LLM-generates keywords/tags + evolves links), then its evaluation
                     flow rewrites the query and expands linked neighbors; WE generate the benchmark-standard
                     answer through the separately configured OpenRouter reader role.
  --method memoryos  MemoryOS (Kang et al., EMNLP'25; pip memoryos-pro). `Memoryos`: add_memory then
                     get_response(q) which retrieves AND generates internally. ⚠ UNTESTED here (needs its pip
                     package); get_response may mutate state, so per-context reuse is a POD-VERIFY for it.

PER-CONTEXT REUSE (docs/baselines/TIER2_HOSTING.md, via src/memory/eval/tier2_common.run_grouped): build the
memory store ONCE per distinct context and answer every question sharing it — the local prompt-cache analog.
MAB = 36 ingests for 3,071 Q (retrieval is read-only, safe to reuse); LongMemEval = ingest per question
(unique histories). Same deterministic scorers + JSON shape as Tier-1.

`--help` works without the framework installed (lazy import). Running needs the cloned A-mem repo
(`--repo-dir`, default ~/tier2_repos/A-mem) + `OPENROUTER_API_KEY`. RESUMABLE + crash-safe.

Example:  OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --dataset longmemeval --max-examples 5

Shared GPU embeddings (start once, then point every lightweight worker at it):
  python scripts/baselines/amem_embedding_service.py --device cuda --port 8765
  OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --dataset longmemeval \
      --embedding-service-url http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_CONTROLLER_LLM = "mistralai/mistral-nemo"
_DEFAULT_CONTROLLER_PROVIDER = "dekallm"
_DEFAULT_ANSWER_LLM = "meta-llama/llama-3.1-8b-instruct"       # shared Phase-2 reader
_DEFAULT_ANSWER_PROVIDER = "deepinfra"
_DEFAULT_EMBED = "all-MiniLM-L6-v2"
_DEFAULT_MAB_NOTE_CHARS = 800  # leaves room for generated attributes under MiniLM's 256-token limit
_DEFAULT_ANSWER_TOKENS = {"longmemeval": 1024, "memoryagentbench": 256}
_DEFAULT_REPO_DIR = str(REPO.parent / "baselines" / "A-mem")  # local master/baselines; git clone github.com/WujiangXu/A-mem

_QUERY_KEYWORDS_PROMPT = """Given the following question, generate several keywords, using 'cosmos' as the separator.

Question: {question}

Format your response as a JSON object with a "keywords" field containing the selected text.

Example response format:
{{"keywords": "keyword1, keyword2, keyword3"}}"""
_QUERY_KEYWORDS_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "response",
    "schema": {
        "type": "object",
        "properties": {"keywords": {"type": "string"}},
        "required": ["keywords"],
        "additionalProperties": False,
    },
    "strict": True,
}}
_TURN_RE = re.compile(r"(?m)^(User|Assistant):\s*")
_SESSION_DATE_RE = re.compile(r"^\[Session .*?—\s*([^\]]+)\]\s*", re.DOTALL)


class _UsageMeter:
    """Aggregate API-reported token use by A-MEM phase without changing upstream responses."""

    def __init__(self):
        self.rows = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                         "cached_prompt_tokens": 0, "reported_cost_usd": 0.0})

    def add(self, phase: str, usage) -> None:
        if usage is None:
            return
        get = (lambda k, default=0: usage.get(k, default)) if isinstance(usage, dict) \
            else (lambda k, default=0: getattr(usage, k, default))
        row = self.rows[phase]
        row["calls"] += 1
        row["prompt_tokens"] += int(get("prompt_tokens", 0) or 0)
        row["completion_tokens"] += int(get("completion_tokens", 0) or 0)
        details = get("prompt_tokens_details", None)
        if details is not None:
            cached = details.get("cached_tokens", 0) if isinstance(details, dict) \
                else getattr(details, "cached_tokens", 0)
            row["cached_prompt_tokens"] += int(cached or 0)
        # OpenRouter includes the provider charge in usage when accounting is enabled. Keep it alongside the
        # portable token counts; callers can still recompute at a fixed price for apples-to-apples reporting.
        row["reported_cost_usd"] += float(get("cost", 0.0) or 0.0)

    def as_dict(self) -> dict:
        out = {k: dict(v) for k, v in sorted(self.rows.items())}
        out["TOTAL"] = {
            key: sum(v[key] for v in self.rows.values())
            for key in ("calls", "prompt_tokens", "completion_tokens", "cached_prompt_tokens",
                        "reported_cost_usd")
        }
        return out


def _validate_openrouter_key(api_key: str) -> None:
    """Fail before loading models/data when OpenRouter has already declared the supplied key invalid."""
    import httpx
    try:
        response = httpx.get(f"{_OPENROUTER_BASE}/key",
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=20.0)
    except Exception as exc:  # a transient preflight failure must not prevent the real retrying calls
        print(f"[run_agentmem] WARN: OpenRouter key preflight unavailable: {type(exc).__name__}: {exc}")
        return
    if response.status_code in (401, 403):
        raise RuntimeError(f"OpenRouter rejected OPENROUTER_API_KEY (HTTP {response.status_code})")


def _strict_schema_provider_issue(endpoint_payload: dict, provider: str) -> str | None:
    """Return an actionable incompatibility reason from OpenRouter's public endpoint metadata."""
    wanted = re.sub(r"[^a-z0-9]", "", provider.lower())
    endpoints = (endpoint_payload.get("data") or {}).get("endpoints") or []
    matching = [ep for ep in endpoints
                if re.sub(r"[^a-z0-9]", "", str(ep.get("provider_name", "")).lower()) == wanted]
    if not matching:
        return f"no current endpoint named {provider!r}"
    if not any("structured_outputs" in (ep.get("supported_parameters") or []) for ep in matching):
        return (f"provider {provider!r} does not advertise strict structured_outputs required by A-MEM's "
                "json_schema calls")
    return None


def _answer_provider_issue(endpoint_payload: dict, provider: str) -> str | None:
    """Final answers need a real endpoint but do not require structured-output support."""
    wanted = re.sub(r"[^a-z0-9]", "", provider.lower())
    endpoints = (endpoint_payload.get("data") or {}).get("endpoints") or []
    if not any(re.sub(r"[^a-z0-9]", "", str(ep.get("provider_name", "")).lower()) == wanted
               for ep in endpoints):
        return f"no current endpoint named {provider!r}"
    return None


def _validate_amem_provider(model: str, provider: str | None) -> None:
    """Fail before model/data loading when a pinned provider cannot run upstream A-MEM's strict schemas."""
    if not provider:
        return
    import httpx
    try:
        response = httpx.get(f"{_OPENROUTER_BASE}/models/{model}/endpoints", timeout=20.0)
        response.raise_for_status()
        issue = _strict_schema_provider_issue(response.json(), provider)
    except httpx.HTTPError as exc:
        print(f"[run_agentmem] WARN: provider capability preflight unavailable: {type(exc).__name__}: {exc}")
        return
    if issue:
        raise RuntimeError(f"OpenRouter provider pin is incompatible: {issue}")


def _validate_answer_provider(model: str, provider: str | None) -> None:
    """Fail early if the pinned final reader endpoint does not exist for the selected model."""
    if not provider:
        return
    import httpx
    try:
        response = httpx.get(f"{_OPENROUTER_BASE}/models/{model}/endpoints", timeout=20.0)
        response.raise_for_status()
        issue = _answer_provider_issue(response.json(), provider)
    except httpx.HTTPError as exc:
        print(f"[run_agentmem] WARN: answer-provider preflight unavailable: {type(exc).__name__}: {exc}")
        return
    if issue:
        raise RuntimeError(f"OpenRouter answer provider pin is incompatible: {issue}")


def _phase_for_internal_prompt(prompt: str) -> str:
    if "Generate a structured analysis of the following content" in (prompt or ""):
        return "metadata"
    if "memory evolution agent" in (prompt or ""):
        return "evolution"
    if "generate several keywords" in (prompt or ""):
        return "query_keywords"
    return "internal_other"


def _json_schema_issue(value, schema: dict, path: str = "$") -> str | None:
    """Return why a decoded value violates the strict-schema subset used by upstream A-MEM."""
    expected = schema.get("type")
    type_ok = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: type(v) is int,
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if expected in type_ok and not type_ok[expected](value):
        return f"{path} must be {expected}, got {type(value).__name__}"
    if expected == "object":
        for key in schema.get("required", []):
            if key not in value:
                return f"{path} missing required property {key!r}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                return f"{path} has unexpected properties {sorted(extra)!r}"
        for key, child_schema in properties.items():
            if key in value:
                issue = _json_schema_issue(value[key], child_schema, f"{path}.{key}")
                if issue:
                    return issue
    elif expected == "array" and schema.get("items"):
        for idx, item in enumerate(value):
            issue = _json_schema_issue(item, schema["items"], f"{path}[{idx}]")
            if issue:
                return issue
    return None


def _structured_response_issue(response, response_format: dict | None) -> str | None:
    """Reject syntactically valid but schema-invalid OpenRouter responses (notably JSON ``null``)."""
    if not response_format or response_format.get("type") != "json_schema":
        return None
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return "response has no first message content"
    if not isinstance(content, str):
        return f"message content must be str, got {type(content).__name__}"
    cleaned = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"message content is not JSON: {exc}"
    schema = (response_format.get("json_schema") or {}).get("schema") or {}
    return _json_schema_issue(value, schema)


def _attach_usage_meter(mem, meter: _UsageMeter, provider: str | None = None,
                        retry_max_tokens: int = 4000) -> None:
    """Meter/retry upstream calls and, when requested, pin every call to one OpenRouter provider."""
    completions = mem.llm_controller.llm.client.chat.completions
    original = completions.create

    def create(*args, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        phase = _phase_for_internal_prompt(prompt)
        if provider:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
            kwargs["extra_body"] = extra_body
        for attempt in range(5):
            try:
                response = original(*args, **kwargs)
                meter.add(phase, getattr(response, "usage", None))
                issue = _structured_response_issue(response, kwargs.get("response_format"))
                if issue:
                    try:
                        finish_reason = getattr(response.choices[0], "finish_reason", None)
                    except (AttributeError, IndexError, TypeError):
                        finish_reason = None
                    if finish_reason == "length":
                        current = int(kwargs.get("max_tokens") or 1000)
                        kwargs["max_tokens"] = min(max(current * 2, current + 1), retry_max_tokens)
                    raise ValueError(f"schema-invalid controller response: {issue}")
                return response
            except Exception as exc:  # noqa: BLE001 — retry transient transport and malformed responses
                if attempt == 4:
                    raise
                print(f"[run_agentmem] controller retry phase={phase} attempt={attempt + 1}/5: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                time.sleep(2 ** attempt)

    completions.create = create


def _longmemeval_turn_units(session: str) -> list[tuple[str, str | None]]:
    """Map a rendered session to upstream A-MEM's one-note-per-conversation-turn protocol."""
    mt = _SESSION_DATE_RE.match(session or "")
    when = mt.group(1).strip() if mt else None
    body = (session or "")[mt.end():] if mt else (session or "")
    matches = list(_TURN_RE.finditer(body))
    units: list[tuple[str, str | None]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[match.end():end].strip()
        if content:
            units.append((f"Speaker {match.group(1)} says : {content}", when))
    return units


def _amem_ingest_units(dataset: str, ctx: str, first_item: dict, chunk_chars: int,
                       ingest_granularity: str = "turn") -> list[tuple[str, str | None]]:
    if dataset == "longmemeval":
        sessions = first_item.get("sessions") or [ctx]
        if ingest_granularity == "session":
            out = []
            for sess in sessions:
                mt = _SESSION_DATE_RE.match(sess or "")
                out.append((sess, mt.group(1).strip() if mt else None))
            return out
        return [unit for sess in sessions for unit in _longmemeval_turn_units(sess)]
    from src.memory.data.memoryagentbench import _chunk
    return [(chunk, None) for chunk in _chunk(ctx, chunk_chars=chunk_chars)]


def _generate_query_keywords(mem, question: str) -> str:
    """Upstream `advancedMemAgent.generate_query_llm`, with a safe parse fallback."""
    response = mem.llm_controller.llm.get_completion(
        _QUERY_KEYWORDS_PROMPT.format(question=question), response_format=_QUERY_KEYWORDS_SCHEMA)
    try:
        keywords = str(json.loads((response or "").strip()).get("keywords", "")).strip()
        return keywords or question
    except (json.JSONDecodeError, AttributeError, TypeError):
        return (response or "").strip() or question


def _drop_invalid_amem_links(mem) -> int:
    """Discard only learned links that cannot name an existing note.

    Upstream asks the controller for integer ``suggested_connections`` but does not constrain their range.
    A hallucinated index is stored verbatim and later crashes ``find_related_memories_raw``.  Existing note
    indices are stable because A-MEM appends to an insertion-ordered dict and never deletes notes, so filtering
    to that range preserves every meaningful learned edge while treating non-existent targets as no edge.
    """
    memories = list(getattr(mem, "memories", {}).values())
    n_memories = len(memories)
    dropped = 0
    for note in memories:
        links = list(getattr(note, "links", []) or [])
        valid = [link for link in links if type(link) is int and 0 <= link < n_memories]
        dropped += len(links) - len(valid)
        note.links = valid
    return dropped


def _retrieve_a_mem(mem, question: str, k: int, query_mode: str, expand_links: bool):
    query = _generate_query_keywords(mem, question) if query_mode == "upstream_keywords" else question
    if expand_links:
        return query, mem.find_related_memories_raw(query, k=k)
    context, _indices = mem.find_related_memories(query, k=k)
    return query, context


def _answer_messages(build_messages, it: dict, dataset: str, context: str) -> list[dict]:
    """Use each benchmark's native answer prompt; notably, do not double-wrap MAB's task template."""
    kwargs = {"full_history": context or "", "char_budget": 10 ** 9}
    if dataset == "memoryagentbench":
        kwargs.update(system=it.get("system"), question_template=it.get("question_template"),
                      context_header=it.get("context_header") or "# Context")
    else:
        kwargs["question_date"] = it.get("question_date")
    return build_messages("full_context", question=it["question"], **kwargs)[0]


def _answer_token_cap(dataset: str, requested: int | None) -> int:
    """Dataset-aware default derived from gold and prior Llama output lengths; explicit CLI always wins."""
    cap = _DEFAULT_ANSWER_TOKENS[dataset] if requested is None else requested
    if cap < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    return cap


def _resolve_amem_llm_roles(args) -> None:
    """Resolve split-role defaults while retaining the old one-model CLI as an explicit compatibility path."""
    if args.llm_model and (args.controller_model or args.answer_model):
        raise ValueError("--llm-model cannot be combined with --controller-model/--answer-model")
    if args.openrouter_provider and (args.controller_provider or args.answer_provider):
        raise ValueError("--openrouter-provider cannot be combined with --controller-provider/--answer-provider")
    if args.llm_model:
        args.controller_model = args.answer_model = args.llm_model
    else:
        args.controller_model = args.controller_model or _DEFAULT_CONTROLLER_LLM
        args.answer_model = args.answer_model or _DEFAULT_ANSWER_LLM
    if args.openrouter_provider:
        args.controller_provider = args.answer_provider = args.openrouter_provider
    else:
        args.controller_provider = args.controller_provider or _DEFAULT_CONTROLLER_PROVIDER
        args.answer_provider = args.answer_provider or _DEFAULT_ANSWER_PROVIDER


def _amem_knob_tag(args) -> str:
    emb = hashlib.md5(args.embed_model.encode()).hexdigest()[:6]
    models = hashlib.md5(f"{args.controller_model}\0{args.answer_model}".encode()).hexdigest()[:8]
    controller_provider = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.controller_provider or "auto")
    answer_provider = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.answer_provider or "auto")
    embedding_backend = "svc" if args.embedding_service_url else "local"
    service_health = getattr(args, "embedding_service_health", {}) or {}
    embedding_device = service_health.get("device", args.embed_device) if args.embedding_service_url \
        else args.embed_device
    shard = f"-sh{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""
    return (f"k{args.retrieve_k}-ing{args.ingest_granularity}-q{args.query_mode}-"
            f"links{int(not args.no_link_expansion)}-emb{emb}-dev{embedding_device}-"
            f"c{args.ingest_chunk_chars}-eb{embedding_backend}-cr{args.controller_retry_max_tokens}-m{models}-"
            f"cp{controller_provider}-ap{answer_provider}{shard}")


def _shard_items(items: list[dict], dataset: str, args) -> tuple[list[dict], list[int]]:
    """LPT-partition whole contexts by A-MEM call count, preserving MAB's ingest-once reuse."""
    from src.memory.eval.tier2_common import group_by_context
    groups = group_by_context(items)
    weighted = []
    for ctx, group in groups.items():
        notes = len(_amem_ingest_units(dataset, ctx, group[0], args.ingest_chunk_chars,
                                       args.ingest_granularity))
        q_calls = 2 if args.query_mode == "upstream_keywords" else 1
        weighted.append((2 * notes + q_calls * len(group), ctx, group))
    if args.num_shards == 1:
        return items, [sum(row[0] for row in weighted)]
    loads = [0] * args.num_shards
    assigned: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for work, ctx, group in sorted(weighted, key=lambda row: (-row[0], row[1])):
        del ctx
        shard = min(range(args.num_shards), key=lambda i: (loads[i], i))
        assigned[shard].extend(group)
        loads[shard] += work
    return assigned[args.shard_idx], loads


def _openrouter_chat(model: str, messages: list[dict], api_key: str, max_tokens: int,
                     provider: str | None = None):
    """Synchronous OpenRouter chat (the answer step for A-MEM), with the Tier-1 safeguards (audit #7): retry
    on 429/5xx with backoff, treat a choice-level error or terminal finish_reason (error/content_filter) as
    an ERROR not an empty answer, and surface a length cutoff. Returns (text, error, finish_reason, usage)."""
    import time
    import httpx
    last = "exhausted retries"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if provider:
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}
    for attempt in range(5):
        try:
            r = httpx.post(f"{_OPENROUTER_BASE}/chat/completions",
                           headers={"Authorization": f"Bearer {api_key}"},
                           json=payload, timeout=120.0)
        except Exception as e:  # noqa: BLE001 — pre-response: retryable
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:120]}"
            elif r.status_code != 200:
                return "", f"HTTP {r.status_code}: {r.text[:200]}", None, None
            else:
                d = r.json()
                if d.get("error") or not d.get("choices"):
                    return "", str(d.get("error") or "200 with no choices"), None, d.get("usage")
                choice = d["choices"][0]
                fr = choice.get("finish_reason")
                if choice.get("error") or fr in ("error", "content_filter"):
                    return "", str(choice.get("error") or f"terminal finish_reason={fr}"), fr, d.get("usage")
                return ((choice.get("message") or {}).get("content") or ""), None, fr, d.get("usage")
        if attempt < 4:
            time.sleep(2 ** attempt)
    return "", last, None, None


def _quiet(verbose: bool):
    """Suppress A-MEM's very chatty internal print()s (per-note evolution JSONs) unless --verbose — a 500-item
    run prints GBs of them otherwise. Our own run_grouped progress lines print OUTSIDE this, so stay visible."""
    import contextlib
    import os
    if verbose:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _silence():
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            yield

    return _silence()


def run_a_mem(args, items, api_key, repo_dir, store, dataset, meta_out) -> None:
    """A-MEM with per-context reuse: ingest a context's sessions ONCE (Zettelkasten notes with link-evolution),
    then per question retrieve top-k and generate via OpenRouter. Retrieval is read-only → safe to reuse."""
    sys.path.insert(0, repo_dir)
    if args.embedding_service_url:
        from src.memory.eval.amem_embeddings import install_lightweight_upstream_imports
        install_lightweight_upstream_imports(args.embedding_service_url)
    # Upstream commit 0c8039f calls `re.sub` in MemoryNote.analyze_content without importing `re`. Without
    # this compatibility shim every valid metadata response is silently discarded and replaced by empty
    # keywords / "General" context. Injecting the missing stdlib module restores the code's intended path
    # while leaving the pinned upstream checkout untouched.
    import memory_layer as amem_memory_layer
    if not hasattr(amem_memory_layer, "re"):
        amem_memory_layer.re = re
    AgenticMemorySystem = amem_memory_layer.AgenticMemorySystem
    from src.memory.eval.tier2_common import group_by_context, run_grouped
    from src.memory.eval.baselines import build_messages

    meter = _UsageMeter()

    def _ingest_units(ctx, first_item):
        """LongMemEval defaults to upstream's one-note-per-turn protocol. MAB has no canonical A-MEM mapping,
        so it uses bounded document chunks as an explicitly reported adaptation."""
        return _amem_ingest_units(dataset, ctx, first_item, args.ingest_chunk_chars,
                                  args.ingest_granularity)

    def encode_ctx(ctx, first_item):
        mem = AgenticMemorySystem(model_name=args.embed_model, llm_backend="openai",
                                  llm_model=args.controller_model, api_key=api_key,
                                  api_base=_OPENROUTER_BASE)
        _attach_usage_meter(mem, meter, args.controller_provider, args.controller_retry_max_tokens)
        with _quiet(args.verbose):
            for note, when in _ingest_units(ctx, first_item):
                mem.add_note(note, time=when) if when else mem.add_note(note)
        return mem

    def answer(mem, it):
        # Faithful A-MEM retrieval = LLM keyword rewrite + linked-neighborhood expansion. The final answer
        # prompt stays benchmark-standard so every Phase-2 system uses the same answer protocol.
        dropped = _drop_invalid_amem_links(mem)
        meta_out["invalid_links_dropped"] = meta_out.get("invalid_links_dropped", 0) + dropped
        with _quiet(args.verbose):
            query, ctx_str = _retrieve_a_mem(mem, it["question"], args.retrieve_k, args.query_mode,
                                             not args.no_link_expansion)
        samples = meta_out.setdefault("retrieval_debug", [])
        if len(samples) < 20:
            samples.append({"question_id": str(it["question_id"]), "query": query,
                            "retrieved_chars": len(ctx_str or "")})
        msgs = _answer_messages(build_messages, it, dataset, ctx_str or "")
        hyp, err, fr, usage = _openrouter_chat(args.answer_model, msgs, api_key, args.max_new_tokens,
                                               args.answer_provider)
        meter.add("answer", usage)
        if err:
            raise RuntimeError(err)                 # let run_grouped record it as an error (retryable)
        return hyp, (fr or "stop")

    # audit #10: A-MEM runs ~2 LLM calls per note (metadata + link-evolution) — estimate up front so a full
    # run's cost/time is known (it is the SLOWEST baseline). Resume is PER-CONTEXT: a context whose questions
    # are all done is skipped (not re-ingested); a mid-context crash re-ingests that one context on rerun.
    groups = group_by_context(items)
    n_units = sum(len(_ingest_units(ctx, its[0])) for ctx, its in groups.items())
    est_calls = n_units * 2 + len(items) * (2 if args.query_mode == "upstream_keywords" else 1)
    meta_out.update({"n_contexts": len(groups), "n_ingest_units": n_units, "est_llm_calls": est_calls,
                     "a_mem_protocol": {
                         "ingest_granularity": args.ingest_granularity,
                         "query_mode": args.query_mode,
                         "expand_links": not args.no_link_expansion,
                         "embedding_model": args.embed_model,
                         "embedding_device": (args.embedding_service_health.get("device")
                                              if args.embedding_service_url else args.embed_device),
                         "embedding_backend": ("shared_service" if args.embedding_service_url else "local"),
                         "embedding_service": (args.embedding_service_health if args.embedding_service_url else None),
                         "controller_model": args.controller_model,
                         "controller_provider": args.controller_provider or "auto",
                         "controller_provider_fallbacks": not bool(args.controller_provider),
                         "controller_retry_max_tokens": args.controller_retry_max_tokens,
                         "answer_model": args.answer_model,
                         "answer_provider": args.answer_provider or "auto",
                         "answer_provider_fallbacks": not bool(args.answer_provider),
                         "token_usage_phase_roles": {
                             "controller": ["metadata", "evolution", "query_keywords", "internal_other"],
                             "answer": ["answer"],
                         },
                         "answer_prompt": "shared_phase2_benchmark_prompt",
                         "mab_note_policy": "document_chunks_no_canonical_paper_mapping",
                         "mab_chunk_chars": args.ingest_chunk_chars,
                     }})
    print(f"[run_agentmem] A-MEM estimate: {len(groups)} contexts, ~{n_units} ingest notes → ~{est_calls} LLM "
          f"calls (2/note + query rewrite + answer). Slowest baseline; resume is per-context.", flush=True)

    run_grouped(items, encode_ctx, answer, store, "[run_agentmem a-mem]")
    meta_out["token_usage"] = meter.as_dict()


def run_memoryos(args, items, api_key, repo_dir, store, dataset, meta_out) -> None:
    """MemoryOS: ingest, then get_response (retrieves + generates internally). ⚠ UNTESTED here (needs
    memoryos-pro). audit #1: get_response MUTATES memory with every query+answer, so reusing one instance
    across a context's questions would let later questions SEE earlier benchmark Q&A → contamination. We
    therefore build a FRESH instance PER QUESTION (no cross-question reuse), and detect the upstream
    "Error: Could not get response from LLM." string (which it returns instead of raising) → recorded as an
    error, NOT frozen as a valid answer."""
    del meta_out
    from memoryos import Memoryos   # pip install memoryos-pro
    from src.memory.eval.tier2_common import format_query, make_record

    done = store.done_ids()
    for i, it in enumerate(items):
        if str(it["question_id"]) in done:
            continue
        try:
            state_dir = REPO / "outputs" / "baselines" / "memoryos_state" / str(it["question_id"])
            if state_dir.exists():
                shutil.rmtree(state_dir, ignore_errors=True)
            memo = Memoryos(user_id=f"q_{it['question_id']}", assistant_id="assistant",
                            openai_api_key=api_key, openai_base_url=_OPENROUTER_BASE,
                            llm_model=args.llm_model, embedding_model_name=args.embed_model,
                            data_storage_path=str(state_dir))
            for sess in (it.get("sessions") or [it["full_history"]]):
                memo.add_memory(user_input=sess, agent_response="")     # fresh memory, this question only
            hyp = memo.get_response(query=format_query(it, dataset)) or ""
            if hyp.strip().startswith("Error: Could not get response"):
                store.append(make_record(it, error="memoryos: upstream 'Could not get response from LLM'"))
            else:
                store.append(make_record(it, hyp=hyp, finish_reason="stop"))
        except Exception as e:  # noqa: BLE001 — crash-safe per question
            print(f"[run_agentmem memoryos] ERROR on {it['question_id']}: {type(e).__name__}: {e}")
            store.append(make_record(it, error=f"{type(e).__name__}: {e}"))
        if (i + 1) % 25 == 0:
            print(f"[run_agentmem memoryos] {i + 1}/{len(items)} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["a-mem", "memoryos"], required=True)
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], default="longmemeval")
    ap.add_argument("--repo-dir", default=_DEFAULT_REPO_DIR, help="cloned A-mem repo (on sys.path)")
    ap.add_argument("--controller-model", default=None,
                    help=f"A-MEM metadata/evolution/query model (default: {_DEFAULT_CONTROLLER_LLM})")
    ap.add_argument("--controller-provider", default=None,
                    help=f"strict-schema controller provider pin (default: {_DEFAULT_CONTROLLER_PROVIDER})")
    ap.add_argument("--answer-model", default=None,
                    help=f"shared final-answer reader model (default: {_DEFAULT_ANSWER_LLM})")
    ap.add_argument("--answer-provider", default=None,
                    help=f"final-answer provider pin (default: {_DEFAULT_ANSWER_PROVIDER})")
    ap.add_argument("--llm-model", default=None,
                    help="legacy compatibility: use one model for both A-MEM control and final answers")
    ap.add_argument("--openrouter-provider", default=None,
                    help="legacy compatibility: pin both controller and answer calls to one provider")
    ap.add_argument("--embed-model", default=_DEFAULT_EMBED,
                    help="sentence embedder (local CPU, or served once when --embedding-service-url is set)")
    ap.add_argument("--embedding-service-url", default=None,
                    help="shared MiniLM service URL (for example http://127.0.0.1:8765); keeps workers "
                         "torch-free and batches embeddings on one GPU")
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="final-answer cap (default: LongMemEval=1024, MAB=256; internal A-MEM structured "
                         "calls retain upstream's 1000-token cap)")
    ap.add_argument("--controller-retry-max-tokens", type=int, default=4000,
                    help="only after an upstream 1000-token structured response is length-truncated, retries "
                         "may double its cap up to this value (default: 4000)")
    ap.add_argument("--retrieve-k", type=int, default=10, help="(a-mem) top-k memories to retrieve")
    ap.add_argument("--query-mode", choices=["upstream_keywords", "raw_question"],
                    default="upstream_keywords",
                    help="upstream A-MEM generates LLM keywords before embedding retrieval; raw is an ablation")
    ap.add_argument("--no-link-expansion", action="store_true",
                    help="disable upstream linked-neighborhood expansion (ablation)")
    ap.add_argument("--ingest-granularity", choices=["turn", "session"], default="turn",
                    help="LongMemEval note unit; upstream A-MEM uses one note per conversation turn")
    ap.add_argument("--embed-device", choices=["cpu", "auto"], default="cpu",
                    help="CPU guarantees this API baseline does not reserve GPU memory")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="parallel workers: partition whole contexts across this many independent processes")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="zero-based shard handled by this process (use once for each 0..num-shards-1)")
    ap.add_argument("--ingest-chunk-chars", type=int, default=_DEFAULT_MAB_NOTE_CHARS,
                    help="(MAB only) document-note size. 800 chars is an explicit benchmark adaptation that "
                         "keeps content plus generated attributes near all-MiniLM-L6-v2's 256-token embedding "
                         "limit; LongMemEval ignores this and uses upstream atomic conversation turns.")
    ap.add_argument("--verbose", action="store_true", help="show A-MEM's internal evolution logging (very noisy)")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()
    try:
        args.max_new_tokens = _answer_token_cap(args.dataset, args.max_new_tokens)
        if args.method == "a-mem":
            _resolve_amem_llm_roles(args)
        else:
            args.llm_model = args.llm_model or _DEFAULT_ANSWER_LLM
    except ValueError as exc:
        ap.error(str(exc))
    if args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        ap.error("require --num-shards >= 1 and 0 <= --shard-idx < --num-shards")
    if args.controller_retry_max_tokens < 1000:
        ap.error("--controller-retry-max-tokens must be >= upstream's initial 1000-token cap")
    if args.embedding_service_url and args.method != "a-mem":
        ap.error("--embedding-service-url currently applies only to --method a-mem")

    if args.embedding_service_url:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.embed_device == "cpu":
        # Set before importing sentence-transformers/torch through the upstream repository.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("[run_agentmem] set OPENROUTER_API_KEY (the reader LLM runs over OpenRouter).")
    try:
        _validate_openrouter_key(api_key)
        if args.method == "a-mem":
            _validate_amem_provider(args.controller_model, args.controller_provider)
            _validate_answer_provider(args.answer_model, args.answer_provider)
            args.embedding_service_health = {}
            if args.embedding_service_url:
                from src.memory.eval.amem_embeddings import RemoteSentenceTransformer
                live_health = RemoteSentenceTransformer(
                    args.embed_model, base_url=args.embedding_service_url).health()
                served_model = str(live_health.get("model", ""))
                if served_model.split("/")[-1] != args.embed_model.split("/")[-1]:
                    raise RuntimeError(f"embedding service model {served_model!r} does not match "
                                       f"--embed-model {args.embed_model!r}")
                # Request counters are useful operational telemetry, but are volatile and must not become
                # part of a result protocol/checkpoint identity (or otherwise-identical shards would differ).
                args.embedding_service_health = {
                    key: value for key, value in live_health.items() if key != "stats"
                }
    except RuntimeError as exc:
        sys.exit(f"[run_agentmem] {exc}")
    # A-MEM's `openai` backend (OpenAIController) does NOT accept api_base → route via the openai SDK's
    # OPENAI_BASE_URL env fallback (set OPENAI_API_KEY too) so its internal metadata/evolution calls hit
    # OpenRouter, not the default OpenAI endpoint (which would 401 the key). MemoryOS forwards its own base_url.
    os.environ["OPENAI_BASE_URL"] = _OPENROUTER_BASE
    os.environ["OPENAI_API_KEY"] = api_key

    repo_dir = str(Path(args.repo_dir).expanduser())
    from src.memory.eval.tier2_common import git_commit, load_items, build_tag, finalize
    from src.memory.eval.results import ResultStore

    roles = (f"controller={args.controller_model}@{args.controller_provider or 'auto'} "
             f"answer={args.answer_model}@{args.answer_provider or 'auto'}") \
        if args.method == "a-mem" else f"llm={args.llm_model}"
    print(f"[run_agentmem] method={args.method} dataset={args.dataset} {roles} "
          f"variant={args.variant} max_examples={args.max_examples}")
    if args.method == "a-mem" and args.embedding_service_url:
        from src.memory.eval.amem_data import load_items as load_lightweight_items
        items = load_lightweight_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    else:
        items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    if args.num_shards > 1:
        items, shard_loads = _shard_items(items, args.dataset, args)
        print(f"[run_agentmem] SHARD {args.shard_idx}/{args.num_shards}: {len(items)} questions; "
              f"projected calls={shard_loads[args.shard_idx]} "
              f"(fleet min/max={min(shard_loads)}/{max(shard_loads)})")
    if args.method == "a-mem" and args.embedding_service_url:
        from src.memory.eval.amem_data import trim_process_heap
        trim_process_heap()
    types = {t: sum(1 for i in items if i["question_type"] == t)
             for t in sorted({i["question_type"] for i in items})}
    print(f"[run_agentmem] {len(items)} items; types={types}")

    commit = git_commit(REPO)
    knob = _amem_knob_tag(args) if args.method == "a-mem" else f"emb{args.embed_model}"
    result_model = args.answer_model if args.method == "a-mem" else args.llm_model
    tag = build_tag(args.dataset, args.method, result_model.split("/")[-1], args.variant, len(items),
                    knob, args.max_new_tokens, 0, commit)
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    n_done = len(store.done_ids())
    if n_done:
        print(f"[run_agentmem] resume: {n_done}/{len(items)} already done — answering the rest")

    meta_out: dict = {}
    runner = run_a_mem if args.method == "a-mem" else run_memoryos
    runner(args, items, api_key, repo_dir, store, args.dataset, meta_out)

    finalize(args.dataset, args.method, result_model, items, store, use_bem=not args.no_bem,
             extra_meta={"variant": args.variant, "retrieve_k": args.retrieve_k,
                         "max_new_tokens": args.max_new_tokens, "commit": commit,
                         "num_shards": args.num_shards, "shard_idx": args.shard_idx,
                         "upstream_commit": git_commit(repo_dir), **meta_out},
             out_dir=out_dir, tag=tag, log_prefix="[run_agentmem]")


if __name__ == "__main__":
    main()
