import threading
from types import SimpleNamespace

import numpy as np

import pytest

from scripts.baselines.run_agentmem import (
    _UsageMeter,
    _answer_provider_issue,
    _answer_token_cap,
    _answer_messages,
    _amem_ingest_units,
    _amem_knob_tag,
    _attach_usage_meter,
    _drop_invalid_amem_links,
    _generate_query_keywords,
    _longmemeval_turn_units,
    _retrieve_a_mem,
    _resolve_amem_llm_roles,
    _shard_items,
    _strict_schema_provider_issue,
    _structured_response_issue,
)
from scripts.baselines.merge_agentmem_shards import _sum_meta_counters, _sum_usage
from src.memory.eval.amem_embeddings import EmbeddingBatcher, _pack_vectors, _unpack_vectors


def test_longmemeval_ingests_one_note_per_turn_with_real_date():
    session = ("[Session abc — 2024/01/02 (Tue) 03:04]\n"
               "User: First line\ncontinued\n"
               "Assistant: Reply here")
    units = _longmemeval_turn_units(session)
    assert units == [
        ("Speaker User says : First line\ncontinued", "2024/01/02 (Tue) 03:04"),
        ("Speaker Assistant says : Reply here", "2024/01/02 (Tue) 03:04"),
    ]


def test_longmemeval_session_mode_is_explicit_ablation():
    session = "[Session abc — 2024/01/02]\nUser: hello\nAssistant: hi"
    item = {"sessions": [session]}
    assert len(_amem_ingest_units("longmemeval", session, item, 8000, "turn")) == 2
    assert _amem_ingest_units("longmemeval", session, item, 8000, "session") == [
        (session, "2024/01/02")
    ]


class _FakeLLM:
    def __init__(self, response='{"keywords":"degree, graduation"}'):
        self.response = response
        self.calls = []

    def get_completion(self, prompt, response_format):
        self.calls.append((prompt, response_format))
        return self.response


class _FakeMem:
    def __init__(self, response='{"keywords":"degree, graduation"}'):
        self.llm_controller = SimpleNamespace(llm=_FakeLLM(response))
        self.raw_calls = []
        self.plain_calls = []

    def find_related_memories_raw(self, query, k):
        self.raw_calls.append((query, k))
        return "linked context"

    def find_related_memories(self, query, k):
        self.plain_calls.append((query, k))
        return "plain context", [0]


def test_upstream_query_rewrite_and_link_expansion_are_default_fidelity_path():
    mem = _FakeMem()
    query, context = _retrieve_a_mem(mem, "What degree?", 10, "upstream_keywords", True)
    assert query == "degree, graduation"
    assert context == "linked context"
    assert mem.raw_calls == [("degree, graduation", 10)]
    assert not mem.plain_calls


def test_invalid_hallucinated_link_targets_are_dropped_without_losing_valid_links():
    notes = [SimpleNamespace(links=[0, 2, 3, -1, "1"]), SimpleNamespace(links=[1]),
             SimpleNamespace(links=None)]
    mem = SimpleNamespace(memories={str(i): note for i, note in enumerate(notes)})
    assert _drop_invalid_amem_links(mem) == 3
    assert [note.links for note in notes] == [[0, 2], [1], []]


def test_query_keyword_parse_falls_back_safely():
    mem = _FakeMem("degree, graduation")
    assert _generate_query_keywords(mem, "What degree?") == "degree, graduation"
    mem = _FakeMem("")
    assert _generate_query_keywords(mem, "What degree?") == "What degree?"


def test_mab_answer_prompt_is_native_and_not_double_wrapped():
    seen = {}

    def build(mode, **kwargs):
        seen.update(mode=mode, **kwargs)
        return ["messages"], {}

    item = {"question": "classify me", "system": "mab-system",
            "question_template": "Only label: {question}", "context_header": "# Context"}
    assert _answer_messages(build, item, "memoryagentbench", "retrieved") == ["messages"]
    assert seen == {
        "mode": "full_context", "question": "classify me", "full_history": "retrieved",
        "char_budget": 10 ** 9, "system": "mab-system",
        "question_template": "Only label: {question}", "context_header": "# Context",
    }


def test_answer_caps_are_dataset_aware_and_explicit_override_wins():
    assert _answer_token_cap("longmemeval", None) == 1024
    assert _answer_token_cap("memoryagentbench", None) == 256
    assert _answer_token_cap("longmemeval", 77) == 77


def test_split_llm_roles_default_to_nemo_controller_and_shared_llama_reader():
    args = SimpleNamespace(llm_model=None, openrouter_provider=None,
                           controller_model=None, controller_provider=None,
                           answer_model=None, answer_provider=None)
    _resolve_amem_llm_roles(args)
    assert (args.controller_model, args.controller_provider) == ("mistralai/mistral-nemo", "dekallm")
    assert (args.answer_model, args.answer_provider) == (
        "meta-llama/llama-3.1-8b-instruct", "deepinfra")


def test_legacy_one_model_flags_still_set_both_roles():
    args = SimpleNamespace(llm_model="legacy/model", openrouter_provider="legacy-provider",
                           controller_model=None, controller_provider=None,
                           answer_model=None, answer_provider=None)
    _resolve_amem_llm_roles(args)
    assert args.controller_model == args.answer_model == "legacy/model"
    assert args.controller_provider == args.answer_provider == "legacy-provider"


def test_legacy_and_split_role_flags_cannot_be_mixed():
    args = SimpleNamespace(llm_model="legacy/model", openrouter_provider=None,
                           controller_model="new/controller", controller_provider=None,
                           answer_model=None, answer_provider=None)
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_amem_llm_roles(args)


def test_provider_capability_check_distinguishes_strict_structured_outputs():
    payload = {"data": {"endpoints": [
        {"provider_name": "DeepInfra", "supported_parameters": ["response_format"]},
        {"provider_name": "WandB", "supported_parameters": ["response_format", "structured_outputs"]},
    ]}}
    assert "does not advertise" in _strict_schema_provider_issue(payload, "deepinfra")
    assert _strict_schema_provider_issue(payload, "wandb") is None
    assert "no current endpoint" in _strict_schema_provider_issue(payload, "cloudflare")
    assert _answer_provider_issue(payload, "deepinfra") is None
    assert "no current endpoint" in _answer_provider_issue(payload, "cloudflare")


def test_usage_meter_classifies_upstream_calls_without_changing_response():
    usage = SimpleNamespace(prompt_tokens=123, completion_tokens=17,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=11))
    response = SimpleNamespace(usage=usage)

    class Completions:
        def create(self, **kwargs):
            return response

    mem = SimpleNamespace(llm_controller=SimpleNamespace(llm=SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())))))
    meter = _UsageMeter()
    _attach_usage_meter(mem, meter)
    got = mem.llm_controller.llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "Generate a structured analysis of the following content"}])
    assert got is response
    assert meter.as_dict()["metadata"] == {
        "calls": 1, "prompt_tokens": 123, "completion_tokens": 17, "cached_prompt_tokens": 11,
        "reported_cost_usd": 0.0,
    }


def test_usage_wrapper_pins_openrouter_provider_without_fallbacks():
    seen = {}
    response = SimpleNamespace(usage=None)

    class Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return response

    mem = SimpleNamespace(llm_controller=SimpleNamespace(llm=SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())))))
    _attach_usage_meter(mem, _UsageMeter(), "deepinfra")
    got = mem.llm_controller.llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}], extra_body={"other": True})
    assert got is response
    assert seen["extra_body"] == {
        "other": True,
        "provider": {"order": ["deepinfra"], "allow_fallbacks": False},
    }


def test_usage_wrapper_retries_transient_internal_call(monkeypatch):
    response = SimpleNamespace(usage=None)

    class Completions:
        calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return response

    completions = Completions()
    mem = SimpleNamespace(llm_controller=SimpleNamespace(llm=SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))))
    monkeypatch.setattr("scripts.baselines.run_agentmem.time.sleep", lambda _seconds: None)
    _attach_usage_meter(mem, _UsageMeter())
    got = mem.llm_controller.llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}])
    assert got is response
    assert completions.calls == 2


def test_usage_wrapper_retries_json_null_and_meters_both_billed_attempts(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1, prompt_tokens_details=None)
    invalid = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content="null"))])
    valid = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"keywords": ["x"]}'))])

    class Completions:
        calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            return invalid if self.calls == 1 else valid

    schema = {"type": "json_schema", "json_schema": {"schema": {
        "type": "object",
        "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
        "required": ["keywords"], "additionalProperties": False,
    }}}
    completions = Completions()
    mem = SimpleNamespace(llm_controller=SimpleNamespace(llm=SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))))
    meter = _UsageMeter()
    monkeypatch.setattr("scripts.baselines.run_agentmem.time.sleep", lambda _seconds: None)
    _attach_usage_meter(mem, meter)
    got = mem.llm_controller.llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "Generate a structured analysis of the following content"}],
        response_format=schema)
    assert got is valid
    assert completions.calls == 2
    assert meter.as_dict()["metadata"]["calls"] == 2


def test_usage_wrapper_expands_cap_only_after_length_truncated_schema(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1000, prompt_tokens_details=None)
    invalid = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content='{"keywords": ["cut off"'))])
    valid = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content='{"keywords": ["complete"]}'))])

    class Completions:
        seen_caps = []

        def create(self, **kwargs):
            self.seen_caps.append(kwargs["max_tokens"])
            return invalid if len(self.seen_caps) == 1 else valid

    schema = {"type": "json_schema", "json_schema": {"schema": {
        "type": "object",
        "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
        "required": ["keywords"], "additionalProperties": False,
    }}}
    completions = Completions()
    mem = SimpleNamespace(llm_controller=SimpleNamespace(llm=SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))))
    monkeypatch.setattr("scripts.baselines.run_agentmem.time.sleep", lambda _seconds: None)
    _attach_usage_meter(mem, _UsageMeter(), retry_max_tokens=4000)
    mem.llm_controller.llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "Generate a structured analysis of the following content"}],
        response_format=schema, max_tokens=1000)
    assert completions.seen_caps == [1000, 2000]


def test_structured_response_validator_reports_missing_required_property():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    schema = {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"keywords": {"type": "string"}},
        "required": ["keywords"], "additionalProperties": False,
    }}}
    assert "missing required property" in _structured_response_issue(response, schema)


def test_structured_response_validator_handles_missing_choices():
    response = SimpleNamespace(choices=None)
    schema = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    assert _structured_response_issue(response, schema) == "response has no first message content"


def test_cache_tag_captures_every_generation_affecting_amem_knob():
    base = dict(retrieve_k=10, ingest_granularity="turn", query_mode="upstream_keywords",
                no_link_expansion=False, embed_model="all-MiniLM-L6-v2", embed_device="cpu",
                ingest_chunk_chars=8000, num_shards=1, shard_idx=0,
                controller_model="mistralai/mistral-nemo", controller_provider="dekallm",
                answer_model="meta-llama/llama-3.1-8b-instruct", answer_provider="deepinfra",
                embedding_service_url=None, controller_retry_max_tokens=4000)
    a = _amem_knob_tag(SimpleNamespace(**base))
    for key, value in (("retrieve_k", 5), ("ingest_granularity", "session"),
                       ("query_mode", "raw_question"), ("no_link_expansion", True),
                       ("embed_model", "another-embedder"), ("embed_device", "auto"),
                       ("ingest_chunk_chars", 4000), ("num_shards", 2),
                       ("controller_model", "another/controller"),
                       ("controller_provider", "novita"), ("answer_model", "another/reader"),
                       ("answer_provider", "groq"),
                       ("embedding_service_url", "http://127.0.0.1:8765"),
                       ("controller_retry_max_tokens", 8000)):
        changed = dict(base)
        changed[key] = value
        assert _amem_knob_tag(SimpleNamespace(**changed)) != a

    service_args = dict(base, embedding_service_url="http://127.0.0.1:8765",
                        embedding_service_health={"device": "cuda"})
    assert "-devcuda-" in _amem_knob_tag(SimpleNamespace(**service_args))


def test_sharding_is_disjoint_balanced_and_keeps_context_groups_whole():
    items = [
        {"question_id": "a1", "full_history": "A", "sessions": ["[Session 1 — d]\nUser: x"]},
        {"question_id": "a2", "full_history": "A", "sessions": ["[Session 1 — d]\nUser: x"]},
        {"question_id": "b1", "full_history": "B", "sessions": ["[Session 2 — d]\nUser: y"]},
        {"question_id": "c1", "full_history": "C", "sessions": ["[Session 3 — d]\nUser: z"]},
    ]
    base = dict(num_shards=2, ingest_chunk_chars=8000, ingest_granularity="turn",
                query_mode="upstream_keywords")
    s0, loads0 = _shard_items(items, "longmemeval", SimpleNamespace(**base, shard_idx=0))
    s1, loads1 = _shard_items(items, "longmemeval", SimpleNamespace(**base, shard_idx=1))
    assert loads0 == loads1
    ids0, ids1 = {x["question_id"] for x in s0}, {x["question_id"] for x in s1}
    assert not ids0 & ids1
    assert ids0 | ids1 == {"a1", "a2", "b1", "c1"}
    assert ({"a1", "a2"} <= ids0) != ({"a1", "a2"} <= ids1)


def test_merge_sums_phase_usage_and_recomputes_total():
    payloads = [
        {"meta": {"token_usage": {"metadata": {"calls": 2, "prompt_tokens": 20,
          "completion_tokens": 4, "cached_prompt_tokens": 3, "reported_cost_usd": 0.01},
          "TOTAL": {"calls": 999}}}},
        {"meta": {"token_usage": {"metadata": {"calls": 1, "prompt_tokens": 9,
          "completion_tokens": 2, "cached_prompt_tokens": 1, "reported_cost_usd": 0.005},
          "answer": {"calls": 1, "prompt_tokens": 8, "completion_tokens": 3,
          "cached_prompt_tokens": 0, "reported_cost_usd": 0.002}}}},
    ]
    got = _sum_usage(payloads)
    assert got["metadata"] == {"calls": 3, "prompt_tokens": 29, "completion_tokens": 6,
                               "cached_prompt_tokens": 4, "reported_cost_usd": 0.015}
    assert got["TOTAL"] == {"calls": 4, "prompt_tokens": 37, "completion_tokens": 9,
                            "cached_prompt_tokens": 4, "reported_cost_usd": 0.017}


def test_merge_sums_workload_and_invalid_link_counters():
    payloads = [
        {"meta": {"n_contexts": 2, "n_ingest_units": 20, "est_llm_calls": 44,
                  "invalid_links_dropped": 3}},
        {"meta": {"n_contexts": 1, "n_ingest_units": 9, "est_llm_calls": 20}},
    ]
    assert _sum_meta_counters(payloads) == {
        "n_contexts": 3, "n_ingest_units": 29, "est_llm_calls": 64,
        "invalid_links_dropped": 3,
    }


def test_embedding_binary_payload_round_trip_is_exact_float32():
    vectors = np.arange(24, dtype=np.float32).reshape(3, 8) / 7
    got = _unpack_vectors(_pack_vectors(vectors))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, vectors)


def test_embedding_batcher_coalesces_concurrent_workers():
    class Encoder:
        calls = []

        def encode(self, sentences, **kwargs):
            self.calls.append((list(sentences), kwargs))
            return np.asarray([[len(text), sum(map(ord, text))] for text in sentences], dtype=np.float32)

    encoder = Encoder()
    batcher = EmbeddingBatcher(encoder, batch_size=16, max_wait_ms=50)
    barrier = threading.Barrier(3)
    results = [None, None]

    def run(index, values):
        barrier.wait()
        results[index] = batcher.encode(values)

    threads = [threading.Thread(target=run, args=(0, ["a", "bb"])),
               threading.Thread(target=run, args=(1, ["ccc"]))]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    batcher.close()

    assert len(encoder.calls) == 1
    assert batcher.stats() == {"batches": 1, "requests": 2, "sentences": 3,
                               "max_batch_sentences": 3, "failures": 0}
    assert encoder.calls[0][0] in (["a", "bb", "ccc"], ["ccc", "a", "bb"])
    np.testing.assert_array_equal(results[0], [[1, 97], [2, 196]])
    np.testing.assert_array_equal(results[1], [[3, 297]])


def test_lightweight_longmemeval_text_loader_matches_canonical(monkeypatch):
    import src.memory.data.longmemeval as canonical
    import src.memory.eval.amem_data as lightweight

    raw = [
        {"question_id": "q1", "question": "Q?", "answer": "A", "question_date": "2024/01/03",
         "question_type": "temporal-reasoning", "haystack_session_ids": [7],
         "haystack_dates": ["2024/01/01"], "haystack_sessions": [[
             {"role": "user", "content": " hello "}, {"role": "assistant", "content": "hi"}]]},
        {"question_id": "q2_abs", "question": "Missing?", "answer": "unknown", "question_date": "",
         "question_type": "single-session-user", "haystack_session_ids": [8],
         "haystack_dates": ["2024/01/02"], "haystack_sessions": [[
             {"role": "user", "content": "nothing relevant"}]]},
    ]
    monkeypatch.setattr(canonical, "_load_raw", lambda _variant: raw)
    monkeypatch.setattr(lightweight, "_load_raw", lambda _variant: raw)
    assert lightweight.load_longmemeval_text("s") == canonical.load_longmemeval_text("s")
