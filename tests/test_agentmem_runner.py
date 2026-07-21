from types import SimpleNamespace

from scripts.baselines.run_agentmem import (
    _UsageMeter,
    _answer_token_cap,
    _answer_messages,
    _amem_ingest_units,
    _amem_knob_tag,
    _attach_usage_meter,
    _generate_query_keywords,
    _longmemeval_turn_units,
    _retrieve_a_mem,
    _shard_items,
)
from scripts.baselines.merge_agentmem_shards import _sum_usage


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


def test_cache_tag_captures_every_generation_affecting_amem_knob():
    base = dict(retrieve_k=10, ingest_granularity="turn", query_mode="upstream_keywords",
                no_link_expansion=False, embed_model="all-MiniLM-L6-v2", embed_device="cpu",
                ingest_chunk_chars=8000, num_shards=1, shard_idx=0, openrouter_provider=None)
    a = _amem_knob_tag(SimpleNamespace(**base))
    for key, value in (("retrieve_k", 5), ("ingest_granularity", "session"),
                       ("query_mode", "raw_question"), ("no_link_expansion", True),
                       ("embed_model", "another-embedder"), ("embed_device", "auto"),
                       ("ingest_chunk_chars", 4000), ("num_shards", 2),
                       ("openrouter_provider", "deepinfra")):
        changed = dict(base)
        changed[key] = value
        assert _amem_knob_tag(SimpleNamespace(**changed)) != a


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
