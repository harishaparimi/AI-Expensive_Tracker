import pytest

from app import config
from app.services import llm

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def reply(content, finish_reason="stop"):
    return {"model": "m:free", "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


@pytest.fixture
def post_returns(monkeypatch):
    """Make the HTTP call return a chosen body, with a key set, and record the payload."""
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    sent = {}

    def setup(body):
        def fake_post(url, headers, json, timeout):
            sent["payload"] = json
            return FakeResponse(body)

        monkeypatch.setattr(llm.requests, "post", fake_post)
        return sent

    return setup


def test_every_answer_has_a_length_limit(post_returns):
    sent = post_returns(reply('{"a": 1}'))
    llm.call_openrouter("system", "user", {}, ["m:free"])
    assert sent["payload"]["max_tokens"] == llm.REQUEST_SETTINGS["max_tokens"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("UPI-SWIGGY-swiggy@ybl-YESB0000001-612345678901-PAYMENT", "UPI-SWIGGY-[ID]-YESB[NUM]-[NUM]-PAYMENT"),
        ("call 9876543210 now", "call [NUM] now"),
        ("rahul.v@oksbi paid", "[ID] paid"),
        ("T2610031234567890", "T[NUM]"),
        ("amount 100000.00 and 1,25,000.50", "amount 100000.00 and 1,25,000.50"),
        ("short 12345 stays", "short 12345 stays"),
    ],
)
def test_mask(text, expected):
    assert llm.mask(text) == expected


def test_parse_json_reads_fenced_answers():
    assert llm.parse_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", "{not valid}"])
def test_parse_json_rejects_bad_answers(text):
    with pytest.raises(llm.LLMError):
        llm.parse_json(text)


@pytest.mark.parametrize("models", [[], ["openai/gpt-paid"], ["good/model:free", "openai/gpt-paid"]])
def test_only_free_models_are_allowed(monkeypatch, models):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")

    def must_not_post(*args, **kwargs):
        raise AssertionError("no request should be sent")

    monkeypatch.setattr(llm.requests, "post", must_not_post)
    with pytest.raises(llm.LLMError, match="free"):
        llm.call_openrouter("system", "user", SCHEMA, models)


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    with pytest.raises(llm.LLMError, match="OPENROUTER_API_KEY"):
        llm.call_openrouter("system", "user", SCHEMA, ["m:free"])


def test_successful_call_returns_answer_and_model(post_returns):
    sent = post_returns(reply('{"answer": "hi"}'))
    answer, model = llm.call_openrouter("system", "user", SCHEMA, ["m:free"])
    assert answer == {"answer": "hi"}
    assert model == "m:free"
    assert sent["payload"]["models"] == ["m:free"]
    assert sent["payload"]["response_format"]["json_schema"]["schema"] == SCHEMA


def test_requests_use_the_default_settings(post_returns):
    sent = post_returns(reply('{"answer": "hi"}'))
    llm.call_openrouter("system", "user", SCHEMA, ["m:free"])
    assert sent["payload"]["reasoning"] == {"enabled": False}
    assert sent["payload"]["temperature"] == 0.2


def test_options_replace_the_default_settings(post_returns):
    """The evals rely on this, so that an empty setting really means the model's own default."""
    sent = post_returns(reply('{"answer": "hi"}'))
    llm.call_openrouter("system", "user", SCHEMA, ["m:free"], {"seed": 7})
    assert sent["payload"]["seed"] == 7
    assert "reasoning" not in sent["payload"]
    assert sent["payload"]["models"] == ["m:free"]

    llm.call_openrouter("system", "user", SCHEMA, ["m:free"], {})
    assert "reasoning" not in sent["payload"]


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": 429, "message": "Rate limit"}},
        {"choices": [{"error": {"code": 502, "message": "Provider failed"}, "finish_reason": "error"}]},
        reply('{"answer": "cut', finish_reason="length"),
    ],
)
def test_error_bodies_raise(post_returns, body):
    post_returns(body)
    with pytest.raises(llm.LLMError):
        llm.call_openrouter("system", "user", SCHEMA, ["m:free"])


def test_repeated_prompt_uses_saved_answer(monkeypatch):
    calls = []

    def fake(system, user, schema, models):
        calls.append(user)
        return {"answer": "hi"}, "m:free"

    monkeypatch.setattr(llm, "call_openrouter", fake)
    assert llm.ask_llm("system", "same prompt", SCHEMA) == {"answer": "hi"}
    assert llm.ask_llm("system", "same prompt", SCHEMA) == {"answer": "hi"}
    assert len(calls) == 1


def test_use_saved_off_always_sends_and_saves_nothing(monkeypatch):
    calls = []

    def fake(system, user, schema, models):
        calls.append(user)
        return {"answer": "hi"}, "m:free"

    monkeypatch.setattr(llm, "call_openrouter", fake)
    llm.ask_llm("system", "same prompt", SCHEMA, use_saved=False)
    llm.ask_llm("system", "same prompt", SCHEMA, use_saved=False)
    assert len(calls) == 2
    llm.ask_llm("system", "same prompt", SCHEMA)  # nothing was saved for this to reuse
    assert len(calls) == 3


def test_ask_llm_sends_masked_text(monkeypatch):
    calls = []

    def fake(system, user, schema, models):
        calls.append(user)
        return {"answer": "hi"}, "m:free"

    monkeypatch.setattr(llm, "call_openrouter", fake)
    llm.ask_llm("system", "pay rahul@oksbi account 123456789012", SCHEMA)
    assert calls == ["pay [ID] account [NUM]"]


def test_answer_missing_a_field_is_not_saved(monkeypatch):
    answers = [({"wrong": 1}, "m:free"), ({"answer": "hi"}, "m:free")]
    monkeypatch.setattr(llm, "call_openrouter", lambda *args: answers.pop(0))
    with pytest.raises(llm.LLMError, match="missing"):
        llm.ask_llm("system", "prompt", SCHEMA)
    assert llm.ask_llm("system", "prompt", SCHEMA) == {"answer": "hi"}
