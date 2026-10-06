"""Адаптер OpenAI-совместимых API (Gemini, Groq, OpenAI) на поддельном клиенте, без сети."""
from __future__ import annotations

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from aq_agents.llm import LLMError, ToolResult
from aq_agents.llm import openai_compat_client as oc


def completion(message: dict, finish: str = "tool_calls") -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    })


def rate_limit_error(message: str = "429") -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return openai.RateLimitError(message, response=httpx.Response(429, request=request), body=None)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_conv(outcomes) -> tuple[oc.OpenAICompatConversation, FakeCompletions]:
    llm = oc.OpenAICompatLLM.__new__(oc.OpenAICompatLLM)
    fake = FakeCompletions(outcomes)
    llm.client = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
    llm.model = "m"
    tools = [{"name": "geocode_city", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    return llm.start("data_agent", "sys", "задача", tools), fake


def test_provider_fields_of_tool_call_are_sent_back():
    """Регрессия: Gemini отклоняет запрос, если не вернуть thought_signature из extra_content."""
    tool_call = {"id": "call_1", "type": "function",
                 "function": {"name": "geocode_city", "arguments": '{"name": "Алматы"}'},
                 "extra_content": {"google": {"thought_signature": "SIG"}}}
    conv, fake = make_conv([
        completion({"role": "assistant", "content": None, "tool_calls": [tool_call]}),
        completion({"role": "assistant", "content": '{"ok": true}'}, finish="stop"),
    ])
    first = conv.step()
    assert first.tool_calls[0].arguments == {"name": "Алматы"}
    conv.add_tool_results([ToolResult("call_1", '{"lat": 43.25}')])
    conv.step()

    sent = fake.requests[1]["messages"]
    assert sent[2]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert sent[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"lat": 43.25}'}


def test_rate_limit_waits_and_retries(monkeypatch):
    waits = []
    monkeypatch.setattr(oc.time, "sleep", waits.append)
    conv, _ = make_conv([rate_limit_error(), rate_limit_error(),
                         completion({"role": "assistant", "content": "{}"}, finish="stop")])
    assert conv.step().text == "{}"
    assert waits == [10, 20]


def test_daily_quota_fails_fast_without_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(oc.time, "sleep", waits.append)
    conv, fake = make_conv([rate_limit_error("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(LLMError, match="дневной лимит"):
        conv.step()
    assert waits == [] and len(fake.requests) == 1


def test_rate_limit_gives_clear_error_after_all_waits(monkeypatch):
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    conv, _ = make_conv([rate_limit_error()] * 4)
    with pytest.raises(LLMError, match="лимит запросов"):
        conv.step()
