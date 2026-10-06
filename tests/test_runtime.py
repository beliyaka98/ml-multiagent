"""Надёжность цикла агента: ошибки инструментов, повтор невалидного ответа, лимиты, зацикливание."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from aq_agents.runtime import BaseAgent, Budget, LimitExceeded, Tool
from aq_agents.tools.http import ToolError
from scripted_llm import ScriptedLLM


class Answer(BaseModel):
    value: int


class EchoAgent(BaseAgent):
    name = "echo"
    system_prompt = "test"
    decision_model = Answer

    def prompt(self, inp, ctx):
        return "go"

    def make_tools(self, inp, ctx, facts):
        def add(a: int, b: int) -> dict:
            return {"sum": a + b}

        def broken() -> dict:
            raise ToolError("сервис недоступен")

        schema = {"type": "object", "properties": {}}
        return [Tool("add", "сложить", schema, add), Tool("broken", "всегда падает", schema, broken)]


def test_tool_error_is_returned_to_llm(make_ctx):
    llm = ScriptedLLM({"echo": [[{"name": "broken"}], '{"value": 1}']})
    ctx = make_ctx(llm)
    assert EchoAgent().run(None, ctx).value == 1
    result = llm.conversations[0].tool_results[0]
    assert result.is_error and "сервис недоступен" in result.content


def test_unknown_tool_and_bad_arguments_do_not_crash(make_ctx):
    llm = ScriptedLLM({"echo": [[{"name": "nope"}, {"name": "add", "arguments": {"a": 1}}], '{"value": 2}']})
    ctx = make_ctx(llm)
    assert EchoAgent().run(None, ctx).value == 2
    errors = [r for r in llm.conversations[0].tool_results if r.is_error]
    assert len(errors) == 2


def test_invalid_output_is_repaired(make_ctx):
    llm = ScriptedLLM({"echo": ["не JSON", '{"value": "abc"}', '{"value": 3}']})
    ctx = make_ctx(llm)
    assert EchoAgent().run(None, ctx).value == 3
    assert len(llm.conversations[0].user_messages) == 2  # два замечания модели
    assert sum(e["kind"] == "output_rejected" for e in ctx.log.events) == 2


def test_json_with_markdown_and_trailing_text_is_accepted(make_ctx):
    answer = 'Готово:\n```json\n{"value": 4}\n```\nЕщё раз: {"value": 4}'
    llm = ScriptedLLM({"echo": [answer]})
    assert EchoAgent().run(None, make_ctx(llm)).value == 4


def test_gives_up_after_max_repairs(make_ctx):
    llm = ScriptedLLM({"echo": ["нет", "нет", "нет", "нет"]})
    with pytest.raises(Exception, match="не прошёл проверку"):
        EchoAgent().run(None, make_ctx(llm))


def test_loop_detection(make_ctx):
    same = [{"name": "add", "arguments": {"a": 1, "b": 1}}]
    llm = ScriptedLLM({"echo": [same, same, same, '{"value": 0}']})
    with pytest.raises(LimitExceeded, match="зацикливание"):
        EchoAgent().run(None, make_ctx(llm))


def test_agent_step_limit(make_ctx):
    calls = [[{"name": "add", "arguments": {"a": i, "b": 0}}] for i in range(10)]
    llm = ScriptedLLM({"echo": calls})
    with pytest.raises(LimitExceeded, match="лимит шагов агента"):
        EchoAgent().run(None, make_ctx(llm))


def test_task_budget_limit(make_ctx):
    calls = [[{"name": "add", "arguments": {"a": i, "b": 0}}] for i in range(10)]
    llm = ScriptedLLM({"echo": calls})
    with pytest.raises(LimitExceeded, match="общий лимит шагов"):
        EchoAgent().run(None, make_ctx(llm, budget=Budget(max_steps=3, timeout_s=60)))


def test_task_timeout(make_ctx):
    llm = ScriptedLLM({"echo": ['{"value": 1}']})
    with pytest.raises(LimitExceeded, match="время"):
        EchoAgent().run(None, make_ctx(llm, budget=Budget(max_steps=10, timeout_s=-1)))


def test_every_call_is_logged(make_ctx):
    llm = ScriptedLLM({"echo": [[{"name": "add", "arguments": {"a": 2, "b": 3}}], '{"value": 5}']})
    ctx = make_ctx(llm)
    EchoAgent().run(None, ctx)
    lines = [json.loads(x) for x in ctx.log.path.read_text(encoding="utf-8").splitlines()]
    kinds = [e["kind"] for e in lines]
    assert kinds.count("llm_call") == 2 and kinds.count("tool_call") == 1
    assert ctx.log.load_report()["total_calls"] == 3
