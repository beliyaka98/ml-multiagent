"""Фейковый LLM для тестов: для каждого агента выдаёт заранее заданную последовательность ответов.

Элемент сценария:
- строка — финальный текстовый ответ;
- список словарей {"name", "arguments"} — вызовы инструментов;
- функция conv -> одно из вышеперечисленного (для ответов, зависящих от результатов инструментов).
"""
from __future__ import annotations

import json

from aq_agents.llm import LLMError, LLMResponse, ToolCall, ToolResult


class ScriptedLLM:
    def __init__(self, scripts: dict[str, list]):
        self.scripts = {agent: list(items) for agent, items in scripts.items()}
        self.conversations: list[ScriptedConversation] = []

    def start(self, agent: str, system: str, user: str, tools: list[dict]) -> "ScriptedConversation":
        conv = ScriptedConversation(self, agent, system, user, tools)
        self.conversations.append(conv)
        return conv


class ScriptedConversation:
    def __init__(self, llm: ScriptedLLM, agent: str, system: str, user: str, tools: list[dict]):
        self.llm, self.agent, self.system, self.user, self.tools = llm, agent, system, user, tools
        self.tool_results: list[ToolResult] = []
        self.user_messages: list[str] = []
        self._n = 0

    def last_result(self, tool_name: str) -> dict:
        """JSON последнего успешного результата инструмента (по имени вызова)."""
        for r in reversed(self.tool_results):
            if r.call_id.startswith(tool_name) and not r.is_error:
                return json.loads(r.content)
        raise KeyError(tool_name)

    def step(self) -> LLMResponse:
        queue = self.llm.scripts.get(self.agent)
        if not queue:
            raise LLMError(f"сценарий для {self.agent} закончился")
        item = queue.pop(0)
        if callable(item):
            item = item(self)
        if isinstance(item, str):
            return LLMResponse(text=item, stop_reason="end_turn", input_tokens=10, output_tokens=5)
        calls = []
        for c in item:
            self._n += 1
            calls.append(ToolCall(f"{c['name']}#{self._n}", c["name"], c.get("arguments", {})))
        return LLMResponse(text="", tool_calls=calls, stop_reason="tool_use", input_tokens=10, output_tokens=5)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.tool_results.extend(results)

    def add_user(self, text: str) -> None:
        self.user_messages.append(text)
