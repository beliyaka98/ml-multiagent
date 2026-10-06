"""Общий интерфейс к LLM, не зависящий от провайдера.

Агент работает только с `Conversation`: делает шаг (`step`), получает ответ
с текстом и/или вызовами инструментов, возвращает результаты инструментов.
Как именно эти сообщения выглядят у Anthropic или OpenAI — скрыто в адаптерах.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(Exception):
    """Ошибка провайдера LLM (нет ключа, сеть, лимиты) с понятным текстом."""


class Conversation(Protocol):
    def step(self) -> LLMResponse: ...

    def add_tool_results(self, results: list[ToolResult]) -> None: ...

    def add_user(self, text: str) -> None: ...


class LLM(Protocol):
    def start(self, agent: str, system: str, user: str, tools: list[dict]) -> Conversation:
        """tools: [{"name", "description", "parameters": JSON Schema}]"""
        ...
