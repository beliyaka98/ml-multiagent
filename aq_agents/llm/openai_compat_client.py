"""Адаптер для OpenAI-совместимых API: OpenAI, Groq, Gemini, локальная Ollama и т.п.

Провайдер задаётся через OPENAI_BASE_URL, ключ — через OPENAI_API_KEY
(для Ollama подойдёт любое значение, например "ollama").
Модель должна поддерживать вызов инструментов (tool calling).
"""
from __future__ import annotations

import json
import time

import openai

from .base import LLMError, LLMResponse, ToolCall, ToolResult

RATE_LIMIT_WAITS_S = (10, 20, 30)  # паузы перед повтором после ошибки 429


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None = None):
        try:
            self.client = openai.OpenAI(base_url=base_url, max_retries=3, timeout=120.0)
        except openai.OpenAIError as e:
            raise LLMError("Не задан OPENAI_API_KEY (проверьте .env)") from e
        self.model = model

    def start(self, agent: str, system: str, user: str, tools: list[dict]) -> "OpenAICompatConversation":
        return OpenAICompatConversation(self, system, user, tools)


class OpenAICompatConversation:
    def __init__(self, llm: OpenAICompatLLM, system: str, user: str, tools: list[dict]):
        self.llm = llm
        self.tools = [{"type": "function", "function": t} for t in tools]
        self.messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def step(self) -> LLMResponse:
        kwargs: dict = {"model": self.llm.model, "messages": self.messages}
        if self.tools:
            kwargs["tools"] = self.tools
        resp = self._create_with_rate_limit_waits(kwargs)

        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        raw_calls: list[dict] = []
        for tc in msg.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": fn.arguments}  # агент вернёт модели ошибку аргументов
            calls.append(ToolCall(tc.id, fn.name, args))
            # Возвращаем вызов целиком, со служебными полями провайдера: например, Gemini
            # кладёт в extra_content подпись рассуждения и без неё отклоняет следующий запрос.
            raw = tc.model_dump(exclude_none=True)
            raw["function"]["arguments"] = fn.arguments or "{}"
            raw_calls.append(raw)

        assistant: dict = {"role": "assistant", "content": msg.content or ""}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        self.messages.append(assistant)

        usage = resp.usage
        return LLMResponse(
            msg.content or "",
            calls,
            choice.finish_reason or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    def _create_with_rate_limit_waits(self, kwargs: dict):
        # Бесплатные тарифы (например, Gemini) ограничивают число запросов в минуту.
        # Быстрые повторы SDK не успевают дождаться нового окна, поэтому ждём дольше.
        for wait_s in (*RATE_LIMIT_WAITS_S, None):
            try:
                return self.llm.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                if "PerDay" in str(e):  # дневную квоту ожиданием не дождаться
                    raise LLMError(f"Исчерпан дневной лимит бесплатного тарифа для модели '{self.llm.model}'. "
                                   "Укажите в .env другую модель (LLM_MODEL) или повторите завтра") from e
                if wait_s is None:
                    raise LLMError("Превышен лимит запросов к API — подождите минуту и повторите") from e
                time.sleep(wait_s)
            except openai.AuthenticationError as e:
                raise LLMError("Неверный OPENAI_API_KEY (проверьте .env)") from e
            except openai.NotFoundError as e:
                raise LLMError(f"Модель '{self.llm.model}' не найдена у провайдера") from e
            except openai.APIStatusError as e:
                raise LLMError(f"Ошибка API {e.status_code}: {e.message}") from e
            except openai.APIConnectionError as e:
                raise LLMError("Нет соединения с API провайдера") from e

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
