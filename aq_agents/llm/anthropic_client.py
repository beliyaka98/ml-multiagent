"""Адаптер для Claude (официальный SDK `anthropic`)."""
from __future__ import annotations

import anthropic

from .base import LLMError, LLMResponse, ToolCall, ToolResult

# Модели, для которых включаем серверный fallback: если классификатор безопасности
# отклонит запрос, API сам повторит его на рекомендованной модели.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


class AnthropicLLM:
    def __init__(self, model: str, effort: str | None = None):
        # Ключ берётся из ANTHROPIC_API_KEY. SDK сам повторяет 429/5xx и сетевые ошибки.
        self.client = anthropic.Anthropic(max_retries=3, timeout=120.0)
        self.model = model
        self.effort = effort

    def start(self, agent: str, system: str, user: str, tools: list[dict]) -> "AnthropicConversation":
        return AnthropicConversation(self, system, user, tools)


class AnthropicConversation:
    def __init__(self, llm: AnthropicLLM, system: str, user: str, tools: list[dict]):
        self.llm = llm
        self.system = system
        self.tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
        self.messages: list[dict] = [{"role": "user", "content": user}]

    def step(self) -> LLMResponse:
        kwargs: dict = {
            "model": self.llm.model,
            "max_tokens": 16000,
            "system": self.system,
            "messages": self.messages,
        }
        if self.tools:
            kwargs["tools"] = self.tools
        if self.llm.effort:
            kwargs["output_config"] = {"effort": self.llm.effort}
        try:
            if self.llm.model in FALLBACK_MODELS:
                resp = self.llm.client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self.llm.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("Неверный или отсутствующий ANTHROPIC_API_KEY (проверьте .env)") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Модель '{self.llm.model}' не найдена — проверьте LLM_MODEL в .env") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"API отклонил запрос: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Превышен лимит запросов к API (после повторных попыток)") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Ошибка API {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Нет соединения с API Anthropic (проверьте интернет)") from e
        except TypeError as e:
            if "authentication" in str(e):  # SDK не нашёл ни ключа, ни профиля `ant auth login`
                raise LLMError("Не найден ключ API: укажите ANTHROPIC_API_KEY в .env") from e
            raise

        if resp.stop_reason == "refusal":
            raise LLMError("Модель отказалась обрабатывать запрос (refusal)")

        # История только дополняется: ответ модели добавляем целиком, без изменений.
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls, resp.stop_reason or "", resp.usage.input_tokens, resp.usage.output_tokens)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # Все результаты одного шага — в одном сообщении пользователя.
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                for r in results
            ],
        })

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
