"""Выбор провайдера LLM по настройкам (.env)."""
from __future__ import annotations

from ..config import Settings
from .base import LLM, Conversation, LLMError, LLMResponse, ToolCall, ToolResult


def make_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "anthropic":
        from .anthropic_client import AnthropicLLM
        return AnthropicLLM(settings.llm_model, settings.llm_effort)
    if settings.llm_provider == "openai":
        from .openai_compat_client import OpenAICompatLLM
        return OpenAICompatLLM(settings.llm_model, settings.openai_base_url)
    raise LLMError(f"Неизвестный LLM_PROVIDER='{settings.llm_provider}'. Допустимо: anthropic, openai")


__all__ = ["LLM", "Conversation", "LLMError", "LLMResponse", "ToolCall", "ToolResult", "make_llm"]
