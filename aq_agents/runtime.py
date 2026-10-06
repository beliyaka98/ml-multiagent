"""Общий «двигатель» агента: цикл LLM ↔ инструменты с лимитами, логированием и проверкой ответа.

Цикл одного агента:
  1. LLM получает роль (system prompt), задачу и список инструментов.
  2. Если LLM просит вызвать инструменты — выполняем их и возвращаем результаты.
  3. Если LLM даёт финальный ответ — проверяем его по Pydantic-схеме и бизнес-правилам
     агента (finalize). Не прошёл — отправляем замечание и даём исправить (до 2 раз).
Защита: лимит шагов агента, общий лимит шагов и времени задачи, обнаружение зацикливания.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from .config import Settings
from .llm import LLM, LLMError, ToolCall, ToolResult
from .logger import RunLogger
from .messages import Message
from .state import TaskState
from .tools.http import ToolError

MAX_REPAIRS = 2          # сколько раз агент может исправить невалидный ответ
MAX_SAME_TOOL_CALL = 2   # третий одинаковый вызов инструмента = зацикливание


class AgentError(Exception):
    """Агент не смог выполнить задачу. Текст показывается пользователю."""


class LimitExceeded(AgentError):
    """Сработал лимит шагов, времени или защита от зацикливания."""


class OutputRejected(ValueError):
    """Ответ LLM не прошёл проверку — агент получит замечание и попробует снова."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema аргументов
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class Budget:
    """Общий бюджет задачи: число шагов (вызовы LLM + инструментов) и время."""

    def __init__(self, max_steps: int, timeout_s: float):
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.steps = 0
        self._t0 = time.monotonic()

    def check(self) -> None:
        if self.steps >= self.max_steps:
            raise LimitExceeded(f"Превышен общий лимит шагов задачи ({self.max_steps})")
        if time.monotonic() - self._t0 > self.timeout_s:
            raise LimitExceeded(f"Превышено время выполнения задачи ({self.timeout_s:.0f} с)")

    def tick(self) -> None:
        self.steps += 1


@dataclass
class AgentContext:
    task_id: str
    state: TaskState
    log: RunLogger
    budget: Budget
    llm: LLM
    settings: Settings


class BaseAgent:
    name: str = "base"
    system_prompt: str = ""
    decision_model: type[BaseModel]  # что именно должна вернуть LLM в финальном ответе
    accepts: tuple[str, ...] = ()    # какие типы входящих сообщений понимает агент

    # --- переопределяется в конкретных агентах ---

    def parse_input(self, msg: Message, ctx: AgentContext) -> Any:
        return msg.payload

    def prompt(self, inp: Any, ctx: AgentContext) -> str:
        raise NotImplementedError

    def make_tools(self, inp: Any, ctx: AgentContext, facts: dict) -> list[Tool]:
        """Инструменты агента. В `facts` инструменты складывают проверенные факты для finalize."""
        return []

    def finalize(self, decision: BaseModel, inp: Any, ctx: AgentContext, facts: dict) -> BaseModel:
        """Решение LLM + факты из инструментов → выходной payload. Может бросить OutputRejected."""
        return decision

    # --- общий цикл ---

    def handle(self, msg: Message, ctx: AgentContext) -> BaseModel:
        if self.accepts and msg.type not in self.accepts:
            raise AgentError(f"{self.name}: неожиданный тип сообщения '{msg.type}'")
        return self.run(self.parse_input(msg, ctx), ctx)

    def run(self, inp: Any, ctx: AgentContext) -> BaseModel:
        facts: dict = {}
        tools = {t.name: t for t in self.make_tools(inp, ctx, facts)}
        system = self.system_prompt.strip() + "\n\n" + _output_instructions(self.decision_model)
        conv = ctx.llm.start(self.name, system, self.prompt(inp, ctx), [t.spec() for t in tools.values()])
        ctx.log.event("agent_start", self.name, tools=list(tools))

        seen: Counter[str] = Counter()
        repairs = 0
        for step in range(1, ctx.settings.max_agent_steps + 1):
            ctx.budget.check()
            t0 = time.monotonic()
            try:
                resp = conv.step()
            except LLMError as e:
                ctx.log.event("llm_error", self.name, step=step, error=str(e))
                raise AgentError(f"{self.name}: ошибка LLM — {e}") from e
            ctx.budget.tick()
            ctx.log.event("llm_call", self.name, step=step, latency_s=round(time.monotonic() - t0, 2),
                          input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                          stop_reason=resp.stop_reason, tool_calls=[c.name for c in resp.tool_calls])

            if resp.tool_calls:
                conv.add_tool_results([self._run_tool(c, tools, seen, ctx) for c in resp.tool_calls])
                continue

            try:
                decision = _parse_json(resp.text, self.decision_model)
                result = self.finalize(decision, inp, ctx, facts)
            except ValueError as e:  # ValidationError и OutputRejected — подклассы ValueError
                repairs += 1
                ctx.log.event("output_rejected", self.name, attempt=repairs, error=str(e)[:500])
                if repairs > MAX_REPAIRS:
                    raise AgentError(f"{self.name}: ответ модели не прошёл проверку {repairs} раза: {e}") from e
                conv.add_user(f"Ответ не прошёл проверку: {e}\nИсправь и верни только JSON по схеме.")
                continue

            ctx.log.event("agent_end", self.name, steps=step)
            return result

        raise LimitExceeded(f"{self.name}: превышен лимит шагов агента ({ctx.settings.max_agent_steps})")

    def _run_tool(self, call: ToolCall, tools: dict[str, Tool], seen: Counter, ctx: AgentContext) -> ToolResult:
        signature = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        seen[signature] += 1
        if seen[signature] > MAX_SAME_TOOL_CALL:
            ctx.log.event("loop_detected", self.name, tool=call.name, args=call.arguments)
            raise LimitExceeded(f"{self.name}: зацикливание — {call.name} вызван "
                                f"{seen[signature]} раза с одинаковыми аргументами")
        ctx.budget.check()

        t0 = time.monotonic()
        tool = tools.get(call.name)
        ok = False
        if tool is None:
            content = f"Инструмента '{call.name}' нет. Доступны: {', '.join(tools)}"
        else:
            try:
                content = json.dumps(tool.fn(**call.arguments), ensure_ascii=False, default=str)
                ok = True
            except TypeError as e:
                content = f"Неверные аргументы: {e}"
            except (ToolError, KeyError, ValueError) as e:
                content = f"Ошибка: {e}"
            except Exception as e:  # noqa: BLE001 — любая ошибка инструмента не должна ронять систему
                content = f"Непредвиденная ошибка {type(e).__name__}: {e}"
        ctx.budget.tick()
        ctx.log.event("tool_call", self.name, tool=call.name, args=call.arguments, ok=ok,
                      latency_s=round(time.monotonic() - t0, 2), result_preview=content[:300])
        return ToolResult(call.id, content, is_error=not ok)


def _output_instructions(model: type[BaseModel]) -> str:
    schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
    return ("Когда работа закончена, ответь ТОЛЬКО одним JSON-объектом (без markdown и пояснений), "
            f"который соответствует JSON Schema:\n{schema}")


def _parse_json(text: str, model: type[BaseModel]) -> BaseModel:
    """Берёт первый полный JSON-объект из ответа: модели иногда добавляют markdown или лишний текст."""
    start = text.find("{")
    if start < 0:
        raise OutputRejected("в ответе нет JSON-объекта")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise OutputRejected(f"некорректный JSON: {e}") from e
    return model.model_validate(data)
