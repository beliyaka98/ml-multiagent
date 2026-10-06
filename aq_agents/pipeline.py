"""Диспетчер: выполняет план Orchestrator'а и доставляет сообщения между агентами.

Схема: Orchestrator строит план → диспетчер (обычный код, не LLM) по очереди вызывает
агентов из плана → выход каждого агента адресован следующему агенту плана.
Большие данные лежат в общем состоянии задачи (TaskState), в сообщениях — только ссылки.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel

from .agents import DataAgent, ForecastAgent, Orchestrator
from .config import Settings, get_settings
from .llm import LLM, LLMError, make_llm
from .logger import RunLogger
from .messages import DataPackage, ForecastResult, Message, TaskPlan, UserQuery
from .runtime import AgentContext, AgentError, BaseAgent, Budget
from .state import TaskState

# Агенты, реализованные в прототипе (Ассайнмент 2). Остальные появятся в Ассайнменте 4.
IMPLEMENTED: dict[str, BaseAgent] = {
    "orchestrator": Orchestrator(),
    "data_agent": DataAgent(),
    "forecast_agent": ForecastAgent(),
}
OUTPUT_TYPE = {
    "orchestrator": "task_plan",
    "data_agent": "data_package",
    "forecast_agent": "forecast_result",
}


@dataclass
class RunResult:
    task_id: str
    status: str  # done | failed
    answer: str
    error: str | None
    load: dict
    run_dir: Path
    state: TaskState


def run_task(query: str, settings: Settings | None = None, llm: LLM | None = None,
             verbose: bool = True, on_event: Callable[[dict], None] | None = None) -> RunResult:
    settings = settings or get_settings()
    task_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    run_dir = settings.runs_dir / task_id
    state = TaskState(task_id, run_dir)
    log = RunLogger(run_dir / "log.jsonl", verbose=verbose, on_event=on_event)
    log.event("task_start", "dispatcher", query=query, provider=settings.llm_provider, model=settings.llm_model)

    try:
        ctx = AgentContext(task_id, state, log, Budget(settings.max_total_steps, settings.task_timeout_s),
                           llm or make_llm(settings), settings)
        msg = Message(task_id=task_id, sender="user", receiver="orchestrator", type="user_query",
                      payload=UserQuery(text=query).model_dump())
        _record(msg, ctx)

        # 1. Orchestrator строит план; его сообщение адресовано первому агенту плана.
        msg = _deliver(msg, ctx, route=lambda plan: _next_step(plan, after=None))
        state.plan = TaskPlan.model_validate(msg.payload)
        log.event("plan", "orchestrator", **state.plan.model_dump())
        for skipped in (s for s in state.plan.steps if s not in IMPLEMENTED):
            log.event("agent_skipped", skipped, reason="будет реализован в Ассайнменте 4")

        # 2. Агенты плана по очереди; выход каждого адресован следующему агенту.
        while msg.receiver in IMPLEMENTED:
            step = msg.receiver
            msg = _deliver(msg, ctx, route=lambda _p, s=step: _next_step(state.plan, after=s))
            if step == "data_agent" and msg.payload["status"] == "error":
                raise AgentError("Не удалось получить данные: " + msg.payload["summary"])
        state.status = "done"
        log.event("task_done", "dispatcher")
    except (AgentError, LLMError) as e:
        state.status, state.error = "failed", str(e)
        log.event("task_failed", "dispatcher", error=str(e))
    except Exception as e:  # noqa: BLE001 — последний рубеж: пользователь видит сообщение, а не трейсбек
        state.status, state.error = "failed", f"внутренняя ошибка {type(e).__name__}: {e}"
        log.event("task_failed", "dispatcher", error=state.error, traceback=traceback.format_exc())
    finally:
        state.save()

    load = log.load_report()
    (run_dir / "load_report.json").write_text(json.dumps(load, ensure_ascii=False, indent=2), encoding="utf-8")
    return RunResult(task_id, state.status, render_answer(state), state.error, load, run_dir, state)


def _next_step(plan: TaskPlan, after: str | None) -> str:
    """Следующий реализованный агент плана после `after` (или первый), иначе 'user'."""
    runnable = [s for s in plan.steps if s in IMPLEMENTED]
    idx = runnable.index(after) + 1 if after in runnable else 0
    return runnable[idx] if idx < len(runnable) else "user"


def _record(msg: Message, ctx: AgentContext) -> None:
    ctx.state.add_message(msg)
    ctx.log.event("message", msg.sender, receiver=msg.receiver, type=msg.type, message_id=msg.message_id)


def _deliver(msg: Message, ctx: AgentContext, route: Callable[[BaseModel], str]) -> Message:
    """Передать сообщение агенту-получателю; его ответ адресовать тому, кого вернёт route."""
    agent = IMPLEMENTED[msg.receiver]
    payload = agent.handle(msg, ctx)
    ctx.state.results[agent.name] = payload.model_dump()
    out = Message(task_id=ctx.task_id, sender=agent.name, receiver=route(payload),
                  type=OUTPUT_TYPE[agent.name], payload=payload.model_dump())
    _record(out, ctx)
    return out


def render_answer(state: TaskState) -> str:
    """Текст ответа для прототипа. В Ассайнменте 4 его будет писать Reporter Agent."""
    if state.status == "failed":
        return f"Не удалось выполнить запрос: {state.error}"
    plan = state.plan
    if plan is None or plan.intent == "out_of_scope":
        return "Я отвечаю только на вопросы о качестве воздуха в городах Казахстана."

    lines = []
    if "data_agent" in state.results:
        data = DataPackage.model_validate(state.results["data_agent"])
        city = data.city
        lines.append(f"Город: {city.name} ({city.lat}, {city.lon})" if city else "Город: ?")
        if data.latest_observed:
            lo = data.latest_observed
            lines.append(f"Последнее значение: {lo.time} — PM2.5 {lo.pm2_5} мкг/м³")
        lines.append(f"Данные: {data.summary}")
        if data.issues:
            lines.append("Проблемы данных: " + "; ".join(data.issues))
    if "forecast_agent" in state.results:
        fc = ForecastResult.model_validate(state.results["forecast_agent"])
        lines.append("\nПрогноз среднесуточного PM2.5 (мкг/м³):")
        for d in fc.days:
            ml = f"ML {d.ml_pm25} [{d.ml_low}–{d.ml_high}]" if d.ml_pm25 is not None else "ML —"
            cams = f"CAMS {d.cams_pm25}" if d.cams_pm25 is not None else "CAMS —"
            lines.append(f"  {d.date}: {d.final_pm25} — {d.category} "
                         f"(источник: {d.source}, уверенность: {d.confidence}; {ml}; {cams})")
        lines.append(f"Обоснование: {fc.rationale}")
    skipped = [s for s in plan.steps if s not in IMPLEMENTED]
    if skipped:
        lines.append(f"\n[прототип] Агенты {', '.join(skipped)} будут добавлены в Ассайнменте 4.")
    return "\n".join(lines)
