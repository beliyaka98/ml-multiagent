"""Форматы сообщений между агентами.

Каждое сообщение — это конверт `Message` (кто, кому, тип) и `payload` — одна из
типизированных моделей ниже. Все модели — Pydantic, поэтому любое сообщение
проверяется по схеме и сериализуется в JSON.

Модели для агентов Ассайнмента 4 (Analyst, Health, Reporter, Critic) уже описаны
здесь, чтобы спецификация форматов была полной с первого этапа.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

AgentName = Literal[
    "orchestrator",
    "data_agent",
    "forecast_agent",
    "analyst_agent",
    "health_agent",
    "reporter_agent",
    "critic_agent",
]
Intent = Literal["forecast", "current", "explain", "health", "out_of_scope"]
UserGroup = Literal["general", "children", "elderly", "respiratory", "athletes"]
MessageType = Literal[
    "user_query",
    "task_plan",
    "data_package",
    "forecast_result",
    "analysis_report",
    "health_advice",
    "final_report",
    "critic_verdict",
    "error",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Message(BaseModel):
    """Конверт любого сообщения в системе."""

    message_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_id: str
    sender: str  # имя агента или "user"
    receiver: str  # имя агента или "user"
    type: MessageType
    created_at: str = Field(default_factory=_now)
    payload: dict


# ---------- Пользователь → Orchestrator ----------

class UserQuery(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


# ---------- Orchestrator → Data Agent ----------

class TaskPlan(BaseModel):
    intent: Intent
    city: str | None = Field(None, description="Город на русском, например 'Алматы'")
    days_ahead: int = Field(1, ge=0, le=3, description="0 = сегодня, 1 = завтра, 2 = послезавтра, 3 = через 3 дня")
    user_group: UserGroup = "general"
    steps: list[AgentName] = Field(description="Агенты в порядке выполнения")
    rationale: str = Field(max_length=500, description="Кратко: почему выбран такой план")


# ---------- Data Agent → Forecast Agent ----------

class CityInfo(BaseModel):
    name: str
    lat: float
    lon: float
    timezone: str = "Asia/Almaty"


class LatestValues(BaseModel):
    time: str
    pm2_5: float | None
    pm10: float | None


class DataPackage(BaseModel):
    status: Literal["ok", "partial", "error"]
    city: CityInfo | None
    air_ref: str | None = Field(description="Ссылка на почасовые данные о воздухе в общем состоянии задачи")
    weather_ref: str | None = Field(description="Ссылка на почасовые данные о погоде в общем состоянии задачи")
    period_start: str | None
    period_end: str | None
    air_missing_pct: float | None
    weather_missing_pct: float | None
    latest_observed: LatestValues | None
    data_source: Literal["network", "cache", "stale_cache", "none"]
    summary: str
    issues: list[str] = []


# ---------- Forecast Agent → следующий агент ----------

class DayForecast(BaseModel):
    date: str
    ml_pm25: float | None
    ml_low: float | None
    ml_high: float | None
    cams_pm25: float | None
    final_pm25: float
    source: Literal["ml", "cams", "blend"]
    category: str
    confidence: Literal["low", "medium", "high"]


class ModelInfo(BaseModel):
    name: str
    trained_until: str
    test_mae_by_horizon: dict[str, float]
    baseline_mae_by_horizon: dict[str, float]


class ForecastResult(BaseModel):
    status: Literal["ok", "error"]
    city: str
    days: list[DayForecast]
    model: ModelInfo | None
    rationale: str


# ---------- Агенты Ассайнмента 4 (формат зафиксирован заранее) ----------

class Driver(BaseModel):
    factor: str  # например "низкий пограничный слой (инверсия)"
    evidence: str  # цифры из данных
    strength: Literal["weak", "moderate", "strong"]


class AnalysisReport(BaseModel):
    drivers: list[Driver]
    trend: Literal["rising", "stable", "falling"]
    vs_seasonal_norm_pct: float | None
    summary: str


class HealthAdvice(BaseModel):
    category: str
    risk_level: Literal["low", "moderate", "high", "very_high"]
    who_guideline_exceeded: bool
    groups_at_risk: list[UserGroup]
    recommendations: list[str]


class FinalReport(BaseModel):
    answer: str
    key_numbers: dict[str, float]
    sources: list[str]
    disclaimer: str


class CriticIssue(BaseModel):
    severity: Literal["minor", "major"]
    field: str
    message: str


class CriticVerdict(BaseModel):
    approved: bool
    issues: list[CriticIssue]
    revise_agent: AgentName | None = None


class ErrorPayload(BaseModel):
    agent: str
    error_type: str
    message: str
    retriable: bool
