"""Forecast Agent — прогноз среднесуточного PM2.5 на сегодня и до 3 дней вперёд.

Два источника: собственная ML-модель и физическая модель CAMS (из данных Data Agent).
LLM решает: какому источнику доверять для каждого дня и насколько уверен прогноз.
Код считает: сами прогнозы, итоговое значение по выбранному источнику и категорию качества воздуха.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..messages import DataPackage, DayForecast, ForecastResult, Message, ModelInfo
from ..runtime import AgentContext, AgentError, BaseAgent, OutputRejected, Tool
from ..tools import forecast_model
from ..tools.aqi import pm25_category
from ..tools.features import to_daily
from ..tools.openmeteo import today_local

SYSTEM = """
Ты — Forecast Agent многоагентной системы мониторинга качества воздуха в Казахстане.
Твоя роль — прогноз среднесуточной концентрации PM2.5 (мкг/м³) на нужные дни. Советов не даёшь.

Источники прогноза:
- run_ml_forecast — наша ML-модель (градиентный бустинг, обучена на истории 8 городов Казахстана),
  возвращает значение и 80%-интервал, а также ошибку модели на тесте;
- get_cams_forecast — прогноз физической модели CAMS (Copernicus) из уже загруженных данных.

Вызови оба инструмента (можно одновременно), сравни прогнозы и для каждой даты выбери источник:
- ml — CAMS недоступен или прогноз CAMS выглядит неправдоподобно на фоне истории;
- cams — ML-прогноз недоступен для этой даты или интервал ML очень широкий;
- blend — источники близки или нет явных причин предпочесть один (итог = среднее двух).
Расхождение = |ML − CAMS| / среднее(ML, CAMS).
confidence: high — расхождение меньше 25%; medium — 25–60%; low — больше 60% или доступен только
один источник. Уверенность можно ставить ниже (например, при проблемах с данными), но не выше —
код это проверяет.
Итоговые числа и категорию качества воздуха посчитает код по твоему выбору — сам их не пиши.
rationale — 1–3 предложения о том, почему выбраны такие источники.
"""

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def max_confidence(ml: float | None, cams: float | None) -> tuple[str, float | None]:
    """Наибольшая допустимая уверенность по расхождению источников (считает код, а не LLM)."""
    if ml is None or cams is None:
        return "low", None
    mean = (ml + cams) / 2
    divergence = abs(ml - cams) / mean if mean > 0 else 0.0
    level = "high" if divergence < 0.25 else "medium" if divergence <= 0.60 else "low"
    return level, divergence


class DayChoice(BaseModel):
    date: str = Field(description="Дата YYYY-MM-DD")
    source: Literal["ml", "cams", "blend"]
    confidence: Literal["low", "medium", "high"]


class ForecastDecision(BaseModel):
    days: list[DayChoice]
    rationale: str = Field(max_length=600)


class ForecastAgent(BaseAgent):
    name = "forecast_agent"
    system_prompt = SYSTEM
    decision_model = ForecastDecision
    accepts = ("data_package",)

    def parse_input(self, msg: Message, ctx: AgentContext) -> DataPackage:
        pkg = DataPackage.model_validate(msg.payload)
        if pkg.status == "error" or not (pkg.air_ref and pkg.weather_ref):
            raise AgentError("forecast_agent: нет данных для прогноза (Data Agent вернул ошибку)")
        return pkg

    def _target_dates(self, ctx: AgentContext) -> list[date]:
        days_ahead = ctx.state.plan.days_ahead if ctx.state.plan else 1
        today = today_local()
        return [today + timedelta(days=i) for i in range(days_ahead + 1)]

    def prompt(self, pkg: DataPackage, ctx: AgentContext) -> str:
        dates = ", ".join(d.isoformat() for d in self._target_dates(ctx))
        return (f"Город: {pkg.city.name if pkg.city else '?'}. Данные: {pkg.summary}\n"
                f"Проблемы данных: {pkg.issues or 'нет'}.\n"
                f"Нужен прогноз на даты: {dates} (days_ahead = {len(self._target_dates(ctx)) - 1}).")

    def make_tools(self, pkg: DataPackage, ctx: AgentContext, facts: dict) -> list[Tool]:
        air = ctx.state.get_dataset(pkg.air_ref)
        weather = ctx.state.get_dataset(pkg.weather_ref)
        targets = self._target_dates(ctx)

        def run_ml_forecast(days_ahead: int) -> dict:
            bundle = forecast_model.load_bundle(ctx.settings.model_path)
            base_day = today_local() - timedelta(days=1)  # последний полный день наблюдений
            horizons = [(d - base_day).days for d in targets[: days_ahead + 1]]
            daily = to_daily(air, weather)
            preds = forecast_model.predict(bundle, daily, base_day, horizons)
            facts["ml"] = {p["date"]: p for p in preds}
            facts["model_card"] = bundle["card"]
            card = bundle["card"]
            return {"base_day": base_day.isoformat(), "forecasts": preds,
                    "test_mae_by_horizon": card["test_mae_by_horizon"],
                    "baseline_mae_by_horizon": card["baseline_mae_by_horizon"]}

        def get_cams_forecast(days_ahead: int) -> dict:
            pm = air["pm2_5"]
            out = []
            for d in targets[: days_ahead + 1]:
                day = pm[pm.index.normalize() == pd.Timestamp(d)].dropna()
                value = round(float(day.mean()), 1) if len(day) >= 18 else None
                out.append({"date": d.isoformat(), "cams_pm25": value, "hours_available": int(len(day))})
            facts["cams"] = {o["date"]: o for o in out}
            return {"forecasts": out}

        arg = {"type": "object",
               "properties": {"days_ahead": {"type": "integer", "minimum": 0, "maximum": 3}},
               "required": ["days_ahead"], "additionalProperties": False}
        return [
            Tool("run_ml_forecast", "Прогноз среднесуточного PM2.5 нашей ML-моделью на сегодня и days_ahead "
                 "дней вперёд, с 80%-интервалом и ошибкой модели на тесте.", arg, run_ml_forecast),
            Tool("get_cams_forecast", "Среднесуточный PM2.5 по прогнозу модели CAMS (Copernicus) на сегодня и "
                 "days_ahead дней вперёд.", arg, get_cams_forecast),
        ]

    def finalize(self, decision: ForecastDecision, pkg: DataPackage, ctx: AgentContext,
                 facts: dict) -> ForecastResult:
        if "ml" not in facts and "cams" not in facts:
            raise OutputRejected("сначала вызови run_ml_forecast и get_cams_forecast")
        ml, cams = facts.get("ml", {}), facts.get("cams", {})
        choices = {c.date: c for c in decision.days}

        days = []
        for d in (t.isoformat() for t in self._target_dates(ctx)):
            choice = choices.get(d)
            if choice is None:
                raise OutputRejected(f"нет выбора источника для даты {d}")
            m = ml.get(d, {}).get("ml_pm25")
            c = cams.get(d, {}).get("cams_pm25")
            if choice.source == "ml" and m is None:
                raise OutputRejected(f"{d}: ML-прогноз недоступен, выбери cams")
            if choice.source == "cams" and c is None:
                raise OutputRejected(f"{d}: прогноз CAMS недоступен, выбери ml")
            if choice.source == "blend" and (m is None or c is None):
                raise OutputRejected(f"{d}: для blend нужны оба прогноза")
            cap, divergence = max_confidence(m, c)
            if CONFIDENCE_RANK[choice.confidence] > CONFIDENCE_RANK[cap]:
                spread = f"расходятся на {divergence:.0%}" if divergence is not None else "доступен один источник"
                raise OutputRejected(f"{d}: уверенность '{choice.confidence}' завышена — ML {m} и CAMS {c} "
                                     f"{spread}, допустимо не выше '{cap}'")
            final = m if choice.source == "ml" else c if choice.source == "cams" else round((m + c) / 2, 1)
            days.append(DayForecast(
                date=d, ml_pm25=m, ml_low=ml.get(d, {}).get("ml_low"), ml_high=ml.get(d, {}).get("ml_high"),
                cams_pm25=c, final_pm25=final, source=choice.source,
                category=pm25_category(final), confidence=choice.confidence,
            ))

        card = facts.get("model_card")
        model = ModelInfo(name=card["name"], trained_until=card["trained_until"],
                          test_mae_by_horizon=card["test_mae_by_horizon"],
                          baseline_mae_by_horizon=card["baseline_mae_by_horizon"]) if card else None
        return ForecastResult(status="ok", city=pkg.city.name if pkg.city else "?", days=days,
                              model=model, rationale=decision.rationale)
