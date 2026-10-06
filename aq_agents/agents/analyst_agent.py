"""Analyst Agent — объясняет, ПОЧЕМУ уровень PM2.5 такой: погодные и сезонные факторы.

LLM решает: какие факторы главные, насколько они сильны, как объяснить это простым языком.
Код считает: корреляции, дни застоя воздуха, сезонную норму, тренд — и проверяет, что каждое
объяснение LLM подтверждено числом, которое реально вернули инструменты (защита от выдумок).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from pydantic import BaseModel, Field

from ..messages import AnalysisReport, DataPackage, Driver, ForecastResult, Message
from ..runtime import AgentContext, AgentError, BaseAgent, OutputRejected, Tool
from ..tools.history import load_history, monthly_profile, nearest_history_city
from ..tools.openmeteo import TZ, today_local

SYSTEM = """
Ты — Analyst Agent многоагентной системы мониторинга качества воздуха в Казахстане.
Твоя роль — объяснить, ПОЧЕМУ уровень PM2.5 такой: какие погодные и сезонные факторы на него влияют.
Прогнозов не делаешь и советов по здоровью не даёшь.

Инструменты (их можно вызвать одновременно):
- compute_correlations — связь PM2.5 с погодой за последние дни: корреляции и сравнение погоды
  в самые грязные и самые чистые часы;
- detect_inversion — признаки застоя воздуха по дням: низкий ночной пограничный слой и слабый ветер
  (при таких условиях выбросы остаются у земли, как при температурной инверсии);
- compare_with_seasonal_norm — сезонная норма по истории 2023–2026: средние по месяцам
  и отклонение последних дней от нормы этого месяца.

Сформируй от 1 до 4 факторов (drivers):
- factor — коротко, что влияет (например, «слабый ветер», «низкий пограничный слой», «зимний сезон»);
- evidence — подтверждение с конкретными числами ИЗ РЕЗУЛЬТАТОВ ИНСТРУМЕНТОВ (код проверит, что числа
  взяты оттуда);
- strength — weak, moderate или strong. Для корреляций: |r| ≥ 0.5 — strong, 0.3–0.5 — moderate, < 0.3 — weak.
Не придумывай источники выбросов, которых нет в данных. Об отопительном сезоне можно говорить,
если месяц попадает в период с 15 октября по 15 апреля.
summary — 2–3 предложения простым языком для пользователя.
"""

WEATHER = {  # колонка → подпись для LLM
    "wind_speed_10m": "ветер, км/ч",
    "boundary_layer_height": "пограничный слой, м",
    "temperature_2m": "температура, °C",
    "relative_humidity_2m": "влажность, %",
}
# Пороги подобраны по данным Алматы: ночной слой ниже 150 м там почти каждую ночь, а вот сочетание
# «слой < 100 м и ветер < 3 км/ч» отделяет более грязные дни от остальных.
STAGNATION_BLH_M = 100   # ночной минимум пограничного слоя ниже этого — выбросы «заперты» у земли
STAGNATION_WIND_KMH = 3  # и ветер слабый — воздух не перемешивается
TREND_THRESHOLD = 0.15   # изменение больше 15% — рост или снижение


class AnalystDecision(BaseModel):
    drivers: list[Driver] = Field(min_length=1, max_length=4)
    summary: str = Field(max_length=800)


@dataclass
class AnalystInput:
    data: DataPackage
    forecast: ForecastResult | None


def _num(x) -> float | None:
    """Число для JSON: округление и None вместо NaN."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", ".")) for m in re.findall(r"-?\d+(?:[.,]\d+)?", text)]


def _grounded(value: float, known: list[float]) -> bool:
    """Число из объяснения совпадает с каким-то числом из инструментов (с учётом округления)."""
    for k in known:
        diff = abs(value - k)
        if diff <= 0.051 or diff <= abs(k) * 0.01 or (value.is_integer() and diff < 0.5):
            return True
    return False


class AnalystAgent(BaseAgent):
    name = "analyst_agent"
    system_prompt = SYSTEM
    decision_model = AnalystDecision
    accepts = ("forecast_result", "data_package")

    def parse_input(self, msg: Message, ctx: AgentContext) -> AnalystInput:
        if msg.type == "forecast_result":
            forecast = ForecastResult.model_validate(msg.payload)
            data = DataPackage.model_validate(ctx.state.results.get("data_agent") or {})
        else:
            forecast, data = None, DataPackage.model_validate(msg.payload)
        if data.status == "error" or not (data.air_ref and data.weather_ref and data.city):
            raise AgentError("analyst_agent: нет данных для анализа (Data Agent вернул ошибку)")
        return AnalystInput(data, forecast)

    def prompt(self, inp: AnalystInput, ctx: AgentContext) -> str:
        question = ctx.state.messages[0].payload.get("text", "") if ctx.state.messages else ""
        lines = [f"Сегодня {today_local().isoformat()}. Город: {inp.data.city.name}.",
                 f"Вопрос пользователя: {question}", f"Данные: {inp.data.summary}"]
        if inp.forecast:
            days = "; ".join(f"{d.date}: {d.final_pm25} мкг/м³ ({d.category})" for d in inp.forecast.days)
            lines.append(f"Прогноз Forecast Agent: {days}")
        lines.append("Объясни, почему уровень PM2.5 такой.")
        return "\n".join(lines)

    def make_tools(self, inp: AnalystInput, ctx: AgentContext, facts: dict) -> list[Tool]:
        air = ctx.state.get_dataset(inp.data.air_ref)
        weather = ctx.state.get_dataset(inp.data.weather_ref)
        hourly = air.join(weather, how="inner")
        now = pd.Timestamp.now(tz=TZ).tz_localize(None)
        observed = hourly[hourly.index <= now]
        today = today_local()
        city = inp.data.city

        def compute_correlations(days: int = 7) -> dict:
            recent = observed[observed.index > now - pd.Timedelta(days=days)].dropna(subset=["pm2_5"])
            if len(recent) < 24:
                raise ValueError("мало данных для корреляций (меньше суток наблюдений)")
            hi, lo = recent["pm2_5"].quantile([0.75, 0.25])
            dirty, clean = recent[recent["pm2_5"] >= hi], recent[recent["pm2_5"] <= lo]
            out = {
                "period_days": days,
                "hours": len(recent),
                "pm25_mean_dirty_hours": _num(dirty["pm2_5"].mean()),
                "pm25_mean_clean_hours": _num(clean["pm2_5"].mean()),
                "correlation_with_pm25": {label: _num(recent["pm2_5"].corr(recent[col]))
                                          if recent[col].std() > 0 else None  # постоянная величина — связи нет
                                          for col, label in WEATHER.items()},
                "weather_dirty_vs_clean_hours": {label: {"грязные часы": _num(dirty[col].mean()),
                                                         "чистые часы": _num(clean[col].mean())}
                                                 for col, label in WEATHER.items()},
            }
            facts["correlations"] = out
            return out

        def detect_inversion() -> dict:
            days = []
            window = hourly[hourly.index >= pd.Timestamp(today - timedelta(days=5))]
            for day, g in window.groupby(window.index.normalize()):
                night = g[(g.index.hour <= 6) | (g.index.hour >= 22)]
                blh_min, wind = night["boundary_layer_height"].min(), night["wind_speed_10m"].mean()
                d = day.date()
                days.append({
                    "date": d.isoformat(),
                    "kind": "сегодня" if d == today else "наблюдение" if d < today else "прогноз",
                    "night_blh_min_m": _num(blh_min),
                    "night_wind_kmh": _num(wind),
                    "pm25_daily_mean": _num(g["pm2_5"].mean()),
                    "stagnation": bool(blh_min < STAGNATION_BLH_M and wind < STAGNATION_WIND_KMH),
                })
            def mean_pm(flag: bool) -> float | None:
                values = [d["pm25_daily_mean"] for d in days if d["stagnation"] == flag and d["pm25_daily_mean"]]
                return _num(sum(values) / len(values)) if values else None

            out = {"rule": f"застой = ночной минимум пограничного слоя < {STAGNATION_BLH_M} м "
                           f"и ночной ветер < {STAGNATION_WIND_KMH} км/ч",
                   "stagnation_days": sum(d["stagnation"] for d in days),
                   "pm25_mean_stagnation_days": mean_pm(True),
                   "pm25_mean_other_days": mean_pm(False),
                   "days": days}
            facts["inversion"] = out
            return out

        def compare_with_seasonal_norm() -> dict:
            hist_city, km = nearest_history_city(city.name, city.lat, city.lon)
            profile = monthly_profile(load_history(ctx.settings.history_path), hist_city)
            norm = float(profile.loc[today.month, "pm25"])
            daily = observed["pm2_5"].resample("D").mean()
            recent = daily[daily.index < pd.Timestamp(today)].dropna().tail(3)
            series = [{"date": str(k.date()), "pm25": _num(v), "kind": "наблюдение"} for k, v in recent.items()]
            if inp.forecast:
                series += [{"date": d.date, "pm25": d.final_pm25, "kind": "прогноз"} for d in inp.forecast.days]
            values = [s["pm25"] for s in series if s["pm25"] is not None]
            change = (values[-1] - values[0]) / max(sum(values) / len(values), 0.1) if len(values) >= 2 else 0.0
            trend = "rising" if change > TREND_THRESHOLD else "falling" if change < -TREND_THRESHOLD else "stable"
            recent_mean = float(recent.mean()) if len(recent) else None
            out = {
                "history_city": hist_city,
                "distance_to_history_city_km": km,
                "history_years": "2023–2026",
                "month": today.month,
                "norm_pm25_this_month": _num(norm),
                "recent_3d_mean_pm25": _num(recent_mean),
                "deviation_from_norm_pct": _num((recent_mean - norm) / norm * 100) if recent_mean and norm else None,
                "who_daily_guideline": 15,
                "trend": trend,
                "daily_series": series,
                "monthly_profile": [{"month": int(m), **{k: _num(v) for k, v in row.items()}}
                                    for m, row in profile.iterrows()],
            }
            facts["seasonal"] = out
            return out

        none = {"type": "object", "properties": {}, "additionalProperties": False}
        return [
            Tool("compute_correlations", "Связь PM2.5 с погодой за последние days дней (по часам): корреляции "
                 "и средняя погода в самые грязные и самые чистые часы.",
                 {"type": "object", "properties": {"days": {"type": "integer", "minimum": 2, "maximum": 14}},
                  "additionalProperties": False}, compute_correlations),
            Tool("detect_inversion", "Признаки застоя воздуха по дням (последние 5 дней и прогноз): ночной минимум "
                 "пограничного слоя, ночной ветер, среднесуточный PM2.5.", none, detect_inversion),
            Tool("compare_with_seasonal_norm", "Сезонная норма PM2.5 из истории 2023–2026: средние по месяцам "
                 "(с погодой), норма текущего месяца, отклонение последних дней от нормы, тренд.",
                 none, compare_with_seasonal_norm),
        ]

    def finalize(self, decision: AnalystDecision, inp: AnalystInput, ctx: AgentContext,
                 facts: dict) -> AnalysisReport:
        if not facts:
            raise OutputRejected("сначала вызови инструменты анализа")
        known = _numbers(json.dumps(facts, ensure_ascii=False))
        for d in decision.drivers:
            numbers = _numbers(d.evidence)
            if not numbers:
                raise OutputRejected(f"в evidence фактора «{d.factor}» нет чисел — подтверди его данными инструментов")
            if not any(_grounded(n, known) for n in numbers):
                raise OutputRejected(f"числа в evidence фактора «{d.factor}» ({', '.join(map(str, numbers))}) "
                                     "не найдены в результатах инструментов — используй их значения")
        seasonal = facts.get("seasonal") or {}
        monthly = {str(r["month"]): r["pm25"] for r in seasonal.get("monthly_profile", []) if r["pm25"] is not None}
        return AnalysisReport(
            drivers=decision.drivers,
            trend=seasonal.get("trend", "stable"),
            vs_seasonal_norm_pct=seasonal.get("deviation_from_norm_pct"),
            summary=decision.summary,
            history_city=seasonal.get("history_city"),
            monthly_pm25=monthly or None,
        )
