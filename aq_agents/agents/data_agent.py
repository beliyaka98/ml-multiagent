"""Data Agent — собирает почасовые данные о воздухе и погоде и проверяет их качество.

LLM решает: как найти город (и что делать, если не нашёлся), какой период загрузить,
что делать при ошибках API, достаточно ли качественные данные.
Код считает: пропуски, последние наблюдения, источник данных — эти цифры не доверяются LLM.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..messages import CityInfo, DataPackage, LatestValues, Message, TaskPlan
from ..runtime import AgentContext, BaseAgent, OutputRejected, Tool
from ..tools import openmeteo
from ..tools.openmeteo import TZ, today_local

SYSTEM = """
Ты — Data Agent многоагентной системы мониторинга качества воздуха в Казахстане.
Твоя роль — собрать почасовые данные о загрязнении воздуха и погоде для города и проверить их качество.
Ты не делаешь прогнозов и не даёшь советов.

Порядок работы:
1. Найди координаты города инструментом geocode_city. Если город не найден, попробуй другое написание
   (например, «Нур-Султан» → «Астана», «Alma-Ata» → «Алматы»).
2. Загрузи fetch_air_quality и fetch_weather с одинаковыми past_days и forecast_days — их можно вызвать
   одновременно. past_days — не меньше 8 (модели прогноза нужна история за 7 дней).
   forecast_days = days_ahead + 2, но не больше 5.
3. Проверь результаты: долю пропусков, источник (network, cache или stale_cache — устаревший кэш),
   правдоподобность значений PM2.5.
4. Если инструмент вернул ошибку, повтори вызов один раз; если снова ошибка — заверши работу
   со статусом partial или error и опиши проблему в issues.

status: ok — оба набора данных получены, пропусков меньше 20%; partial — данные есть, но с проблемами;
error — данных нет. summary — 1–2 предложения: что загружено и последние наблюдаемые значения PM2.5.
"""


class DataDecision(BaseModel):
    status: Literal["ok", "partial", "error"]
    summary: str = Field(max_length=600)
    issues: list[str] = []


def _now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ).tz_localize(None)


def _missing_pct(df: pd.DataFrame) -> float:
    return round(float(df.isna().mean().mean() * 100), 1)


def _air_summary(ref: str, df: pd.DataFrame, source: str, attempts: int) -> dict:
    observed = df[df.index <= _now_local()]
    last = observed["pm2_5"].dropna()
    daily = observed["pm2_5"].resample("D").mean().dropna().tail(3).round(1)
    return {
        "ref": ref,
        "rows": len(df),
        "period": [str(df.index.min()), str(df.index.max())],
        "missing_pct": _missing_pct(df),
        "last_observed": {"time": str(last.index[-1]), "pm2_5": round(float(last.iloc[-1]), 1)} if len(last) else None,
        "daily_pm25_last3": {str(k.date()): v for k, v in daily.items()},
        "max_pm25": round(float(df["pm2_5"].max()), 1) if df["pm2_5"].notna().any() else None,
        "source": source,
        "network_attempts": attempts,
    }


def _weather_summary(ref: str, df: pd.DataFrame, source: str, attempts: int) -> dict:
    return {
        "ref": ref,
        "rows": len(df),
        "period": [str(df.index.min()), str(df.index.max())],
        "missing_pct": _missing_pct(df),
        "mean_temp_c": round(float(df["temperature_2m"].mean()), 1),
        "mean_wind_kmh": round(float(df["wind_speed_10m"].mean()), 1),
        "source": source,
        "network_attempts": attempts,
    }


class DataAgent(BaseAgent):
    name = "data_agent"
    system_prompt = SYSTEM
    decision_model = DataDecision
    accepts = ("task_plan",)

    def parse_input(self, msg: Message, ctx: AgentContext) -> TaskPlan:
        return TaskPlan.model_validate(msg.payload)

    def prompt(self, plan: TaskPlan, ctx: AgentContext) -> str:
        return (f"Сегодня {today_local().isoformat()}.\n"
                f"Город: {plan.city}. Тип запроса: {plan.intent}. days_ahead = {plan.days_ahead}.\n"
                "Собери данные о качестве воздуха и погоде.")

    def make_tools(self, plan: TaskPlan, ctx: AgentContext, facts: dict) -> list[Tool]:
        def geocode_city(name: str) -> dict:
            city = openmeteo.geocode(name)
            facts["city"] = city
            return city

        def fetch_air_quality(lat: float, lon: float, past_days: int, forecast_days: int) -> dict:
            f = openmeteo.air_quality(lat, lon, past_days, forecast_days)
            ref = ctx.state.put_dataset("air", f.df)
            facts["air"] = (ref, f)
            return _air_summary(ref, f.df, f.source, f.attempts)

        def fetch_weather(lat: float, lon: float, past_days: int, forecast_days: int) -> dict:
            f = openmeteo.weather(lat, lon, past_days, forecast_days)
            ref = ctx.state.put_dataset("weather", f.df)
            facts["weather"] = (ref, f)
            return _weather_summary(ref, f.df, f.source, f.attempts)

        coords = {
            "lat": {"type": "number", "description": "Широта"},
            "lon": {"type": "number", "description": "Долгота"},
            "past_days": {"type": "integer", "minimum": 1, "maximum": 30},
        }
        return [
            Tool("geocode_city", "Найти координаты города Казахстана по названию (Open-Meteo Geocoding).",
                 {"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"], "additionalProperties": False},
                 geocode_city),
            Tool("fetch_air_quality",
                 "Загрузить почасовые PM2.5, PM10, NO2 (мкг/м³, модель CAMS): past_days дней истории и "
                 "forecast_days дней прогноза. Данные сохраняются в общее состояние задачи, возвращается сводка.",
                 {"type": "object",
                  "properties": {**coords, "forecast_days": {"type": "integer", "minimum": 1, "maximum": 5}},
                  "required": ["lat", "lon", "past_days", "forecast_days"], "additionalProperties": False},
                 fetch_air_quality),
            Tool("fetch_weather",
                 "Загрузить почасовую погоду (температура, влажность, ветер, давление, осадки, высота "
                 "пограничного слоя): past_days дней истории и forecast_days дней прогноза.",
                 {"type": "object",
                  "properties": {**coords, "forecast_days": {"type": "integer", "minimum": 1, "maximum": 7}},
                  "required": ["lat", "lon", "past_days", "forecast_days"], "additionalProperties": False},
                 fetch_weather),
        ]

    def finalize(self, decision: DataDecision, plan: TaskPlan, ctx: AgentContext, facts: dict) -> DataPackage:
        city, air, weather = facts.get("city"), facts.get("air"), facts.get("weather")
        if decision.status == "error":
            return DataPackage(status="error", city=CityInfo(**city) if city else None, air_ref=None,
                               weather_ref=None, period_start=None, period_end=None, air_missing_pct=None,
                               weather_missing_pct=None, latest_observed=None, data_source="none",
                               summary=decision.summary, issues=decision.issues)
        if not (city and air and weather):
            raise OutputRejected("статус ok/partial возможен только после geocode_city, fetch_air_quality "
                                 "и fetch_weather; загрузи данные или верни status=error")

        (air_ref, air_f), (wx_ref, wx_f) = air, weather
        observed = air_f.df[air_f.df.index <= _now_local()].dropna(subset=["pm2_5"])
        latest = None
        if len(observed):
            row = observed.iloc[-1]
            latest = LatestValues(time=str(observed.index[-1]), pm2_5=round(float(row["pm2_5"]), 1),
                                  pm10=None if pd.isna(row["pm10"]) else round(float(row["pm10"]), 1))

        air_missing, wx_missing = _missing_pct(air_f.df), _missing_pct(wx_f.df)
        status, issues = decision.status, list(decision.issues)
        if status == "ok" and max(air_missing, wx_missing) > 20:  # правило проверяется кодом
            status = "partial"
            issues.append(f"много пропусков: воздух {air_missing}%, погода {wx_missing}%")
        sources = {air_f.source, wx_f.source}
        source = next(s for s in ("stale_cache", "cache", "network") if s in sources)
        if source == "stale_cache":
            issues.append("API недоступен, использован устаревший кэш")

        return DataPackage(
            status=status,
            city=CityInfo(name=city["name"], lat=city["lat"], lon=city["lon"], timezone=city.get("timezone", TZ)),
            air_ref=air_ref,
            weather_ref=wx_ref,
            period_start=str(air_f.df.index.min()),
            period_end=str(air_f.df.index.max()),
            air_missing_pct=air_missing,
            weather_missing_pct=wx_missing,
            latest_observed=latest,
            data_source=source,
            summary=decision.summary,
            issues=issues,
        )
