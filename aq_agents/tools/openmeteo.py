"""Внешние источники данных: Open-Meteo (бесплатно, без ключа).

- Geocoding API — координаты города;
- Air Quality API — PM2.5, PM10, NO2 (модель CAMS), история + прогноз до 5 дней;
- Forecast API — погода (история за последние дни + прогноз);
- Archive API — погодный архив (ERA5) для обучения модели.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .http import ToolError, get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TZ = "Asia/Almaty"
AIR_VARS = ["pm2_5", "pm10", "nitrogen_dioxide"]
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",  # низкий пограничный слой = инверсия, смог «заперт» у земли
]


def today_local() -> date:
    """Сегодняшняя дата по времени Казахстана (а не по часам компьютера)."""
    return pd.Timestamp.now(tz=TZ).date()


@dataclass
class Fetched:
    df: pd.DataFrame
    source: str  # network | cache | stale_cache
    attempts: int


def geocode(name: str, country_code: str = "KZ") -> dict:
    params = {"name": name, "count": 1, "language": "ru", "countryCode": country_code}
    data, _, _ = get_json(GEOCODING_URL, params, ttl_s=30 * 86400)
    results = data.get("results") or []
    if not results:
        raise ToolError(f"Город '{name}' не найден в Казахстане. Попробуйте другое написание.")
    r = results[0]
    return {
        "name": r["name"],
        "lat": round(r["latitude"], 4),
        "lon": round(r["longitude"], 4),
        "timezone": r.get("timezone", TZ),
        "region": r.get("admin1"),
        "population": r.get("population"),
    }


def _hourly(data: dict, columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[columns].astype(float)


def air_quality(lat: float, lon: float, past_days: int, forecast_days: int) -> Fetched:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(AIR_VARS),
              "past_days": past_days, "forecast_days": forecast_days, "timezone": TZ}
    data, source, attempts = get_json(AIR_URL, params)
    return Fetched(_hourly(data, AIR_VARS), source, attempts)


def weather(lat: float, lon: float, past_days: int, forecast_days: int) -> Fetched:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
              "past_days": past_days, "forecast_days": forecast_days, "timezone": TZ}
    data, source, attempts = get_json(FORECAST_URL, params)
    return Fetched(_hourly(data, WEATHER_VARS), source, attempts)


# ---------- История для обучения модели ----------

def air_quality_history(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(AIR_VARS),
              "start_date": start, "end_date": end, "timezone": TZ}
    data, _, _ = get_json(AIR_URL, params, ttl_s=365 * 86400)
    return _hourly(data, AIR_VARS)


def weather_history(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
              "start_date": start, "end_date": end, "timezone": TZ}
    data, _, _ = get_json(ARCHIVE_URL, params, ttl_s=365 * 86400)
    return _hourly(data, WEATHER_VARS)
