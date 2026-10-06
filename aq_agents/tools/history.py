"""История качества воздуха и погоды (data/training_daily.csv): сезонная норма для Analyst Agent.

Тот же файл используется для обучения ML-модели. В нём среднесуточные данные 8 городов
Казахстана за 2023–2026 годы (CAMS + ERA5 из Open-Meteo).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .http import ToolError

TRAINING_CITIES = {
    "Алматы": (43.24, 76.89),
    "Астана": (51.17, 71.45),
    "Шымкент": (42.32, 69.60),
    "Караганда": (49.80, 73.10),
    "Усть-Каменогорск": (49.95, 82.62),
    "Павлодар": (52.29, 76.95),
    "Актобе": (50.28, 57.17),
    "Атырау": (47.11, 51.88),
}


@lru_cache(maxsize=2)
def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ToolError("Нет исторических данных: запустите python scripts/train_forecast_model.py --refresh")
    return pd.read_csv(path, parse_dates=["date"])


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def nearest_history_city(name: str, lat: float, lon: float) -> tuple[str, int]:
    """Город с историей: тот же, а если его нет в датасете — ближайший (и расстояние до него, км)."""
    if name in TRAINING_CITIES:
        return name, 0
    best = min(TRAINING_CITIES, key=lambda c: _distance_km(lat, lon, *TRAINING_CITIES[c]))
    return best, round(_distance_km(lat, lon, *TRAINING_CITIES[best]))


def monthly_profile(history: pd.DataFrame, city: str) -> pd.DataFrame:
    """Средние по месяцам: PM2.5, высота пограничного слоя, ветер, температура."""
    h = history[history["city"] == city]
    return h.groupby(h["date"].dt.month).agg(
        pm25=("pm25_mean", "mean"), blh_m=("blh", "mean"), wind_kmh=("wind", "mean"), temp_c=("temp", "mean"),
    ).round(1)
