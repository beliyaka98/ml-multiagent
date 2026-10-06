"""Признаки для ML-модели прогноза PM2.5.

Один и тот же код используется при обучении и при прогнозе, чтобы признаки
совпадали. Базовый день (base) — последний полный день наблюдений, целевой
день (target) = base + horizon, horizon = 1..4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 4)
MIN_HOURS_PER_DAY = 18

FEATURES = [
    "horizon",
    "pm25_d0",          # среднее PM2.5 в базовый день
    "pm25_d1",          # среднее PM2.5 за день до базового
    "pm25_7d",          # среднее за 7 дней
    "pm25_max_d0",      # максимум в базовый день
    "temp_t",           # погода в целевой день
    "rh_t",
    "wind_t",
    "pressure_t",
    "precip_t",
    "blh_t",            # высота пограничного слоя (среднее)
    "blh_min_t",        # минимум пограничного слоя (ночная инверсия)
    "temp_change",      # изменение температуры base → target
    "pressure_change",
    "month_sin",
    "month_cos",
    "dow_t",
    "heating_season",   # отопительный сезон (15 окт — 15 апр)
]


def to_daily(air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Почасовые данные → среднесуточные."""
    pm = air["pm2_5"]
    daily = pd.DataFrame({
        "pm25_mean": pm.resample("D").mean(),
        "pm25_max": pm.resample("D").max(),
    })
    hours = pm.resample("D").count()
    daily.loc[hours < MIN_HOURS_PER_DAY, ["pm25_mean", "pm25_max"]] = np.nan

    w = weather
    daily = daily.join(pd.DataFrame({
        "temp": w["temperature_2m"].resample("D").mean(),
        "rh": w["relative_humidity_2m"].resample("D").mean(),
        "wind": w["wind_speed_10m"].resample("D").mean(),
        "pressure": w["surface_pressure"].resample("D").mean(),
        "precip": w["precipitation"].resample("D").sum(min_count=MIN_HOURS_PER_DAY),
        "blh": w["boundary_layer_height"].resample("D").mean(),
        "blh_min": w["boundary_layer_height"].resample("D").min(),
    }), how="outer")
    return daily.asfreq("D")


def _heating(dates: pd.DatetimeIndex) -> np.ndarray:
    m, d = dates.month, dates.day
    return ((m >= 11) | (m <= 3) | ((m == 10) & (d >= 15)) | ((m == 4) & (d <= 15))).astype(int)


def build_frame(daily: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Строки (base_date, horizon) с признаками и целевой переменной `target`.

    Ничего не выбрасывает: при обучении строки с NaN отбрасываются отдельно,
    при прогнозе target, конечно, неизвестен.
    """
    d = daily.asfreq("D")
    base = pd.DataFrame(index=d.index)
    base["pm25_d0"] = d["pm25_mean"]
    base["pm25_d1"] = d["pm25_mean"].shift(1)
    base["pm25_7d"] = d["pm25_mean"].rolling(7, min_periods=5).mean()
    base["pm25_max_d0"] = d["pm25_max"]

    frames = []
    for h in horizons:
        t = d.shift(-h)  # значения целевого дня, выровненные по базовому дню
        target_dates = d.index + pd.Timedelta(days=h)
        x = base.copy()
        x["horizon"] = h
        x["temp_t"] = t["temp"]
        x["rh_t"] = t["rh"]
        x["wind_t"] = t["wind"]
        x["pressure_t"] = t["pressure"]
        x["precip_t"] = t["precip"]
        x["blh_t"] = t["blh"]
        x["blh_min_t"] = t["blh_min"]
        x["temp_change"] = t["temp"] - d["temp"]
        x["pressure_change"] = t["pressure"] - d["pressure"]
        x["month_sin"] = np.sin(2 * np.pi * target_dates.month / 12)
        x["month_cos"] = np.cos(2 * np.pi * target_dates.month / 12)
        x["dow_t"] = target_dates.dayofweek
        x["heating_season"] = _heating(target_dates)
        x["target"] = t["pm25_mean"]
        x["base_date"] = d.index
        x["target_date"] = target_dates
        frames.append(x)
    return pd.concat(frames, ignore_index=True)
