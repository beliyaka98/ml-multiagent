"""Загрузка обученной модели и прогноз PM2.5 на 1–4 дня вперёд.

Модель обучается скриптом scripts/train_forecast_model.py и хранится в
models/pm25_forecast.joblib вместе с «паспортом» (метрики, интервалы).
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, build_frame
from .http import ToolError


@lru_cache(maxsize=2)
def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise ToolError("ML-модель не обучена: запустите python scripts/train_forecast_model.py")
    return joblib.load(path)


def predict(bundle: dict, daily: pd.DataFrame, base_day: date, horizons: list[int]) -> list[dict]:
    """Прогноз на base_day + h для каждого h. Если признаков не хватает — значение None и причина."""
    frame = build_frame(daily, horizons)
    rows = frame[frame["base_date"] == pd.Timestamp(base_day)]
    q = bundle["card"]["interval_log_residuals"]  # {"1": [q10, q90], ...}
    out = []
    for _, row in rows.iterrows():
        h = int(row["horizon"])
        item = {"date": row["target_date"].date().isoformat(), "horizon": h,
                "ml_pm25": None, "ml_low": None, "ml_high": None}
        missing = [f for f in FEATURES if pd.isna(row[f])]
        if missing:
            item["reason"] = f"не хватает данных для признаков: {', '.join(missing)}"
        else:
            x = pd.DataFrame([row[FEATURES].astype(float)])
            log_pred = float(bundle["model"].predict(x)[0])
            lo, hi = q[str(h)]
            item["ml_pm25"] = round(float(np.expm1(log_pred)), 1)
            item["ml_low"] = round(float(np.expm1(log_pred + lo)), 1)
            item["ml_high"] = round(float(np.expm1(log_pred + hi)), 1)
        out.append(item)
    return out
