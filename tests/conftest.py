from __future__ import annotations

import dataclasses
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from aq_agents.config import get_settings
from aq_agents.logger import RunLogger
from aq_agents.runtime import AgentContext, Budget
from aq_agents.state import TaskState
from aq_agents.tools import forecast_model, openmeteo
from aq_agents.tools.features import FEATURES, HORIZONS, build_frame, to_daily
from aq_agents.tools.openmeteo import Fetched, today_local


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(get_settings(), runs_dir=tmp_path / "runs", cache_path=tmp_path / "cache.sqlite",
                               max_agent_steps=6, max_total_steps=40, task_timeout_s=60)


@pytest.fixture
def make_ctx(settings, tmp_path):
    def _make(llm, budget: Budget | None = None) -> AgentContext:
        state = TaskState("test", tmp_path / "run")
        log = RunLogger(tmp_path / "run" / "log.jsonl", verbose=False)
        return AgentContext("test", state, log, budget or Budget(40, 60), llm, settings)
    return _make


def synthetic_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Почасовые «данные» за 12 дней назад и 5 дней вперёд с суточным циклом."""
    start = pd.Timestamp(today_local()) - timedelta(days=12)
    idx = pd.date_range(start, periods=17 * 24, freq="h")
    hours = np.arange(len(idx))
    air = pd.DataFrame({
        "pm2_5": 20 + 8 * np.sin(2 * np.pi * hours / 24),
        "pm10": 35 + 10 * np.sin(2 * np.pi * hours / 24),
        "nitrogen_dioxide": 15.0,
    }, index=idx)
    weather = pd.DataFrame({
        "temperature_2m": 5 + 3 * np.sin(2 * np.pi * hours / 24),
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 8.0,
        "surface_pressure": 920.0,
        "precipitation": 0.0,
        "boundary_layer_height": 300 + 200 * np.sin(2 * np.pi * hours / 24),
    }, index=idx)
    return air, weather


@pytest.fixture
def offline_openmeteo(monkeypatch):
    """Подменяет обращения к Open-Meteo синтетическими данными (тесты без сети)."""
    air, weather = synthetic_hourly()
    monkeypatch.setattr(openmeteo, "geocode", lambda name, country_code="KZ": {
        "name": "Алматы", "lat": 43.25, "lon": 76.95, "timezone": "Asia/Almaty", "region": "Алматы",
        "population": 2000000})
    monkeypatch.setattr(openmeteo, "air_quality", lambda lat, lon, past_days, forecast_days: Fetched(air, "network", 1))
    monkeypatch.setattr(openmeteo, "weather", lambda lat, lon, past_days, forecast_days: Fetched(weather, "network", 1))
    return air, weather


@pytest.fixture
def tiny_model(monkeypatch):
    """Маленькая модель на синтетических данных вместо настоящей (тест не зависит от обучения)."""
    air, weather = synthetic_hourly()
    frame = build_frame(to_daily(air, weather)).dropna(subset=FEATURES + ["target"])
    model = HistGradientBoostingRegressor(max_iter=20).fit(frame[FEATURES], np.log1p(frame["target"]))
    card = {"name": "tiny", "trained_until": "2026-01-01",
            "test_mae_by_horizon": {str(h): 1.0 for h in HORIZONS},
            "baseline_mae_by_horizon": {str(h): 2.0 for h in HORIZONS},
            "interval_log_residuals": {str(h): [-0.2, 0.2] for h in HORIZONS}}
    bundle = {"model": model, "features": FEATURES, "card": card}
    monkeypatch.setattr(forecast_model, "load_bundle", lambda path: bundle)
    return bundle
