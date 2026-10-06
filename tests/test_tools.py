"""Инструменты: категории AQI, признаки, HTTP-повторы и кэш, схемы сообщений."""
from __future__ import annotations

import pandas as pd
import pytest
import requests
from pydantic import ValidationError

from aq_agents.messages import Message, TaskPlan
from aq_agents.tools import http
from aq_agents.tools.aqi import pm25_category
from aq_agents.tools.features import build_frame, to_daily
from conftest import synthetic_hourly


@pytest.mark.parametrize("value, label", [
    (5, "хорошо"), (9.0, "хорошо"), (20, "умеренно"), (40, "вредно для чувствительных групп"),
    (100, "вредно"), (200, "очень вредно"), (300, "опасно"),
])
def test_pm25_category(value, label):
    assert pm25_category(value) == label


def test_features_align_target_with_horizon():
    air, weather = synthetic_hourly()
    daily = to_daily(air, weather)
    frame = build_frame(daily, horizons=(2,))
    row = frame.dropna(subset=["target"]).iloc[3]
    assert row["target_date"] == row["base_date"] + pd.Timedelta(days=2)
    assert row["target"] == pytest.approx(daily.loc[row["target_date"], "pm25_mean"])
    assert row["pm25_d0"] == pytest.approx(daily.loc[row["base_date"], "pm25_mean"])


def test_task_plan_schema():
    with pytest.raises(ValidationError):
        TaskPlan(intent="forecast", city="Алматы", days_ahead=7, steps=["data_agent"], rationale="")
    with pytest.raises(ValidationError):
        TaskPlan(intent="weather", city="Алматы", steps=["data_agent"], rationale="")
    msg = Message(task_id="t", sender="user", receiver="orchestrator", type="user_query", payload={"text": "hi"})
    assert Message.model_validate_json(msg.model_dump_json()) == msg


class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


def test_http_retries_then_succeeds_and_caches(settings, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    responses = [requests.ConnectionError("нет сети"), _Resp(503, {}), _Resp(200, {"ok": 1})]

    def fake_get(url, params, timeout):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(http.requests, "get", fake_get)
    data, source, attempts = http.get_json("https://x.test/api", {"a": 1}, cfg=settings)
    assert (data, source, attempts) == ({"ok": 1}, "network", 3)
    assert http.get_json("https://x.test/api", {"a": 1}, cfg=settings)[1] == "cache"


def test_http_falls_back_to_stale_cache(settings, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    monkeypatch.setattr(http.requests, "get", lambda url, params, timeout: _Resp(200, {"v": 1}))
    http.get_json("https://x.test/api", {}, cfg=settings)

    def down(url, params, timeout):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(http.requests, "get", down)
    data, source, _ = http.get_json("https://x.test/api", {}, ttl_s=-1, cfg=settings)  # кэш «протух»
    assert data == {"v": 1} and source == "stale_cache"


def test_http_error_without_cache_is_clear(settings, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    monkeypatch.setattr(http.requests, "get", lambda url, params, timeout: (_ for _ in ()).throw(
        requests.ConnectionError("нет сети")))
    with pytest.raises(http.ToolError, match="недоступен после 3 попыток"):
        http.get_json("https://y.test/api", {}, cfg=settings)


def test_http_client_error_is_not_retried(settings, monkeypatch):
    calls = []

    def bad(url, params, timeout):
        calls.append(1)
        return _Resp(400, {"error": True, "reason": "Parameter 'past_days' is out of range"})

    monkeypatch.setattr(http.requests, "get", bad)
    with pytest.raises(http.ToolError, match="past_days"):
        http.get_json("https://z.test/api", {}, cfg=settings)
    assert len(calls) == 1
