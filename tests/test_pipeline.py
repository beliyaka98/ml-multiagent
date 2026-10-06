"""Прототип целиком: Orchestrator → Data Agent → Forecast Agent обмениваются сообщениями (без сети и LLM)."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from aq_agents.agents.forecast_agent import max_confidence
from aq_agents.messages import DataPackage, ForecastResult
from aq_agents.pipeline import run_task
from scripted_llm import ScriptedLLM

PLAN = json.dumps({"intent": "forecast", "city": "Алматы", "days_ahead": 1, "user_group": "general",
                   "steps": ["critic_agent", "data_agent", "forecast_agent", "reporter_agent"],
                   "rationale": "прогноз на завтра"}, ensure_ascii=False)

DATA_CALLS = [
    [{"name": "geocode_city", "arguments": {"name": "Алматы"}}],
    [{"name": "fetch_air_quality", "arguments": {"lat": 43.25, "lon": 76.95, "past_days": 10, "forecast_days": 3}},
     {"name": "fetch_weather", "arguments": {"lat": 43.25, "lon": 76.95, "past_days": 10, "forecast_days": 3}}],
    json.dumps({"status": "ok", "summary": "Данные загружены", "issues": []}, ensure_ascii=False),
]


def forecast_decision(conv, confidence=None) -> str:
    ml = conv.last_result("run_ml_forecast")["forecasts"]
    cams = {d["date"]: d["cams_pm25"] for d in conv.last_result("get_cams_forecast")["forecasts"]}
    days = [{"date": d["date"], "source": "blend",
             "confidence": confidence or max_confidence(d["ml_pm25"], cams[d["date"]])[0]} for d in ml]
    return json.dumps({"days": days, "rationale": "источники близки"})


def scripts(**override):
    base = {
        "orchestrator": [PLAN],
        "data_agent": list(DATA_CALLS),
        "forecast_agent": [[{"name": "run_ml_forecast", "arguments": {"days_ahead": 1}},
                            {"name": "get_cams_forecast", "arguments": {"days_ahead": 1}}],
                           forecast_decision],
    }
    base.update(override)
    return base


def test_agents_exchange_messages(settings, offline_openmeteo, tiny_model):
    res = run_task("Какой будет воздух завтра в Алматы?", settings=settings, llm=ScriptedLLM(scripts()),
                   verbose=False)
    assert res.status == "done", res.error

    route = [(m.sender, m.receiver, m.type) for m in res.state.messages]
    assert route == [
        ("user", "orchestrator", "user_query"),
        ("orchestrator", "data_agent", "task_plan"),
        ("data_agent", "forecast_agent", "data_package"),   # Data Agent → Forecast Agent
        ("forecast_agent", "user", "forecast_result"),
    ]
    # план нормализован в канонический порядок
    assert res.state.plan.steps == ["data_agent", "forecast_agent", "reporter_agent", "critic_agent"]

    data = DataPackage.model_validate(res.state.results["data_agent"])
    assert data.status == "ok" and data.air_ref and data.latest_observed is not None

    fc = ForecastResult.model_validate(res.state.results["forecast_agent"])
    assert len(fc.days) == 2
    for d in fc.days:  # итог посчитан кодом как среднее ML и CAMS
        assert d.final_pm25 == round((d.ml_pm25 + d.cams_pm25) / 2, 1)

    assert (res.run_dir / "state.json").exists() and (res.run_dir / "log.jsonl").exists()
    assert res.load["total_calls"] == 1 + (3 + 3) + (2 + 2)
    assert "Прогноз" in res.answer


def test_out_of_scope(settings):
    plan = json.dumps({"intent": "out_of_scope", "city": None, "days_ahead": 0, "user_group": "general",
                       "steps": ["data_agent"], "rationale": "не про воздух"})
    res = run_task("Сколько будет 2+2?", settings=settings, llm=ScriptedLLM({"orchestrator": [plan]}), verbose=False)
    assert res.status == "done"
    assert res.state.plan.steps == []
    assert "только на вопросы о качестве воздуха" in res.answer


def test_orchestrator_plan_is_validated(settings, offline_openmeteo, tiny_model):
    bad = json.dumps({"intent": "forecast", "city": "Алматы", "days_ahead": 1, "user_group": "general",
                      "steps": ["data_agent"], "rationale": "забыл прогноз"})
    res = run_task("Воздух завтра?", settings=settings,
                   llm=ScriptedLLM(scripts(orchestrator=[bad, PLAN])), verbose=False)
    assert res.status == "done"
    assert any(e["kind"] == "output_rejected" for e in json.loads(
        "[" + ",".join((res.run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()) + "]"))


@pytest.mark.parametrize("ml, cams, expected", [
    (16.9, 15.6, "high"), (15.5, 10.8, "medium"), (5.0, 15.0, "low"), (5.0, None, "low"),
])
def test_max_confidence(ml, cams, expected):
    assert max_confidence(ml, cams)[0] == expected


def test_overconfident_forecast_is_rejected_and_fixed(settings, offline_openmeteo, monkeypatch):
    """Модель завысила уверенность — код возвращает замечание, модель исправляется."""
    from aq_agents.tools import forecast_model
    monkeypatch.setattr(forecast_model, "predict", lambda bundle, daily, base_day, horizons: [
        {"date": (base_day + timedelta(days=h)).isoformat(), "horizon": h,
         "ml_pm25": 15.5, "ml_low": 8.7, "ml_high": 22.0} for h in horizons])
    monkeypatch.setattr(forecast_model, "load_bundle", lambda path: {"card": {
        "name": "m", "trained_until": "2026-01-01", "test_mae_by_horizon": {}, "baseline_mae_by_horizon": {}}})
    llm = ScriptedLLM(scripts(forecast_agent=[
        [{"name": "run_ml_forecast", "arguments": {"days_ahead": 1}},
         {"name": "get_cams_forecast", "arguments": {"days_ahead": 1}}],
        lambda conv: forecast_decision(conv, confidence="high"),
        forecast_decision,
    ]))
    # CAMS в синтетических данных ≈ 20 мкг/м³, ML = 15.5: расхождение ~25–30% → не выше medium
    res = run_task("Воздух завтра?", settings=settings, llm=llm, verbose=False)
    assert res.status == "done", res.error
    forecast_conv = [c for c in llm.conversations if c.agent == "forecast_agent"][0]
    assert "завышена" in forecast_conv.user_messages[0]
    assert all(d["confidence"] != "high" for d in res.state.results["forecast_agent"]["days"])


def test_data_error_stops_pipeline_with_clear_message(settings, monkeypatch, tiny_model):
    from aq_agents.tools import openmeteo
    from aq_agents.tools.http import ToolError

    def down(*args, **kwargs):
        raise ToolError("Сервис air-quality-api.open-meteo.com недоступен после 3 попыток")

    monkeypatch.setattr(openmeteo, "geocode", lambda name, country_code="KZ": {
        "name": "Алматы", "lat": 43.25, "lon": 76.95, "timezone": "Asia/Almaty"})
    monkeypatch.setattr(openmeteo, "air_quality", down)
    data_calls = [
        [{"name": "geocode_city", "arguments": {"name": "Алматы"}}],
        [{"name": "fetch_air_quality", "arguments": {"lat": 43.25, "lon": 76.95, "past_days": 10, "forecast_days": 3}}],
        [{"name": "fetch_air_quality", "arguments": {"lat": 43.25, "lon": 76.95, "past_days": 9, "forecast_days": 3}}],
        json.dumps({"status": "error", "summary": "API качества воздуха недоступен", "issues": ["нет данных"]},
                   ensure_ascii=False),
    ]
    res = run_task("Воздух завтра в Алматы?", settings=settings,
                   llm=ScriptedLLM(scripts(data_agent=data_calls)), verbose=False)
    assert res.status == "failed"
    assert "API качества воздуха недоступен" in res.answer
    assert "forecast_agent" not in res.state.results
