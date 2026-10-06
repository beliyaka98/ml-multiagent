"""Analyst Agent: инструменты анализа, проверка чисел в объяснениях, маршруты сообщений."""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from aq_agents.agents.analyst_agent import AnalystAgent, AnalystDecision, AnalystInput
from aq_agents.messages import AnalysisReport, CityInfo, DataPackage, Driver
from aq_agents.pipeline import run_task
from aq_agents.runtime import OutputRejected
from aq_agents.tools.openmeteo import today_local
from scripted_llm import ScriptedLLM


def stagnant_city_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Чем слабее ветер и ниже пограничный слой, тем грязнее воздух; ночи — застойные."""
    idx = pd.date_range(pd.Timestamp(today_local()) - timedelta(days=8), periods=12 * 24, freq="h")
    phase = -np.cos(2 * np.pi * (idx.hour - 3) / 24)  # минимум в 3 часа ночи
    wind = 5 + 4 * phase          # ночью ветер ~1–3 км/ч
    blh = 400 + 330 * phase       # ночью пограничный слой ~70–150 м
    air = pd.DataFrame({"pm2_5": 40 - 3 * wind, "pm10": 50.0, "nitrogen_dioxide": 10.0}, index=idx)
    weather = pd.DataFrame({"temperature_2m": -5.0, "relative_humidity_2m": 70.0, "wind_speed_10m": wind,
                            "surface_pressure": 920.0, "precipitation": 0.0, "boundary_layer_height": blh},
                           index=idx)
    return air, weather


@pytest.fixture
def analyst_setup(make_ctx):
    ctx = make_ctx(llm=None)
    air, weather = stagnant_city_data()
    data = DataPackage(status="ok", city=CityInfo(name="Алматы", lat=43.25, lon=76.95),
                       air_ref=ctx.state.put_dataset("air", air), weather_ref=ctx.state.put_dataset("weather", weather),
                       period_start=None, period_end=None, air_missing_pct=0, weather_missing_pct=0,
                       latest_observed=None, data_source="network", summary="ok")
    facts: dict = {}
    agent = AnalystAgent()
    inp = AnalystInput(data, None)
    tools = {t.name: t for t in agent.make_tools(inp, ctx, facts)}
    return agent, inp, ctx, facts, tools


def test_tools_find_weather_drivers(analyst_setup):
    agent, inp, ctx, facts, tools = analyst_setup
    corr = tools["compute_correlations"].fn(days=5)
    assert corr["correlation_with_pm25"]["ветер, км/ч"] == pytest.approx(-1, abs=0.01)
    assert corr["correlation_with_pm25"]["пограничный слой, м"] < -0.9
    dirty_clean = corr["weather_dirty_vs_clean_hours"]["ветер, км/ч"]
    assert dirty_clean["грязные часы"] < dirty_clean["чистые часы"]

    inversion = tools["detect_inversion"].fn()
    assert inversion["stagnation_days"] >= 5
    assert {d["kind"] for d in inversion["days"]} >= {"наблюдение", "сегодня", "прогноз"}

    seasonal = tools["compare_with_seasonal_norm"].fn()
    assert seasonal["history_city"] == "Алматы" and seasonal["month"] == today_local().month
    assert len(seasonal["monthly_profile"]) == 12 and seasonal["trend"] in {"rising", "stable", "falling"}
    assert set(tools) == {"compute_correlations", "detect_inversion", "compare_with_seasonal_norm"}


def test_evidence_must_use_numbers_from_tools(analyst_setup):
    agent, inp, ctx, facts, tools = analyst_setup
    corr = tools["compute_correlations"].fn(days=5)
    tools["compare_with_seasonal_norm"].fn()
    r = corr["correlation_with_pm25"]["ветер, км/ч"]

    def decision(evidence: str) -> AnalystDecision:
        return AnalystDecision(drivers=[Driver(factor="слабый ветер", evidence=evidence, strength="strong")],
                               summary="Воздух грязнее, когда ветер слабый.")

    with pytest.raises(OutputRejected, match="нет чисел"):
        agent.finalize(decision("ветер почти отсутствовал"), inp, ctx, facts)
    with pytest.raises(OutputRejected, match="не найдены"):
        agent.finalize(decision("корреляция с ветром 0.123, ветер 99.9 км/ч"), inp, ctx, facts)

    report = agent.finalize(decision(f"корреляция PM2.5 с ветром r = {r:.2f}"), inp, ctx, facts)
    assert isinstance(report, AnalysisReport) and report.history_city == "Алматы"
    assert report.monthly_pm25 and len(report.monthly_pm25) == 12


def test_nearest_history_city_for_unknown_city(analyst_setup):
    from aq_agents.tools.history import nearest_history_city
    assert nearest_history_city("Алматы", 43.2, 76.9) == ("Алматы", 0)
    city, km = nearest_history_city("Талдыкорган", 45.02, 78.37)
    assert city == "Алматы" and 200 < km < 300


def analyst_script():
    def decide(conv) -> str:
        norm = conv.last_result("compare_with_seasonal_norm")["norm_pm25_this_month"]
        return json.dumps({"drivers": [{"factor": "сезон", "evidence": f"норма месяца {norm} мкг/м³",
                                        "strength": "moderate"}],
                           "summary": "Уровень близок к сезонной норме."}, ensure_ascii=False)
    return [[{"name": "compute_correlations", "arguments": {"days": 7}},
             {"name": "detect_inversion"}, {"name": "compare_with_seasonal_norm"}], decide]


def plan(intent: str, steps: list[str]) -> str:
    return json.dumps({"intent": intent, "city": "Алматы", "days_ahead": 1, "user_group": "general",
                       "steps": steps, "rationale": "тест"}, ensure_ascii=False)


DATA_CALLS = [
    [{"name": "geocode_city", "arguments": {"name": "Алматы"}}],
    [{"name": "fetch_air_quality", "arguments": {"lat": 43.25, "lon": 76.95, "past_days": 10, "forecast_days": 3}},
     {"name": "fetch_weather", "arguments": {"lat": 43.25, "lon": 76.95, "past_days": 10, "forecast_days": 3}}],
    json.dumps({"status": "ok", "summary": "Данные загружены", "issues": []}, ensure_ascii=False),
]


def test_explain_route_data_to_analyst(settings, offline_openmeteo):
    llm = ScriptedLLM({"orchestrator": [plan("explain", ["data_agent", "analyst_agent", "critic_agent"])],
                       "data_agent": list(DATA_CALLS), "analyst_agent": analyst_script()})
    res = run_task("Почему в Алматы грязный воздух?", settings=settings, llm=llm, verbose=False)
    assert res.status == "done", res.error
    route = [(m.sender, m.receiver, m.type) for m in res.state.messages]
    assert route[-2:] == [("data_agent", "analyst_agent", "data_package"),
                          ("analyst_agent", "user", "analysis_report")]
    assert "Почему так" in res.answer


def test_forecast_then_analyst_route(settings, offline_openmeteo, tiny_model):
    from test_pipeline import forecast_decision
    llm = ScriptedLLM({
        "orchestrator": [plan("forecast", ["data_agent", "forecast_agent", "analyst_agent"])],
        "data_agent": list(DATA_CALLS),
        "forecast_agent": [[{"name": "run_ml_forecast", "arguments": {"days_ahead": 1}},
                            {"name": "get_cams_forecast", "arguments": {"days_ahead": 1}}], forecast_decision],
        "analyst_agent": analyst_script(),
    })
    res = run_task("Почему завтра в Алматы будет смог?", settings=settings, llm=llm, verbose=False)
    assert res.status == "done", res.error
    route = [(m.sender, m.receiver, m.type) for m in res.state.messages]
    assert route[-3:] == [("data_agent", "forecast_agent", "data_package"),
                          ("forecast_agent", "analyst_agent", "forecast_result"),
                          ("analyst_agent", "user", "analysis_report")]
    analyst_conv = [c for c in llm.conversations if c.agent == "analyst_agent"][0]
    assert "Прогноз Forecast Agent" in analyst_conv.user  # аналитик видит прогноз
    assert res.load["total_calls"] == 1 + 6 + 4 + (2 + 3)
