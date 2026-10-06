"""Веб-интерфейс: наглядная работа многоагентной системы.

Запуск:  python -m streamlit run app.py

Два режима:
- «Новый запрос» — настоящий запуск агентов; каждое событие логгера сразу отображается на схеме;
- «Повтор прогона» — проигрывает сохранённый запуск по его логу (без интернета и без расхода квоты LLM).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st

from aq_agents.config import ROOT, get_settings
from aq_agents.logger import LOAD_LIMIT, compute_load, format_event, read_events
from aq_agents.messages import TaskPlan
from aq_agents.pipeline import IMPLEMENTED, render_answer, run_task
from aq_agents.tools.aqi import WHO_PM25_24H, pm25_category

st.set_page_config(page_title="AirQ-KZ · агенты", page_icon="🌫️", layout="wide")

AGENTS = [  # порядок на схеме: (ключ, название, роль)
    ("orchestrator", "Orchestrator", "понимает запрос, строит план"),
    ("data_agent", "Data Agent", "данные о воздухе и погоде"),
    ("forecast_agent", "Forecast Agent", "прогноз PM2.5: ML и CAMS"),
    ("analyst_agent", "Analyst Agent", "причины загрязнения"),
    ("health_agent", "Health Agent", "рекомендации для здоровья"),
    ("reporter_agent", "Reporter Agent", "итоговый ответ"),
    ("critic_agent", "Critic Agent", "проверка ответа"),
]
NAMES = {key: title for key, title, _ in AGENTS}
EXAMPLES = [
    "Какой будет воздух завтра в Алматы?",
    "Можно ли послезавтра гулять с ребёнком в Астане?",
    "Какой сейчас воздух в Шымкенте?",
    "Почему в Алматы зимой смог?",
    "Сколько будет 2+2?",
]
INTENTS = {"forecast": "прогноз", "current": "текущее состояние", "explain": "объяснение причин",
           "health": "здоровье", "out_of_scope": "не по теме"}
GROUPS = {"general": "все", "children": "дети", "elderly": "пожилые", "respiratory": "болезни дыхания",
          "athletes": "спортсмены"}
STATUS = {  # статус агента на схеме: (значок, подпись)
    "waiting": ("○", "ожидает"),
    "working": ("◔", "работает…"),
    "done": ("✓", "готово"),
    "error": ("✕", "ошибка"),
    "later": ("◇", "Ассайнмент 4"),
    "unused": ("–", "не нужен"),
}

# Цвета графиков: проверенная палитра (различима, в том числе при дальтонизме)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK_2, MUTED, GRID, AXIS, SURFACE, CRITICAL = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#d03b3b"
ML_LABEL, CAMS_LABEL = "ML-прогноз, среднее за сутки (80% интервал)", "CAMS, по часам"

CSS = """
<style>
.pipeline {display:flex; align-items:stretch; gap:4px; overflow-x:auto; margin:4px 0 14px; padding:4px 2px}
.agent {flex:1 1 0; min-width:96px; border:1px solid #d9d8d2; border-radius:10px;
        padding:9px 8px 7px; background:#fcfcfb; transition:all .25s}
.agent .t {font-weight:600; font-size:13px; color:#0b0b0b}
.agent .r {font-size:11.5px; color:#52514e; min-height:32px; line-height:1.3}
.agent .b {font-size:12px; font-weight:600; margin-top:6px; color:#898781}
.agent .c {font-size:11px; color:#898781; margin-top:2px}
.arrow {align-self:center; color:#898781; font-size:14px}
.agent.working {border:2px solid #2a78d6; animation:pulse 1.1s ease-in-out infinite}
.agent.working .b {color:#2a78d6}
.agent.done {border:2px solid #0ca30c}
.agent.done .b {color:#006300}
.agent.error {border:2px solid #d03b3b}
.agent.error .b {color:#d03b3b}
.agent.later {border-style:dashed; opacity:.6}
.agent.unused {opacity:.45}
@keyframes pulse {0%,100% {box-shadow:0 0 0 0 rgba(42,120,214,.35)} 50% {box-shadow:0 0 0 6px rgba(42,120,214,0)}}
</style>
"""


class Board:
    """Состояние схемы агентов. Обновляется по событиям лога — и вживую, и при повторе прогона."""

    def __init__(self):
        self.status = {key: "waiting" for key, *_ in AGENTS}
        self.llm = {key: 0 for key in self.status}
        self.tools = {key: 0 for key in self.status}
        self.plan: dict | None = None
        self.feed: list[str] = []
        self.outcome: str | None = None

    def apply(self, e: dict) -> None:
        kind, agent = e["kind"], e["agent"]
        known = agent in self.status
        if kind == "agent_start" and known:
            self.status[agent] = "working"
        elif kind == "agent_end" and known:
            self.status[agent] = "done"
        elif kind == "agent_skipped" and known:
            self.status[agent] = "later"
        elif kind == "llm_call" and known:
            self.llm[agent] += 1
        elif kind == "tool_call" and known:
            self.tools[agent] += 1
        elif kind == "llm_error" and known:
            self.status[agent] = "error"
        elif kind == "plan":
            self.plan = e
            for key, *_ in AGENTS[1:]:
                if key not in e["steps"]:
                    self.status[key] = "unused"
        elif kind == "task_done":
            self.outcome = "done"
        elif kind == "task_failed":
            self.outcome = "failed"
            for key, s in self.status.items():
                if s == "working":
                    self.status[key] = "error"
        self.feed.append(feed_line(e))


def feed_line(e: dict) -> str:
    if e["kind"] == "task_start":
        return f"[{e['t']:6.1f}s] вопрос: {e['query']}"
    if e["kind"] == "plan":
        return f"[{e['t']:6.1f}s] orchestrator    план: {' → '.join(e['steps']) or 'не по теме'}"
    if e["kind"] == "task_failed":
        return f"[{e['t']:6.1f}s] ✕ ошибка: {e['error']}"
    return format_event(e)


def pipeline_html(board: Board) -> str:
    cards = []
    for key, title, role in AGENTS:
        status = board.status[key]
        icon, label = STATUS[status]
        calls = f"LLM {board.llm[key]} · инструменты {board.tools[key]}" if key in IMPLEMENTED else "в разработке"
        cards.append(f'<div class="agent {status}"><div class="t">{title}</div><div class="r">{role}</div>'
                     f'<div class="b">{icon} {label}</div><div class="c">{calls}</div></div>')
    return '<div class="pipeline">' + '<div class="arrow">→</div>'.join(cards) + "</div>"


def plan_markdown(plan: dict | None) -> str:
    if plan is None:
        return "**План Orchestrator'а** появится после первого шага."
    days = {0: "сегодня", 1: "завтра", 2: "послезавтра", 3: "через 3 дня"}.get(plan["days_ahead"], "")
    steps = " → ".join(NAMES[s] for s in plan["steps"]) or "—"
    return (f"**План Orchestrator'а**\n\n"
            f"- Тип запроса: **{INTENTS.get(plan['intent'], plan['intent'])}**\n"
            f"- Город: **{plan.get('city') or '—'}**, горизонт: **{days}**\n"
            f"- Группа: **{GROUPS.get(plan['user_group'], plan['user_group'])}**\n"
            f"- Шаги: {steps}\n\n> {plan['rationale']}")


# ---------- Графики ----------

def _style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (chart.configure(background=SURFACE)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=MUTED,
                            titleColor=INK_2, labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=INK_2, labelFontSize=12, labelLimit=0, symbolStrokeWidth=2))


def forecast_chart(air: pd.DataFrame, days: list[dict], latest_time: str | None) -> alt.TopLevelMixin:
    anchor = pd.Timestamp(days[0]["date"]) if days else pd.Timestamp(latest_time or air.index.max()).normalize()
    start = anchor - pd.Timedelta(days=3)
    hourly = air.loc[air.index >= start, ["pm2_5"]].dropna().reset_index()
    hourly.columns = ["Время", "pm25"]  # без точки в имени: Vega-Lite читает её как вложенное поле
    hourly["Ряд"] = CAMS_LABEL
    daily = pd.DataFrame([{
        "Время": pd.Timestamp(d["date"]) + pd.Timedelta(hours=12), "pm25": d["ml_pm25"],
        "low": d["ml_low"], "high": d["ml_high"], "Ряд": ML_LABEL, "Дата": d["date"],
        "Итог": d["final_pm25"], "Категория": d["category"],
    } for d in days if d["ml_pm25"] is not None])

    x = alt.X("Время:T", title=None, axis=alt.Axis(format="%d.%m", tickCount="day"))
    y = alt.Y("pm25:Q", title="PM2.5, мкг/м³")
    if daily.empty:  # один ряд — легенда не нужна, его называет подпись под графиком
        color = alt.Color("Ряд:N", scale=alt.Scale(domain=[CAMS_LABEL], range=[ORANGE]), legend=None)
    else:
        color = alt.Color("Ряд:N", scale=alt.Scale(domain=[ML_LABEL, CAMS_LABEL], range=[BLUE, ORANGE]),
                          legend=alt.Legend(orient="top", title=None))
    hover = alt.selection_point(fields=["Время"], nearest=True, on="pointerover", empty=False)

    line = alt.Chart(hourly).mark_line(strokeWidth=2, interpolate="monotone").encode(x, y, color)
    crosshair = alt.Chart(hourly).mark_rule(color=AXIS).encode(x).transform_filter(hover)
    hover_dots = alt.Chart(hourly).mark_point(size=70, filled=True, stroke=SURFACE, strokeWidth=2).encode(
        x, y, color, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Время:T", format="%d.%m %H:%M"), alt.Tooltip("pm25:Q", title="PM2.5", format=".1f")],
    ).add_params(hover)
    layers = [line, crosshair, hover_dots]

    if not daily.empty:
        tooltip = [alt.Tooltip("Дата:N"), alt.Tooltip("pm25:Q", title="ML", format=".1f"),
                   alt.Tooltip("low:Q", title="от", format=".1f"), alt.Tooltip("high:Q", title="до", format=".1f"),
                   alt.Tooltip("Итог:Q", format=".1f"), alt.Tooltip("Категория:N")]
        layers += [
            alt.Chart(daily).mark_rule(strokeWidth=2).encode(x, alt.Y("low:Q"), y2="high:Q", color=color),
            alt.Chart(daily).mark_point(size=120, filled=True, opacity=1, stroke=SURFACE, strokeWidth=2)
            .encode(x, y, color, tooltip=tooltip),
        ]

    who = pd.DataFrame({"y": [WHO_PM25_24H]})  # подпись пунктира — под графиком, чтобы не наезжала на линию
    layers.append(alt.Chart(who).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q"))
    if latest_time:
        now = pd.DataFrame({"t": [pd.Timestamp(latest_time)]})
        layers += [
            alt.Chart(now).mark_rule(color=MUTED).encode(x="t:T"),
            alt.Chart(now).mark_text(align="left", baseline="top", dx=4, y=4, color=INK_2, fontSize=11)
            .encode(x="t:T", text=alt.value("последнее наблюдение")),
        ]
    return _style(alt.layer(*layers).properties(height=330))


def load_chart(rows: list[dict]) -> alt.TopLevelMixin:
    df = pd.DataFrame(rows)
    df["Агент"] = df["agent"].map(NAMES)
    x = alt.X("share:Q", title="доля всех вызовов (LLM + инструменты)", axis=alt.Axis(format="%"),
              scale=alt.Scale(domain=[0, max(0.7, df["share"].max() + 0.1)]))
    y = alt.Y("Агент:N", sort="-x", title=None)
    tooltip = ["Агент:N", alt.Tooltip("llm_calls:Q", title="вызовы LLM"),
               alt.Tooltip("tool_calls:Q", title="вызовы инструментов"), alt.Tooltip("share:Q", format=".0%")]
    bars = alt.Chart(df).mark_bar(size=22, cornerRadiusEnd=4, color=BLUE).encode(x, y, tooltip=tooltip)
    labels = alt.Chart(df).mark_text(align="left", dx=5, color=INK_2, fontSize=12).encode(
        x, y, text=alt.Text("share:Q", format=".0%"))
    limit = pd.DataFrame({"x": [LOAD_LIMIT]})
    rule = alt.Chart(limit).mark_rule(color=CRITICAL, strokeDash=[4, 4]).encode(x="x:Q")
    rule_txt = alt.Chart(limit).mark_text(align="left", dx=4, y=-8, color=CRITICAL, fontSize=11).encode(
        x="x:Q", text=alt.value("лимит 40%"))
    return _style(alt.layer(bars, labels, rule, rule_txt).properties(height=46 * len(df) + 30))


MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def season_chart(monthly: dict[str, float], current_month: int) -> alt.TopLevelMixin:
    """Средний PM2.5 по месяцам: текущий месяц выделен цветом, остальные — серым."""
    df = pd.DataFrame([{"Месяц": MONTHS[int(m) - 1], "n": int(m), "pm25": v,
                        "Текущий": int(m) == current_month} for m, v in monthly.items()])
    x = alt.X("Месяц:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0))
    bars = alt.Chart(df).mark_bar(size=22, cornerRadiusEnd=4).encode(
        x, alt.Y("pm25:Q", title="PM2.5, мкг/м³"),
        color=alt.condition("datum['Текущий']", alt.value(BLUE), alt.value(AXIS)),
        tooltip=["Месяц:N", alt.Tooltip("pm25:Q", title="средний PM2.5", format=".1f")])
    who = alt.Chart(pd.DataFrame({"y": [WHO_PM25_24H]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return _style(alt.layer(bars, who).properties(height=240))


def show_analysis(analysis: dict, current_month: int) -> None:
    st.subheader("Почему так — Analyst Agent")
    trend = {"rising": "↗ растёт", "stable": "→ стабильно", "falling": "↘ снижается"}[analysis["trend"]]
    deviation = analysis.get("vs_seasonal_norm_pct")
    c1, c2 = st.columns(2)
    c1.metric("Тренд PM2.5 (последние дни и прогноз)", trend, border=True)
    c2.metric(f"Отклонение от нормы ({MONTHS[current_month - 1]})",
              f"{deviation:+.0f}%" if deviation is not None else "—", border=True)
    strength = {"weak": "слабый", "moderate": "умеренный", "strong": "сильный"}
    for d in analysis["drivers"]:
        st.markdown(f"- **{d['factor']}** · {strength[d['strength']]} фактор — {d['evidence']}")
    st.markdown(analysis["summary"])
    if analysis.get("monthly_pm25"):
        st.altair_chart(season_chart(analysis["monthly_pm25"], current_month), width="stretch", theme=None)
        st.caption(f"Средний PM2.5 по месяцам, {analysis.get('history_city') or ''} (2023–2026, CAMS). "
                   "Синим — текущий месяц, пунктир — суточная норма ВОЗ (15 мкг/м³).")


# ---------- Результаты прогона ----------

def show_results(run_dir: Path) -> None:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    events = read_events(run_dir / "log.jsonl")
    results = state["results"]
    view = SimpleNamespace(status=state["status"], error=state["error"], results=results,
                           plan=TaskPlan.model_validate(state["plan"]) if state["plan"] else None)

    tab_answer, tab_msgs, tab_load, tab_log = st.tabs(
        ["Ответ и прогноз", "Сообщения между агентами", "Нагрузка на агентов", "Журнал вызовов"])

    with tab_answer:
        if state["status"] == "failed":
            st.error(render_answer(view))
        elif "data_agent" not in results:
            st.info(render_answer(view))
        else:
            show_forecast(run_dir, results)
            if "analyst_agent" in results:
                latest = (results["data_agent"].get("latest_observed") or {}).get("time")
                month = pd.Timestamp(latest).month if latest else int(state["task_id"][4:6])
                st.divider()
                show_analysis(results["analyst_agent"], month)
        with st.expander("Текстовый ответ (как в консоли)"):
            st.text(render_answer(view))

    with tab_msgs:
        st.caption("Каждое сообщение — JSON по схеме из aq_agents/messages.py. Большие таблицы не передаются: "
                   "в data_package лежат только ссылки air_ref и weather_ref на данные в общем состоянии задачи.")
        for i, m in enumerate(state["messages"], 1):
            with st.expander(f"{i}. {m['sender']} → {m['receiver']}  ·  {m['type']}", expanded=i == 3):
                st.json(m["payload"])

    with tab_load:
        report = compute_load(events)
        if report["agents"]:
            st.altair_chart(load_chart(report["agents"]), width="stretch", theme=None)
            table = pd.DataFrame(report["agents"])
            table["share"] = table["share"].map("{:.0%}".format)
            table = table.rename(columns={
                "agent": "агент", "llm_calls": "вызовы LLM", "tool_calls": "вызовы инструментов",
                "total": "всего", "share": "доля"})
            st.dataframe(table, hide_index=True, width="stretch")
            if report["balanced"]:
                st.success(f"✓ Максимальная доля {report['max_share']:.0%} — в пределах лимита 40%.")
            elif len(IMPLEMENTED) < len(AGENTS):
                st.warning(f"⚠ Максимальная доля {report['max_share']:.0%}: в прототипе работают "
                           f"{len(IMPLEMENTED)} агента из {len(AGENTS)}, и в этом запросе вызовы разделились "
                           "между немногими агентами. В полной системе ожидается около 23% "
                           "(docs/05_load_balance.md).")
            else:
                st.error(f"✕ Максимальная доля {report['max_share']:.0%} превышает лимит 40%.")

    with tab_log:
        rows = [{"t, c": e["t"], "агент": e["agent"], "событие": e["kind"],
                 "детали": e.get("tool") or ", ".join(e.get("tool_calls") or []) or e.get("type")
                 or e.get("error") or e.get("reason") or "",
                 "время вызова, c": e.get("latency_s"), "токены (вход/выход)":
                 f"{e['input_tokens']}/{e['output_tokens']}" if "input_tokens" in e else ""}
                for e in events]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
        st.caption(f"Полный лог: {run_dir / 'log.jsonl'}")


def show_forecast(run_dir: Path, results: dict) -> None:
    data = results["data_agent"]
    fc = results.get("forecast_agent") or {"days": [], "rationale": None}  # для «какой воздух сейчас» прогноза нет
    latest = data.get("latest_observed") or {}

    cols = st.columns(1 + len(fc["days"]))
    cols[0].metric(f"{data['city']['name']}: сейчас, мкг/м³", latest.get("pm2_5", "—"), border=True)
    if latest.get("pm2_5") is not None:
        cols[0].caption(pm25_category(latest["pm2_5"]) + " (по часовому значению)")
    cols[0].caption(f"{latest.get('time', '')} · источник: {data['data_source']}")
    for col, d in zip(cols[1:], fc["days"]):
        col.metric(f"Прогноз на {pd.Timestamp(d['date']):%d.%m}, мкг/м³", d["final_pm25"], border=True)
        col.caption(f"{d['category']} · уверенность: {d['confidence']}")

    air_path = run_dir / "datasets" / f"{data['air_ref']}.csv"
    if air_path.exists():
        air = pd.read_csv(air_path, parse_dates=["time"], index_col="time")
        st.altair_chart(forecast_chart(air, fc["days"], latest.get("time")), width="stretch", theme=None)
        st.caption("Оранжевая линия — PM2.5 по данным CAMS (по часам): левее вертикальной линии последнего наблюдения — "
                   "прошедшие часы, правее — прогноз CAMS. Пунктир — суточная норма ВОЗ (15 мкг/м³). "
                   "Наведите курсор, чтобы увидеть значения.")

    if fc["days"]:
        table = pd.DataFrame(fc["days"])[["date", "final_pm25", "category", "source", "confidence",
                                          "ml_pm25", "ml_low", "ml_high", "cams_pm25"]]
        table.columns = ["дата", "итог", "категория", "источник", "уверенность", "ML", "ML от", "ML до", "CAMS"]
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown(f"**Почему такой источник (Forecast Agent):** {fc['rationale']}")
    st.markdown(f"**Данные (Data Agent):** {data['summary']}")


def saved_runs() -> dict[str, Path]:
    """Сохранённые прогоны: сначала примеры из репозитория, потом свои запуски (новые сверху)."""
    found: dict[str, Path] = {}
    folders = sorted((ROOT / "examples").glob("*/")) + sorted((ROOT / "runs").glob("*/"), reverse=True)
    for d in folders:
        if not (d / "log.jsonl").exists() or not (d / "state.json").exists():
            continue
        first = read_events(d / "log.jsonl")[0]
        prefix = "Пример" if d.parent.name == "examples" else d.name
        found[f"{prefix}: {first.get('query', '?')}"] = d
    return found


# ---------- Страница ----------

st.markdown(CSS, unsafe_allow_html=True)
st.title("🌫️ AirQ-KZ — многоагентная система качества воздуха")
st.caption("Прототип (Ассайнмент 2): Orchestrator → Data Agent → Forecast Agent → Analyst Agent. "
           "Пунктиром — агенты, которые появятся в Ассайнменте 4.")

settings = get_settings()
with st.sidebar:
    st.header("Режим")
    mode = st.radio("Режим", ["Новый запрос", "Повтор сохранённого прогона"], label_visibility="collapsed")
    if mode == "Новый запрос":
        st.caption(f"LLM: `{settings.llm_provider}` · `{settings.llm_model}`")
        st.info("Бесплатный Gemini: около 20 запросов в день на модель, один запуск — около 6 запросов. "
                "Для показа без расхода квоты используйте повтор сохранённого прогона.")
    else:
        runs = saved_runs()
        choice = st.selectbox("Прогон", list(runs)) if runs else None
        speed = st.slider("Скорость воспроизведения", 1, 10, 4)

if mode == "Новый запрос":
    example = st.selectbox("Пример вопроса", EXAMPLES)
    query = st.text_input("Вопрос (можно изменить)", value=example)
    start = st.button("▶ Запустить агентов", type="primary")
    target = Path(st.session_state["live_run"]) if "live_run" in st.session_state else None
else:
    target = runs[choice] if choice else None
    start = st.button("▶ Воспроизвести прогон", type="primary", disabled=target is None)

board = Board()
pipe_ph = st.empty()
feed_col, plan_col = st.columns([3, 2])
with feed_col:
    st.markdown("**Ход работы**")
    feed_ph = st.empty()
plan_ph = plan_col.empty()


def draw() -> None:
    pipe_ph.markdown(pipeline_html(board), unsafe_allow_html=True)
    feed_ph.code("\n".join(board.feed[-16:]) or "Ожидание запуска…", language=None, height=330, wrap_lines=True)
    plan_ph.markdown(plan_markdown(board.plan))


def on_event(e: dict) -> None:
    board.apply(e)
    draw()


if start and mode == "Новый запрос":
    result = run_task(query, settings=settings, verbose=False, on_event=on_event)
    st.session_state["live_run"] = str(result.run_dir)
    target = result.run_dir
elif start and target is not None:
    prev_t = 0.0
    for e in read_events(target / "log.jsonl"):
        time.sleep(max(0.12, min((e["t"] - prev_t) / speed, 1.5)))
        prev_t = e["t"]
        on_event(e)
elif target is not None:
    for e in read_events(target / "log.jsonl"):  # без анимации: сразу итоговое состояние
        board.apply(e)
    draw()
else:
    draw()

if target is not None:
    st.divider()
    show_results(target)
