# 4. Форматы сообщений

Все сообщения — типизированные объекты Pydantic ([`aq_agents/messages.py`](../aq_agents/messages.py)),
которые сериализуются в JSON. Любое сообщение проверяется по схеме при отправке и при получении:
агент, получивший сообщение не того типа или с некорректными полями, сразу сообщает об ошибке.

## Конверт

```json
{
  "message_id": "58316d222920",
  "task_id": "20261006-150014-0683b7",
  "sender": "data_agent",
  "receiver": "forecast_agent",
  "type": "data_package",
  "created_at": "2026-10-06T10:00:18+00:00",
  "payload": { "...": "одна из моделей ниже" }
}
```

## Типы сообщений

| type | Отправитель → получатель | Payload | Статус |
|---|---|---|---|
| `user_query` | user → orchestrator | `UserQuery` | ✅ |
| `task_plan` | orchestrator → первый агент плана | `TaskPlan` | ✅ |
| `data_package` | data_agent → forecast_agent | `DataPackage` | ✅ |
| `forecast_result` | forecast_agent → analyst_agent (в прототипе → user) | `ForecastResult` | ✅ |
| `analysis_report` | analyst_agent → health_agent (в прототипе → user) | `AnalysisReport` | ✅ |
| `health_advice` | health_agent → reporter_agent | `HealthAdvice` | A4 |
| `final_report` | reporter_agent → critic_agent | `FinalReport` | A4 |
| `critic_verdict` | critic_agent → orchestrator | `CriticVerdict` | A4 |
| `error` | любой агент → orchestrator | `ErrorPayload` | A4 |

## Примеры из реального прогона прототипа

Модель — Gemini 3.5 Flash, данные Open-Meteo, 06.10.2026, лог [`examples/run_health_astana/`](../examples/run_health_astana/).
Запрос: *«Можно ли послезавтра гулять с ребёнком в Нур-Султане?»* (Orchestrator сам привёл
название города к «Астана»).

### `task_plan` (Orchestrator → Data Agent)

```json
{
  "intent": "health",
  "city": "Астана",
  "days_ahead": 2,
  "user_group": "children",
  "steps": ["data_agent", "forecast_agent", "health_agent", "reporter_agent", "critic_agent"],
  "rationale": "Запрос о возможности прогулки с ребенком (группа children) послезавтра (days_ahead: 2) в Астане. Требуется собрать текущие данные, построить прогноз на 2 дня вперед, сформировать рекомендации по здоровью для детей, подготовить финальный отчет и верифицировать его."
}
```

### `data_package` (Data Agent → Forecast Agent)

Большие таблицы не передаются: `air_ref` и `weather_ref` — ссылки на данные в общем состоянии задачи.

```json
{
  "status": "ok",
  "city": {"name": "Астана", "lat": 51.1801, "lon": 71.446, "timezone": "Asia/Almaty"},
  "air_ref": "air_1",
  "weather_ref": "weather_2",
  "period_start": "2026-09-28 00:00:00",
  "period_end": "2026-10-09 23:00:00",
  "air_missing_pct": 0.0,
  "weather_missing_pct": 0.0,
  "latest_observed": {"time": "2026-10-06 15:00:00", "pm2_5": 2.6, "pm10": 2.8},
  "data_source": "network",
  "summary": "Успешно загружены почасовые данные о качестве воздуха и погоде для города Астана за период с 28 сентября по 9 октября 2026 года (пропуски отсутствуют). Последнее наблюдаемое значение PM2.5 на 15:00 6 октября составляет 2.6 мкг/м³, среднее за сегодня — 5.4 мкг/м³.",
  "issues": []
}
```

### `forecast_result` (Forecast Agent → следующий агент)

```json
{
  "status": "ok",
  "city": "Астана",
  "days": [
    {"date": "2026-10-06", "ml_pm25": 5.1, "ml_low": 2.5, "ml_high": 7.4, "cams_pm25": 5.0,
     "final_pm25": 5.0, "source": "blend", "category": "хорошо", "confidence": "high"},
    {"date": "2026-10-07", "ml_pm25": 3.9, "ml_low": 1.8, "ml_high": 6.1, "cams_pm25": 4.4,
     "final_pm25": 4.2, "source": "blend", "category": "хорошо", "confidence": "high"},
    {"date": "2026-10-08", "ml_pm25": 6.0, "ml_low": 3.0, "ml_high": 9.1, "cams_pm25": 4.5,
     "final_pm25": 5.2, "source": "blend", "category": "хорошо", "confidence": "medium"}
  ],
  "model": {
    "name": "HistGradientBoostingRegressor(log1p PM2.5), горизонт 1–4 дня",
    "trained_until": "2026-09-29",
    "test_mae_by_horizon": {"1": 3.25, "2": 3.51, "3": 3.56, "4": 3.6},
    "baseline_mae_by_horizon": {"1": 3.63, "2": 4.64, "3": 4.94, "4": 5.11}
  },
  "rationale": "Для прогноза на 6–8 октября использовано усреднение (blend) нашей ML-модели и физического прогноза CAMS, так как оба источника дают очень близкие и правдоподобные низкие значения. Уверенность высокая на первые два дня из-за расхождения менее 12%, и средняя на третий день из-за расхождения в 28%."
}
```

> Тексты `summary` и `rationale` пишет LLM, поэтому в каждом прогоне формулировки свои.
> Числа (PM2.5, прогнозы, итог, категория) считает код.

## Схемы агентов Ассайнмента 4 (зафиксированы заранее)

```jsonc
// analysis_report
{"drivers": [{"factor": "низкий пограничный слой (инверсия)", "evidence": "ночной минимум 60 м при норме 300 м",
              "strength": "strong"}],
 "trend": "rising", "vs_seasonal_norm_pct": 35.0, "summary": "..."}

// health_advice
{"category": "вредно для чувствительных групп", "risk_level": "high", "who_guideline_exceeded": true,
 "groups_at_risk": ["children", "respiratory"], "recommendations": ["...", "..."]}

// final_report
{"answer": "...", "key_numbers": {"pm25_2026-10-07": 42.1}, "sources": ["CAMS", "ML-модель"],
 "disclaimer": "Не является медицинской рекомендацией"}

// critic_verdict
{"approved": false,
 "issues": [{"severity": "major", "field": "answer", "message": "в тексте 24.1 мкг/м³, в прогнозе 42.1"}],
 "revise_agent": "reporter_agent"}

// error
{"agent": "data_agent", "error_type": "ToolError", "message": "Сервис недоступен после 3 попыток", "retriable": true}
```

## Внутренний формат «решения LLM»

LLM каждого агента возвращает только свою часть решения. Выходное сообщение собирает код
(`finalize()`), добавляя факты из инструментов:

| Агент | LLM возвращает | Код добавляет |
|---|---|---|
| Orchestrator | весь `TaskPlan` | нормализацию шагов, город по умолчанию |
| Data Agent | `{status, summary, issues}` | город, ссылки на данные, пропуски, последнее наблюдение, источник |
| Forecast Agent | `{days: [{date, source, confidence}], rationale}` | значения ML и CAMS, итог, категорию, метрики модели |
