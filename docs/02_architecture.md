# 2. Архитектура

## Схема взаимодействия: оркестратор + конвейер по плану + общая память

- **Orchestrator** (LLM) только планирует: определяет тип запроса, город, горизонт, группу
  пользователей и упорядоченный список агентов. Предметной работы не выполняет, инструментов нет.
- **Диспетчер** (обычный код, не LLM) выполняет план: передаёт сообщение очередному агенту
  и адресует его ответ следующему агенту плана. Это конвейер, собранный под конкретный запрос.
- **Общая память (TaskState)** хранит план, все сообщения, почасовые наборы данных и результаты
  агентов. Большие таблицы не передаются в сообщениях: передаются ссылки (`air_ref`, `weather_ref`).
- **Обратная связь от критика:** вердикт Critic Agent возвращается Orchestrator'у, и тот решает,
  кого отправить на доработку. Доработок не больше двух, это защита от зацикливания.

## Схема агентов, связей и инструментов

```mermaid
flowchart TB
    U([Пользователь]) -->|user_query| O

    subgraph CTRL[Управление]
        O["Orchestrator<br/>планирует, без инструментов"]
        D{{"Диспетчер (код)<br/>доставка сообщений по плану"}}
    end

    O -->|task_plan| D
    D --> DA

    subgraph WORK[Агенты-исполнители]
        DA["Data Agent<br/>сбор и проверка данных"]
        FA["Forecast Agent<br/>прогноз PM2.5"]
        AA["Analyst Agent<br/>причины загрязнения"]
        HA["Health Agent<br/>рекомендации"]
        RA["Reporter Agent<br/>итоговый ответ"]
        CA["Critic Agent<br/>проверка ответа"]
        DA -->|data_package| FA
        FA -->|forecast_result| AA
        AA -->|analysis_report| HA
        HA -->|health_advice| RA
        RA -->|final_report| CA
    end

    CA -->|"critic_verdict"| O
    O -.->|"доработка (не более 2 раз)"| RA
    CA -->|одобренный ответ| U

    DA --- T1[("Open-Meteo Geocoding API")]
    DA --- T2[("Open-Meteo Air Quality API (CAMS)")]
    DA --- T3[("Open-Meteo Weather API")]
    DA -.- T4[("OpenAQ: наземные станции")]
    FA --- T5[["ML-модель PM2.5 (joblib)"]]
    AA --- T6[["Корреляции, застой воздуха, сезонная норма (pandas + история 2023–2026)"]]
    HA -.- T7[["Нормы ВОЗ и шкала AQI (локальная база)"]]
    CA -.- T8[["Проверка чисел в ответе"]]

    S[("TaskState — общая память<br/>runs/&lt;task_id&gt;/state.json")]
    L[("Логи — runs/&lt;task_id&gt;/log.jsonl")]
    C[("SQLite-кэш ответов API")]
    WORK <-.-> S
    T1 & T2 & T3 --- C

    classDef planned stroke-dasharray: 5 5
    class HA,RA,CA,T4,T7,T8 planned
```

Пунктир — агенты и инструменты Ассайнмента 4. Сплошная линия — уже реализовано в прототипе.

## Типовой сценарий (последовательность сообщений)

Запрос: *«Можно ли завтра утром бегать в Алматы?»*

```mermaid
sequenceDiagram
    autonumber
    actor U as Пользователь
    participant O as Orchestrator
    participant DA as Data Agent
    participant FA as Forecast Agent
    participant AA as Analyst Agent
    participant HA as Health Agent
    participant RA as Reporter Agent
    participant CA as Critic Agent

    U->>O: user_query
    O->>DA: task_plan (intent=health, city=Алматы, days_ahead=1, group=athletes)
    DA->>DA: geocode_city → fetch_air_quality ∥ fetch_weather
    DA->>FA: data_package (air_ref, weather_ref, качество данных)
    FA->>FA: run_ml_forecast ∥ get_cams_forecast → выбор источника
    FA->>AA: forecast_result
    AA->>HA: analysis_report (факторы: ветер, инверсия…)
    HA->>RA: health_advice (риск, рекомендации для бегунов)
    RA->>CA: final_report
    CA->>O: critic_verdict
    alt ответ одобрен
        O-->>U: ответ
    else найдены ошибки (не более 2 раз)
        O->>RA: доработать по замечаниям
    end
```

В прототипе (Ассайнмент 2) работают шаги 1–6: Orchestrator → Data Agent → Forecast Agent → Analyst Agent.
Агенты, которых ещё нет, диспетчер пропускает и пишет в лог событие `agent_skipped`.

## Состояние задачи

| Что хранится | Где | Кто пишет |
|---|---|---|
| План задачи (`TaskPlan`) | `TaskState.plan` | Orchestrator |
| Все сообщения (конверт + payload) | `TaskState.messages` | Диспетчер |
| Почасовые данные воздуха и погоды | `TaskState` + `runs/<id>/datasets/*.csv` | Инструменты Data Agent |
| Результат каждого агента | `TaskState.results` | Диспетчер |
| Итог задачи | `runs/<id>/state.json` | В конце задачи, в том числе при ошибке |
| Кэш ответов API | `data/cache.sqlite` | HTTP-слой |

## Надёжность и безопасность

| Механизм | Где реализован | Значение по умолчанию |
|---|---|---|
| Лимит вызовов LLM внутри агента | `runtime.BaseAgent.run` | `MAX_AGENT_STEPS=6` |
| Общий лимит шагов (LLM + инструменты) на задачу | `runtime.Budget` | `MAX_TOTAL_STEPS=40` |
| Лимит времени на задачу | `runtime.Budget` | `TASK_TIMEOUT_S=180` |
| Защита от зацикливания: третий одинаковый вызов инструмента — остановка | `runtime._run_tool` | 2 повтора |
| Ответ модели не прошёл проверку схемы или правил — замечание и повтор | `runtime.BaseAgent.run` | до 2 исправлений |
| Повторы HTTP с экспоненциальной паузой (сеть, 429, 5xx) | `tools/http.get_json` | 3 попытки |
| Кэш API и переход на устаревший кэш при недоступности | `tools/http.get_json` | TTL 1 час |
| Повторы запросов к LLM (429, 5xx, сеть) | SDK провайдера | 3 попытки |
| Ошибка инструмента не роняет систему: возвращается агенту с `is_error` | `runtime._run_tool` | — |
| Понятное сообщение пользователю при сбое | `pipeline.run_task` → `render_answer` | — |
| Логирование всех вызовов LLM и инструментов | `logger.RunLogger` → `log.jsonl` | — |
| Ключи API только в `.env` (в `.gitignore`), в коде ключей нет | `config.py` | — |
| Числа считает код, а не LLM (категории, итоговый прогноз, пропуски) | `finalize()` агентов | — |

## Технологии

- Python 3.11+ и собственный минимальный «движок» агентов без фреймворков: каждую строку
  можно объяснить на защите.
- LLM: любая модель с вызовом инструментов, выбирается в `.env`: Claude (`claude-opus-5-5`,
  официальный SDK `anthropic`) или OpenAI-совместимые API (Gemini, OpenAI, Groq, локальная Ollama).
  Прототип проверен на Gemini 3.5 Flash (бесплатный тариф).
- Данные: Open-Meteo (бесплатно, без ключа). ML: scikit-learn. Валидация сообщений: Pydantic.
- Интерфейс: веб-интерфейс на Streamlit (`app.py`: схема агентов в реальном времени, графики, сообщения,
  нагрузка; режим повтора сохранённого прогона) и CLI (`main.py`).
