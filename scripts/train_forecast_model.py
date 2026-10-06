"""Обучение ML-модели прогноза среднесуточного PM2.5 (горизонт 1–4 дня).

Данные: Open-Meteo (CAMS — качество воздуха, ERA5 — погода) для 8 городов
Казахстана с 2023-01-01. Модель: HistGradientBoostingRegressor на log1p(PM2.5).
Оценка: временное разделение (тест — последний год, включая зиму),
сравнение с наивным прогнозом «завтра как сегодня» (persistence).

Запуск:  python scripts/train_forecast_model.py            (использует data/training_daily.csv, если есть)
         python scripts/train_forecast_model.py --refresh  (заново скачать данные)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aq_agents.config import ROOT  # noqa: E402
from aq_agents.tools import openmeteo  # noqa: E402
from aq_agents.tools.features import FEATURES, HORIZONS, build_frame, to_daily  # noqa: E402
from aq_agents.tools.history import TRAINING_CITIES  # noqa: E402

CITIES = TRAINING_CITIES
START = date(2023, 1, 1)
CALIB_FROM = pd.Timestamp("2025-04-01")  # калибровка интервала: апрель–сентябрь 2025
TEST_FROM = pd.Timestamp("2025-10-01")   # тест: последний год, включая зиму 2025/26
DATA_PATH = ROOT / "data" / "training_daily.csv"
MODEL_PATH = ROOT / "models" / "pm25_forecast.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"


def year_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        stop = min(date(cur.year, 12, 31), end)
        yield cur.isoformat(), stop.isoformat()
        cur = stop + timedelta(days=1)


def download(end: date) -> pd.DataFrame:
    parts = []
    for city, (lat, lon) in CITIES.items():
        air, wx = [], []
        for s, e in year_chunks(START, end):
            print(f"  {city}: {s} … {e}")
            air.append(openmeteo.air_quality_history(lat, lon, s, e))
            wx.append(openmeteo.weather_history(lat, lon, s, e))
        daily = to_daily(pd.concat(air), pd.concat(wx))
        daily["city"] = city
        parts.append(daily)
    df = pd.concat(parts)
    df.index.name = "date"
    return df


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="заново скачать данные")
    args = ap.parse_args()

    if args.refresh or not DATA_PATH.exists():
        end = date.today() - timedelta(days=7)  # архив погоды отстаёт на несколько дней
        print(f"Скачиваю данные {START} … {end} для {len(CITIES)} городов")
        daily_all = download(end)
        DATA_PATH.parent.mkdir(exist_ok=True)
        daily_all.to_csv(DATA_PATH)
    daily_all = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

    frames = [build_frame(g.drop(columns="city")).assign(city=city) for city, g in daily_all.groupby("city")]
    data = pd.concat(frames, ignore_index=True).dropna(subset=FEATURES + ["target"])
    # train → calib (подбор ширины интервала) → test (честная оценка, сюда модель «не заглядывала»)
    train = data[data["target_date"] < CALIB_FROM]
    calib = data[(data["target_date"] >= CALIB_FROM) & (data["target_date"] < TEST_FROM)]
    test = data[data["target_date"] >= TEST_FROM]
    print(f"Строк: train={len(train)}, calib={len(calib)}, test={len(test)}")

    model = make_model().fit(train[FEATURES], np.log1p(train["target"]))
    calib = calib.assign(pred=np.expm1(model.predict(calib[FEATURES])))
    test = test.assign(pred=np.expm1(model.predict(test[FEATURES])))

    metrics, baseline, intervals, coverage = {}, {}, {}, {}
    print(f"\n{'h':>2} {'MAE модель':>11} {'MAE baseline':>13} {'зима: модель':>13} {'зима: baseline':>15} {'покрытие 80%':>13}")
    for h in HORIZONS:
        t = test[test["horizon"] == h]
        tw = t[t["target_date"].dt.month.isin([11, 12, 1, 2, 3])]
        mae = mean_absolute_error(t["target"], t["pred"])
        base = mean_absolute_error(t["target"], t["pm25_d0"])
        # 80%-интервал: квантили остатков (в лог-пространстве) на калибровочном периоде,
        # покрытие проверяется на тесте
        c = calib[calib["horizon"] == h]
        q10, q90 = np.quantile(np.log1p(c["target"]) - np.log1p(c["pred"]), [0.10, 0.90])
        inside = ((np.log1p(t["target"]) >= np.log1p(t["pred"]) + q10) &
                  (np.log1p(t["target"]) <= np.log1p(t["pred"]) + q90)).mean()
        coverage[str(h)] = round(float(inside), 3)
        metrics[str(h)] = round(mae, 2)
        baseline[str(h)] = round(base, 2)
        intervals[str(h)] = [round(float(q10), 4), round(float(q90), 4)]
        print(f"{h:>2} {mae:>11.2f} {base:>13.2f} "
              f"{mean_absolute_error(tw['target'], tw['pred']):>13.2f} "
              f"{mean_absolute_error(tw['target'], tw['pm25_d0']):>15.2f} {inside:>13.0%}")

    # Финальная модель обучается на всех данных; метрики выше — честная оценка на отложенном годе.
    final = make_model().fit(data[FEATURES], np.log1p(data["target"]))
    card = {
        "name": "HistGradientBoostingRegressor(log1p PM2.5), горизонт 1–4 дня",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cities": list(CITIES),
        "train_period": [str(data["base_date"].min().date()), str(data["target_date"].max().date())],
        "trained_until": str(data["target_date"].max().date()),
        "test_from": str(TEST_FROM.date()),
        "n_rows": int(len(data)),
        "features": FEATURES,
        "test_mae_by_horizon": metrics,
        "baseline_mae_by_horizon": baseline,
        "interval_log_residuals": intervals,
        "interval_coverage_on_test": coverage,
        "notes": "Целевая переменная — данные CAMS (Open-Meteo), а не наземные станции.",
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "card": card}, MODEL_PATH)
    CARD_PATH.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nМодель сохранена: {MODEL_PATH.relative_to(ROOT)}; паспорт: {CARD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
