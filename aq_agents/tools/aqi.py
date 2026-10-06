"""Категории качества воздуха по среднесуточному PM2.5.

Границы — шкала AQI US EPA (редакция 2024 года). Порог ВОЗ (2021) для
среднесуточного PM2.5 — 15 мкг/м³.
"""
from __future__ import annotations

import math

WHO_PM25_24H = 15.0

PM25_CATEGORIES = [
    (9.0, "хорошо"),
    (35.4, "умеренно"),
    (55.4, "вредно для чувствительных групп"),
    (125.4, "вредно"),
    (225.4, "очень вредно"),
    (math.inf, "опасно"),
]


def pm25_category(value: float) -> str:
    for upper, label in PM25_CATEGORIES:
        if value <= upper:
            return label
    return PM25_CATEGORIES[-1][1]
