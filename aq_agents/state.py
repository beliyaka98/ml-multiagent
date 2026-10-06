"""Явное состояние задачи (общая память, blackboard).

Агенты не передают друг другу большие таблицы в сообщениях: почасовые данные
лежат здесь, а в сообщениях передаются только ссылки на них (air_ref, weather_ref).
После завершения задачи состояние сохраняется в runs/<task_id>/state.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .messages import Message, TaskPlan


class TaskState:
    def __init__(self, task_id: str, run_dir: Path):
        self.task_id = task_id
        self.run_dir = run_dir
        (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
        self.status = "running"
        self.error: str | None = None
        self.plan: TaskPlan | None = None
        self.messages: list[Message] = []
        self.results: dict[str, dict] = {}  # имя агента -> его выходной payload
        self._datasets: dict[str, pd.DataFrame] = {}

    def put_dataset(self, kind: str, df: pd.DataFrame) -> str:
        ref = f"{kind}_{len(self._datasets) + 1}"
        self._datasets[ref] = df
        df.to_csv(self.run_dir / "datasets" / f"{ref}.csv")
        return ref

    def get_dataset(self, ref: str) -> pd.DataFrame:
        if ref not in self._datasets:
            raise KeyError(f"В состоянии задачи нет набора данных '{ref}'")
        return self._datasets[ref]

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)

    def save(self) -> Path:
        path = self.run_dir / "state.json"
        data = {
            "task_id": self.task_id,
            "status": self.status,
            "error": self.error,
            "plan": self.plan.model_dump() if self.plan else None,
            "results": self.results,
            "datasets": sorted(self._datasets),
            "messages": [m.model_dump() for m in self.messages],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path
