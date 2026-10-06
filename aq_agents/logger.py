"""Логирование всех вызовов агентов и инструментов.

Каждое событие пишется отдельной JSON-строкой в runs/<task_id>/log.jsonl.
По этим логам считается распределение нагрузки между агентами (требование ≤ 40%).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

LOAD_LIMIT = 0.40  # ни один агент не должен делать больше 40% вызовов


class RunLogger:
    def __init__(self, path: Path, verbose: bool = True, on_event: Callable[[dict], None] | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.on_event = on_event  # например, веб-интерфейс, который показывает ход работы вживую
        self._t0 = time.monotonic()
        self.events: list[dict] = []

    def event(self, kind: str, agent: str, **data) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t": round(time.monotonic() - self._t0, 3),
            "kind": kind,
            "agent": agent,
            **data,
        }
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.verbose:
            print(format_event(record), file=sys.stderr)
        if self.on_event:
            self.on_event(record)

    def load_report(self) -> dict:
        return compute_load(self.events)


def compute_load(events: list[dict], limit: float = LOAD_LIMIT) -> dict:
    """Доля вызовов (LLM + инструменты) каждого агента."""
    counts: dict[str, dict[str, int]] = {}
    for e in events:
        if e["kind"] not in ("llm_call", "tool_call"):
            continue
        c = counts.setdefault(e["agent"], {"llm_calls": 0, "tool_calls": 0})
        c["llm_calls" if e["kind"] == "llm_call" else "tool_calls"] += 1
    total = sum(c["llm_calls"] + c["tool_calls"] for c in counts.values())
    rows = []
    for agent, c in counts.items():
        n = c["llm_calls"] + c["tool_calls"]
        rows.append({"agent": agent, **c, "total": n, "share": round(n / total, 3) if total else 0.0})
    rows.sort(key=lambda r: r["total"], reverse=True)
    max_share = rows[0]["share"] if rows else 0.0
    return {"total_calls": total, "agents": rows, "max_share": max_share, "limit": limit,
            "balanced": max_share <= limit}


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report_from_file(path: Path) -> dict:
    return compute_load(read_events(path))


def format_event(r: dict) -> str:
    """Короткая строка события для консоли и веб-интерфейса."""
    kind, agent = r["kind"], r["agent"]
    if kind == "llm_call":
        tools = ", ".join(r.get("tool_calls") or []) or "финальный ответ"
        return f"[{r['t']:6.1f}s] {agent:<15} LLM #{r.get('step')} → {tools} ({r.get('latency_s')}s)"
    if kind == "tool_call":
        mark = "ok" if r.get("ok") else "ОШИБКА"
        return f"[{r['t']:6.1f}s] {agent:<15}   tool {r.get('tool')} [{mark}] ({r.get('latency_s')}s)"
    if kind == "message":
        return f"[{r['t']:6.1f}s] {agent:<15} ✉ {r.get('type')} → {r.get('receiver')}"
    extra = r.get("error") or r.get("reason") or ""
    return f"[{r['t']:6.1f}s] {agent:<15} {kind} {extra}".rstrip()
