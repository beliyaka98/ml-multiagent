"""Настройки системы.

Все значения берутся из переменных окружения (файл .env). Ключи API в коде не хранятся:
SDK провайдеров сами читают ANTHROPIC_API_KEY / OPENAI_API_KEY из окружения.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str          # anthropic | openai
    llm_model: str
    llm_effort: str | None     # low | medium | high (только для Claude)
    openai_base_url: str | None
    max_agent_steps: int       # лимит вызовов LLM внутри одного агента
    max_total_steps: int       # лимит всех шагов (LLM + инструменты) на задачу
    task_timeout_s: float      # лимит времени на задачу
    http_timeout_s: float
    http_retries: int
    cache_ttl_s: int
    runs_dir: Path
    cache_path: Path
    model_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "anthropic").strip().lower(),
            llm_model=os.getenv("LLM_MODEL", "claude-opus-5-5").strip(),
            llm_effort=os.getenv("LLM_EFFORT", "").strip() or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "6")),
            max_total_steps=int(os.getenv("MAX_TOTAL_STEPS", "40")),
            task_timeout_s=float(os.getenv("TASK_TIMEOUT_S", "180")),
            http_timeout_s=float(os.getenv("HTTP_TIMEOUT_S", "20")),
            http_retries=int(os.getenv("HTTP_RETRIES", "3")),
            cache_ttl_s=int(os.getenv("CACHE_TTL_S", "3600")),
            runs_dir=ROOT / os.getenv("RUNS_DIR", "runs"),
            cache_path=ROOT / "data" / "cache.sqlite",
            model_path=ROOT / "models" / "pm25_forecast.joblib",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
