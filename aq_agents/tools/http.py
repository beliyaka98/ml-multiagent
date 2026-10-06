"""HTTP-запросы к внешним API с повторными попытками и кэшем в SQLite.

- до N попыток с экспоненциальной паузой при сетевых ошибках, 429 и 5xx;
- успешные ответы кэшируются (TTL), чтобы не перегружать API;
- если API недоступен, возвращаем устаревший кэш (stale_cache) и честно помечаем это.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from urllib.parse import urlencode, urlparse

import requests

from ..config import Settings, get_settings


class ToolError(Exception):
    """Ошибка инструмента. Текст уходит агенту как результат вызова с пометкой ошибки."""


class _Retryable(Exception):
    pass


def _cache_key(url: str, params: dict) -> str:
    query = urlencode(sorted(params.items()), doseq=True)
    return hashlib.sha256(f"{url}?{query}".encode()).hexdigest()


def _cache_get(cfg: Settings, key: str, max_age_s: float | None) -> dict | None:
    if not cfg.cache_path.exists():
        return None
    with sqlite3.connect(cfg.cache_path) as db:
        row = db.execute("SELECT created, body FROM cache WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    created, body = row
    if max_age_s is not None and time.time() - created > max_age_s:
        return None
    return json.loads(body)


def _cache_put(cfg: Settings, key: str, url: str, data: dict) -> None:
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cfg.cache_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, url TEXT, created REAL, body TEXT)")
        db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)", (key, url, time.time(), json.dumps(data)))


def get_json(url: str, params: dict, *, ttl_s: float | None = None,
             cfg: Settings | None = None) -> tuple[dict, str, int]:
    """Возвращает (данные, источник: network|cache|stale_cache, число сетевых попыток)."""
    cfg = cfg or get_settings()
    key = _cache_key(url, params)
    ttl = cfg.cache_ttl_s if ttl_s is None else ttl_s

    cached = _cache_get(cfg, key, max_age_s=ttl)
    if cached is not None:
        return cached, "cache", 0

    last_error: Exception | None = None
    for attempt in range(1, cfg.http_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=cfg.http_timeout_s)
            if r.status_code == 429 or r.status_code >= 500:
                raise _Retryable(f"HTTP {r.status_code}")
            data = r.json()
            if r.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
                reason = data.get("reason") if isinstance(data, dict) else r.status_code
                raise ToolError(f"API отклонил запрос: {reason}")  # повтор не поможет
            _cache_put(cfg, key, url, data)
            return data, "network", attempt
        except (requests.RequestException, _Retryable) as e:
            last_error = e
            if attempt < cfg.http_retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    stale = _cache_get(cfg, key, max_age_s=None)
    if stale is not None:
        return stale, "stale_cache", cfg.http_retries
    host = urlparse(url).netloc
    raise ToolError(f"Сервис {host} недоступен после {cfg.http_retries} попыток: {last_error}")
