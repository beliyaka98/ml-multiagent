"""Распределение нагрузки между агентами по логу прогона.

Запуск: python scripts/load_report.py runs/<task_id>/log.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aq_agents.logger import load_report_from_file  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    report = load_report_from_file(Path(sys.argv[1]))
    print(f"{'агент':<16}{'LLM':>5}{'tools':>7}{'всего':>7}{'доля':>8}")
    for r in report["agents"]:
        print(f"{r['agent']:<16}{r['llm_calls']:>5}{r['tool_calls']:>7}{r['total']:>7}{r['share']:>8.0%}")
    verdict = "OK" if report["balanced"] else "ПРЕВЫШЕН"
    print(f"\nВсего вызовов: {report['total_calls']}. Максимальная доля: {report['max_share']:.0%} "
          f"(лимит {report['limit']:.0%}) — {verdict}")


if __name__ == "__main__":
    main()
