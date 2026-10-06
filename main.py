"""CLI: python main.py "Какой будет воздух завтра в Алматы?" """
from __future__ import annotations

import argparse
import json
import sys

from aq_agents.pipeline import run_task


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Многоагентная система: качество воздуха в городах Казахстана")
    parser.add_argument("query", nargs="?", help="вопрос на естественном языке")
    parser.add_argument("-q", "--quiet", action="store_true", help="не печатать ход работы агентов")
    parser.add_argument("--json", action="store_true", help="напечатать результаты агентов в JSON")
    args = parser.parse_args()

    query = args.query or input("Ваш вопрос: ").strip()
    result = run_task(query, verbose=not args.quiet)

    print("\n" + "=" * 70)
    print(result.answer)
    print("=" * 70)
    if args.json:
        print(json.dumps(result.state.results, ensure_ascii=False, indent=2))

    load = result.load
    print(f"\nРаспределение нагрузки (всего вызовов: {load['total_calls']}, лимит {load['limit']:.0%} на агента):")
    for row in load["agents"]:
        print(f"  {row['agent']:<15} LLM {row['llm_calls']:>2} + инструменты {row['tool_calls']:>2} "
              f"= {row['total']:>2}  ({row['share']:.0%})")
    print(f"\nЛоги и состояние: {result.run_dir}")
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
