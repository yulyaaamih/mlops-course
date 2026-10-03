"""Проверка окружения перед первым занятием: python -m src.smoke_check"""
from __future__ import annotations

import importlib
import sys

REQUIRED = ["numpy", "pandas", "sklearn", "yaml", "joblib", "fastapi", "pydantic", "pytest"]
OPTIONAL = ["mlflow", "dvc", "evidently", "prometheus_client", "prefect"]


def version(mod: str) -> str:
    m = importlib.import_module(mod)
    return getattr(m, "__version__", "?")


def main() -> int:
    print(f"python  : {sys.version.split()[0]}")
    missing = []
    for mod in REQUIRED:
        try:
            print(f"{mod:16}: {version(mod)}")
        except Exception as exc:  # noqa: BLE001
            missing.append(mod)
            print(f"{mod:16}: НЕТ ({exc})")
    for mod in OPTIONAL:
        try:
            print(f"{mod:16}: {version(mod)} (опционально)")
        except Exception:  # noqa: BLE001
            print(f"{mod:16}: нет (понадобится позже)")

    recommended = (3, 11)
    if sys.version_info < recommended:
        print("\nвнимание: курс рассчитан на Python 3.11, у вас "
              f"{sys.version_info.major}.{sys.version_info.minor} — часть шагов может отличаться")
    if missing:
        print(f"\nenvironment: FAIL — не установлены: {', '.join(missing)}")
        print("исправление: pip install -r requirements.txt -r requirements-dev.txt")
        return 1
    print("\nenvironment: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
