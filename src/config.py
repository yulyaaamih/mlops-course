"""Загрузка params.yaml и доступ к настройкам.

Один модуль-точка входа для конфигурации: скрипты не читают YAML сами
и не хранят пути внутри себя. Это то, ради чего на занятии 3 всё выносилось
из кода — здесь видно результат.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "params.yaml"))


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Читает params.yaml. Кэш не нужен: файл маленький, а неявный кэш путает."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Путь из конфига -> абсолютный путь от корня проекта.

    Без этого проект работает только если запускать его из корня —
    классическая причина 'у меня локально работало'.
    """
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def feature_columns(params: dict[str, Any]) -> list:
    f = params["features"]
    return list(f["numeric"]) + list(f["categorical"]) + list(f["binary"])


TARGET = "churn"
