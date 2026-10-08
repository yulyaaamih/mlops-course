"""Стадия validate: проверка data/processed/train.csv перед обучением.

Запуск: python -m src.data.validate

Пороги — в params.yaml (секция validate). Результат каждой проверки
печатается в лог и пишется в reports/validation.json. Если хоть одна
проверка упала, скрипт выходит с кодом 1 и dvc repro не запускает train.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import TARGET, load_params, resolve
from src.logging_setup import setup_logging
from src.train import load_data, save_metrics

log = setup_logging()

REPORT_PATH = "reports/validation.json"

CHECKS = {
    "строк не меньше минимума": lambda df, p: len(df) >= p["min_rows"],
    "доля пропусков в норме": lambda df, p: df.isna().mean().max() <= p["max_missing_share"],
    "доля оттока осмысленна": lambda df, p: p["target_rate"][0] < df[TARGET].mean() < p["target_rate"][1],
    "нет дублей по клиенту": lambda df, p: not df["customer_id"].duplicated().any(),
    "стаж в допустимом диапазоне": lambda df, p: df["tenure_months"].between(0, 200).all(),
}


def run_checks(df: pd.DataFrame, params: dict) -> dict[str, list[str]]:
    report = {"passed": [], "failed": []}
    for name, check in CHECKS.items():
        ok = bool(check(df, params))
        if ok:
            log.info("%s — OK", name)
        else:
            log.error("%s — FAIL", name)
        report["passed" if ok else "failed"].append(name)
    return report


def main() -> None:
    params = load_params()

    # 1. Чтение train.csv после prepare
    train_df = load_data(resolve(params["data"]["processed_dir"]) / "train.csv")

    # 2. Проверки из CHECKS с порогами из params.yaml
    report = run_checks(train_df, params["validate"])

    # 3. Отчёт пишем до выхода: файл нужен, даже если проверки упали
    save_metrics(report, resolve(REPORT_PATH))

    # 4. Хоть одна проверка упала — код 1, dvc repro не запустит train
    if report["failed"]:
        log.error("валидация не пройдена: %s", report["failed"])
        sys.exit(1)


if __name__ == "__main__":
    main()
