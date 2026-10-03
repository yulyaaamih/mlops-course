"""Стадия prepare: сырой CSV -> train/val/test.

TODO (занятие 1):
  1. прочитать data/raw/churn.csv;
  2. обработать пропуски в total_charges осмысленно (не dropna!);
  3. разбить на train/val/test со stratify по churn и random_state из params;
  4. сохранить три CSV в data/processed/.

Проверка: два запуска подряд должны дать одинаковые файлы.
"""
from __future__ import annotations

from src.config import load_params
from src.logging_setup import setup_logging

log = setup_logging()


def main() -> None:
    params = load_params()
    d = params["data"]
    seed = params["seed"]

    # TODO: Студент заполняет этот код
    # Шаги:
    # 1. Прочитать raw данные с помощью pd.read_csv()
    # 2. Обработать пропуски в total_charges (заполнить осмысленно, не dropna!)
    # 3. Разбить на train/val/test со stratify по churn
    # 4. Сохранить три CSV в data/processed/
    #
    # Подсказка: используй train_test_split из sklearn с random_state=seed
    # Проверка: два запуска подряд должны дать одинаковые файлы

    pass


if __name__ == "__main__":
    main()
