"""Стадия train.

TODO (занятие 1): перенести сюда логику из notebooks/baseline_notebook.py,
исправив всё, что вы в ней нашли.

Обязательно:
  * никаких абсолютных путей — только src.config.resolve();
  * никаких магических чисел — только params.yaml;
  * зафиксированный seed;
  * модель сохраняется в models/model.joblib вместе с препроцессором;
  * метрики пишутся в reports/train_metrics.json.

Запуск: python -m src.train
"""
from __future__ import annotations

from src.config import load_params
from src.logging_setup import setup_logging

log = setup_logging()


def main() -> None:
    params = load_params()

    # TODO: Студент заполняет этот код
    #
    # Шаги:
    # 1. Загрузить данные train.csv и val.csv из data/processed/
    # 2. Подготовить X, y для обучения и валидации
    # 3. Создать Pipeline(препроцессор + RandomForestClassifier)
    # 4. Обучить модель на train
    # 5. Предсказать на val
    # 6. Посчитать метрики (roc_auc, f1, pr_auc)
    # 7. Сохранить модель в models/model.joblib
    # 8. Сохранить метрики в reports/train_metrics.json
    #
    # Подсказка: параметры модели берутся из params["model"]
    # Проверка: два запуска подряд должны дать одинаковые метрики

    pass


if __name__ == "__main__":
    main()
