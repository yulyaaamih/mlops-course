"""evaluate: оценка обученной модели на тестовых данных.

Запуск: python -m src.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

from src.config import load_params, resolve
from src.logging_setup import setup_logging
from src.train import MODEL_PATH, compute_metrics, load_data, save_metrics, split_xy

log = setup_logging()

EVAL_METRICS_PATH = "reports/eval_metrics.json"
ROC_PATH = "reports/roc.json"


def predict(model: Pipeline, X: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)[:, 1]
    return proba, (proba >= threshold).astype(int)


def save_roc_curve(y_true: pd.Series, proba: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"roc": points}, f, indent=2)
    log.info("ROC-кривая сохранена в %s", path)


def is_quality_ok(metrics: dict[str, float], min_roc_auc: float) -> bool:
    if metrics["roc_auc"] < min_roc_auc:
        log.error("ROC-AUC %.4f ниже порога %.4f", metrics["roc_auc"], min_roc_auc)
        return False
    log.info("ROC-AUC %.4f не ниже порога %.4f", metrics["roc_auc"], min_roc_auc)
    return True


def main() -> None:
    params = load_params()
    cfg = params["evaluate"]

    # 1. Загрузка обученной модели (Pipeline вместе с препроцессором)
    model = joblib.load(resolve(MODEL_PATH))

    # 2. Загрузка test и разделение на признаки X и таргет y
    test_df = load_data(resolve(params["data"]["processed_dir"]) / "test.csv")
    X_test, y_test = split_xy(test_df, params)

    # 3. Предсказание на test с порогом из params.yaml
    proba, pred = predict(model, X_test, cfg["threshold"])

    # 4. Подсчёт метрик
    metrics = compute_metrics(y_test, proba, pred)
    log.info("метрики на test: %s", metrics)

    # 5. Сохранение метрик (до проверки порога: файл нужен, даже если порог не пройден)
    save_metrics(metrics, resolve(EVAL_METRICS_PATH))
    save_roc_curve(y_test, proba, resolve(ROC_PATH))

    # 6. Проверка порога качества: ниже min_roc_auc — выход с кодом 1
    if not is_quality_ok(metrics, cfg["min_roc_auc"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
