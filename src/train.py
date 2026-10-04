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

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import TARGET, feature_columns, load_params, resolve
from src.features import build_preprocessor
from src.logging_setup import setup_logging

log = setup_logging()

# TODO (занятие 5): пути стадий перенести в params.yaml
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "reports/train_metrics.json"

# Имя модели из params["train"]["model"] -> класс sklearn
MODELS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("прочитано %s строк из %s", len(df), path)
    return df


def split_xy(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns(params)], df[TARGET]


def build_model(params: dict) -> Pipeline:
    name = params["train"]["model"]
    if name not in MODELS:
        raise ValueError(f"неизвестная модель {name!r}, допустимые: {list(MODELS)}")
    hyperparams = params["train"][name]
    log.info("модель: %s, гиперпараметры: %s", name, hyperparams)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(params)),
            ("model", MODELS[name](**hyperparams, random_state=params["seed"])),
        ]
    )


def predict(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return model.predict_proba(X)[:, 1], model.predict(X)


def compute_metrics(y: pd.Series, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, pred)),
    }


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("модель сохранена в %s", path)


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")  # end-of-file-fixer требует перевод строки в конце файла
    log.info("метрики сохранены в %s", path)


def main() -> None:
    params = load_params()
    processed_dir = resolve(params["data"]["processed_dir"])

    np.random.seed(params["seed"])

    # 1. Загрузка train и val
    train_df = load_data(processed_dir / "train.csv")
    val_df = load_data(processed_dir / "val.csv")

    # 2. Разделение на признаки X и таргет y
    X_train, y_train = split_xy(train_df, params)
    X_val, y_val = split_xy(val_df, params)

    # 3. Сборка Pipeline (препроцессор + модель из params["train"]["model"])
    model = build_model(params)

    # 4. Обучение на train
    model.fit(X_train, y_train)
    log.info("модель обучена на %s строках", len(X_train))

    # 5. Предсказание на val
    proba, pred = predict(model, X_val)

    # 6. Подсчёт метрик
    metrics = compute_metrics(y_val, proba, pred)
    log.info("метрики на val: %s", metrics)

    # 7. Сохранение модели вместе с препроцессором (весь Pipeline)
    save_model(model, resolve(MODEL_PATH))

    # 8. Сохранение метрик
    save_metrics(metrics, resolve(METRICS_PATH))


if __name__ == "__main__":
    main()
