"""Построение препроцессора признаков.

TODO (занятие 1): собрать здесь ColumnTransformer.

Требования:
  * числовые признаки: заполнение пропусков + масштабирование;
  * категориальные: заполнение пропусков + OneHotEncoder;
  * бинарные: без изменений;
  * списки колонок берутся из params.yaml, а не пишутся в коде.

Подсказка: почему препроцессор обязан ехать в одном Pipeline с моделью,
разбирается на паре. Если сделать иначе — сервис на занятии 10 сломается.
"""
from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_preprocessor(params: dict[str, Any]) -> ColumnTransformer:
    f = params["features"]
    return ColumnTransformer([
        # числовые: заполнение медианой + масштабирование
        ("num", numeric_pipeline(), f["numeric"]),
        # категориальные: заполнение самой частой категорией + one-hot
        ("cat", categorical_pipeline(), f["categorical"]),
        # бинарные: уже 0/1, оставляем как есть
        ("bin", "passthrough", f["binary"]),
    ])
