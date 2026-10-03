"""Стадия prepare: сырой CSV -> train/val/test.

TODO (занятие 1):
  1. прочитать data/raw/churn.csv;
  2. обработать пропуски в total_charges осмысленно (не dropna!);
  3. разбить на train/val/test со stratify по churn и random_state из params;
  4. сохранить три CSV в data/processed/.

Проверка: два запуска подряд должны дать одинаковые файлы.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET, feature_columns, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("прочитано %s строк из %s", len(df), path)
    return df


def validate_columns(df: pd.DataFrame, params: dict) -> None:
    required = feature_columns(params) + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"в данных нет колонок: {missing}")


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["total_charges"].isna()
    df.loc[mask, "total_charges"] = (
        df.loc[mask, "monthly_charges"] * df.loc[mask, "tenure_months"]
    ).round(2)
    log.info("заполнено пропусков в total_charges: %s", int(mask.sum()))
    return df


def split(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # отрезаем test_df, остаётся part_df
    part_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    # val_size задан в долях от всего датасета, а отделяем мы его от остатка —
    # поэтому долю пересчитываем
    val_share = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        part_df, test_size=val_share, stratify=part_df[TARGET], random_state=seed
    )

    return train_df, val_df, test_df


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        log.info("%s: %s строк, churn rate=%.3f -> %s",
                 name, len(part), part[TARGET].mean(), path)


def main() -> None:
    params = load_params()
    d = params["data"]
    seed = params["seed"]

    # 1. Чтение raw данные с помощью pd.read_csv()
    df = load_raw(resolve(d["raw_path"]))

    # Проверка, что в данных есть все признаки из params.yaml и таргет
    validate_columns(df, params)

    # 2. Заполнение пропусков в total_charges как monthly_charges * tenure_months
    df = fill_missing(df)

    # 3. Разбиение на train/val/test со stratify по churn и random_state=seed
    train, val, test = split(df, d["test_size"], d["val_size"], seed)

    # 4. Сохранение трёх CSV в data/processed/
    save_splits({"train": train, "val": val, "test": test}, resolve(d["processed_dir"]))


if __name__ == "__main__":
    main()
