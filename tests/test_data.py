import pandas as pd
import pytest

from src.config import TARGET, feature_columns
from src.data.generate import generate


def test_schema_is_complete(raw_df, params):
    """Все колонки из params.yaml есть в данных."""
    missing = set(feature_columns(params) + [TARGET]) - set(raw_df.columns)
    assert not missing, f"в датасете нет колонок: {missing}"


def test_target_is_not_degenerate(raw_df):
    """Защита от 'все нули': такой датасет даст accuracy 0.9 и бесполезную модель."""
    rate = raw_df[TARGET].mean()
    assert 0.05 < rate < 0.60, f"подозрительный churn rate: {rate:.3f}"


def test_generation_is_reproducible():
    a, b = generate(n=500, seed=123), generate(n=500, seed=123)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "col, low, high",
    [
        ("tenure_months", 0, 200),
        ("monthly_charges", 0, 500),
        ("num_support_calls", 0, 50),
    ],
)
def test_values_in_range(raw_df, col, low, high):
    values = raw_df[col]
    bad = values[(values < low) | (values > high)]
    assert bad.empty, f"{col}: {len(bad)} значений вне [{low}, {high}], например {bad.head(3).tolist()}"


def test_customer_ids_are_unique(raw_df):
    ids = raw_df["customer_id"]
    dup = ids[ids.duplicated()]
    assert dup.empty, f"повторяются customer_id: {len(dup)} шт., например {dup.head(3).tolist()}"


@pytest.mark.parametrize("forbidden", [TARGET, "customer_id"])
def test_no_leakage_in_features(cols, forbidden):
    assert forbidden not in cols, f"{forbidden} попал в список признаков"
