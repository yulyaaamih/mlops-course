import numpy as np
import pandas as pd

from src.data.prepare import fill_missing as clean


def test_duplicate_row_does_not_change_prediction(trained_pipeline, raw_df, cols):
    """Один и тот же клиент, поданный дважды, получает одинаковый ответ.
    Нарушение означает, что модель смотрит на позицию строки в батче."""
    row = clean(raw_df)[cols].head(1)
    doubled = pd.concat([row, row], ignore_index=True)
    proba = trained_pipeline.predict_proba(doubled)[:, 1]
    assert abs(proba[0] - proba[1]) < 1e-12


def test_add_column_dont_crash_model(trained_pipeline, raw_df, cols):
    """Новая колонка не должна ломать пайплайн"""

    X = clean(raw_df)[cols].head(200).copy()

    proba = trained_pipeline.predict_proba(X)[:, 1]
    X["new_col"] = None
    proba_with_new_col = trained_pipeline.predict_proba(X)[:, 1]

    np.testing.assert_allclose(proba, proba_with_new_col)
