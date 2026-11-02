import numpy as np

from src.data.prepare import fill_missing as clean
from src.features import build_preprocessor


def test_unknown_category_does_not_crash(params, raw_df, cols):
    """На проде рано или поздно придёт категория, которой не было в обучении.
    Сервис обязан ответить предсказанием, а не 500-й ошибкой."""
    df = clean(raw_df)[cols]
    pre = build_preprocessor(params).fit(df)
    unseen = df.head(1).copy()
    unseen.loc[:, "contract_type"] = "lifetime"
    assert pre.transform(unseen).shape[0] == 1


def test_column_order_does_not_matter(params, raw_df, cols):
    """ColumnTransformer выбирает колонки по имени. Проверяем, что это правда:
    иначе клиент, приславший поля в другом порядке, получит мусор."""
    df = clean(raw_df)[cols]
    pre = build_preprocessor(params).fit(df)
    shuffled = df.head(10)[list(reversed(cols))]
    np.testing.assert_allclose(pre.transform(df.head(10)), pre.transform(shuffled))
