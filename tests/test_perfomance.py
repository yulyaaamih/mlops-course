import time

from src.data.prepare import fill_missing as clean


def test_batch_inference_is_fast_enough(trained_pipeline, raw_df, cols):
    """1000 объектов должны считаться быстрее секунды.
    Порог с запасом: ловим катастрофу (кто-то вызвал fit в predict),
    а не колебания на разных машинах."""
    batch = clean(raw_df)[cols].sample(1000, replace=True, random_state=0)
    start = time.perf_counter()
    trained_pipeline.predict_proba(batch)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"инференс 1000 строк занял {elapsed:.2f} с"
