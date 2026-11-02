import uuid

import pytest
from fastapi.testclient import TestClient

VALID = {
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "total_charges": 906.0,
    "contract_type": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "electronic_check",
    "num_support_calls": 2,
    "has_tech_support": 0,
    "is_senior": 0,
    "avg_monthly_gb": 30.0,
}


@pytest.fixture(scope="module")
def client(trained_pipeline):
    """Подсовываем сервису модель из фикстуры: тесты API не должны зависеть
    от того, лежит ли в models/ артефакт с прошлого запуска."""
    from src.service import app as app_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module.holder, "load", lambda: None)
        mp.setattr(app_module.holder, "model", trained_pipeline)
        mp.setattr(app_module.holder, "version", "test:fixture")
        with TestClient(app_module.app) as c:
            yield c


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "test:fixture"}


def test_predict_returns_valid_probability(client):
    body = client.post("/predict", json=VALID).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"]


def test_unknown_category_returns_422(client):
    assert client.post("/predict", json=dict(VALID, contract_type="lifetime")).status_code == 422


def test_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID.items() if k != "tenure_months"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["type"] == "missing" and e["loc"][-1] == "tenure_months" for e in errors)


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure_months", -1),
        ("monthly_charges", 5000),
        ("num_support_calls", 101),
        ("has_tech_support", 2),
    ],
)
def test_out_of_range_returns_422(client, field, value):
    response = client.post("/predict", json=dict(VALID, **{field: value}))

    assert response.status_code == 422
    assert any(e["loc"][-1] == field for e in response.json()["detail"])


def test_batch_matches_single_predictions(client):
    items = [VALID, dict(VALID, contract_type="two_year"), dict(VALID, num_support_calls=8)]
    response = client.post("/predict/batch", json={"items": items})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(items) == len(body["predictions"])

    for item, batch_pred in zip(items, body["predictions"]):
        single_pred = client.post("/predict", json=item).json()
        assert batch_pred == single_pred


def test_empty_batch_returns_422(client):
    response = client.post("/predict/batch", json={"items": []})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["type"] == "too_short" and e["loc"][-1] == "items" for e in errors)


def test_explain_returns_sorted_top5(client):
    response = client.post("/explain", json=VALID)

    assert response.status_code == 200
    body = response.json()
    assert body["method"] in ("linear_contribution", "global_importance")
    assert len(body["top_features"]) == 5

    magnitudes = [abs(f["contribution"]) for f in body["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)

    predicted = client.post("/predict", json=VALID).json()["churn_probability"]
    assert body["churn_probability"] == predicted


def test_request_id_is_echoed_or_generated(client):
    echoed = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert echoed.headers["X-Request-ID"] == "test-123"

    generated = client.get("/health").headers["X-Request-ID"]
    assert str(uuid.UUID(generated)) == generated
