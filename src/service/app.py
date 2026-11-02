import uuid
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.config import feature_columns, load_params
from src.logging_setup import request_id_var, setup_logging
from src.service.model_loader import holder
from src.service.schemas import (
    BatchPrediction,
    BatchRequest,
    Customer,
    Explanation,
    FeatureContribution,
    HealthResponse,
    Prediction,
)

log = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель грузится ОДИН раз на старте. Если грузить в обработчике,
    # каждый запрос будет идти секундами.
    try:
        holder.load()
    except Exception as exc:
        log.error("не удалось загрузить модель: %s", exc)
    yield


app = FastAPI(
    title="Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info("%s %s %s", request.method, request.url.path, response.status_code)
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Жив ли процесс и загружена ли модель."""
    return HealthResponse(
        status="ok" if holder.loaded else "degraded",
        model_loaded=holder.loaded,
        model_version=holder.version,
    )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(request: BatchRequest) -> BatchPrediction:
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([c.model_dump() for c in request.items])[feature_columns(params)]
    probas = holder.model.predict_proba(df)[:, 1]
    predictions = [
        Prediction(
            churn_probability=round(float(p), 6),
            churn=int(p >= threshold),
            threshold=threshold,
            model_version=holder.version,
        )
        for p in probas
    ]
    return BatchPrediction(predictions=predictions, count=len(predictions))


@app.post("/explain", response_model=Explanation, tags=["inference"])
def explain(customer: Customer) -> Explanation:
    params = load_params()
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([customer.model_dump()])[feature_columns(params)]
    preprocessor = holder.model.named_steps["preprocessor"]
    model = holder.model.named_steps["model"]
    names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]

    if hasattr(model, "coef_"):
        row = preprocessor.transform(df)
        row = row.toarray() if hasattr(row, "toarray") else row
        values = row[0] * model.coef_[0]
        method = "linear_contribution"
    else:
        values = model.feature_importances_
        method = "global_importance"

    top = np.argsort(-np.abs(values))[:5]
    return Explanation(
        churn_probability=round(float(holder.model.predict_proba(df)[0, 1]), 6),
        method=method,
        top_features=[
            FeatureContribution(feature=names[i], contribution=round(float(values[i]), 6)) for i in top
        ],
        model_version=holder.version,
    )


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(customer: Customer) -> Prediction:
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([customer.model_dump()])[feature_columns(params)]
    proba = float(holder.model.predict_proba(df)[0, 1])
    return Prediction(
        churn_probability=round(proba, 6),
        churn=int(proba >= threshold),
        threshold=threshold,
        model_version=holder.version,
    )
