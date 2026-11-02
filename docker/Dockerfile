# ---------- стадия сборки ----------
FROM python:3.11-slim@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893 AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-service.txt .
RUN pip install --no-cache-dir --no-compile --prefix=/install -r requirements-service.txt \
    && find /install -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +

# ---------- финальный образ ----------
FROM python:3.11-slim@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893 AS runtime

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
RUN pip uninstall -y setuptools wheel pip

COPY src/ ./src/
COPY params.yaml ./params.yaml
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').json()['model_loaded'] else 1)"

CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
