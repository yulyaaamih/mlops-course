FROM ghcr.io/mlflow/mlflow:v2.22.4

RUN pip install --no-cache-dir psycopg2-binary boto3
