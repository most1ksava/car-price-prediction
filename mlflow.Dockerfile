FROM python:3.9-slim

# Установка зависимостей
RUN pip install --no-cache-dir mlflow==2.11.0 psycopg2-binary==2.9.9 boto3==1.28.0

# Создание директории для MLflow
RUN mkdir -p /mlflow

WORKDIR /mlflow

# Запуск MLflow сервера
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow/", \
     "--host", "0.0.0.0"] 