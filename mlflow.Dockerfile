FROM python:3.9-slim

# Установка зависимостей
RUN pip install --no-cache-dir \
    mlflow==2.21.3 \
    psycopg2-binary==2.9.9 \
    boto3==1.28.0 \
    requests

# Создание директории для MLflow
RUN mkdir -p /mlflow

WORKDIR /mlflow

# Проверка доступности API
HEALTHCHECK --interval=5s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Запуск MLflow сервера
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow/", \
     "--host", "0.0.0.0"] 
