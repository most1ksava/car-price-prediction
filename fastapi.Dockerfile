FROM python:3.9-slim

WORKDIR /app

# Устанавливаем FastAPI и зависимости
RUN pip install --no-cache-dir --default-timeout=100 fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.4.2

# Устанавливаем конкретную версию MLflow, совместимую с MLServer
RUN pip install --no-cache-dir --default-timeout=100 mlflow==2.11.0

# Устанавливаем научные библиотеки
RUN pip install --no-cache-dir --default-timeout=100 numpy==1.24.3 pandas==2.0.3
RUN pip install --no-cache-dir --default-timeout=100 scikit-learn==1.3.0

# Устанавливаем XGBoost точную версию
RUN pip install --no-cache-dir --default-timeout=100 xgboost==2.0.0 --no-deps
# Устанавливаем зависимости XGBoost
RUN pip install --no-cache-dir --default-timeout=100 scipy>=1.11.0

# Устанавливаем прочие зависимости
RUN pip install --no-cache-dir --default-timeout=100 boto3==1.28.0 psycopg2-binary==2.9.9

# Копируем файлы приложения
COPY ./app/main.py ./app/main.py
COPY ./app/__init__.py ./app/__init__.py

# Запускаем FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 