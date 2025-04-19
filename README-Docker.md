# Запуск проекта через Docker Hub

Этот проект доступен как набор готовых Docker-образов на Docker Hub, что позволяет быстро запустить без сборки из исходного кода.

## Предварительные требования

- Docker
- Docker Compose

## Быстрый старт

### 1. Создайте файл docker-compose.yml

Создайте в любой папке файл `docker-compose.yml` со следующим содержимым:

```yaml
services:
  # PostgreSQL для хранения метаданных MLFlow
  postgres:
    image: postgres:14
    container_name: mlflow-postgres
    restart: always
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mlflow"]
      interval: 5s
      retries: 5

  # MinIO для хранения артефактов 
  minio:
    image: minio/minio:latest
    container_name: mlflow-minio
    restart: always
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 5s
      retries: 3

  # MinIO Client для создания бакета
  mc:
    image: minio/mc:latest
    container_name: minio-client
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      sleep 5;
      /usr/bin/mc config host add myminio http://minio:9000 minio minio123;
      /usr/bin/mc mb myminio/mlflow;
      exit 0;
      "

  # MLFlow сервер
  mlflow:
    image: most1ksava/car-price-mlflow:latest
    container_name: mlflow-server
    restart: always
    depends_on:
      - postgres
      - minio
    environment:
      - AWS_ACCESS_KEY_ID=minio
      - AWS_SECRET_ACCESS_KEY=minio123
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow

  # FastAPI сервис
  fastapi:
    image: most1ksava/car-price-fastapi:latest
    container_name: fastapi-service
    restart: always
    depends_on:
      - mlflow
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - AWS_ACCESS_KEY_ID=minio
      - AWS_SECRET_ACCESS_KEY=minio123
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000

volumes:
  postgres_data:
  minio_data:
```

### 2. Запустите сервисы

```bash
docker-compose up -d
```

### 3. Скачайте и распакуйте датасет

Скачайте файл [autos.csv](https://github.com/your-username/car-price-prediction/raw/main/autos.csv) и поместите его в текущую директорию.

### 4. Обучите модель

```bash
docker run --rm -v $(pwd):/workspace -w /workspace --network host yourname/car-price-train:latest
```

### 5. Доступ к сервисам

- MLFlow UI: [http://localhost:5000](http://localhost:5000)
- MinIO Console: [http://localhost:9001](http://localhost:9001) (логин: minio, пароль: minio123)
- FastAPI Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

## Тестирование API

### 1. Проверка статуса сервиса

```bash
curl -X GET http://localhost:8000/health
```

### 2. Получение информации о модели

```bash
curl -X GET http://localhost:8000/model-info
```

### 3. Предсказание цены автомобиля

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vehicle_type": "sedan", "registration_year": 2010, "gearbox": "auto", "power": 150, "model": "golf", "kilometer": 80000, "fuel_type": "petrol", "brand": "volkswagen", "repaired": 0}'
```

## Доступные Docker-образы

1. `most1ksava/car-price-fastapi` - FastAPI сервис для предсказаний
2. `most1ksava/car-price-mlflow` - Сервер MLflow
3. `most1ksava/car-price-train` - Скрипт для обучения модели
