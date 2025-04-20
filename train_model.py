import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import logging
import sys
import socket
import time
import requests
from requests.exceptions import RequestException

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Функция для проверки доступности порта
def is_port_open(host, port, timeout=2):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = False
    try:
        sock.connect((host, port))
        result = True
    except socket.error:
        pass
    finally:
        sock.close()
    return result

# Функция для проверки готовности MLflow сервера
def is_mlflow_ready(url, max_retries=3, timeout=15):
    """готов ли MLflow к обработке запросов"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/api/2.0/mlflow/experiments/list", timeout=timeout)
            if response.status_code == 200:
                return True
        except RequestException:
            logger.warning(f"MLflow сервер не отвечает. Попытка {i+1}/{max_retries}")
            time.sleep(1)
    return False

# Проверка доступности MLflow сервера
logger.info("Проверка доступности MLflow сервера на http://localhost:5000")
if not is_port_open("localhost", 5000):
    logger.error("MLflow сервер не доступен на http://localhost:5000. Убедитесь, что он запущен.")
    sys.exit(1)
else:
    # Проверка готовности MLflow сервера
    if not is_mlflow_ready("http://localhost:5000"):
        logger.error("MLflow сервер запущен, но не готов к обработке запросов.")
        logger.error("Дождитесь полного запуска сервера MLflow.")
        sys.exit(1)
    else:
        logger.info("MLflow сервер готов к работе")

# Проверка доступности MinIO сервера
logger.info("Проверка доступности MinIO сервера на http://localhost:9000")
if not is_port_open("localhost", 9000):
    logger.error("MinIO сервер не доступен на http://localhost:9000. Убедитесь, что он запущен.")
    sys.exit(1)
else:
    logger.info("MinIO сервер готов к работе")

# Настройка переменных среды для MinIO
logger.info("Настройка переменных среды для MinIO")
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

# Настройка MLFlow
logger.info("Попытка подключения к MLflow серверу...")
try:
    # Добавляем таймаут для предотвращения бесконечного ожидания
    mlflow.set_tracking_uri("http://localhost:5000")
    # Проверяем доступность MLflow API - запрос с таймаутом
    start_time = time.time()
    max_wait_time = 10  # максимальное время ожидания в секундах
    mlflow_ready = False
    
    while time.time() - start_time < max_wait_time:
        try:
            # Проверяем доступность API
            client = MlflowClient()
            client.search_experiments()
            mlflow_ready = True
            break
        except Exception as e:
            logger.warning(f"Ожидание готовности MLflow сервера: {e}")
            time.sleep(1)
    
    if not mlflow_ready:
        logger.error(f"MLflow сервер не ответил в течение {max_wait_time} секунд")
        sys.exit(1)
        
    logger.info("Подключение к MLflow успешно установлено")
except Exception as e:
    logger.error(f"Ошибка при подключении к MLflow: {e}")
    sys.exit(1)

try:
    logger.info("Установка эксперимента 'car-price-prediction'")
    # Добавляем таймаут для предотвращения бесконечного ожидания
    start_time = time.time()
    max_wait_time = 10  # максимальное время ожидания в секундах
    experiment_set = False
    
    while time.time() - start_time < max_wait_time:
        try:
            mlflow.set_experiment("car-price-prediction")
            experiment_set = True
            break
        except Exception as e:
            logger.warning(f"Ожидание готовности MLflow для создания эксперимента: {e}")
            time.sleep(1)
    
    if not experiment_set:
        logger.error(f"Не удалось установить эксперимент в течение {max_wait_time} секунд")
        sys.exit(1)
        
    logger.info("Эксперимент успешно установлен")
except Exception as e:
    logger.error(f"Ошибка при установке эксперимента: {e}")
    sys.exit(1)

# Функция для загрузки и подготовки данных
def load_and_prepare_data(file_path="autos.csv"):
    logger.info(f"Загрузка данных из файла {file_path}")
    try:
        # Загрузка данных
        data = pd.read_csv(file_path)
        logger.info(f"Данные успешно загружены, размер: {data.shape}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise
    
    # Базовая обработка данных (аналогичная из Jupyter notebook)
    logger.info("Начало обработки данных")
    data.columns = data.columns.str.lower()
    data = data.rename(columns={
        'datecrawled': 'date_crawled',
        'vehicletype': 'vehicle_type',
        'registrationyear': 'registration_year',
        'registrationmonth': 'registration_month',
        'fueltype': 'fuel_type',
        'notrepaired': 'not_repaired',
        'datecreated': 'date_created',
        'numberofpictures': 'number_of_pictures',
        'postalcode': 'postal_code',
        'lastseen': 'last_seen'
    })
    
    # Удаление ненужных столбцов
    data = data.drop(['date_crawled', 'registration_month', 'date_created',
                      'number_of_pictures', 'postal_code', 'last_seen'], axis=1)
    
    # Удаление дубликатов
    initial_size = data.shape[0]
    data = data.drop_duplicates().reset_index(drop=True)
    logger.info(f"Удалено дубликатов: {initial_size - data.shape[0]}")
    
    # Обработка столбца repaired
    data['repaired'] = data['repaired'].map({'yes': 1, 'no': 0}).fillna(0).astype('int64')
    
    # Фильтрация по году регистрации
    initial_size = data.shape[0]
    data = data.loc[(data['registration_year'] <= 2016) & (data['registration_year'] >= 1900)]
    logger.info(f"Удалено строк по фильтру года регистрации: {initial_size - data.shape[0]}")
    
    # Фильтрация по цене
    initial_size = data.shape[0]
    data = data.loc[(data['price'] > 0) & (data['price'] < 100000)]
    logger.info(f"Удалено строк по фильтру цены: {initial_size - data.shape[0]}")
    
    # Обработка пропущенных значений
    # Заполняем пропуски в категориальных признаках
    data['vehicle_type'] = data['vehicle_type'].fillna('unknown')
    data['gearbox'] = data['gearbox'].fillna('unknown')
    data['model'] = data['model'].fillna('unknown')
    data['fuel_type'] = data['fuel_type'].fillna('unknown')
    
    logger.info(f"Данные успешно обработаны, итоговый размер: {data.shape}")
    return data

# Функция для обучения модели
def train_model(data):
    logger.info("Начало процесса обучения модели")
    # Разделение на признаки и целевую переменную
    X = data.drop('price', axis=1)
    y = data['price']
    logger.info(f"Подготовлены признаки: {X.shape} и целевая переменная: {y.shape}")
    
    # Кодирование категориальных признаков
    categorical_cols = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand']
    logger.info(f"Кодирование категориальных признаков: {categorical_cols}")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Данные разделены на обучающую ({X_train.shape[0]} строк) и тестовую ({X_test.shape[0]} строк) выборки")
    
    # Обучение модели XGBoost с оптимальными параметрами
    logger.info("Попытка запуска MLflow run для логирования обучения модели")
    try:
        with mlflow.start_run(run_name="car-price-model") as run:
            run_id = run.info.run_id
            logger.info(f"Запущен MLflow run с ID: {run_id}")
            
            # Логируем параметры
            logger.info("Логирование параметров модели")
            mlflow.log_param("max_depth", 12)
            mlflow.log_param("learning_rate", 0.15)
            
            # Создаем и обучаем модель
            logger.info("Создание и обучение модели XGBRegressor")
            model = xgb.XGBRegressor(
                max_depth=12,
                learning_rate=0.15,
                booster='gbtree',
                n_estimators=100,
                random_state=42
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Модель обучена за {training_time:.2f} секунд")
            
            # Оценка модели
            logger.info("Оценка модели на тестовых данных")
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f"RMSE на тестовой выборке: {rmse:.2f}")
            
            # Логируем метрики
            logger.info("Логирование метрик в MLflow")
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Регистрируем модель
            logger.info("Регистрация модели в MLflow")
            try:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="car-price-model",
                    registered_model_name="car-price-model"
                )
                logger.info("Модель успешно зарегистрирована")
            except Exception as e:
                logger.error(f"Ошибка при регистрации модели: {e}")
            
            # Логируем информацию о данных
            logger.info("Логирование информации о признаках")
            try:
                mlflow.log_dict({"features": list(X.columns)}, "features.json")
                logger.info("Информация о признаках успешно залогирована")
            except Exception as e:
                logger.error(f"Ошибка при логировании информации о признаках: {e}")
            
            # Статус и результат
            logger.info(f"Обучение модели завершено. RMSE: {rmse:.2f}")
            logger.info(f"Модель зарегистрирована в MLflow с run_id: {run_id}")
            
            return model, encoder, X.columns, rmse
    except Exception as e:
        logger.error(f"Критическая ошибка при работе с MLflow: {e}")
        raise

if __name__ == "__main__":
    try:
        # Проверяем, существует ли файл с данными
        if not os.path.exists("autos.csv"):
            logger.error("Ошибка: Файл 'autos.csv' не найден. Разместите его в текущей директории.")
            sys.exit(1)
        
        # Загружаем и подготавливаем данные
        logger.info("Начинаем загрузку и подготовку данных...")
        data = load_and_prepare_data()
        
        # Обучаем модель
        logger.info("Начинаем обучение модели...")
        model, encoder, features, rmse = train_model(data)
        
        logger.info("Выполнение скрипта успешно завершено!")
    except Exception as e:
        logger.critical(f"Критическая ошибка при выполнении скрипта: {e}")
        sys.exit(1) 
