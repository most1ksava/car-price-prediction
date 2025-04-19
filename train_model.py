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

# Настройка переменных среды для MinIO
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

# Настройка MLFlow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("car-price-prediction")

# Функция для загрузки и подготовки данных
def load_and_prepare_data(file_path="autos.csv"):
    # Загрузка данных
    data = pd.read_csv(file_path)
    
    # Базовая обработка данных (аналогичная из Jupyter notebook)
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
    data = data.drop_duplicates().reset_index(drop=True)
    
    # Обработка столбца repaired
    data['repaired'] = data['repaired'].map({'yes': 1, 'no': 0}).fillna(0).astype('int64')
    
    # Фильтрация по году регистрации
    data = data.loc[(data['registration_year'] <= 2016) & (data['registration_year'] >= 1900)]
    
    # Фильтрация по цене
    data = data.loc[(data['price'] > 0) & (data['price'] < 100000)]
    
    # Обработка пропущенных значений
    # Заполняем пропуски в категориальных признаках
    data['vehicle_type'] = data['vehicle_type'].fillna('unknown')
    data['gearbox'] = data['gearbox'].fillna('unknown')
    data['model'] = data['model'].fillna('unknown')
    data['fuel_type'] = data['fuel_type'].fillna('unknown')
    
    return data

# Функция для обучения модели
def train_model(data):
    # Разделение на признаки и целевую переменную
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Кодирование категориальных признаков
    categorical_cols = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели XGBoost с оптимальными параметрами
    with mlflow.start_run(run_name="car-price-model") as run:
        # Логируем параметры
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("learning_rate", 0.15)
        
        # Создаем и обучаем модель
        model = xgb.XGBRegressor(
            max_depth=12,
            learning_rate=0.15,
            booster='gbtree',
            n_estimators=100,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Логируем метрики
        mlflow.log_metric("rmse", rmse)
        
        # Регистрируем модель
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="car-price-model",
            registered_model_name="car-price-model"
        )
        
        # Логируем информацию о данных
        mlflow.log_dict({"features": list(X.columns)}, "features.json")
        
        # Статус и результат
        print(f"Model training completed. RMSE: {rmse:.2f}")
        print(f"Model registered with MLflow with run_id: {run.info.run_id}")
        
        return model, encoder, X.columns, rmse

if __name__ == "__main__":
    # Проверяем, существует ли файл с данными
    if not os.path.exists("autos.csv"):
        print("Error: File 'autos.csv' not found. Please place it in the current directory.")
        exit(1)
    
    # Загружаем и подготавливаем данные
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    # Обучаем модель
    print("Training model...")
    model, encoder, features, rmse = train_model(data)
    
    print("Done!") 