import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.preprocessing import OrdinalEncoder
from enum import Enum

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Настройка MLflow
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

# Создание FastAPI приложения
app = FastAPI(
    title="Car Price Prediction API",
    description="API для предсказания стоимости автомобилей",
    version="1.0.0"
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель для входных данных
class CarFeatures(BaseModel):
    vehicle_type: Optional[str] = Field(None, description="Тип кузова автомобиля (sedan, coupe, suv, и т.д.)")
    registration_year: int = Field(..., description="Год регистрации автомобиля", ge=1900, le=2023)
    gearbox: Optional[str] = Field(None, description="Тип коробки передач (manual, auto)")
    power: int = Field(..., description="Мощность автомобиля в л.с.", ge=0)
    model: Optional[str] = Field(None, description="Модель автомобиля")
    kilometer: int = Field(..., description="Пробег автомобиля в км", ge=0)
    fuel_type: Optional[str] = Field(None, description="Тип топлива (petrol, gasoline, diesel, и т.д.)")
    brand: str = Field(..., description="Марка автомобиля")
    repaired: int = Field(0, description="Был ли автомобиль в ремонте (0 - не был, 1 - был)", ge=0, le=1)

# Модель для ответа
class PredictionResponse(BaseModel):
    price: float = Field(..., description="Предсказанная стоимость автомобиля в евро")
    model_version: str = Field(..., description="Версия использованной модели")

# Функция для загрузки модели из MLflow
def load_model():
    try:
        logger.info(f"Загрузка модели из MLflow по URI: {mlflow_tracking_uri}")
        model_name = "car-price-model"
        
        # Получаем последнюю версию модели
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        logger.info(f"Загрузка версии модели: {latest_version}")
        
        # Загружаем модель
        model_uri = f"models:/{model_name}/{latest_version}"
        
        # Загружаем модель с явным указанием типа
        try:
            # Сначала пробуем загрузить как XGBoost модель
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Модель успешно загружена как XGBoost: {model_uri}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить как XGBoost модель: {str(e)}")
            # Если не получается, загружаем как универсальную PyFunc модель
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Модель успешно загружена как PyFunc: {model_uri}")
        
        return model, latest_version
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {str(e)}")

# Создаем зависимость для модели
def get_model():
    if not hasattr(get_model, "model"):
        get_model.model, get_model.version = load_model()
    return get_model.model, get_model.version

# Маршрут для проверки статуса API
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Car Price Prediction API is running"}

# Маршрут для проверки здоровья
@app.get("/health")
def health_check():
    try:
        # Проверяем подключение к MLflow
        client.search_experiments()
        return {"status": "healthy", "mlflow_connection": "ok"}
    except Exception as e:
        logger.error(f"Ошибка проверки здоровья: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# Маршрут для предсказания
@app.post("/predict", response_model=PredictionResponse)
def predict(car_features: CarFeatures, model_info: tuple = Depends(get_model)):
    try:
        model, version = model_info
        logger.info(f"Получен запрос на предсказание: {car_features.dict()}")

        # Преобразуем словарь в DataFrame
        input_data = pd.DataFrame([car_features.dict()])
        logger.info(f"Исходные данные: \n{input_data}")
        
        # Максимально упрощаем преобразование данных
        # Заполняем пропуски для строковых данных
        for col in ['vehicle_type', 'gearbox', 'model', 'fuel_type']:
            if col in input_data.columns:
                input_data[col] = input_data[col].fillna('unknown')
        
        # Кодируем категориальные признаки
        dict_mapping = {
            'vehicle_type': {'sedan': 1, 'coupe': 2, 'suv': 3, 'wagon': 4, 'van': 5},
            'gearbox': {'manual': 1, 'auto': 2},
            'fuel_type': {'petrol': 1, 'diesel': 2, 'gas': 3, 'hybrid': 4, 'electric': 5}
        }
        
        # Применяем преобразования
        for col, mapping in dict_mapping.items():
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(lambda x: mapping.get(x, 0) if isinstance(x, str) and x.lower() in mapping else 0)
        
        # Остальные строковые данные преобразуем в порядковые номера
        for col in ['model', 'brand']:
            if col in input_data.columns and input_data[col].dtype == 'object':
                input_data[col] = pd.factorize(input_data[col])[0]
        
        # Преобразуем все в числовой формат
        for col in input_data.columns:
            if input_data[col].dtype != 'float64' and input_data[col].dtype != 'int64':
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
        
        logger.info(f"Преобразованные данные: \n{input_data}")
        logger.info(f"Типы данных: \n{input_data.dtypes}")
        
        # Сохраним данные в файл для отладки
        try:
            input_data.to_csv('/app/debug_input.csv', index=False)
            logger.info("Сохранили данные для отладки в /app/debug_input.csv")
        except Exception as save_error:
            logger.warning(f"Не удалось сохранить данные для отладки: {str(save_error)}")
        
        # Выполняем предсказание
        logger.info("Пытаемся вызвать модель...")
        try:
            prediction = model.predict(input_data)
            logger.info(f"Результат предсказания: {prediction}")
            predicted_price = float(prediction[0])
        except Exception as model_error:
            logger.error(f"Ошибка при предсказании: {str(model_error)}")
            logger.error(f"Тип модели: {type(model)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Ошибка при выполнении предсказания модели: {str(model_error)}"
            )
        
        # Возвращаем результат
        logger.info(f"Предсказание выполнено. Цена: {predicted_price:.2f} евро")
        return PredictionResponse(price=predicted_price, model_version=version)
    
    except HTTPException:
        # Пробрасываем HTTPException дальше
        raise
    except Exception as e:
        logger.error(f"Общая ошибка в функции predict: {str(e)}", exc_info=True)
        import traceback
        error_message = f"Ошибка при обработке запроса: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_message)

# Маршрут для получения информации о модели
@app.get("/model-info")
def model_info(model_info: tuple = Depends(get_model)):
    _, version = model_info
    try:
        model_name = "car-price-model"
        model_details = client.get_model_version(model_name, version)
        
        return {
            "name": model_name,
            "version": version,
            "creation_timestamp": model_details.creation_timestamp,
            "status": model_details.status,
            "description": model_details.description or "No description available"
        }
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении информации о модели: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
