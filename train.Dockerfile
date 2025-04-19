FROM python:3.9-slim

# Установка зависимостей
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Создание рабочей директории
WORKDIR /workspace

# Копирование скрипта обучения
COPY train_model.py /workspace/

# Команда запуска
CMD ["python", "train_model.py"] 