FROM python:3.9-slim

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la API
COPY main.py /app/api/main.py

# Modelo entrenado (se copia desde el build context)
COPY ../model/model.joblib /app/model/model.joblib

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model/model.joblib

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
