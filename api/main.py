"""
api/main.py
Versión final de producción para el TFG.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="PulseML Inference API",
    description="API de inferencia para el proyecto de TFG 'PulseML'",
    version="1.0.0"
)

# Carga del modelo (Estructura aplanada para Hugging Face)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado exitosamente.")
    else:
        print(f"❌ Error: El modelo no se encuentra en {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    label: str

LABEL_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.get("/")
def health_check():
    if model is None:
        return {"status": "warning", "message": "Inference engine offline: Model missing"}
    return {"status": "ok", "message": "PulseML API online 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None:
        return {"prediction": -1, "label": "error: model not loaded"}
    
    features = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    pred = int(model.predict(features)[0])
    
    return PredictionResponse(
        prediction=pred,
        label=LABEL_MAP.get(pred, "unknown")
    )
