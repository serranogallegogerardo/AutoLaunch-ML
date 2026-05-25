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
    feature_importance: dict
    model_type: str

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
        return {"prediction": -1, "label": "error: model not loaded", "feature_importance": {}, "model_type": "none"}
    
    features = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    prediction = model.predict(features)
    
    # Extraer importancia de características real del modelo
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    importances = model.feature_importances_.tolist()
    importance_map = {name: val for name, val in zip(feature_names, importances)}

    return {
        "prediction": int(prediction[0]),
        "label": LABEL_MAP.get(int(prediction[0]), "unknown"),
        "feature_importance": importance_map,
        "model_type": type(model).__name__
    }
