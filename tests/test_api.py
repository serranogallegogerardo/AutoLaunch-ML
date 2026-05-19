"""
tests/test_api.py
Tests unitarios para la API de inferencia.
"""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    """El endpoint raíz debe devolver 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    """El endpoint /health siempre devuelve ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    """Predicción con features válidas debe devolver 200 y tener 'prediction'."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data


def test_predict_invalid_input():
    """Payload incompleto debe devolver error de validación (422)."""
    payload = {"sepal_length": 5.1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
