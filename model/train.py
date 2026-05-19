"""
model/train.py
Entrena un modelo RandomForest sobre el dataset Iris,
registra parámetros y métricas en MLflow, y guarda el
artefacto .joblib para servir desde la API.
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import os

def train():
    print("🚀 Cargando datos...")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🧠 Entrenando modelo...")
    # ---------- MLflow tracking ----------
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("iris_classification_tfg")

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        print(f"✅ Accuracy: {accuracy:.4f}")

        # Log de parámetros y métricas
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        # Log del modelo como artefacto MLflow
        mlflow.sklearn.log_model(clf, "model")

    # Guardado local para FastAPI
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/model.joblib")
    print("💾 Modelo guardado en model/model.joblib")

if __name__ == "__main__":
    train()
