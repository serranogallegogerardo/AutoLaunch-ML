"""
mlflow/config.py
Configuración centralizada para MLflow tracking.
Importá este módulo desde cualquier script que necesite loguear.
"""
import mlflow

TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "iris_classification_tfg"


def setup_mlflow():
    """Configura MLflow con el tracking URI y el experimento."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow
