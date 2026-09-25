"""Publica la versión 2 del prototipo en los Spaces de Hugging Face.

Requiere HF_TOKEN y PULSEML_API_KEY en el entorno. La clave se guarda como secreto de ambos Spaces.
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
REP = HERE.parent
API_SPACE, DASH_SPACE = "gerakp/pulseml-api", "gerakp/pulseml-dashboard"
API_URL = "https://gerakp-pulseml-api.hf.space"

README = """---
title: {title}
emoji: {emoji}
colorFrom: blue
colorTo: {color}
sdk: docker
app_port: 7860
pinned: false
---

{body}
"""

API_DOCKER = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PULSEML_ARTIFACTS=/app/artifacts
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

DASH_DOCKER = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PULSEML_RESULTS=/app/results
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
"""


def stage_api(d: Path):
    for f in ("main.py", "drift.py", "requirements.txt"):
        shutil.copy(HERE / "api" / f, d / f)
    shutil.copytree(HERE / "model" / "artifacts", d / "artifacts")
    (d / "Dockerfile").write_text(API_DOCKER, encoding="utf-8")
    (d / "README.md").write_text(README.format(
        title="Pulseml Api", emoji="👀", color="purple",
        body="PulseML API v2: riesgo de incumplimiento de SLA por incidente (TFG, Universidad Siglo 21). "
             "Endpoints: GET /health, POST /predict, POST /predict/batch, POST /monitor/drift. "
             "Los POST requieren el encabezado X-API-Key. Documentación en /docs."), encoding="utf-8")


def stage_dash(d: Path):
    for f in ("app.py", "requirements.txt"):
        shutil.copy(HERE / "dashboard" / f, d / f)
    (d / "results").mkdir()
    for f in ("drift.csv", "evaluacion.json"):
        shutil.copy(REP / "results" / f, d / "results" / f)
    (d / "Dockerfile").write_text(DASH_DOCKER, encoding="utf-8")
    (d / "README.md").write_text(README.format(
        title="Pulseml Dashboard", emoji="👁", color="gray",
        body="PulseML Dashboard v2: consulta de riesgo de SLA y monitoreo de drift (TFG, Universidad Siglo 21)."),
        encoding="utf-8")


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    key = os.environ["PULSEML_API_KEY"]
    borrar = ["*.py", "*.joblib", "model/*", "Dockerfile", "requirements.txt", "artifacts/*", "results/*"]
    for space, stage in ((API_SPACE, stage_api), (DASH_SPACE, stage_dash)):
        with tempfile.TemporaryDirectory() as tmp:
            stage(Path(tmp))
            api.add_space_secret(space, "PULSEML_API_KEY", key)
            if space == DASH_SPACE:
                api.add_space_variable(space, "API_URL", API_URL)
            info = api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp, delete_patterns=borrar,
                                     commit_message="PulseML v2: modelo de riesgo de SLA con datos reales (UCI)")
            print(space, info.oid if hasattr(info, "oid") else info)


if __name__ == "__main__":
    main()
