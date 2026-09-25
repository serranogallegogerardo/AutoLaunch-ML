# Contexto de build: carpeta 02_reproducible
#   docker build -f prototipo/dashboard/Dockerfile -t pulseml-dashboard .
FROM python:3.12-slim

WORKDIR /app
COPY prototipo/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prototipo/dashboard/ .
COPY results/drift.csv results/evaluacion.json ./results/
ENV PULSEML_RESULTS=/app/results

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
