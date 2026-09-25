# Contexto de build: carpeta 02_reproducible
#   docker build -f prototipo/api/Dockerfile -t pulseml-api .
FROM python:3.12-slim

WORKDIR /app
COPY prototipo/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prototipo/api/ .
COPY prototipo/model/artifacts/ ./artifacts/
ENV PULSEML_ARTIFACTS=/app/artifacts

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
