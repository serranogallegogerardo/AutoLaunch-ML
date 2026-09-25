# Carpeta reproducible del TFG

Análisis del riesgo de incumplimiento de SLA y prototipo MLOps v2 para la mesa de servicio de TechServe Solutions S.A. (organización modelada).

## Datos

- **Fuente:** Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H
- **Licencia:** CC BY 4.0, que permite redistribuir el archivo citando la fuente.
- **Archivo:** `data/incident_event_log.csv`. Tiene 141.712 eventos, 24.918 incidentes y 36 atributos.
- **SHA-256:** `fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`

## Estructura

| Carpeta | Contenido |
|---|---|
| `data/` | Registro de eventos original |
| `scripts/` | Análisis CRISP-DM (`00` a `08`), `pipeline.py` (preparación compartida), `modelos.py`, `plots.py` y los CSV de riesgos y planificación |
| `results/` | Resultados en JSON y CSV, reporte de pytest y ejemplos reales de la API |
| `figures/` | Figuras del documento; `capturas/` guarda las pantallas del prototipo |
| `prototipo/model/` | `train.py` (compuerta de calidad y MLflow) y `artifacts/` (modelo publicado) |
| `prototipo/api/` | API FastAPI con clave de acceso, cálculo de drift y Dockerfile |
| `prototipo/dashboard/` | Tablero Streamlit y Dockerfile |
| `prototipo/tests/` | 11 pruebas de la API y del tablero |
| `prototipo/` | `latencia.py`, `backup.py`, `ejemplos_api.py`, `ci-cd.yml`, `mlruns.db` y los registros de ejecución |

## Ejecución

Todos los comandos se ejecutan desde esta carpeta, con Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt       # en Linux o macOS: .venv/bin/pip
python scripts/00_descargar_datos.py fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef
python scripts/01_perfil.py               # perfil.json, diccionario_crudo.csv, incidentes_preparados.csv
python scripts/02_eda_distribuciones.py   # figuras 01 a 05
python scripts/03_eda_asociaciones.py     # eda.json, figura 06
python scripts/04_modelado.py             # modelado.json, comparacion_modelos.csv, modelo_seleccionado.joblib
python scripts/05_evaluacion.py           # evaluacion.json, figuras 07 a 09
python scripts/06_drift.py                # drift.json, drift.csv, figura 10
python scripts/07_proceso_asis.py         # proceso_asis.json, figura 11
python scripts/08_diagramas.py            # figuras 12 a 19
python prototipo/model/train.py           # entrena, aplica la compuerta, registra en MLflow y publica artifacts/
python -m pytest prototipo/tests          # 11 pruebas
```

Para levantar el prototipo:

```bash
cd prototipo/api && PULSEML_API_KEY=mi-clave uvicorn main:app --port 8000
cd prototipo/dashboard && API_URL=http://127.0.0.1:8000 PULSEML_API_KEY=mi-clave streamlit run app.py
```

Para construir las imágenes Docker (desde esta carpeta):

```bash
docker build -f prototipo/api/Dockerfile -t pulseml-api:v2 .
docker build -f prototipo/dashboard/Dockerfile -t pulseml-dashboard:v2 .
```

## Resultados principales

| Indicador | Valor |
|---|---|
| ROC-AUC en la prueba temporal (HGB) | 0,784 |
| PR-AUC en la prueba temporal | 0,546 (tasa base: 0,229) |
| Incumplimientos detectados en el 20 % de mayor riesgo | 46,1 % |
| Variables con PSI ≥ 0,25 | knowledge, opened_by, category y el score predicho |
| Latencia local de /predict (500 solicitudes) | p50 = 20,7 ms; p95 = 24,2 ms; 0 errores |

## Reproducibilidad

- La semilla es 42 y está definida en `scripts/pipeline.py`.
- La partición es temporal: se entrena con los incidentes abiertos antes del 01/05/2016 y se prueba con los posteriores.
- Al volver a ejecutar los scripts `01` a `08` se obtienen JSON idénticos. La única diferencia son los segundos de entrenamiento que informa `modelado.json`.
- `latencia.json` varía según la máquina que ejecuta la prueba.

## Evidencia de ejecución

- `results/pytest_report.xml` registra las 11 pruebas aprobadas.
- `prototipo/train_log.txt` muestra la compuerta aprobada. `prototipo/gate_rechazo_log.txt` muestra un rechazo forzado con un ROC-AUC mínimo de 0,90.
- `prototipo/backup_log.txt` registra el respaldo, la pérdida simulada del modelo y su restauración verificada por hash.
- `prototipo/mlruns.db` contiene el registro de MLflow con ambas ejecuciones.

## Pendientes que requieren al autor

- Publicar la versión 2 en los Spaces `gerakp/pulseml-api` y `gerakp/pulseml-dashboard`.
- Copiar `prototipo/ci-cd.yml` a `.github/workflows/` y configurar el secreto `HF_TOKEN`.
- Grabar el video de demostración siguiendo el guion del Anexo F del informe.
