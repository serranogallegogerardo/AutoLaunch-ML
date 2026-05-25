# CHANGELOG: PulseML Pro

## [v1.2.0] - 2026-05-25 (Actual)
### Added
- **Módulo de Observabilidad:** Implementado XAI dinámico (Feature Importance) en el dashboard.
- **Data Drift:** Nueva pestaña de monitoreo de salud del modelo con histogramas de distribución.
- **GitOps:** Pipeline de GitHub Actions migrado al moderno comando `hf` para sincronización con Hugging Face Spaces.
- **Arquitectura:** Diagrama técnico interactivo integrado en el dashboard mediante Graphviz.

### Changed
- **Modelo:** RandomForest optimizado y serializado con Joblib.
- **API:** El endpoint `/predict` ahora devuelve metadatos de importancia y tipo de modelo.
- **UI:** Rediseño total del Dashboard con tema oscuro "GitHub-style", métricas en tiempo real y tabs.

## [v1.1.0] - 2026-05-20
### Added
- **Hosting:** Migración de infraestructura de GCP Cloud Run a **Hugging Face Spaces** para mejorar accesibilidad.
- **Docker:** Configuración de Dockerfiles multi-etapa para API y Dashboard.

## [v1.0.0] - Inicial
### Added
- Lanzamiento de la MVP con FastAPI básico y entrenamiento local con MLflow.
