# 💎 PulseML Pro: Ecosistema MLOps End-to-End

Este proyecto es un prototipo tecnológico de grado profesional desarrollado para el **Trabajo Final de Grado (TFG)** en la Licenciatura en Ciencia de Datos (Universidad Siglo 21). Implementa un ecosistema MLOps completo (Nivel 2 de Madurez), orquestado íntegramente en la nube mediante GitOps.

## 🚀 Enlaces en Vivo
*   **Inference API:** [PulseML API (FastAPI)](https://huggingface.co/spaces/gerakp/pulseml-api)
*   **Observability Dashboard:** [PulseML Pro Dashboard (Streamlit)](https://huggingface.co/spaces/gerakp/pulseml-dashboard)

## ✨ Características Principales
*   **Servicio de Inferencia:** API RESTful construida con FastAPI, optimizada para baja latencia y tipado con Pydantic.
*   **Observabilidad & XAI:** Panel interactivo con **Explainability (Feature Importance)** dinámica extraída directamente del modelo en tiempo real.
*   **Monitoreo de Salud:** Módulo de detección de **Data Drift** (Deriva de Datos) con visualizaciones comparativas de distribución (Entrenamiento vs Inferencia).
*   **GitOps (CI/CD):** Pipeline 100% automatizado mediante GitHub Actions que valida el código (Pytest) y despliega a Hugging Face on-push.
*   **Tracking de Experimentos:** Ciclo de vida gestionado con **MLflow** local (SQLite) para registro de métricas y artefactos.
*   **Contenerización:** Despliegue basado en Docker para asegurar reproducibilidad total.

## 🛠️ Elementos "Hardcodeados" (Transparencia Académica)
Para esta Prueba de Concepto (PoC), se han mantenido como estáticos:
1.  **Repo Config:** Los nombres de los espacios (`gerakp/pulseml-api`) están fijos en la configuración de CI/CD.
2.  **Drift Baseline:** Los gráficos de monitoreo utilizan una comparativa entre el dataset de entrenamiento real y una distribución de inferencia simulada para demostrar capacidad operativa.
3.  **Local Storage:** El backend de MLflow es local; escalable a servidores remotos (S3/GCS).

## 📈 Próximos Pasos (Nice to Have)
- [ ] **A/B Testing:** Activar tráfico dividido entre modelos v1 y v2.
- [ ] **Remote Registry:** Migrar los artefactos de MLflow a una nube centralizada.
- [ ] **Alerting:** Notificaciones automáticas ante detección de anomalías en datos de entrada.

---
**Autor:** Gerardo Juan Martín Serrano Gallego  
**TFG:** MLOps aplicado a la Gestión Analítica.
