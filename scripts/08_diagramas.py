"""Diagramas del documento: organigrama, arquitectura, flujo de datos, secuencia, navegación, riesgos y Gantt."""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plots import BLUE, plt, save  # noqa: E402

LIGHT, GREY, RED = "#DDEBF7", "#44546A", "#C00000"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, p1, p2, text="", color=GREY, style="-|>", ls="-", rad=0.0, fs=7):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle=style, color=color, lw=1.1, ls=ls,
                                                      connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12, text, fontsize=fs, ha="center", color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


# 12. Organigrama de TechServe (modelado)
fig, ax = canvas(9, 4.2, (0, 10), (0, 5))
box(ax, 5, 4.4, 3.2, 0.7, "Gerencia de Operaciones y Datos\n(decisión sobre SLA y recursos)", bold=True)
for x, t in [(1.7, "Mesa de Servicio\n(analistas N1/N2,\n69 grupos resolutores)"),
             (5, "Ciencia de Datos\n(modelado y análisis)"),
             (8.3, "Operaciones TI\n(infraestructura y despliegue)")]:
    arrow(ax, (5, 4.05), (x, 3.25))
    box(ax, x, 2.8, 2.8, 0.9, t)
box(ax, 1.7, 1.2, 2.8, 0.8, "Registra y gestiona\nincidentes (ServiceNow)", fc="white")
box(ax, 5, 1.2, 2.8, 0.8, "Entrena modelos en\nnotebooks locales", fc="white")
box(ax, 8.3, 1.2, 2.8, 0.8, "Publica servicios\nmanualmente", fc="white")
for x in (1.7, 5, 8.3):
    arrow(ax, (x, 2.35), (x, 1.62), style="-")
ax.set_title("Estructura del área involucrada (organización modelada)")
save(fig, "fig_12_organigrama.png")

# 13. Arquitectura v2 (funcional verificada en local; publicación en la nube pendiente)
fig, ax = canvas(10, 5, (0, 12), (0, 6))
box(ax, 1.3, 4.8, 2.2, 0.8, "Registro de eventos\nUCI / ServiceNow (CSV)")
box(ax, 4.2, 4.8, 2.4, 0.8, "pipeline.py\npreparación por incidente")
box(ax, 7.3, 4.8, 2.6, 0.8, "train.py\nHGB + compuerta de calidad")
box(ax, 10.4, 4.8, 2.2, 0.8, "MLflow\n(SQLite: parámetros,\nmétricas, tags)")
box(ax, 7.3, 2.9, 2.6, 0.8, "Artefactos del modelo\nmodel.joblib, metadata,\nreference.json")
box(ax, 4.2, 1.1, 2.6, 0.9, "API FastAPI (Docker)\n/predict · /predict/batch\n/monitor/drift · /health")
box(ax, 8.9, 1.1, 2.6, 0.9, "Dashboard Streamlit\n(Docker)\nconsulta y drift")
box(ax, 1.2, 2.9, 2.0, 0.8, "GitHub Actions\npruebas pytest (CI)", fc="#FFF2CC", ec="#BF8F00")
box(ax, 1.2, 1.1, 2.0, 0.8, "Analista de la\nmesa de servicio", fc="#E2EFDA", ec="#548235")
arrow(ax, (2.4, 4.8), (3.0, 4.8)); arrow(ax, (5.4, 4.8), (6.0, 4.8)); arrow(ax, (8.6, 4.8), (9.3, 4.8), "log")
arrow(ax, (7.3, 4.4), (7.3, 3.3), "solo si aprueba")
arrow(ax, (6.9, 2.5), (4.8, 1.55), "carga")
arrow(ax, (7.6, 1.1), (5.5, 1.1), "HTTP + X-API-Key")
arrow(ax, (2.2, 1.1), (2.9, 1.1), "consulta")
arrow(ax, (1.2, 3.3), (6.0, 4.6), "ejecuta", color="#BF8F00", ls="--", rad=-0.15)
ax.set_title("Arquitectura del prototipo v2")
save(fig, "fig_13_arquitectura.png")

# 14. Flujo de datos completo
fig, ax = canvas(10, 4.6, (0, 12), (0, 5.5))
steps = [(1.1, "Fuente\nlog de eventos\n141.712 filas"), (3.3, "Carga\n'?' → nulo\n36 atributos"),
         (5.5, "Agregación\n1 fila por incidente\n24.918 filas"), (7.7, "Variables\n15 predictoras\n(1.er evento)"),
         (9.9, "Partición\ntemporal\n2016-05-01")]
for x, t in steps:
    box(ax, x, 4.2, 1.9, 1.1, t)
for (x1, _), (x2, _) in zip(steps[:-1], steps[1:]):
    arrow(ax, (x1 + 0.95, 4.2), (x2 - 0.95, 4.2))
low = [(2.2, "Entrenamiento\n17.136 incidentes"), (5.0, "Modelo + perfil\nde referencia"),
       (7.8, "Servicio\n/predict"), (10.4, "Monitoreo\nPSI / KS")]
for x, t in low:
    box(ax, x, 1.5, 2.1, 0.9, t, fc="white")
arrow(ax, (9.9, 3.65), (2.2, 1.95), "feb-abr 2016", rad=0.1)
arrow(ax, (3.25, 1.5), (3.95, 1.5)); arrow(ax, (6.05, 1.5), (6.75, 1.5)); arrow(ax, (8.85, 1.5), (9.35, 1.5))
arrow(ax, (10.3, 3.65), (10.4, 1.95), "may 2016 →\n(prueba)")
ax.text(6, 0.35, "Los atributos de cierre (resolved_at, closed_at, made_sla) solo se usan para construir el objetivo.",
        ha="center", fontsize=8, style="italic", color=GREY)
ax.set_title("Flujo de datos desde la fuente hasta el monitoreo")
save(fig, "fig_14_flujo_datos.png")

# 15. Secuencia de una consulta
fig, ax = canvas(10, 4.6, (0, 10), (0, 6))
actors = [(1, "Analista"), (3.7, "Dashboard"), (6.4, "API FastAPI"), (9, "Modelo HGB")]
for x, t in actors:
    box(ax, x, 5.5, 1.8, 0.55, t, bold=True)
    ax.plot([x, x], [5.2, 0.3], color="#BFBFBF", lw=1, ls="--")
msgs = [(1, 3.7, 4.6, "1. completa formulario"), (3.7, 6.4, 3.9, "2. POST /predict + X-API-Key"),
        (6.4, 6.4, 3.3, "3. valida esquema (422 si falla) y clave (401)"), (6.4, 9, 2.7, "4. predict_proba"),
        (9, 6.4, 2.1, "5. probabilidad"), (6.4, 3.7, 1.5, "6. probabilidad, riesgo, umbral, versión"),
        (3.7, 1, 0.9, "7. muestra resultado")]
for x1, x2, y, t in msgs:
    if x1 == x2:
        ax.text(x1 + 0.15, y, t, fontsize=7.5, va="center")
    else:
        arrow(ax, (x1, y), (x2, y), ls="--" if x2 < x1 else "-")
        ax.text((x1 + x2) / 2, y + 0.15, t, fontsize=7.5, ha="center")
ax.set_title("Secuencia de una consulta de riesgo")
save(fig, "fig_15_secuencia.png")

# 16. Navegación del usuario
fig, ax = canvas(10, 3.8, (0, 12), (0, 4.5))
box(ax, 1.2, 2.2, 2.0, 0.9, "Ingreso al\ndashboard", bold=True)
box(ax, 4.2, 2.2, 2.2, 1.1, "Encabezado\nestado API, versión,\nROC-AUC, PR-AUC")
box(ax, 7.4, 3.4, 2.4, 0.9, "Pestaña 1\nConsulta de incidente")
box(ax, 7.4, 1.0, 2.4, 0.9, "Pestaña 2\nMonitoreo de drift")
box(ax, 10.6, 3.4, 2.2, 0.9, "Resultado:\nprobabilidad y riesgo", fc="white")
box(ax, 10.6, 1.0, 2.2, 0.9, "PSI por variable,\nalertas, AUC mensual", fc="white")
arrow(ax, (2.2, 2.2), (3.1, 2.2)); arrow(ax, (5.3, 2.4), (6.2, 3.3)); arrow(ax, (5.3, 2.0), (6.2, 1.1))
arrow(ax, (8.6, 3.4), (9.5, 3.4), "Calcular riesgo"); arrow(ax, (8.6, 1.0), (9.5, 1.0))
ax.set_title("Navegación del usuario en el dashboard")
save(fig, "fig_16_navegacion.png")

# 17. Matriz de riesgos probabilidad x impacto
risks = pd.read_csv(Path(__file__).parent / "riesgos.csv")
fig, ax = plt.subplots(figsize=(6.4, 5))
for p in range(1, 6):
    for i in range(1, 6):
        s = p * i
        ax.add_patch(plt.Rectangle((i - 0.5, p - 0.5), 1, 1, color="#C00000" if s >= 15 else "#ED7D31" if s >= 8 else "#FFD966" if s >= 4 else "#A9D08E", alpha=0.55))
for (p, i), g in risks.groupby(["probabilidad", "impacto"]):
    ax.text(i, p, "\n".join(g["id"]), ha="center", va="center", fontsize=8, weight="bold")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
ax.set_xlabel("Impacto (1 = muy bajo, 5 = muy alto)"); ax.set_ylabel("Probabilidad (1 = rara, 5 = casi segura)")
ax.set_title("Matriz de riesgos (P × I)")
save(fig, "fig_17_matriz_riesgos.png")

# 18. Gantt con dependencias e hitos de control
tasks = pd.read_csv(Path(__file__).parent / "planificacion.csv", parse_dates=["inicio", "fin"])
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, r in tasks.iloc[::-1].reset_index(drop=True).iterrows():
    ax.barh(i, (r.fin - r.inicio).days, left=r.inicio, color="#8EA9DB" if r.tipo == "tarea" else "#C00000",
            height=0.55 if r.tipo == "tarea" else 0.25)
    ax.text(r.inicio, i, f" {r.id}", va="center", fontsize=7)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks["nombre"].iloc[::-1], fontsize=7.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m")); ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(fontsize=7, rotation=45)
ax.grid(axis="x", alpha=0.3)
ax.set_title("Planificación 2026: tareas (azul) e hitos de control (rojo)")
save(fig, "fig_18_gantt.png")
# 19. Diagrama entidad-relación de los datos del prototipo
fig, ax = canvas(10, 5.4, (0, 12), (0, 6.5))
ents = {
    "EVENTO": (2.0, 4.6, "EVENTO (fuente)\nPK: number + sys_mod_count\nincident_state, active,\nreassignment_count, made_sla,\nsys_updated_at, ... (36 atributos)"),
    "INCIDENTE": (6.0, 4.6, "INCIDENTE\nPK: number\n15 predictoras (1.er evento)\nopened_at, resolution_hours\nsla_breach (objetivo)"),
    "RUN": (10.0, 4.6, "EJECUCION_MLFLOW\nPK: run_id\nparámetros, métricas,\ntag quality_gate"),
    "MODELO": (10.0, 1.6, "VERSION_MODELO\nPK: version (run_id[:8])\nmodel.joblib, umbral,\nmétricas de validación"),
    "PERFIL": (6.0, 1.6, "PERFIL_REFERENCIA\nPK: version + variable\ntipo, frecuencias o\nmuestra de valores"),
    "DRIFT": (2.0, 1.6, "REPORTE_DRIFT\nPK: fecha + variable\npsi, nivel, ks, ks_p"),
}
for k, (x, y, t) in ents.items():
    box(ax, x, y, 3.3, 1.7, t, fs=7.5)
rels = [((3.65, 4.6), (4.35, 4.6), "N : 1"), ((7.65, 4.6), (8.35, 4.6), "N : M\n(entrena)"),
        ((10.0, 3.75), (10.0, 2.45), "1 : 0..1\n(si aprueba)"), ((8.35, 1.6), (7.65, 1.6), "1 : N"),
        ((4.35, 1.6), (3.65, 1.6), "1 : N")]
for p1, p2, t in rels:
    arrow(ax, p1, p2, style="-")
    ax.text((p1[0] + p2[0]) / 2 + (0.35 if p1[0] == p2[0] else 0), (p1[1] + p2[1]) / 2 + 0.25, t,
            fontsize=7, ha="center", color=RED)
ax.set_title("Modelo entidad-relación de los datos del prototipo")
save(fig, "fig_19_der.png")
print("ok")
