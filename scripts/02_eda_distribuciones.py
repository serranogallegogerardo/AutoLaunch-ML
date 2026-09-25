"""EDA (1/2): volumen, completitud, incumplimiento por prioridad, categoría y hora."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
from plots import BLUE, plt, save  # noqa: E402

RES = P.ROOT / "results"
inc = pd.read_csv(RES / "incidentes_preparados.csv", parse_dates=["opened_at"])
miss = pd.read_csv(RES / "faltantes_eventos.csv", index_col=0)["proporcion_faltante"]
monthly = pd.read_csv(RES / "volumen_mensual.csv", index_col=0, parse_dates=True)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.bar(monthly.index.strftime("%Y-%m"), monthly["size"], color=BLUE)
ax.set_ylabel("Incidentes abiertos"); ax.set_title("Volumen mensual de incidentes")
plt.xticks(rotation=45, ha="right"); save(fig, "fig_01_volumen_mensual.png")

fig, ax = plt.subplots(figsize=(7.5, 4))
mm = miss[miss > 0]
ax.barh(mm.index[::-1], 100 * mm.values[::-1], color=BLUE)
ax.set_xlabel("% de eventos con valor faltante ('?')")
ax.set_title("Completitud de atributos del registro de eventos"); save(fig, "fig_02_faltantes.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
pr = inc.groupby("priority")[P.TARGET].agg(["mean", "size"])
axes[0].bar(pr.index, 100 * pr["mean"], color=BLUE)
for i, (v, n) in enumerate(zip(pr["mean"], pr["size"])):
    axes[0].text(i, 100 * v + 1, f"n={n}", ha="center", fontsize=8)
axes[0].set_ylabel("% incumplimiento SLA"); axes[0].set_title("Por prioridad")
axes[0].tick_params(axis="x", rotation=20)
top = inc["category"].value_counts().index[:10]
cr = inc[inc["category"].isin(top)].groupby("category")[P.TARGET].mean().sort_values()
axes[1].barh(cr.index, 100 * cr.values, color=BLUE)
axes[1].axvline(100 * inc[P.TARGET].mean(), color="grey", ls="--", lw=1)
axes[1].set_xlabel("% incumplimiento SLA"); axes[1].set_title("10 categorías más frecuentes")
save(fig, "fig_03_incumplimiento_prioridad_categoria.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
lr = np.log10(inc["resolution_hours"].clip(lower=0.01))
axes[0].hist(lr.dropna(), bins=50, color=BLUE)
axes[0].set_xlabel("log10(horas hasta resolución)"); axes[0].set_ylabel("Incidentes")
axes[0].set_title("Tiempo de resolución")
sns.boxplot(x=inc[P.TARGET].map({0: "Cumple SLA", 1: "Incumple SLA"}), y=lr, ax=axes[1], color="#8EA9DB")
axes[1].set_xlabel(""); axes[1].set_ylabel("log10(horas)"); axes[1].set_title("Resolución según SLA")
save(fig, "fig_04_tiempo_resolucion.png")

fig, ax = plt.subplots(figsize=(7.5, 3.2))
hr = inc.groupby("opened_hour")[P.TARGET].mean()
ax.plot(hr.index, 100 * hr.values, marker="o", color=BLUE)
ax.set_xlabel("Hora de apertura"); ax.set_ylabel("% incumplimiento SLA")
ax.set_title("Incumplimiento según hora de apertura"); ax.set_xticks(range(0, 24, 2))
save(fig, "fig_05_hora_apertura.png")
