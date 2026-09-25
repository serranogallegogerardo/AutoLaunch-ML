"""Proceso AS-IS de la mesa de servicio reconstruido desde el registro de eventos (process mining básico)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
from plots import plt, save  # noqa: E402

RES = P.ROOT / "results"
ev = P.load_events().sort_values(["number", "sys_mod_count"])
ev = ev[ev["incident_state"] != "-100"]

seq = ev.groupby("number")["incident_state"].apply(lambda s: [x for i, x in enumerate(s) if i == 0 or x != s.iloc[i - 1]])
pairs = [(a, b) for states in seq for a, b in zip(states[:-1], states[1:])]
tr = pd.Series(pairs).value_counts()
tr.index = [f"{a} -> {b}" for a, b in tr.index]
tr.to_csv(RES / "transiciones_estados.csv", header=["n"])

inc = pd.read_csv(RES / "incidentes_preparados.csv")
first_state = seq.apply(lambda s: s[0]).value_counts()
variants = seq.apply(lambda s: " > ".join(s)).value_counts()
awaiting = seq.apply(lambda s: any(x.startswith("Awaiting") for x in s))
reasig = inc.set_index("number")["reassignment_count"]
out = {
    "incidentes": int(len(seq)),
    "estado_inicial": {k: int(v) for k, v in first_state.items()},
    "variantes_distintas": int(len(variants)),
    "top5_variantes": {k: int(v) for k, v in variants.head(5).items()},
    "pct_con_espera": round(float(awaiting.mean()), 4),
    "pct_con_reasignacion": round(float((reasig > 0).mean()), 4),
    "incumplimiento_con_espera": round(float(inc.set_index("number").loc[awaiting.index[awaiting], P.TARGET].mean()), 4),
    "incumplimiento_sin_espera": round(float(inc.set_index("number").loc[awaiting.index[~awaiting], P.TARGET].mean()), 4),
    "incumplimiento_con_reasignacion": round(float(inc.loc[inc.reassignment_count > 0, P.TARGET].mean()), 4),
    "incumplimiento_sin_reasignacion": round(float(inc.loc[inc.reassignment_count == 0, P.TARGET].mean()), 4),
    "top_transiciones": {k: int(v) for k, v in tr.head(12).items()},
}
(RES / "proceso_asis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))

# Diagrama del proceso con las transiciones más frecuentes
pos = {"New": (0, 2), "Active": (2.4, 2), "Awaiting User Info": (1.2, 3.8), "Awaiting Vendor": (3.6, 3.8),
       "Awaiting Problem": (4.8, 0.3), "Resolved": (4.8, 2), "Closed": (7.2, 2)}
fig, ax = plt.subplots(figsize=(9, 4.6))
tot = tr.sum()
for key, n in tr.items():
    a, b = key.split(" -> ")
    if a not in pos or b not in pos or n < 150:
        continue
    (x1, y1), (x2, y2) = pos[a], pos[b]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.6 + 4 * n / tr.max(), color="#44546A",
                                shrinkA=28, shrinkB=28, connectionstyle="arc3,rad=0.12"))
    dx, dy = x2 - x1, y2 - y1
    norm = (dx ** 2 + dy ** 2) ** 0.5
    off = 0.08 * norm + 0.12
    lx, ly = (x1 + x2) / 2 + off * dy / norm, (y1 + y2) / 2 - off * dx / norm
    ax.text(lx, ly, f"{n:,}".replace(",", "."), fontsize=7, color="#C00000", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
for s, (x, y) in pos.items():
    ax.text(x, y, s.replace("Awaiting ", "Espera:\n"), ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.5", fc="#DDEBF7", ec="#002060"))
ax.set_xlim(-0.8, 8); ax.set_ylim(-0.3, 4.3); ax.axis("off")
ax.set_title("Proceso AS-IS de gestión de incidentes (transiciones con n ≥ 150)")
save(fig, "fig_11_proceso_asis.png")
