"""EDA (2/2): asociaciones (V de Cramér) y resumen numérico en results/eda.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
from plots import plt, save  # noqa: E402

RES = P.ROOT / "results"
inc = pd.read_csv(RES / "incidentes_preparados.csv", parse_dates=["opened_at"])


def cramers_v(x, y):
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt((chi2 / t.values.sum()) / max(1, min(t.shape) - 1)))


cols = ["priority", "impact", "urgency", "category", "contact_type", "assignment_group",
        "u_symptom", "knowledge", P.TARGET]
S = inc[cols].astype(str)
cv = pd.DataFrame([[1.0 if a == b else cramers_v(S[a], S[b]) for b in cols] for a in cols],
                  index=cols, columns=cols)
cv.to_csv(RES / "cramers_v.csv")
fig, ax = plt.subplots(figsize=(6.8, 5.4))
sns.heatmap(cv, annot=True, fmt=".2f", cmap="Blues", ax=ax, annot_kws={"size": 8})
ax.set_title("Asociación entre variables (V de Cramér)"); save(fig, "fig_06_cramers_v.png")

pr = inc.groupby("priority")[P.TARGET].agg(["mean", "size"])
hr = inc.groupby("opened_hour")[P.TARGET].mean()
top = inc["category"].value_counts().index[:10]
cr = inc[inc["category"].isin(top)].groupby("category")[P.TARGET].agg(["mean", "size"])
out = {
    "incumplimiento_por_prioridad": {k: [int(n), round(float(v), 3)] for k, v, n in zip(pr.index, pr["mean"], pr["size"])},
    "cramers_v_con_objetivo": {k: round(float(v), 3) for k, v in cv[P.TARGET].drop(P.TARGET).items()},
    "priority_vs_impact": round(float(cv.loc["priority", "impact"]), 3),
    "priority_vs_urgency": round(float(cv.loc["priority", "urgency"]), 3),
    "hora_min": [int(hr.idxmin()), round(float(hr.min()), 3)],
    "hora_max": [int(hr.idxmax()), round(float(hr.max()), 3)],
    "top10_categorias": {k: [int(n), round(float(v), 3)] for k, v, n in zip(cr.index, cr["mean"], cr["size"])},
    "resolucion_mediana_por_clase": {str(k): round(float(v), 1)
                                     for k, v in inc.groupby(P.TARGET)["resolution_hours"].median().items()},
}
(RES / "eda.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
