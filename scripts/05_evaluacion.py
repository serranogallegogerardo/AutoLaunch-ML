"""Evaluación del modelo seleccionado: curvas, umbral operativo, priorización e importancia."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
from plots import BLUE, plt, save  # noqa: E402

RES = P.ROOT / "results"
inc = pd.read_csv(RES / "incidentes_preparados.csv", parse_dates=["opened_at"])
train, test = P.temporal_split(inc)
Xtr, ytr, Xte, yte = train[P.FEATURES], train[P.TARGET], test[P.FEATURES], test[P.TARGET]
probs = pd.read_csv(RES / "probabilidades_test.csv")
sel = joblib.load(RES / "modelo_seleccionado.joblib")
best, model = sel["name"], sel["model"]
pb = probs[best].values

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for name in [c for c in probs.columns if c not in ("number", "y", "opened_at")]:
    f, t, _ = roc_curve(yte, probs[name]); axes[0].plot(f, t, label=f"{name} ({roc_auc_score(yte, probs[name]):.3f})")
    pr, rc, _ = precision_recall_curve(yte, probs[name]); axes[1].plot(rc, pr, label=name)
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].legend(fontsize=7, title="ROC-AUC", title_fontsize=7)
axes[0].set_xlabel("Tasa de falsos positivos"); axes[0].set_ylabel("Tasa de verdaderos positivos")
axes[0].set_title("Curva ROC (prueba temporal)")
axes[1].axhline(yte.mean(), color="grey", ls="--", lw=0.8)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisión"); axes[1].set_title("Curva precisión-recall")
save(fig, "fig_07_roc_pr.png")

# Umbral: máximo F1 en validación temporal interna (abril 2016), sin usar el periodo de prueba
val = train["opened_at"] >= pd.Timestamp("2016-04-01")
mv = clone(model).fit(Xtr[~val], ytr[~val])
pp, rr, tt = precision_recall_curve(ytr[val], mv.predict_proba(Xtr[val])[:, 1])
f1s = 2 * pp * rr / np.clip(pp + rr, 1e-9, None)
thr = float(tt[np.nanargmax(f1s[:-1])])
yb = (pb >= thr).astype(int)
cm = confusion_matrix(yte, yb)
fig, ax = plt.subplots(figsize=(4.4, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Cumple", "Incumple"], yticklabels=["Cumple", "Incumple"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real"); ax.set_title(f"Matriz de confusión (umbral {thr:.2f})")
save(fig, "fig_08_matriz_confusion.png")

k = int(0.2 * len(pb)); top = np.argsort(-pb)[:k]
prec_top = float(yte.values[top].mean())

sub = Xte.sample(n=3000, random_state=P.SEED)
pi = permutation_importance(model, sub, yte.loc[sub.index], scoring="roc_auc", n_repeats=5, random_state=P.SEED)
imp = pd.Series(pi.importances_mean, index=P.FEATURES).sort_values()
imp.to_csv(RES / "importancia_permutacion.csv")
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color=BLUE); ax.set_xlabel("Caída media de ROC-AUC al permutar")
ax.set_title(f"Importancia por permutación ({best})"); save(fig, "fig_09_importancia.png")

t2 = test.assign(p=pb)
mensual = [{"mes": str(m), "n": int(len(g)), "tasa": round(float(g[P.TARGET].mean()), 3),
            "roc_auc": round(roc_auc_score(g[P.TARGET], g["p"]), 4)}
           for m, g in t2.groupby(t2["opened_at"].dt.to_period("M")) if len(g) >= 50 and g[P.TARGET].nunique() == 2]
out = {"modelo": best,
       "umbral": {"criterio": "máximo F1 en validación temporal (abril 2016)", "valor": round(thr, 3),
                  "precision": round(precision_score(yte, yb), 4), "recall": round(recall_score(yte, yb), 4),
                  "f1": round(f1_score(yte, yb), 4), "matriz_confusion": cm.tolist()},
       "top20": {"revisados": k, "precision": round(prec_top, 4),
                 "incumplimientos_capturados": round(float(yte.values[top].sum() / yte.sum()), 4),
                 "lift": round(prec_top / float(yte.mean()), 2)},
       "importancia": {k_: round(float(v), 4) for k_, v in imp.sort_values(ascending=False).items()},
       "desempeno_mensual": mensual}
(RES / "evaluacion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))
