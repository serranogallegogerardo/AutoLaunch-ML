"""Modelado y evaluación: tres algoritmos, validación cruzada en train y test temporal."""
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
from modelos import candidatos  # noqa: E402

RES = P.ROOT / "results"
inc = pd.read_csv(RES / "incidentes_preparados.csv", parse_dates=["opened_at"])
train, test = P.temporal_split(inc)
Xtr, ytr, Xte, yte = train[P.FEATURES], train[P.TARGET], test[P.FEATURES], test[P.TARGET]
MODELS = candidatos()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
rows = []
probs = pd.DataFrame({"number": test["number"].values, "y": yte.values, "opened_at": test["opened_at"].values})
for name, mdl in MODELS.items():
    t0 = time.time()
    cv = cross_val_score(mdl, Xtr, ytr, cv=skf, scoring="roc_auc")
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    yhat = (p >= 0.5).astype(int)
    rows.append({"modelo": name, "cv_roc_auc": round(cv.mean(), 4), "cv_sd": round(cv.std(), 4),
                 "test_roc_auc": round(roc_auc_score(yte, p), 4),
                 "test_pr_auc": round(average_precision_score(yte, p), 4),
                 "test_f1_0.5": round(f1_score(yte, yhat), 4),
                 "test_precision_0.5": round(precision_score(yte, yhat), 4),
                 "test_recall_0.5": round(recall_score(yte, yhat), 4),
                 "test_brier": round(brier_score_loss(yte, p), 4),
                 "segundos": round(time.time() - t0, 1)})
    probs[name] = p
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "comparacion_modelos.csv", index=False)
probs.to_csv(RES / "probabilidades_test.csv", index=False)
best = res.sort_values("cv_roc_auc", ascending=False).iloc[0]["modelo"]
joblib.dump({"name": best, "model": MODELS[best]}, RES / "modelo_seleccionado.joblib")
out = {"particion": {"corte": str(P.CUTOFF.date()), "train_n": int(len(train)), "test_n": int(len(test)),
                     "train_tasa": round(float(ytr.mean()), 4), "test_tasa": round(float(yte.mean()), 4)},
       "modelos": rows, "seleccionado": best,
       "criterio": "mayor ROC-AUC media en validación cruzada (train)",
       "pr_auc_azar": round(float(yte.mean()), 4)}
(RES / "modelado.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
