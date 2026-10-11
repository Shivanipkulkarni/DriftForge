"""
10_paper_protocol_comparison.py
Same models, evaluated the way many SECOM papers do it: a shuffled, stratified 80/20 split.

THIS IS NOT THE PROJECT'S PRIMARY PROTOCOL. It exists to show how much a random split
flatters the same models compared with the strict forward-in-time evaluation in Steps 2-5 and 9.
Shuffling lets wafers from the same production period land in both train and test, so drift
cannot hurt the model.

Only leakage that the shuffled protocol itself implies is present; nothing else is added:
no feature selection and no oversampling, and the imputer is fitted on the training rows only.

Needs results/pso_best.json (run Step 9 first). Usage, from the repository root:
    python src/10_paper_protocol_comparison.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

REPEATS = 5
RES = "results"

df = pd.read_csv("data/interim/secom_clean.csv")
feats = [c for c in df.columns if c not in ("row_id", "time", "target")]
X, y = df[feats], df["target"].values
frozen = json.load(open(f"{RES}/pso_best.json"))
configs = {m: frozen[m]["params"] for m in ("default", "random", "pso")}


def fit_predict(params, Xtr, ytr, Xte, seed):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    pos = int(ytr.sum())
    clf = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=(len(ytr) - pos) / max(pos, 1),
        random_state=seed,
        n_jobs=-1,
    ).fit(imp.transform(Xtr), ytr)
    return clf.predict_proba(imp.transform(Xte))[:, 1]


rows = []
splitter = StratifiedShuffleSplit(n_splits=REPEATS, test_size=0.2, random_state=0)
for rep, (tr, te) in enumerate(splitter.split(X, y)):
    for m, params in configs.items():
        p = fit_predict(params, X.iloc[tr], y[tr], X.iloc[te], seed=rep)
        yhat = (p >= 0.5).astype(int)
        rows.append(
            dict(
                method=m,
                repeat=rep,
                accuracy=accuracy_score(y[te], yhat),
                precision=precision_score(y[te], yhat, zero_division=0),
                recall=recall_score(y[te], yhat, zero_division=0),
                f1=f1_score(y[te], yhat, zero_division=0),
                roc_auc=roc_auc_score(y[te], p),
                pr_auc=average_precision_score(y[te], p),
            )
        )
    print(f"repeat {rep + 1}/{REPEATS} done")

res = pd.DataFrame(rows)
res.to_csv(f"{RES}/paper_protocol_folds.csv", index=False)
cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
summary = res.groupby("method")[cols].agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()
summary.to_csv(f"{RES}/paper_protocol_summary.csv", index=False)

pd.set_option("display.width", 220)
print(f"\n=== Shuffled stratified 80/20 split, {REPEATS} repeats (NOT leakage-safe) ===")
print(summary.round(3).to_string(index=False))

# Side by side with the strict temporal protocol (only if the final test has been run)
fin = f"{RES}/pso_final_summary.csv"
if os.path.exists(fin):
    t = pd.read_csv(fin)[["method", "pr_auc", "roc_auc"]]
    s = summary[["method", "pr_auc_mean", "roc_auc_mean"]]
    cmp_ = s.merge(t, on="method", suffixes=("_shuffled", "_temporal_final"))
    print("\n=== Same models: shuffled split vs strict temporal final test ===")
    print(cmp_.round(3).to_string(index=False))
    cmp_.to_csv(f"{RES}/protocol_gap.csv", index=False)
else:
    print("\n(Run `python src/09_pso_xgboost.py --final` to add the temporal comparison.)")
