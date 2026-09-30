"""
02_static_baseline.py
Static baselines (train once on the past, never updated) on the frozen expanding-window folds.
All features, no selection yet. The FINAL test period is NOT touched here.

Usage: python src/02_static_baseline.py
"""
import numpy as np
import pandas as pd

from common import (RESULTS, bootstrap_macro_ci, fit_predict, load, metrics,
                    pick_threshold, pooled_threshold_metrics)

df, feats, cfg = load()
X, y = df[feats], df["target"]
MODELS = ["xgb", "logreg", "rf"]

fold_rows, oof_rows, summary = [], [], []

for model in MODELS:
    per_fold, preds = [], []
    for f in cfg["folds"]:
        tr, va = slice(*f["train"]), slice(*f["val"])
        Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]

        thr = pick_threshold(model, Xtr, ytr)      # training data only
        p = fit_predict(model, Xtr, ytr, Xva)      # validation labels never used for fitting
        r = metrics(yva.values, p, thr)
        r.update(model=model, fold=f["fold"], n_features=len(feats))
        per_fold.append(r)
        preds.append((yva.values, p))
        oof_rows.append(pd.DataFrame({"model": model, "fold": f["fold"],
                                      "row_id": df["row_id"].iloc[va].values,
                                      "y": yva.values, "score": p, "thr": thr}))
        print(f"{model:<7} fold {f['fold']}: PR-AUC={r['pr_auc']:.3f} "
              f"(prevalence {r['prevalence']:.3f})  ROC-AUC={r['roc_auc']:.3f}  "
              f"recall={r['recall']:.2f}  precision={r['precision']:.2f}")
    fold_rows += per_fold

    ap_lo, ap_hi = bootstrap_macro_ci(preds, "ap")
    roc_lo, roc_hi = bootstrap_macro_ci(preds, "roc")
    pooled = pooled_threshold_metrics(per_fold)
    summary.append(dict(
        model=model,
        pr_auc_mean=np.mean([r["pr_auc"] for r in per_fold]), pr_auc_ci=f"[{ap_lo:.3f}, {ap_hi:.3f}]",
        pr_lift_mean=np.mean([r["pr_lift"] for r in per_fold]),
        roc_auc_mean=np.mean([r["roc_auc"] for r in per_fold]), roc_auc_ci=f"[{roc_lo:.3f}, {roc_hi:.3f}]",
        **pooled))

# reference: a random scorer has PR-AUC = prevalence and ROC-AUC = 0.5
fr = pd.DataFrame(fold_rows)
prev = fr.drop_duplicates("fold")["prevalence"].mean()
summary.append(dict(model="random (reference)", pr_auc_mean=prev, pr_lift_mean=1.0, roc_auc_mean=0.5))

fr.to_csv(RESULTS / "static_baseline_folds.csv", index=False)
pd.concat(oof_rows).to_csv(RESULTS / "static_baseline_oof_scores.csv", index=False)
sm = pd.DataFrame(summary)
sm.to_csv(RESULTS / "static_baseline_summary.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Per-fold results ===")
print(fr[["model", "fold", "n", "fails", "prevalence", "pr_auc", "pr_lift", "roc_auc",
          "precision", "recall", "f1", "bal_acc", "fnr"]].round(3).to_string(index=False))
print("\n=== Summary over folds (threshold metrics pooled) ===")
print(sm.round(3).to_string(index=False))
print("\nSaved results/static_baseline_*.csv")
