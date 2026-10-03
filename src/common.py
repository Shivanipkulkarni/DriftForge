"""
common.py - shared helpers used by every experiment script (steps 2-5).
Rule: nothing in here may look at validation/test labels when fitting or choosing anything.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA = Path("data/interim/secom_clean.csv")
CFG = Path("configs/folds.json")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]  # predictions are averaged over these seeds


def load():
    """Returns (df, feature_names, folds_config). df is time-ordered and NOT imputed."""
    df = pd.read_csv(DATA, parse_dates=["time"])
    cfg = json.loads(CFG.read_text())
    feats = [c for c in df.columns if c not in ("row_id", "time", "target")]
    return df, feats, cfg


def make_model(name, y_train, seed):
    """Every model starts with median imputation fitted on the TRAINING rows only (pipeline)."""
    imp = SimpleImputer(strategy="median")
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    if name == "xgb":
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=neg
            / max(pos, 1),  # class imbalance handled per training set
            random_state=seed,
            n_jobs=-1,
        )
        return make_pipeline(imp, clf)
    if name == "logreg":
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        return make_pipeline(imp, StandardScaler(), clf)
    if name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        return make_pipeline(imp, clf)
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xte):
    seeds = [0] if name == "logreg" else SEEDS  # logistic regression is deterministic
    ps = []
    for s in seeds:
        m = make_model(name, ytr, s)
        m.fit(Xtr, ytr)
        ps.append(m.predict_proba(Xte)[:, 1])
    return np.mean(ps, axis=0)


def pick_threshold(name, Xtr, ytr, inner_frac=0.6):
    """
    Decision threshold chosen from TRAINING data only, out-of-time:
    fit on the first 60% of the training rows, score the last 40%, take the F1-best score.
    """
    k = int(len(Xtr) * inner_frac)
    p = fit_predict(name, Xtr.iloc[:k], ytr.iloc[:k], Xtr.iloc[k:])
    y_in = ytr.iloc[k:].values
    if y_in.sum() == 0:
        return float(
            np.quantile(p, 1 - max(ytr.mean(), 0.01))
        )  # fallback: flag top-prevalence share
    cands = np.unique(p)
    scores = [f1_score(y_in, p >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(scores))])


def metrics(y, p, thr):
    y = np.asarray(y)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prev = y.mean()
    ap = average_precision_score(y, p)
    return dict(
        n=len(y),
        fails=int(y.sum()),
        prevalence=prev,
        pr_auc=ap,
        pr_lift=ap / prev,
        roc_auc=roc_auc_score(y, p),
        thr=thr,
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        bal_acc=balanced_accuracy_score(y, yhat),
        fnr=fn / max(fn + tp, 1),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def pooled_threshold_metrics(fold_rows):
    """Micro-average the confusion counts over folds (each fold used its own training-only threshold)."""
    tp, fp, fn, tn = (sum(r[k] for r in fold_rows) for k in ("tp", "fp", "fn", "tn"))
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    spec = tn / max(tn + fp, 1)
    return dict(
        precision=prec,
        recall=rec,
        f1=2 * prec * rec / max(prec + rec, 1e-12),
        bal_acc=(rec + spec) / 2,
        fnr=1 - rec,
    )


def bootstrap_macro_ci(preds, metric="ap", n_boot=1000, seed=0):
    """
    95% CI of the fold-averaged metric. preds = [(y_fold1, p_fold1), ...].
    Rows are resampled inside each fold; resamples with no failure are skipped.
    """
    fn = average_precision_score if metric == "ap" else roc_auc_score
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        vals = []
        for y, p in preds:
            idx = rng.integers(0, len(y), len(y))
            if 0 < y[idx].sum() < len(idx):
                vals.append(fn(y[idx], p[idx]))
        if vals:
            out.append(np.mean(vals))
    return tuple(np.percentile(out, [2.5, 97.5]))
