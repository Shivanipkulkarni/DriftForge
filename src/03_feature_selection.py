# src/03_feature_selection.py
# Self-contained. Needs data/interim/secom_clean.csv and configs/folds.json from Step 1.
# Compares feature-selection strategies INSIDE each temporal fold (training rows only).
import json, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, false_discovery_control
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEEDS = [0, 1, 2, 3, 4]
SIZES = [20, 40, 60, 100]
STRATS = ["mi", "xgb", "consensus", "drift_aware"]
DRIFT_KS_MAX = 0.30   # PRE-SPECIFIED: drop a feature if its within-training drift has KS > 0.30 and FDR q < 0.05

df = pd.read_csv("data/interim/secom_clean.csv")
cfg = json.load(open("configs/folds.json"))
feats = [c for c in df.columns if c not in ("row_id", "time", "target")]
X, y = df[feats], df["target"]


def xgb_model(ytr, seed):
    pos = int(ytr.sum()); neg = len(ytr) - pos
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
                         importance_type="gain", scale_pos_weight=neg / max(pos, 1),
                         random_state=seed, n_jobs=-1)


def score_features(Xtr_raw, ytr):
    """All three signals are computed from the TRAINING rows of one fold only."""
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(Xtr_raw), columns=Xtr_raw.columns, index=Xtr_raw.index)
    mi = np.mean([mutual_info_classif(Xi, ytr, random_state=s) for s in SEEDS[:3]], axis=0)
    gain = np.mean([xgb_model(ytr, s).fit(Xi, ytr).feature_importances_ for s in SEEDS], axis=0)
    n = len(Xtr_raw)
    a, b = Xtr_raw.iloc[: n // 3], Xtr_raw.iloc[-(n // 3):]      # earliest vs latest third of training
    ks, pv = [], []
    for c in Xtr_raw.columns:
        u, v = a[c].dropna(), b[c].dropna()
        if len(u) < 20 or len(v) < 20:
            ks.append(0.0); pv.append(1.0)
        else:
            r = ks_2samp(u, v); ks.append(r.statistic); pv.append(r.pvalue)
    q = false_discovery_control(np.array(pv), method="bh")
    return pd.DataFrame({"mi": mi, "gain": gain, "ks": ks, "q": q}, index=Xtr_raw.columns)


def select(strategy, sc, k):
    s = sc.copy()
    r_mi, r_xgb = s.mi.rank(ascending=False), s.gain.rank(ascending=False)
    score = -r_mi if strategy == "mi" else -r_xgb if strategy == "xgb" else -(r_mi + r_xgb) / 2
    if strategy == "drift_aware":                       # filter unstable features, then mean-rank
        keep = ~((s.q < 0.05) & (s.ks > DRIFT_KS_MAX))
        score = score[keep]
    kk = len(score) if k is None else min(k, len(score))
    return score.sort_values(ascending=False).index[:kk].tolist()


def evaluate(cols, Xtr, ytr, Xva, yva):
    imp = SimpleImputer(strategy="median").fit(Xtr[cols])
    A, B = imp.transform(Xtr[cols]), imp.transform(Xva[cols])
    p = np.mean([xgb_model(ytr, s).fit(A, ytr).predict_proba(B)[:, 1] for s in SEEDS], axis=0)
    ap = average_precision_score(yva, p)
    return ap, ap / yva.mean(), roc_auc_score(yva, p)


CONFIGS = [("all_features", "consensus", None), ("drift_filter_only", "drift_aware", None)]
CONFIGS += [(f"{s}_{k}", s, k) for s in STRATS for k in SIZES]

rows, sel = [], {}
for f in cfg["folds"]:
    tr, va = slice(*f["train"]), slice(*f["val"])
    Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
    sc = score_features(Xtr, ytr)
    n_drop = int(((sc.q < 0.05) & (sc.ks > DRIFT_KS_MAX)).sum())
    print(f"Fold {f['fold']}: {n_drop} of {len(sc)} features flagged as unstable inside training")
    for name, strat, k in CONFIGS:
        cols = select(strat, sc, k)
        ap, lift, roc = evaluate(cols, Xtr, ytr, Xva, yva)
        sel.setdefault(name, {})[f["fold"]] = set(cols)
        rows.append(dict(fold=f["fold"], config=name, strategy=strat, k=k or 0, n_sel=len(cols),
                         pr_auc=ap, lift=lift, roc_auc=roc))
    print(f"  fold {f['fold']} done")

res = pd.DataFrame(rows)
base = res[res.config == "all_features"].set_index("fold")["pr_auc"]
res["beats_all"] = res.apply(lambda r: r.pr_auc > base[r.fold], axis=1)
agg = res.groupby("config").agg(strategy=("strategy", "first"), k=("k", "first"),
                                n_sel=("n_sel", "mean"), pr_auc=("pr_auc", "mean"),
                                pr_sd=("pr_auc", "std"), lift=("lift", "mean"),
                                roc_auc=("roc_auc", "mean"), folds_beating_all=("beats_all", "sum")).reset_index()
fids = sorted(fd["fold"] for fd in cfg["folds"])
jac = lambda a, b: len(a & b) / len(a | b)
agg["stability"] = agg.config.map(lambda c: np.mean([jac(sel[c][i], sel[c][j]) for i, j in zip(fids, fids[1:])]))

# PRE-SPECIFIED choice rule: best mean PR-AUC; among configs within 1 SE of it, take the fewest features.
best = agg.loc[agg.pr_auc.idxmax()]
se = best.pr_sd / np.sqrt(len(fids))
cand = agg[agg.pr_auc >= best.pr_auc - se]
choice = cand.sort_values(["n_sel", "pr_auc"], ascending=[True, False]).iloc[0]

pd.set_option("display.width", 200)
print("\n=== Feature-selection comparison (mean over folds) ===")
print(agg.sort_values("pr_auc", ascending=False).round(3).to_string(index=False))
print(f"\nBest mean PR-AUC: {best.config} ({best.pr_auc:.3f}); 1-SE = {se:.3f}")
print(f"CHOSEN by rule: {choice.config}  (strategy={choice.strategy}, k={int(choice.k)}, n_features~{choice.n_sel:.0f})")

os.makedirs("results", exist_ok=True)
res.to_csv("results/featsel_folds.csv", index=False)
agg.to_csv("results/featsel_summary.csv", index=False)
json.dump({"config": choice.config, "strategy": choice.strategy, "k": int(choice.k)},
          open("results/selected_config.json", "w"))