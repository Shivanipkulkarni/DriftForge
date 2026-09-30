# src/lib.py  -- shared code for Steps 5+ (same logic as Steps 3-4)
import json, warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, false_discovery_control
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEEDS = [0, 1, 2, 3, 4]
KS_MIN, Q_MAX = 0.30, 0.05          # a feature "drifts" if KS > 0.30 and FDR q < 0.05
ALERT = 0.10                        # alert budget: flag the top 10% riskiest wafers per block
MODEL_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def load_all():
    df = pd.read_csv("data/interim/secom_clean.csv")
    cfg = json.load(open("configs/folds.json"))
    sel = json.load(open("results/selected_config.json"))
    feats = [c for c in df.columns if c not in ("row_id", "time", "target")]
    return df, cfg, sel, feats


def xgb_model(ytr, seed):
    pos = int(ytr.sum()); neg = len(ytr) - pos
    return XGBClassifier(**MODEL_PARAMS, objective="binary:logistic", eval_metric="aucpr",
                         importance_type="gain", scale_pos_weight=neg / max(pos, 1),
                         random_state=seed, n_jobs=-1)


def score_features(Xtr_raw, ytr):
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(Xtr_raw), columns=Xtr_raw.columns, index=Xtr_raw.index)
    mi = np.mean([mutual_info_classif(Xi, ytr, random_state=s) for s in SEEDS[:3]], axis=0)
    gain = np.mean([xgb_model(ytr, s).fit(Xi, ytr).feature_importances_ for s in SEEDS], axis=0)
    n = len(Xtr_raw)
    a, b = Xtr_raw.iloc[: n // 3], Xtr_raw.iloc[-(n // 3):]
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
    if strategy == "drift_aware":
        score = score[~((s.q < Q_MAX) & (s.ks > KS_MIN))]
    kk = len(score) if k is None else min(k, len(score))
    return score.sort_values(ascending=False).index[:kk].tolist()


class Deployer:
    """Trains (and caches) models on row windows, predicts, and measures label-free drift."""
    def __init__(self, X, y, sel):
        self.X, self.y, self.Xv = X, y, X.values
        self.strategy = sel["strategy"]
        self.k = None if sel["k"] == 0 else sel["k"]
        self.models, self.dcache = {}, {}

    def train(self, start, end):
        key = (start, end)
        if key not in self.models:
            Xtr, ytr = self.X.iloc[start:end], self.y.iloc[start:end]
            cols = select(self.strategy, score_features(Xtr, ytr), self.k)
            imp = SimpleImputer(strategy="median").fit(Xtr[cols])
            A = imp.transform(Xtr[cols])
            self.models[key] = dict(cols=cols, imp=imp, window=key,
                                    models=[xgb_model(ytr, s).fit(A, ytr) for s in SEEDS])
        return self.models[key]

    def predict(self, bundle, lo, hi):
        B = bundle["imp"].transform(self.X.iloc[lo:hi][bundle["cols"]])
        return np.mean([m.predict_proba(B)[:, 1] for m in bundle["models"]], axis=0)

    def drift_share(self, ref, new):
        """Share of ALL process features with KS > KS_MIN and FDR q < Q_MAX between two row ranges."""
        if (ref, new) not in self.dcache:
            ks, pv = [], []
            for j in range(self.Xv.shape[1]):
                u, v = self.Xv[ref[0]:ref[1], j], self.Xv[new[0]:new[1], j]
                u, v = u[~np.isnan(u)], v[~np.isnan(v)]
                if len(u) < 20 or len(v) < 20:
                    ks.append(0.0); pv.append(1.0)
                else:
                    r = ks_2samp(u, v); ks.append(r.statistic); pv.append(r.pvalue)
            q = false_discovery_control(np.array(pv), method="bh")
            self.dcache[(ref, new)] = float(np.mean((q < Q_MAX) & (np.array(ks) > KS_MIN)))
        return self.dcache[(ref, new)]


def alert_metrics(yv, p, sizes):
    yhat, i = np.zeros(len(yv), int), 0
    for sz in sizes:
        k = max(1, int(round(ALERT * sz)))
        yhat[i + np.argsort(-p[i:i + sz])[:k]] = 1
        i += sz
    tn, fp, fn, tp = confusion_matrix(yv, yhat, labels=[0, 1]).ravel()
    rec, prec, spec = tp / max(tp + fn, 1), tp / max(tp + fp, 1), tn / max(tn + fp, 1)
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                bal_acc=(rec + spec) / 2, fnr=1 - rec)


def paired_ci(yv, p, p_ref, n_boot=1000, chunk=20, seed=0):
    """Paired chunk bootstrap of the AP difference (p - p_ref)."""
    rng, n = np.random.default_rng(seed), len(yv)
    starts, diffs = np.arange(0, n, chunk), []
    for _ in range(n_boot):
        pick = rng.integers(0, len(starts), len(starts))
        idx = np.concatenate([np.arange(starts[i], min(starts[i] + chunk, n)) for i in pick])
        if 0 < yv[idx].sum() < len(idx):
            diffs.append(average_precision_score(yv[idx], p[idx]) - average_precision_score(yv[idx], p_ref[idx]))
    return np.percentile(diffs, [2.5, 97.5])