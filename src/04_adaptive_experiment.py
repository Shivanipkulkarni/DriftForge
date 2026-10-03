# src/04_adaptive_experiment.py
# Self-contained. Needs Step 1 outputs (data/interim/secom_clean.csv, configs/folds.json)
# and Step 3's results/selected_config.json.
# Simulates deployment on the DEVELOPMENT period only (final test untouched).
import json, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, false_discovery_control
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEEDS = [0, 1, 2, 3, 4]
INIT_BLOCKS = (
    3  # initial model trains on the first 3 blocks, deployment starts at block 3
)
LAG = 0  # label delay in blocks (0 = outcomes known before the next block); try 1 later
KS_MIN, Q_MAX = (
    0.30,
    0.05,
)  # a feature "drifts" if KS > 0.30 and FDR q < 0.05 (same rule as Step 3)
TRIGGERS = [
    0.05,
    0.10,
    0.20,
]  # drift-alarm thresholds = share of monitored features drifting
ALERT = 0.10  # fixed alert budget: flag the top 10% highest-risk wafers in each block
MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)

df = pd.read_csv("data/interim/secom_clean.csv")
cfg = json.load(open("configs/folds.json"))
SEL = json.load(open("results/selected_config.json"))
SEL_K = None if SEL["k"] == 0 else SEL["k"]
feats = [c for c in df.columns if c not in ("row_id", "time", "target")]
X, y = df[feats], df["target"]
Xv = X.values
n_dev, n_blocks = cfg["n_dev"], cfg["n_blocks"]
blocks = np.array_split(np.arange(n_dev), n_blocks)
bs = [int(b[0]) for b in blocks] + [n_dev]  # block boundaries (row ids)
INIT_END = bs[INIT_BLOCKS]
BLOCK_ROWS = int(np.median(np.diff(bs)))
print(f"Selected feature config from Step 3: {SEL}")


def xgb_model(ytr, seed):
    pos = int(ytr.sum())
    neg = len(ytr) - pos
    return XGBClassifier(
        **MODEL_PARAMS,
        objective="binary:logistic",
        eval_metric="aucpr",
        importance_type="gain",
        scale_pos_weight=neg / max(pos, 1),
        random_state=seed,
        n_jobs=-1,
    )


def score_features(Xtr_raw, ytr):
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(
        imp.fit_transform(Xtr_raw), columns=Xtr_raw.columns, index=Xtr_raw.index
    )
    mi = np.mean(
        [mutual_info_classif(Xi, ytr, random_state=s) for s in SEEDS[:3]], axis=0
    )
    gain = np.mean(
        [xgb_model(ytr, s).fit(Xi, ytr).feature_importances_ for s in SEEDS], axis=0
    )
    n = len(Xtr_raw)
    a, b = Xtr_raw.iloc[: n // 3], Xtr_raw.iloc[-(n // 3) :]
    ks, pv = [], []
    for c in Xtr_raw.columns:
        u, v = a[c].dropna(), b[c].dropna()
        if len(u) < 20 or len(v) < 20:
            ks.append(0.0)
            pv.append(1.0)
        else:
            r = ks_2samp(u, v)
            ks.append(r.statistic)
            pv.append(r.pvalue)
    q = false_discovery_control(np.array(pv), method="bh")
    return pd.DataFrame(
        {"mi": mi, "gain": gain, "ks": ks, "q": q}, index=Xtr_raw.columns
    )


def select(strategy, sc, k):
    s = sc.copy()
    r_mi, r_xgb = s.mi.rank(ascending=False), s.gain.rank(ascending=False)
    score = (
        -r_mi
        if strategy == "mi"
        else -r_xgb if strategy == "xgb" else -(r_mi + r_xgb) / 2
    )
    if strategy == "drift_aware":
        score = score[~((s.q < Q_MAX) & (s.ks > KS_MIN))]
    kk = len(score) if k is None else min(k, len(score))
    return score.sort_values(ascending=False).index[:kk].tolist()


# ---------- model bundles (cached: the same training window always gives the same model) ----------
MODELS = {}


def train(start, end):
    key = (start, end)
    if key not in MODELS:
        Xtr, ytr = X.iloc[start:end], y.iloc[start:end]
        cols = select(SEL["strategy"], score_features(Xtr, ytr), SEL_K)
        imp = SimpleImputer(strategy="median").fit(Xtr[cols])
        A = imp.transform(Xtr[cols])
        MODELS[key] = dict(
            cols=cols,
            imp=imp,
            window=key,
            models=[xgb_model(ytr, s).fit(A, ytr) for s in SEEDS],
        )
    return MODELS[key]


def predict(bundle, lo, hi):
    B = bundle["imp"].transform(X.iloc[lo:hi][bundle["cols"]])
    return np.mean([m.predict_proba(B)[:, 1] for m in bundle["models"]], axis=0)


# ---------- label-free drift detector: share of ALL process features that drifted ----------
DRIFT = {}


def drift_share(ref, new):
    if (ref, new) not in DRIFT:
        ks, pv = [], []
        for j in range(Xv.shape[1]):
            u, v = Xv[ref[0] : ref[1], j], Xv[new[0] : new[1], j]
            u, v = u[~np.isnan(u)], v[~np.isnan(v)]
            if len(u) < 20 or len(v) < 20:
                ks.append(0.0)
                pv.append(1.0)
            else:
                r = ks_2samp(u, v)
                ks.append(r.statistic)
                pv.append(r.pvalue)
        q = false_discovery_control(np.array(pv), method="bh")
        DRIFT[(ref, new)] = float(np.mean((q < Q_MAX) & (np.array(ks) > KS_MIN)))
    return DRIFT[(ref, new)]


# ---------- policies ----------
def run_policy(kind, param=None, window=None):
    """kind: static | periodic | drift.  window=None -> expanding, else last `window` blocks."""
    cur = train(0, INIT_END)
    retrains, fired = 0, []
    preds = []
    for t in range(INIT_BLOCKS, n_blocks):
        new_rng = (bs[t], bs[t + 1])
        end_b = max(INIT_BLOCKS, t - LAG)  # labels available up to this block
        start_row = 0 if window is None else bs[max(0, end_b - window)]
        retrain = False
        if kind == "periodic":
            retrain = (t - INIT_BLOCKS) % param == 0
        elif kind == "drift":
            w0, w1 = cur["window"]
            share = drift_share(
                (max(w0, w1 - BLOCK_ROWS), w1), new_rng
            )  # newest data vs newest training data
            retrain = share > param
            fired.append(round(share, 3))
        if retrain:
            nxt = train(start_row, bs[end_b])
            if nxt is not cur:
                retrains += 1
                cur = nxt
        preds.append(predict(cur, *new_rng))
    return np.concatenate(preds), retrains, fired


POLICIES = [
    ("static", "static", None, None),
    ("periodic_every_1_expanding", "periodic", 1, None),
    ("periodic_every_2_expanding", "periodic", 2, None),
    ("periodic_every_1_sliding3", "periodic", 1, 3),
]
POLICIES += [(f"drift>{int(t*100)}%_expanding", "drift", t, None) for t in TRIGGERS]

rows0, rows1 = bs[INIT_BLOCKS], n_dev
yv = y.iloc[rows0:rows1].values
sizes = [bs[t + 1] - bs[t] for t in range(INIT_BLOCKS, n_blocks)]

print(
    "\n=== Stream blocks: prevalence and drift vs the initial model's newest training data ==="
)
ref0 = (INIT_END - BLOCK_ROWS, INIT_END)
for t in range(INIT_BLOCKS, n_blocks):
    yy = y.iloc[bs[t] : bs[t + 1]]
    print(
        f"block {t}: rows {bs[t]}-{bs[t+1]}  fails={int(yy.sum())}  "
        f"prevalence={yy.mean():.3f}  drifting features={drift_share(ref0, (bs[t], bs[t+1])):.1%}"
    )


def alert_metrics(p):
    yhat, i = np.zeros(len(yv), int), 0
    for sz in sizes:
        k = max(1, int(round(ALERT * sz)))
        yhat[i + np.argsort(-p[i : i + sz])[:k]] = 1
        i += sz
    tn, fp, fn, tp = confusion_matrix(yv, yhat, labels=[0, 1]).ravel()
    rec, prec, spec = tp / max(tp + fn, 1), tp / max(tp + fp, 1), tn / max(tn + fp, 1)
    return dict(
        recall=rec,
        precision=prec,
        f1=2 * prec * rec / max(prec + rec, 1e-12),
        bal_acc=(rec + spec) / 2,
        fnr=1 - rec,
    )


def paired_ci(p, p_static, n_boot=1000, chunk=20, seed=0):
    """Paired chunk bootstrap of the AP difference (policy - static); keeps short-range time structure."""
    rng, n = np.random.default_rng(seed), len(yv)
    starts, diffs = np.arange(0, n, chunk), []
    for _ in range(n_boot):
        pick = rng.integers(0, len(starts), len(starts))
        idx = np.concatenate(
            [np.arange(starts[i], min(starts[i] + chunk, n)) for i in pick]
        )
        if 0 < yv[idx].sum() < len(idx):
            diffs.append(
                average_precision_score(yv[idx], p[idx])
                - average_precision_score(yv[idx], p_static[idx])
            )
    return np.percentile(diffs, [2.5, 97.5])


results, scores, static_p = [], [], None
for name, kind, param, window in POLICIES:
    p, retrains, fired = run_policy(kind, param, window)
    if name == "static":
        static_p = p
    lo, hi = paired_ci(p, static_p) if name != "static" else (np.nan, np.nan)
    ap = average_precision_score(yv, p)
    results.append(
        dict(
            policy=name,
            retrains=retrains,
            pr_auc=ap,
            d_pr_auc_vs_static=ap - average_precision_score(yv, static_p),
            ci95=("" if name == "static" else f"[{lo:+.3f}, {hi:+.3f}]"),
            roc_auc=roc_auc_score(yv, p),
            **alert_metrics(p),
            drift_shares_seen=fired,
        )
    )
    scores.append(
        pd.DataFrame(
            {"policy": name, "row_id": np.arange(rows0, rows1), "y": yv, "score": p}
        )
    )
    print(f"done: {name}")

res = pd.DataFrame(results)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(
    f"\nStream: rows {rows0}-{rows1}, n={len(yv)}, failures={int(yv.sum())}, prevalence={yv.mean():.3f}"
)
print(f"Alert budget: top {int(ALERT*100)}% per block\n")
print(res.drop(columns="drift_shares_seen").round(3).to_string(index=False))
print("\nDrift shares seen by each triggered policy (per block):")
print(
    res[res.policy.str.startswith("drift")][["policy", "drift_shares_seen"]].to_string(
        index=False
    )
)

os.makedirs("results", exist_ok=True)
res.to_csv("results/adaptive_summary.csv", index=False)
pd.concat(scores).to_csv("results/adaptive_stream_scores.csv", index=False)
