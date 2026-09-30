# src/05_final_test.py
# FINAL, frozen evaluation on the untouched last 20% of rows. RUN ONCE.
# Nothing below may be changed after you have seen the output.
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from lib import (ALERT, Deployer, alert_metrics, load_all, paired_ci)

if os.path.exists("results/final_test_summary.csv") and os.environ.get("FORCE") != "1":
    sys.exit("Final test already run (results/final_test_summary.csv exists). "
             "Re-running and re-picking is not allowed; set FORCE=1 only for a documented re-run.")

# ---------------- frozen protocol ----------------
FINAL_BLOCKS = 4                 # the 314 final rows arrive as 4 batches (~78 rows) -> 3 chances to adapt
TRIGGERS = [0.05, 0.10, 0.20]    # PRIMARY = 0.10 (declared before seeing any result); others = sensitivity
LAGS = [0, 1]                    # label delay in test blocks (0 = labels known before the next block)

df, cfg, sel, feats = load_all()
X, y = df[feats], df["target"]
n_dev, n = cfg["n_dev"], cfg["n_rows"]
BLOCK_ROWS = len(np.array_split(np.arange(n_dev), cfg["n_blocks"])[0])   # dev block size, drift reference length
tb = [int(b[0]) for b in np.array_split(np.arange(n_dev, n), FINAL_BLOCKS)] + [n]
sizes = [tb[t + 1] - tb[t] for t in range(FINAL_BLOCKS)]
yv = y.iloc[n_dev:n].values
dep = Deployer(X, y, sel)
print(f"Feature config (chosen in Step 3, development data only): {sel}")
print(f"Final test: rows {n_dev}-{n}, n={len(yv)}, failures={int(yv.sum())}, prevalence={yv.mean():.3f}")
print(f"Period: {df['time'].iloc[n_dev]}  ->  {df['time'].iloc[n-1]}\n")


def run(kind, param, lag):
    cur = dep.train(0, n_dev)                         # model trained on ALL development data
    retrains, shares, preds = 0, [], []
    for t in range(FINAL_BLOCKS):
        new = (tb[t], tb[t + 1])
        retrain = False
        if kind == "periodic":
            retrain = True
        elif kind == "drift":
            w0, w1 = cur["window"]
            share = dep.drift_share((max(w0, w1 - BLOCK_ROWS), w1), new)   # label-free, before predicting
            shares.append(round(share, 3))
            retrain = share > param
        if retrain:
            nxt = dep.train(0, tb[max(0, t - lag)])   # only labels already available at this point
            if nxt is not cur:
                retrains += 1; cur = nxt
        preds.append(dep.predict(cur, *new))
    return np.concatenate(preds), retrains, shares


print("=== Final-test blocks: prevalence and drift vs the newest development data ===")
ref0 = (n_dev - BLOCK_ROWS, n_dev)
for t in range(FINAL_BLOCKS):
    yy = y.iloc[tb[t]:tb[t + 1]]
    print(f"test block {t}: rows {tb[t]}-{tb[t+1]}  fails={int(yy.sum())}  prevalence={yy.mean():.3f}  "
          f"drifting features={dep.drift_share(ref0, (tb[t], tb[t+1])):.1%}")

POLICIES = [("static", "static", None), ("periodic_every_block", "periodic", None)]
POLICIES += [(f"drift>{int(t*100)}%", "drift", t) for t in TRIGGERS]

rows, scores = [], []
for lag in LAGS:
    static_p, _, _ = run("static", None, lag)
    for name, kind, param in POLICIES:
        p, retrains, shares = run(kind, param, lag)
        ap = average_precision_score(yv, p)
        ap_s = average_precision_score(yv, static_p)
        lo, hi = (np.nan, np.nan) if name == "static" else paired_ci(yv, p, static_p)
        rows.append(dict(lag=lag, policy=name, retrains=retrains, pr_auc=ap, pr_lift=ap / yv.mean(),
                         d_pr_auc_vs_static=ap - ap_s,
                         ci95="" if name == "static" else f"[{lo:+.3f}, {hi:+.3f}]",
                         roc_auc=roc_auc_score(yv, p), **alert_metrics(yv, p, sizes), drift_shares=shares))
        scores.append(pd.DataFrame({"lag": lag, "policy": name, "row_id": np.arange(n_dev, n), "y": yv, "score": p}))
    print(f"done: lag={lag}")

res = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for lag in LAGS:
    print(f"\n=== FINAL TEST, label lag = {lag} block(s); alert budget = top {int(ALERT*100)}% per block ===")
    print(res[res.lag == lag].drop(columns=["lag", "drift_shares"]).round(3).to_string(index=False))
print("\nPRIMARY comparison: static vs drift>10% at lag 0 (everything else is sensitivity).")

os.makedirs("results", exist_ok=True)
res.to_csv("results/final_test_summary.csv", index=False)
pd.concat(scores).to_csv("results/final_test_scores.csv", index=False)