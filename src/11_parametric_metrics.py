"""
11_parametric_metrics.py
Regression-style metrics (MAE, MSE, RMSE, R2), efficiency metrics, and drift before/after figures.

Why lot-level yield: Xu, Qin & Sun (2022) predict a continuous wafer yield and report RMSE, MAE and R2.
SECOM only has a binary pass/fail label per row, so row-level errors are computed on predicted failure
probabilities, and a yield-like quantity is built by grouping consecutive rows into lots of 25 (the
paper's lot size):  actual yield = 1 - failure fraction,  predicted yield = 1 - mean predicted probability.

The metric definitions match the paper's (RMSE, MAE, R2 with the mean of the evaluated set).
The data and the target are different, so the numbers are NOT directly comparable with the paper's.
The "noise floor" printed below is the smallest lot-level RMSE even a perfect probability model could
reach on binary data, and it explains most of the gap.

Predicted probabilities come from models trained with scale_pos_weight, which inflates them.
They are converted back with the exact prior correction  p' = p / (p + w(1-p)),  w = neg/pos of the
model's own training rows, so errors measure probability quality, not the class-weight setting.

Needs Steps 1-5 and 9 (run 09 with --final first). Usage, from the repository root:
    python src/11_parametric_metrics.py
"""

import importlib
import json
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ks_2samp

from lib import KS_MIN, Q_MAX, load_all

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = "results", "paper/figures"
LOT = 25  # wafers per lot, as in the paper
PAPER = {  # Table 1 of Xu, Qin & Sun (2022), IFAC PapersOnLine 55-10; MSE is derived (RMSE squared)
    "paper_IXGBoost": dict(rmse=0.0061, mae=0.0042, r2=0.8591),
    "paper_BAG": dict(rmse=0.0084, mae=0.0061, r2=0.7248),
    "paper_BPNN": dict(rmse=0.0118, mae=0.0092, r2=0.4546),
}

df, cfg, sel, feats = load_all()
y_all = df["target"].values
n_rows, n_dev = len(df), cfg["n_dev"]
# same blocks as Steps 1/4/8: n_blocks development blocks, then the final-test period as 2 blocks
bs = np.array_split(np.arange(n_dev), cfg["n_blocks"]) + np.array_split(
    np.arange(n_dev, n_rows), 2
)
N_DEV_BLOCKS = cfg["n_blocks"]


def prior_correct(p, w):
    return p / (p + w * (1.0 - p))


def weight(rows_end):
    pos = y_all[:rows_end].sum()
    return (rows_end - pos) / max(pos, 1)


def reg(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(
        mae=float(np.mean(np.abs(err))),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        r2=float(1 - np.sum(err**2) / sst) if sst > 0 else np.nan,
    )


def lots(y, p, size=LOT):
    n = (len(y) // size) * size  # drop the incomplete last lot
    ya = y[:n].reshape(-1, size).mean(axis=1)
    pa = p[:n].reshape(-1, size).mean(axis=1)
    return 1 - ya, 1 - pa  # actual yield, predicted yield


# ================================================================ final test: parametric table
sc = pd.read_csv(f"{RES}/pso_final_scores.csv").sort_values("row_id")
y = sc["y"].values
w_final = weight(n_dev)
base_p = np.full(
    len(y), y_all[:n_dev].mean()
)  # constant predictor = development failure rate
methods = ["default", "random", "pso"]
probs = {m: prior_correct(sc[f"score_{m}"].values, w_final) for m in methods}
probs["constant_baseline"] = base_p

row_level, lot_level, lot_cache = [], [], {}
for m, p in probs.items():
    row_level.append(dict(model=m, **reg(y.astype(float), p)))
    ya, pa = lots(y.astype(float), p)
    lot_cache[m] = (ya, pa)
    lot_level.append(dict(model=m, n_lots=len(ya), **reg(ya, pa)))
row_level, lot_level = pd.DataFrame(row_level), pd.DataFrame(lot_level)
prev = y.mean()
floor_rmse = float(np.sqrt(prev * (1 - prev) / LOT))
floor = dict(
    model="noise_floor_approx",
    n_lots=lot_level.n_lots[0],
    mae=floor_rmse * np.sqrt(2 / np.pi),
    mse=floor_rmse**2,
    rmse=floor_rmse,
    r2=np.nan,
)

row_level.round(4).to_csv(f"{RES}/parametric_row_level.csv", index=False)
lot_level = pd.concat([lot_level, pd.DataFrame([floor])], ignore_index=True)
lot_level.round(4).to_csv(f"{RES}/parametric_lot_level.csv", index=False)

# side-by-side with the paper
cmp_rows = []
for k, name in (
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("mse", "MSE (derived for paper)"),
    ("r2", "R2"),
):
    r = {"metric": name}
    for pn, pv in PAPER.items():
        r[pn] = pv["rmse"] ** 2 if k == "mse" else pv[k]
    for _, row in lot_level.iterrows():
        r[f"ours_{row['model']}"] = row[k]
    cmp_rows.append(r)
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(f"{RES}/paper_comparison.csv", index=False)

# ================================================================ efficiency / improvement metrics
fin = pd.read_csv(f"{RES}/pso_final_summary.csv").set_index("method")
p9 = importlib.import_module("09_pso_xgboost")
X, yy = df[feats], df["target"]
wins = p9.Windows(X, yy, sel)
A, ytr, B, _ = wins.get(cfg["final"]["train"], cfg["final"]["test"])
frozen = json.load(open(f"{RES}/pso_best.json"))
eff = []
mse_base = row_level.set_index("model").loc["constant_baseline", "mse"]
for m in methods:
    params = frozen[m]["params"]
    t0 = time.perf_counter()
    mdl = p9.fit_model(params, A, ytr, 0)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    mdl.predict_proba(B)
    t_pred = time.perf_counter() - t0
    mse_m = row_level.set_index("model").loc[m, "mse"]
    eff.append(
        dict(
            model=m,
            pr_auc=fin.loc[m, "pr_auc"],
            pr_lift=fin.loc[m, "pr_lift"],
            roc_auc=fin.loc[m, "roc_auc"],
            recall_at_10pct_alert=fin.loc[m, "recall"],
            precision_at_10pct_alert=fin.loc[m, "precision"],
            failures_caught_per_100_flagged=100 * fin.loc[m, "precision"],
            random_failures_per_100=100 * prev,
            brier_skill_vs_baseline=1 - mse_m / mse_base,
            n_trees=params["n_estimators"],
            train_seconds=t_fit,
            predict_ms_per_1000_wafers=1000 * t_pred / len(B) * 1000,
            search_evaluations=0 if m == "default" else frozen["budget_evaluations"],
        )
    )
eff = pd.DataFrame(eff)
eff.round(4).to_csv(f"{RES}/efficiency_metrics.csv", index=False)

# ================================================================ drift before / after
shap = pd.read_csv(f"{RES}/shap_global.csv")
names = shap["sensor"].tolist()
# map "Sensor 59" style names back to column names used in the cleaned data
col_of = {}
for c in feats:
    col_of[f"Sensor {c}"] = c
    col_of[str(c)] = c
top = [col_of[s] for s in names if s in col_of][:20]
ref_rows = bs[0]
ks = pd.DataFrame(index=top, columns=range(1, len(bs)), dtype=float)
for c in top:
    ref = df.loc[ref_rows, c].dropna()
    for b in range(1, len(bs)):
        new = df.loc[bs[b], c].dropna()
        ks.loc[c, b] = (
            ks_2samp(ref, new).statistic if len(ref) > 5 and len(new) > 5 else np.nan
        )

# share of ALL features drifting per block (same rule as Steps 3-4: KS > 0.30 and FDR q < 0.05)
share = [0.0]
for b in range(1, len(bs)):
    stats, pv = [], []
    for c in feats:
        u, v = df.loc[ref_rows, c].dropna(), df.loc[bs[b], c].dropna()
        if len(u) > 5 and len(v) > 5:
            r = ks_2samp(u, v)
            stats.append(r.statistic)
            pv.append(r.pvalue)
    q = false_discovery_control(np.array(pv)) if pv else np.array([])
    share.append(float(np.mean((np.array(stats) > KS_MIN) & (q < Q_MAX))) * 100)
share = np.array(share)
pd.DataFrame(
    {
        "block": range(len(bs)),
        "start_row": [b[0] for b in bs],
        "pct_features_drifting": share,
    }
).to_csv(f"{RES}/drift_by_block.csv", index=False)

# ---- Figure 10: distributions before (first block) vs after (last 3 blocks) for the most-drifting top features
after_rows = np.concatenate(bs[-3:])
worst = ks.max(axis=1).sort_values(ascending=False).index[:6]
fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
for ax, c in zip(axes.ravel(), worst):
    u, v = df.loc[ref_rows, c].dropna(), df.loc[after_rows, c].dropna()
    lo, hi = np.percentile(np.r_[u, v], [1, 99])
    bins = np.linspace(lo, hi, 25)
    ax.hist(
        u,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#1f4e79",
        label="Before (first block)",
    )
    ax.hist(
        v,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#c0392b",
        label="After (last 3 blocks)",
    )
    ax.set_title(f"Sensor {c} (KS = {ks_2samp(u, v).statistic:.2f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle(
    "Feature distributions before vs after drift (six most-shifted of the top-20 SHAP features)"
)
fig.tight_layout()
fig.savefig(f"{FIG}/fig10_drift_before_after.png", dpi=300)
plt.close(fig)

# ---- Figure 11: when drift appears (KS vs first block, top-20 SHAP features)
fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(ks.values.astype(float), aspect="auto", cmap="magma_r", vmin=0, vmax=0.6)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([str(c) for c in top], fontsize=8)
starts = [df["time"].iloc[b[0]] for b in bs]
ax.set_xticks(range(len(bs) - 1))
ax.set_xticklabels([str(s)[5:10] for s in starts[1:]], rotation=45, fontsize=8)
ax.set_xlabel("Time block (start date)")
ax.set_ylabel("Feature (ordered by SHAP importance)")
fig.colorbar(im, ax=ax, label=f"KS distance from first block (drift rule: > {KS_MIN})")
ax.set_title("When drift appears in the features the model relies on")
fig.tight_layout()
fig.savefig(f"{FIG}/fig11_drift_heatmap.png", dpi=300)
plt.close(fig)

# ---- Figure 12: lot-level yield, actual vs predicted, final test
fig, ax = plt.subplots(figsize=(7.5, 4))
ya = lot_cache["default"][0]
ax.plot(range(1, len(ya) + 1), ya, "k-o", lw=2, label="Actual yield")
for m, c in (("default", "#1f4e79"), ("random", "#e08a1e"), ("pso", "#2e8b57")):
    ax.plot(
        range(1, len(ya) + 1), lot_cache[m][1], "--", color=c, label=f"Predicted ({m})"
    )
ax.set_xlabel(f"Lot of {LOT} consecutive wafers (final test period)")
ax.set_ylabel("Yield (1 - failure fraction)")
ax.set_title("Lot-level yield on the final test")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig12_lot_yield_tracking.png", dpi=300)
plt.close(fig)

# ---- Figure 13: error before vs after adaptation across the development stream, with drift level
st = pd.read_csv(f"{RES}/adaptive_stream_scores.csv")
rows_per_block = {}
for pol in ("static", "periodic_every_1_expanding"):
    s = st[st.policy == pol].sort_values("row_id")
    for b in range(3, N_DEV_BLOCKS):
        sub = s[s.row_id.isin(bs[b])]
        if sub.empty:
            continue
        w = weight(bs[3][0]) if pol == "static" else weight(bs[b][0])
        p = prior_correct(sub["score"].values, w)
        yv = sub["y"].values.astype(float)
        rows_per_block[(pol, b)] = dict(
            policy=pol,
            block=b,
            n=len(sub),
            yield_error=abs((1 - yv.mean()) - (1 - p.mean())),
            row_mse=float(np.mean((yv - p) ** 2)),
            row_mae=float(np.mean(np.abs(yv - p))),
        )
blk = pd.DataFrame(rows_per_block.values())
blk.round(4).to_csv(f"{RES}/adaptation_error_by_block.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4))
for pol, c, lab in (
    ("static", "#c0392b", "Static (never retrained)"),
    ("periodic_every_1_expanding", "#1f4e79", "Retrained every block"),
):
    d = blk[blk.policy == pol]
    ax.plot(d.block, d.yield_error, "-o", color=c, label=lab)
ax.set_xlabel("Time block")
ax.set_ylabel("Block-level yield error (absolute)")
ax2 = ax.twinx()
ax2.plot(range(len(bs)), share, ":", color="gray", label="% features drifting (right)")
ax2.set_ylabel("% features drifting vs first block")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
ax.set_title("Yield error before vs after adapting, against drift level")
fig.tight_layout()
fig.savefig(f"{FIG}/fig13_adaptation_vs_drift.png", dpi=300)
plt.close(fig)

# ================================================================ print
pd.set_option("display.width", 220)
print(
    f"Final test: {len(y)} wafers, failure rate {prev:.3f}; lots of {LOT} -> {lot_level.n_lots[0]} lots\n"
)
print("=== Row level (predicted failure probability vs 0/1 label) ===")
print(row_level.round(4).to_string(index=False))
print("\n=== Lot level (yield = 1 - failure fraction) ===")
print(lot_level.round(4).to_string(index=False))
print(
    "\n=== Next to the paper (different data and target; see the notes in this script) ==="
)
print(cmp_df.round(4).to_string(index=False))
print("\n=== Efficiency / improvement ===")
print(eff.round(3).T.to_string(header=False))
print("\n=== Error before vs after adapting (dev stream) ===")
print(blk.round(4).to_string(index=False))
print(
    "\nSaved results/parametric_*.csv, paper_comparison.csv, efficiency_metrics.csv, drift_by_block.csv,"
)
print("adaptation_error_by_block.csv and paper/figures/fig10-fig13.")
