# src/06_shap_registry.py
# Needs src/lib.py plus outputs of Steps 1 and 3. Uses the DEVELOPMENT-trained model; no labels are used for explanations.
import json, os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, false_discovery_control
from lib import Deployer, KS_MIN, Q_MAX, load_all

df, cfg, sel, feats = load_all()
X, y = df[feats], df["target"]
n_dev, n = cfg["n_dev"], cfg["n_rows"]
blocks = np.array_split(np.arange(n_dev), cfg["n_blocks"])
BLOCK_ROWS, INIT_END = len(blocks[0]), int(
    blocks[3][0]
)  # initial model = first 3 blocks (as in Step 4)
for d in ("data/metadata", "results"):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------ 1. Feature metadata registry
# Rule: physical meaning stays EMPTY until someone supplies it WITH a source. The chatbot may only
# say "Sensor <id>" plus data-derived facts (percentiles, drift) unless meaning_status == "provided".
REG = "data/metadata/feature_registry.csv"
dev = X.iloc[:n_dev]
reg = pd.DataFrame({"feature_id": feats})
reg["display_name"] = "Sensor " + reg.feature_id
reg["dev_missing_pct"] = dev.isna().mean().values * 100
reg["dev_p05"], reg["dev_p50"], reg["dev_p95"] = (
    dev.quantile(q).values for q in (0.05, 0.50, 0.95)
)
human_cols = ["physical_meaning", "unit", "meaning_source", "meaning_status"]
if os.path.exists(REG):  # keep anything teammates already filled in
    old = pd.read_csv(REG, dtype={"feature_id": str})
    reg = reg.merge(old[["feature_id"] + human_cols], on="feature_id", how="left")
else:
    for c in human_cols:
        reg[c] = None
reg["meaning_status"] = reg["meaning_status"].fillna("unknown")
reg.to_csv(REG, index=False)
regi = reg.set_index("feature_id")
print(
    f"Registry saved: {REG} ({len(reg)} features; meaning known for "
    f"{int((reg.meaning_status == 'provided').sum())})"
)

# ------------------------------------------------------------------ 2. SHAP for the development-trained model
dep = Deployer(X, y, sel)
final_m, init_m = dep.train(0, n_dev), dep.train(0, INIT_END)


def shap_for(bundle, lo, hi):
    A = bundle["imp"].transform(X.iloc[lo:hi][bundle["cols"]])
    svs, bases = [], []
    for m in bundle["models"]:
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(A)
        svs.append(sv[-1] if isinstance(sv, list) else sv)
        bases.append(float(np.ravel(ex.expected_value)[-1]))
    return np.mean(svs, axis=0), float(np.mean(bases)), A


sv, base, A = shap_for(final_m, n_dev, n)  # explain the final-period wafers
cols = final_m["cols"]
glob = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)

# label-free drift of each feature: newest development block vs the final period
Xv = X.values
ref, new = Xv[n_dev - BLOCK_ROWS : n_dev], Xv[n_dev:n]
ks, pv = [], []
for j in range(Xv.shape[1]):
    u, v = ref[:, j][~np.isnan(ref[:, j])], new[:, j][~np.isnan(new[:, j])]
    if len(u) < 20 or len(v) < 20:
        ks.append(0.0)
        pv.append(1.0)
    else:
        r = ks_2samp(u, v)
        ks.append(r.statistic)
        pv.append(r.pvalue)
drift = pd.DataFrame(
    {"ks": ks, "q": false_discovery_control(np.array(pv), method="bh")}, index=feats
)
drift["drifting"] = (drift.q < Q_MAX) & (drift.ks > KS_MIN)

top = pd.DataFrame(
    {
        "sensor": ["Sensor " + c for c in glob.index],
        "mean_abs_shap": glob.values,
        "share_of_total_%": 100 * glob.values / glob.sum(),
        "drift_ks": drift.loc[glob.index, "ks"].values,
        "drifting": drift.loc[glob.index, "drifting"].values,
    }
)
top.to_csv("results/shap_global.csv", index=False)
print(
    "\n=== Global SHAP (model trained on all development data, explained on the final period) ==="
)
print("Note: SHAP describes what the MODEL uses, not what physically causes failures.")
print(top.head(10).round(3).to_string(index=False))

sv0, _, _ = shap_for(init_m, n_dev, n)
top10_now, top10_init = set(glob.index[:10]), set(
    pd.Series(np.abs(sv0).mean(0), index=init_m["cols"]).nlargest(10).index
)
print(
    f"\nTop-10 overlap between the initial model (first 3 blocks) and the development model: "
    f"{len(top10_now & top10_init)}/10 features (Jaccard {len(top10_now & top10_init) / len(top10_now | top10_init):.2f})"
)
print(
    f"Top-10 features that are also drifting in the final period: {int(top.head(10).drifting.sum())}/10"
)

# summary plot
plt.figure()
shap.summary_plot(
    sv,
    pd.DataFrame(A, columns=["Sensor " + c for c in cols]),
    show=False,
    max_display=15,
)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=200)
plt.close()
pd.DataFrame(sv, columns=cols, index=np.arange(n_dev, n)).rename_axis("row_id").to_csv(
    "results/shap_test_values.csv"
)

# ------------------------------------------------------------------ 3. Local explanation cards (structured output for the chatbot)
p = dep.predict(final_m, n_dev, n)
cards = []
for rng in np.array_split(
    np.arange(n_dev, n), 4
):  # highest-risk wafer in each final block
    i = int(rng[np.argmax(p[rng - n_dev])])
    contrib = []
    for j in np.argsort(-np.abs(sv[i - n_dev]))[:5]:
        col, val = cols[j], X.iloc[i][cols[j]]
        dv = dev[col].dropna().values
        contrib.append(
            dict(
                feature_id=col,
                display_name=regi.loc[col, "display_name"],
                physical_meaning=(
                    None
                    if pd.isna(regi.loc[col, "physical_meaning"])
                    else regi.loc[col, "physical_meaning"]
                ),
                meaning_status=regi.loc[col, "meaning_status"],
                value=None if pd.isna(val) else float(val),
                was_missing_and_imputed=bool(pd.isna(val)),
                dev_percentile=(
                    None if pd.isna(val) else round(100 * float((dv <= val).mean()), 1)
                ),
                shap_log_odds=round(float(sv[i - n_dev][j]), 4),
                effect=(
                    "raises predicted risk"
                    if sv[i - n_dev][j] > 0
                    else "lowers predicted risk"
                ),
                drifting_in_final_period=bool(drift.loc[col, "drifting"]),
            )
        )
    cards.append(
        dict(
            row_id=i,
            time=str(df["time"].iloc[i]),
            risk_score=round(float(p[i - n_dev]), 4),
            score_note="model score, not a calibrated probability",
            base_value_log_odds=round(base, 4),
            top_contributors=contrib,
        )
    )
json.dump(cards, open("results/explanations.json", "w"), indent=2)
print("\n=== Example explanation card ===")
print(json.dumps(cards[0], indent=2))
print(
    "\nSaved: registry CSV, results/shap_global.csv, shap_summary.png, shap_test_values.csv, explanations.json"
)
