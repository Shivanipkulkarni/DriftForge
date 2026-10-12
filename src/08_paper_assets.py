import importlib.metadata as md
import json, os, platform, re, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, false_discovery_control
from lib import KS_MIN, Q_MAX, load_all

OUT, FIG = "paper", "paper/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
tables = {}


def read(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  (skipped: {path} not found)")
    return None


def parse_ci(s):
    if not isinstance(s, str) or not s.strip():
        return np.nan, np.nan
    a = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return float(a[0]), float(a[1])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=300)
    plt.close(fig)
    print(f"  figure: {FIG}/{name}.png")


def forest(ax, d, title):
    d = d[d.policy != "static"].reset_index(drop=True)
    lo, hi = zip(*d.ci95.map(parse_ci))
    lo, hi, x = np.array(lo), np.array(hi), d.d_pr_auc_vs_static.values
    ypos = np.arange(len(d))[::-1]
    ax.errorbar(
        x,
        ypos,
        xerr=[np.maximum(x - lo, 0), np.maximum(hi - x, 0)],
        fmt="o",
        capsize=3,
        color="#1f4e79",
    )
    ax.axvline(0, ls="--", c="gray")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{p} ({r} retrains)" for p, r in zip(d.policy, d.retrains)])
    ax.set_xlabel("Change in PR-AUC vs static (95% paired bootstrap CI)")
    ax.set_title(title)


df, cfg, sel, feats = load_all()
n_dev, n = cfg["n_dev"], cfg["n_rows"]
y, Xv, tstr = df["target"].values, df[feats].values, df["time"].astype(str)

# ------------------------------------------------ Table 1 + Figure 1: data, protocol, prevalence and drift over time
rows = []


def span_row(name, lo, hi):
    yy = y[lo:hi]
    rows.append(
        dict(
            split=name,
            rows=hi - lo,
            failures=int(yy.sum()),
            failure_rate_pct=round(100 * yy.mean(), 2),
            start=tstr.iloc[lo][:10],
            end=tstr.iloc[hi - 1][:10],
        )
    )


for f in cfg["folds"]:
    span_row(f"Fold {f['fold']} train", *f["train"])
    span_row(f"Fold {f['fold']} validation", *f["val"])
span_row("Development (final-model training)", 0, n_dev)
span_row("Final test (untouched)", n_dev, n)
tables["T1_protocol"] = pd.DataFrame(rows)

blocks = np.array_split(np.arange(n_dev), cfg["n_blocks"]) + np.array_split(
    np.arange(n_dev, n), 2
)


def drift_share(a, b):
    ks, pv = [], []
    for j in range(Xv.shape[1]):
        u, v = Xv[a, j], Xv[b, j]
        u, v = u[~np.isnan(u)], v[~np.isnan(v)]
        if len(u) < 20 or len(v) < 20:
            ks.append(0.0)
            pv.append(1.0)
        else:
            r = ks_2samp(u, v)
            ks.append(r.statistic)
            pv.append(r.pvalue)
    q = false_discovery_control(np.array(pv), method="bh")
    return float(np.mean((q < Q_MAX) & (np.array(ks) > KS_MIN)))


prev = [100 * y[b].mean() for b in blocks]
shares = [0.0] + [100 * drift_share(blocks[0], b) for b in blocks[1:]]
fig, ax = plt.subplots(figsize=(9, 4.2))
bars = ax.bar(
    range(len(blocks)), prev, color=["#8fb4d9"] * cfg["n_blocks"] + ["#e3a857"] * 2
)
for i, b in enumerate(blocks):
    ax.text(i, prev[i] + 0.3, f"{int(y[b].sum())}", ha="center", fontsize=8)
ax.set_ylabel("Failure rate (%)  [labels = failures]")
ax.set_xlabel("Time block (start date)")
ax.set_xticks(range(len(blocks)))
ax.set_xticklabels([tstr.iloc[b[0]][5:10] for b in blocks], rotation=45)
ax.axvline(cfg["n_blocks"] - 0.5, ls="--", c="gray")
ax2 = ax.twinx()
ax2.plot(range(len(blocks)), shares, "o-", color="#b03a2e")
ax2.set_ylabel("Features drifting vs first block (%)", color="#b03a2e")
ax2.spines["right"].set_visible(True)
ax.set_title(
    "Failure rate and label-free feature drift over time (orange = final test period)"
)
save(fig, "fig1_prevalence_and_drift")

# ------------------------------------------------ Table 2 + Figure 2: static baselines
s = read("results/static_baseline_summary.csv")
fl = read("results/static_baseline_folds.csv")
if s is not None and fl is not None:
    tables["T2_static_baselines"] = s.round(3)
    pv = fl.pivot(index="fold", columns="model", values="pr_auc")
    prevf = fl.drop_duplicates("fold").set_index("fold")["prevalence"]
    fig, ax = plt.subplots(figsize=(7, 4))
    pv.plot.bar(ax=ax, rot=0, width=0.75)
    for i, f in enumerate(pv.index):
        ax.hlines(prevf[f], i - 0.4, i + 0.4, colors="k", linestyles="dotted")
    ax.set_xlabel("Validation fold")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Static baselines per fold (dotted = prevalence = random scorer)")
    save(fig, "fig2_static_baselines")

# ------------------------------------------------ Table 3 + Figure 3: feature selection
fs = read("results/featsel_summary.csv")
if fs is not None:
    fs = fs.sort_values("pr_auc", ascending=False)
    tables["T3_feature_selection"] = fs[
        [
            "config",
            "strategy",
            "n_sel",
            "pr_auc",
            "pr_sd",
            "lift",
            "roc_auc",
            "folds_beating_all",
            "stability",
        ]
    ].round(3)
    d = fs.iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    a1.errorbar(
        d.pr_auc, range(len(d)), xerr=d.pr_sd, fmt="o", capsize=2, color="#1f4e79"
    )
    base = fs.loc[fs.config == "all_features", "pr_auc"]
    if len(base):
        a1.axvline(float(base.iloc[0]), ls="--", c="gray")
    a1.set_yticks(range(len(d)))
    a1.set_yticklabels(d.config)
    a1.set_xlabel("Mean PR-AUC over folds (± SD); dashed = all features")
    a2.barh(range(len(d)), d.stability, color="#7fa37f")
    a2.set_xlabel("Selection stability (Jaccard between consecutive folds)")
    save(fig, "fig3_feature_selection")

# ------------------------------------------------ Table 4 + Figure 4: adaptive experiment on development stream
ad = read("results/adaptive_summary.csv")
if ad is not None:
    tables["T4_adaptive_dev"] = ad.drop(
        columns=["drift_shares_seen"], errors="ignore"
    ).round(3)
    fig, ax = plt.subplots(figsize=(8, 4))
    forest(ax, ad, "Development stream: adaptation vs static")
    save(fig, "fig4_adaptive_dev")

# ------------------------------------------------ Table 5 + Figure 5: final test
ft = read("results/final_test_summary.csv")
if ft is not None:
    tables["T5_final_test"] = ft.drop(columns=["drift_shares"], errors="ignore").round(
        3
    )
    lags = sorted(ft.lag.unique())
    fig, axes = plt.subplots(1, len(lags), figsize=(6.5 * len(lags), 4), squeeze=False)
    for ax, lag in zip(axes[0], lags):
        forest(ax, ft[ft.lag == lag], f"Final test, label lag = {lag} block(s)")
    save(fig, "fig5_final_test")

# ------------------------------------------------ Table 6 + Figure 6: SHAP
sg = read("results/shap_global.csv")
if sg is not None:
    tables["T6_shap_global"] = sg.head(15).round(3)
    t = sg.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(
        t.sensor,
        t["share_of_total_%"],
        color=["#b03a2e" if d else "#7f8c8d" for d in t.drifting],
    )
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.set_title("Features the model relies on (red = also drifting in final period)")
    save(fig, "fig6_shap_top10")
    if os.path.exists("results/shap_summary.png"):
        import shutil

        shutil.copy("results/shap_summary.png", f"{FIG}/fig7_shap_beeswarm.png")

# ------------------------------------------------ Table 7: chatbot evaluation
if os.path.exists("results/chatbot_eval.json"):
    ev = json.load(open("results/chatbot_eval.json"))
    if isinstance(ev, dict):  # newer format: {"provider": ..., "model": ..., "results": [...]}
        ev = ev["results"]
    tables["T7_chatbot_eval"] = pd.DataFrame(
        [
            dict(
                question=r["question"],
                status=r["status"],
                tools_called=len(r["tools_called"]),
                guard_rejections=len(
                    r.get("violations_before_accept", r.get("violations", []))
                ),
                soft_warnings=len(r.get("warnings", [])),
                answer=r["answer"],
            )
            for r in ev
        ]
    )
else:
    print("  (skipped: results/chatbot_eval.json not found)")

# ------------------------------------------------ Tables 8-11 + Figure 9: PSO-XGBoost and protocol gap (Steps 9-10)
ph = read("results/pso_holdout_summary.csv")
if ph is not None:
    tables["T8_pso_holdout"] = ph.round(3)
pf = read("results/pso_final_summary.csv")
if pf is not None:
    tables["T9_pso_final"] = pf.round(3)
if os.path.exists("results/pso_best.json"):
    best = json.load(open("results/pso_best.json"))
    tables["T10_pso_params"] = (
        pd.DataFrame(
            {
                m: {**best[m]["params"], "search_pr_auc": best[m]["search_pr_auc"]}
                for m in ("default", "random", "pso")
            }
        )
        .round(4)
        .rename_axis("parameter")
        .reset_index()
    )
gap = read("results/protocol_gap.csv")
if gap is not None:
    tables["T11_protocol_gap"] = gap.rename(
        columns={
            "pr_auc_mean": "pr_auc_shuffled",
            "roc_auc_mean": "roc_auc_shuffled",
            "pr_auc": "pr_auc_temporal_final",
            "roc_auc": "roc_auc_temporal_final",
        }
    ).round(3)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    xs, w = np.arange(len(gap)), 0.38
    for ax, (a, b, lab) in zip(
        axes, [("pr_auc_mean", "pr_auc", "PR-AUC"), ("roc_auc_mean", "roc_auc", "ROC-AUC")]
    ):
        ax.bar(xs - w / 2, gap[a], w, color="#c0392b", label="Shuffled split (leaky)")
        ax.bar(xs + w / 2, gap[b], w, color="#1f4e79", label="Temporal final test")
        ax.set_xticks(xs)
        ax.set_xticklabels(gap.method)
        ax.set_ylabel(lab)
        ax.set_ylim(0, max(gap[a].max(), gap[b].max()) * 1.3)  # headroom for the legend
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Same models, two evaluation protocols")
    save(fig, "fig9_protocol_gap")

# ------------------------------------------------ write workbook + reproducibility manifest
with pd.ExcelWriter(f"{OUT}/tables.xlsx", engine="openpyxl") as xl:
    for name, t in tables.items():
        t.to_excel(xl, sheet_name=name[:31], index=False)
print(f"  workbook: {OUT}/tables.xlsx  sheets: {', '.join(tables)}")


def ver(p):
    try:
        return md.version(p)
    except Exception:
        return None


try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    commit = None
json.dump(
    dict(
        python=platform.python_version(),
        git_commit=commit,
        packages={
            p: ver(p)
            for p in [
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
                "shap",
                "scipy",
                "openai",
            ]
        },
        selected_feature_config=sel,
        n_rows=n,
        n_dev=n_dev,
        folds=cfg["folds"],
        final=cfg["final"],
    ),
    open(f"{OUT}/manifest.json", "w"),
    indent=2,
)
print(f"  manifest: {OUT}/manifest.json")
