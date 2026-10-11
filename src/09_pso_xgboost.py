"""
09_pso_xgboost.py
Does particle-swarm-tuned XGBoost beat default XGBoost on time-ordered SECOM data?

Three configurations are compared, all using the frozen feature config from Step 3
(results/selected_config.json) and the same frozen temporal folds:
    default  : the hyperparameters already used everywhere else (lib.MODEL_PARAMS)
    random   : random search with the SAME number of evaluations as PSO (the control)
    pso      : particle swarm optimisation over the XGBoost hyperparameters

Protocol (nothing here may look at held-out labels when choosing anything):
    search folds : folds 1 and 2 of configs/folds.json. The swarm only ever sees these.
    held-out fold: fold 3. Not used by the search; used once to compare the frozen configs.
    final test   : the untouched last 20% of rows. Only touched with --final, once.

Usage (from the repository root):
    python src/09_pso_xgboost.py                      # search + held-out fold comparison
    python src/09_pso_xgboost.py --particles 6 --iters 3   # quick smoke test
    python src/09_pso_xgboost.py --final              # frozen final-test run (refuses to overwrite)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from lib import (
    MODEL_PARAMS,
    SEEDS,
    alert_metrics,
    load_all,
    paired_ci,
    score_features,
    select,
)

RES = "results"
FIG = "paper/figures"
SEARCH_SEEDS = [0, 1]  # seeds averaged inside the search (cheap but less noisy than one)
HOLDOUT_BLOCKS, FINAL_BLOCKS = 2, 4  # alert-budget blocks, same convention as Step 5

# name, low, high, scale. "log" spreads samples evenly across orders of magnitude.
SPACE = [
    ("n_estimators", 50, 500, "int"),
    ("max_depth", 2, 8, "int"),
    ("learning_rate", 0.01, 0.30, "log"),
    ("subsample", 0.5, 1.0, "lin"),
    ("colsample_bytree", 0.3, 1.0, "lin"),
    ("min_child_weight", 1.0, 10.0, "log"),
    ("reg_lambda", 0.1, 10.0, "log"),
    ("gamma", 0.0, 5.0, "lin"),
]
DEFAULT = dict(MODEL_PARAMS, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0)


# ------------------------------------------------------------------ search space
def decode(u):
    """Unit-cube position -> XGBoost hyperparameters."""
    p = {}
    for (name, lo, hi, kind), v in zip(SPACE, u):
        if kind == "int":
            p[name] = int(round(lo + v * (hi - lo)))
        elif kind == "log":
            p[name] = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
        else:
            p[name] = float(lo + v * (hi - lo))
    return p


def fit_model(params, A, ytr, seed):
    pos = int(np.sum(ytr))
    neg = len(ytr) - pos
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=neg / max(pos, 1),  # same imbalance handling as the rest of the repo
        random_state=seed,
        n_jobs=-1,
    ).fit(A, ytr)


def mean_proba(params, A, ytr, B, seeds):
    return np.mean([fit_model(params, A, ytr, s).predict_proba(B)[:, 1] for s in seeds], axis=0)


# ------------------------------------------------------------------ data windows
class Windows:
    """Feature selection + imputation fitted on the TRAINING rows of each window only (cached)."""

    def __init__(self, X, y, sel):
        self.X, self.y = X, y
        self.strategy = sel["strategy"]
        self.k = None if sel["k"] == 0 else sel["k"]
        self.cache = {}

    def get(self, tr, va):
        key = (tuple(tr), tuple(va))
        if key not in self.cache:
            Xtr, ytr = self.X.iloc[tr[0] : tr[1]], self.y.iloc[tr[0] : tr[1]]
            cols = select(self.strategy, score_features(Xtr, ytr), self.k)
            imp = SimpleImputer(strategy="median").fit(Xtr[cols])
            Xva, yva = self.X.iloc[va[0] : va[1]], self.y.iloc[va[0] : va[1]]
            self.cache[key] = (
                imp.transform(Xtr[cols]),
                ytr.values,
                imp.transform(Xva[cols]),
                yva.values,
            )
        return self.cache[key]


# ------------------------------------------------------------------ optimisers
def make_fitness(wins, folds):
    """Mean average-precision over the search folds (validation labels are only scored, never fit)."""
    data = [wins.get(f["train"], f["val"]) for f in folds]

    def fitness(params):
        aps = []
        for A, ytr, B, yva in data:
            aps.append(average_precision_score(yva, mean_proba(params, A, ytr, B, SEARCH_SEEDS)))
        return float(np.mean(aps))

    return fitness


def run_pso(fitness, n_particles, iters, rng, w=0.72, c1=1.49, c2=1.49, vmax=0.25):
    """
    Standard global-best PSO in the unit cube (Clerc-Kennedy constriction-style constants).
    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x);  x <- clip(x + v, 0, 1)
    """
    dim = len(SPACE)
    X = rng.random((n_particles, dim))
    V = rng.uniform(-0.1, 0.1, (n_particles, dim))
    log, evals = [], 0

    def evaluate(x, it, role):
        nonlocal evals
        params = decode(x)
        f = fitness(params)
        evals += 1
        log.append(dict(method="pso", eval=evals, iteration=it, role=role, fitness=f, **params))
        return f

    pbest, pbest_f = X.copy(), np.array([evaluate(x, 0, "init") for x in X])
    g = int(np.argmax(pbest_f))
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    print(f"  PSO iter 0/{iters}: best search AP = {gbest_f:.4f}  ({evals} evals)")
    for it in range(1, iters + 1):
        r1, r2 = rng.random((2, n_particles, dim))
        V = np.clip(w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X), -vmax, vmax)
        X = np.clip(X + V, 0.0, 1.0)
        for i in range(n_particles):
            f = evaluate(X[i], it, "move")
            if f > pbest_f[i]:
                pbest[i], pbest_f[i] = X[i].copy(), f
                if f > gbest_f:
                    gbest, gbest_f = X[i].copy(), f
        print(f"  PSO iter {it}/{iters}: best search AP = {gbest_f:.4f}  ({evals} evals)")
    return decode(gbest), gbest_f, pd.DataFrame(log)


def run_random(fitness, n_evals, rng):
    """Control: same evaluation budget, points drawn uniformly from the same search space."""
    best_f, best_p, log = -1.0, None, []
    for i in range(1, n_evals + 1):
        params = decode(rng.random(len(SPACE)))
        f = fitness(params)
        log.append(dict(method="random", eval=i, iteration=np.nan, role="sample", fitness=f, **params))
        if f > best_f:
            best_f, best_p = f, params
    print(f"  Random search: best search AP = {best_f:.4f}  ({n_evals} evals)")
    return best_p, best_f, pd.DataFrame(log)


# ------------------------------------------------------------------ evaluation
def block_sizes(n, k):
    return [len(a) for a in np.array_split(np.arange(n), k)]


def evaluate_configs(configs, wins, tr, va, blocks):
    A, ytr, B, yv = wins.get(tr, va)
    probs = {m: mean_proba(p, A, ytr, B, SEEDS) for m, p in configs.items()}
    prev, sizes = yv.mean(), block_sizes(len(yv), blocks)
    ap_def = average_precision_score(yv, probs["default"])
    rows = []
    for m, p in probs.items():
        ap = average_precision_score(yv, p)
        lo, hi = (np.nan, np.nan) if m == "default" else paired_ci(yv, p, probs["default"])
        rows.append(
            dict(
                method=m,
                n=len(yv),
                fails=int(yv.sum()),
                prevalence=prev,
                pr_auc=ap,
                pr_lift=ap / prev,
                roc_auc=roc_auc_score(yv, p),
                d_pr_auc_vs_default=ap - ap_def,
                ci95="" if m == "default" else f"[{lo:+.3f}, {hi:+.3f}]",
                **alert_metrics(yv, p, sizes),
            )
        )
    return pd.DataFrame(rows), probs, yv


def plot_convergence(hist, default_fit, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, c in (("pso", "tab:blue"), ("random", "tab:orange")):
        h = hist[hist.method == m].sort_values("eval")
        ax.plot(h["eval"], h["fitness"].cummax(), color=c, lw=2, label=f"{m.upper() if m == 'pso' else 'Random search'} (best so far)")
        ax.scatter(h["eval"], h["fitness"], color=c, s=8, alpha=0.25)
    ax.axhline(default_fit, color="gray", ls="--", label="Default XGBoost")
    ax.set_xlabel("Model evaluations")
    ax.set_ylabel("Search-fold PR-AUC (folds 1-2)")
    ax.set_title("PSO vs random search at equal budget")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="frozen final-test evaluation (run once)")
    args = ap.parse_args()

    df, cfg, sel, feats = load_all()
    X, y = df[feats], df["target"]
    wins = Windows(X, y, sel)
    print(f"Feature config (Step 3, development data only): {sel}")

    # ============================== FINAL TEST (separate, frozen stage)
    if args.final:
        out = f"{RES}/pso_final_summary.csv"
        if os.path.exists(out) and os.environ.get("FORCE") != "1":
            sys.exit(f"Final PSO test already run ({out} exists). FORCE=1 only for a documented re-run.")
        if not os.path.exists(f"{RES}/pso_best.json"):
            sys.exit("Run the search stage first (python src/09_pso_xgboost.py).")
        frozen = json.load(open(f"{RES}/pso_best.json"))
        configs = {m: frozen[m]["params"] for m in ("default", "random", "pso")}
        fin = cfg["final"]
        summ, probs, yv = evaluate_configs(configs, wins, fin["train"], fin["test"], FINAL_BLOCKS)
        summ.to_csv(out, index=False)
        pd.DataFrame(
            {"row_id": df["row_id"].iloc[fin["test"][0] : fin["test"][1]].values, "y": yv,
             **{f"score_{m}": p for m, p in probs.items()}}
        ).to_csv(f"{RES}/pso_final_scores.csv", index=False)
        pd.set_option("display.width", 220)
        print("\n=== FINAL TEST (frozen configs, trained on all development rows) ===")
        print(summ.round(3).to_string(index=False))
        print(f"\nSaved {out}")
        return

    # ============================== SEARCH STAGE
    search_folds, holdout = cfg["folds"][:2], cfg["folds"][2]
    fitness = make_fitness(wins, search_folds)
    t0 = time.time()
    default_fit = fitness(DEFAULT)
    print(f"Default XGBoost, search-fold PR-AUC: {default_fit:.4f}  ({time.time() - t0:.1f}s per evaluation)")

    print(f"\nPSO: {args.particles} particles x {args.iters} iterations")
    pso_best, pso_fit, pso_hist = run_pso(fitness, args.particles, args.iters, np.random.default_rng(args.seed))
    n_evals = len(pso_hist)
    print(f"\nRandom search control: {n_evals} evaluations")
    rnd_best, rnd_fit, rnd_hist = run_random(fitness, n_evals, np.random.default_rng(args.seed + 1))

    hist = pd.concat([pso_hist, rnd_hist], ignore_index=True)
    hist.to_csv(f"{RES}/pso_search_history.csv", index=False)
    frozen = {
        "default": dict(params=DEFAULT, search_pr_auc=default_fit),
        "random": dict(params=rnd_best, search_pr_auc=rnd_fit),
        "pso": dict(params=pso_best, search_pr_auc=pso_fit),
        "budget_evaluations": n_evals,
        "search_folds": [f["fold"] for f in search_folds],
        "heldout_fold": holdout["fold"],
    }
    json.dump(frozen, open(f"{RES}/pso_best.json", "w"), indent=2)  # frozen BEFORE the held-out look

    configs = {m: frozen[m]["params"] for m in ("default", "random", "pso")}
    summ, _, _ = evaluate_configs(configs, wins, holdout["train"], holdout["val"], HOLDOUT_BLOCKS)
    summ.to_csv(f"{RES}/pso_holdout_summary.csv", index=False)
    os.makedirs(FIG, exist_ok=True)
    plot_convergence(hist, default_fit, f"{FIG}/fig8_pso_convergence.png")

    pd.set_option("display.width", 220)
    print(f"\n=== Held-out fold {holdout['fold']} (never seen by the search; n={summ.n[0]}, fails={summ.fails[0]}) ===")
    print(summ.round(3).to_string(index=False))
    print("\nFrozen params -> results/pso_best.json")
    print("Run  python src/09_pso_xgboost.py --final  once to score the frozen configs on the final test.")


if __name__ == "__main__":
    main()
