"""
01_prepare_data.py
Load raw SECOM, repair the corrupted timestamps, clean WITHOUT leakage,
and freeze the temporal validation folds.

Usage (from the project root):
    python src/01_prepare_data.py data/raw/uci-secom.csv

Outputs:
    data/interim/secom_clean.csv   time-ordered, NOT imputed (imputation happens inside each fold)
    configs/folds.json             frozen fold boundaries -> commit this, nobody edits it afterwards
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/uci-secom.csv")
OUT_DIR, CFG_DIR = Path("data/interim"), Path("configs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG_DIR.mkdir(parents=True, exist_ok=True)

DEV_FRAC = 0.80  # last 20% of rows (in time order) = final untouched test
N_BLOCKS = 8  # development period is cut into 8 equal time blocks
MISSING_MAX = 0.75  # drop features with >75% missing (decided on development rows only)

# Expanding-window folds, as lists of block numbers (0-based).
# Validation windows are 2 blocks wide where possible because single blocks
# can contain only ~2 failures, which makes any metric meaningless.
FOLDS = [
    ([0, 1, 2], [3, 4]),
    ([0, 1, 2, 3, 4], [5, 6]),
    ([0, 1, 2, 3, 4, 5, 6], [7]),
]

# ---------------------------------------------------------------- load + repair time
df = pd.read_csv(RAW)
feat_cols = [c for c in df.columns if c not in ("Time", "Pass/Fail")]


def repair(s: str) -> pd.Timestamp:
    """Raw strings with day <= 12 were written with day and month swapped."""
    d = pd.to_datetime(s)
    return d.replace(month=d.day, day=d.month) if d.day <= 12 else d


time = df["Time"].astype(str).map(repair)
n_viol = int((time.diff() < pd.Timedelta(0)).sum())
span_days = (time.max() - time.min()).days
print(f"Repaired time range : {time.min()} -> {time.max()} ({span_days} days)")
print(f"Order violations vs file order: {n_viol}")
if n_viol > 0 or span_days > 120:
    print(
        "WARNING: repair did not fully restore chronology - inspect before continuing!"
    )

order = np.argsort(
    time.values, kind="stable"
)  # no-op if the file order is already correct
df = df.iloc[order].reset_index(drop=True)
time = time.iloc[order].reset_index(drop=True)

y = df["Pass/Fail"].map({-1: 0, 1: 1}).astype(int)
X = df[feat_cols].astype(float)
n = len(X)
n_dev = int(n * DEV_FRAC)

# ---------------------------------------------------------------- leakage-safe cleaning
# Feature filters use development rows only; the final test period never influences them.
miss = X.iloc[:n_dev].isna().mean()
drop_missing = miss[miss > MISSING_MAX].index.tolist()
X = X.drop(columns=drop_missing)
nun = X.iloc[:n_dev].nunique(dropna=True)
drop_const = nun[nun <= 1].index.tolist()
X = X.drop(columns=drop_const)
print(
    f"Dropped >{int(MISSING_MAX*100)}% missing: {len(drop_missing)} | constant: {len(drop_const)} "
    f"| features kept: {X.shape[1]}"
)

meta = pd.DataFrame({"row_id": np.arange(n), "time": time, "target": y})
clean = pd.concat([meta, X], axis=1)
clean.to_csv(OUT_DIR / "secom_clean.csv", index=False)

# ---------------------------------------------------------------- temporal folds
blocks = np.array_split(np.arange(n_dev), N_BLOCKS)


def span(bl):
    return [
        int(blocks[bl[0]][0]),
        int(blocks[bl[-1]][-1]) + 1,
    ]  # [start, end) in row_id


folds = [
    {"fold": i + 1, "train": span(tr), "val": span(va)}
    for i, (tr, va) in enumerate(FOLDS)
]
cfg = {
    "n_rows": n,
    "n_dev": n_dev,
    "n_blocks": N_BLOCKS,
    "folds": folds,
    "final": {"train": [0, n_dev], "test": [n_dev, n]},
    "dropped_missing": drop_missing,
    "dropped_constant": drop_const,
}
(CFG_DIR / "folds.json").write_text(json.dumps(cfg, indent=2))


def describe(name, lo, hi):
    yy, tt = y.iloc[lo:hi], time.iloc[lo:hi]
    print(
        f"{name:<11} rows {lo:>4}-{hi:<4} n={hi - lo:<4} fails={int(yy.sum()):<3} "
        f"rate={100 * yy.mean():5.2f}%  {tt.iloc[0]:%Y-%m-%d} -> {tt.iloc[-1]:%Y-%m-%d}"
    )


print("\nFold layout (train / validation):")
for f in folds:
    describe(f"F{f['fold']} train", *f["train"])
    describe(f"F{f['fold']} val", *f["val"])
print()
describe("FINAL train", 0, n_dev)
describe("FINAL test", n_dev, n)
print("\nSaved data/interim/secom_clean.csv and configs/folds.json")
