# DriftForge

DriftForge is a leakage-safe research pipeline for semiconductor failure prediction on time-ordered SECOM data. It compares static and adaptive XGBoost models, evaluates feature-selection strategies, measures label-free feature drift, tests whether particle-swarm-tuned XGBoost (PSO-XGBoost) helps under drift, and produces SHAP explanations and paper-ready assets. It builds on prior work on XGBoost-based wafer yield prediction (for example Xu, Qin & Sun, 2022, who identify poor cross-batch generalization as a core problem) and differs from it in evaluation: strictly forward in time, on the public SECOM dataset.

## Repository layout

```text
configs/       Frozen temporal fold definitions
data/raw/      Local source dataset (not committed)
data/interim/  Cleaned, time-ordered dataset (generated)
data/           Local raw, intermediate, and metadata data
paper/         Paper figures, tables, and manifest
results/       Experiment scores, summaries, and explanations
src/           Numbered pipeline scripts and shared helpers
```

The checked-in configuration and reported results make the experiment protocol auditable. Dataset files are intentionally kept local; place the source SECOM CSV at `data/raw/uci-secom.csv` before running the pipeline.

## Requirements

- Python 3.10 or newer recommended
- A raw SECOM CSV file
- Dependencies listed in [requirements.txt](requirements.txt)

The optional chatbot uses the OpenAI Python client with an OpenAI-compatible provider such as Groq or Gemini. It is not required for the modeling pipeline.

## Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline

Run the numbered scripts from the repository root:

```powershell
python src/01_prepare_data.py data/raw/uci-secom.csv
python src/02_static_baseline.py
python src/03_feature_selection.py
python src/04_adaptive_experiment.py
python src/05_final_test.py
python src/06_shap_registry.py
python src/07_chatbot.py --selftest
python src/08_paper_assets.py
python src/09_pso_xgboost.py
python src/09_pso_xgboost.py --final
python src/10_paper_protocol_comparison.py
```

Run the scripts in order from the repository root. Steps 2-4 operate on the development period, Step 5 evaluates the frozen final test period, and Steps 6-8 consume the generated results.

### Step 1: Prepare data

`01_prepare_data.py` repairs timestamps, sorts rows chronologically, removes features using development data only, and creates:

- `data/interim/secom_clean.csv`
- `configs/folds.json`

The raw CSV is expected to contain `Time`, `Pass/Fail`, and sensor columns.

### Steps 2-5: Model experiments

- `02_static_baseline.py` evaluates XGBoost, logistic regression, and random forest baselines.
- `03_feature_selection.py` compares mutual-information, XGBoost, consensus, and drift-aware selection.
- `04_adaptive_experiment.py` evaluates periodic and drift-triggered retraining on the development stream.
- `05_final_test.py` runs the frozen evaluation on the untouched final 20% of rows.

The final test refuses to overwrite existing results. A documented rerun requires:

```powershell
$env:FORCE = "1"
python src/05_final_test.py
```

### Step 6: Explanations

`06_shap_registry.py` creates the feature registry, global SHAP results, local explanation cards, and SHAP plots. Feature meanings remain unknown unless they are supplied with a source in `data/metadata/feature_registry.csv`.

### Step 7: Optional chatbot

The chatbot requires outputs from Steps 1, 3, 5, and 6. It supports two OpenAI-compatible providers:

```powershell
$env:GROQ_API_KEY = "your-key"
python src/07_chatbot.py --provider groq
```

```powershell
$env:GEMINI_API_KEY = "your-key"
python src/07_chatbot.py --provider gemini
```

The chatbot opens in your default browser and serves its UI locally at `127.0.0.1:8765`. Keep the Python process running while chatting; press `Ctrl+C` in the terminal to stop it. Use `--port` to choose a different local port, or add `--terminal` to use the original terminal interface.

You can override the provider and model with `LLM_PROVIDER` and `CHATBOT_MODEL`. The self-test does not call an API:

```powershell
python src/07_chatbot.py --selftest
```

The interactive chatbot is instructed to explain metrics and model outputs in plain language for non-technical readers.
It still notes when the data cannot support a conclusion, such as whether a sensor physically causes failures.

The adversarial evaluation writes `results/chatbot_eval.json`:

```powershell
python src/07_chatbot.py --eval --provider groq
```

### Step 8: Paper assets

`08_paper_assets.py` reads available result files and writes 300-DPI figures and an Excel workbook to `paper/`.

### Steps 9-10: PSO-XGBoost and the protocol comparison

`09_pso_xgboost.py` compares default XGBoost, random search, and PSO over eight XGBoost hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `gamma`), all with the frozen feature config from Step 3 and the frozen temporal folds. Random search gets the same number of evaluations as PSO, so any gain has to come from the swarm and not from tuning in general.

- The search only sees folds 1-2. Fold 3 is held out from the search and used once to compare the frozen configurations.
- The frozen parameters are written to `results/pso_best.json` before the held-out fold is scored.
- `--final` scores the frozen configurations on the untouched final test. Like Step 5, it refuses to overwrite existing results unless `FORCE=1` is set.
- Useful options: `--particles` (default 12), `--iters` (default 10), `--seed`.

`10_paper_protocol_comparison.py` evaluates the same configurations with a shuffled, stratified 80/20 split, the way many SECOM papers do. It is **not** the project's primary protocol and is not leakage-safe with respect to time. It exists to show how much a random split flatters the same models, and it writes `results/protocol_gap.csv`.

**Reference run (one environment, 16 particles x 12 iterations, 208 evaluations per method).** Search-fold PR-AUC rose from 0.128 (default) to 0.314 (PSO) and 0.274 (random search). On held-out fold 3, PSO matched default XGBoost (PR-AUC 0.074 vs 0.074). On the final test, PSO scored PR-AUC 0.055 vs 0.068 for default (paired 95% CI of the difference: -0.028 to +0.003). The improvement on the search folds did not carry forward in time, so on this data PSO tuning mostly fits the folds it was tuned on. Final-test numbers depend on library versions; rerun Steps 9-10 in your own environment before quoting them.

## Main outputs

- `results/static_baseline_*.csv`
- `results/featsel_*.csv`
- `results/adaptive_*.csv`
- `results/final_test_*.csv`
- `results/pso_search_history.csv`, `results/pso_best.json`, `results/pso_holdout_summary.csv`, `results/pso_final_*.csv`
- `results/paper_protocol_*.csv`, `results/protocol_gap.csv`
- `results/shap_global.csv`
- `results/shap_test_values.csv`
- `results/explanations.json`
- `paper/figures/*.png`
- `paper/tables.xlsx`

Generated data and intermediate files are ignored by Git. Keep the frozen configuration and any reported result files under version control when reproducibility requires them.

## Git and data policy

- `.env` files, virtual environments, Python caches, and the entire `data/` directory are ignored.
- `configs/`, `results/`, and `paper/` remain visible to Git so experiment context and reported artifacts can be reviewed.
- Never commit API keys. Use environment variables such as `GROQ_API_KEY` or `GEMINI_API_KEY` for the optional chatbot.

## Reproducibility notes

The scripts use fixed random seeds and temporal folds. Imputation, feature selection, threshold selection, and model fitting are performed without using validation or final-test labels for their decisions. Run the scripts from the repository root because they use project-relative paths.