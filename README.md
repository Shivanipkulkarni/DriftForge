# DriftForge

DriftForge is a leakage-safe research pipeline for semiconductor failure prediction on time-ordered SECOM data. It compares static and adaptive XGBoost models, evaluates feature-selection strategies, measures label-free feature drift, and produces SHAP explanations and paper-ready assets.

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

You can override the provider and model with `LLM_PROVIDER` and `CHATBOT_MODEL`. The self-test does not call an API:

```powershell
python src/07_chatbot.py --selftest
```

The adversarial evaluation writes `results/chatbot_eval.json`:

```powershell
python src/07_chatbot.py --eval --provider groq
```

### Step 8: Paper assets

`08_paper_assets.py` reads available result files and writes 300-DPI figures and an Excel workbook to `paper/`.

## Main outputs

- `results/static_baseline_*.csv`
- `results/featsel_*.csv`
- `results/adaptive_*.csv`
- `results/final_test_*.csv`
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